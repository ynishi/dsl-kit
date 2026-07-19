//! Reference flow DSL for `dsl-kit`.
//!
//! This example demonstrates the α authoring path end to end:
//!
//! 1. The AST is a single `enum` with named fields per variant, one of
//!    which is `id: NodeId`. Recursive fields (`Box<Flow>`, `Vec<Flow>`,
//!    `Option<Flow>`) are picked up by the derive automatically.
//! 2. `#[derive(DslNode)]` provides `node_id()`, `Walk`, and `WalkMut` in
//!    one line.
//! 3. A hand-rolled stepper drives the AST through the engine's event
//!    stream and demonstrates suspend / resume around effect calls.
//! 4. The `EngineError` type carries a full `NodeContext`, so a failure
//!    is enough to locate the offending node.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use dsl_kit::{
    BreakCondition, BreakpointSet, CallFrameId, DslNode, EngineError, EngineResult, Event,
    EventSink, IdGen, Iteration, NodeContext, NodeId, Path, Phase, StepOutcome, Stepper,
    SuspendReason, Walk,
};

#[derive(Debug, DslNode)]
enum Flow {
    /// Runs its children in order.
    Seq { id: NodeId, children: Vec<Flow> },
    /// Runs its children concurrently (this example schedules them
    /// sequentially, which is enough to demonstrate the event shape).
    Par { id: NodeId, children: Vec<Flow> },
    /// Denotes an external effect; the stepper yields once and resumes.
    Call { id: NodeId, label: String },
    /// Wraps a single inner flow with a label; the wrapper adds no
    /// semantics of its own beyond emitting a frame boundary.
    Scope { id: NodeId, label: String, body: Box<Flow> },
    /// Optionally runs an inner flow.
    Maybe { id: NodeId, body: Option<Box<Flow>> },
}

// ---------- Static analysis over the AST via `Walk` ---------------------

/// Produces an indented pretty-print of the tree, driven only through the
/// derived `Walk::walk` traversal.
fn pretty(flow: &Flow) -> String {
    let mut out = String::new();
    let mut depth: usize = 0;
    flow.walk(&mut |node, phase| match phase {
        Phase::Pre => {
            for _ in 0..depth {
                out.push_str("  ");
            }
            let _ = writeln!(out, "{} {}", node.node_id(), summary(node));
            depth += 1;
        }
        Phase::Post => {
            depth = depth.saturating_sub(1);
        }
    });
    out
}

fn summary(flow: &Flow) -> String {
    match flow {
        Flow::Seq { .. } => "Seq".into(),
        Flow::Par { .. } => "Par".into(),
        Flow::Call { label, .. } => format!("Call {label:?}"),
        Flow::Scope { label, .. } => format!("Scope {label:?}"),
        Flow::Maybe { .. } => "Maybe".into(),
    }
}

/// Confirms every node ID in the tree is unique. A stable `NodeId`
/// discipline is not automatic; the kit provides the type but callers
/// still have to allocate them consistently.
fn check_unique_ids(flow: &Flow) -> EngineResult<()> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut duplicate: Option<NodeId> = None;

    flow.walk(&mut |node, phase| {
        if phase != Phase::Pre {
            return;
        }
        if !seen.insert(node.node_id()) && duplicate.is_none() {
            duplicate = Some(node.node_id());
        }
    });

    match duplicate {
        None => Ok(()),
        Some(id) => Err(EngineError::Malformed {
            at: NodeContext::at(id, Path::root().push(id)),
            detail: format!("node id {id} appears more than once"),
        }),
    }
}

// ---------- Stepper ------------------------------------------------------

struct FlowStepper<'a> {
    stack: Vec<Frame<'a>>,
    events: CountingSink,
    next_frame: u64,
    suspend_pending: bool,
    results: HashMap<NodeId, String>,
}

struct Frame<'a> {
    node: &'a Flow,
    path: Path,
    state: FrameState<'a>,
    frame_id: Option<CallFrameId>,
}

enum FrameState<'a> {
    Enter,
    SeqNext { children: std::slice::Iter<'a, Flow>, index: u64 },
    ParNext { children: std::slice::Iter<'a, Flow>, index: u64 },
    ScopePending { body: &'a Flow },
    ScopeDone,
    MaybePending { body: Option<&'a Flow> },
    MaybeDone,
    CallSuspending,
    CallDone,
}

/// Silent sink that keeps a small histogram of event kinds; useful when
/// the demo wants to summarise "how much happened" without spelling out
/// every step.
#[derive(Default)]
struct CountingSink {
    visit_pre: u32,
    visit_post: u32,
    frame_enter: u32,
    frame_leave: u32,
    iteration_tick: u32,
    suspend: u32,
    resume: u32,
}

impl EventSink for CountingSink {
    fn emit(&mut self, event: &Event) {
        match event {
            Event::VisitPre { .. } => self.visit_pre += 1,
            Event::VisitPost { .. } => self.visit_post += 1,
            Event::FrameEnter { .. } => self.frame_enter += 1,
            Event::FrameLeave { .. } => self.frame_leave += 1,
            Event::IterationTick { .. } => self.iteration_tick += 1,
            Event::Suspend { .. } => self.suspend += 1,
            Event::Resume { .. } => self.resume += 1,
            _ => {}
        }
    }
}

impl CountingSink {
    fn summarise(&self) -> String {
        format!(
            "pre={} post={} frame_enter={} frame_leave={} iter={} suspend={} resume={}",
            self.visit_pre,
            self.visit_post,
            self.frame_enter,
            self.frame_leave,
            self.iteration_tick,
            self.suspend,
            self.resume,
        )
    }
}

impl<'a> FlowStepper<'a> {
    fn new(root: &'a Flow) -> Self {
        let path = Path::root().push(root.node_id());
        Self {
            stack: vec![Frame { node: root, path, state: FrameState::Enter, frame_id: None }],
            events: CountingSink::default(),
            next_frame: 1,
            suspend_pending: false,
            results: HashMap::new(),
        }
    }

    fn ctx(&self, frame: &Frame<'_>) -> NodeContext {
        NodeContext {
            node: frame.node.node_id(),
            path: frame.path.clone(),
            frame: frame.frame_id,
            depth: self.stack.len() as u32,
            iteration: None,
        }
    }

    /// Returns the `(id, label)` of the `Call` node that suspended the
    /// stepper, if the stepper is currently suspended on a call.
    fn suspended_call(&self) -> Option<(NodeId, &str)> {
        let frame = self.stack.last()?;
        if !matches!(frame.state, FrameState::CallSuspending) {
            return None;
        }
        match frame.node {
            Flow::Call { id, label } => Some((*id, label.as_str())),
            _ => None,
        }
    }

    /// Records the result the host produced while the stepper was
    /// suspended.
    fn record_result(&mut self, id: NodeId, result: String) {
        self.results.insert(id, result);
    }

    fn into_results(self) -> HashMap<NodeId, String> {
        self.results
    }

    fn event_summary(&self) -> String {
        self.events.summarise()
    }
}

impl<'a> Stepper for FlowStepper<'a> {
    type Value = ();
    type Error = EngineError;

    fn step(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error> {
        if self.stack.is_empty() {
            return Ok(StepOutcome::Done(()));
        }
        let depth_before = self.stack.len() as u32;

        let ctx = {
            let frame = self.stack.last().expect("non-empty");
            self.ctx(frame)
        };

        let frame = self.stack.last_mut().expect("non-empty");
        let path = frame.path.clone();

        match &mut frame.state {
            FrameState::Enter => {
                self.events.emit(&Event::VisitPre { at: ctx.clone() });
                match frame.node {
                    Flow::Seq { children, .. } => {
                        frame.state = FrameState::SeqNext { children: children.iter(), index: 0 };
                    }
                    Flow::Par { children, .. } => {
                        let call_id = CallFrameId(self.next_frame);
                        self.next_frame += 1;
                        let frame = self.stack.last_mut().expect("non-empty");
                        frame.frame_id = Some(call_id);
                        frame.state = FrameState::ParNext { children: children.iter(), index: 0 };
                        let mut ctx = ctx.clone();
                        ctx.frame = Some(call_id);
                        ctx.depth = depth_before;
                        self.events.emit(&Event::FrameEnter { at: ctx });
                    }
                    Flow::Scope { body, .. } => {
                        frame.state = FrameState::ScopePending { body: body.as_ref() };
                    }
                    Flow::Maybe { body, .. } => {
                        frame.state =
                            FrameState::MaybePending { body: body.as_deref() };
                    }
                    Flow::Call { .. } => {
                        frame.state = FrameState::CallSuspending;
                        self.suspend_pending = true;
                        self.events.emit(&Event::Suspend {
                            at: ctx,
                            reason: SuspendReason::AwaitEffect,
                        });
                    }
                }
                Ok(StepOutcome::Advanced)
            }
            FrameState::SeqNext { children, index } => {
                if let Some(next) = children.next() {
                    let child_path = path.push(next.node_id());
                    *index += 1;
                    let iter = Iteration(*index);
                    let mut ctx = ctx.clone();
                    ctx.iteration = Some(iter);
                    self.events.emit(&Event::IterationTick { at: ctx });
                    self.stack.push(Frame {
                        node: next,
                        path: child_path,
                        state: FrameState::Enter,
                        frame_id: None,
                    });
                    Ok(StepOutcome::Advanced)
                } else {
                    self.events.emit(&Event::VisitPost { at: ctx });
                    self.stack.pop();
                    Ok(StepOutcome::Advanced)
                }
            }
            FrameState::ParNext { children, index } => {
                if let Some(next) = children.next() {
                    let child_path = path.push(next.node_id());
                    *index += 1;
                    let iter = Iteration(*index);
                    let mut ctx = ctx.clone();
                    ctx.iteration = Some(iter);
                    self.events.emit(&Event::IterationTick { at: ctx });
                    self.stack.push(Frame {
                        node: next,
                        path: child_path,
                        state: FrameState::Enter,
                        frame_id: None,
                    });
                    Ok(StepOutcome::Advanced)
                } else {
                    let leave_ctx = ctx.clone();
                    self.events.emit(&Event::FrameLeave { at: leave_ctx });
                    self.events.emit(&Event::VisitPost { at: ctx });
                    self.stack.pop();
                    Ok(StepOutcome::Advanced)
                }
            }
            FrameState::ScopePending { body } => {
                let body = *body;
                let child_path = path.push(body.node_id());
                let child_state = FrameState::Enter;
                self.stack.last_mut().expect("non-empty").state = FrameState::ScopeDone;
                self.stack.push(Frame {
                    node: body,
                    path: child_path,
                    state: child_state,
                    frame_id: None,
                });
                Ok(StepOutcome::Advanced)
            }
            FrameState::ScopeDone => {
                self.events.emit(&Event::VisitPost { at: ctx });
                self.stack.pop();
                Ok(StepOutcome::Advanced)
            }
            FrameState::MaybePending { body } => {
                let body = *body;
                self.stack.last_mut().expect("non-empty").state = FrameState::MaybeDone;
                if let Some(body) = body {
                    let child_path = path.push(body.node_id());
                    self.stack.push(Frame {
                        node: body,
                        path: child_path,
                        state: FrameState::Enter,
                        frame_id: None,
                    });
                }
                Ok(StepOutcome::Advanced)
            }
            FrameState::MaybeDone => {
                self.events.emit(&Event::VisitPost { at: ctx });
                self.stack.pop();
                Ok(StepOutcome::Advanced)
            }
            FrameState::CallSuspending => {
                if self.suspend_pending {
                    self.suspend_pending = false;
                    return Ok(StepOutcome::Suspended {
                        reason: SuspendReason::AwaitEffect,
                        at: ctx,
                    });
                }
                self.events.emit(&Event::Resume { at: ctx });
                frame.state = FrameState::CallDone;
                Ok(StepOutcome::Advanced)
            }
            FrameState::CallDone => {
                self.events.emit(&Event::VisitPost { at: ctx });
                self.stack.pop();
                Ok(StepOutcome::Advanced)
            }
        }
    }
}

// ---------- Effect resolver ---------------------------------------------

/// Canned responses keyed by call label.
///
/// A real host would forward each call to an LLM, a tool, or an MCP
/// server. Here we return prewritten strings so the demo runs offline.
fn canned_response(label: &str) -> String {
    match label {
        "fetch_query" => "How does miette structure diagnostics?".into(),
        "search_arxiv" => "arxiv: 3 papers on structured diagnostics".into(),
        "search_github" => "github: miette (rust-lang), ariadne, codespan-reporting".into(),
        "search_web" => "web: rust blog posts, docs.rs entries".into(),
        "synthesise" => {
            "miette layers a Diagnostic trait over thiserror errors, adding code / severity / labels."
                .into()
        }
        "citation_check" => "citations: 3 sources cross-verified".into(),
        "write_report" => "report: 380 words, 3 citations, ready".into(),
        other => format!("<no handler for {other}>"),
    }
}

// ---------- Demo pipeline -----------------------------------------------

/// Builds a small research pipeline expressed in the flow DSL.
///
/// The shape is:
///
/// ```text
/// Seq(
///     Call "fetch_query",
///     Scope "web_research" {
///         Par(
///             Call "search_arxiv",
///             Call "search_github",
///             Call "search_web",
///         )
///     },
///     Call "synthesise",
///     Maybe(
///         Call "citation_check",
///     ),
///     Call "write_report",
/// )
/// ```
fn research_pipeline(ids: &IdGen) -> Flow {
    Flow::Seq {
        id: ids.node(),
        children: vec![
            Flow::Call { id: ids.node(), label: "fetch_query".into() },
            Flow::Scope {
                id: ids.node(),
                label: "web_research".into(),
                body: Box::new(Flow::Par {
                    id: ids.node(),
                    children: vec![
                        Flow::Call { id: ids.node(), label: "search_arxiv".into() },
                        Flow::Call { id: ids.node(), label: "search_github".into() },
                        Flow::Call { id: ids.node(), label: "search_web".into() },
                    ],
                }),
            },
            Flow::Call { id: ids.node(), label: "synthesise".into() },
            Flow::Maybe {
                id: ids.node(),
                body: Some(Box::new(Flow::Call {
                    id: ids.node(),
                    label: "citation_check".into(),
                })),
            },
            Flow::Call { id: ids.node(), label: "write_report".into() },
        ],
    }
}

/// Illustrates the breakpoint condition surface by walking the AST once
/// and reporting which nodes each condition matches.
///
/// A real host would keep a `BreakpointSet` alive across a debug session
/// and consult it inside the stepper's suspend logic; this demo runs
/// them off-line so the output stays focused on the matching rules.
fn demonstrate_breakpoints(program: &Flow) {
    let mut set = BreakpointSet::new();
    let by_id = set.add(BreakCondition::at_node(NodeId(4)));
    let deep = set.add(BreakCondition::at_depth_at_least(4));
    let inside_research = set.add(BreakCondition::under_path(
        Path::root().push(NodeId(0)).push(NodeId(2)),
    ));
    let combined = set.add(
        BreakCondition::at_depth_at_least(3).and(BreakCondition::at_iteration(2)),
    );

    println!("  registered:");
    println!("    {by_id}: at node n4");
    println!("    {deep}: depth >= 4");
    println!("    {inside_research}: under path /n0/n2");
    println!("    {combined}: depth >= 3 AND iteration == 2");

    // Walk the tree, synthesising a NodeContext per node and printing any
    // breakpoint hits. Real steppers would emit these from their event
    // loop; the demo builds them by hand so the code is shorter.
    fn walk(
        node: &Flow,
        path: &Path,
        depth: u32,
        set: &BreakpointSet,
        hits: &mut Vec<(NodeId, Vec<dsl_kit::BreakpointId>)>,
    ) {
        let ctx = NodeContext {
            node: node.node_id(),
            path: path.clone(),
            frame: None,
            depth,
            iteration: None,
        };
        let m = set.matches(&ctx);
        if !m.is_empty() {
            hits.push((node.node_id(), m));
        }
        for child in node.children() {
            let child_path = path.push(child.node_id());
            walk(child, &child_path, depth + 1, set, hits);
        }
    }

    let mut hits = Vec::new();
    let root_path = Path::root().push(program.node_id());
    walk(program, &root_path, 1, &set, &mut hits);

    println!("  hits:");
    for (node, ids) in &hits {
        let names: Vec<String> = ids.iter().map(ToString::to_string).collect();
        println!("    {node}: {}", names.join(", "));
    }
    if hits.is_empty() {
        println!("    (none)");
    }
}

fn main() -> miette::Result<()> {
    let ids = IdGen::new();
    let program = research_pipeline(&ids);

    println!("=== Research pipeline: AST ===");
    print!("{}", pretty(&program));

    check_unique_ids(&program)?;

    println!("\n=== Running the pipeline ===");
    let mut stepper = FlowStepper::new(&program);
    loop {
        match stepper.run_to_yield()? {
            StepOutcome::Advanced => {}
            StepOutcome::Suspended { reason: _, at } => {
                if let Some((id, label)) = stepper.suspended_call() {
                    let response = canned_response(label);
                    println!("  {id:>4} {label:<15} -> {response}   ({at})");
                    stepper.record_result(id, response);
                }
            }
            StepOutcome::Done(()) => break,
        }
    }

    println!("\n=== Event summary ===");
    println!("  {}", stepper.event_summary());

    println!("\n=== Recorded results ===");
    let mut results: Vec<(NodeId, String)> = stepper.into_results().into_iter().collect();
    results.sort_by_key(|(id, _)| id.0);
    for (id, text) in &results {
        println!("  {id}: {text}");
    }

    println!("\n=== Breakpoints ===");
    demonstrate_breakpoints(&program);

    println!("\n=== Error rendering (malformed AST) ===");
    let broken = Flow::Seq {
        id: NodeId(100),
        children: vec![
            Flow::Call { id: NodeId(101), label: "one".into() },
            Flow::Call { id: NodeId(101), label: "two".into() },
        ],
    };
    if let Err(err) = check_unique_ids(&broken) {
        println!("{:?}", miette::Report::new(err));
    }

    Ok(())
}
