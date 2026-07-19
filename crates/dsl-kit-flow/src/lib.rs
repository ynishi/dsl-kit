//! Reference flow DSL for `dsl-kit`.
//!
//! This crate defines a small orchestration DSL — `Seq`, `Par`, `Call`,
//! `Scope`, `Maybe` — with a hand-rolled stepper that drives the AST
//! through the engine's event stream. It is used by the flow example
//! and by the MCP server as a concrete DSL to debug.
//!
//! The DSL is deliberately tiny; its role is to exercise every engine
//! primitive (traversal, breakpoints, suspend / resume, structured
//! errors) end to end.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use dsl_kit::{
    BreakpointSet, CallFrameId, DslNode, EngineError, EngineResult, Event, EventSink, IdGen,
    Iteration, NodeContext, NodeId, Path, Phase, StepOutcome, Stepper, SuspendReason, Walk,
};

/// AST of the flow DSL.
///
/// Every variant carries an `id: NodeId` slot so the derive can attach
/// traversal and identification without extra attributes.
#[derive(Debug, DslNode)]
pub enum Flow {
    /// Runs its children in order.
    Seq { id: NodeId, children: Vec<Flow> },
    /// Runs its children concurrently (this reference stepper schedules
    /// them sequentially, which is enough to demonstrate the event
    /// shape).
    Par { id: NodeId, children: Vec<Flow> },
    /// Denotes an external effect; the stepper yields once and resumes
    /// once the host has provided a result.
    Call { id: NodeId, label: String },
    /// Wraps a single inner flow with a label; the wrapper adds no
    /// semantics of its own beyond delineating a section.
    Scope { id: NodeId, label: String, body: Box<Flow> },
    /// Optionally runs an inner flow.
    Maybe { id: NodeId, body: Option<Box<Flow>> },
}

impl Flow {
    /// One-line summary of a node's shape, used by pretty-printers.
    pub fn summary(&self) -> String {
        match self {
            Flow::Seq { .. } => "Seq".into(),
            Flow::Par { .. } => "Par".into(),
            Flow::Call { label, .. } => format!("Call {label:?}"),
            Flow::Scope { label, .. } => format!("Scope {label:?}"),
            Flow::Maybe { .. } => "Maybe".into(),
        }
    }
}

/// Renders a `Flow` as an indented text tree using the derived
/// `Walk::walk` traversal.
pub fn pretty(flow: &Flow) -> String {
    let mut out = String::new();
    let mut depth: usize = 0;
    flow.walk(&mut |node, phase| match phase {
        Phase::Pre => {
            for _ in 0..depth {
                out.push_str("  ");
            }
            let _ = writeln!(out, "{} {}", node.node_id(), node.summary());
            depth += 1;
        }
        Phase::Post => {
            depth = depth.saturating_sub(1);
        }
    });
    out
}

/// Confirms that every `NodeId` in the tree is unique.
///
/// Callers usually allocate ids with a shared [`IdGen`], but hand-
/// authored trees can accidentally reuse an id and this check exists to
/// surface that as a structured error.
pub fn check_unique_ids(flow: &Flow) -> EngineResult<()> {
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

/// Canned effect responses keyed by call label.
///
/// A real host would forward each call to an LLM, a tool, or an MCP
/// server. The reference DSL ships with prewritten strings so tests and
/// demos run offline.
pub fn canned_response(label: &str) -> String {
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

/// Builds a small research pipeline expressed in the flow DSL.
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
pub fn research_pipeline(ids: &IdGen) -> Flow {
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

/// Silent event sink that counts each event kind. Handy when the demo
/// wants to summarise "how much happened" without spelling every step
/// out.
#[derive(Debug, Default, Clone, Copy)]
pub struct CountingSink {
    pub visit_pre: u32,
    pub visit_post: u32,
    pub frame_enter: u32,
    pub frame_leave: u32,
    pub iteration_tick: u32,
    pub suspend: u32,
    pub resume: u32,
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
    pub fn summarise(&self) -> String {
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

/// Stepper over a `Flow` program.
///
/// The stepper walks the AST depth-first, emitting the observable
/// events (`VisitPre` / `VisitPost` / `FrameEnter` / `FrameLeave` /
/// `IterationTick` / `Suspend` / `Resume`) at each transition, and
/// yielding `Suspended { reason: AwaitEffect, .. }` at every `Call`
/// node so the host can supply the response.
pub struct FlowStepper<'a> {
    stack: Vec<Frame<'a>>,
    events: CountingSink,
    next_frame: u64,
    suspend_pending: bool,
    results: HashMap<NodeId, String>,
    /// Set on the step that follows a breakpoint yield, so the next
    /// `step_with_breakpoints` call skips the recheck and lets the
    /// underlying `step()` proceed with normal semantics.
    breakpoint_yielded: bool,
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

impl<'a> FlowStepper<'a> {
    /// Creates a fresh stepper anchored at `root`.
    pub fn new(root: &'a Flow) -> Self {
        let path = Path::root().push(root.node_id());
        Self {
            stack: vec![Frame { node: root, path, state: FrameState::Enter, frame_id: None }],
            events: CountingSink::default(),
            next_frame: 1,
            suspend_pending: false,
            results: HashMap::new(),
            breakpoint_yielded: false,
        }
    }

    /// Runs a single step, first checking whether the next node's
    /// `Enter` phase matches any registered breakpoint. When a
    /// breakpoint fires the stepper yields
    /// `Suspended { reason: Breakpoint, .. }` without advancing;
    /// the next call transitions normally.
    pub fn step_with_breakpoints(
        &mut self,
        breakpoints: &BreakpointSet,
    ) -> Result<StepOutcome<()>, EngineError> {
        if self.breakpoint_yielded {
            self.breakpoint_yielded = false;
            return self.step();
        }
        if breakpoints.is_empty() || self.stack.is_empty() {
            return self.step();
        }

        let frame = self.stack.last().expect("non-empty");
        if matches!(frame.state, FrameState::Enter) {
            let ctx = self.ctx(frame);
            if !breakpoints.matches(&ctx).is_empty() {
                self.breakpoint_yielded = true;
                self.events.emit(&Event::Suspend {
                    at: ctx.clone(),
                    reason: SuspendReason::Breakpoint,
                });
                return Ok(StepOutcome::Suspended {
                    reason: SuspendReason::Breakpoint,
                    at: ctx,
                });
            }
        }

        self.step()
    }

    /// Loops [`Self::step_with_breakpoints`] until suspension,
    /// completion, or error.
    pub fn run_to_yield_with_breakpoints(
        &mut self,
        breakpoints: &BreakpointSet,
    ) -> Result<StepOutcome<()>, EngineError> {
        loop {
            match self.step_with_breakpoints(breakpoints)? {
                StepOutcome::Advanced => continue,
                other => return Ok(other),
            }
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

    /// If the stepper is currently paused on a `Call` node, returns the
    /// node's `(id, label)`.
    pub fn suspended_call(&self) -> Option<(NodeId, &str)> {
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
    pub fn record_result(&mut self, id: NodeId, result: String) {
        self.results.insert(id, result);
    }

    /// Read access to the results recorded so far.
    pub fn results(&self) -> &HashMap<NodeId, String> {
        &self.results
    }

    /// Consumes the stepper and returns the accumulated results.
    pub fn into_results(self) -> HashMap<NodeId, String> {
        self.results
    }

    /// One-line summary of the events observed so far.
    pub fn event_summary(&self) -> String {
        self.events.summarise()
    }

    /// Snapshot of the counting sink.
    pub fn events(&self) -> CountingSink {
        self.events
    }

    /// Current stack depth (0 when finished, 1 at the root, etc.).
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Path to the top of the stack, if any.
    pub fn current_path(&self) -> Option<Path> {
        self.stack.last().map(|f| f.path.clone())
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
                        frame.state = FrameState::MaybePending { body: body.as_deref() };
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
