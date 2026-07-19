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

use std::collections::HashSet;
use std::fmt::Write as _;

use dsl_kit::{
    CallFrameId, DslNode, EngineError, EngineResult, Event, EventSink, IdGen, Iteration, NodeContext,
    NodeId, Path, Phase, StepOutcome, Stepper, SuspendReason, Walk,
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
            let _ = writeln!(out, "{} {}", node.node_id(), variant_name(node));
            depth += 1;
        }
        Phase::Post => {
            depth = depth.saturating_sub(1);
        }
    });
    out
}

fn variant_name(flow: &Flow) -> &'static str {
    match flow {
        Flow::Seq { .. } => "Seq",
        Flow::Par { .. } => "Par",
        Flow::Call { .. } => "Call",
        Flow::Scope { .. } => "Scope",
        Flow::Maybe { .. } => "Maybe",
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
    events: PrintSink,
    next_frame: u64,
    suspend_pending: bool,
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

struct PrintSink;

impl EventSink for PrintSink {
    fn emit(&mut self, event: &Event) {
        println!("{event:?}");
    }
}

impl<'a> FlowStepper<'a> {
    fn new(root: &'a Flow) -> Self {
        let path = Path::root().push(root.node_id());
        Self {
            stack: vec![Frame { node: root, path, state: FrameState::Enter, frame_id: None }],
            events: PrintSink,
            next_frame: 1,
            suspend_pending: false,
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
                    Flow::Scope { body, label, .. } => {
                        println!("<scope> {label}");
                        frame.state = FrameState::ScopePending { body: body.as_ref() };
                    }
                    Flow::Maybe { body, .. } => {
                        frame.state =
                            FrameState::MaybePending { body: body.as_deref() };
                    }
                    Flow::Call { label, .. } => {
                        println!("<call> {label}");
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
                    return Ok(StepOutcome::Suspended(SuspendReason::AwaitEffect));
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

// ---------- Demo ---------------------------------------------------------

fn build_program(ids: &IdGen) -> Flow {
    Flow::Seq {
        id: ids.node(),
        children: vec![
            Flow::Call { id: ids.node(), label: "greet".into() },
            Flow::Scope {
                id: ids.node(),
                label: "research".into(),
                body: Box::new(Flow::Par {
                    id: ids.node(),
                    children: vec![
                        Flow::Call { id: ids.node(), label: "search".into() },
                        Flow::Call { id: ids.node(), label: "summarise".into() },
                    ],
                }),
            },
            Flow::Maybe {
                id: ids.node(),
                body: Some(Box::new(Flow::Call { id: ids.node(), label: "review".into() })),
            },
        ],
    }
}

fn main() -> miette::Result<()> {
    let ids = IdGen::new();
    let program = build_program(&ids);

    println!("--- AST ---");
    print!("{}", pretty(&program));

    check_unique_ids(&program)?;

    println!("\n--- Nodes located by ID ---");
    if let Some(node) = program.find_by_id(NodeId(4)) {
        println!("find_by_id({}) = {}", NodeId(4), variant_name(node));
    }

    println!("\n--- Stepped execution ---");
    let mut stepper = FlowStepper::new(&program);
    loop {
        match stepper.run_to_yield()? {
            StepOutcome::Advanced => {}
            StepOutcome::Suspended(reason) => {
                println!("<host> resolving effect ({reason})");
            }
            StepOutcome::Done(()) => {
                println!("<done>");
                break;
            }
        }
    }

    println!("\n--- Error rendering ---");
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
