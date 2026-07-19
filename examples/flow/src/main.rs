//! A minimal flow DSL driven directly against the engine core primitives.
//!
//! This example builds three flow nodes by hand (no parser, no derive
//! attributes yet), wraps them in a hand-written stepper, and runs the
//! stepper to completion while printing every event.

use dsl_kit::{
    CallFrameId, DslNode, Event, EventSink, IdGen, Iteration, NodeId, Path, StepOutcome, Stepper,
    SuspendReason,
};

/// A tiny flow DSL.
///
/// - `Seq` runs its children in order.
/// - `Par` "runs" its children in parallel (this example resolves them
///   sequentially for simplicity; the point is that the engine can observe
///   the structure).
/// - `Call` denotes an external effect and yields once before completing.
#[derive(Debug, DslNode)]
enum Flow {
    Seq(SeqNode),
    Par(ParNode),
    Call(CallNode),
}

#[derive(Debug)]
struct SeqNode {
    id: NodeId,
    children: Vec<Flow>,
}

#[derive(Debug)]
struct ParNode {
    id: NodeId,
    children: Vec<Flow>,
}

#[derive(Debug)]
struct CallNode {
    id: NodeId,
    label: String,
}

impl DslNode for SeqNode {
    fn node_id(&self) -> NodeId {
        self.id
    }
}

impl DslNode for ParNode {
    fn node_id(&self) -> NodeId {
        self.id
    }
}

impl DslNode for CallNode {
    fn node_id(&self) -> NodeId {
        self.id
    }
}

/// Stepper state.
///
/// A real implementation would compile the AST into an explicit state
/// machine; this example maintains a stack of unfinished nodes and walks
/// them one `step()` call at a time.
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
    call_id: Option<CallFrameId>,
}

enum FrameState<'a> {
    Enter,
    SeqNext { children: std::slice::Iter<'a, Flow>, index: u64 },
    ParNext { children: std::slice::Iter<'a, Flow>, index: u64 },
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
            stack: vec![Frame { node: root, path, state: FrameState::Enter, call_id: None }],
            events: PrintSink,
            next_frame: 1,
            suspend_pending: false,
        }
    }
}

impl<'a> Stepper for FlowStepper<'a> {
    type Value = ();
    type Error = std::convert::Infallible;

    fn step(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error> {
        if self.stack.is_empty() {
            return Ok(StepOutcome::Done(()));
        }
        let depth_before = self.stack.len() as u32;

        let frame = self.stack.last_mut().expect("non-empty");
        let node_id = frame.node.node_id();
        let path = frame.path.clone();

        match &mut frame.state {
            FrameState::Enter => {
                self.events.emit(&Event::VisitPre { node: node_id, path: path.clone() });
                match frame.node {
                    Flow::Seq(node) => {
                        frame.state =
                            FrameState::SeqNext { children: node.children.iter(), index: 0 };
                    }
                    Flow::Par(node) => {
                        let call_id = CallFrameId(self.next_frame);
                        self.next_frame += 1;
                        let frame = self.stack.last_mut().expect("non-empty");
                        frame.call_id = Some(call_id);
                        frame.state =
                            FrameState::ParNext { children: node.children.iter(), index: 0 };
                        self.events.emit(&Event::FrameEnter {
                            node: node_id,
                            frame: call_id,
                            depth: depth_before,
                        });
                    }
                    Flow::Call(node) => {
                        println!("<call> {}", node.label);
                        frame.state = FrameState::CallSuspending;
                        self.suspend_pending = true;
                        self.events.emit(&Event::Suspend {
                            node: node_id,
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
                    let idx = Iteration(*index);
                    self.events.emit(&Event::IterationTick { node: node_id, iteration: idx });
                    self.stack.push(Frame {
                        node: next,
                        path: child_path,
                        state: FrameState::Enter,
                        call_id: None,
                    });
                    Ok(StepOutcome::Advanced)
                } else {
                    self.events.emit(&Event::VisitPost { node: node_id, path });
                    self.stack.pop();
                    Ok(StepOutcome::Advanced)
                }
            }
            FrameState::ParNext { children, index } => {
                if let Some(next) = children.next() {
                    let child_path = path.push(next.node_id());
                    *index += 1;
                    let idx = Iteration(*index);
                    self.events.emit(&Event::IterationTick { node: node_id, iteration: idx });
                    self.stack.push(Frame {
                        node: next,
                        path: child_path,
                        state: FrameState::Enter,
                        call_id: None,
                    });
                    Ok(StepOutcome::Advanced)
                } else {
                    if let Some(call_id) = frame.call_id {
                        self.events.emit(&Event::FrameLeave {
                            node: node_id,
                            frame: call_id,
                            depth: self.stack.len() as u32,
                        });
                    }
                    self.events.emit(&Event::VisitPost { node: node_id, path });
                    self.stack.pop();
                    Ok(StepOutcome::Advanced)
                }
            }
            FrameState::CallSuspending => {
                if self.suspend_pending {
                    self.suspend_pending = false;
                    return Ok(StepOutcome::Suspended(SuspendReason::AwaitEffect));
                }
                self.events.emit(&Event::Resume { node: node_id });
                frame.state = FrameState::CallDone;
                Ok(StepOutcome::Advanced)
            }
            FrameState::CallDone => {
                self.events.emit(&Event::VisitPost { node: node_id, path });
                self.stack.pop();
                Ok(StepOutcome::Advanced)
            }
        }
    }
}

fn main() {
    let ids = IdGen::new();

    let program = Flow::Seq(SeqNode {
        id: ids.node(),
        children: vec![
            Flow::Call(CallNode { id: ids.node(), label: "greet".into() }),
            Flow::Par(ParNode {
                id: ids.node(),
                children: vec![
                    Flow::Call(CallNode { id: ids.node(), label: "search".into() }),
                    Flow::Call(CallNode { id: ids.node(), label: "summarise".into() }),
                ],
            }),
        ],
    });

    let mut stepper = FlowStepper::new(&program);
    loop {
        match stepper.run_to_yield().expect("infallible") {
            StepOutcome::Advanced => {} // won't happen: run_to_yield exits on non-Advanced
            StepOutcome::Suspended(reason) => {
                println!("<host> resolving effect ({reason:?})");
            }
            StepOutcome::Done(()) => {
                println!("<done>");
                break;
            }
        }
    }
}
