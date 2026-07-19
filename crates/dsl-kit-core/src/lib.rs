//! Engine primitives for `dsl-kit`.
//!
//! This crate defines the observable primitives every DSL built with the kit
//! carries from day one: stable node identifiers, call frame identifiers with
//! depth, iteration counters, root-to-node paths, an event stream, and a
//! stepper trait that models evaluation as a state machine.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Stable identifier assigned to every AST node when the tree is built.
///
/// A `NodeId` is orthogonal to source location: two nodes at the same
/// source span but in different expansions carry different IDs, and the same
/// node retains its ID across evaluation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Identifier for a single activation of a function-like node.
///
/// Recursion is uniquely identified by pairing `CallFrameId` with `depth`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallFrameId(pub u64);

impl fmt::Display for CallFrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f{}", self.0)
    }
}

/// Position within the current iteration of a loop-shaped node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Iteration(pub u64);

/// Root-to-node identifier chain, usable for path-shaped breakpoint
/// conditions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path(pub Vec<NodeId>);

impl Path {
    /// Returns an empty path (i.e. the root).
    #[inline]
    pub fn root() -> Self {
        Path(Vec::new())
    }

    /// Returns a new path with `node` pushed onto the end.
    pub fn push(&self, node: NodeId) -> Self {
        let mut next = self.0.clone();
        next.push(node);
        Path(next)
    }

    /// Depth of the path (number of node IDs).
    #[inline]
    pub fn depth(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for id in &self.0 {
            if !first {
                write!(f, "/")?;
            }
            write!(f, "{id}")?;
            first = false;
        }
        Ok(())
    }
}

/// Monotonic ID generator for nodes and call frames.
///
/// Kept as a plain atomic counter so it can be shared across threads without
/// synchronisation overhead.
#[derive(Debug, Default)]
pub struct IdGen {
    next: AtomicU64,
}

impl IdGen {
    pub const fn new() -> Self {
        Self { next: AtomicU64::new(0) }
    }

    /// Allocates a fresh `NodeId`.
    pub fn node(&self) -> NodeId {
        NodeId(self.next.fetch_add(1, Ordering::Relaxed))
    }

    /// Allocates a fresh `CallFrameId`.
    pub fn frame(&self) -> CallFrameId {
        CallFrameId(self.next.fetch_add(1, Ordering::Relaxed))
    }
}

/// One observation from the evaluator.
///
/// Backends (tracer, debugger, MCP tool, replay recorder) attach to the same
/// event stream. New variants may be added as the kit grows; downstream
/// consumers should treat the enum as non-exhaustive.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Event {
    /// Emitted before a node's semantics run.
    VisitPre { node: NodeId, path: Path },
    /// Emitted after a node's semantics complete.
    VisitPost { node: NodeId, path: Path },
    /// A function-like node started a new call frame.
    FrameEnter { node: NodeId, frame: CallFrameId, depth: u32 },
    /// A function-like node's frame ended.
    FrameLeave { node: NodeId, frame: CallFrameId, depth: u32 },
    /// A loop node advanced to a new iteration.
    IterationTick { node: NodeId, iteration: Iteration },
    /// The stepper is about to yield to the outside world.
    Suspend { node: NodeId, reason: SuspendReason },
    /// The stepper resumed after a yield.
    Resume { node: NodeId },
}

/// Why the stepper yielded control.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SuspendReason {
    /// A hit breakpoint held execution.
    Breakpoint,
    /// The semantics awaited an external effect (LLM call, tool call, MCP).
    AwaitEffect,
    /// The scheduler chose to yield cooperatively.
    Cooperative,
}

/// Sink for the event stream.
///
/// The trait is intentionally simple: implementors decide whether to trace,
/// print, forward over MCP, or record for replay.
pub trait EventSink {
    fn emit(&mut self, event: &Event);
}

/// A no-op sink useful for tests and for evaluators that do not need
/// observation.
#[derive(Debug, Default)]
pub struct NullSink;

impl EventSink for NullSink {
    #[inline]
    fn emit(&mut self, _event: &Event) {}
}

/// One step of evaluation.
///
/// The stepper is the central abstraction: rather than modelling evaluation
/// as an async function that awaits internally, the kit exposes it as a
/// state machine that yields to the outside world at every observable point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome<V> {
    /// The stepper advanced one node and is ready to be stepped again.
    Advanced,
    /// The stepper yielded and is waiting to be resumed.
    Suspended(SuspendReason),
    /// Evaluation completed with a value.
    Done(V),
}

/// Something that can be driven one step at a time.
///
/// Implementors are typically produced by an interpreter over a specific
/// DSL. The trait is deliberately synchronous at its surface: async effects
/// appear as `Suspended(AwaitEffect)` yields and the host drives the effect
/// externally before resuming.
pub trait Stepper {
    type Value;
    type Error;

    /// Runs the next node's semantics.
    fn step(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error>;

    /// Runs steps until completion, suspension, or error.
    fn run_to_yield(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error> {
        loop {
            match self.step()? {
                StepOutcome::Advanced => continue,
                other => return Ok(other),
            }
        }
    }
}

/// Marker trait for AST enums recognised by the kit.
///
/// The derive macro in `dsl-kit-macros` implements this trait automatically;
/// hand-written implementations are supported for advanced use.
pub trait DslNode {
    /// Returns the stable ID assigned to this node.
    fn node_id(&self) -> NodeId;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_push_extends_depth() {
        let ids = IdGen::new();
        let a = ids.node();
        let b = ids.node();

        let root = Path::root();
        assert_eq!(root.depth(), 0);

        let one = root.push(a);
        assert_eq!(one.depth(), 1);

        let two = one.push(b);
        assert_eq!(two.depth(), 2);
    }

    #[test]
    fn id_generator_yields_distinct_ids() {
        let ids = IdGen::new();
        let a = ids.node();
        let b = ids.node();
        assert_ne!(a, b);
    }

    #[test]
    fn null_sink_accepts_events() {
        let mut sink = NullSink;
        let event = Event::VisitPre { node: NodeId(0), path: Path::root() };
        sink.emit(&event);
    }
}
