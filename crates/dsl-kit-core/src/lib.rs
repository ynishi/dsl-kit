//! Engine primitives for `dsl-kit`.
//!
//! This crate defines the observable primitives every DSL built with the kit
//! carries from day one: stable node identifiers, call frame identifiers with
//! depth, iteration counters, root-to-node paths, an event stream, a stepper
//! trait that models evaluation as a state machine, an AST traversal trait,
//! and a structured error type that always carries the location at which the
//! error happened.

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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
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

    /// Returns the node ID at the tip of the path, if any.
    pub fn tip(&self) -> Option<NodeId> {
        self.0.last().copied()
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "/");
        }
        for id in &self.0 {
            write!(f, "/{id}")?;
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

/// Location context attached to every emitted event and to every error.
///
/// A `NodeContext` is a snapshot of "where in the evaluation are we right
/// now?" — enough for a debugger UI to jump to the node, for an error
/// message to point at the source, and for a replay recorder to reconstruct
/// the state.
#[derive(Debug, Clone)]
pub struct NodeContext {
    pub node: NodeId,
    pub path: Path,
    pub frame: Option<CallFrameId>,
    pub depth: u32,
    pub iteration: Option<Iteration>,
}

impl NodeContext {
    /// Builds a context for a node visited without any active frame.
    pub fn at(node: NodeId, path: Path) -> Self {
        Self { node, path, frame: None, depth: 0, iteration: None }
    }
}

impl fmt::Display for NodeContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.node)?;
        if let Some(frame) = self.frame {
            write!(f, " (frame {}, depth {})", frame, self.depth)?;
        }
        if let Some(it) = self.iteration {
            write!(f, " iter {}", it.0)?;
        }
        write!(f, " at {}", self.path)
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
    VisitPre { at: NodeContext },
    /// Emitted after a node's semantics complete.
    VisitPost { at: NodeContext },
    /// A function-like node started a new call frame.
    FrameEnter { at: NodeContext },
    /// A function-like node's frame ended.
    FrameLeave { at: NodeContext },
    /// A loop node advanced to a new iteration.
    IterationTick { at: NodeContext },
    /// The stepper is about to yield to the outside world.
    Suspend { at: NodeContext, reason: SuspendReason },
    /// The stepper resumed after a yield.
    Resume { at: NodeContext },
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

impl fmt::Display for SuspendReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SuspendReason::Breakpoint => write!(f, "breakpoint"),
            SuspendReason::AwaitEffect => write!(f, "await-effect"),
            SuspendReason::Cooperative => write!(f, "cooperative"),
        }
    }
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

// ---------- Errors --------------------------------------------------------

/// Structured error type for engine and evaluator failures.
///
/// Every variant carries a [`NodeContext`] so that error messages, MCP tool
/// responses, and log lines can always answer "at which node did this
/// happen?". Diagnostic codes are stable, machine-readable identifiers under
/// the `dsl_kit::` namespace and are suitable for cross-referencing with an
/// error catalogue.
///
/// The kit uses [`miette`] for the diagnostic surface, so downstream callers
/// can render pretty reports (`miette::Report::new(err)`) without further
/// wiring.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[non_exhaustive]
pub enum EngineError {
    /// The evaluator explicitly aborted the current step.
    #[error("evaluation aborted at {at}: {reason}")]
    #[diagnostic(
        code(dsl_kit::eval::aborted),
        help("The interpreter returned an `Aborted` outcome. Inspect the reason and the node context to locate the abort site.")
    )]
    Aborted {
        at: NodeContext,
        reason: String,
    },

    /// A node's semantics returned an error.
    #[error("evaluator failed at {at}")]
    #[diagnostic(
        code(dsl_kit::eval::failed),
        help("The interpreter returned an error while evaluating this node. The `#[source]` chain points at the underlying failure.")
    )]
    EvalFailed {
        at: NodeContext,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A traversal or stepper reached a node that was expected to have a
    /// specific shape but did not.
    #[error("malformed AST at {at}: {detail}")]
    #[diagnostic(
        code(dsl_kit::ast::malformed),
        help("An AST invariant was violated. This usually means a hand-constructed tree omitted a required child.")
    )]
    Malformed {
        at: NodeContext,
        detail: String,
    },

    /// The host tried to resume a suspended stepper before it had suspended,
    /// or resumed twice for a single suspension.
    #[error("stepper protocol violation at {at}: {detail}")]
    #[diagnostic(
        code(dsl_kit::stepper::protocol),
        help("The stepper's suspend/resume contract was broken. Each `Suspended` outcome must be resumed exactly once before further steps.")
    )]
    StepperProtocol {
        at: NodeContext,
        detail: String,
    },
}

/// Result alias for engine and evaluator operations.
pub type EngineResult<T> = Result<T, EngineError>;

// ---------- Stepper ------------------------------------------------------

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

// ---------- Traversal ----------------------------------------------------

/// Which side of a node's traversal the visitor is currently on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Before descending into the node's children.
    Pre,
    /// After all children have been visited.
    Post,
}

/// AST traversal contract.
///
/// The derive macro generates a [`Walk`] impl for any enum whose variants
/// carry a `NodeId`-typed `id` field and zero or more directly-recursive
/// fields (`T`, `Box<T>`, `Option<T>`, `Vec<T>` where `T` is the enum
/// itself). Hand-written implementations are welcome for advanced shapes.
pub trait Walk: DslNode + Sized {
    /// Direct children of this node, in traversal order.
    fn children(&self) -> Vec<&Self>;

    /// Depth-first pre / post traversal.
    ///
    /// The closure is called twice per node: once with [`Phase::Pre`]
    /// before descending, and once with [`Phase::Post`] after all
    /// descendants have been visited.
    fn walk<F>(&self, visitor: &mut F)
    where
        F: FnMut(&Self, Phase),
    {
        visitor(self, Phase::Pre);
        for child in self.children() {
            child.walk(visitor);
        }
        visitor(self, Phase::Post);
    }

    /// Depth-first traversal with early exit.
    ///
    /// The closure may return `Some(value)` at any point to halt the
    /// traversal and yield that value. Useful for search-shaped queries
    /// such as "find the first node whose id is X".
    fn walk_until<F, T>(&self, visitor: &mut F) -> Option<T>
    where
        F: FnMut(&Self, Phase) -> Option<T>,
    {
        if let Some(t) = visitor(self, Phase::Pre) {
            return Some(t);
        }
        for child in self.children() {
            if let Some(t) = child.walk_until(visitor) {
                return Some(t);
            }
        }
        visitor(self, Phase::Post)
    }

    /// Locates a node by its stable ID.
    fn find_by_id(&self, target: NodeId) -> Option<&Self> {
        self.walk_until(&mut |node, phase| {
            if phase == Phase::Pre && node.node_id() == target {
                Some(node as *const Self)
            } else {
                None
            }
        })
        .map(|ptr| unsafe { &*ptr })
    }
}

/// Mutable counterpart of [`Walk`].
///
/// Generated by the derive macro alongside [`Walk`]. The pre / post
/// invariant is the same; the closure receives `&mut Self` in both phases.
pub trait WalkMut: DslNode + Sized {
    /// Direct children of this node, mutably, in traversal order.
    fn children_mut(&mut self) -> Vec<&mut Self>;

    /// Depth-first pre / post traversal with mutable access.
    fn walk_mut<F>(&mut self, visitor: &mut F)
    where
        F: FnMut(&mut Self, Phase),
    {
        visitor(self, Phase::Pre);
        for child in self.children_mut() {
            child.walk_mut(visitor);
        }
        visitor(self, Phase::Post);
    }
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
        assert_eq!(two.tip(), Some(b));
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
        let event = Event::VisitPre {
            at: NodeContext::at(NodeId(0), Path::root()),
        };
        sink.emit(&event);
    }

    #[test]
    fn engine_error_carries_stable_code() {
        let err = EngineError::Aborted {
            at: NodeContext::at(NodeId(1), Path::root().push(NodeId(1))),
            reason: "user requested".into(),
        };
        // miette exposes the code() method via the Diagnostic trait.
        use miette::Diagnostic;
        assert_eq!(err.code().map(|c| c.to_string()).as_deref(), Some("dsl_kit::eval::aborted"));
    }
}
