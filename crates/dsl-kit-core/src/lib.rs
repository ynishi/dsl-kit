//! Engine primitives for `dsl-kit`.
//!
//! This crate defines the observable primitives every DSL built with the kit
//! carries from day one: stable node identifiers, call frame identifiers with
//! depth, iteration counters, root-to-node paths, an event stream, a stepper
//! trait that models evaluation as a state machine, an AST traversal trait,
//! and a structured error type that always carries the location at which the
//! error happened.
//!
//! ## Architecture
//!
//! - `NodeId` / `CallFrameId` / `Iteration` / `Path` — the observation
//!   primitives every event and every error carry.
//! - `Event` / `EventSink` — the tap between the evaluator and any
//!   observer (tracer, debugger, MCP tool, replay recorder).
//! - `Stepper` / `AsyncStepper` — evaluators expressed as state
//!   machines driven from the outside, with `Suspended` yields
//!   externalising every effect.
//! - `Walk` / `WalkMut` / `DslNode` — the traversal contract every
//!   AST derives via the `#[derive(DslNode)]` macro in `dsl-kit-macros`.
//! - `EngineError` / `NodeContext` — structured errors that always
//!   know where they happened.
//! - `BreakCondition` / `BreakpointSet` — composable boolean predicates
//!   over `NodeContext` used to describe conditional breakpoints.

#![warn(missing_docs)]

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
    /// Creates a fresh generator initialised at zero.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeContext {
    /// The node currently being evaluated.
    pub node: NodeId,
    /// The root-to-node path leading to `node`.
    pub path: Path,
    /// The active call frame, if any.
    pub frame: Option<CallFrameId>,
    /// Depth of the current call / evaluation stack.
    pub depth: u32,
    /// Iteration counter when the surrounding node is loop-shaped.
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
    VisitPre {
        /// Where the observation happened.
        at: NodeContext,
    },
    /// Emitted after a node's semantics complete.
    VisitPost {
        /// Where the observation happened.
        at: NodeContext,
    },
    /// A function-like node started a new call frame.
    FrameEnter {
        /// Where the observation happened.
        at: NodeContext,
    },
    /// A function-like node's frame ended.
    FrameLeave {
        /// Where the observation happened.
        at: NodeContext,
    },
    /// A loop node advanced to a new iteration.
    IterationTick {
        /// Where the observation happened.
        at: NodeContext,
    },
    /// The stepper is about to yield to the outside world.
    Suspend {
        /// Where the observation happened.
        at: NodeContext,
        /// Why the stepper is yielding.
        reason: SuspendReason,
    },
    /// The stepper resumed after a yield.
    Resume {
        /// Where the observation happened.
        at: NodeContext,
    },
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
    /// Consumes one event.
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
        /// Where the abort happened.
        at: NodeContext,
        /// Why the evaluator aborted.
        reason: String,
    },

    /// A node's semantics returned an error.
    #[error("evaluator failed at {at}")]
    #[diagnostic(
        code(dsl_kit::eval::failed),
        help("The interpreter returned an error while evaluating this node. The `#[source]` chain points at the underlying failure.")
    )]
    EvalFailed {
        /// Where the failure happened.
        at: NodeContext,
        /// The underlying error returned by the interpreter.
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
        /// Where the invariant violation was detected.
        at: NodeContext,
        /// Human-readable description of the invariant that failed.
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
        /// Where the protocol violation was detected.
        at: NodeContext,
        /// Human-readable description of the misuse.
        detail: String,
    },
}

/// Result alias for engine and evaluator operations.
pub type EngineResult<T> = Result<T, EngineError>;

/// One entry in an error catalogue: a stable machine-readable code paired
/// with the human-readable help text explaining how to react to it.
///
/// Produced by [`engine_error_catalog`] for the built-in [`EngineError`]
/// variants, and extensible by DSL hosts that want to expose their own
/// codes through the same channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorCatalogEntry {
    /// Stable diagnostic code, e.g. `"dsl_kit::eval::aborted"`.
    pub code: String,
    /// Help text mirroring the miette `help(..)` attribute on the variant.
    pub help: String,
}

/// Returns the built-in [`EngineError`] catalogue.
///
/// Each entry is produced by instantiating a sample of the variant with a
/// dummy [`NodeContext`] and asking miette for its `code()` and `help()`
/// strings — so the catalogue stays in lock-step with the derive attributes
/// on `EngineError` without a second source of truth.
pub fn engine_error_catalog() -> Vec<ErrorCatalogEntry> {
    use miette::Diagnostic;

    fn ctx() -> NodeContext {
        NodeContext::at(NodeId(0), Path::root())
    }

    let samples: Vec<EngineError> = vec![
        EngineError::Aborted { at: ctx(), reason: String::new() },
        EngineError::EvalFailed {
            at: ctx(),
            source: Box::new(std::io::Error::other("")),
        },
        EngineError::Malformed { at: ctx(), detail: String::new() },
        EngineError::StepperProtocol { at: ctx(), detail: String::new() },
    ];

    samples
        .into_iter()
        .filter_map(|e| {
            let code = e.code()?.to_string();
            let help = e.help()?.to_string();
            Some(ErrorCatalogEntry { code, help })
        })
        .collect()
}

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
    /// The stepper yielded and is waiting to be resumed. The `at` field
    /// pinpoints where the suspension happened so hosts can resolve
    /// effects without asking the stepper for extra state.
    Suspended {
        /// Why the stepper is yielding.
        reason: SuspendReason,
        /// Where the suspension happened.
        at: NodeContext,
    },
    /// Evaluation completed with a value.
    Done(V),
}

/// Something that can be driven one step at a time.
///
/// Implementors are typically produced by an interpreter over a specific
/// DSL. The trait is deliberately synchronous at its surface: async effects
/// appear as `Suspended { reason: AwaitEffect, .. }` yields and the host
/// drives the effect externally before resuming.
pub trait Stepper {
    /// Value the stepper produces on completion.
    type Value;
    /// Error type surfaced from `step` and `run_to_yield`.
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

/// Async counterpart of [`Stepper`].
///
/// Implementors let the visit code await external work (network calls,
/// tool invocations, MCP round-trips) inside the semantics rather than
/// externalising them through suspend / resume. When the sync surface is
/// enough — which it usually is — prefer [`Stepper`]: it composes better
/// with debuggers and with the [`drive_async`] helper below.
pub trait AsyncStepper {
    /// Value the stepper produces on completion.
    type Value;
    /// Error type surfaced from `step_async` and `run_to_yield_async`.
    type Error;

    /// Runs the next node's semantics, awaiting any effect the semantics
    /// wants to perform inline.
    fn step_async(
        &mut self,
    ) -> impl std::future::Future<Output = Result<StepOutcome<Self::Value>, Self::Error>>;

    /// Drives async steps until completion, suspension, or error.
    fn run_to_yield_async(
        &mut self,
    ) -> impl std::future::Future<Output = Result<StepOutcome<Self::Value>, Self::Error>>
    {
        async {
            loop {
                match self.step_async().await? {
                    StepOutcome::Advanced => continue,
                    other => return Ok(other),
                }
            }
        }
    }
}

/// Host-side callback invoked whenever a stepper yields with
/// [`SuspendReason::AwaitEffect`].
///
/// The resolver performs the external effect and, on success, is expected
/// to have arranged for the stepper's state to be updated before returning
/// (typically via a channel, a shared map, or a method on the concrete
/// stepper type).
pub trait EffectResolver {
    /// Error surfaced from `resolve` when the effect cannot be produced.
    type Error;

    /// Performs the effect corresponding to `reason` at location `at`.
    fn resolve(
        &mut self,
        at: &NodeContext,
        reason: &SuspendReason,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>>;
}

/// Drives an [`AsyncStepper`] to completion, handing each `AwaitEffect`
/// suspension to a resolver.
///
/// Breakpoint suspensions are passed to the resolver too; a resolver that
/// wants to distinguish them can inspect `reason` and short-circuit.
pub async fn drive_async<S, R>(stepper: &mut S, resolver: &mut R) -> Result<S::Value, S::Error>
where
    S: AsyncStepper,
    R: EffectResolver,
    R::Error: Into<S::Error>,
{
    loop {
        match stepper.step_async().await? {
            StepOutcome::Advanced => continue,
            StepOutcome::Suspended { reason, at } => {
                resolver.resolve(&at, &reason).await.map_err(Into::into)?;
            }
            StepOutcome::Done(v) => return Ok(v),
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

// ---------- Breakpoints --------------------------------------------------

/// Identifier assigned by a [`BreakpointSet`] to each added condition.
///
/// Callers keep the id so they can later remove or disable the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BreakpointId(pub u64);

impl fmt::Display for BreakpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bp{}", self.0)
    }
}

/// A boolean predicate over [`NodeContext`] used to describe conditional
/// breakpoints.
///
/// Conditions can be combined with [`and`](Self::and), [`or`](Self::or),
/// and [`not`](Self::not); the composed tree is evaluated against each
/// context the stepper produces. Because the underlying data
/// (`NodeId` / `Path` / `depth` / `iteration` / call frame) is uniform
/// across every DSL built with the kit, an agent can synthesise a
/// condition purely from the observable event stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BreakCondition {
    /// Matches when the node ID is exactly `id`.
    Node(NodeId),
    /// Matches when the current path has the given prefix.
    PathPrefix(Path),
    /// Matches when the current path equals the given path exactly.
    PathExact(Path),
    /// Matches when the current call-frame depth is at least `n`.
    DepthAtLeast(u32),
    /// Matches when the current call-frame depth is at most `n`.
    DepthAtMost(u32),
    /// Matches when the current call-frame depth equals `n` exactly.
    DepthEquals(u32),
    /// Matches when the iteration counter equals `n`. Nodes without an
    /// active iteration never match this variant.
    Iteration(u64),
    /// Matches when the current call frame equals `frame`.
    CallFrame(CallFrameId),
    /// Matches when any child matches.
    Any(Vec<BreakCondition>),
    /// Matches when every child matches.
    All(Vec<BreakCondition>),
    /// Matches when the inner does not match.
    Not(Box<BreakCondition>),
    /// Always matches (useful as a debugger's "break on every node" mode).
    Always,
    /// Never matches (useful as a disabled placeholder).
    Never,
}

impl BreakCondition {
    /// Convenience constructor.
    pub fn at_node(id: NodeId) -> Self {
        Self::Node(id)
    }

    /// Convenience constructor.
    pub fn under_path(path: Path) -> Self {
        Self::PathPrefix(path)
    }

    /// Convenience constructor.
    pub fn at_path(path: Path) -> Self {
        Self::PathExact(path)
    }

    /// Convenience constructor.
    pub fn at_depth_at_least(n: u32) -> Self {
        Self::DepthAtLeast(n)
    }

    /// Convenience constructor.
    pub fn at_depth_at_most(n: u32) -> Self {
        Self::DepthAtMost(n)
    }

    /// Convenience constructor.
    pub fn at_depth(n: u32) -> Self {
        Self::DepthEquals(n)
    }

    /// Convenience constructor.
    pub fn at_iteration(n: u64) -> Self {
        Self::Iteration(n)
    }

    /// Convenience constructor.
    pub fn in_call_frame(frame: CallFrameId) -> Self {
        Self::CallFrame(frame)
    }

    /// Logical AND.
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::All(mut xs), Self::All(ys)) => {
                xs.extend(ys);
                Self::All(xs)
            }
            (Self::All(mut xs), other) => {
                xs.push(other);
                Self::All(xs)
            }
            (this, Self::All(mut ys)) => {
                ys.insert(0, this);
                Self::All(ys)
            }
            (this, other) => Self::All(vec![this, other]),
        }
    }

    /// Logical OR.
    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::Any(mut xs), Self::Any(ys)) => {
                xs.extend(ys);
                Self::Any(xs)
            }
            (Self::Any(mut xs), other) => {
                xs.push(other);
                Self::Any(xs)
            }
            (this, Self::Any(mut ys)) => {
                ys.insert(0, this);
                Self::Any(ys)
            }
            (this, other) => Self::Any(vec![this, other]),
        }
    }

    /// Logical NOT.
    pub fn not(self) -> Self {
        Self::Not(Box::new(self))
    }

    /// Evaluates the condition against a context.
    pub fn matches(&self, ctx: &NodeContext) -> bool {
        match self {
            Self::Node(id) => ctx.node == *id,
            Self::PathPrefix(prefix) => {
                ctx.path.0.len() >= prefix.0.len()
                    && ctx.path.0[..prefix.0.len()] == prefix.0[..]
            }
            Self::PathExact(path) => ctx.path == *path,
            Self::DepthAtLeast(n) => ctx.depth >= *n,
            Self::DepthAtMost(n) => ctx.depth <= *n,
            Self::DepthEquals(n) => ctx.depth == *n,
            Self::Iteration(n) => ctx.iteration.map(|i| i.0 == *n).unwrap_or(false),
            Self::CallFrame(frame) => ctx.frame == Some(*frame),
            Self::Any(children) => children.iter().any(|c| c.matches(ctx)),
            Self::All(children) => children.iter().all(|c| c.matches(ctx)),
            Self::Not(inner) => !inner.matches(ctx),
            Self::Always => true,
            Self::Never => false,
        }
    }
}

/// A registry of breakpoint conditions and their assigned IDs.
///
/// Hosts typically hold one `BreakpointSet` per debug session, add
/// conditions in response to user or agent instructions, and query
/// [`matches`](Self::matches) against every stepper event they observe.
/// The set is deliberately host-side data: the [`Stepper`] trait knows
/// nothing about it, which keeps the engine's contract minimal.
#[derive(Debug, Default)]
pub struct BreakpointSet {
    next: u64,
    entries: Vec<(BreakpointId, BreakCondition)>,
}

impl BreakpointSet {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self { next: 1, entries: Vec::new() }
    }

    /// Adds a condition and returns its assigned id.
    pub fn add(&mut self, condition: BreakCondition) -> BreakpointId {
        let id = BreakpointId(self.next);
        self.next += 1;
        self.entries.push((id, condition));
        id
    }

    /// Removes an entry by id. Returns whether an entry was found.
    pub fn remove(&mut self, id: BreakpointId) -> bool {
        let before = self.entries.len();
        self.entries.retain(|(entry_id, _)| *entry_id != id);
        self.entries.len() != before
    }

    /// Returns the ids of all entries whose condition matches `ctx`.
    ///
    /// The empty vector means "no breakpoint fires here"; the caller
    /// typically checks `.is_empty()` before deciding whether to yield
    /// with [`SuspendReason::Breakpoint`].
    pub fn matches(&self, ctx: &NodeContext) -> Vec<BreakpointId> {
        self.entries
            .iter()
            .filter(|(_, cond)| cond.matches(ctx))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns whether the set has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries currently registered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterates over the registered `(id, condition)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (BreakpointId, &BreakCondition)> {
        self.entries.iter().map(|(id, c)| (*id, c))
    }
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

    fn ctx_at(node: NodeId, path: Path, depth: u32) -> NodeContext {
        NodeContext { node, path, frame: None, depth, iteration: None }
    }

    #[test]
    fn breakpoint_node_matches_exact_id() {
        let cond = BreakCondition::at_node(NodeId(7));
        assert!(cond.matches(&ctx_at(NodeId(7), Path::root().push(NodeId(7)), 1)));
        assert!(!cond.matches(&ctx_at(NodeId(8), Path::root().push(NodeId(8)), 1)));
    }

    #[test]
    fn breakpoint_path_prefix_matches_descendants() {
        let prefix = Path::root().push(NodeId(1)).push(NodeId(2));
        let cond = BreakCondition::under_path(prefix.clone());
        let inside = ctx_at(NodeId(3), prefix.push(NodeId(3)), 3);
        let outside = ctx_at(NodeId(5), Path::root().push(NodeId(5)), 1);
        assert!(cond.matches(&inside));
        assert!(!cond.matches(&outside));
    }

    #[test]
    fn breakpoint_depth_bounds() {
        let cond = BreakCondition::at_depth_at_least(3).and(BreakCondition::at_depth_at_most(5));
        assert!(!cond.matches(&ctx_at(NodeId(0), Path::root(), 2)));
        assert!(cond.matches(&ctx_at(NodeId(0), Path::root(), 3)));
        assert!(cond.matches(&ctx_at(NodeId(0), Path::root(), 5)));
        assert!(!cond.matches(&ctx_at(NodeId(0), Path::root(), 6)));
    }

    #[test]
    fn breakpoint_composition_flattens() {
        let a = BreakCondition::at_node(NodeId(1));
        let b = BreakCondition::at_node(NodeId(2));
        let c = BreakCondition::at_node(NodeId(3));
        let combined = a.and(b).and(c);
        match combined {
            BreakCondition::All(ref xs) => assert_eq!(xs.len(), 3),
            _ => panic!("expected All variant"),
        }
    }

    #[test]
    fn breakpoint_set_add_matches_remove() {
        let mut set = BreakpointSet::new();
        let hit = set.add(BreakCondition::at_node(NodeId(4)));
        let miss = set.add(BreakCondition::at_node(NodeId(99)));

        let ctx = ctx_at(NodeId(4), Path::root().push(NodeId(4)), 1);
        assert_eq!(set.matches(&ctx), vec![hit]);

        assert!(set.remove(hit));
        assert!(set.matches(&ctx).is_empty());
        assert_eq!(set.len(), 1);
        assert!(set.iter().any(|(id, _)| id == miss));
    }

    #[test]
    fn breakpoint_iteration_and_frame() {
        let cond = BreakCondition::at_iteration(3).and(BreakCondition::in_call_frame(CallFrameId(9)));
        let mut ctx = ctx_at(NodeId(0), Path::root(), 1);
        assert!(!cond.matches(&ctx));
        ctx.iteration = Some(Iteration(3));
        ctx.frame = Some(CallFrameId(9));
        assert!(cond.matches(&ctx));
    }

    // ---- Async stepper smoke test ---------------------------------------

    struct CountdownStepper {
        remaining: u32,
        yielded_once: bool,
    }

    impl AsyncStepper for CountdownStepper {
        type Value = u32;
        type Error = EngineError;

        async fn step_async(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error> {
            if self.remaining == 0 {
                return Ok(StepOutcome::Done(0));
            }
            if !self.yielded_once {
                self.yielded_once = true;
                return Ok(StepOutcome::Suspended {
                    reason: SuspendReason::AwaitEffect,
                    at: NodeContext::at(NodeId(self.remaining as u64), Path::root()),
                });
            }
            self.remaining -= 1;
            self.yielded_once = false;
            Ok(StepOutcome::Advanced)
        }
    }

    struct NoopResolver {
        calls: u32,
    }

    impl EffectResolver for NoopResolver {
        type Error = EngineError;

        async fn resolve(
            &mut self,
            _at: &NodeContext,
            _reason: &SuspendReason,
        ) -> Result<(), Self::Error> {
            self.calls += 1;
            Ok(())
        }
    }

    #[test]
    fn drive_async_runs_stepper_and_calls_resolver() {
        let mut stepper = CountdownStepper { remaining: 3, yielded_once: false };
        let mut resolver = NoopResolver { calls: 0 };
        let value = futures::executor::block_on(drive_async(&mut stepper, &mut resolver))
            .expect("drive succeeded");
        assert_eq!(value, 0);
        assert_eq!(resolver.calls, 3);
    }

    #[test]
    fn engine_error_catalog_covers_every_variant() {
        let entries = engine_error_catalog();
        let codes: Vec<&str> = entries.iter().map(|e| e.code.as_str()).collect();
        assert!(codes.contains(&"dsl_kit::eval::aborted"));
        assert!(codes.contains(&"dsl_kit::eval::failed"));
        assert!(codes.contains(&"dsl_kit::ast::malformed"));
        assert!(codes.contains(&"dsl_kit::stepper::protocol"));
        for entry in &entries {
            assert!(!entry.help.is_empty(), "help empty for {}", entry.code);
        }
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
