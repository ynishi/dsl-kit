//! Engine primitives for `dsl-kit` (v3 engine shape).
//!
//! This crate defines the observable primitives every DSL built with the
//! kit carries: stable node identifiers, call frame identifiers with
//! depth, iteration counters, root-to-node paths, an event stream, the
//! `Stepper` trait modelling evaluation as an externally-driven state
//! machine, an AST traversal contract, and structured errors that always
//! carry the location at which the error happened.
//!
//! ## v3 shape (Round 13)
//!
//! - `Stepper` is a sync trait; async is a host concern. Interpretation is
//!   observable, steppable, and supports first-class fan-out with structured
//!   join through `Par` frames.
//! - Suspensions are identified by `SuspensionId` and can be N-in-flight;
//!   the host resolves each with `Ok(Value)` or `Err(EffectError)`.
//! - Fan-out is expressed by a `Par` frame with a `JoinPolicy` (shape +
//!   fail); reducers fold the collected slots into the parent value.
//! - Cancellation is cooperative; the host drains cancelled ids via
//!   `take_cancellations()` and looks them up in its own runtime-handle
//!   map.
//!
//! The full behavioural specification lives in the rustdoc of
//! [`Engine`], [`Stepper`], and the item-level contracts they link.

#![warn(missing_docs)]

// Re-exported for `#[derive(DslExec)]`-generated code (`Call` payload
// construction); not part of the public API surface.
#[doc(hidden)]
pub use serde_json;

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod drive;
mod engine;
pub mod suggest;
pub use drive::{AsyncEffectResolver, DriveOutcome, EffectResolver, drive, drive_async};
pub use engine::{Ast, Engine, ExecError, LoopDecision, NodeKind};
pub use suggest::{NoopSuggester, Suggester, SuggesterHandle, Suggestion, noop_handle};

// ---------- Observation primitives --------------------------------------

/// Stable identifier assigned to every AST node when the tree is built.
///
/// A `NodeId` is orthogonal to source location: two nodes at the same
/// source span but in different expansions carry different IDs, and the
/// same node retains its ID across evaluation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Identifier for a single activation of a function-like node.
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
#[derive(Debug, Default)]
pub struct IdGen {
    next: AtomicU64,
}

impl IdGen {
    /// Creates a fresh generator initialised at zero.
    pub const fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
        }
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
        Self {
            node,
            path,
            frame: None,
            depth: 0,
            iteration: None,
        }
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

// ---------- Suspension identity ----------------------------------------

/// Stable identifier for one in-flight suspension.
///
/// Allocated by the engine when a `Frame::PendingEffect` leaf is created.
/// The host uses it as the key on [`Stepper::resolve`] and as the map key
/// for its own `SuspensionId -> runtime handle` bookkeeping when
/// cooperative cancellation is required. Monotonic within one `Stepper`
/// instance; never reused, even after cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuspensionId(pub u64);

impl fmt::Display for SuspensionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// One live suspension.
#[derive(Debug, Clone)]
pub struct Pending {
    /// Stable identifier.
    pub id: SuspensionId,
    /// Why the engine yielded.
    pub reason: SuspendReason,
    /// Where the suspension happened.
    pub at: NodeContext,
}

// ---------- Suspend reason ---------------------------------------------

/// Why the engine yielded control.
///
/// Cancellation runtime handles are NOT stored here — the host maintains
/// its own `SuspensionId -> handle` map.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SuspendReason {
    /// A first-class external effect (LLM call, tool invocation, …).
    Call {
        /// Host-defined effect descriptor.
        spec: CallSpec,
    },
    /// A registered breakpoint fired.
    Breakpoint,
    /// The scheduler chose to yield cooperatively.
    Cooperative,
    /// Host-defined effect the DSL wants to model beyond `Call`.
    User {
        /// Short tag identifying the effect kind.
        tag: Cow<'static, str>,
    },
}

impl fmt::Display for SuspendReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SuspendReason::Call { spec } => write!(f, "call({})", spec.label),
            SuspendReason::Breakpoint => write!(f, "breakpoint"),
            SuspendReason::Cooperative => write!(f, "cooperative"),
            SuspendReason::User { tag } => write!(f, "user({tag})"),
        }
    }
}

/// Host-defined effect descriptor. Opaque to the engine.
#[derive(Debug, Clone)]
pub struct CallSpec {
    /// Short human-readable label (also fed into the DSL's canned
    /// responses when applicable).
    pub label: String,
    /// Effect payload; the engine stores it verbatim and the host
    /// interprets it per the DSL's semantics.
    pub payload: serde_json::Value,
}

// ---------- Event stream -----------------------------------------------

/// One observation from the evaluator.
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
    /// A suspension was cancelled (tracer convenience; the authoritative
    /// channel is [`Stepper::take_cancellations`]).
    Cancel {
        /// The cancelled suspension.
        id: SuspensionId,
    },
}

/// Sink for the event stream.
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

/// Silent event sink that counts each event kind.
///
/// Used by the engine as its default observation channel so hosts
/// can inspect basic activity without wiring a custom sink.
#[derive(Debug, Default, Clone, Copy)]
pub struct CountingSink {
    /// Number of `VisitPre` events observed.
    pub visit_pre: u32,
    /// Number of `VisitPost` events observed.
    pub visit_post: u32,
    /// Number of `FrameEnter` events observed.
    pub frame_enter: u32,
    /// Number of `FrameLeave` events observed.
    pub frame_leave: u32,
    /// Number of `IterationTick` events observed.
    pub iteration_tick: u32,
    /// Number of `Suspend` events observed.
    pub suspend: u32,
    /// Number of `Resume` events observed.
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
    /// Returns a single-line human-readable event histogram.
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

/// Recording event sink that keeps every observed [`Event`] in order.
///
/// Complements [`CountingSink`] (built-in on `Engine`) by preserving the
/// full event history for downstream inspection — debuggers, tracers,
/// test assertions, etc. Attach one to an engine via
/// `Engine::attach_sink` and drain / snapshot it between mutations.
#[derive(Debug, Default, Clone)]
pub struct BufferingSink {
    events: Vec<Event>,
}

impl BufferingSink {
    /// Creates an empty, unbounded buffer.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Creates an empty buffer with pre-allocated capacity `cap`.
    ///
    /// The buffer is still logically unbounded; `cap` is only a
    /// hint to avoid early re-allocations.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            events: Vec::with_capacity(cap),
        }
    }

    /// Returns a slice view of the recorded events without draining.
    pub fn snapshot(&self) -> &[Event] {
        &self.events
    }

    /// Takes every recorded event and clears the buffer.
    pub fn drain(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// Discards every recorded event without returning them.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Number of recorded events currently held.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True when no events are held.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl EventSink for BufferingSink {
    fn emit(&mut self, event: &Event) {
        self.events.push(event.clone());
    }
}

// ---------- Errors ------------------------------------------------------

/// Structured error type for engine and evaluator failures.
///
/// Every applicable variant carries a [`NodeContext`] or a
/// [`SuspensionId`] / [`ReducerId`] so that error messages, MCP tool
/// responses, and log lines can always answer "at which node / for
/// which id did this happen?". Diagnostic codes are stable,
/// machine-readable identifiers under the `dsl_kit::` namespace.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[non_exhaustive]
pub enum EngineError {
    /// The evaluator explicitly aborted the current step.
    #[error("evaluation aborted at {at}: {reason}")]
    #[diagnostic(
        code(dsl_kit::eval::aborted),
        help(
            "The interpreter returned an `Aborted` outcome. Inspect the reason and the node context to locate the abort site."
        )
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
        help(
            "The interpreter returned an error while evaluating this node. The `#[source]` chain points at the underlying failure."
        )
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
        help(
            "An AST invariant was violated. This usually means a hand-constructed tree omitted a required child."
        )
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
        help(
            "The stepper's suspend/resume contract was broken. Each `PendingEffect` leaf must be resolved exactly once before further steps observe its value."
        )
    )]
    StepperProtocol {
        /// Where the protocol violation was detected.
        at: NodeContext,
        /// Human-readable description of the misuse.
        detail: String,
    },

    /// The host called `resolve` with a `SuspensionId` the engine does
    /// not know about (never created, already resolved, or belonging to
    /// a cancelled leg). Hosts treat this as a benign no-op — it
    /// happens routinely for cancelled effects whose external work
    /// completes after abort.
    #[error("unknown suspension {id}")]
    #[diagnostic(
        code(dsl_kit::stepper::unknown_suspension),
        help(
            "The host called resolve() with a SuspensionId the engine does not know. This is the expected outcome for a late response arriving after the effect's branch was cancelled."
        )
    )]
    UnknownSuspension {
        /// The unknown suspension id.
        id: SuspensionId,
    },

    /// A `ParFrame` referenced a `ReducerId` that is not registered in
    /// the active [`ReducerRegistry`].
    #[error("unknown reducer {id:?}")]
    #[diagnostic(
        code(dsl_kit::reducer::unknown),
        help(
            "The ParFrame's reducer_id was not found in the ReducerRegistry. Register the reducer at Stepper build time or fix the AST to reference a known id."
        )
    )]
    UnknownReducer {
        /// The unknown reducer id.
        id: ReducerId,
    },

    /// An `Apply` node referenced an [`OpId`] that is not registered in
    /// the active [`OpRegistry`].
    #[error("unknown op {id:?}")]
    #[diagnostic(
        code(dsl_kit::op::unknown),
        help(
            "The Apply node's op_id was not found in the OpRegistry. Register the op at Stepper build time (Engine::new_with_ops) or fix the AST to reference a known id."
        )
    )]
    UnknownOp {
        /// The unknown op id.
        id: OpId,
    },
}

/// Result alias for engine and evaluator operations.
pub type EngineResult<T> = Result<T, EngineError>;

/// One entry in an error catalogue: a stable machine-readable code paired
/// with the human-readable help text explaining how to react to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorCatalogEntry {
    /// Stable diagnostic code, e.g. `"dsl_kit::eval::aborted"`.
    pub code: String,
    /// Help text mirroring the miette `help(..)` attribute on the variant.
    pub help: String,
}

/// Returns the built-in [`EngineError`] catalogue.
pub fn engine_error_catalog() -> Vec<ErrorCatalogEntry> {
    use miette::Diagnostic;

    fn ctx() -> NodeContext {
        NodeContext::at(NodeId(0), Path::root())
    }

    let samples: Vec<EngineError> = vec![
        EngineError::Aborted {
            at: ctx(),
            reason: String::new(),
        },
        EngineError::EvalFailed {
            at: ctx(),
            source: Box::new(std::io::Error::other("")),
        },
        EngineError::Malformed {
            at: ctx(),
            detail: String::new(),
        },
        EngineError::StepperProtocol {
            at: ctx(),
            detail: String::new(),
        },
        EngineError::UnknownSuspension {
            id: SuspensionId(0),
        },
        EngineError::UnknownReducer {
            id: ReducerId(String::new()),
        },
        EngineError::UnknownOp {
            id: OpId(String::new()),
        },
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

// ---------- Step outcome & Stepper -------------------------------------

/// Result of one [`Stepper::step`] call.
///
/// The three states name the driver's next action verbatim:
///
/// - `Ready` — the engine made progress and can make more without host
///   input. Call `step()` again immediately.
/// - `Blocked { newly_pending }` — the engine did everything it can
///   without host input. Do NOT call `step()` again until at least one
///   `resolve` has been fed back, or a cancellation observed via
///   `take_cancellations()`.
/// - `Done(v)` — the root frame produced a value; interpretation is
///   complete.
///
/// Terminal failure is returned as `Err(Stepper::Error)` from `step()`;
/// there is no `Failed` variant. Cancellations queued during the failing
/// step are still drainable via `take_cancellations()` after the `Err`.
#[derive(Debug, Clone)]
pub enum StepOutcome<V> {
    /// Engine advanced; keep looping.
    Ready,
    /// Engine is blocked awaiting host input. `newly_pending` names
    /// suspensions created this call (may be empty when the block is
    /// on already-live pending).
    Blocked {
        /// Suspensions newly created during this `step` call.
        newly_pending: smallvec::SmallVec<[Pending; 1]>,
    },
    /// Interpretation completed with a value.
    Done(V),
}

/// Something that can be driven one step at a time from the outside.
///
/// v3 shape: sync trait, generic in `V` / `C` / `D` / `EffectError` /
/// `Error`. Async is a host concern — a host wrapping the stepper in
/// `FuturesUnordered<ResolverFuture>` is the canonical async story.
pub trait Stepper {
    /// Value the stepper produces on completion.
    type Value;
    /// DSL-specific node cursor state stored inside `Frame::Node`.
    type Cursor;
    /// DSL-specific state delta / write log stored inside `Env` and
    /// merged by reducers at join.
    type Delta;
    /// DSL-specific effect-side failure surfaced via `resolve`.
    type EffectError: std::error::Error + Send + Sync + 'static;
    /// DSL / engine-level error surfaced from `step` and `resolve`.
    type Error: From<Self::EffectError> + std::error::Error + Send + Sync + 'static;

    /// Advance the interpretation as far as possible without host input.
    fn step(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error>;

    /// Feed a response back for one pending suspension.
    ///
    /// Order-independent; the host may resolve pending IDs in any
    /// order. Unknown IDs return an error the host may treat as a
    /// benign no-op (see [`EngineError::UnknownSuspension`]).
    fn resolve(
        &mut self,
        id: SuspensionId,
        result: Result<Self::Value, Self::EffectError>,
    ) -> Result<(), Self::Error>;

    /// Snapshot of every suspension currently blocking the tree.
    /// Excludes ids under `Frame::Cancelled` sub-trees. Order matches
    /// DFS pre-order over the frame tree.
    fn pending(&self) -> &[Pending];

    /// Drain the set of suspensions the engine has just cancelled.
    /// Must be called after every `Ok(Ready)` AND every `Err(_)`
    /// return from `step()`.
    fn take_cancellations(&mut self) -> Vec<SuspensionId>;

    /// Root of the live frame tree. Debugger read.
    fn frame_tree(&self) -> &FrameTree<Self::Value, Self::Cursor, Self::Delta, Self::EffectError>;

    /// True when the root has produced a value.
    fn is_done(&self) -> bool;
}

// ---------- Frame tree --------------------------------------------------

/// Which child slot in a `ParFrame` a value corresponds to.
pub type ChildIndex = usize;

/// Why a sub-tree was cancelled.
#[derive(Debug, Clone)]
pub enum CancelReason {
    /// Cancelled because the enclosing `Par` policy fired
    /// (`Any` / `FirstK` winner selected).
    ParPolicyFired,
    /// Cancelled because a sibling failed (FailFast).
    SiblingFailed,
}

/// One node in the live frame tree.
#[derive(Debug)]
pub enum Frame<V, C, D, EE> {
    /// An interior interpretation node currently executing.
    Node {
        /// AST node this frame is interpreting.
        node: NodeId,
        /// Shared environment snapshot.
        env: EnvRef<D>,
        /// DSL-specific cursor (opaque to the engine).
        cursor: C,
    },
    /// A fan-out root. Children live on the enclosing `FrameTree.kids`.
    Par(ParFrame<V, D, EE>),
    /// A labelled section wrapper (no fan-out semantics).
    Scope {
        /// Section label.
        label: String,
        /// Shared environment snapshot.
        env: EnvRef<D>,
    },
    /// A leaf blocked on an external effect.
    PendingEffect {
        /// Stable id (also present in `Stepper::pending`).
        id: SuspensionId,
    },
    /// A leaf whose value is ready and waiting to be consumed by its
    /// parent frame's continuation.
    Value(V),
    /// A sub-tree the engine has cancelled. Never selected for `step()`
    /// progress; kept so the debugger can render it.
    Cancelled {
        /// Why the sub-tree was cancelled.
        reason: CancelReason,
    },
}

/// Tree of frames. `kids[i]` under a `Par` root is the `i`-th sibling
/// child, matched positionally with `ParFrame.slots[i]` and
/// `ParFrame.deltas[i]`.
#[derive(Debug)]
pub struct FrameTree<V, C, D, EE> {
    /// The frame at this node.
    pub root: Frame<V, C, D, EE>,
    /// Child sub-trees.
    pub kids: Vec<FrameTree<V, C, D, EE>>,
}

/// Fan-out metadata.
///
/// `children` are stored on the enclosing `FrameTree.kids`; `slots`,
/// `deltas`, and `completion_order` are indexed against that same
/// positional list.
#[derive(Debug)]
pub struct ParFrame<V, D, EE> {
    /// Policy (shape + fail).
    pub policy: JoinPolicy,
    /// `slots[i]` is `Some(v)` once child i completed successfully;
    /// `None` while still running, pending, or cancelled without a
    /// value. FailFast: reroutes failure to `Err` before reducer.
    /// CollectAll: per-child failures live in `failures[i]`; combine
    /// slot + failure at reducer-invocation time.
    pub slots: Vec<Option<V>>,
    /// Per-child effect-side failures (CollectAll only). `Some(e)`
    /// once the child resolved with `Err(e)`; always `None` under
    /// FailFast (failures reroute to `Err` before the reducer runs).
    pub failures: Vec<Option<EE>>,
    /// Per-child accumulated delta.
    pub deltas: Vec<Option<D>>,
    /// Order in which children completed successfully.
    pub completion_order: Vec<ChildIndex>,
    /// Registry key for the reducer.
    pub reducer_id: ReducerId,
    /// Resolved reducer handle (FailFast or CollectAll, matching
    /// `policy.fail`).
    pub reducer: ReducerHandle<V, D, EE>,
    /// True once the shape has fired.
    pub joined: bool,
}

// ---------- Join policy -------------------------------------------------

/// Full policy for a `ParFrame`: success-side `shape` + failure-side
/// `fail`.
#[derive(Debug, Clone, Copy)]
pub struct JoinPolicy {
    /// Success completion condition.
    pub shape: JoinShape,
    /// Failure handling.
    pub fail: FailPolicy,
}

/// Success-side completion condition.
#[derive(Debug, Clone, Copy)]
pub enum JoinShape {
    /// Every child must complete successfully.
    All,
    /// The first successful child wins; remaining children are cancelled.
    Any,
    /// The first `k` successful children win; remaining children are
    /// cancelled.
    FirstK(usize),
}

/// Failure-side policy.
#[derive(Debug, Clone, Copy)]
pub enum FailPolicy {
    /// A single child failure aborts the `Par` immediately.
    FailFast,
    /// Failures are collected into slots; shape completion still decides
    /// based on the count of successes.
    CollectAll,
}

// ---------- Reducer trait + registry -----------------------------------

/// Serialisable name for a registered reducer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReducerId(pub String);

impl fmt::Display for ReducerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ReducerId {
    fn from(s: &str) -> Self {
        ReducerId(s.to_string())
    }
}

impl From<String> for ReducerId {
    fn from(s: String) -> Self {
        ReducerId(s)
    }
}

/// FailFast reducer. Sees successful joins only.
pub trait Reducer<V, D>: Send + Sync {
    /// Fold slots + deltas + completion order into a joined `(V, D)`.
    fn reduce(
        &self,
        slots: &[Option<V>],
        deltas: &[Option<D>],
        winners: &[ChildIndex],
    ) -> Result<(V, D), EngineError>;
}

/// CollectAll reducer. Sees the full success/failure slot vector.
pub trait ReducerCollectAll<V, D, EffectError>: Send + Sync {
    /// Fold slots (each `Some(Ok)` / `Some(Err)` / `None`) + deltas +
    /// completion order into a joined `(V, D)`.
    fn reduce(
        &self,
        slots: &[Option<Result<V, EffectError>>],
        deltas: &[Option<D>],
        winners: &[ChildIndex],
    ) -> Result<(V, D), EngineError>;
}

/// Runtime handle for the resolved reducer of a `ParFrame`.
///
/// Which variant is present is dictated by `JoinPolicy.fail`. The
/// `CollectAll` variant is typed directly in the effect-error `EE`
/// so the engine can invoke `reduce` on a `&[Option<Result<V, EE>>]`
/// slot vector at fold time without any downcast.
pub enum ReducerHandle<V, D, EE> {
    /// FailFast-typed reducer.
    FailFast(Arc<dyn Reducer<V, D>>),
    /// CollectAll-typed reducer, typed in the effect-error `EE`.
    CollectAll(Arc<dyn ReducerCollectAll<V, D, EE>>),
}

impl<V, D, EE> fmt::Debug for ReducerHandle<V, D, EE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReducerHandle::FailFast(_) => f.write_str("ReducerHandle::FailFast(..)"),
            ReducerHandle::CollectAll(_) => f.write_str("ReducerHandle::CollectAll(..)"),
        }
    }
}

impl<V, D, EE> Clone for ReducerHandle<V, D, EE> {
    fn clone(&self) -> Self {
        match self {
            ReducerHandle::FailFast(a) => ReducerHandle::FailFast(a.clone()),
            ReducerHandle::CollectAll(a) => ReducerHandle::CollectAll(a.clone()),
        }
    }
}

/// Runtime lookup table.
///
/// Constructed by the host at Stepper build time; a DSL typically ships
/// a helper `default_registry()` that registers its standard reducers
/// under the canonical ids (`"reduce_all_ordered"` etc.).
pub struct ReducerRegistry<V, D, EffectError> {
    fail_fast: HashMap<ReducerId, Arc<dyn Reducer<V, D>>>,
    collect_all: HashMap<ReducerId, Arc<dyn ReducerCollectAll<V, D, EffectError>>>,
}

impl<V, D, EE> Default for ReducerRegistry<V, D, EE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, D, EE> ReducerRegistry<V, D, EE> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            fail_fast: HashMap::new(),
            collect_all: HashMap::new(),
        }
    }

    /// Registers a FailFast-typed reducer under `id`.
    pub fn register_fail_fast(
        &mut self,
        id: impl Into<ReducerId>,
        reducer: Arc<dyn Reducer<V, D>>,
    ) {
        self.fail_fast.insert(id.into(), reducer);
    }

    /// Registers a CollectAll-typed reducer under `id`.
    pub fn register_collect_all(
        &mut self,
        id: impl Into<ReducerId>,
        reducer: Arc<dyn ReducerCollectAll<V, D, EE>>,
    ) {
        self.collect_all.insert(id.into(), reducer);
    }

    /// Resolves the reducer for a `ParFrame` at construction time.
    ///
    /// The `fail` argument disambiguates when the same id has been
    /// registered under both traits. Returns
    /// `EngineError::UnknownReducer` on miss.
    pub fn resolve(
        &self,
        id: &ReducerId,
        fail: FailPolicy,
    ) -> Result<ReducerHandle<V, D, EE>, EngineError> {
        match fail {
            FailPolicy::FailFast => self
                .fail_fast
                .get(id)
                .cloned()
                .map(ReducerHandle::FailFast)
                .ok_or_else(|| EngineError::UnknownReducer { id: id.clone() }),
            FailPolicy::CollectAll => self
                .collect_all
                .get(id)
                .cloned()
                .map(ReducerHandle::CollectAll)
                .ok_or_else(|| EngineError::UnknownReducer { id: id.clone() }),
        }
    }
}

// ---------- Op trait + registry -----------------------------------------

/// Serialisable name for a registered op.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpId(pub String);

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for OpId {
    fn from(s: &str) -> Self {
        OpId(s.to_string())
    }
}

impl From<String> for OpId {
    fn from(s: String) -> Self {
        OpId(s)
    }
}

/// A pure value fold applied by an `Apply` node once every child
/// argument has produced a value.
///
/// Ops never suspend: effects belong in `Call` children, whose values
/// arrive here as ordinary `args` entries. Domain failures (division by
/// zero, type mismatch on a dynamic value) are reported as
/// [`EngineError::EvalFailed`] with the node context supplied by the
/// engine at the call site.
pub trait Op<V>: Send + Sync {
    /// Fold the evaluated child values into the node's value.
    ///
    /// `args` holds one entry per `Apply` child, in declaration order.
    /// Zero-child `Apply` nodes invoke the op with an empty slice.
    fn apply(&self, node: NodeId, args: &[V]) -> Result<V, EngineError>;
}

/// Runtime handle for the resolved op of an `Apply` frame.
pub type OpHandle<V> = Arc<dyn Op<V>>;

/// Runtime lookup table for [`Op`]s, the `Apply` counterpart of
/// [`ReducerRegistry`].
///
/// Constructed by the host at Stepper build time; a DSL typically ships
/// a helper that registers its standard ops under canonical ids
/// (`"add"`, `"mul"`, ...). Passed to [`Engine::new_with_ops`]; DSLs
/// with no `Apply` nodes can stay on [`Engine::new`], which uses an
/// empty registry.
///
/// [`Engine::new`]: crate::Engine::new
/// [`Engine::new_with_ops`]: crate::Engine::new_with_ops
pub struct OpRegistry<V> {
    ops: HashMap<OpId, OpHandle<V>>,
}

impl<V> Default for OpRegistry<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> OpRegistry<V> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
        }
    }

    /// Registers an op under `id`.
    pub fn register(&mut self, id: impl Into<OpId>, op: OpHandle<V>) {
        self.ops.insert(id.into(), op);
    }

    /// Resolves the op for an `Apply` frame at spawn / validation time.
    /// Returns [`EngineError::UnknownOp`] on miss.
    pub fn resolve(&self, id: &OpId) -> Result<OpHandle<V>, EngineError> {
        self.ops
            .get(id)
            .cloned()
            .ok_or_else(|| EngineError::UnknownOp { id: id.clone() })
    }
}

// ---------- Derived execution (DslExec / DslSemantics / DerivedAst) ----

/// Mechanical node classification, normally supplied by
/// `#[derive(DslExec)]` from `dsl-kit-macros`.
///
/// This is the derivable half of an [`Ast`]: which engine [`NodeKind`]
/// each variant maps to, and the payload of `value`-marked literal
/// variants. The semantic half (value type, truthiness, bindings)
/// lives in [`DslSemantics`]; [`DerivedAst`] zips the two into an
/// engine-ready [`Ast`].
pub trait DslExec: Walk {
    /// Payload type of `#[dsl_exec(value)]` variants. `()` when the
    /// enum has no value variants.
    type LitValue: Clone + fmt::Debug;

    /// Engine classification of this node.
    fn exec_kind(&self) -> NodeKind;

    /// Payload of a value variant; `None` for every other variant.
    fn exec_literal(&self) -> Option<Self::LitValue>;
}

/// The semantic half an author writes next to `#[derive(DslExec)]`:
/// the runtime value type and the hooks whose meaning the kit cannot
/// derive. Every hook mirrors its [`Ast`] counterpart and keeps the
/// same default, so a DSL only implements what its node kinds use.
pub trait DslSemantics {
    /// Runtime value type produced by the DSL.
    type Value: Clone + fmt::Debug;
    /// Per-child state delta accumulated across a `Par` and extended
    /// by `Bind`.
    type Delta: Clone + Default + fmt::Debug;
    /// Effect-side failure the host reports through `resolve`.
    type EffectError: std::error::Error + Clone + Send + Sync + 'static;
    /// Per-frame cursor surfaced through the projected frame tree.
    type Cursor: Clone + fmt::Debug + Default;

    /// See [`Ast::unit_value`].
    fn unit_value(&self) -> Self::Value;

    /// See [`Ast::truthy`].
    fn truthy(&self, _value: &Self::Value) -> Option<bool> {
        None
    }

    /// See [`Ast::continue_loop`].
    fn continue_loop(&self, _node: NodeId, _last: &Self::Value, _iteration: usize) -> LoopDecision {
        LoopDecision::Break
    }

    /// See [`Ast::bind_delta`].
    fn bind_delta(&self, _name: &str, _value: &Self::Value) -> Option<Self::Delta> {
        None
    }

    /// See [`Ast::lookup`].
    fn lookup(&self, _delta: &Self::Delta, _name: &str) -> Option<Self::Value> {
        None
    }
}

/// Engine-ready [`Ast`] assembled from a `#[derive(DslExec)]` node
/// type plus a [`DslSemantics`] implementation.
///
/// Construction walks the tree once to build the `NodeId -> node`
/// lookup; `node_kind` / `literal` then delegate to the derived
/// classification and every semantic hook to `S`.
pub struct DerivedAst<'a, N, S> {
    root: &'a N,
    by_id: HashMap<NodeId, &'a N>,
    sem: S,
}

impl<'a, N: DslExec, S> DerivedAst<'a, N, S> {
    /// Builds the lookup for `root` and attaches the semantics.
    pub fn new(root: &'a N, sem: S) -> Self {
        fn collect<'a, N: Walk>(node: &'a N, map: &mut HashMap<NodeId, &'a N>) {
            map.insert(node.node_id(), node);
            for child in node.children() {
                collect(child, map);
            }
        }
        let mut by_id = HashMap::new();
        collect(root, &mut by_id);
        Self { root, by_id, sem }
    }

    fn node(&self, id: NodeId) -> &'a N {
        self.by_id.get(&id).copied().expect("unknown NodeId")
    }
}

impl<N, S> Ast for DerivedAst<'_, N, S>
where
    N: DslExec,
    S: DslSemantics,
    S::Value: From<N::LitValue>,
{
    type Value = S::Value;
    type Delta = S::Delta;
    type EffectError = S::EffectError;
    type Cursor = S::Cursor;

    fn root(&self) -> NodeId {
        self.root.node_id()
    }

    fn node_kind(&self, id: NodeId) -> NodeKind {
        self.node(id).exec_kind()
    }

    fn unit_value(&self) -> Self::Value {
        self.sem.unit_value()
    }

    fn truthy(&self, value: &Self::Value) -> Option<bool> {
        self.sem.truthy(value)
    }

    fn continue_loop(&self, node: NodeId, last: &Self::Value, iteration: usize) -> LoopDecision {
        self.sem.continue_loop(node, last, iteration)
    }

    fn bind_delta(&self, name: &str, value: &Self::Value) -> Option<Self::Delta> {
        self.sem.bind_delta(name, value)
    }

    fn lookup(&self, delta: &Self::Delta, name: &str) -> Option<Self::Value> {
        self.sem.lookup(delta, name)
    }

    fn literal(&self, node: NodeId) -> Option<Self::Value> {
        self.node(node).exec_literal().map(S::Value::from)
    }
}

// ---------- Environment sharing ----------------------------------------

/// Shared environment handle.
///
/// Reads are cheap `Arc` clones. Writes accumulate in a per-child `D`
/// (state delta) inside a `Par` and are merged by the reducer at join.
#[derive(Debug)]
pub struct EnvRef<D>(pub Arc<Env<D>>);

impl<D> Clone for EnvRef<D> {
    fn clone(&self) -> Self {
        EnvRef(self.0.clone())
    }
}

/// The `Env` a `Frame::Node` or `Frame::Scope` holds. Generic over the
/// DSL-owned `D` (state delta); the engine does not interpret `D`.
#[derive(Debug)]
pub struct Env<D> {
    /// The DSL's own state delta / bindings snapshot.
    pub delta: D,
    /// Optional parent for lexical lookup.
    pub parent: Option<Arc<Env<D>>>,
}

// ---------- Marker traits + traversal ----------------------------------

/// Marker trait for AST enums recognised by the kit.
pub trait DslNode {
    /// Returns the stable ID assigned to this node.
    fn node_id(&self) -> NodeId;

    /// Returns the enum variant name of this node as a static string
    /// (e.g. `"Seq"`, `"Par"`, `"Call"`).
    ///
    /// The name matches the variant's Rust source ident and lines up
    /// with `dsl_kit_schema::VariantSchema::name` when the type also
    /// derives `DslSchema`. Consumers use this to correlate a runtime
    /// node with its type-level schema entry — schema-aware lint
    /// rules, editor tooling, and any other component that needs to
    /// go from an AST value to "which variant is this" without
    /// downcasting.
    fn variant_name(&self) -> &'static str;
}

// ---------- Breakpoints -------------------------------------------------

/// Identifier assigned by a [`BreakpointSet`] to each added condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BreakpointId(pub u64);

impl fmt::Display for BreakpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bp{}", self.0)
    }
}

/// A boolean predicate over [`NodeContext`] used to describe conditional
/// breakpoints.
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
    /// Matches when the iteration counter equals `n`.
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

    /// Evaluates the condition against a context.
    pub fn matches(&self, ctx: &NodeContext) -> bool {
        match self {
            Self::Node(id) => ctx.node == *id,
            Self::PathPrefix(prefix) => {
                ctx.path.0.len() >= prefix.0.len() && ctx.path.0[..prefix.0.len()] == prefix.0[..]
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

impl std::ops::Not for BreakCondition {
    type Output = Self;

    /// Logical NOT: `!cond` matches exactly when `cond` does not.
    fn not(self) -> Self {
        Self::Not(Box::new(self))
    }
}

/// A registry of breakpoint conditions and their assigned IDs.
#[derive(Debug, Default)]
pub struct BreakpointSet {
    next: u64,
    entries: Vec<(BreakpointId, BreakCondition)>,
}

impl BreakpointSet {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self {
            next: 1,
            entries: Vec::new(),
        }
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

// ---------- Traversal ---------------------------------------------------

/// Which side of a node's traversal the visitor is currently on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Before descending into the node's children.
    Pre,
    /// After all children have been visited.
    Post,
}

/// AST traversal contract.
pub trait Walk: DslNode + Sized {
    /// Direct children of this node, in traversal order.
    fn children(&self) -> Vec<&Self>;

    /// Depth-first pre / post traversal.
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
        NodeContext {
            node,
            path,
            frame: None,
            depth,
            iteration: None,
        }
    }

    #[test]
    fn breakpoint_node_matches_exact_id() {
        let cond = BreakCondition::at_node(NodeId(7));
        assert!(cond.matches(&ctx_at(NodeId(7), Path::root().push(NodeId(7)), 1)));
        assert!(!cond.matches(&ctx_at(NodeId(8), Path::root().push(NodeId(8)), 1)));
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
    fn engine_error_carries_stable_code() {
        let err = EngineError::Aborted {
            at: NodeContext::at(NodeId(1), Path::root().push(NodeId(1))),
            reason: "user requested".into(),
        };
        use miette::Diagnostic;
        assert_eq!(
            err.code().map(|c| c.to_string()).as_deref(),
            Some("dsl_kit::eval::aborted")
        );
    }

    #[test]
    fn engine_error_catalog_covers_every_variant() {
        let entries = engine_error_catalog();
        let codes: Vec<&str> = entries.iter().map(|e| e.code.as_str()).collect();
        assert!(codes.contains(&"dsl_kit::eval::aborted"));
        assert!(codes.contains(&"dsl_kit::eval::failed"));
        assert!(codes.contains(&"dsl_kit::ast::malformed"));
        assert!(codes.contains(&"dsl_kit::stepper::protocol"));
        assert!(codes.contains(&"dsl_kit::stepper::unknown_suspension"));
        assert!(codes.contains(&"dsl_kit::reducer::unknown"));
        for entry in &entries {
            assert!(!entry.help.is_empty(), "help empty for {}", entry.code);
        }
    }

    #[test]
    fn reducer_id_from_str_and_string() {
        let a: ReducerId = "foo".into();
        let b: ReducerId = String::from("foo").into();
        assert_eq!(a, b);
    }

    #[test]
    fn reducer_registry_resolves_fail_fast_and_collect_all_independently() {
        // A trivial FailFast reducer that returns the first non-None
        // slot verbatim.
        struct FirstNonNone;
        impl Reducer<i64, ()> for FirstNonNone {
            fn reduce(
                &self,
                slots: &[Option<i64>],
                _deltas: &[Option<()>],
                _winners: &[ChildIndex],
            ) -> Result<(i64, ()), EngineError> {
                let v = slots.iter().find_map(|s| *s).unwrap_or(0);
                Ok((v, ()))
            }
        }

        // A trivial CollectAll reducer.
        struct CountOk;
        impl ReducerCollectAll<i64, (), std::io::Error> for CountOk {
            fn reduce(
                &self,
                slots: &[Option<Result<i64, std::io::Error>>],
                _deltas: &[Option<()>],
                _winners: &[ChildIndex],
            ) -> Result<(i64, ()), EngineError> {
                let count = slots.iter().filter(|s| matches!(s, Some(Ok(_)))).count();
                Ok((count as i64, ()))
            }
        }

        let mut reg: ReducerRegistry<i64, (), std::io::Error> = ReducerRegistry::new();
        reg.register_fail_fast("first", Arc::new(FirstNonNone));
        reg.register_collect_all("count_ok", Arc::new(CountOk));

        let ff = reg.resolve(&"first".into(), FailPolicy::FailFast).unwrap();
        assert!(matches!(ff, ReducerHandle::FailFast(_)));
        let ca = reg
            .resolve(&"count_ok".into(), FailPolicy::CollectAll)
            .unwrap();
        assert!(matches!(ca, ReducerHandle::CollectAll(_)));

        let miss = reg.resolve(&"nope".into(), FailPolicy::FailFast);
        assert!(matches!(miss, Err(EngineError::UnknownReducer { .. })));
    }
}
