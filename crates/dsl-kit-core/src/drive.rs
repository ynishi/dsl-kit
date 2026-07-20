//! sans-io drive layer: the run loop that owns effect resolution.
//!
//! The engine itself is sync and executor-agnostic (design doc §2-A):
//! it never awaits, never spawns, never sleeps. Everything
//! runtime-flavoured lives here, in the *driver* — a small loop that
//! alternates `run_to_yield_with_breakpoints` with effect resolution
//! supplied by an [`EffectResolver`] (sync backends) or an
//! [`AsyncEffectResolver`] (any async runtime — Tokio, smol, wasm
//! bindings; the driver awaits, the caller picks the executor).
//!
//! Backend switching is therefore a resolver + loop choice, not an
//! engine concern: the same `Engine` value can be driven blocking from
//! a CLI, async under Tokio, or single-stepped by a debugger host that
//! bypasses this module entirely and works the [`Stepper`] surface by
//! hand.
//!
//! Breakpoints integrate at the outcome level: when the engine halts on
//! a breakpoint (or any non-`Call` suspension), the driver returns
//! [`DriveOutcome::Break`] and hands control back to the caller.
//! Calling the drive function again resumes past the halt — the
//! engine's one-shot breakpoint marker semantics carry through
//! unchanged.
//!
//! v1 resolves suspensions **sequentially** (one resolve per loop
//! round, engine re-stepped in between). Under an `Any` / `FirstK`
//! join this means the driver never resolves suspensions the join has
//! already discarded — cancelled ids are reported through
//! [`EffectResolver::cancelled`] instead. Concurrent fan-out resolution
//! (resolving N pending calls in parallel) is intentionally left to
//! hosts with an executor opinion; the [`Stepper`] surface remains
//! fully public for that.

use crate::engine::{Ast, Engine, ExecError};
use crate::{
    BreakpointSet, EngineError, NodeContext, Path, Pending, StepOutcome, Stepper, SuspendReason,
    SuspensionId,
};

/// How a drive loop ended while the interpretation is still healthy.
#[derive(Debug)]
pub enum DriveOutcome<V> {
    /// The interpretation completed with the root value.
    Done(V),
    /// The engine halted on a breakpoint (or another non-`Call`
    /// suspension) and control returns to the caller. `at` snapshots
    /// every suspension live at the halt — the synthetic breakpoint
    /// entry plus any `Call`s that were already in flight. Call the
    /// drive function again to resume past the halt.
    Break {
        /// Suspensions live at the halt, in DFS pre-order.
        at: Vec<Pending>,
    },
}

/// Sync effect backend: resolves one suspended `Call` at a time,
/// blocking the driving thread.
///
/// Implementations receive the full [`Pending`] (id, reason, node
/// context) and return the effect's result; the driver feeds it back
/// through [`Stepper::resolve`] and re-steps the engine.
pub trait EffectResolver<A: Ast> {
    /// Produce the result for one suspended `Call`.
    fn resolve(&mut self, pending: &Pending) -> Result<A::Value, A::EffectError>;

    /// Notification that the engine cancelled a suspension (join
    /// policy fired or a sibling failed). Default: ignore. Backends
    /// holding external handles (HTTP requests, subprocesses) abort
    /// them here.
    fn cancelled(&mut self, _sid: SuspensionId) {}
}

/// Async effect backend: same contract as [`EffectResolver`] with an
/// awaitable `resolve`. Runtime-neutral — this crate never touches an
/// executor; whichever runtime polls [`drive_async`] is the backend.
pub trait AsyncEffectResolver<A: Ast> {
    /// Produce the result for one suspended `Call`.
    fn resolve(
        &mut self,
        pending: &Pending,
    ) -> impl Future<Output = Result<A::Value, A::EffectError>> + Send;

    /// Notification that the engine cancelled a suspension. Default:
    /// ignore. See [`EffectResolver::cancelled`].
    fn cancelled(&mut self, _sid: SuspensionId) {}
}

/// Drive `engine` to completion (or to a breakpoint halt) on the
/// current thread, resolving every suspended `Call` through `resolver`.
///
/// Pass [`BreakpointSet::new()`] when no breakpoints are wanted.
/// Errors surface exactly as from [`Stepper::step`] /
/// [`Stepper::resolve`]: engine-level [`EngineError`]s and effect
/// failures the join policy escalated.
pub fn drive<A, R>(
    engine: &mut Engine<A>,
    resolver: &mut R,
    bps: &BreakpointSet,
) -> Result<DriveOutcome<A::Value>, ExecError<A::EffectError>>
where
    A: Ast,
    R: EffectResolver<A>,
{
    loop {
        let outcome = engine.run_to_yield_with_breakpoints(bps)?;
        notify_cancellations(engine, |sid| resolver.cancelled(sid));
        match outcome {
            StepOutcome::Done(v) => return Ok(DriveOutcome::Done(v)),
            StepOutcome::Ready => unreachable!("run_to_yield never returns Ready"),
            StepOutcome::Blocked { .. } => match next_call(engine)? {
                NextCall::Break(at) => return Ok(DriveOutcome::Break { at }),
                NextCall::Resolve(p) => {
                    let result = resolver.resolve(&p);
                    engine.resolve(p.id, result)?;
                    notify_cancellations(engine, |sid| resolver.cancelled(sid));
                }
            },
        }
    }
}

/// Async twin of [`drive`]: identical loop, awaiting the resolver.
/// Runtime-neutral — run it under any executor.
pub async fn drive_async<A, R>(
    engine: &mut Engine<A>,
    resolver: &mut R,
    bps: &BreakpointSet,
) -> Result<DriveOutcome<A::Value>, ExecError<A::EffectError>>
where
    A: Ast,
    R: AsyncEffectResolver<A>,
{
    loop {
        let outcome = engine.run_to_yield_with_breakpoints(bps)?;
        notify_cancellations(engine, |sid| resolver.cancelled(sid));
        match outcome {
            StepOutcome::Done(v) => return Ok(DriveOutcome::Done(v)),
            StepOutcome::Ready => unreachable!("run_to_yield never returns Ready"),
            StepOutcome::Blocked { .. } => match next_call(engine)? {
                NextCall::Break(at) => return Ok(DriveOutcome::Break { at }),
                NextCall::Resolve(p) => {
                    let result = resolver.resolve(&p).await;
                    engine.resolve(p.id, result)?;
                    notify_cancellations(engine, |sid| resolver.cancelled(sid));
                }
            },
        }
    }
}

/// What the driver does next for a blocked engine.
enum NextCall {
    /// Hand control back to the caller (breakpoint / non-`Call`
    /// suspension present).
    Break(Vec<Pending>),
    /// Resolve this `Call` and re-step.
    Resolve(Pending),
}

/// Pick the next action for a blocked engine: break out on any
/// non-`Call` suspension, otherwise resolve the first pending `Call`.
/// A blocked engine with zero suspensions violates the stepper
/// contract and surfaces as [`EngineError::StepperProtocol`].
fn next_call<A: Ast>(engine: &Engine<A>) -> Result<NextCall, ExecError<A::EffectError>> {
    let pending = engine.pending();
    if pending
        .iter()
        .any(|p| !matches!(p.reason, SuspendReason::Call { .. }))
    {
        return Ok(NextCall::Break(pending.to_vec()));
    }
    match pending.first() {
        Some(p) => Ok(NextCall::Resolve(p.clone())),
        None => {
            let root = engine.root_node();
            Err(ExecError::Engine(EngineError::StepperProtocol {
                at: NodeContext::at(root, Path::root().push(root)),
                detail: "engine blocked with no pending suspensions".into(),
            }))
        }
    }
}

/// Drain the engine's cancellation channel into the resolver hook.
fn notify_cancellations<A: Ast>(engine: &mut Engine<A>, mut f: impl FnMut(SuspensionId)) {
    for sid in engine.take_cancellations() {
        f(sid);
    }
}
