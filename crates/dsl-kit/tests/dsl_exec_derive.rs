//! End-to-end coverage of `#[derive(DslExec)]` + `DslSemantics` +
//! `DerivedAst` / `OwnedDerivedAst`, exercising the forms the
//! `expr-example` does not use (`seq` / `scope` / `maybe` / `call` /
//! `repeat`), plus the owned projection that carries no lifetime.

use std::sync::Arc;

use dsl_kit::{
    DerivedAst, DslSemantics, Engine, IdGen, LoopDecision, NodeId, Op, OpRegistry, OwnedDerivedAst,
    ReducerRegistry, StepOutcome, Stepper, SuspendReason,
};
use dsl_kit_macros::{DslExec, DslNode};

#[derive(Debug, DslNode, DslExec)]
enum Script {
    #[dsl_exec(seq)]
    Block { id: NodeId, steps: Vec<Script> },
    #[dsl_exec(scope(label))]
    Section {
        id: NodeId,
        label: String,
        body: Box<Script>,
    },
    #[dsl_exec(maybe)]
    Optional {
        id: NodeId,
        body: Option<Box<Script>>,
    },
    #[dsl_exec(call(label))]
    Effect { id: NodeId, label: String },
    #[dsl_exec(repeat)]
    Retry { id: NodeId, body: Box<Script> },
    #[dsl_exec(value)]
    Text { id: NodeId, content: String },
    #[dsl_exec(apply = "join")]
    Join { id: NodeId, parts: Vec<Script> },
}

#[derive(Clone)]
struct ScriptSemantics;

impl DslSemantics for ScriptSemantics {
    type Value = String;
    type Delta = ();
    type EffectError = std::convert::Infallible;
    type Cursor = ();

    fn unit_value(&self) -> String {
        String::new()
    }

    fn continue_loop(&self, _node: NodeId, last: &String, _iteration: usize) -> LoopDecision {
        if last == "retry" {
            LoopDecision::Continue
        } else {
            LoopDecision::Break
        }
    }
}

struct JoinOp;

impl Op<String> for JoinOp {
    fn apply(&self, _node: NodeId, args: &[String]) -> Result<String, dsl_kit::EngineError> {
        Ok(args.join("+"))
    }
}

fn ops() -> Arc<OpRegistry<String>> {
    let mut r: OpRegistry<String> = OpRegistry::new();
    r.register("join", Arc::new(JoinOp));
    Arc::new(r)
}

fn engine(root: &Script) -> Engine<DerivedAst<'_, Script, ScriptSemantics>> {
    Engine::new_with_ops(
        DerivedAst::new(root, ScriptSemantics),
        Arc::new(ReducerRegistry::new()),
        ops(),
    )
    .expect("script validates")
}

/// Owned-projection counterpart of [`engine`]: the returned engine holds
/// no borrow of `root` (its `OwnedDerivedAst` copied the classification
/// out), so it can outlive the source tree.
fn owned_engine(root: &Script) -> Engine<OwnedDerivedAst<String, ScriptSemantics>> {
    Engine::new_with_ops(
        OwnedDerivedAst::new(root, ScriptSemantics),
        Arc::new(ReducerRegistry::new()),
        ops(),
    )
    .expect("script validates")
}

fn resolve_sole(e: &mut Engine<DerivedAst<'_, Script, ScriptSemantics>>, v: &str) {
    let sid = e.pending().first().expect("one pending").id;
    e.resolve(sid, Ok(v.to_string())).unwrap();
}

#[test]
fn seq_scope_maybe_value_apply_compose() {
    let ids = IdGen::new();
    let root = Script::Section {
        id: ids.node(),
        label: "outer".into(),
        body: Box::new(Script::Block {
            id: ids.node(),
            steps: vec![
                Script::Optional {
                    id: ids.node(),
                    body: None,
                },
                Script::Join {
                    id: ids.node(),
                    parts: vec![
                        Script::Text {
                            id: ids.node(),
                            content: "a".into(),
                        },
                        Script::Text {
                            id: ids.node(),
                            content: "b".into(),
                        },
                    ],
                },
            ],
        }),
    };
    let mut e = engine(&root);
    let out = e.step().unwrap();
    assert!(matches!(out, StepOutcome::Done(ref s) if s == "a+b"));
}

#[test]
fn call_suspends_with_its_label() {
    let ids = IdGen::new();
    let root = Script::Effect {
        id: ids.node(),
        label: "fetch".into(),
    };
    let mut e = engine(&root);
    let out = e.step().unwrap();
    match out {
        StepOutcome::Blocked { newly_pending } => match &newly_pending[0].reason {
            SuspendReason::Call { spec } => assert_eq!(spec.label, "fetch"),
            other => panic!("expected Call reason, got {other:?}"),
        },
        other => panic!("expected Blocked, got {other:?}"),
    }
    resolve_sole(&mut e, "payload");
    let out = e.step().unwrap();
    assert!(matches!(out, StepOutcome::Done(ref s) if s == "payload"));
}

#[test]
fn repeat_loops_until_semantics_break() {
    let ids = IdGen::new();
    let root = Script::Retry {
        id: ids.node(),
        body: Box::new(Script::Effect {
            id: ids.node(),
            label: "attempt".into(),
        }),
    };
    let mut e = engine(&root);
    e.step().unwrap();
    resolve_sole(&mut e, "retry");
    e.step().unwrap();
    resolve_sole(&mut e, "retry");
    e.step().unwrap();
    resolve_sole(&mut e, "ok");
    let out = e.step().unwrap();
    assert!(matches!(out, StepOutcome::Done(ref s) if s == "ok"));
    assert_eq!(e.events().iteration_tick, 2, "two respawns before break");
}

// ---------- Owned projection (OwnedDerivedAst) --------------------------

#[test]
fn owned_ast_outlives_the_source_tree() {
    // Build an engine over an owned projection, then drop the source
    // tree *before* driving. The engine must still complete: it holds
    // no borrow of `root` — the whole point of `OwnedDerivedAst`.
    let mut e = {
        let ids = IdGen::new();
        let root = Script::Section {
            id: ids.node(),
            label: "outer".into(),
            body: Box::new(Script::Join {
                id: ids.node(),
                parts: vec![
                    Script::Text {
                        id: ids.node(),
                        content: "a".into(),
                    },
                    Script::Text {
                        id: ids.node(),
                        content: "b".into(),
                    },
                ],
            }),
        };
        let e = owned_engine(&root);
        // `root` is dropped here at the end of the block; `e` escapes,
        // proving it borrows nothing from the tree.
        e
    };
    let out = e.step().unwrap();
    assert!(matches!(out, StepOutcome::Done(ref s) if s == "a+b"));
}

#[test]
fn owned_ast_supports_reset_style_reconstruction() {
    // A long-lived host keeps `root` and re-projects a fresh engine on
    // reset. Both runs drive independently to completion.
    let ids = IdGen::new();
    let root = Script::Effect {
        id: ids.node(),
        label: "fetch".into(),
    };

    let mut first = owned_engine(&root);
    first.step().unwrap();
    resolve_sole_owned(&mut first, "payload");
    assert!(matches!(first.step().unwrap(), StepOutcome::Done(ref s) if s == "payload"));

    // Reset: rebuild from the retained owned program and run again.
    let mut second = owned_engine(&root);
    second.step().unwrap();
    resolve_sole_owned(&mut second, "again");
    assert!(matches!(second.step().unwrap(), StepOutcome::Done(ref s) if s == "again"));
}

#[test]
fn owned_ast_clone_backs_a_pristine_engine() {
    // With `S: Clone` + `L: Clone`, the projection is `Clone`, so a host
    // can keep a pristine copy and stamp out fresh engines from it
    // without re-walking the tree.
    let ids = IdGen::new();
    let root = Script::Text {
        id: ids.node(),
        content: "hi".into(),
    };
    let ast = OwnedDerivedAst::new(&root, ScriptSemantics);
    let pristine = ast.clone();

    let mut e = Engine::new_with_ops(pristine, Arc::new(ReducerRegistry::new()), ops())
        .expect("script validates");
    assert!(matches!(e.step().unwrap(), StepOutcome::Done(ref s) if s == "hi"));
}

fn resolve_sole_owned(e: &mut Engine<OwnedDerivedAst<String, ScriptSemantics>>, v: &str) {
    let sid = e.pending().first().expect("one pending").id;
    e.resolve(sid, Ok(v.to_string())).unwrap();
}
