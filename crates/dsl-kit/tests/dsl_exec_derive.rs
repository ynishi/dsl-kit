//! End-to-end coverage of `#[derive(DslExec)]` + `DslSemantics` +
//! `DerivedAst`, exercising the forms the `expr-example` does not use
//! (`seq` / `scope` / `maybe` / `call` / `repeat`).

use std::sync::Arc;

use dsl_kit::{
    DerivedAst, DslSemantics, Engine, IdGen, LoopDecision, NodeId, Op, OpRegistry, ReducerRegistry,
    StepOutcome, Stepper, SuspendReason,
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
