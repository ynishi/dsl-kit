//! A small configuration DSL built on **keyed child slots**.
//!
//! `Cfg` is the third reference DSL for `dsl-kit`, and the first one
//! whose children are keyed: `Env` and `Overrides` hold
//! `BTreeMap<String, _>` rather than a positional `Box` / `Vec`. Where
//! `flow-dsl` models an orchestration graph and `expr-dsl` an
//! arithmetic language, `Cfg` models a configuration document — the
//! shape where "which child" is a name, not an index.
//!
//! The two keyed variants deliberately differ in one respect only:
//! `Env` boxes its values (`BTreeMap<String, Box<Cfg>>`, the derive's
//! `MapBoxed` path) and `Overrides` does not
//! (`BTreeMap<String, Cfg>`, the `Map` path). Both report
//! `Multiplicity::Map` in the schema and both read the same way in
//! text and JSON; keeping both in one DSL is what stops the two derive
//! arms from drifting apart.
//!
//! ## Where the keys live
//!
//! `Walk` iterates keyed slots by value, in the map's own
//! (sorted-by-key) order — it does not surface the keys. So the engine
//! sees an ordered child sequence, and anything that needs the names
//! reads them off the AST. [`pretty`] and [`flatten`] are the two
//! places in this crate that do exactly that.
//!
//! This crate carries the AST plus its engine wiring; the `DslHost`
//! adapter lives in `cfg-host` and the MCP binary in `cfg-mcp`.

#![warn(missing_docs)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;

use dsl_kit::{
    DslExec, DslNode, DslSemantics, Engine, EngineError, ExecError, IdGen, NodeContext, NodeId, Op,
    OpRegistry, OwnedDerivedAst, Path, Phase, ReducerRegistry, StepOutcome, Stepper, SuspendReason,
    SuspensionId, Walk,
};

/// AST of the configuration DSL.
///
/// Every variant carries a `#[dsl_exec(...)]` form, so the engine
/// classification is derived; [`CfgSemantics`] supplies the handful of
/// judgments the kit cannot derive.
#[derive(
    Debug, DslNode, dsl_kit_macros::DslSchema, dsl_kit_macros::DslBuild, dsl_kit_macros::DslExec,
)]
pub enum Cfg {
    /// A named block. Its bindings are evaluated in key order and the
    /// block resolves to the last one's value.
    ///
    /// Keyed slot with boxed values — the derive's `MapBoxed` path.
    #[dsl_exec(seq)]
    Env {
        /// Stable node id.
        id: NodeId,
        /// Child settings, keyed by name.
        bindings: BTreeMap<String, Box<Cfg>>,
    },
    /// A stack of override layers, keyed by layer name. Layers are
    /// folded through the registered `"last_wins"` op, so the
    /// highest-sorting layer supplies the value (`10-base` loses to
    /// `20-prod`). An empty stack folds to the empty string.
    ///
    /// Keyed slot without `Box` — the derive's `Map` path. Identical
    /// schema shape to [`Cfg::Env`]; the difference is Rust storage.
    #[dsl_exec(apply = "last_wins")]
    Overrides {
        /// Stable node id.
        id: NodeId,
        /// Layers, keyed by layer name.
        entries: BTreeMap<String, Cfg>,
    },
    /// A value the document does not carry: the engine asks the host
    /// for it, exactly as an unbound `Read` does everywhere else in
    /// the kit. This is what makes `dsl_kit_resolve` meaningful on the
    /// MCP surface.
    #[dsl_exec(read(name))]
    Ref {
        /// Stable node id.
        id: NodeId,
        /// Name handed to the host when the lookup misses.
        name: String,
    },
    /// A literal setting — the terminating case.
    #[dsl_exec(value)]
    Leaf {
        /// Stable node id.
        id: NodeId,
        /// Literal value.
        value: String,
    },
}

impl Cfg {
    /// One-line summary of a node's shape.
    pub fn summary(&self) -> String {
        match self {
            Cfg::Env { bindings, .. } => format!("Env ({} bindings)", bindings.len()),
            Cfg::Overrides { entries, .. } => format!("Overrides ({} layers)", entries.len()),
            Cfg::Ref { name, .. } => format!("Ref {name:?}"),
            Cfg::Leaf { value, .. } => format!("Leaf {value:?}"),
        }
    }

    /// Keyed children of this node as `(key, child)` pairs, in key
    /// order. Empty for the leaf variants.
    ///
    /// `Walk::children` drops the keys by design (the engine has no
    /// use for them); this is the accessor for callers that do.
    pub fn keyed_children(&self) -> Vec<(&str, &Cfg)> {
        match self {
            Cfg::Env { bindings, .. } => bindings
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_ref()))
                .collect(),
            Cfg::Overrides { entries, .. } => {
                entries.iter().map(|(k, v)| (k.as_str(), v)).collect()
            }
            Cfg::Ref { .. } | Cfg::Leaf { .. } => Vec::new(),
        }
    }
}

/// Renders a `Cfg` as an indented text tree, keys included.
///
/// Written as an explicit recursion rather than through `Walk`,
/// because `Walk` surfaces values only — the keys are read off the
/// AST via [`Cfg::keyed_children`].
pub fn pretty(cfg: &Cfg) -> String {
    fn go(node: &Cfg, key: Option<&str>, depth: usize, out: &mut String) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        match key {
            Some(k) => {
                let _ = writeln!(out, "{} {k}: {}", node.node_id(), node.summary());
            }
            None => {
                let _ = writeln!(out, "{} {}", node.node_id(), node.summary());
            }
        }
        for (child_key, child) in node.keyed_children() {
            go(child, Some(child_key), depth + 1, out);
        }
    }
    let mut out = String::new();
    go(cfg, None, 0, &mut out);
    out
}

/// Flattens the document into `(dotted.path, node)` pairs, in key
/// order — the projection a config consumer actually wants.
///
/// The root contributes the empty path; every keyed step appends its
/// key. Nothing here is engine machinery: it is ordinary AST reading,
/// which is where keyed slots pay off.
pub fn flatten(cfg: &Cfg) -> Vec<(String, &Cfg)> {
    fn go<'a>(node: &'a Cfg, prefix: &str, out: &mut Vec<(String, &'a Cfg)>) {
        out.push((prefix.to_string(), node));
        for (key, child) in node.keyed_children() {
            let path = if prefix.is_empty() {
                key.to_string()
            } else {
                format!("{prefix}.{key}")
            };
            go(child, &path, out);
        }
    }
    let mut out = Vec::new();
    go(cfg, "", &mut out);
    out
}

/// Counts every node in the AST (pre-order visits).
pub fn count_nodes(cfg: &Cfg) -> usize {
    let mut count = 0usize;
    cfg.walk(&mut |_, phase| {
        if phase == Phase::Pre {
            count += 1;
        }
    });
    count
}

/// Effect-side error for `Cfg` — what a host reports when it answers a
/// [`Cfg::Ref`] suspension with `Err(_)`.
#[derive(Debug, Clone)]
pub struct CfgEffectError {
    /// Human-readable description supplied by the host.
    pub message: String,
}

impl std::fmt::Display for CfgEffectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CfgEffectError {}

/// The semantic half of the DSL. Config values are strings, and the
/// document carries no binding form — so `Delta` is `()` and every
/// [`Cfg::Ref`] lookup misses, which is precisely what turns it into a
/// host-answered suspension.
pub struct CfgSemantics;

impl DslSemantics for CfgSemantics {
    type Value = String;
    type Delta = ();
    type EffectError = CfgEffectError;
    type Cursor = ();

    fn unit_value(&self) -> String {
        String::new()
    }
}

/// Engine-ready [`Ast`](dsl_kit::Ast) over `Cfg`: derived
/// classification zipped with [`CfgSemantics`].
///
/// Owned projection ([`OwnedDerivedAst`]), so a long-lived host can own
/// its document and engine together without `Box::leak`.
pub type CfgAst = OwnedDerivedAst<<Cfg as DslExec>::LitValue, CfgSemantics>;

/// The fold behind [`Cfg::Overrides`]: the highest-sorting layer wins.
///
/// The op sees layer *values* only — keys stay on the AST — so
/// "highest-sorting" is expressed as "last argument", which holds
/// because keyed slots reach the engine in key order.
struct LastWins;

impl Op<String> for LastWins {
    fn apply(&self, _node: NodeId, args: &[String]) -> Result<String, EngineError> {
        Ok(args.last().cloned().unwrap_or_default())
    }
}

/// The op table for the configuration surface.
pub fn cfg_ops() -> Arc<OpRegistry<String>> {
    let mut ops: OpRegistry<String> = OpRegistry::new();
    ops.register("last_wins", Arc::new(LastWins));
    Arc::new(ops)
}

/// Builds a fresh engine over `cfg` with the standard op table.
pub fn cfg_engine(cfg: &Cfg) -> Result<Engine<CfgAst>, EngineError> {
    Engine::new_with_ops(
        OwnedDerivedAst::new(cfg, CfgSemantics),
        Arc::new(ReducerRegistry::new()),
        cfg_ops(),
    )
}

/// Builds the demo document:
///
/// ```text
/// Env(bindings: {
///   app: Env(bindings: { name: Leaf("dsl-kit"), port: Ref("PORT") }),
///   log: Overrides(entries: { 10-base: Leaf("info"), 20-prod: Leaf("warn") }),
/// })
/// ```
///
/// Resolving it suspends once (on `PORT`) and settles on `"warn"`:
/// the root `Env` is a `Seq`, so its value is the last binding's in
/// key order (`log`), and that layer stack folds last-wins to
/// `20-prod`.
pub fn demo_document(ids: &IdGen) -> Cfg {
    let app = Cfg::Env {
        id: ids.node(),
        bindings: BTreeMap::from([
            (
                "name".to_string(),
                Box::new(Cfg::Leaf {
                    id: ids.node(),
                    value: "dsl-kit".into(),
                }),
            ),
            (
                "port".to_string(),
                Box::new(Cfg::Ref {
                    id: ids.node(),
                    name: "PORT".into(),
                }),
            ),
        ]),
    };
    let log = Cfg::Overrides {
        id: ids.node(),
        entries: BTreeMap::from([
            (
                "10-base".to_string(),
                Cfg::Leaf {
                    id: ids.node(),
                    value: "info".into(),
                },
            ),
            (
                "20-prod".to_string(),
                Cfg::Leaf {
                    id: ids.node(),
                    value: "warn".into(),
                },
            ),
        ]),
    };
    Cfg::Env {
        id: ids.node(),
        bindings: BTreeMap::from([
            ("app".to_string(), Box::new(app)),
            ("log".to_string(), Box::new(log)),
        ]),
    }
}

/// Resolves `cfg` to completion on the engine, answering every
/// [`Cfg::Ref`] through `resolver`.
///
/// Returns `EngineError::Malformed` when a reference is not covered by
/// the resolver. The whole reduction happens inside the engine; this
/// function only answers the suspensions it yields.
pub fn resolve_all<F>(cfg: &Cfg, mut resolver: F) -> Result<String, EngineError>
where
    F: FnMut(&str) -> Option<String>,
{
    let mut engine = cfg_engine(cfg)?;
    loop {
        match engine.step() {
            Ok(StepOutcome::Done(v)) => return Ok(v),
            Ok(StepOutcome::Ready) => continue,
            Ok(StepOutcome::Blocked { .. }) => {
                let outstanding: Vec<(SuspensionId, NodeId, String)> = engine
                    .pending()
                    .iter()
                    .filter_map(|p| match &p.reason {
                        SuspendReason::Call { spec } => Some((p.id, p.at.node, spec.label.clone())),
                        _ => None,
                    })
                    .collect();
                if outstanding.is_empty() {
                    return Err(EngineError::Malformed {
                        at: NodeContext::at(cfg.node_id(), Path::root().push(cfg.node_id())),
                        detail: "engine blocked without a resolvable suspension".into(),
                    });
                }
                for (sid, node, name) in outstanding {
                    match resolver(&name) {
                        Some(v) => {
                            engine.resolve(sid, Ok(v)).map_err(|e| match e {
                                ExecError::Engine(e) => e,
                                ExecError::Effect(e) => EngineError::EvalFailed {
                                    at: NodeContext::at(node, Path::root().push(node)),
                                    source: Box::new(e),
                                },
                            })?;
                        }
                        None => {
                            return Err(EngineError::Malformed {
                                at: NodeContext::at(node, Path::root().push(node)),
                                detail: format!("unresolved reference {name:?}"),
                            });
                        }
                    }
                }
            }
            Err(ExecError::Engine(e)) => return Err(e),
            Err(ExecError::Effect(e)) => {
                return Err(EngineError::EvalFailed {
                    at: NodeContext::at(cfg.node_id(), Path::root().push(cfg.node_id())),
                    source: Box::new(e),
                });
            }
        }
    }
}
