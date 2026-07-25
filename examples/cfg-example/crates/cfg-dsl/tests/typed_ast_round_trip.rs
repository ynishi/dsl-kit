//! Typed-AST round-trip: a `Cfg` value built in Rust is rendered to
//! canonical text, parsed back through the schema-generated grammar,
//! and compared against the original.
//!
//! The other two suites start from a document and end at the AST; this
//! one starts at the AST, which is where a DSL author actually lives.
//! The renderer below is deliberately part of the test rather than the
//! crate: emitting canonical text is not something the kit promises
//! yet, and pinning the syntax from the outside is what catches a
//! grammar change that would strand hand-built ASTs.

use std::collections::BTreeMap;

use cfg_dsl::{Cfg, flatten, resolve_all};
use dsl_kit::{DslNode, IdGen, Walk};
use dsl_kit_parse::{DslBuild, check_conformance, schema_gen::checked_grammar_from_schema};
use dsl_kit_schema::{DslSchema, Multiplicity};

/// Renders a `Cfg` in the canonical text syntax, keys quoted.
fn render(cfg: &Cfg) -> String {
    fn entries(pairs: Vec<(&str, &Cfg)>) -> String {
        pairs
            .into_iter()
            .map(|(key, child)| format!("{:?}: {}", key, render(child)))
            .collect::<Vec<_>>()
            .join(", ")
    }
    match cfg {
        Cfg::Leaf { value, .. } => format!("Leaf(value: {value:?})"),
        Cfg::Ref { name, .. } => format!("Ref(name: {name:?})"),
        Cfg::Env { .. } => format!("Env(bindings: {{ {} }})", entries(cfg.keyed_children())),
        Cfg::Overrides { .. } => {
            format!(
                "Overrides(entries: {{ {} }})",
                entries(cfg.keyed_children())
            )
        }
    }
}

/// `dotted.path -> summary` lines; `NodeId`s are minted per build and
/// are meant to differ, so they stay out of the comparison.
fn shape(cfg: &Cfg) -> Vec<String> {
    flatten(cfg)
        .into_iter()
        .map(|(path, node)| format!("{path} -> {}", node.summary()))
        .collect()
}

/// Hand-built document mixing both keyed variants, a reference and a
/// literal, with one key that has to be quoted.
fn hand_built(ids: &IdGen) -> Cfg {
    let app = Cfg::Env {
        id: ids.node(),
        bindings: BTreeMap::from([
            (
                "log level".to_string(),
                Box::new(Cfg::Leaf {
                    id: ids.node(),
                    value: "debug".into(),
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
    let layers = Cfg::Overrides {
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
            ("log".to_string(), Box::new(layers)),
        ]),
    }
}

#[test]
fn typed_ast_survives_a_text_round_trip() {
    let original = hand_built(&IdGen::new());
    let text = render(&original);

    let schema = Cfg::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new()).expect("generates");
    let tree = grammar
        .parse(&text)
        .unwrap_or_else(|e| panic!("rendered text must parse: {:?}\n  {text}", e.diagnostics));
    assert!(check_conformance(&tree, &schema).is_empty());
    let rebuilt = Cfg::from_parse_tree(&tree, &IdGen::new()).expect("rebuilds");

    assert_eq!(shape(&original), shape(&rebuilt));
    assert_eq!(
        text,
        render(&rebuilt),
        "rendering is stable across the trip"
    );
}

#[test]
fn both_keyed_variants_report_multiplicity_map() {
    let schema = Cfg::schema();
    let env = schema.variant("Env").expect("Env declared");
    assert_eq!(env.children[0].name, "bindings");
    assert_eq!(env.children[0].multiplicity, Multiplicity::Map);

    let overrides = schema.variant("Overrides").expect("Overrides declared");
    assert_eq!(overrides.children[0].name, "entries");
    assert_eq!(overrides.children[0].multiplicity, Multiplicity::Map);

    // The leaf variants have no child slots at all.
    assert!(
        schema
            .variant("Leaf")
            .expect("Leaf declared")
            .children
            .is_empty()
    );
    assert!(
        schema
            .variant("Ref")
            .expect("Ref declared")
            .children
            .is_empty()
    );
}

#[test]
fn walk_sees_keyed_values_in_key_order_without_the_keys() {
    let document = hand_built(&IdGen::new());
    let Cfg::Env { bindings, .. } = &document else {
        panic!("expected Env");
    };
    // `Walk` yields values only; the names come from the AST.
    let walked: Vec<String> = document
        .children()
        .into_iter()
        .map(|c| c.summary())
        .collect();
    let keyed: Vec<String> = bindings.values().map(|c| c.summary()).collect();
    assert_eq!(walked, keyed);
    assert_eq!(
        document
            .keyed_children()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<Vec<_>>(),
        vec!["app", "log"],
    );
}

#[test]
fn node_id_is_the_variants_own_not_a_keyed_childs() {
    let ids = IdGen::new();
    let child_id = ids.node();
    let variant_id = ids.node();
    let document = Cfg::Env {
        id: variant_id,
        bindings: BTreeMap::from([(
            "only".to_string(),
            Box::new(Cfg::Leaf {
                id: child_id,
                value: "v".into(),
            }),
        )]),
    };
    assert_eq!(document.node_id(), variant_id);
    assert_ne!(document.node_id(), child_id);
}

#[test]
fn a_hand_built_document_resolves_through_the_engine() {
    let document = hand_built(&IdGen::new());
    let value = resolve_all(&document, |name| match name {
        "PORT" => Some("8080".to_string()),
        _ => None,
    })
    .expect("resolution settles");
    assert_eq!(value, "warn");
}
