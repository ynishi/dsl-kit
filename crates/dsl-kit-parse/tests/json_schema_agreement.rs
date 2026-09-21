//! `dsl_kit_schema::json_schema` ⇄ `serde_bridge` agreement.
//!
//! The rendered JSON Schema is the *published* contract of a DSL's
//! JSON surface; the serde bridge plus `check_conformance` is the
//! *enforced* one. They must agree, or clients validate against a
//! promise the parser does not keep:
//!
//! - every document the front-end accepts validates against the
//!   schema — pinned on the canonical dump of a parsed fixture and on
//!   hand-written spellings of every optional / omission form;
//! - every document the front-end rejects for a structural reason
//!   (unknown key, missing field, wrong payload type, wrong slot
//!   shape, empty or absent `non_empty` slot, unknown variant) is
//!   rejected by the schema as well.

use std::collections::BTreeMap;

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslDump, DslNode, DslSchema};
use dsl_kit_parse::{DslBuild as _, check_conformance, dump_canonical_json, serde_bridge};
use dsl_kit_schema::DslSchema as _;
use dsl_kit_schema::json_schema::{RefStyle, TypeMap};
use serde_json::{Value, json};

/// Every slot shape plus the built-in payload mappings.
#[derive(Debug, DslNode, DslSchema, DslBuild, DslDump)]
enum Cfg {
    Leaf {
        id: NodeId,
        name: String,
        count: u32,
        on: bool,
        note: Option<String>,
        tags: Vec<String>,
    },
    Node {
        id: NodeId,
        head: Box<Cfg>,
        fallback: Option<Box<Cfg>>,
        items: Vec<Cfg>,
        env: BTreeMap<String, Cfg>,
        labels: BTreeMap<String, String>,
    },
    All {
        id: NodeId,
        #[dsl_schema(non_empty)]
        items: Vec<Cfg>,
    },
}

fn validator() -> jsonschema::Validator {
    let doc = Cfg::schema()
        .to_json_schema(&TypeMap::new(), RefStyle::Defs)
        .expect("renders");
    jsonschema::validator_for(&doc).expect("valid JSON Schema")
}

fn front_end_accepts(doc: &Value) -> bool {
    let Ok(tree) = serde_bridge::from_json_value(doc, &Cfg::schema()) else {
        return false;
    };
    check_conformance(&tree, &Cfg::schema()).is_empty()
        && Cfg::from_parse_tree(&tree, &IdGen::new()).is_ok()
}

fn leaf(name: &str) -> Value {
    json!({ "type": "Leaf", "name": name, "count": 1, "on": true })
}

#[test]
fn accepted_documents_validate() {
    let v = validator();
    let accepted = [
        leaf("a"),
        json!({ "type": "Leaf", "name": "a", "count": 1, "on": false,
                "note": "n", "tags": ["x", "y"] }),
        json!({ "type": "Leaf", "name": "a", "count": 1, "on": false,
                "note": null, "tags": [] }),
        json!({ "type": "Node", "head": leaf("h") }),
        json!({ "type": "Node", "head": leaf("h"), "fallback": null,
                "items": [], "env": {}, "labels": {} }),
        json!({ "type": "Node", "head": leaf("h"), "fallback": leaf("f"),
                "items": [leaf("i")], "env": { "k": leaf("e") },
                "labels": { "a": "b" } }),
        json!({ "type": "All", "items": [leaf("i")] }),
    ];
    for doc in accepted {
        assert!(front_end_accepts(&doc), "front-end rejected {doc}");
        let errors: Vec<String> = v.iter_errors(&doc).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "schema rejected {doc}: {errors:?}");
    }
}

#[test]
fn canonical_dump_validates_and_reparses() {
    let v = validator();
    let ids = IdGen::new();
    let doc = json!({ "type": "Node", "head": leaf("h"), "fallback": leaf("f"),
                      "items": [leaf("i"), json!({ "type": "All", "items": [leaf("j")] })],
                      "env": { "k": leaf("e") }, "labels": { "a": "b" } });
    let tree = serde_bridge::from_json_value(&doc, &Cfg::schema()).unwrap();
    let ast = Cfg::from_parse_tree(&tree, &ids).unwrap();
    let dumped = dump_canonical_json(&ast).unwrap();
    let errors: Vec<String> = v.iter_errors(&dumped).map(|e| e.to_string()).collect();
    assert!(
        errors.is_empty(),
        "schema rejected the kit's own dump: {errors:?}"
    );
    assert!(front_end_accepts(&dumped));
}

#[test]
fn rejected_documents_fail_validation() {
    let v = validator();
    let rejected = [
        json!({ "type": "Nope" }),
        json!({ "name": "a", "count": 1, "on": true }),
        json!({ "type": "Leaf", "nmae": "a", "count": 1, "on": true }),
        json!({ "type": "Leaf", "count": 1, "on": true }),
        json!({ "type": "Leaf", "name": "a", "count": "1", "on": true }),
        json!({ "type": "Leaf", "name": "a", "count": 1, "on": true, "tags": "x" }),
        json!({ "type": "Leaf", "name": "a", "count": 1, "on": true, "note": 3 }),
        json!({ "type": "Node" }),
        json!({ "type": "Node", "head": [leaf("h")] }),
        json!({ "type": "Node", "head": leaf("h"), "items": leaf("i") }),
        json!({ "type": "Node", "head": leaf("h"), "env": [leaf("e")] }),
        json!({ "type": "Node", "head": leaf("h"), "labels": { "a": 1 } }),
        json!({ "type": "All", "items": [] }),
        json!({ "type": "All" }),
    ];
    for doc in rejected {
        assert!(!front_end_accepts(&doc), "front-end accepted {doc}");
        assert!(!v.is_valid(&doc), "schema accepted {doc}");
    }
}
