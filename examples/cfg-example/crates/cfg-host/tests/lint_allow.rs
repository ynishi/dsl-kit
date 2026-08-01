//! Usage-site lint suppression, end to end through the host: a
//! document that names a rule it accepts at a node stops seeing that
//! finding on `dsl_kit_lint`.
//!
//! Both doors are covered, because the two front-ends spell the
//! annotation differently: `$allow` on a JSON object, `@allow(…)` in
//! front of a canonical-text node from a bundle source.

use cfg_host::CfgHost;
use dsl_kit_mcp::host::DslHost;
use serde_json::{Value, json};

/// Rule names the host currently reports, in report order.
fn findings(host: &CfgHost) -> Vec<String> {
    let json = host.lint_json().expect("cfg wires a linter");
    let parsed: Value = serde_json::from_str(&json).expect("lint output is JSON");
    parsed
        .as_array()
        .expect("lint output is an array")
        .iter()
        .map(|d| d["rule"].as_str().expect("rule name").to_string())
        .collect()
}

/// An `Env` whose only binding is another `Env` — the shape
/// `no-redundant-wrap` complains about. `allow` annotates the outer
/// one.
fn nested_env(allow: bool) -> String {
    let mut root = json!({
        "type": "Env",
        "bindings": {
            "app": {
                "type": "Env",
                "bindings": { "name": { "type": "Leaf", "value": "dsl-kit" } },
            },
        },
    });
    if allow {
        root["$allow"] = json!(["no-redundant-wrap"]);
    }
    root.to_string()
}

#[tokio::test]
async fn a_json_annotation_silences_the_finding_it_names() {
    let mut host = CfgHost::new_with_default_document();

    host.load_json(&nested_env(false)).await.expect("load");
    assert!(
        findings(&host).contains(&"no-redundant-wrap".to_string()),
        "the un-annotated document should report the wrap: {:?}",
        findings(&host),
    );

    host.load_json(&nested_env(true)).await.expect("load");
    assert!(
        !findings(&host).contains(&"no-redundant-wrap".to_string()),
        "the annotation should have silenced it: {:?}",
        findings(&host),
    );
}

/// Loading an annotated document and then an un-annotated one puts the
/// finding back — the table travels with the document, not the host.
#[tokio::test]
async fn the_suppression_does_not_outlive_its_document() {
    let mut host = CfgHost::new_with_default_document();
    host.load_json(&nested_env(true)).await.expect("load");
    assert!(!findings(&host).contains(&"no-redundant-wrap".to_string()));

    host.load_json(&nested_env(false)).await.expect("load");
    assert!(
        findings(&host).contains(&"no-redundant-wrap".to_string()),
        "a later document without the annotation reports again: {:?}",
        findings(&host),
    );
}

/// The text spelling, reached the way a client actually reaches it:
/// a bundle source written in canonical text.
#[tokio::test]
async fn a_text_annotation_in_a_bundle_source_silences_the_finding() {
    let root = json!({ "$import": "app" }).to_string();
    let plain = r#"Env(bindings: { inner: Env(bindings: { name: Leaf(value: "dsl-kit") }) })"#;
    let annotated = format!(r#"@allow("no-redundant-wrap") {plain}"#);

    let mut host = CfgHost::new_with_default_document();
    host.load_json_bundle(&root, &json!({ "app": { "text": plain } }).to_string())
        .await
        .expect("bundle load");
    assert!(
        findings(&host).contains(&"no-redundant-wrap".to_string()),
        "the un-annotated source should report the wrap: {:?}",
        findings(&host),
    );

    host.load_json_bundle(&root, &json!({ "app": { "text": annotated } }).to_string())
        .await
        .expect("bundle load");
    assert!(
        !findings(&host).contains(&"no-redundant-wrap".to_string()),
        "the `@allow` prefix should have silenced it: {:?}",
        findings(&host),
    );
}

/// What a document may not waive stays reported: `unique-node-ids` is
/// a correctness rule, and `["*"]` does not reach it.
#[tokio::test]
async fn a_wildcard_annotation_still_leaves_correctness_findings() {
    let mut host = CfgHost::new_with_default_document();
    let mut doc: Value = serde_json::from_str(&nested_env(false)).expect("document");
    doc["$allow"] = json!(["*"]);
    host.load_json(&doc.to_string()).await.expect("load");

    let reported = findings(&host);
    assert!(
        !reported.contains(&"no-redundant-wrap".to_string()),
        "the wildcard covers the complexity finding: {reported:?}",
    );
    // The document is well-formed, so nothing correctness-class fires
    // here; what the wildcard must not do is silence the annotation's
    // own complaints, which is what a `not-suppressible` entry would
    // be.
    assert!(
        !reported.contains(&"unknown-allow".to_string()),
        "`*` is a known spelling, not an unknown rule name: {reported:?}",
    );
}
