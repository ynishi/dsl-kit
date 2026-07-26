//! End-to-end coverage for the import load phase (`dsl_kit_parse::import`)
//! over the JSON front-end.
//!
//! The consumption path a downstream host walks is
//! `JSON → load_json_str → check_conformance → Cfg::from_parse_tree`,
//! so the happy-path tests drive that whole chain rather than
//! asserting on intermediate shapes alone. What they pin:
//!
//! - **splice semantics** — a `{"$import": "name"}` site is replaced
//!   by the imported document's tree, at positional and keyed slots
//!   alike, transitively;
//! - **graph discipline** — cycles are a chain-rendered diagnostic,
//!   the cache fetches each source once, and every limit fails with
//!   its own stable code;
//! - **fail-loud placeholders** — a placeholder that skips the loader
//!   is rejected at conformance, and malformed placeholder spellings
//!   are rejected at the bridge.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, check_conformance,
    import::{
        IMPORT_VARIANT, ImportLimits, ImportSource, Loaded, MapResolver, SourceId, SourceResolver,
        import_codes, load_json_str,
    },
    serde_bridge::from_json_str,
};
use dsl_kit_schema::DslSchema;
use std::collections::BTreeMap;

/// Self-recursive AST with a positional list, a keyed slot, and a
/// single-child wrapper, so imports have every slot shape to land in.
#[derive(Debug, DslNode, DslSchema, DslBuild)]
enum Cfg {
    /// Leaf holding a payload string.
    Leaf {
        /// Stable node id.
        id: NodeId,
        /// Payload.
        value: String,
    },
    /// Positional list.
    #[allow(clippy::vec_box)]
    Seq {
        /// Stable node id.
        id: NodeId,
        /// Positional children.
        items: Vec<Box<Cfg>>,
    },
    /// Keyed slot.
    Env {
        /// Stable node id.
        id: NodeId,
        /// Keyed children.
        entries: BTreeMap<String, Box<Cfg>>,
    },
    /// Positional single child.
    Wrap {
        /// Stable node id.
        id: NodeId,
        /// Positional child.
        inner: Box<Cfg>,
    },
}

fn load(root: &str, resolver: &mut dyn SourceResolver) -> Result<Loaded, Vec<String>> {
    load_json_str(root, &Cfg::schema(), resolver, &ImportLimits::default())
        .map_err(|e| e.diagnostics.into_iter().map(|d| d.code).collect())
}

/// Builds the typed AST, asserting the linked tree passes shallow
/// conformance at the root first (the derive recurses below).
fn build(loaded: &Loaded) -> Cfg {
    assert_eq!(check_conformance(&loaded.tree, &Cfg::schema()), vec![]);
    Cfg::from_parse_tree(&loaded.tree, &IdGen::new()).expect("typed build")
}

fn leaf_value(cfg: &Cfg) -> &str {
    match cfg {
        Cfg::Leaf { value, .. } => value,
        other => panic!("expected Leaf, got {other:?}"),
    }
}

/// Wrapper that counts `fetch` calls, to pin the once-per-source cache.
struct Counting {
    inner: MapResolver,
    fetches: usize,
}

impl SourceResolver for Counting {
    fn resolve(&mut self, importer: Option<&SourceId>, spec: &str) -> Result<SourceId, String> {
        self.inner.resolve(importer, spec)
    }

    fn fetch(&mut self, id: &SourceId) -> Result<ImportSource, String> {
        self.fetches += 1;
        self.inner.fetch(id)
    }
}

// ---------------------------------------------------------------------------
// Splice semantics
// ---------------------------------------------------------------------------

#[test]
fn import_splices_into_positional_slot_and_builds() {
    let mut resolver = MapResolver::new();
    resolver.insert("lib", r#"{ "type": "Leaf", "value": "shared" }"#);
    let root = r#"{ "type": "Seq", "items": [
        { "$import": "lib" },
        { "type": "Leaf", "value": "inline" }
    ] }"#;

    let loaded = load(root, &mut resolver).expect("load");
    assert_eq!(loaded.dependencies, vec![SourceId::new("lib")]);

    match build(&loaded) {
        Cfg::Seq { items, .. } => {
            assert_eq!(items.len(), 2);
            assert_eq!(leaf_value(&items[0]), "shared");
            assert_eq!(leaf_value(&items[1]), "inline");
        }
        other => panic!("expected Seq, got {other:?}"),
    }
}

#[test]
fn import_splices_into_keyed_slot() {
    let mut resolver = MapResolver::new();
    resolver.insert("db", r#"{ "type": "Leaf", "value": "postgres" }"#);
    let root = r#"{ "type": "Env", "entries": {
        "cache": { "type": "Leaf", "value": "redis" },
        "database": { "$import": "db" }
    } }"#;

    let loaded = load(root, &mut resolver).expect("load");
    match build(&loaded) {
        Cfg::Env { entries, .. } => {
            assert_eq!(leaf_value(&entries["database"]), "postgres");
            assert_eq!(leaf_value(&entries["cache"]), "redis");
        }
        other => panic!("expected Env, got {other:?}"),
    }
}

#[test]
fn transitive_imports_expand_to_fixpoint() {
    let mut resolver = MapResolver::new();
    resolver.insert("a", r#"{ "type": "Wrap", "inner": { "$import": "b" } }"#);
    resolver.insert("b", r#"{ "type": "Wrap", "inner": { "$import": "c" } }"#);
    resolver.insert("c", r#"{ "type": "Leaf", "value": "deep" }"#);
    let root = r#"{ "$import": "a" }"#;

    let loaded = load(root, &mut resolver).expect("load");
    assert_eq!(
        loaded.dependencies,
        vec![SourceId::new("a"), SourceId::new("b"), SourceId::new("c")]
    );

    let mut cfg = build(&loaded);
    for _ in 0..2 {
        cfg = match cfg {
            Cfg::Wrap { inner, .. } => *inner,
            other => panic!("expected Wrap, got {other:?}"),
        };
    }
    assert_eq!(leaf_value(&cfg), "deep");
}

#[test]
fn diamond_import_fetches_each_source_once() {
    let mut inner = MapResolver::new();
    inner.insert("shared", r#"{ "type": "Leaf", "value": "once" }"#);
    let mut resolver = Counting { inner, fetches: 0 };
    let root = r#"{ "type": "Seq", "items": [
        { "$import": "shared" },
        { "$import": "shared" }
    ] }"#;

    let loaded = load(root, &mut resolver).expect("load");
    assert_eq!(resolver.fetches, 1);
    assert_eq!(loaded.dependencies, vec![SourceId::new("shared")]);

    match build(&loaded) {
        Cfg::Seq { items, .. } => {
            assert_eq!(leaf_value(&items[0]), "once");
            assert_eq!(leaf_value(&items[1]), "once");
        }
        other => panic!("expected Seq, got {other:?}"),
    }
}

#[test]
fn root_without_imports_has_no_dependencies() {
    let mut resolver = MapResolver::new();
    let loaded = load(r#"{ "type": "Leaf", "value": "plain" }"#, &mut resolver).expect("load");
    assert_eq!(loaded.dependencies, vec![]);
    assert_eq!(leaf_value(&build(&loaded)), "plain");
}

// ---------------------------------------------------------------------------
// Graph discipline
// ---------------------------------------------------------------------------

#[test]
fn cycle_reports_full_chain() {
    let mut resolver = MapResolver::new();
    resolver.insert("a", r#"{ "type": "Wrap", "inner": { "$import": "b" } }"#);
    resolver.insert("b", r#"{ "type": "Wrap", "inner": { "$import": "a" } }"#);

    let err = load_json_str(
        r#"{ "$import": "a" }"#,
        &Cfg::schema(),
        &mut resolver,
        &ImportLimits::default(),
    )
    .expect_err("cycle");

    let cycle = err
        .diagnostics
        .iter()
        .find(|d| d.code == import_codes::CYCLE)
        .expect("cycle diagnostic");
    assert!(
        cycle.message.contains("<root> → a → b → a"),
        "chain missing from: {}",
        cycle.message
    );
}

#[test]
fn self_import_is_a_cycle() {
    let mut resolver = MapResolver::new();
    resolver.insert("me", r#"{ "type": "Wrap", "inner": { "$import": "me" } }"#);

    let err = load(r#"{ "$import": "me" }"#, &mut resolver).expect_err("cycle");
    assert!(err.contains(&import_codes::CYCLE.to_string()), "{err:?}");
}

#[test]
fn depth_limit_fails_loudly() {
    let mut resolver = MapResolver::new();
    resolver.insert("a", r#"{ "type": "Wrap", "inner": { "$import": "b" } }"#);
    resolver.insert("b", r#"{ "type": "Leaf", "value": "deep" }"#);
    let limits = ImportLimits {
        max_depth: 1,
        ..ImportLimits::default()
    };

    let err = load_json_str(
        r#"{ "$import": "a" }"#,
        &Cfg::schema(),
        &mut resolver,
        &limits,
    )
    .expect_err("depth");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == import_codes::DEPTH_EXCEEDED),
        "{err}"
    );
}

#[test]
fn source_limit_fails_loudly() {
    let mut resolver = MapResolver::new();
    resolver.insert("a", r#"{ "type": "Leaf", "value": "a" }"#);
    resolver.insert("b", r#"{ "type": "Leaf", "value": "b" }"#);
    let limits = ImportLimits {
        max_sources: 1,
        ..ImportLimits::default()
    };
    let root = r#"{ "type": "Seq", "items": [
        { "$import": "a" },
        { "$import": "b" }
    ] }"#;

    let err =
        load_json_str(root, &Cfg::schema(), &mut resolver, &limits).expect_err("source limit");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == import_codes::SOURCE_LIMIT),
        "{err}"
    );
}

#[test]
fn byte_limit_fails_loudly() {
    let mut resolver = MapResolver::new();
    resolver.insert("big", r#"{ "type": "Leaf", "value": "0123456789" }"#);
    let limits = ImportLimits {
        max_total_bytes: 8,
        ..ImportLimits::default()
    };

    let err = load_json_str(
        r#"{ "$import": "big" }"#,
        &Cfg::schema(),
        &mut resolver,
        &limits,
    )
    .expect_err("byte limit");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == import_codes::BYTE_LIMIT),
        "{err}"
    );
}

#[test]
fn missing_source_is_fetch_failed() {
    let mut resolver = MapResolver::new();
    let err = load(r#"{ "$import": "nowhere" }"#, &mut resolver).expect_err("fetch");
    assert!(
        err.contains(&import_codes::FETCH_FAILED.to_string()),
        "{err:?}"
    );
}

#[test]
fn failed_source_is_cached_and_replayed_per_site() {
    let mut resolver = Counting {
        inner: MapResolver::new(),
        fetches: 0,
    };
    let root = r#"{ "type": "Seq", "items": [
        { "$import": "nowhere" },
        { "$import": "nowhere" }
    ] }"#;

    let err = load(root, &mut resolver).expect_err("fetch");
    // Both sites report, but the broken source was fetched only once.
    assert_eq!(
        err.iter()
            .filter(|c| *c == import_codes::FETCH_FAILED)
            .count(),
        2
    );
    assert_eq!(resolver.fetches, 1);
}

// ---------------------------------------------------------------------------
// Fail-loud placeholders
// ---------------------------------------------------------------------------

#[test]
fn placeholder_with_extra_keys_is_rejected() {
    let mut resolver = MapResolver::new();
    let err = load(r#"{ "$import": "lib", "type": "Leaf" }"#, &mut resolver).expect_err("shape");
    assert!(
        err.contains(&import_codes::SPEC_SHAPE.to_string()),
        "{err:?}"
    );
}

#[test]
fn placeholder_with_non_string_spec_is_rejected() {
    let mut resolver = MapResolver::new();
    let err = load(r#"{ "$import": 42 }"#, &mut resolver).expect_err("shape");
    assert!(
        err.contains(&import_codes::SPEC_SHAPE.to_string()),
        "{err:?}"
    );
}

#[test]
fn unexpanded_placeholder_is_rejected_at_conformance() {
    let tree = from_json_str(r#"{ "$import": "lib" }"#, &Cfg::schema()).expect("bridge");
    assert_eq!(tree.variant, IMPORT_VARIANT);

    let diags = check_conformance(&tree, &Cfg::schema());
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code, import_codes::UNEXPANDED);
}

#[test]
fn parse_error_inside_import_carries_chain_context() {
    let mut resolver = MapResolver::new();
    resolver.insert("bad", r#"{ "type": "Nope" }"#);

    let err = load_json_str(
        r#"{ "$import": "bad" }"#,
        &Cfg::schema(),
        &mut resolver,
        &ImportLimits::default(),
    )
    .expect_err("nested parse error");

    assert_eq!(err.diagnostics[0].code, import_codes::IN_IMPORT);
    assert!(
        err.diagnostics[0].message.contains("<root> → bad"),
        "chain missing from: {}",
        err.diagnostics[0].message
    );
    assert!(
        err.diagnostics[1..]
            .iter()
            .any(|d| d.code == dsl_kit_parse::codes::UNKNOWN_VARIANT),
        "{err}"
    );
}
