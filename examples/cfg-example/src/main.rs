//! Runs the configuration DSL end to end.
//!
//! The binary shows four things:
//!
//! 1. The hand-built demo document, printed with its keys, flattened
//!    into dotted paths, and resolved through a plain closure.
//! 2. The schema the derive produces — the keyed slots report
//!    `multiplicity: map`, which is what an MCP client sees through
//!    `dsl_kit_schema`.
//! 3. A text round-trip through the **schema-generated grammar**: the
//!    parser for `Env(bindings: { app: Leaf(value: "…") })` is derived
//!    from `Cfg::schema()` at runtime — nobody wrote a grammar.
//! 4. The same document driven through the `DslHost` trait, which is
//!    exactly what `cfg-mcp` exposes over stdio.

use cfg_dsl::{Cfg, demo_document, flatten, pretty, resolve_all};
use cfg_host::CfgHost;
use dsl_kit::{BreakpointSet, IdGen};
use dsl_kit_mcp::host::DslHost;
use dsl_kit_parse::{DslBuild, check_conformance, schema_gen};
use dsl_kit_schema::{DslSchema, Multiplicity};

/// The demo document in canonical text form — the same shape
/// `demo_document` builds in Rust.
const DEMO_TEXT: &str = r#"
    Env(bindings: {
        app: Env(bindings: {
            name: Leaf(value: "dsl-kit"),
            port: Ref(name: "PORT")
        }),
        log: Overrides(entries: {
            "10-base": Leaf(value: "info"),
            "20-prod": Leaf(value: "warn")
        })
    })
"#;

/// Prints every keyed slot the schema declares, with its multiplicity.
fn report_keyed_slots() {
    let schema = Cfg::schema();
    for variant in &schema.variants {
        for child in &variant.children {
            let marker = if child.multiplicity == Multiplicity::Map {
                " <- keyed"
            } else {
                ""
            };
            println!(
                "  {}.{}: multiplicity={}{marker}",
                variant.name,
                child.name,
                child.multiplicity.as_str(),
            );
        }
    }
}

/// Parses canonical text with the schema-generated grammar, builds the
/// typed AST, and resolves it.
fn run_schema_generated_grammar_demo() -> miette::Result<()> {
    let grammar = schema_gen::checked_grammar_from_schema(&Cfg::schema(), &IdGen::new())
        .map_err(|e| miette::miette!("grammar generation failed: {:?}", e.diagnostics))?;
    println!(
        "generated {} rules from Cfg::schema() (start rule `{}`)",
        grammar.rules.len(),
        grammar.start
    );

    let tree = grammar
        .parse(DEMO_TEXT)
        .map_err(|e| miette::miette!("parse failed: {:?}", e.diagnostics))?;
    let diags = check_conformance(&tree, &Cfg::schema());
    println!(
        "parsed `{}` tree; conformance diagnostics: {}",
        tree.variant,
        diags.len()
    );

    let document = Cfg::from_parse_tree(&tree, &IdGen::new())
        .map_err(|e| miette::miette!("typed build failed: {:?}", e.diagnostics))?;
    print!("{}", pretty(&document));
    let value = resolve_all(&document, |name| Some(format!("<{name}>")))?;
    println!("text-parsed document resolves to {value:?}");

    // A repeated key parses but does not build: dropping one of the
    // two subtrees would be the wrong kind of forgiving.
    let duplicate = r#"Env(bindings: { k: Leaf(value: "a"), k: Leaf(value: "b") })"#;
    let tree = grammar
        .parse(duplicate)
        .map_err(|e| miette::miette!("parse failed: {:?}", e.diagnostics))?;
    if let Err(e) = Cfg::from_parse_tree(&tree, &IdGen::new()) {
        println!("deliberate error (repeated key):");
        for d in &e.diagnostics {
            println!("  [{}] {}", d.code, d.message);
        }
    }
    Ok(())
}

/// Splits the demo across three sources — a JSON fragment, a text
/// fragment, and a root that `$import`s both — and links them with the
/// import loader. Same document as the other sections, assembled from
/// pieces.
fn run_import_demo() -> miette::Result<()> {
    use dsl_kit_parse::import::{Loader, MapResolver, add_import_syntax};

    let ids = IdGen::new();
    let schema = Cfg::schema();
    let mut grammar = schema_gen::checked_grammar_from_schema(&schema, &ids)
        .map_err(|e| miette::miette!("grammar generation failed: {:?}", e.diagnostics))?;
    add_import_syntax(&mut grammar, &ids)
        .map_err(|e| miette::miette!("import injection failed: {:?}", e.diagnostics))?;

    let mut resolver = MapResolver::new();
    resolver.insert_text(
        "app",
        r#"Env(bindings: { name: Leaf(value: "dsl-kit"), port: Ref(name: "PORT") })"#,
    );
    resolver.insert(
        "logging",
        r#"{ "type": "Overrides", "entries": {
            "10-base": { "type": "Leaf", "value": "info" },
            "20-prod": { "type": "Leaf", "value": "warn" } } }"#,
    );
    let root = r#"Env(bindings: { app: @import "app", log: @import "logging" })"#;

    let loaded = Loader::new(&schema)
        .with_grammar(&grammar)
        .load_text(root, &mut resolver)
        .map_err(|e| miette::miette!("import load failed: {:?}", e.diagnostics))?;
    let deps: Vec<&str> = loaded.dependencies.iter().map(|d| d.as_str()).collect();
    println!(
        "linked {} sources {:?}; graph digest {}",
        deps.len(),
        deps,
        loaded.digest()
    );

    let document = Cfg::from_parse_tree(&loaded.tree, &IdGen::new())
        .map_err(|e| miette::miette!("typed build failed: {:?}", e.diagnostics))?;
    let value = resolve_all(&document, |name| match name {
        "PORT" => Some("8080".to_string()),
        _ => None,
    })?;
    println!("linked document resolves to {value:?}");
    Ok(())
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    // ---- 1. The document, its keys, and a synchronous resolve ------
    let ids = IdGen::new();
    let document = demo_document(&ids);

    println!("=== Config document ===");
    print!("{}", pretty(&document));

    println!("\n=== Flattened (dotted paths read off the AST) ===");
    for (path, node) in flatten(&document) {
        let label = if path.is_empty() { "<root>" } else { &path };
        println!("  {label} -> {}", node.summary());
    }

    let value = resolve_all(&document, |name| match name {
        "PORT" => Some("8080".to_string()),
        _ => None,
    })?;
    println!("\nsynchronous resolve with PORT=8080 -> {value:?}");

    // ---- 2. What the schema says about the keyed slots -------------
    println!("\n=== Schema (child slots) ===");
    report_keyed_slots();

    // ---- 3. Text round-trip via the schema-generated grammar -------
    println!("\n=== Schema-generated grammar (text -> typed AST -> resolve) ===");
    run_schema_generated_grammar_demo()?;

    // ---- 3b. The same document assembled from imported sources -----
    println!("\n=== Import loader (root + 2 sources -> one linked document) ===");
    run_import_demo()?;

    // ---- 4. Driving the same document through DslHost --------------
    println!("\n=== DslHost run ===");
    let mut host = CfgHost::new_with_default_document();
    let bp = BreakpointSet::new();

    let first = host.step_to_yield(&bp).await.expect("step ok");
    print_outcome("step to_yield", &first);
    let resolved = host.resolve(Some("8080".into())).await.expect("resolve ok");
    println!("resolved: {} = {}", resolved.label, resolved.result);

    let second = host.step_to_yield(&bp).await.expect("step ok");
    print_outcome("step to_yield", &second);

    let snap = host.snapshot();
    println!("\nresults:");
    for (id, entry) in &snap.results {
        println!("  n{id}: {entry}");
    }

    println!("\n(Install cfg-mcp and point an MCP client at it to drive");
    println!(" the same document through dsl_kit_schema / load / step.)");

    Ok(())
}

fn print_outcome(label: &str, outcome: &dsl_kit_mcp::host::HostOutcome) {
    match outcome {
        dsl_kit_mcp::host::HostOutcome::Advanced => println!("{label}: advanced"),
        dsl_kit_mcp::host::HostOutcome::Suspended { reason, at } => {
            println!(
                "{label}: suspended (reason={reason}, node=n{}, path={:?}, depth={})",
                at.node, at.path, at.depth
            );
        }
        dsl_kit_mcp::host::HostOutcome::Done => println!("{label}: done"),
    }
}
