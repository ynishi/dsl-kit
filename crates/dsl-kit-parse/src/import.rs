//! Load/link phase: expands `$import` placeholders into a single
//! fully-linked [`ParseTree`] before conformance and [`DslBuild`].
//!
//! ## Model
//!
//! Imports are resolved in a **separate load phase, not as a runtime
//! effect**: the engine never sees an import, and the existing
//! suspend/resume `Call` surface keeps meaning "domain effect", never
//! "go read a file". The loader walks the import graph to a fixpoint
//! and hands downstream consumers one linked tree — the same
//! resolution-before-evaluation split Bazel's `load()` and Dhall's
//! import judgment standardise.
//!
//! Three properties carry the design:
//!
//! - **Literal-only specifiers.** An import site names its source with
//!   a literal string (`{"$import": "lib"}` in the JSON front-end), so
//!   the full document graph is derivable by a pure syntactic pass —
//!   [`Loaded::dependencies`] reports it without executing anything.
//! - **The host owns all IO.** The loader itself never touches the
//!   filesystem; every byte enters through a caller-supplied
//!   [`SourceResolver`]. The kit ships [`MapResolver`] (named
//!   in-memory sources) as the canonical sandboxed implementation;
//!   a filesystem resolver is just another impl a host may write.
//! - **Bounded expansion.** Cycles are detected with a `Pending`
//!   sentinel in the source cache and reported as the full chain
//!   (`<root> → a → b → a`); [`ImportLimits`] caps depth, source
//!   count, and total fetched bytes so reference-expansion blowups
//!   fail loudly instead of exhausting memory.
//!
//! ## Placeholder shape
//!
//! Front-ends represent an import site as a [`ParseTree`] whose
//! variant is [`IMPORT_VARIANT`] (`"$import"`) carrying a single
//! [`IMPORT_SPEC_FIELD`] (`"spec"`) payload. The JSON bridge produces
//! it from `{"$import": "name"}` at any node position; the canonical
//! text front-end spells the same placeholder `@import "name"` once
//! [`add_import_syntax`] has been applied to the grammar. The loader
//! replaces each placeholder with the parsed-and-expanded tree of the
//! resolved source; placeholders never survive into conformance — a
//! leftover one is a [`import_codes::UNEXPANDED`] diagnostic there.
//!
//! Sharing is by value: two sites importing the same source each
//! receive a clone of the expanded tree. Node identity is minted later
//! (by [`DslBuild`] via `IdGen`), so clones cannot collide.
//!
//! ## Front-ends and mixing
//!
//! A fetched source declares its front-end via [`ImportSource`]:
//! `Json` parses through [`serde_bridge::from_json_str`], `Text`
//! through the [`Loader`]'s configured [`Grammar`]. The two mix
//! freely — a text root may import JSON sources and vice versa —
//! because both land in the same [`ParseTree`] trunk before splicing.
//! A `Text` source arriving at a [`Loader`] with no grammar is a
//! loud [`import_codes::TEXT_UNSUPPORTED`] failure, not a fallback.
//!
//! One known limitation: [`ParseTree::span`]s inside an expanded tree
//! are byte offsets **relative to the source that parsed that
//! subtree**, and the tree does not record which source that is.
//! Loader diagnostics carry the resolution chain instead; per-subtree
//! source attribution is a planned follow-up.
//!
//! ## Example
//!
//! ```ignore
//! use dsl_kit_parse::import::{ImportLimits, MapResolver, load_json_str};
//!
//! let mut resolver = MapResolver::new();
//! resolver.insert("lib", r#"{ "type": "Leaf", "value": "shared" }"#);
//! let root = r#"{ "type": "Seq", "items": [ { "$import": "lib" } ] }"#;
//! let loaded = load_json_str(root, &Cfg::schema(), &mut resolver, &ImportLimits::default())?;
//! assert_eq!(loaded.dependencies.len(), 1);
//! ```
//!
//! [`DslBuild`]: crate::DslBuild

use crate::peg::{self, Grammar, Peg};
use crate::{BuildError, Diagnostic, ParseTree, RawValue, serde_bridge};
use dsl_kit_core::IdGen;
use dsl_kit_schema::NodeSchema;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;

/// Reserved [`ParseTree::variant`] spelling for an unexpanded import
/// placeholder.
///
/// The `$` sigil cannot appear in a Rust enum variant name, so the
/// spelling can never collide with a DSL author's variant.
pub const IMPORT_VARIANT: &str = "$import";

/// Field name on an [`IMPORT_VARIANT`] placeholder that carries the
/// literal source specifier.
pub const IMPORT_SPEC_FIELD: &str = "spec";

/// Diagnostic codes emitted by the import loader (and the shared
/// placeholder machinery in the front-ends / conformance).
pub mod import_codes {
    /// The import graph contains a cycle. The message renders the full
    /// resolution chain (`<root> → a → b → a`).
    pub const CYCLE: &str = "dsl_kit::parse::import::cycle";
    /// Expansion exceeded [`ImportLimits::max_depth`](super::ImportLimits).
    pub const DEPTH_EXCEEDED: &str = "dsl_kit::parse::import::depth_exceeded";
    /// Expansion exceeded [`ImportLimits::max_sources`](super::ImportLimits).
    pub const SOURCE_LIMIT: &str = "dsl_kit::parse::import::source_limit";
    /// Expansion exceeded [`ImportLimits::max_total_bytes`](super::ImportLimits).
    pub const BYTE_LIMIT: &str = "dsl_kit::parse::import::byte_limit";
    /// [`SourceResolver::resolve`](super::SourceResolver::resolve)
    /// returned an error for a specifier.
    pub const RESOLVE_FAILED: &str = "dsl_kit::parse::import::resolve_failed";
    /// [`SourceResolver::fetch`](super::SourceResolver::fetch) returned
    /// an error for a resolved source id.
    pub const FETCH_FAILED: &str = "dsl_kit::parse::import::fetch_failed";
    /// An import placeholder is malformed: the `$import` value is not
    /// a string, or the object carries keys besides `$import`, or a
    /// hand-built placeholder tree lacks a usable
    /// [`IMPORT_SPEC_FIELD`](super::IMPORT_SPEC_FIELD) payload.
    pub const SPEC_SHAPE: &str = "dsl_kit::parse::import::spec_shape";
    /// Context marker prepended to diagnostics that originate inside
    /// an imported source, naming the source and the resolution chain.
    pub const IN_IMPORT: &str = "dsl_kit::parse::import::in_import";
    /// An [`IMPORT_VARIANT`](super::IMPORT_VARIANT) placeholder
    /// reached [`check_conformance`](crate::check_conformance) —
    /// the document was not run through the loader.
    pub const UNEXPANDED: &str = "dsl_kit::parse::import::unexpanded";
    /// A [`ImportSource::Text`](super::ImportSource::Text) source was
    /// fetched (or a text root was loaded) but the [`Loader`](super::Loader)
    /// has no [`Grammar`](crate::peg::Grammar) configured — call
    /// [`Loader::with_grammar`](super::Loader::with_grammar).
    pub const TEXT_UNSUPPORTED: &str = "dsl_kit::parse::import::text_unsupported";
    /// A sources bundle handed to
    /// [`MapResolver::from_sources_json`](super::MapResolver::from_sources_json)
    /// is malformed: not a JSON object, an entry that is not a
    /// single-key `{"json": "…"}` / `{"text": "…"}` object, or an
    /// unknown front-end tag.
    pub const BAD_SOURCES: &str = "dsl_kit::parse::import::bad_sources";
}

// ---------------------------------------------------------------------------
// SourceId / ImportSource / SourceResolver
// ---------------------------------------------------------------------------

/// Canonical identity of a source, as produced by
/// [`SourceResolver::resolve`].
///
/// The loader caches, cycle-checks, and reports dependencies on this
/// id — two specifiers that resolve to the same id are the same
/// source, fetched once. Resolvers should canonicalise here (e.g.
/// collapse `./` path segments) so aliased spellings don't defeat the
/// cache the way Lua's `package.loaded` string-keying does.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(String);

impl SourceId {
    /// Wraps a canonical id string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The canonical id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A fetched source document, tagged with the front-end that should
/// parse it.
///
/// `#[non_exhaustive]`: further front-ends may be added; match with a
/// catch-all.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImportSource {
    /// A JSON document, parsed through
    /// [`serde_bridge::from_json_str`].
    Json(String),
    /// A canonical-text document, parsed through the [`Loader`]'s
    /// configured [`Grammar`]. Requires [`Loader::with_grammar`];
    /// otherwise the fetch fails with
    /// [`import_codes::TEXT_UNSUPPORTED`].
    Text(String),
}

/// Host-supplied import IO: turns literal specifiers into canonical
/// ids and canonical ids into source text.
///
/// This is the only place bytes enter the loader — the kit core stays
/// io-free, mirroring Jsonnet's `import_callback` / Starlark's
/// embedder-owned `load` resolution. `resolve` receives the importing
/// source (or `None` for the root document) so relative-path schemes
/// can canonicalise against the importer, and is split from `fetch`
/// so the loader can dedupe on the canonical id *before* paying for
/// IO.
pub trait SourceResolver {
    /// Canonicalises `spec` (as written at the import site inside
    /// `importer`) into a [`SourceId`].
    ///
    /// Errors are reported as [`import_codes::RESOLVE_FAILED`].
    fn resolve(&mut self, importer: Option<&SourceId>, spec: &str) -> Result<SourceId, String>;

    /// Produces the source document for a resolved id.
    ///
    /// Called at most once per id per [`load_json_str`] call; errors
    /// are reported as [`import_codes::FETCH_FAILED`] and cached, so
    /// a broken source fails every site that imports it without being
    /// re-fetched.
    fn fetch(&mut self, id: &SourceId) -> Result<ImportSource, String>;
}

/// The canonical sandboxed [`SourceResolver`]: a map of named
/// in-memory sources.
///
/// Specifiers are used verbatim as canonical ids. Nothing outside the
/// map is reachable, which makes this the default-deny resolver shape
/// suited to MCP-style hosts where the client supplies every source
/// inline.
#[derive(Debug, Clone, Default)]
pub struct MapResolver {
    sources: BTreeMap<String, ImportSource>,
}

impl MapResolver {
    /// Constructs an empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers (or replaces) a named JSON source.
    pub fn insert(&mut self, name: impl Into<String>, json: impl Into<String>) {
        self.sources
            .insert(name.into(), ImportSource::Json(json.into()));
    }

    /// Registers (or replaces) a named canonical-text source.
    pub fn insert_text(&mut self, name: impl Into<String>, text: impl Into<String>) {
        self.sources
            .insert(name.into(), ImportSource::Text(text.into()));
    }

    /// Builds a resolver from a JSON sources bundle — the wire shape
    /// MCP-style hosts receive from a client:
    ///
    /// ```json
    /// { "lib":  { "json": "{ \"type\": \"Leaf\", … }" },
    ///   "frag": { "text": "Leaf(value: \"x\")" } }
    /// ```
    ///
    /// Every entry must be a single-key object tagging the front-end
    /// explicitly (`json` / `text`) with a string payload — the
    /// flavour lives in the data, never inferred. Anything else is a
    /// [`import_codes::BAD_SOURCES`] diagnostic; problems across
    /// entries are collected before failing.
    pub fn from_sources_json(sources_json: &str) -> Result<Self, BuildError> {
        let value: Value = serde_json::from_str(sources_json).map_err(|e| {
            BuildError::single(Diagnostic::error(
                import_codes::BAD_SOURCES,
                format!("sources bundle is not valid JSON: {e}"),
            ))
        })?;
        let Value::Object(map) = value else {
            return Err(BuildError::single(Diagnostic::error(
                import_codes::BAD_SOURCES,
                "sources bundle must be a JSON object mapping names to \
                 single-key {\"json\": …} / {\"text\": …} objects",
            )));
        };

        let mut resolver = Self::new();
        let mut diags = Vec::new();
        for (name, entry) in map {
            let bad = |what: &str| {
                Diagnostic::error(
                    import_codes::BAD_SOURCES,
                    format!(
                        "source `{name}` {what} (expected a single-key \
                         {{\"json\": \"…\"}} or {{\"text\": \"…\"}} object)"
                    ),
                )
            };
            let Value::Object(entry) = entry else {
                diags.push(bad("is not an object"));
                continue;
            };
            if entry.len() != 1 {
                diags.push(bad("must carry exactly one front-end key"));
                continue;
            }
            let (kind, payload) = entry.iter().next().expect("len checked");
            let Value::String(payload) = payload else {
                diags.push(bad("has a non-string payload"));
                continue;
            };
            match kind.as_str() {
                "json" => resolver.insert(name, payload.clone()),
                "text" => resolver.insert_text(name, payload.clone()),
                other => diags.push(bad(&format!("uses unknown front-end tag `{other}`"))),
            }
        }
        if diags.is_empty() {
            Ok(resolver)
        } else {
            Err(BuildError::new(diags))
        }
    }
}

impl SourceResolver for MapResolver {
    fn resolve(&mut self, _importer: Option<&SourceId>, spec: &str) -> Result<SourceId, String> {
        Ok(SourceId::new(spec))
    }

    fn fetch(&mut self, id: &SourceId) -> Result<ImportSource, String> {
        match self.sources.get(id.as_str()) {
            Some(source) => Ok(source.clone()),
            None => Err(format!("no source registered under `{id}`")),
        }
    }
}

// ---------------------------------------------------------------------------
// Limits / Loaded
// ---------------------------------------------------------------------------

/// Hard caps on import expansion.
///
/// Reference expansion is a real denial-of-service class (the
/// billion-laughs family), so every bound fails loudly with a
/// dedicated diagnostic instead of degrading. The defaults are
/// generous for configuration-sized documents; hosts embedding
/// untrusted input should tighten them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportLimits {
    /// Maximum import nesting depth (root = depth 0). Exceeding it is
    /// [`import_codes::DEPTH_EXCEEDED`].
    pub max_depth: usize,
    /// Maximum number of distinct sources fetched in one load.
    /// Exceeding it is [`import_codes::SOURCE_LIMIT`].
    pub max_sources: usize,
    /// Maximum total bytes fetched across all sources in one load.
    /// Exceeding it is [`import_codes::BYTE_LIMIT`].
    pub max_total_bytes: usize,
}

impl Default for ImportLimits {
    fn default() -> Self {
        Self {
            max_depth: 32,
            max_sources: 256,
            max_total_bytes: 8 * 1024 * 1024,
        }
    }
}

/// A fully-linked document plus its resolved import graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    /// The root tree with every placeholder replaced by its expanded
    /// source. Ready for [`check_conformance`](crate::check_conformance)
    /// and [`DslBuild`](crate::DslBuild).
    pub tree: ParseTree,
    /// Every source the load touched, as canonical ids, sorted
    /// ascending and deduplicated. Empty when the root document has
    /// no imports.
    pub dependencies: Vec<SourceId>,
}

impl Loaded {
    /// Stable digest of the resolved graph: the fully-linked tree's
    /// content plus the sorted dependency ids, hashed with FNV-1a 64.
    ///
    /// Loads that link to the same tree content produce the same
    /// digest — so "did anything change?" is a single string
    /// comparison, in the spirit of Dhall's normalized-expression
    /// hash. Spans are excluded (surface detail: reformatting a
    /// source must not disturb the digest); field payloads are hashed
    /// through their [`RawValue`], so the digest is sensitive to
    /// which front-end parsed a payload (`Text` vs `Json`) — compare
    /// digests produced by the same front-end mix. Non-cryptographic
    /// — a change detector, not an integrity check.
    pub fn digest(&self) -> String {
        let mut h = Fnv1a::new();
        feed_tree(&mut h, &self.tree);
        for d in &self.dependencies {
            h.write(b"\x1fdep\x1f");
            h.write(d.as_str().as_bytes());
        }
        format!("{:016x}", h.finish())
    }
}

/// FNV-1a 64-bit, hand-rolled so the digest is stable across Rust
/// releases (`DefaultHasher` makes no such promise) without pulling a
/// hashing dependency into the parse trunk.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Feeds a tree into the digest in a canonical order. `\x1f` (unit
/// separator) delimits framing so `("ab", "c")` and `("a", "bc")`
/// cannot collide by concatenation.
fn feed_tree(h: &mut Fnv1a, tree: &ParseTree) {
    h.write(b"\x1fnode\x1f");
    h.write(tree.variant.as_bytes());
    for (name, value) in &tree.fields {
        h.write(b"\x1ffield\x1f");
        h.write(name.as_bytes());
        h.write(b"\x1f");
        match value {
            RawValue::Text(t) => {
                h.write(b"t\x1f");
                h.write(t.as_bytes());
            }
            // Default serde_json maps are BTree-backed, so `to_string`
            // is deterministic for equal values.
            RawValue::Json(v) => {
                h.write(b"j\x1f");
                h.write(v.to_string().as_bytes());
            }
        }
    }
    for (slot, children) in &tree.children {
        h.write(b"\x1fslot\x1f");
        h.write(slot.as_bytes());
        for c in children {
            feed_tree(h, c);
        }
    }
    for (slot, entries) in &tree.keyed_children {
        h.write(b"\x1fkeyed\x1f");
        h.write(slot.as_bytes());
        for (key, c) in entries {
            h.write(b"\x1fkey\x1f");
            h.write(key.as_bytes());
            feed_tree(h, c);
        }
    }
    h.write(b"\x1fend\x1f");
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Import loader: a schema, an optional text [`Grammar`], and the
/// expansion limits, bundled so every front-end entry point shares one
/// configuration.
///
/// Every load walks the same pipeline: parse the root with its
/// front-end, then resolve every [`IMPORT_VARIANT`] placeholder — at
/// any depth, in positional and keyed child slots alike — through the
/// caller's [`SourceResolver`], parse each fetched source with the
/// front-end its [`ImportSource`] arm names, recursively expand, and
/// splice in place. Diagnostics from distinct import sites are
/// collected before failing; diagnostics from *inside* an imported
/// source are prefixed with an [`import_codes::IN_IMPORT`] context
/// marker naming the source and the resolution chain.
#[derive(Debug, Clone)]
pub struct Loader<'a> {
    schema: &'a NodeSchema,
    grammar: Option<&'a Grammar>,
    limits: ImportLimits,
}

impl<'a> Loader<'a> {
    /// Constructs a loader with default [`ImportLimits`] and no text
    /// grammar (JSON sources only).
    pub fn new(schema: &'a NodeSchema) -> Self {
        Self {
            schema,
            grammar: None,
            limits: ImportLimits::default(),
        }
    }

    /// Enables the canonical text front-end: text roots and
    /// [`ImportSource::Text`] sources parse through `grammar`.
    ///
    /// Pass a grammar that has been through [`add_import_syntax`],
    /// otherwise text sources cannot spell `@import` themselves.
    pub fn with_grammar(mut self, grammar: &'a Grammar) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Replaces the default [`ImportLimits`].
    pub fn with_limits(mut self, limits: ImportLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Loads a JSON root document and expands its imports to a
    /// fixpoint.
    pub fn load_json_str(
        &self,
        root: &str,
        resolver: &mut dyn SourceResolver,
    ) -> Result<Loaded, BuildError> {
        let tree = serde_bridge::from_json_str(root, self.schema)?;
        self.expand_root(tree, resolver)
    }

    /// [`Loader::load_json_str`] over an already-deserialized
    /// [`serde_json::Value`].
    pub fn load_json_value(
        &self,
        root: &Value,
        resolver: &mut dyn SourceResolver,
    ) -> Result<Loaded, BuildError> {
        let tree = serde_bridge::from_json_value(root, self.schema)?;
        self.expand_root(tree, resolver)
    }

    /// Loads a canonical-text root document and expands its imports
    /// to a fixpoint. Requires [`Loader::with_grammar`].
    pub fn load_text(
        &self,
        root: &str,
        resolver: &mut dyn SourceResolver,
    ) -> Result<Loaded, BuildError> {
        let Some(grammar) = self.grammar else {
            return Err(BuildError::single(Diagnostic::error(
                import_codes::TEXT_UNSUPPORTED,
                "cannot load a text root: this Loader has no grammar — \
                 configure one with `Loader::with_grammar`",
            )));
        };
        let tree = grammar.parse(root)?;
        self.expand_root(tree, resolver)
    }

    fn expand_root(
        &self,
        tree: ParseTree,
        resolver: &mut dyn SourceResolver,
    ) -> Result<Loaded, BuildError> {
        let mut state = LoadState {
            resolver,
            limits: &self.limits,
            schema: self.schema,
            grammar: self.grammar,
            cache: BTreeMap::new(),
            stack: Vec::new(),
            total_bytes: 0,
        };
        let tree = expand(tree, None, 0, &mut state).map_err(BuildError::new)?;
        let dependencies = state.cache.into_keys().collect();
        Ok(Loaded { tree, dependencies })
    }
}

/// Loads a JSON root document with default limits and no text
/// front-end. Convenience wrapper over [`Loader`].
pub fn load_json_str(
    root: &str,
    schema: &NodeSchema,
    resolver: &mut dyn SourceResolver,
    limits: &ImportLimits,
) -> Result<Loaded, BuildError> {
    Loader::new(schema)
        .with_limits(limits.clone())
        .load_json_str(root, resolver)
}

/// [`load_json_str`] over an already-deserialized
/// [`serde_json::Value`].
pub fn load_json_value(
    root: &Value,
    schema: &NodeSchema,
    resolver: &mut dyn SourceResolver,
    limits: &ImportLimits,
) -> Result<Loaded, BuildError> {
    Loader::new(schema)
        .with_limits(limits.clone())
        .load_json_value(root, resolver)
}

// ---------------------------------------------------------------------------
// Text syntax injection
// ---------------------------------------------------------------------------

/// Adds the reserved `@import "name"` spelling to a text grammar.
///
/// Appends a rule named [`IMPORT_VARIANT`] whose body is
/// `Node("$import", %kw:@import Field("spec", %str))`, and makes it an
/// alternative of the grammar's start rule — for grammars generated by
/// [`crate::schema_gen`], the start rule is the `node` choice that
/// every child slot references, so the spelling becomes available at
/// every node position. A start rule whose body is not a [`Peg::Choice`]
/// is wrapped in one.
///
/// Opt-in by design: a grammar that never passes through this function
/// accepts no import syntax, and the reserved rule is invisible to
/// [`crate::example_gen`] (examples never spell `@import`) and exempt
/// from [`crate::grammar_check`]'s schema-consistency pass.
///
/// Idempotent — a grammar that already carries the reserved rule is
/// returned unchanged. Fails with [`crate::peg::codes::UNKNOWN_RULE`]
/// if the start rule is not defined.
pub fn add_import_syntax(grammar: &mut Grammar, ids: &IdGen) -> Result<(), BuildError> {
    let already = grammar
        .rules
        .iter()
        .any(|r| matches!(r, Peg::Rule { name, .. } if name == IMPORT_VARIANT));
    if already {
        return Ok(());
    }

    let start = grammar.start.clone();
    let start_rule = grammar
        .rules
        .iter_mut()
        .find(|r| matches!(r, Peg::Rule { name, .. } if *name == start));
    let Some(Peg::Rule { body, .. }) = start_rule else {
        return Err(BuildError::single(Diagnostic::error(
            peg::codes::UNKNOWN_RULE,
            format!("cannot add import syntax: start rule `{start}` is not defined"),
        )));
    };

    match body.as_mut() {
        Peg::Choice { alts, .. } => alts.push(peg::rule_ref(ids, IMPORT_VARIANT)),
        _ => {
            let dummy = peg::token(ids, "");
            let old = std::mem::replace(body.as_mut(), dummy);
            **body = peg::choice(ids, vec![old, peg::rule_ref(ids, IMPORT_VARIANT)]);
        }
    }

    grammar.rules.push(peg::rule(
        ids,
        IMPORT_VARIANT,
        peg::node(
            ids,
            IMPORT_VARIANT,
            peg::seq(
                ids,
                vec![
                    peg::token(ids, "%kw:@import"),
                    peg::field(ids, IMPORT_SPEC_FIELD, peg::token(ids, "%str")),
                ],
            ),
        ),
    ));
    Ok(())
}

/// Per-load expansion state.
struct LoadState<'a> {
    resolver: &'a mut dyn SourceResolver,
    limits: &'a ImportLimits,
    schema: &'a NodeSchema,
    grammar: Option<&'a Grammar>,
    /// Source cache keyed on canonical id. Both outcomes are memoised
    /// — a broken source stays broken for every later site instead of
    /// being re-fetched and re-parsed per importer.
    cache: BTreeMap<SourceId, CacheEntry>,
    /// The in-flight resolution chain, for cycle detection and chain
    /// rendering in diagnostics.
    stack: Vec<SourceId>,
    total_bytes: usize,
}

enum CacheEntry {
    /// Expansion in progress — hitting this from below is a cycle.
    Pending,
    /// Fully expanded, ready to splice as a clone.
    Ready(ParseTree),
    /// Fetch / parse / expansion failed; the diagnostics replay at
    /// every importing site.
    Failed(Vec<Diagnostic>),
}

impl LoadState<'_> {
    /// Renders the current resolution chain, optionally extended with
    /// one more hop, as `<root> → a → b`.
    fn chain(&self, tail: Option<&SourceId>) -> String {
        let mut parts = vec!["<root>".to_string()];
        parts.extend(self.stack.iter().map(SourceId::to_string));
        if let Some(t) = tail {
            parts.push(t.to_string());
        }
        parts.join(" → ")
    }
}

/// Recursively replaces every placeholder in `tree`.
///
/// `importer` is the source the tree came from (`None` = the root
/// document); `depth` counts import hops, not tree depth.
fn expand(
    mut tree: ParseTree,
    importer: Option<&SourceId>,
    depth: usize,
    state: &mut LoadState<'_>,
) -> Result<ParseTree, Vec<Diagnostic>> {
    if tree.variant == IMPORT_VARIANT {
        let spec = placeholder_spec(&tree)?;
        return resolve_import(&spec, importer, depth, state);
    }

    let mut diags = Vec::new();
    for (_, slot) in &mut tree.children {
        for child in std::mem::take(slot) {
            match expand(child, importer, depth, state) {
                Ok(t) => slot.push(t),
                Err(ds) => diags.extend(ds),
            }
        }
    }
    for (_, entries) in &mut tree.keyed_children {
        for (key, child) in std::mem::take(entries) {
            match expand(child, importer, depth, state) {
                Ok(t) => entries.push((key, t)),
                Err(ds) => diags.extend(ds),
            }
        }
    }
    if diags.is_empty() {
        Ok(tree)
    } else {
        Err(diags)
    }
}

/// Extracts the literal specifier from a placeholder tree.
fn placeholder_spec(tree: &ParseTree) -> Result<String, Vec<Diagnostic>> {
    match tree.field(IMPORT_SPEC_FIELD) {
        Some(RawValue::Json(Value::String(s))) => Ok(s.clone()),
        Some(RawValue::Text(s)) => Ok(s.clone()),
        _ => Err(vec![
            Diagnostic::error(
                import_codes::SPEC_SHAPE,
                format!(
                    "`{IMPORT_VARIANT}` placeholder lacks a string `{IMPORT_SPEC_FIELD}` payload",
                ),
            )
            .with_span(tree.span),
        ]),
    }
}

/// Resolves one import site: cache lookup, limit checks, fetch, parse,
/// recursive expansion.
fn resolve_import(
    spec: &str,
    importer: Option<&SourceId>,
    depth: usize,
    state: &mut LoadState<'_>,
) -> Result<ParseTree, Vec<Diagnostic>> {
    if depth + 1 > state.limits.max_depth {
        return Err(vec![Diagnostic::error(
            import_codes::DEPTH_EXCEEDED,
            format!(
                "import nesting exceeds max_depth {} at `{spec}` (chain: {})",
                state.limits.max_depth,
                state.chain(None),
            ),
        )]);
    }

    let id = state.resolver.resolve(importer, spec).map_err(|e| {
        vec![Diagnostic::error(
            import_codes::RESOLVE_FAILED,
            format!(
                "cannot resolve import `{spec}`: {e} (chain: {})",
                state.chain(None)
            ),
        )]
    })?;

    match state.cache.get(&id) {
        Some(CacheEntry::Pending) => {
            return Err(vec![Diagnostic::error(
                import_codes::CYCLE,
                format!("import cycle detected: {}", state.chain(Some(&id))),
            )]);
        }
        Some(CacheEntry::Ready(tree)) => return Ok(tree.clone()),
        Some(CacheEntry::Failed(diags)) => return Err(diags.clone()),
        None => {}
    }

    if state.cache.len() + 1 > state.limits.max_sources {
        return Err(vec![Diagnostic::error(
            import_codes::SOURCE_LIMIT,
            format!(
                "import expansion exceeds max_sources {} at `{id}` (chain: {})",
                state.limits.max_sources,
                state.chain(None),
            ),
        )]);
    }

    let fetched = match state.resolver.fetch(&id) {
        Ok(s) => s,
        Err(e) => {
            let diags = vec![Diagnostic::error(
                import_codes::FETCH_FAILED,
                format!(
                    "cannot fetch import `{id}`: {e} (chain: {})",
                    state.chain(None)
                ),
            )];
            state.cache.insert(id, CacheEntry::Failed(diags.clone()));
            return Err(diags);
        }
    };

    let source_len = match &fetched {
        ImportSource::Json(text) | ImportSource::Text(text) => text.len(),
    };
    state.total_bytes += source_len;
    if state.total_bytes > state.limits.max_total_bytes {
        let diags = vec![Diagnostic::error(
            import_codes::BYTE_LIMIT,
            format!(
                "import expansion exceeds max_total_bytes {} at `{id}` (chain: {})",
                state.limits.max_total_bytes,
                state.chain(None),
            ),
        )];
        state.cache.insert(id, CacheEntry::Failed(diags.clone()));
        return Err(diags);
    }

    let parsed = match fetched {
        ImportSource::Json(text) => serde_bridge::from_json_str(&text, state.schema),
        ImportSource::Text(text) => match state.grammar {
            Some(grammar) => grammar.parse(&text),
            None => {
                let diags = vec![Diagnostic::error(
                    import_codes::TEXT_UNSUPPORTED,
                    format!(
                        "import `{id}` is a text source but this Loader has no \
                         grammar — configure one with `Loader::with_grammar` \
                         (chain: {})",
                        state.chain(None),
                    ),
                )];
                state.cache.insert(id, CacheEntry::Failed(diags.clone()));
                return Err(diags);
            }
        },
    };

    let parsed = match parsed {
        Ok(tree) => tree,
        Err(err) => {
            let diags = in_import(&id, state, err.diagnostics);
            state.cache.insert(id, CacheEntry::Failed(diags.clone()));
            return Err(diags);
        }
    };

    state.cache.insert(id.clone(), CacheEntry::Pending);
    state.stack.push(id.clone());
    let expanded = expand(parsed, Some(&id), depth + 1, state);
    state.stack.pop();

    match expanded {
        Ok(tree) => {
            state.cache.insert(id, CacheEntry::Ready(tree.clone()));
            Ok(tree)
        }
        Err(diags) => {
            let diags = in_import(&id, state, diags);
            state.cache.insert(id, CacheEntry::Failed(diags.clone()));
            Err(diags)
        }
    }
}

/// Prefixes diagnostics that originate inside an imported source with
/// a context marker naming the source and the resolution chain.
///
/// Nested failures are wrapped once per hop, so a deep failure reads
/// outermost-first. Diagnostics already carrying an
/// [`import_codes::IN_IMPORT`] marker for this exact source are not
/// re-wrapped (replayed cache entries arrive pre-wrapped).
fn in_import(id: &SourceId, state: &LoadState<'_>, diags: Vec<Diagnostic>) -> Vec<Diagnostic> {
    let marker = format!("in import `{id}`");
    if diags
        .first()
        .is_some_and(|d| d.code == import_codes::IN_IMPORT && d.message.starts_with(&marker))
    {
        return diags;
    }
    let mut out = vec![Diagnostic::error(
        import_codes::IN_IMPORT,
        format!("{marker} (chain: {})", state.chain(Some(id))),
    )];
    out.extend(diags);
    out
}
