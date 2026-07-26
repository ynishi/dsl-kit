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
//! it from `{"$import": "name"}` at any node position. The loader
//! replaces each placeholder with the parsed-and-expanded tree of the
//! resolved source; placeholders never survive into conformance — a
//! leftover one is a [`import_codes::UNEXPANDED`] diagnostic there.
//!
//! Sharing is by value: two sites importing the same source each
//! receive a clone of the expanded tree. Node identity is minted later
//! (by [`DslBuild`] via `IdGen`), so clones cannot collide.
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

use crate::{BuildError, Diagnostic, ParseTree, RawValue, serde_bridge};
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
/// `#[non_exhaustive]`: a `Text` arm (PEG front-end) is planned; match
/// with a catch-all.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImportSource {
    /// A JSON document, parsed through
    /// [`serde_bridge::from_json_str`].
    Json(String),
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
/// in-memory JSON sources.
///
/// Specifiers are used verbatim as canonical ids. Nothing outside the
/// map is reachable, which makes this the default-deny resolver shape
/// suited to MCP-style hosts where the client supplies every source
/// inline.
#[derive(Debug, Clone, Default)]
pub struct MapResolver {
    sources: BTreeMap<String, String>,
}

impl MapResolver {
    /// Constructs an empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers (or replaces) a named JSON source.
    pub fn insert(&mut self, name: impl Into<String>, json: impl Into<String>) {
        self.sources.insert(name.into(), json.into());
    }
}

impl SourceResolver for MapResolver {
    fn resolve(&mut self, _importer: Option<&SourceId>, spec: &str) -> Result<SourceId, String> {
        Ok(SourceId::new(spec))
    }

    fn fetch(&mut self, id: &SourceId) -> Result<ImportSource, String> {
        match self.sources.get(id.as_str()) {
            Some(json) => Ok(ImportSource::Json(json.clone())),
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

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Parses a root JSON document and expands its imports to a fixpoint.
///
/// The root is parsed with [`serde_bridge::from_json_str`], then every
/// [`IMPORT_VARIANT`] placeholder — at any depth, in positional and
/// keyed child slots alike — is resolved through `resolver`, parsed,
/// recursively expanded, and spliced in place. Diagnostics from
/// distinct import sites are collected before failing; diagnostics
/// from *inside* an imported source are prefixed with an
/// [`import_codes::IN_IMPORT`] context marker naming the source and
/// the resolution chain.
pub fn load_json_str(
    root: &str,
    schema: &NodeSchema,
    resolver: &mut dyn SourceResolver,
    limits: &ImportLimits,
) -> Result<Loaded, BuildError> {
    let tree = serde_bridge::from_json_str(root, schema)?;
    expand_root(tree, schema, resolver, limits)
}

/// [`load_json_str`] over an already-deserialized
/// [`serde_json::Value`].
pub fn load_json_value(
    root: &Value,
    schema: &NodeSchema,
    resolver: &mut dyn SourceResolver,
    limits: &ImportLimits,
) -> Result<Loaded, BuildError> {
    let tree = serde_bridge::from_json_value(root, schema)?;
    expand_root(tree, schema, resolver, limits)
}

fn expand_root(
    tree: ParseTree,
    schema: &NodeSchema,
    resolver: &mut dyn SourceResolver,
    limits: &ImportLimits,
) -> Result<Loaded, BuildError> {
    let mut state = LoadState {
        resolver,
        limits,
        schema,
        cache: BTreeMap::new(),
        stack: Vec::new(),
        total_bytes: 0,
    };
    let tree = expand(tree, None, 0, &mut state).map_err(BuildError::new)?;
    let dependencies = state.cache.into_keys().collect();
    Ok(Loaded { tree, dependencies })
}

/// Per-load expansion state.
struct LoadState<'a> {
    resolver: &'a mut dyn SourceResolver,
    limits: &'a ImportLimits,
    schema: &'a NodeSchema,
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

    let parsed = match fetched {
        ImportSource::Json(text) => {
            state.total_bytes += text.len();
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
            serde_bridge::from_json_str(&text, state.schema)
        }
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
