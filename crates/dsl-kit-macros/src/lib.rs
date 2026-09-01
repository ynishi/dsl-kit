//! Derive macros for `dsl-kit`.
//!
//! ## Design
//!
//! One input shape, five derives. Every macro accepts the same enum
//! form (named-field variants, one `id: NodeId` each), so a DSL opts
//! into traversal, schema reflection, parse-tree building, engine
//! execution, and semantic checking by adding derives — never by
//! restating its shape.
//!
//! `#[derive(DslNode)]` accepts an `enum` whose every variant uses named
//! fields, exactly one of which is called `id` and typed `NodeId`. The
//! macro generates three impls in one shot:
//!
//! - `DslNode` — returns the `id` field for each variant.
//! - `Walk` — returns direct children by inspecting each variant's
//!   field types. Any field of type `T`, `Box<T>`, `Option<T>`,
//!   `Vec<T>`, or `BTreeMap<String, T>` (each in its `Box`-wrapped
//!   form too), where `T` is the enum itself, is treated as a child.
//!   Keyed slots iterate in the map's own (sorted-by-key) order —
//!   keys themselves are not surfaced through the walk.
//! - `WalkMut` — mutable counterpart of `Walk`.
//!
//! `#[derive(DslSchema)]` accepts the same shape and emits an
//! `impl DslSchema` returning a `NodeSchema` — the type-level view of
//! variants, non-recursive payload fields, and child-field
//! multiplicity. See `dsl-kit-schema` for the target types.
//!
//! `#[derive(DslBuild)]` (G-1) accepts the same shape and emits an
//! `impl DslBuild` that converts a validated `ParseTree` into a typed
//! AST value, minting fresh `NodeId`s from the caller's `IdGen`. It
//! delegates payload deserialization to `dsl_kit_parse::build_field`
//! and child-slot recursion to `build_child_one` / `build_child_optional`
//! / `build_child_many` / `build_child_map`. Types deriving `DslBuild` must also derive
//! `DslSchema` (used for the level-scoped conformance check). Payload
//! fields whose type has no `FromStr` route (e.g. `Option<T>`, which
//! the orphan rule bars downstream crates from implementing) opt out of
//! `build_field` with `#[dsl_build(with = path)]` — the named function
//! converts the field itself. See [`derive_dsl_build`].
//!
//! `#[derive(DslDump)]` is `DslBuild`'s inverse: it emits an
//! `impl DslDump` that re-serializes a typed AST value back into the
//! `ParseTree` shape the build derive accepts, so the pair round-trips
//! (modulo fresh `NodeId`s). Chain the emitted tree through
//! `dsl_kit_parse::serde_bridge::to_canonical_json` (or use the
//! `dsl_kit_parse::dump_canonical_json` convenience) to serialize an
//! in-memory AST as canonical bridge JSON. Payload fields must
//! implement `serde::Serialize`; a field carrying
//! `#[dsl_build(with = ...)]` must carry the dual
//! `#[dsl_dump(with = path)]` serializer. See [`derive_dsl_dump`].
//!
//! `#[derive(DslCheck)]` accepts the same shape and emits an
//! `impl DslCheck` returning a `CheckProgram` — the semantic judgement
//! rules of the DSL as data. Variants carry `#[dsl_check(requires =
//! "state(A)", produces = "state(B)")]` for the sequential half and
//! `#[dsl_check(requires(cond = "type(Bool)"), concludes = "type($a)")]`
//! for the tree-typing half; `bind(var = "field")` wires a `$var` to a
//! payload field, and a `Vec<Self>` child slot carries
//! `#[dsl_check(fold(initial = "state(Atom)"))]` to declare that its
//! elements are ordered and thread a state. The macro compiles those
//! strings into a `CheckProgram` construction expression exactly as
//! `#[derive(DslSchema)]` compiles a shape into a `NodeSchema` one.
//! See [`derive_dsl_check`].
//!
//! `#[derive(DslExec)]` accepts the same shape and emits an
//! `impl DslExec` — the mechanical half of an engine `Ast`. Every
//! variant names its engine `NodeKind` through a `#[dsl_exec(...)]`
//! annotation (`value` / `read(field)` / `apply = "op"` / `bind(field)`
//! / `branch` / `repeat` / `seq` / `scope(field)` / `maybe` /
//! `call(field)`, the last of which may name an effect payload —
//! `call(label, payload(src, dst))`); recursive child fields are picked up in declaration
//! order. Pair the impl with a `DslSemantics` implementation via
//! `dsl_kit_core::DerivedAst` to obtain a runnable `Ast`. See
//! [`derive_dsl_exec`].
//!
//! Variants may carry additional fields of unrelated types (payload); those
//! fields are ignored by the traversal.
//!
//! Advanced shapes (indirect recursion through a struct, mixed tuple /
//! named variants, generic ASTs) can implement the traits by hand.

#![warn(missing_docs)]

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    Data, DeriveInput, Fields, GenericArgument, Ident, PathArguments, Type, TypePath,
    parse_macro_input,
};

#[derive(Clone, Copy)]
enum Recursion {
    /// `T`
    Direct,
    /// `Box<T>`
    Boxed,
    /// `Option<T>`
    Optional,
    /// `Option<Box<T>>`
    OptionalBoxed,
    /// `Vec<T>`
    Many,
    /// `Vec<Box<T>>`
    ManyBoxed,
    /// `BTreeMap<String, T>` — self-recursive keyed slot with a bare
    /// enum value. Rare in practice (recursive fields usually spell
    /// the enum through `Box` to keep the enum object-safe), but
    /// recognised for symmetry with [`Recursion::Direct`] /
    /// [`Recursion::Many`].
    Map,
    /// `BTreeMap<String, Box<T>>` — the common self-recursive keyed
    /// slot spelling.
    MapBoxed,
}

/// Returns the last path segment of a `Type::Path`, if that's what `ty` is.
fn last_segment(ty: &Type) -> Option<&syn::PathSegment> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return None;
    };
    path.segments.last()
}

/// If `seg` is `Wrapper<Inner>` (single generic type argument), returns
/// `Inner`.
fn single_generic_type(seg: &syn::PathSegment) -> Option<&Type> {
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let GenericArgument::Type(inner) = args.args.first()? else {
        return None;
    };
    Some(inner)
}

/// If `seg` is `Wrapper<A, B>` (exactly two generic type arguments,
/// both types), returns `(A, B)`. Used to unpack
/// `BTreeMap<String, T>` and its siblings for keyed-slot
/// recognition; returns `None` for any other arity or shape.
fn two_generic_types(seg: &syn::PathSegment) -> Option<(&Type, &Type)> {
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    if args.args.len() != 2 {
        return None;
    }
    let mut iter = args.args.iter();
    let (Some(GenericArgument::Type(a)), Some(GenericArgument::Type(b))) =
        (iter.next(), iter.next())
    else {
        return None;
    };
    Some((a, b))
}

/// Returns true if `ty` is a `Type::Path` whose last segment is
/// `String` (no generic arguments). Used to gate keyed-slot
/// recognition — the first generic parameter of a keyed map must be
/// a `String` key.
fn is_string_type(ty: &Type) -> bool {
    match last_segment(ty) {
        Some(seg) => seg.ident == "String" && matches!(seg.arguments, PathArguments::None),
        None => false,
    }
}

/// Returns true if `ty` is a `Type::Path` whose last segment matches
/// `enum_name` and carries no generic arguments.
fn matches_enum(ty: &Type, enum_name: &Ident) -> bool {
    match last_segment(ty) {
        Some(seg) => seg.ident == *enum_name && matches!(seg.arguments, PathArguments::None),
        None => false,
    }
}

/// Payload-side (non-recursive) type shape recognized by the derive.
///
/// Mirrors the child-side [`Recursion`] enum but only tracks the two
/// wrappers whose absence has an obvious default: `Option<T>` (→ `None`)
/// and `Vec<T>` (→ empty vec). Any other payload type is treated as
/// `PayloadShape::Bare` and takes the `build_field` path.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PayloadShape {
    /// Plain payload type: uses `dsl_kit_parse::build_field`.
    Bare,
    /// `Option<T>` where `T` is not the enum itself. Absent field →
    /// `None`.
    OptionInner,
    /// `Vec<T>` where `T` is not the enum itself. Absent field →
    /// `vec![]`.
    VecInner,
}

/// Classifies a payload field by outer wrapper. Recursive shapes
/// (detected by [`detect_recursion`]) are handled separately and never
/// pass through here.
fn payload_shape(ty: &Type, enum_name: &Ident) -> (PayloadShape, Option<Type>) {
    let Some(seg) = last_segment(ty) else {
        return (PayloadShape::Bare, None);
    };
    let Some(inner) = single_generic_type(seg) else {
        return (PayloadShape::Bare, None);
    };
    // `Option<T>` / `Vec<T>` where `T` itself is the enum are recursive
    // shapes handled by `detect_recursion`; skip them here.
    if matches_enum(inner, enum_name) {
        return (PayloadShape::Bare, None);
    }
    match seg.ident.to_string().as_str() {
        "Option" => (PayloadShape::OptionInner, Some(inner.clone())),
        "Vec" => (PayloadShape::VecInner, Some(inner.clone())),
        _ => (PayloadShape::Bare, None),
    }
}

/// If `ty` is a `BTreeMap<String, V>` whose value type `V` is **not**
/// the enclosing enum (nor `Box<Self>`), returns the value type. This
/// is Shape 1 of the tracking issue — scalar-valued keyed slots
/// (`BTreeMap<String, String>`, `BTreeMap<String, i64>`, etc.).
///
/// `HashMap` and other keyed containers are deliberately not matched:
/// the schema layer commits to deterministic iteration order for
/// keyed slots, and only `BTreeMap` provides it structurally.
///
/// The self-recursive keyed shapes (`BTreeMap<String, Self>` /
/// `BTreeMap<String, Box<Self>>`) are handled by
/// [`detect_recursion`] and never reach this helper — the derive
/// checks recursion first.
fn detect_scalar_map(ty: &Type, enum_name: &Ident) -> Option<Type> {
    let seg = last_segment(ty)?;
    if seg.ident != "BTreeMap" {
        return None;
    }
    let (k, v) = two_generic_types(seg)?;
    if !is_string_type(k) {
        return None;
    }
    // `Self` and `Box<Self>` are the recursive shapes — leave them
    // to `detect_recursion`.
    if matches_enum(v, enum_name) {
        return None;
    }
    if let Some(inner_seg) = last_segment(v)
        && inner_seg.ident == "Box"
    {
        let inner_inner = single_generic_type(inner_seg)?;
        if matches_enum(inner_inner, enum_name) {
            return None;
        }
    }
    Some(v.clone())
}

fn detect_recursion(ty: &Type, enum_name: &Ident) -> Option<Recursion> {
    if matches_enum(ty, enum_name) {
        return Some(Recursion::Direct);
    }

    let seg = last_segment(ty)?;

    // `BTreeMap<K, V>` takes two type parameters; try that shape first
    // so the single-generic-type extraction below does not silently
    // succeed on the first parameter alone.
    if seg.ident == "BTreeMap" {
        let (k, v) = two_generic_types(seg)?;
        if !is_string_type(k) {
            return None;
        }
        if matches_enum(v, enum_name) {
            return Some(Recursion::Map);
        }
        if let Some(inner_seg) = last_segment(v)
            && inner_seg.ident == "Box"
        {
            let inner_inner = single_generic_type(inner_seg)?;
            if matches_enum(inner_inner, enum_name) {
                return Some(Recursion::MapBoxed);
            }
        }
        return None;
    }

    let inner = single_generic_type(seg)?;

    match seg.ident.to_string().as_str() {
        "Box" => {
            if matches_enum(inner, enum_name) {
                Some(Recursion::Boxed)
            } else {
                None
            }
        }
        "Option" => {
            if matches_enum(inner, enum_name) {
                Some(Recursion::Optional)
            } else if let Some(inner_seg) = last_segment(inner) {
                if inner_seg.ident == "Box" {
                    let inner_inner = single_generic_type(inner_seg)?;
                    if matches_enum(inner_inner, enum_name) {
                        return Some(Recursion::OptionalBoxed);
                    }
                }
                None
            } else {
                None
            }
        }
        "Vec" => {
            if matches_enum(inner, enum_name) {
                Some(Recursion::Many)
            } else if let Some(inner_seg) = last_segment(inner) {
                if inner_seg.ident == "Box" {
                    let inner_inner = single_generic_type(inner_seg)?;
                    if matches_enum(inner_inner, enum_name) {
                        return Some(Recursion::ManyBoxed);
                    }
                }
                None
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Derives `dsl_kit_core::DslNode`, `Walk`, and `WalkMut` for an enum
/// whose variants use named fields and carry an `id: NodeId` slot.
///
/// See the crate-level docs for the expected shape and the way
/// recursive fields are picked up.
#[proc_macro_derive(DslNode)]
pub fn derive_dsl_node(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(&input, "#[derive(DslNode)] currently supports enums only")
            .to_compile_error()
            .into();
    };

    let mut node_arms = Vec::new();
    let mut variant_name_arms = Vec::new();
    let mut child_arms = Vec::new();
    let mut child_mut_arms = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;
        let variant_name_str = variant_ident.to_string();

        let Fields::Named(fields) = &variant.fields else {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslNode)] requires every variant to use named fields",
            )
            .to_compile_error()
            .into();
        };

        // Locate the `id` field.
        let has_id = fields
            .named
            .iter()
            .any(|f| f.ident.as_ref().is_some_and(|ident| ident == "id"));
        if !has_id {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslNode)] requires each variant to have an `id: NodeId` field",
            )
            .to_compile_error()
            .into();
        }

        // Collect recursive fields (ident, kind).
        let mut recursive: Vec<(Ident, Recursion)> = Vec::new();
        for f in &fields.named {
            let Some(ident) = &f.ident else { continue };
            if ident == "id" {
                continue;
            }
            if let Some(kind) = detect_recursion(&f.ty, &name) {
                recursive.push((ident.clone(), kind));
            }
        }

        // node_id arm.
        node_arms.push(quote! {
            Self::#variant_ident { id, .. } => *id,
        });

        // variant_name arm.
        variant_name_arms.push(quote! {
            Self::#variant_ident { .. } => #variant_name_str,
        });

        // children arm.
        let push_stmts = recursive.iter().map(|(field_ident, kind)| match kind {
            Recursion::Direct => quote! { _v.push(#field_ident); },
            Recursion::Boxed => quote! { _v.push(&**#field_ident); },
            Recursion::Optional => quote! {
                if let Some(inner) = #field_ident.as_ref() { _v.push(inner); }
            },
            Recursion::OptionalBoxed => quote! {
                if let Some(inner) = #field_ident.as_deref() { _v.push(inner); }
            },
            Recursion::Many => quote! { _v.extend(#field_ident.iter()); },
            Recursion::ManyBoxed => quote! {
                _v.extend(#field_ident.iter().map(::std::convert::AsRef::as_ref));
            },
            // `BTreeMap<String, T>` — iterate values in the map's
            // (sorted) key order. Keys are traversal-invisible; a
            // future keyed-walk API can expose them separately.
            Recursion::Map => quote! { _v.extend(#field_ident.values()); },
            // `BTreeMap<String, Box<T>>` — same iteration order, unbox
            // each value on the way out.
            Recursion::MapBoxed => quote! {
                _v.extend(#field_ident.values().map(::std::convert::AsRef::as_ref));
            },
        });
        let field_binds = recursive.iter().map(|(id, _)| quote!(#id));
        child_arms.push(quote! {
            Self::#variant_ident { #(#field_binds,)* .. } => {
                let mut _v: ::std::vec::Vec<&#name> = ::std::vec::Vec::new();
                #(#push_stmts)*
                _v
            }
        });

        // children_mut arm.
        let push_mut_stmts = recursive.iter().map(|(field_ident, kind)| match kind {
            Recursion::Direct => quote! { _v.push(#field_ident); },
            Recursion::Boxed => quote! { _v.push(&mut **#field_ident); },
            Recursion::Optional => quote! {
                if let Some(inner) = #field_ident.as_mut() { _v.push(inner); }
            },
            Recursion::OptionalBoxed => quote! {
                if let Some(inner) = #field_ident.as_deref_mut() { _v.push(inner); }
            },
            Recursion::Many => quote! { _v.extend(#field_ident.iter_mut()); },
            Recursion::ManyBoxed => quote! {
                _v.extend(#field_ident.iter_mut().map(::std::convert::AsMut::as_mut));
            },
            Recursion::Map => quote! { _v.extend(#field_ident.values_mut()); },
            Recursion::MapBoxed => quote! {
                _v.extend(#field_ident.values_mut().map(::std::convert::AsMut::as_mut));
            },
        });
        let field_binds_mut = recursive.iter().map(|(id, _)| quote!(#id));
        child_mut_arms.push(quote! {
            Self::#variant_ident { #(#field_binds_mut,)* .. } => {
                let mut _v: ::std::vec::Vec<&mut #name> = ::std::vec::Vec::new();
                #(#push_mut_stmts)*
                _v
            }
        });
    }

    let expanded: TokenStream2 = quote! {
        impl #impl_generics ::dsl_kit_core::DslNode for #name #ty_generics #where_clause {
            fn node_id(&self) -> ::dsl_kit_core::NodeId {
                match self {
                    #(#node_arms)*
                }
            }

            fn variant_name(&self) -> &'static str {
                match self {
                    #(#variant_name_arms)*
                }
            }
        }

        impl #impl_generics ::dsl_kit_core::Walk for #name #ty_generics #where_clause {
            fn children(&self) -> ::std::vec::Vec<&Self> {
                match self {
                    #(#child_arms)*
                }
            }
        }

        impl #impl_generics ::dsl_kit_core::WalkMut for #name #ty_generics #where_clause {
            fn children_mut(&mut self) -> ::std::vec::Vec<&mut Self> {
                match self {
                    #(#child_mut_arms)*
                }
            }
        }
    };

    expanded.into()
}

/// One parsed `#[dsl_schema(scalar(<kind> = Variant::field))]`
/// declaration on a child-slot field.
struct ScalarShorthandDecl {
    /// Declared kind key: `"string"` / `"int"` / `"bool"`.
    kind: String,
    /// Target variant name (first path segment).
    variant: String,
    /// Target payload field name (second path segment).
    field: String,
    /// The declaration's path, kept for error spans.
    path: syn::Path,
}

/// Integer payload types a `scalar(int = ...)` shorthand may target.
/// Mirrors `dsl_kit_parse::schema_gen`'s built-in `%int` mapping.
const SCALAR_INT_TYPES: &[&str] = &[
    "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "usize", "isize",
];

/// Parsed `#[dsl_schema(...)]` annotations on one field.
#[derive(Default)]
struct DslSchemaAttrs {
    /// `scalar(<kind> = Variant::field)` declarations.
    scalars: Vec<ScalarShorthandDecl>,
    /// Bare `non_empty` flag.
    non_empty: bool,
}

/// Parses a field's `#[dsl_schema(...)]` annotations
/// (`scalar(<kind> = Variant::field)` and/or `non_empty`).
/// `Ok(Default::default())` when the field carries no `dsl_schema`
/// attribute.
fn dsl_schema_attrs(f: &syn::Field) -> syn::Result<DslSchemaAttrs> {
    let mut out = DslSchemaAttrs::default();
    for attr in &f.attrs {
        if !attr.path().is_ident("dsl_schema") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("non_empty") {
                out.non_empty = true;
                return Ok(());
            }
            if !meta.path.is_ident("scalar") {
                return Err(meta.error(
                    "unsupported #[dsl_schema(...)] key; expected \
                     `scalar(<kind> = Variant::field)` or `non_empty`",
                ));
            }
            meta.parse_nested_meta(|inner| {
                let kind = inner
                    .path
                    .get_ident()
                    .map(ToString::to_string)
                    .unwrap_or_default();
                if !matches!(kind.as_str(), "string" | "int" | "bool") {
                    return Err(
                        inner.error("unsupported scalar kind; expected `string`, `int`, or `bool`")
                    );
                }
                let path: syn::Path = inner.value()?.parse()?;
                if path.segments.len() != 2 {
                    return Err(syn::Error::new_spanned(
                        &path,
                        "expected a `Variant::field` path (exactly two segments)",
                    ));
                }
                let variant = path.segments[0].ident.to_string();
                let field = path.segments[1].ident.to_string();
                out.scalars.push(ScalarShorthandDecl {
                    kind,
                    variant,
                    field,
                    path,
                });
                Ok(())
            })
        })?;
    }
    Ok(out)
}

/// Whether the field carries any `#[dsl_schema(...)]` attribute —
/// used to reject the attribute on positions it has no meaning for.
fn has_dsl_schema_attr(f: &syn::Field) -> bool {
    f.attrs.iter().any(|a| a.path().is_ident("dsl_schema"))
}

/// Validates parsed shorthand declarations against the whole enum:
/// no duplicate kinds, target variant exists, target field is a plain
/// payload field of a type the kind can carry. Mirrors the runtime
/// pre-flight in `dsl_kit_parse::schema_gen` so derive users get the
/// report at compile time instead.
fn validate_scalar_decls(
    decls: &[ScalarShorthandDecl],
    data: &syn::DataEnum,
    enum_name: &Ident,
) -> syn::Result<()> {
    let mut seen: Vec<&str> = Vec::new();
    for d in decls {
        if seen.contains(&d.kind.as_str()) {
            return Err(syn::Error::new_spanned(
                &d.path,
                format!(
                    "duplicate `scalar({} = ...)` declaration on this slot",
                    d.kind
                ),
            ));
        }
        seen.push(&d.kind);
        let Some(target) = data.variants.iter().find(|v| v.ident == d.variant) else {
            return Err(syn::Error::new_spanned(
                &d.path,
                format!(
                    "shorthand target variant `{}` is not a variant of this enum",
                    d.variant
                ),
            ));
        };
        let Fields::Named(fields) = &target.fields else {
            return Err(syn::Error::new_spanned(
                &d.path,
                format!(
                    "shorthand target variant `{}` must use named fields",
                    d.variant
                ),
            ));
        };
        let Some(target_field) = fields
            .named
            .iter()
            .find(|f| f.ident.as_ref().is_some_and(|i| i == d.field.as_str()))
        else {
            return Err(syn::Error::new_spanned(
                &d.path,
                format!(
                    "shorthand target variant `{}` has no field `{}`",
                    d.variant, d.field
                ),
            ));
        };
        if d.field == "id" || detect_recursion(&target_field.ty, enum_name).is_some() {
            return Err(syn::Error::new_spanned(
                &d.path,
                format!(
                    "shorthand target `{}::{}` must be a plain payload field",
                    d.variant, d.field
                ),
            ));
        }
        let ty = normalize_type_str(&target_field.ty.to_token_stream().to_string());
        let compatible = match d.kind.as_str() {
            "string" => ty == "String",
            "int" => SCALAR_INT_TYPES.contains(&ty.as_str()),
            "bool" => ty == "bool",
            _ => false,
        };
        if !compatible {
            return Err(syn::Error::new_spanned(
                &d.path,
                format!(
                    "scalar kind `{}` cannot target `{}::{}` of type `{}`",
                    d.kind, d.variant, d.field, ty
                ),
            ));
        }
    }
    Ok(())
}

/// Derives `dsl_kit_schema::DslSchema` for the same enum shape accepted
/// by [`DslNode`]. The generated `schema()` method returns a
/// `NodeSchema` describing every variant, its non-recursive payload
/// fields, and the multiplicity of each recursive child field.
///
/// The `id: NodeId` field is stripped from the schema — it is an
/// implementation detail of the observability layer, not part of the
/// DSL's external shape. Recursive fields (`T` / `Box<T>` /
/// `Option<T>` / `Option<Box<T>>` / `Vec<T>` / `Vec<Box<T>>` /
/// `BTreeMap<String, T>` / `BTreeMap<String, Box<T>>` where `T` is
/// the enum itself) become `ChildSchema` entries with
/// `ChildValueShape::Recursive`; the two keyed-map shapes report
/// `Multiplicity::Map`. Scalar-valued keyed maps
/// (`BTreeMap<String, V>` where `V` is a payload type such as
/// `String` / `i64` / `bool` — Shape 1 of the tracking issue) also
/// become `ChildSchema` entries with `Multiplicity::Map` and
/// `ChildValueShape::Scalar { ty }`; the value type is captured as
/// Rust source text. Every remaining named field becomes a
/// `FieldSchema` carrying the Rust type source text.
///
/// ## Scalar shorthands (`#[dsl_schema(scalar(...))]`)
///
/// A `One` / `Optional` child slot may declare that a bare scalar in
/// its position lowers to a named variant:
///
/// ```ignore
/// FsWrite {
///     id: NodeId,
///     path: String,
///     #[dsl_schema(scalar(string = Literal::value))]
///     content: Box<Self>,
/// },
/// Literal { id: NodeId, value: String },
/// ```
///
/// The declaration is recorded in the schema
/// (`ChildSchema::scalar_shorthands`) so every consumer — the JSON
/// bridge, the generated canonical-text grammar, wire formats — lowers
/// the same way. Kinds: `string` / `int` / `bool`, at most one each
/// per slot; the target must be a plain payload field of a matching
/// type on a variant of the same enum. The macro validates all of
/// that at compile time.
///
/// ## Non-empty collections (`#[dsl_schema(non_empty)]`)
///
/// A `Many` / `Map` collection slot (recursive or keyed-scalar) may
/// declare that it must hold at least one element:
///
/// ```ignore
/// Pipeline {
///     id: NodeId,
///     #[dsl_schema(non_empty)]
///     stages: Vec<Self>,
/// },
/// ```
///
/// Recorded as `ChildSchema::non_empty`; `check_conformance` rejects
/// a violating tree (`ARITY_NON_EMPTY`), generated grammars require
/// an element, and the `no-empty-child-slots` lint reports declared
/// violations only. Rejected at compile time on `One` / `Optional`
/// slots and on payload fields.
#[proc_macro_derive(DslSchema, attributes(dsl_schema))]
pub fn derive_dsl_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let name_str = name.to_string();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(
            &input,
            "#[derive(DslSchema)] currently supports enums only",
        )
        .to_compile_error()
        .into();
    };

    let mut variant_ctors = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;
        let variant_name = variant_ident.to_string();

        let Fields::Named(fields) = &variant.fields else {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslSchema)] requires every variant to use named fields",
            )
            .to_compile_error()
            .into();
        };

        let mut field_ctors = Vec::new();
        let mut child_ctors = Vec::new();

        for f in &fields.named {
            let Some(ident) = &f.ident else { continue };
            let ident_str = ident.to_string();
            if ident_str == "id" {
                continue;
            }

            if let Some(kind) = detect_recursion(&f.ty, &name) {
                let attrs = match dsl_schema_attrs(f) {
                    Ok(attrs) => attrs,
                    Err(e) => return e.to_compile_error().into(),
                };
                let one_or_optional = matches!(
                    kind,
                    Recursion::Direct
                        | Recursion::Boxed
                        | Recursion::Optional
                        | Recursion::OptionalBoxed
                );
                if !attrs.scalars.is_empty() && !one_or_optional {
                    return syn::Error::new_spanned(
                        f,
                        "#[dsl_schema(scalar(...))] applies to `One` / `Optional` child \
                         slots only (`T` / `Box<T>` / `Option<T>` / `Option<Box<T>>`)",
                    )
                    .to_compile_error()
                    .into();
                }
                if attrs.non_empty && one_or_optional {
                    return syn::Error::new_spanned(
                        f,
                        "#[dsl_schema(non_empty)] applies to `Many` / `Map` collection \
                         slots only (`Vec<T>` / `BTreeMap<String, T>` and their boxed \
                         forms) — `One` is inherently non-empty and `Optional` \
                         inherently permits absence",
                    )
                    .to_compile_error()
                    .into();
                }
                if let Err(e) = validate_scalar_decls(&attrs.scalars, data, &name) {
                    return e.to_compile_error().into();
                }
                let non_empty = attrs.non_empty;
                let shorthand_ctors: Vec<TokenStream2> = attrs
                    .scalars
                    .iter()
                    .map(|d| {
                        let kind_ident = match d.kind.as_str() {
                            "string" => quote!(Str),
                            "int" => quote!(Int),
                            _ => quote!(Bool),
                        };
                        let variant = &d.variant;
                        let field = &d.field;
                        quote! {
                            ::dsl_kit_schema::ScalarShorthand {
                                kind: ::dsl_kit_schema::ScalarKind::#kind_ident,
                                variant: #variant.to_string(),
                                field: #field.to_string(),
                            }
                        }
                    })
                    .collect();
                let mult = match kind {
                    Recursion::Direct | Recursion::Boxed => quote!(One),
                    Recursion::Optional | Recursion::OptionalBoxed => quote!(Optional),
                    Recursion::Many | Recursion::ManyBoxed => quote!(Many),
                    Recursion::Map | Recursion::MapBoxed => quote!(Map),
                };
                child_ctors.push(quote! {
                    ::dsl_kit_schema::ChildSchema {
                        name: #ident_str.to_string(),
                        multiplicity: ::dsl_kit_schema::Multiplicity::#mult,
                        value_shape: ::dsl_kit_schema::ChildValueShape::Recursive,
                        scalar_shorthands: ::std::vec![#(#shorthand_ctors),*],
                        non_empty: #non_empty,
                    }
                });
            } else if let Some(value_ty) = detect_scalar_map(&f.ty, &name) {
                // Shape 1: `BTreeMap<String, ScalarType>` — keyed
                // slot whose values are non-recursive payloads.
                // Reported as `Multiplicity::Map` with
                // `ChildValueShape::Scalar { ty }`; the value type is
                // stored as Rust source text so schema consumers can
                // dispatch on it without a semantic type system.
                // `non_empty` is legal here (a scalar map is a `Map`
                // slot); scalar shorthands are not.
                let attrs = match dsl_schema_attrs(f) {
                    Ok(attrs) => attrs,
                    Err(e) => return e.to_compile_error().into(),
                };
                if !attrs.scalars.is_empty() {
                    return syn::Error::new_spanned(
                        f,
                        "#[dsl_schema(scalar(...))] applies to `One` / `Optional` child \
                         slots, not keyed scalar slots",
                    )
                    .to_compile_error()
                    .into();
                }
                let non_empty = attrs.non_empty;
                let value_ty_src = normalize_type_str(&value_ty.to_token_stream().to_string());
                child_ctors.push(quote! {
                    ::dsl_kit_schema::ChildSchema {
                        name: #ident_str.to_string(),
                        multiplicity: ::dsl_kit_schema::Multiplicity::Map,
                        value_shape: ::dsl_kit_schema::ChildValueShape::Scalar {
                            ty: #value_ty_src.to_string(),
                        },
                        scalar_shorthands: ::std::vec![],
                        non_empty: #non_empty,
                    }
                });
            } else {
                if has_dsl_schema_attr(f) {
                    return syn::Error::new_spanned(
                        f,
                        "#[dsl_schema(...)] applies to `One` / `Optional` child slots, \
                         not payload fields",
                    )
                    .to_compile_error()
                    .into();
                }
                let ty_src = normalize_type_str(&f.ty.to_token_stream().to_string());
                let (shape, _) = payload_shape(&f.ty, &name);
                let optional = matches!(shape, PayloadShape::OptionInner | PayloadShape::VecInner,);
                field_ctors.push(quote! {
                    ::dsl_kit_schema::FieldSchema {
                        name: #ident_str.to_string(),
                        ty: #ty_src.to_string(),
                        optional: #optional,
                    }
                });
            }
        }

        variant_ctors.push(quote! {
            ::dsl_kit_schema::VariantSchema {
                name: #variant_name.to_string(),
                fields: ::std::vec![#(#field_ctors),*],
                children: ::std::vec![#(#child_ctors),*],
            }
        });
    }

    let expanded: TokenStream2 = quote! {
        impl #impl_generics ::dsl_kit_schema::DslSchema for #name #ty_generics #where_clause {
            fn schema() -> ::dsl_kit_schema::NodeSchema {
                ::dsl_kit_schema::NodeSchema {
                    name: #name_str.to_string(),
                    variants: ::std::vec![#(#variant_ctors),*],
                }
            }
        }
    };

    expanded.into()
}

/// Derives `dsl_kit_parse::DslBuild` for the same enum shape accepted
/// by `DslNode` and `DslSchema`. The generated `from_parse_tree`
/// method:
///
/// 1. Runs a level-scoped `check_conformance` against
///    `Self::schema()` and returns any diagnostics before proceeding.
///    Types deriving `DslBuild` must therefore also derive `DslSchema`.
/// 2. Dispatches on the `ParseTree`'s `variant` name against the
///    enum's variants.
/// 3. For each named field, calls `dsl_kit_parse::build_field` — the field's
///    Rust type must implement both `serde::de::DeserializeOwned` and
///    `FromStr`. `RawValue::Json` payloads dispatch through serde;
///    `RawValue::Text` payloads (the PEG front-end's natural output)
///    dispatch through `FromStr`. A field annotated
///    `#[dsl_build(with = path)]` bypasses `build_field`: `path` must
///    name a function
///    `fn(&ParseTree, &str) -> Result<T, BuildError>` (the field name
///    is passed as the second argument), lifting the two trait bounds.
///    This is the build-layer twin of `schema_gen::SyntaxOverrides` —
///    required for types like `Option<JoinPolicy>` where the orphan
///    rule bars a downstream `FromStr` impl. The attribute applies to
///    payload fields only; annotating a recursive child field is a
///    compile error.
/// 4. For each recursive child field, calls the appropriate helper
///    (`build_child_one` / `_optional` / `_many` / `_map`) and
///    re-wraps the result in `Box` where the source field is boxed.
///    Keyed slots (`BTreeMap<String, T>`) read the tree's keyed half
///    and keep their keys.
/// 5. Constructs the variant with a fresh `NodeId` from the
///    caller-supplied `IdGen`, first recording the tree's `$allow`
///    annotation (if any) against that id via `IdGen::record_allows`.
///    The caller reads the accumulated table back with
///    `IdGen::take_allows` once the whole tree is built.
///
/// (The named helpers live in `dsl_kit_parse`; this proc-macro crate
/// cannot intra-doc-link across crates it does not depend on.)
#[proc_macro_derive(DslBuild, attributes(dsl_build))]
pub fn derive_dsl_build(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(
            &input,
            "#[derive(DslBuild)] currently supports enums only",
        )
        .to_compile_error()
        .into();
    };

    let mut variant_arms = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;
        let variant_name_str = variant_ident.to_string();

        let Fields::Named(fields) = &variant.fields else {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslBuild)] requires every variant to use named fields",
            )
            .to_compile_error()
            .into();
        };

        let has_id = fields
            .named
            .iter()
            .any(|f| f.ident.as_ref().is_some_and(|ident| ident == "id"));
        if !has_id {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslBuild)] requires each variant to have an `id: NodeId` field",
            )
            .to_compile_error()
            .into();
        }

        let mut let_bindings = Vec::new();
        let mut ctor_fields = Vec::new();

        for f in &fields.named {
            let Some(ident) = &f.ident else { continue };
            if ident == "id" {
                continue;
            }
            let ident_str = ident.to_string();

            let with = match dsl_build_with_attr(f) {
                Ok(w) => w,
                Err(e) => return e.to_compile_error().into(),
            };
            if let Some(path) = &with {
                if detect_recursion(&f.ty, &name).is_some() {
                    return syn::Error::new_spanned(
                        f,
                        "#[dsl_build(with = ...)] applies to payload fields only, \
                         not recursive child fields",
                    )
                    .to_compile_error()
                    .into();
                }
                // If the field's type is `Option<T>` or `Vec<T>` the
                // schema marks it optional (see `derive_dsl_schema`),
                // so `check_conformance` no longer diagnoses absence.
                // The converter would still error on the missing field,
                // so short-circuit to the natural default before we
                // call it: `None` for `Option<T>`, `vec![]` for
                // `Vec<T>`. Bare payload types still delegate
                // unconditionally.
                let (shape, _) = payload_shape(&f.ty, &name);
                let call = match shape {
                    PayloadShape::OptionInner => quote! {
                        let #ident = if tree.field(#ident_str).is_some() {
                            #path(tree, #ident_str)?
                        } else {
                            ::std::option::Option::None
                        };
                    },
                    PayloadShape::VecInner => quote! {
                        let #ident = if tree.field(#ident_str).is_some() {
                            #path(tree, #ident_str)?
                        } else {
                            ::std::vec::Vec::new()
                        };
                    },
                    PayloadShape::Bare => quote! {
                        let #ident = #path(tree, #ident_str)?;
                    },
                };
                let_bindings.push(call);
                ctor_fields.push(quote! { #ident });
                continue;
            }

            if let Some(kind) = detect_recursion(&f.ty, &name) {
                let helper_call = match kind {
                    Recursion::Direct => quote! {
                        ::dsl_kit_parse::build_child_one::<#name>(tree, #ident_str, ids)?
                    },
                    Recursion::Boxed => quote! {
                        ::std::boxed::Box::new(
                            ::dsl_kit_parse::build_child_one::<#name>(tree, #ident_str, ids)?
                        )
                    },
                    Recursion::Optional => quote! {
                        ::dsl_kit_parse::build_child_optional::<#name>(tree, #ident_str, ids)?
                    },
                    Recursion::OptionalBoxed => quote! {
                        ::dsl_kit_parse::build_child_optional::<#name>(tree, #ident_str, ids)?
                            .map(::std::boxed::Box::new)
                    },
                    Recursion::Many => quote! {
                        ::dsl_kit_parse::build_child_many::<#name>(tree, #ident_str, ids)?
                    },
                    Recursion::ManyBoxed => quote! {
                        ::dsl_kit_parse::build_child_many::<#name>(tree, #ident_str, ids)?
                            .into_iter()
                            .map(::std::boxed::Box::new)
                            .collect::<::std::vec::Vec<_>>()
                    },
                    // Keyed slots (`BTreeMap<String, T>` /
                    // `BTreeMap<String, Box<T>>`) read from the tree's
                    // keyed half; `build_child_map` hands back a
                    // `BTreeMap<String, T>` already keyed by the
                    // front-end's keys.
                    Recursion::Map => quote! {
                        ::dsl_kit_parse::build_child_map::<#name>(tree, #ident_str, ids)?
                    },
                    // Same call, then re-box each value. Collected
                    // back into a `BTreeMap` (not the `Vec` idiom used
                    // by `ManyBoxed`) so the keys survive the rewrap.
                    Recursion::MapBoxed => quote! {
                        ::dsl_kit_parse::build_child_map::<#name>(tree, #ident_str, ids)?
                            .into_iter()
                            .map(|(k, v)| (k, ::std::boxed::Box::new(v)))
                            .collect::<::std::collections::BTreeMap<_, _>>()
                    },
                };
                let_bindings.push(quote! { let #ident = #helper_call; });
                ctor_fields.push(quote! { #ident });
            } else if let Some(value_ty) = detect_scalar_map(&f.ty, &name) {
                // Shape 1: `BTreeMap<String, ScalarType>` — keyed
                // slot whose values are scalar payloads. Reads from
                // the tree's keyed half via
                // `build_scalar_map`, which deserializes each entry's
                // payload with `build_field`'s FromStr / serde route.
                let_bindings.push(quote! {
                    let #ident =
                        ::dsl_kit_parse::build_scalar_map::<#value_ty>(tree, #ident_str)?;
                });
                ctor_fields.push(quote! { #ident });
            } else {
                let ty = &f.ty;
                let (shape, inner) = payload_shape(&f.ty, &name);
                match (shape, inner) {
                    (PayloadShape::OptionInner, Some(inner_ty)) => {
                        let_bindings.push(quote! {
                            let #ident: #ty =
                                ::dsl_kit_parse::build_field_optional::<#inner_ty>(
                                    tree, #ident_str,
                                )?;
                        });
                    }
                    (PayloadShape::VecInner, Some(inner_ty)) => {
                        let_bindings.push(quote! {
                            let #ident: #ty =
                                ::dsl_kit_parse::build_field_vec::<#inner_ty>(
                                    tree, #ident_str,
                                )?;
                        });
                    }
                    _ => {
                        let_bindings.push(quote! {
                            let #ident: #ty =
                                ::dsl_kit_parse::build_field(tree, #ident_str)?;
                        });
                    }
                }
                ctor_fields.push(quote! { #ident });
            }
        }

        // The node's id is minted into a local first so a `$allow`
        // annotation on the tree can be recorded against it before the
        // variant is constructed. `__dsl_kit_`-prefixed so the binding
        // cannot shadow a field ident.
        variant_arms.push(quote! {
            #variant_name_str => {
                #(#let_bindings)*
                let __dsl_kit_id = ids.node();
                if !tree.allows.is_empty() {
                    ids.record_allows(__dsl_kit_id, tree.allows.clone());
                }
                ::std::result::Result::Ok(Self::#variant_ident {
                    id: __dsl_kit_id,
                    #(#ctor_fields,)*
                })
            }
        });
    }

    let expanded: TokenStream2 = quote! {
        impl #impl_generics ::dsl_kit_parse::DslBuild for #name #ty_generics #where_clause {
            fn from_parse_tree(
                tree: &::dsl_kit_parse::ParseTree,
                ids: &::dsl_kit_core::IdGen,
            ) -> ::std::result::Result<Self, ::dsl_kit_parse::BuildError> {
                let __level_diags = ::dsl_kit_parse::check_conformance(
                    tree,
                    &<Self as ::dsl_kit_schema::DslSchema>::schema(),
                );
                if !__level_diags.is_empty() {
                    return ::std::result::Result::Err(
                        ::dsl_kit_parse::BuildError::new(__level_diags),
                    );
                }
                match tree.variant.as_str() {
                    #(#variant_arms)*
                    other => ::std::unreachable!(
                        "check_conformance accepted unknown variant `{}` — this is a bug",
                        other,
                    ),
                }
            }
        }
    };

    expanded.into()
}

/// Derives `dsl_kit_parse::DslDump` — the inverse of
/// `#[derive(DslBuild)]` — for an enum whose variants use named fields
/// and carry an `id: NodeId` field.
///
/// The generated `to_parse_tree_with` re-emits the `ParseTree` shape
/// the same enum's `DslBuild` derive accepts, so the pair round-trips
/// (`from_parse_tree(&ast.to_parse_tree()?, &ids)` rebuilds an
/// equivalent AST, modulo fresh `NodeId`s). Field routing mirrors the
/// build derive exactly:
///
/// 1. The `id` field is not serialized; it is only used to look up the
///    node's `$allow` annotation in the caller-supplied `AllowTable`.
/// 2. Payload fields emit through `dump_field` /
///    `dump_field_optional` (types must implement
///    `serde::Serialize`). An absent `Option` omits its key; `Vec`
///    payloads always emit, including `[]`.
/// 3. A field annotated `#[dsl_dump(with = path)]` bypasses
///    `dump_field`: `path` must name a function
///    `fn(&T) -> Result<Option<serde_json::Value>, BuildError>`, where
///    `Ok(None)` omits the key. This is the dual of
///    `#[dsl_build(with = path)]`, and the derive **requires** it on
///    any field that carries `#[dsl_build(with = ...)]` — a custom
///    build converter cannot be inverted mechanically, so the dual
///    serializer must be spelled out. The attribute applies to payload
///    fields only; annotating a recursive child field is a compile
///    error.
/// 4. Recursive child fields emit through `dump_child_one` /
///    `_optional` / `_many` / `_map`, unwrapping `Box` where the
///    source field is boxed. Keyed slots (`BTreeMap<String, T>`) emit
///    in map iteration order — already ascending by key, as
///    conformance demands. Scalar-valued keyed slots emit through
///    `dump_scalar_map`.
///
/// (The named helpers live in `dsl_kit_parse::dump`; this proc-macro
/// crate cannot intra-doc-link across crates it does not depend on.)
#[proc_macro_derive(DslDump, attributes(dsl_dump))]
pub fn derive_dsl_dump(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(&input, "#[derive(DslDump)] currently supports enums only")
            .to_compile_error()
            .into();
    };

    let mut variant_arms = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;
        let variant_name_str = variant_ident.to_string();

        let Fields::Named(fields) = &variant.fields else {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslDump)] requires every variant to use named fields",
            )
            .to_compile_error()
            .into();
        };

        let has_id = fields
            .named
            .iter()
            .any(|f| f.ident.as_ref().is_some_and(|ident| ident == "id"));
        if !has_id {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslDump)] requires each variant to have an `id: NodeId` field",
            )
            .to_compile_error()
            .into();
        }

        let mut bind_idents = Vec::new();
        let mut emit_stmts = Vec::new();

        for f in &fields.named {
            let Some(ident) = &f.ident else { continue };
            if ident == "id" {
                continue;
            }
            bind_idents.push(ident.clone());
            let ident_str = ident.to_string();

            let dump_with = match dsl_dump_with_attr(f) {
                Ok(w) => w,
                Err(e) => return e.to_compile_error().into(),
            };
            let build_with = match dsl_build_with_attr(f) {
                Ok(w) => w,
                Err(e) => return e.to_compile_error().into(),
            };
            // A scalar keyed slot (`BTreeMap<String, V>`) is a keyed
            // Map slot in the schema, but a dump-side `with` fn can
            // only land its value in `tree.fields` — every dump would
            // then fail conformance with CHILD_AS_FIELD. (The build
            // side is fine: its `with` fn receives the whole
            // `&ParseTree` and can read the keyed half.) Reject at
            // compile time instead of emitting a serializer that can
            // never conform.
            if (build_with.is_some() || dump_with.is_some())
                && detect_scalar_map(&f.ty, &name).is_some()
            {
                return syn::Error::new_spanned(
                    f,
                    "custom converters on scalar keyed slots (`BTreeMap<String, _>`) cannot \
                     be expressed by #[derive(DslDump)] — the dump-side `with` output lands \
                     in payload fields while the schema declares a keyed Map slot; write a \
                     hand-written DslDump impl for this enum instead",
                )
                .to_compile_error()
                .into();
            }
            if build_with.is_some() && dump_with.is_none() {
                return syn::Error::new_spanned(
                    f,
                    "this field has #[dsl_build(with = ...)] but no #[dsl_dump(with = ...)]; \
                     a custom build converter cannot be inverted mechanically — provide the \
                     dual serializer `fn(&T) -> Result<Option<serde_json::Value>, BuildError>`",
                )
                .to_compile_error()
                .into();
            }
            if let Some(path) = &dump_with {
                if detect_recursion(&f.ty, &name).is_some() {
                    return syn::Error::new_spanned(
                        f,
                        "#[dsl_dump(with = ...)] applies to payload fields only, \
                         not recursive child fields",
                    )
                    .to_compile_error()
                    .into();
                }
                emit_stmts.push(quote! {
                    if let ::std::option::Option::Some(__dsl_kit_v) = #path(#ident)? {
                        __dsl_kit_tree.fields.push((
                            #ident_str.to_string(),
                            ::dsl_kit_parse::RawValue::Json(__dsl_kit_v),
                        ));
                    }
                });
                continue;
            }

            if let Some(kind) = detect_recursion(&f.ty, &name) {
                let call = match kind {
                    Recursion::Direct => quote! {
                        ::dsl_kit_parse::dump_child_one(
                            &mut __dsl_kit_tree, #ident_str, #ident, __dsl_kit_allows,
                        )?;
                    },
                    Recursion::Boxed => quote! {
                        ::dsl_kit_parse::dump_child_one(
                            &mut __dsl_kit_tree, #ident_str, &**#ident, __dsl_kit_allows,
                        )?;
                    },
                    Recursion::Optional => quote! {
                        ::dsl_kit_parse::dump_child_optional(
                            &mut __dsl_kit_tree, #ident_str, #ident.as_ref(), __dsl_kit_allows,
                        )?;
                    },
                    Recursion::OptionalBoxed => quote! {
                        ::dsl_kit_parse::dump_child_optional(
                            &mut __dsl_kit_tree, #ident_str, #ident.as_deref(), __dsl_kit_allows,
                        )?;
                    },
                    Recursion::Many => quote! {
                        ::dsl_kit_parse::dump_child_many(
                            &mut __dsl_kit_tree, #ident_str, #ident.iter(), __dsl_kit_allows,
                        )?;
                    },
                    Recursion::ManyBoxed => quote! {
                        ::dsl_kit_parse::dump_child_many(
                            &mut __dsl_kit_tree,
                            #ident_str,
                            #ident.iter().map(|__dsl_kit_b| &**__dsl_kit_b),
                            __dsl_kit_allows,
                        )?;
                    },
                    Recursion::Map => quote! {
                        ::dsl_kit_parse::dump_child_map(
                            &mut __dsl_kit_tree, #ident_str, #ident.iter(), __dsl_kit_allows,
                        )?;
                    },
                    Recursion::MapBoxed => quote! {
                        ::dsl_kit_parse::dump_child_map(
                            &mut __dsl_kit_tree,
                            #ident_str,
                            #ident.iter().map(|(__dsl_kit_k, __dsl_kit_v)| {
                                (__dsl_kit_k, &**__dsl_kit_v)
                            }),
                            __dsl_kit_allows,
                        )?;
                    },
                };
                emit_stmts.push(call);
            } else if detect_scalar_map(&f.ty, &name).is_some() {
                emit_stmts.push(quote! {
                    ::dsl_kit_parse::dump_scalar_map(&mut __dsl_kit_tree, #ident_str, #ident)?;
                });
            } else {
                let (shape, _) = payload_shape(&f.ty, &name);
                let call = match shape {
                    PayloadShape::OptionInner => quote! {
                        ::dsl_kit_parse::dump_field_optional(
                            &mut __dsl_kit_tree, #ident_str, #ident,
                        )?;
                    },
                    _ => quote! {
                        ::dsl_kit_parse::dump_field(&mut __dsl_kit_tree, #ident_str, #ident)?;
                    },
                };
                emit_stmts.push(call);
            }
        }

        variant_arms.push(quote! {
            Self::#variant_ident { id: __dsl_kit_id, #(#bind_idents,)* } => {
                let mut __dsl_kit_tree = ::dsl_kit_parse::ParseTree::new(#variant_name_str);
                if let ::std::option::Option::Some(__dsl_kit_node_allows) =
                    __dsl_kit_allows.get(__dsl_kit_id)
                {
                    __dsl_kit_tree.allows = __dsl_kit_node_allows.clone();
                }
                #(#emit_stmts)*
                ::std::result::Result::Ok(__dsl_kit_tree)
            }
        });
    }

    let expanded: TokenStream2 = quote! {
        impl #impl_generics ::dsl_kit_parse::DslDump for #name #ty_generics #where_clause {
            // `__dsl_kit_`-prefixed parameter so a user field named
            // `allows` cannot shadow the table inside the variant arms
            // (same hygiene convention as `__dsl_kit_id` on the build
            // side).
            fn to_parse_tree_with(
                &self,
                __dsl_kit_allows: &::dsl_kit_core::AllowTable,
            ) -> ::std::result::Result<
                ::dsl_kit_parse::ParseTree,
                ::dsl_kit_parse::BuildError,
            > {
                match self {
                    #(#variant_arms)*
                }
            }
        }
    };

    expanded.into()
}

/// Per-variant classification parsed from `#[dsl_exec(...)]`.
enum ExecSpec {
    /// `#[dsl_exec(value)]` — literal leaf; the single payload field is
    /// the value.
    Value,
    /// `#[dsl_exec(read(FIELD))]` — env read named by `FIELD`.
    Read { name_field: Ident },
    /// `#[dsl_exec(apply = "OP")]` — op fold over the recursive fields.
    Apply { op: String },
    /// `#[dsl_exec(bind(FIELD))]` — binding named by `FIELD`; the two
    /// recursive fields are the value and the body, in order.
    Bind { name_field: Ident },
    /// `#[dsl_exec(branch)]` — the recursive fields are cond / then /
    /// optional else, in order.
    Branch,
    /// `#[dsl_exec(repeat)]` — loop over the single recursive field.
    Repeat,
    /// `#[dsl_exec(seq)]` — sequential children.
    Seq,
    /// `#[dsl_exec(scope(FIELD))]` — labelled wrapper around the single
    /// recursive field.
    Scope { label_field: Ident },
    /// `#[dsl_exec(maybe)]` — optional body.
    Maybe,
    /// `#[dsl_exec(call(FIELD))]` — effect leaf labelled by `FIELD`,
    /// optionally carrying an effect payload (see [`CallPayload`]).
    Call {
        label_field: Ident,
        payload: CallPayload,
    },
}

/// What `#[dsl_exec(call(LABEL, ...))]` puts in `NodeKind::Call`'s
/// payload — the effect's argument channel, handed to the host verbatim
/// through `CallSpec::payload`.
#[derive(Debug, Clone)]
enum CallPayload {
    /// `call(LABEL)` — no arguments beyond the label; payload is `null`.
    None,
    /// `call(LABEL, payload)` — every non-recursive field except the
    /// label, as a JSON object keyed by field name.
    AllFields,
    /// `call(LABEL, payload(a, b))` — the listed fields only, as a JSON
    /// object keyed by field name.
    Fields(Vec<Ident>),
    /// `call(LABEL, payload = FIELD)` — that one field serialised on its
    /// own, with no surrounding object.
    Single(Ident),
}

/// Parses the variant's `#[dsl_exec(...)]` annotation. Errors when the
/// attribute is missing or carries an unknown form.
fn dsl_exec_attr(variant: &syn::Variant) -> syn::Result<ExecSpec> {
    for attr in &variant.attrs {
        if !attr.path().is_ident("dsl_exec") {
            continue;
        }
        let mut spec: Option<ExecSpec> = None;
        let field_of = |meta: &syn::meta::ParseNestedMeta| -> syn::Result<Ident> {
            let mut field: Option<Ident> = None;
            meta.parse_nested_meta(|inner| match inner.path.get_ident() {
                Some(ident) if field.is_none() => {
                    field = Some(ident.clone());
                    Ok(())
                }
                _ => Err(inner.error("expected a single field name")),
            })?;
            field.ok_or_else(|| meta.error("expected a field name argument"))
        };
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("value") {
                spec = Some(ExecSpec::Value);
            } else if meta.path.is_ident("read") {
                spec = Some(ExecSpec::Read {
                    name_field: field_of(&meta)?,
                });
            } else if meta.path.is_ident("apply") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                spec = Some(ExecSpec::Apply { op: lit.value() });
            } else if meta.path.is_ident("bind") {
                spec = Some(ExecSpec::Bind {
                    name_field: field_of(&meta)?,
                });
            } else if meta.path.is_ident("branch") {
                spec = Some(ExecSpec::Branch);
            } else if meta.path.is_ident("repeat") {
                spec = Some(ExecSpec::Repeat);
            } else if meta.path.is_ident("seq") {
                spec = Some(ExecSpec::Seq);
            } else if meta.path.is_ident("scope") {
                spec = Some(ExecSpec::Scope {
                    label_field: field_of(&meta)?,
                });
            } else if meta.path.is_ident("maybe") {
                spec = Some(ExecSpec::Maybe);
            } else if meta.path.is_ident("call") {
                let (label_field, payload) = call_form(&meta)?;
                spec = Some(ExecSpec::Call {
                    label_field,
                    payload,
                });
            } else {
                return Err(meta.error(
                    "unknown #[dsl_exec(...)] form; expected one of value / read(field) / \
                     apply = \"op\" / bind(field) / branch / repeat / seq / scope(field) / \
                     maybe / call(field[, payload | payload(a, b) | payload = field])",
                ));
            }
            Ok(())
        })?;
        return spec
            .ok_or_else(|| syn::Error::new_spanned(attr, "#[dsl_exec(...)] requires a form"));
    }
    Err(syn::Error::new_spanned(
        variant,
        "#[derive(DslExec)] requires a #[dsl_exec(...)] annotation on every variant \
         (or implement Ast by hand for advanced shapes)",
    ))
}

/// Parses the inside of `call(...)`: a mandatory label field name plus
/// an optional `payload` clause in one of three shapes —
/// `payload` (every other non-recursive field), `payload(a, b)` (those
/// fields), `payload = field` (that field, unwrapped).
///
/// A variant whose *label* field is literally named `payload` has to
/// implement `exec_kind` by hand; `call(payload)` reads as the clause.
fn call_form(meta: &syn::meta::ParseNestedMeta) -> syn::Result<(Ident, CallPayload)> {
    let mut label: Option<Ident> = None;
    let mut payload = CallPayload::None;
    meta.parse_nested_meta(|inner| {
        if inner.path.is_ident("payload") {
            if inner.input.peek(syn::token::Paren) {
                let mut fields: Vec<Ident> = Vec::new();
                inner.parse_nested_meta(|f| match f.path.get_ident() {
                    Some(ident) => {
                        fields.push(ident.clone());
                        Ok(())
                    }
                    None => Err(f.error("expected a field name")),
                })?;
                if fields.is_empty() {
                    return Err(inner.error("payload(...) needs at least one field name"));
                }
                payload = CallPayload::Fields(fields);
            } else if inner.input.peek(syn::Token![=]) {
                payload = CallPayload::Single(inner.value()?.parse()?);
            } else {
                payload = CallPayload::AllFields;
            }
            return Ok(());
        }
        match inner.path.get_ident() {
            Some(ident) if label.is_none() => {
                label = Some(ident.clone());
                Ok(())
            }
            Some(_) => Err(inner.error(
                "expected a single label field name, then an optional \
                 `payload` / `payload(a, b)` / `payload = field` clause",
            )),
            None => Err(inner.error("expected a field name")),
        }
    })?;
    let label = label.ok_or_else(|| meta.error("expected a label field name argument"))?;
    Ok((label, payload))
}

/// Emits the NodeId expression for a single-node recursive field
/// (`T` or `Box<T>`); errors on optional / repeated multiplicity.
fn single_child_id(
    field: &(Ident, Recursion),
    variant: &syn::Variant,
    role: &str,
) -> syn::Result<TokenStream2> {
    let (ident, kind) = field;
    match kind {
        Recursion::Direct => Ok(quote! { ::dsl_kit_core::DslNode::node_id(#ident) }),
        Recursion::Boxed => Ok(quote! { ::dsl_kit_core::DslNode::node_id(&**#ident) }),
        _ => Err(syn::Error::new_spanned(
            variant,
            format!("#[dsl_exec] {role} field `{ident}` must be `Self` or `Box<Self>`"),
        )),
    }
}

/// Derives `dsl_kit_core::DslExec`: the mechanical half of an engine
/// `Ast`. Every variant carries a `#[dsl_exec(...)]` form naming its
/// engine `NodeKind`; recursive child fields are picked up in
/// declaration order exactly like `#[derive(DslNode)]` does. Pair the
/// generated impl with a `DslSemantics` implementation via
/// `dsl_kit_core::DerivedAst` to obtain a runnable `Ast`.
///
/// # Effect payloads
///
/// `call(LABEL)` suspends with a `null` payload — the label is the whole
/// message. When the effect takes arguments, name them so they reach the
/// host verbatim through `CallSpec::payload` and the resolver never has
/// to look the node up in the DSL's own state:
///
/// | form | payload |
/// |---|---|
/// | `call(label)` | `null` |
/// | `call(label, payload)` | every non-recursive field except `label`, as an object |
/// | `call(label, payload(src, dst))` | those fields, as an object |
/// | `call(label, payload = args)` | `args` alone, unwrapped |
///
/// Every field feeding a payload must implement `serde::Serialize`.
///
/// ```ignore
/// #[derive(DslNode, DslExec)]
/// enum Flow {
///     #[dsl_exec(call(label, payload(src, dst)))]
///     Transfer { id: NodeId, label: String, src: String, dst: String },
/// }
/// // resolver side: spec.payload["src"] / spec.payload["dst"]
/// ```
#[proc_macro_derive(DslExec, attributes(dsl_exec))]
pub fn derive_dsl_exec(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(&input, "#[derive(DslExec)] currently supports enums only")
            .to_compile_error()
            .into();
    };

    let mut kind_arms = Vec::new();
    let mut literal_arms = Vec::new();
    let mut lit_ty: Option<(String, Type)> = None;

    for variant in &data.variants {
        let variant_ident = &variant.ident;

        let Fields::Named(fields) = &variant.fields else {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslExec)] requires every variant to use named fields",
            )
            .to_compile_error()
            .into();
        };

        let spec = match dsl_exec_attr(variant) {
            Ok(s) => s,
            Err(e) => return e.to_compile_error().into(),
        };

        // Recursive fields in declaration order, payload fields besides.
        let mut recursive: Vec<(Ident, Recursion)> = Vec::new();
        let mut payload: Vec<(Ident, Type)> = Vec::new();
        for f in &fields.named {
            let Some(ident) = &f.ident else { continue };
            if ident == "id" {
                continue;
            }
            if let Some(kind) = detect_recursion(&f.ty, &name) {
                recursive.push((ident.clone(), kind));
            } else {
                payload.push((ident.clone(), f.ty.clone()));
            }
        }

        let err = |msg: String| {
            syn::Error::new_spanned(variant, msg)
                .to_compile_error()
                .into()
        };

        // `children`-style collection over every recursive field.
        let collect_children = |recursive: &[(Ident, Recursion)]| {
            let push_stmts = recursive.iter().map(|(ident, kind)| match kind {
                Recursion::Direct => quote! {
                    _children.push(::dsl_kit_core::DslNode::node_id(#ident));
                },
                Recursion::Boxed => quote! {
                    _children.push(::dsl_kit_core::DslNode::node_id(&**#ident));
                },
                Recursion::Optional => quote! {
                    if let ::std::option::Option::Some(inner) = #ident.as_ref() {
                        _children.push(::dsl_kit_core::DslNode::node_id(inner));
                    }
                },
                Recursion::OptionalBoxed => quote! {
                    if let ::std::option::Option::Some(inner) = #ident.as_deref() {
                        _children.push(::dsl_kit_core::DslNode::node_id(inner));
                    }
                },
                Recursion::Many => quote! {
                    _children.extend(#ident.iter().map(::dsl_kit_core::DslNode::node_id));
                },
                Recursion::ManyBoxed => quote! {
                    _children.extend(
                        #ident.iter().map(|c| ::dsl_kit_core::DslNode::node_id(c.as_ref())),
                    );
                },
                // Keyed slots iterate in the map's own order
                // (`BTreeMap` sorts by key) so the child sequence the
                // engine sees is deterministic. Semantic choices
                // (join policy, per-key dispatch, …) belong on the
                // engine side; the derive only surfaces the children.
                Recursion::Map => quote! {
                    _children.extend(#ident.values().map(::dsl_kit_core::DslNode::node_id));
                },
                Recursion::MapBoxed => quote! {
                    _children.extend(
                        #ident.values().map(|c| ::dsl_kit_core::DslNode::node_id(c.as_ref())),
                    );
                },
            });
            quote! {
                let mut _children: ::std::vec::Vec<::dsl_kit_core::NodeId> =
                    ::std::vec::Vec::new();
                #(#push_stmts)*
            }
        };

        let binds = recursive.iter().map(|(id, _)| quote!(#id));
        let binds = quote! { #(#binds,)* };

        match spec {
            ExecSpec::Value => {
                if !recursive.is_empty() {
                    return err("#[dsl_exec(value)] variants must have no child fields".into());
                }
                let [(field, ty)] = payload.as_slice() else {
                    return err(
                        "#[dsl_exec(value)] variants must have exactly one payload field".into(),
                    );
                };
                let ty_str = ty.to_token_stream().to_string();
                match &lit_ty {
                    None => lit_ty = Some((ty_str, ty.clone())),
                    Some((seen, _)) if *seen == ty_str => {}
                    Some((seen, _)) => {
                        return err(format!(
                            "#[dsl_exec(value)] fields must share one type; \
                             saw `{seen}` and `{ty_str}`"
                        ));
                    }
                }
                kind_arms.push(quote! {
                    Self::#variant_ident { .. } => ::dsl_kit_core::NodeKind::Lit,
                });
                literal_arms.push(quote! {
                    Self::#variant_ident { #field, .. } =>
                        ::std::option::Option::Some(#field.clone()),
                });
            }
            ExecSpec::Read { name_field } => {
                kind_arms.push(quote! {
                    Self::#variant_ident { #name_field, .. } =>
                        ::dsl_kit_core::NodeKind::Read { name: #name_field.clone() },
                });
            }
            ExecSpec::Apply { op } => {
                let collect = collect_children(&recursive);
                kind_arms.push(quote! {
                    Self::#variant_ident { #binds .. } => {
                        #collect
                        ::dsl_kit_core::NodeKind::Apply {
                            op_id: ::dsl_kit_core::OpId::from(#op),
                            children: _children,
                        }
                    }
                });
            }
            ExecSpec::Bind { name_field } => {
                let [value, body] = recursive.as_slice() else {
                    return err(
                        "#[dsl_exec(bind(..))] variants must have exactly two child fields \
                         (value, then body)"
                            .into(),
                    );
                };
                let value_id = match single_child_id(value, variant, "value") {
                    Ok(t) => t,
                    Err(e) => return e.to_compile_error().into(),
                };
                let body_id = match single_child_id(body, variant, "body") {
                    Ok(t) => t,
                    Err(e) => return e.to_compile_error().into(),
                };
                kind_arms.push(quote! {
                    Self::#variant_ident { #name_field, #binds .. } =>
                        ::dsl_kit_core::NodeKind::Bind {
                            name: #name_field.clone(),
                            value: #value_id,
                            body: #body_id,
                        },
                });
            }
            ExecSpec::Branch => {
                if recursive.len() < 2 || recursive.len() > 3 {
                    return err(
                        "#[dsl_exec(branch)] variants must have two or three child fields \
                         (cond, then, optional else)"
                            .into(),
                    );
                }
                let cond_id = match single_child_id(&recursive[0], variant, "cond") {
                    Ok(t) => t,
                    Err(e) => return e.to_compile_error().into(),
                };
                let then_id = match single_child_id(&recursive[1], variant, "then") {
                    Ok(t) => t,
                    Err(e) => return e.to_compile_error().into(),
                };
                let else_expr = match recursive.get(2) {
                    None => quote! { ::std::option::Option::None },
                    Some((ident, Recursion::Direct)) => quote! {
                        ::std::option::Option::Some(::dsl_kit_core::DslNode::node_id(#ident))
                    },
                    Some((ident, Recursion::Boxed)) => quote! {
                        ::std::option::Option::Some(::dsl_kit_core::DslNode::node_id(&**#ident))
                    },
                    Some((ident, Recursion::Optional)) => quote! {
                        #ident.as_ref().map(::dsl_kit_core::DslNode::node_id)
                    },
                    Some((ident, Recursion::OptionalBoxed)) => quote! {
                        #ident.as_deref().map(::dsl_kit_core::DslNode::node_id)
                    },
                    Some(_) => {
                        return err(
                            "#[dsl_exec(branch)] else field must be a single or optional child"
                                .into(),
                        );
                    }
                };
                kind_arms.push(quote! {
                    Self::#variant_ident { #binds .. } =>
                        ::dsl_kit_core::NodeKind::Branch {
                            cond: #cond_id,
                            then_branch: #then_id,
                            else_branch: #else_expr,
                        },
                });
            }
            ExecSpec::Repeat => {
                let [body] = recursive.as_slice() else {
                    return err(
                        "#[dsl_exec(repeat)] variants must have exactly one child field".into(),
                    );
                };
                let body_id = match single_child_id(body, variant, "body") {
                    Ok(t) => t,
                    Err(e) => return e.to_compile_error().into(),
                };
                kind_arms.push(quote! {
                    Self::#variant_ident { #binds .. } =>
                        ::dsl_kit_core::NodeKind::Loop { body: #body_id },
                });
            }
            ExecSpec::Seq => {
                let collect = collect_children(&recursive);
                kind_arms.push(quote! {
                    Self::#variant_ident { #binds .. } => {
                        #collect
                        ::dsl_kit_core::NodeKind::Seq { children: _children }
                    }
                });
            }
            ExecSpec::Scope { label_field } => {
                let [body] = recursive.as_slice() else {
                    return err(
                        "#[dsl_exec(scope(..))] variants must have exactly one child field".into(),
                    );
                };
                let body_id = match single_child_id(body, variant, "body") {
                    Ok(t) => t,
                    Err(e) => return e.to_compile_error().into(),
                };
                kind_arms.push(quote! {
                    Self::#variant_ident { #label_field, #binds .. } =>
                        ::dsl_kit_core::NodeKind::Scope {
                            label: #label_field.clone(),
                            body: #body_id,
                        },
                });
            }
            ExecSpec::Maybe => {
                let [(ident, kind)] = recursive.as_slice() else {
                    return err(
                        "#[dsl_exec(maybe)] variants must have exactly one child field".into(),
                    );
                };
                let body_expr = match kind {
                    Recursion::Optional => quote! {
                        #ident.as_ref().map(::dsl_kit_core::DslNode::node_id)
                    },
                    Recursion::OptionalBoxed => quote! {
                        #ident.as_deref().map(::dsl_kit_core::DslNode::node_id)
                    },
                    _ => {
                        return err("#[dsl_exec(maybe)] child field must be `Option<Self>` or \
                             `Option<Box<Self>>`"
                            .into());
                    }
                };
                kind_arms.push(quote! {
                    Self::#variant_ident { #binds .. } =>
                        ::dsl_kit_core::NodeKind::Maybe { body: #body_expr },
                });
            }
            ExecSpec::Call {
                label_field,
                payload: payload_form,
            } => {
                if !recursive.is_empty() {
                    return err("#[dsl_exec(call(..))] variants must have no child fields".into());
                }
                // Which fields feed the payload, and how they are shaped.
                let known: Vec<&Ident> = payload.iter().map(|(id, _)| id).collect();
                let selected: Vec<Ident> = match &payload_form {
                    CallPayload::None => Vec::new(),
                    CallPayload::AllFields => known
                        .iter()
                        .filter(|id| **id != &label_field)
                        .map(|id| (*id).clone())
                        .collect(),
                    CallPayload::Fields(fields) => fields.clone(),
                    CallPayload::Single(field) => vec![field.clone()],
                };
                if let Some(unknown) = selected.iter().find(|f| !known.contains(f)) {
                    return err(format!(
                        "#[dsl_exec(call(..))] payload field `{unknown}` is not a \
                         non-recursive field of this variant"
                    ));
                }
                // The label field is already bound by the pattern.
                let extra_binds = selected.iter().filter(|f| **f != label_field);
                // A serialisation failure is reported *in* the payload
                // rather than silently dropped — losing an effect's
                // arguments without a trace is the failure mode this
                // whole channel exists to avoid.
                let or_error = quote! {
                    let _value = |
                        r: ::std::result::Result<
                            ::dsl_kit_core::serde_json::Value,
                            ::dsl_kit_core::serde_json::Error,
                        >,
                    | -> ::dsl_kit_core::serde_json::Value {
                        match r {
                            ::std::result::Result::Ok(v) => v,
                            ::std::result::Result::Err(e) => {
                                let mut _err = ::dsl_kit_core::serde_json::Map::new();
                                _err.insert(
                                    ::std::string::String::from("__payload_error"),
                                    ::dsl_kit_core::serde_json::Value::String(
                                        ::std::string::ToString::to_string(&e),
                                    ),
                                );
                                ::dsl_kit_core::serde_json::Value::Object(_err)
                            }
                        }
                    };
                };
                let payload_expr = match &payload_form {
                    CallPayload::None => quote! { ::dsl_kit_core::serde_json::Value::Null },
                    CallPayload::Single(field) => quote! {{
                        #or_error
                        _value(::dsl_kit_core::serde_json::to_value(#field))
                    }},
                    _ => {
                        let inserts = selected.iter().map(|f| {
                            quote! {
                                _payload.insert(
                                    ::std::string::String::from(::std::stringify!(#f)),
                                    _value(::dsl_kit_core::serde_json::to_value(#f)),
                                );
                            }
                        });
                        quote! {{
                            #or_error
                            let mut _payload = ::dsl_kit_core::serde_json::Map::new();
                            #(#inserts)*
                            ::dsl_kit_core::serde_json::Value::Object(_payload)
                        }}
                    }
                };
                kind_arms.push(quote! {
                    Self::#variant_ident { #label_field, #(#extra_binds,)* .. } =>
                        ::dsl_kit_core::NodeKind::Call {
                            label: #label_field.clone(),
                            payload: #payload_expr,
                        },
                });
            }
        }
    }

    let lit_ty_tokens = match &lit_ty {
        Some((_, ty)) => quote!(#ty),
        None => quote!(()),
    };
    let literal_body = if literal_arms.is_empty() {
        quote! { ::std::option::Option::None }
    } else {
        quote! {
            match self {
                #(#literal_arms)*
                _ => ::std::option::Option::None,
            }
        }
    };

    let expanded: TokenStream2 = quote! {
        impl #impl_generics ::dsl_kit_core::DslExec for #name #ty_generics #where_clause {
            type LitValue = #lit_ty_tokens;

            fn exec_kind(&self) -> ::dsl_kit_core::NodeKind {
                match self {
                    #(#kind_arms)*
                }
            }

            fn exec_literal(&self) -> ::std::option::Option<Self::LitValue> {
                #literal_body
            }
        }
    };

    expanded.into()
}

// ---------------------------------------------------------------------------
// DslCheck
// ---------------------------------------------------------------------------

/// One argument term inside a `#[dsl_check(...)]` judgement.
///
/// The vocabulary is the Check IR's own term algebra minus
/// `Term::FieldRef`, which has no surface syntax of its own: a payload
/// field reaches a judgement by binding a `$var` with `bind(var =
/// "field")`, so the wiring is declared once per variant instead of
/// being spelled inside every fact.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CheckTermDecl {
    /// Ground constant — `SystemReady`, `Int`.
    Atom(String),
    /// Rule variable — `$name`.
    Var(String),
    /// Constructor application — `ServiceRunning($name)`.
    Ctor(String, Vec<CheckTermDecl>),
}

impl CheckTermDecl {
    /// Appends every `$var` name the term mentions, in order.
    fn collect_vars(&self, out: &mut Vec<String>) {
        match self {
            CheckTermDecl::Atom(_) => {}
            CheckTermDecl::Var(name) => out.push(name.clone()),
            CheckTermDecl::Ctor(_, args) => {
                for arg in args {
                    arg.collect_vars(out);
                }
            }
        }
    }
}

/// A `pred(term, …)` judgement parsed out of a `#[dsl_check(...)]`
/// string literal.
///
/// The predicate is any identifier (`state`, `type`, `cap`, whatever
/// the DSL author invents) and the arguments are terms: ground atoms,
/// rule variables (`$name`), and constructor applications
/// (`ServiceRunning($name)`), nested arbitrarily.
#[derive(Debug)]
struct CheckFactDecl {
    /// Predicate name.
    pred: String,
    /// Argument terms, in order.
    args: Vec<CheckTermDecl>,
}

impl CheckFactDecl {
    /// Appends every `$var` name the fact mentions, in order.
    fn collect_vars(&self, out: &mut Vec<String>) {
        for arg in &self.args {
            arg.collect_vars(out);
        }
    }
}

/// One `bind(var = "field")` entry: the payload field a `$var` reads
/// its value from at check time.
#[derive(Debug, Clone)]
struct CheckBind {
    /// Variable name, without the `$` sigil.
    var: String,
    /// Payload field on the annotated variant.
    field: String,
    /// The `"field"` literal, kept for error spans.
    lit: syn::LitStr,
}

/// One parsed premise of a variant's rule.
#[derive(Debug)]
enum CheckPremiseDecl {
    /// `requires = "pred(…)"` — the running fold state must match.
    State(CheckFactDecl),
    /// `requires(slot = "pred(…)")` — the conclusion of every child in
    /// `slot` must match.
    Child {
        /// Child slot name (an ident, kept for error spans).
        slot: Ident,
        /// Pattern the child's conclusion must match.
        expect: CheckFactDecl,
    },
}

/// Emits the `dsl_kit_check::Term` literal an argument stands for.
///
/// `binds` is the variant's `bind(var = "field")` table: a `$var` it
/// names becomes a `Term::FieldRef` — the solver resolves it against
/// the node's payload before unifying — and every other `$var` stays a
/// rule-local `Term::Var`.
fn emit_check_term(decl: &CheckTermDecl, binds: &[CheckBind]) -> TokenStream2 {
    match decl {
        CheckTermDecl::Atom(name) => quote! { ::dsl_kit_check::Term::Atom(#name.to_string()) },
        CheckTermDecl::Var(name) => match binds.iter().find(|b| &b.var == name) {
            Some(bind) => {
                let field = &bind.field;
                quote! { ::dsl_kit_check::Term::FieldRef(#field.to_string()) }
            }
            None => quote! { ::dsl_kit_check::Term::Var(#name.to_string()) },
        },
        CheckTermDecl::Ctor(name, args) => {
            let args = args.iter().map(|a| emit_check_term(a, binds));
            quote! {
                ::dsl_kit_check::Term::Ctor(#name.to_string(), ::std::vec![#(#args),*])
            }
        }
    }
}

/// Emits the `dsl_kit_check::Fact` literal a declaration stands for.
fn emit_check_fact(decl: &CheckFactDecl, binds: &[CheckBind]) -> TokenStream2 {
    let pred = &decl.pred;
    let args = decl.args.iter().map(|a| emit_check_term(a, binds));
    quote! {
        ::dsl_kit_check::Fact {
            pred: #pred.to_string(),
            args: ::std::vec![#(#args),*],
        }
    }
}

/// Recursive-descent cursor over the text of one `#[dsl_check(...)]`
/// judgement literal.
///
/// Nested constructors (`state(ServiceRunning($name))`) rule out the
/// split-on-comma shortcut, so the grammar is spelled out:
///
/// ```text
/// fact := ident [ "(" args ")" ]
/// args := term { "," term }
/// term := "$" ident | ident [ "(" args ")" ]
/// ```
///
/// Every error is spanned to the string literal, so the caret lands on
/// the annotation rather than on the whole variant.
struct CheckFactParser<'a> {
    text: &'a str,
    pos: usize,
    lit: &'a syn::LitStr,
}

impl<'a> CheckFactParser<'a> {
    fn new(text: &'a str, lit: &'a syn::LitStr) -> Self {
        Self { text, pos: 0, lit }
    }

    fn error(&self, message: String) -> syn::Error {
        syn::Error::new_spanned(self.lit, message)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    fn bump(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    /// Reads one Rust-style identifier, or fails naming `role`.
    fn ident(&mut self, role: &str) -> syn::Result<String> {
        let start = self.pos;
        if matches!(self.peek(), Some(c) if c.is_ascii_alphabetic() || c == '_') {
            self.bump();
            while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
                self.bump();
            }
        }
        if start == self.pos {
            let rest = self.text[start..].trim_end();
            return Err(self.error(format!(
                "invalid {role} in `{}`{}; expected an identifier such as `state` or \
                 `SystemReady`",
                self.text,
                if rest.is_empty() {
                    String::new()
                } else {
                    format!(" at `{rest}`")
                }
            )));
        }
        Ok(self.text[start..self.pos].to_string())
    }

    /// Reads `( term, … )`, cursor sitting on the opening parenthesis.
    fn args(&mut self) -> syn::Result<Vec<CheckTermDecl>> {
        self.bump(); // `(`
        let mut args = Vec::new();
        self.skip_ws();
        if self.peek() == Some(')') {
            self.bump();
            return Ok(args);
        }
        loop {
            args.push(self.term()?);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some(')') => {
                    self.bump();
                    return Ok(args);
                }
                Some(_) => {
                    return Err(self.error(format!(
                        "invalid argument in `{}`; expected `,` or `)` after an argument",
                        self.text
                    )));
                }
                None => {
                    return Err(self.error(format!(
                        "unbalanced parentheses in `{}`; expected `pred(term, …)`",
                        self.text
                    )));
                }
            }
        }
    }

    fn term(&mut self) -> syn::Result<CheckTermDecl> {
        self.skip_ws();
        if self.peek() == Some('$') {
            self.bump();
            return Ok(CheckTermDecl::Var(self.ident("rule variable")?));
        }
        let name = self.ident("argument")?;
        self.skip_ws();
        if self.peek() == Some('(') {
            return Ok(CheckTermDecl::Ctor(name, self.args()?));
        }
        Ok(CheckTermDecl::Atom(name))
    }

    fn fact(&mut self) -> syn::Result<CheckFactDecl> {
        self.skip_ws();
        let pred = self.ident("predicate name")?;
        self.skip_ws();
        let args = if self.peek() == Some('(') {
            self.args()?
        } else {
            Vec::new()
        };
        self.skip_ws();
        if self.pos != self.text.len() {
            return Err(self.error(format!(
                "trailing text `{}` in `{}`; a judgement is one `pred(term, …)`",
                &self.text[self.pos..],
                self.text
            )));
        }
        Ok(CheckFactDecl { pred, args })
    }
}

/// Parses `"pred(term, …)"` / `"pred"` out of a `#[dsl_check(...)]`
/// string literal.
fn parse_check_fact(lit: &syn::LitStr) -> syn::Result<CheckFactDecl> {
    let raw = lit.value();
    CheckFactParser::new(raw.trim(), lit).fact()
}

/// [`parse_check_fact`] for positions that admit no variables — a
/// fold's initial state, which is the seed of the sequence and has
/// nothing to bind against.
fn parse_check_fact_ground(lit: &syn::LitStr) -> syn::Result<CheckFactDecl> {
    let decl = parse_check_fact(lit)?;
    let mut vars = Vec::new();
    decl.collect_vars(&mut vars);
    if let Some(var) = vars.first() {
        return Err(syn::Error::new_spanned(
            lit,
            format!(
                "rule variable `${var}` cannot appear in a fold's initial state: the seed is \
                 supplied by the declaration, before any child has bound anything"
            ),
        ));
    }
    Ok(decl)
}

/// Parsed `#[dsl_check(...)]` annotations on one variant.
#[derive(Debug, Default)]
struct DslCheckAttrs {
    /// Premises in declaration order — `requires = "pred(…)"` (state)
    /// and `requires(slot = "pred(…)")` (child slot).
    premises: Vec<CheckPremiseDecl>,
    /// `produces = "pred(…)"` — becomes `Rule::state_after`.
    produces: Option<CheckFactDecl>,
    /// `concludes = "pred(…)"` — becomes `Rule::conclusion`, the
    /// synthesised attribute a parent's child premise reads.
    concludes: Option<CheckFactDecl>,
    /// `bind(var = "field")` entries wiring `$var` to payload fields.
    binds: Vec<CheckBind>,
    /// `message = "…"` — overrides the generated wording.
    message: Option<String>,
    /// `code = "…"` — overrides the default diagnostic slug.
    code: Option<syn::LitStr>,
    /// Whether the variant carried the attribute at all. A variant
    /// without it contributes no rule (the check layer is opt-in per
    /// variant, exactly as the solver's "no rule = pass through").
    seen: bool,
}

impl DslCheckAttrs {
    /// Whether a `requires = "pred(…)"` (state) premise is present.
    fn has_state_premise(&self) -> bool {
        self.premises
            .iter()
            .any(|p| matches!(p, CheckPremiseDecl::State(_)))
    }

    /// Every `$var` the variant's judgements mention, deduplicated.
    fn mentioned_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for premise in &self.premises {
            match premise {
                CheckPremiseDecl::State(fact) => fact.collect_vars(&mut vars),
                CheckPremiseDecl::Child { expect, .. } => expect.collect_vars(&mut vars),
            }
        }
        for fact in [self.produces.as_ref(), self.concludes.as_ref()]
            .into_iter()
            .flatten()
        {
            fact.collect_vars(&mut vars);
        }
        vars.dedup();
        vars
    }
}

/// Parses a variant's `#[dsl_check(...)]` annotations (`requires` /
/// `produces` / `concludes` / `bind` / `message` / `code`).
/// `Ok(Default::default())` when the variant carries none.
///
/// Unknown keys are a compile error rather than a silent skip — the
/// same discipline `dsl_schema_attrs` follows, and the reason the
/// whole vocabulary check lives in one function.
fn dsl_check_attrs(variant: &syn::Variant) -> syn::Result<DslCheckAttrs> {
    let mut out = DslCheckAttrs::default();
    for attr in &variant.attrs {
        if !attr.path().is_ident("dsl_check") {
            continue;
        }
        out.seen = true;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("requires") {
                // Two shapes on one key: `requires = "state(A)"` is the
                // running fold state, `requires(cond = "type(Bool)")`
                // names child slots. The `=` decides which.
                if meta.input.peek(syn::Token![=]) {
                    if out.has_state_premise() {
                        return Err(meta.error("duplicate `requires` on this variant"));
                    }
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    out.premises
                        .push(CheckPremiseDecl::State(parse_check_fact(&lit)?));
                    return Ok(());
                }
                let mut any = false;
                meta.parse_nested_meta(|inner| {
                    let Some(slot) = inner.path.get_ident().cloned() else {
                        return Err(inner.error(
                            "expected a child slot name, as in `requires(cond = \"type(Bool)\")`",
                        ));
                    };
                    if out
                        .premises
                        .iter()
                        .any(|p| matches!(p, CheckPremiseDecl::Child { slot: s, .. } if *s == slot))
                    {
                        return Err(inner
                            .error(format!("duplicate child slot `{slot}` in `requires(...)`")));
                    }
                    let lit: syn::LitStr = inner.value()?.parse()?;
                    let expect = parse_check_fact(&lit)?;
                    out.premises.push(CheckPremiseDecl::Child { slot, expect });
                    any = true;
                    Ok(())
                })?;
                if !any {
                    return Err(meta.error(
                        "`requires(...)` needs at least one `slot = \"pred(term, …)\"` entry",
                    ));
                }
            } else if meta.path.is_ident("produces") {
                if out.produces.is_some() {
                    return Err(meta.error("duplicate `produces` on this variant"));
                }
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.produces = Some(parse_check_fact(&lit)?);
            } else if meta.path.is_ident("concludes") {
                if out.concludes.is_some() {
                    return Err(meta.error("duplicate `concludes` on this variant"));
                }
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.concludes = Some(parse_check_fact(&lit)?);
            } else if meta.path.is_ident("bind") {
                let mut any = false;
                meta.parse_nested_meta(|inner| {
                    let Some(var) = inner.path.get_ident().cloned() else {
                        return Err(inner.error(
                            "expected a rule variable name, as in `bind(name = \"name\")`",
                        ));
                    };
                    let var = var.to_string();
                    if out.binds.iter().any(|b| b.var == var) {
                        return Err(inner.error(format!("duplicate `bind({var} = …)`")));
                    }
                    let lit: syn::LitStr = inner.value()?.parse()?;
                    out.binds.push(CheckBind {
                        var,
                        field: lit.value(),
                        lit,
                    });
                    any = true;
                    Ok(())
                })?;
                if !any {
                    return Err(
                        meta.error("`bind(...)` needs at least one `var = \"field\"` entry")
                    );
                }
            } else if meta.path.is_ident("message") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.message = Some(lit.value());
            } else if meta.path.is_ident("code") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.code = Some(lit);
            } else {
                return Err(meta.error(
                    "unsupported #[dsl_check(...)] key; expected `requires = \"pred(term)\"`, \
                     `requires(slot = \"pred(term)\")`, `produces = \"pred(term)\"`, \
                     `concludes = \"pred(term)\"`, `bind(var = \"field\")`, `message = \"…\"`, \
                     or `code = \"…\"`",
                ));
            }
            Ok(())
        })?;
    }
    if out.seen && out.premises.is_empty() && out.produces.is_none() && out.concludes.is_none() {
        return Err(syn::Error::new_spanned(
            variant,
            "#[dsl_check(...)] needs at least one of `requires` / `produces` / `concludes` — a \
             rule with neither a premise nor a conclusion says nothing",
        ));
    }
    Ok(out)
}

/// Cross-checks a variant's parsed annotations against its shape:
/// every `bind(var = "field")` must name a payload field the variant
/// declares, every `requires(slot = …)` must name one of its child
/// slots, and every bound `$var` must actually appear in a judgement.
///
/// Kept separate from [`dsl_check_attrs`] because it needs the enum
/// name to tell a child slot from a payload field — the same
/// `detect_recursion` classification the other derives run on.
fn validate_check_attrs(
    variant: &syn::Variant,
    attrs: &DslCheckAttrs,
    enum_name: &Ident,
) -> syn::Result<()> {
    if attrs.binds.is_empty()
        && !attrs
            .premises
            .iter()
            .any(|p| matches!(p, CheckPremiseDecl::Child { .. }))
    {
        return Ok(());
    }

    let Fields::Named(fields) = &variant.fields else {
        return Err(syn::Error::new_spanned(
            variant,
            "`bind(...)` and `requires(slot = …)` name fields, so they apply to named-field \
             variants only",
        ));
    };

    for bind in &attrs.binds {
        let field = fields
            .named
            .iter()
            .find(|f| f.ident.as_ref().is_some_and(|i| *i == bind.field));
        let Some(field) = field else {
            return Err(syn::Error::new_spanned(
                &bind.lit,
                format!(
                    "`bind({} = \"{}\")` names a field variant `{}` does not declare",
                    bind.var, bind.field, variant.ident
                ),
            ));
        };
        if detect_recursion(&field.ty, enum_name).is_some() {
            return Err(syn::Error::new_spanned(
                &bind.lit,
                format!(
                    "`bind({} = \"{}\")` names a child slot, not a payload field — a `$var` \
                     reads a scalar value, and a child's judgement reaches the rule through \
                     `requires({} = \"…\")` instead",
                    bind.var, bind.field, bind.field
                ),
            ));
        }
    }

    for premise in &attrs.premises {
        let CheckPremiseDecl::Child { slot, .. } = premise else {
            continue;
        };
        let field = fields
            .named
            .iter()
            .find(|f| f.ident.as_ref().is_some_and(|i| i == slot));
        let Some(field) = field else {
            return Err(syn::Error::new_spanned(
                slot,
                format!(
                    "`requires({slot} = …)` names a slot variant `{}` does not declare",
                    variant.ident
                ),
            ));
        };
        if detect_recursion(&field.ty, enum_name).is_none() {
            return Err(syn::Error::new_spanned(
                slot,
                format!(
                    "`requires({slot} = …)` names a payload field, not a child slot — only a \
                     child node carries a conclusion to match against"
                ),
            ));
        }
    }

    let mentioned = attrs.mentioned_vars();
    for bind in &attrs.binds {
        if !mentioned.contains(&bind.var) {
            return Err(syn::Error::new_spanned(
                &bind.lit,
                format!(
                    "`bind({} = \"{}\")` binds `${}`, which no judgement on this variant \
                     mentions",
                    bind.var, bind.field, bind.var
                ),
            ));
        }
    }

    Ok(())
}

/// Parses a child slot's `#[dsl_check(fold(initial = "pred(Atom)"))]`
/// annotation. `Ok(None)` when the field carries no `dsl_check`
/// attribute.
fn dsl_check_fold_attr(f: &syn::Field) -> syn::Result<Option<CheckFactDecl>> {
    let mut initial: Option<CheckFactDecl> = None;
    for attr in &f.attrs {
        if !attr.path().is_ident("dsl_check") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("fold") {
                return Err(meta.error(
                    "unsupported #[dsl_check(...)] key on a child slot; expected \
                     `fold(initial = \"state(Atom)\")` (`requires` / `produces` belong on \
                     the variant)",
                ));
            }
            let mut got = false;
            meta.parse_nested_meta(|inner| {
                if !inner.path.is_ident("initial") {
                    return Err(inner.error(
                        "unsupported `fold(...)` key; expected `initial = \"state(Atom)\"`",
                    ));
                }
                let lit: syn::LitStr = inner.value()?.parse()?;
                initial = Some(parse_check_fact_ground(&lit)?);
                got = true;
                Ok(())
            })?;
            if !got {
                return Err(meta.error("`fold(...)` requires `initial = \"state(Atom)\"`"));
            }
            Ok(())
        })?;
    }
    Ok(initial)
}

/// Derives `dsl_kit_check::DslCheck` for the same enum shape accepted
/// by `DslNode` / `DslSchema` / `DslBuild`. The generated
/// `check_program()` method returns a `CheckProgram` — the DSL's
/// semantic judgement rules as data, evaluated against a `ParseTree` by
/// `dsl_kit_check::check_semantics`.
///
/// The derive is deliberately **opt-in and separate** from
/// `#[derive(DslSchema)]`: the emitted code names `::dsl_kit_check::…`
/// types, so a DSL that does not check semantics never acquires the
/// dependency.
///
/// ## Variant annotations
///
/// ```ignore
/// #[derive(DslNode, DslSchema, DslBuild, DslCheck)]
/// enum Phase {
///     Plan {
///         id: NodeId,
///         #[dsl_check(fold(initial = "state(Raw)"))]
///         steps: Vec<Phase>,
///     },
///
///     #[dsl_check(produces = "state(SystemReady)")]
///     SystemPkg { id: NodeId, packages: Vec<String> },
///
///     #[dsl_check(requires = "state(SystemReady)", produces = "state(PythonEnv)")]
///     PythonInstall { id: NodeId, version: String },
///
///     // A parameterised state: `$name` reads the `name` payload field.
///     #[dsl_check(
///         requires = "state(ComfyUIInstalled)",
///         produces = "state(ServiceRunning($name))",
///         bind(name = "name")
///     )]
///     ComfyUIService { id: NodeId, name: String },
///
///     #[dsl_check(requires = "state(ServiceRunning($target))", bind(target = "target"))]
///     Readiness { id: NodeId, target: String, port: u16 },
/// }
/// ```
///
/// - `requires = "pred(term, …)"` becomes a `Premise::State` — the
///   running fold state must unify with it before the variant may
///   appear.
/// - `requires(slot = "pred(term, …)")` becomes a `Premise::Child` per
///   named slot — the conclusion of every child in that slot must
///   unify with the pattern. This is the tree-typing half
///   (`requires(cond = "type(Bool)")`).
/// - `produces = "pred(term, …)"` becomes `Rule::state_after` — where
///   the variant leaves the fold.
/// - `concludes = "pred(term, …)"` becomes `Rule::conclusion` — the
///   synthesised attribute the parent's `requires(slot = …)` reads.
/// - `bind(var = "field")` wires `$var` to a payload field: every
///   occurrence of `$var` in that variant's judgements is emitted as a
///   `Term::FieldRef`, which the solver resolves against the node
///   before unifying. An unbound `$var` stays a rule-local
///   `Term::Var`, scoped to one attempt at one rule (the `$a` in
///   `requires(then_branch = "type($a)", else_branch = "type($a)")`).
/// - `message = "…"` overrides the generated wording (holes:
///   `{expected}` / `{found}` / `{provenance}` / `{slot}` / `{$var}`).
///   Omitted, the variant gets a default naming itself: "`Name`
///   requires {expected}, found {found} (from {provenance})".
/// - `code = "…"` overrides the diagnostic slug, which otherwise is
///   `CHECK_STATE_MISMATCH` for a rule that talks about state and
///   `CHECK_TYPE_MISMATCH` for one that only constrains children.
/// - A variant with no `#[dsl_check(...)]` contributes no rule and is
///   waved through by the solver: annotating is opt-in per variant.
///
/// Every name a judgement mentions is checked against the variant's
/// shape at derive time: `bind(...)` must name a payload field,
/// `requires(slot = …)` must name a child slot, and a `bind(...)` no
/// judgement uses is an error rather than dead wiring.
///
/// ## Fold slots
///
/// `#[dsl_check(fold(initial = "pred(Atom)"))]` on a `Vec<Self>` /
/// `Vec<Box<Self>>` field emits the `SeqSlotDecl` that makes the slot
/// ordered — its children thread a state from `initial` left to right.
/// The declaration lives on the field rather than in a stringly-typed
/// enum-level attribute so the `(variant, slot)` pair cannot drift from
/// the shape it names. Slots that are not annotated stay unordered
/// (`SeqMode::All`, the solver's default), and a program needing a
/// declaration the shape cannot express (a fold over another enum's
/// slot, say) can still be assembled with `CheckProgram::builder()`.
///
/// A fold's `initial` must be ground: it is the seed of the sequence,
/// supplied before any child has bound anything, so a `$var` there is
/// a compile error.
#[proc_macro_derive(DslCheck, attributes(dsl_check))]
pub fn derive_dsl_check(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    if let Some(attr) = input.attrs.iter().find(|a| a.path().is_ident("dsl_check")) {
        return syn::Error::new_spanned(
            attr,
            "#[dsl_check(...)] belongs on a variant (`requires` / `produces`) or on a \
             `Vec<Self>` child slot (`fold(initial = \"…\")`), not on the enum itself",
        )
        .to_compile_error()
        .into();
    }

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(
            &input,
            "#[derive(DslCheck)] currently supports enums only",
        )
        .to_compile_error()
        .into();
    };

    let mut rule_ctors: Vec<TokenStream2> = Vec::new();
    let mut seq_ctors: Vec<TokenStream2> = Vec::new();

    for variant in &data.variants {
        let variant_name = variant.ident.to_string();

        let attrs = match dsl_check_attrs(variant) {
            Ok(attrs) => attrs,
            Err(e) => return e.to_compile_error().into(),
        };
        if let Err(e) = validate_check_attrs(variant, &attrs, &name) {
            return e.to_compile_error().into();
        }

        if attrs.seen {
            let binds = &attrs.binds;
            let premise_ctors = attrs.premises.iter().map(|premise| match premise {
                CheckPremiseDecl::State(decl) => {
                    let expect = emit_check_fact(decl, binds);
                    quote! { ::dsl_kit_check::Premise::State { expect: #expect } }
                }
                CheckPremiseDecl::Child { slot, expect } => {
                    let slot = slot.to_string();
                    let expect = emit_check_fact(expect, binds);
                    quote! {
                        ::dsl_kit_check::Premise::Child {
                            slot: #slot.to_string(),
                            expect: #expect,
                        }
                    }
                }
            });
            let premises = quote! { ::std::vec![#(#premise_ctors),*] };
            let state_after = match &attrs.produces {
                Some(decl) => {
                    let fact = emit_check_fact(decl, binds);
                    quote! { ::std::option::Option::Some(#fact) }
                }
                None => quote! { ::std::option::Option::None },
            };
            let conclusion = match &attrs.concludes {
                Some(decl) => {
                    let fact = emit_check_fact(decl, binds);
                    quote! { ::std::option::Option::Some(#fact) }
                }
                None => quote! { ::std::option::Option::None },
            };
            let template = attrs.message.clone().unwrap_or_else(|| {
                format!(
                    "`{variant_name}` requires {{expected}}, found {{found}} (from {{provenance}})"
                )
            });
            // A rule that talks about the fold state reports under the
            // state slug; one that only constrains its children is a
            // typing judgement.
            let code = match &attrs.code {
                Some(lit) => quote! { #lit },
                None if attrs.has_state_premise() || attrs.produces.is_some() => {
                    quote! { ::dsl_kit_check::codes::CHECK_STATE_MISMATCH }
                }
                None => quote! { ::dsl_kit_check::codes::CHECK_TYPE_MISMATCH },
            };
            rule_ctors.push(quote! {
                ::dsl_kit_check::Rule {
                    variant: #variant_name.to_string(),
                    premises: #premises,
                    conclusion: #conclusion,
                    state_after: #state_after,
                    message: ::dsl_kit_check::MessageTemplate {
                        code: #code,
                        template: #template.to_string(),
                    },
                }
            });
        }

        match &variant.fields {
            Fields::Named(fields) => {
                for f in &fields.named {
                    let initial = match dsl_check_fold_attr(f) {
                        Ok(initial) => initial,
                        Err(e) => return e.to_compile_error().into(),
                    };
                    let (Some(initial), Some(ident)) = (initial, f.ident.as_ref()) else {
                        continue;
                    };
                    if !matches!(
                        detect_recursion(&f.ty, &name),
                        Some(Recursion::Many) | Some(Recursion::ManyBoxed)
                    ) {
                        return syn::Error::new_spanned(
                            f,
                            "#[dsl_check(fold(...))] applies to `Many` child slots only \
                             (`Vec<Self>` / `Vec<Box<Self>>`) — a fold threads its state \
                             through an ordered sequence, and every other slot is unordered",
                        )
                        .to_compile_error()
                        .into();
                    }
                    let slot = ident.to_string();
                    // A fold seed is ground by construction
                    // (`parse_check_fact_ground`), so it has no `$var`
                    // for a bind table to resolve.
                    let initial = emit_check_fact(&initial, &[]);
                    seq_ctors.push(quote! {
                        ::dsl_kit_check::SeqSlotDecl {
                            variant: #variant_name.to_string(),
                            slot: #slot.to_string(),
                            initial: #initial,
                            mode: ::dsl_kit_check::SeqMode::Fold,
                        }
                    });
                }
            }
            Fields::Unnamed(fields) => {
                let annotated = fields
                    .unnamed
                    .iter()
                    .find(|f| f.attrs.iter().any(|a| a.path().is_ident("dsl_check")));
                if let Some(f) = annotated {
                    return syn::Error::new_spanned(
                        f,
                        "#[dsl_check(...)] applies to named fields only",
                    )
                    .to_compile_error()
                    .into();
                }
            }
            Fields::Unit => {}
        }
    }

    let expanded: TokenStream2 = quote! {
        impl #impl_generics ::dsl_kit_check::DslCheck for #name #ty_generics #where_clause {
            fn check_program() -> ::dsl_kit_check::CheckProgram {
                ::dsl_kit_check::CheckProgram {
                    rules: ::std::vec![#(#rule_ctors),*],
                    seq_slots: ::std::vec![#(#seq_ctors),*],
                }
            }
        }
    };

    expanded.into()
}

/// Parses a field's `#[<attr_name>(with = path)]` annotation, if
/// present. `Ok(None)` when the field carries no such attribute.
///
/// Shared by the `dsl_build` and `dsl_dump` sides so the two parsers
/// cannot drift. A duplicated `with` — whether inside one attribute or
/// across repeated attributes — is rejected rather than silently
/// last-wins.
fn with_attr(f: &syn::Field, attr_name: &str) -> syn::Result<Option<syn::Path>> {
    let mut with: Option<syn::Path> = None;
    for attr in &f.attrs {
        if !attr.path().is_ident(attr_name) {
            continue;
        }
        let mut seen_here = false;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("with") {
                if with.is_some() {
                    return Err(meta.error(format!(
                        "duplicate #[{attr_name}(with = ...)] — a field takes one converter"
                    )));
                }
                with = Some(meta.value()?.parse::<syn::Path>()?);
                seen_here = true;
                Ok(())
            } else {
                Err(meta.error(format!(
                    "unsupported #[{attr_name}(...)] key; expected `with = <path>`"
                )))
            }
        })?;
        if !seen_here {
            return Err(syn::Error::new_spanned(
                attr,
                format!("#[{attr_name}] requires `with = <path>`"),
            ));
        }
    }
    Ok(with)
}

/// Parses a field's `#[dsl_dump(with = path)]` annotation, if present.
/// `Ok(None)` when the field carries no `dsl_dump` attribute.
fn dsl_dump_with_attr(f: &syn::Field) -> syn::Result<Option<syn::Path>> {
    with_attr(f, "dsl_dump")
}

/// Parses a field's `#[dsl_build(with = path)]` annotation, if present.
/// `Ok(None)` when the field carries no `dsl_build` attribute.
fn dsl_build_with_attr(f: &syn::Field) -> syn::Result<Option<syn::Path>> {
    with_attr(f, "dsl_build")
}

/// Collapses the whitespace that `TokenStream::to_string` inserts between
/// tokens so a type renders in its canonical, source-like form
/// (`Option < String >` → `Option<String>`). Pure string transform used
/// only for the human-facing `FieldSchema.ty` label; every consumer
/// (`BuildError` diagnostics, schema JSON export, generated docs) inherits
/// the tidied spelling.
///
/// The stringified token stream is a sequence of single-space-separated
/// tokens, so normalization reduces to deciding the separator between each
/// adjacent pair:
///
/// - no space before `<` `>` `,` `::` `(` `)`
/// - no space after `<` `::` `&` `(`
/// - exactly one space after `,` (the default separator)
///
/// This yields `HashMap<String, u32>`, `std::string::String`, `&str`, and
/// `&'a str` (the lifetime/type gap is preserved because neither `'a` nor
/// the following token is a special punctuation token).
fn normalize_type_str(s: &str) -> String {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    let mut out = String::with_capacity(s.len());
    for (i, tok) in tokens.iter().enumerate() {
        if i > 0 {
            let prev = tokens[i - 1];
            let no_space = matches!(*tok, "<" | ">" | "," | "::" | "(" | ")")
                || matches!(prev, "<" | "::" | "&" | "(");
            if !no_space {
                out.push(' ');
            }
        }
        out.push_str(tok);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        CheckPremiseDecl, CheckTermDecl, DslCheckAttrs, dsl_check_attrs, dsl_check_fold_attr,
        normalize_type_str, validate_check_attrs,
    };

    /// The state premise (`requires = "…"`) of a parsed variant.
    fn state_premise(attrs: &DslCheckAttrs) -> Option<&super::CheckFactDecl> {
        attrs.premises.iter().find_map(|p| match p {
            CheckPremiseDecl::State(fact) => Some(fact),
            CheckPremiseDecl::Child { .. } => None,
        })
    }

    /// The child premises (`requires(slot = "…")`), as `(slot, fact)`.
    fn child_premises(attrs: &DslCheckAttrs) -> Vec<(String, &super::CheckFactDecl)> {
        attrs
            .premises
            .iter()
            .filter_map(|p| match p {
                CheckPremiseDecl::Child { slot, expect } => Some((slot.to_string(), expect)),
                CheckPremiseDecl::State(_) => None,
            })
            .collect()
    }

    fn enum_name() -> syn::Ident {
        syn::parse_str("Phase").expect("test enum name parses")
    }

    /// Parses one enum variant from source, attributes included.
    fn variant(src: &str) -> syn::Variant {
        syn::parse_str(src).expect("test variant parses")
    }

    /// Parses one named field by wrapping it in a throwaway variant —
    /// `syn::Field` has no `Parse` impl of its own.
    fn field(src: &str) -> syn::Field {
        let v = variant(&format!("Wrapper {{ {src} }}"));
        let syn::Fields::Named(named) = v.fields else {
            panic!("test field is named");
        };
        named.named.into_iter().next().expect("one field")
    }

    #[test]
    fn dsl_check_reads_the_stage_one_vocabulary() {
        let v = variant(
            "#[dsl_check(requires = \"state(SystemReady)\", produces = \"state(PythonEnv)\", \
             message = \"nope\", code = \"my::code\")] \
             PythonInstall { id: NodeId, version: String }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");

        assert!(attrs.seen);
        let requires = state_premise(&attrs).expect("requires parsed");
        assert_eq!(requires.pred, "state");
        assert_eq!(requires.args, [CheckTermDecl::Atom("SystemReady".into())]);
        let produces = attrs.produces.as_ref().expect("produces parsed");
        assert_eq!(produces.pred, "state");
        assert_eq!(produces.args, [CheckTermDecl::Atom("PythonEnv".into())]);
        assert_eq!(attrs.message.as_deref(), Some("nope"));
        assert_eq!(
            attrs.code.as_ref().map(|c| c.value()).as_deref(),
            Some("my::code")
        );
        validate_check_attrs(&v, &attrs, &enum_name()).expect("shape agrees");
    }

    #[test]
    fn a_variant_without_the_attribute_contributes_nothing() {
        let attrs = dsl_check_attrs(&variant("Plain { id: NodeId }")).expect("no annotation");
        assert!(!attrs.seen);
        assert!(attrs.premises.is_empty());
        assert!(attrs.produces.is_none());
        assert!(attrs.concludes.is_none());
    }

    #[test]
    fn an_unknown_dsl_check_key_is_a_compile_error() {
        // A typo must not be silently ignored: the whole point of
        // routing every key through one parser is that an unknown one
        // fails loudly at derive time.
        let v = variant("#[dsl_check(prodcues = \"state(Ready)\")] SystemPkg { id: NodeId }");
        let err = dsl_check_attrs(&v).expect_err("unknown key rejected");
        assert!(
            err.to_string()
                .contains("unsupported #[dsl_check(...)] key"),
            "message = {err}"
        );
    }

    #[test]
    fn a_bare_dsl_check_needs_a_judgement() {
        let v = variant("#[dsl_check(message = \"hi\")] SystemPkg { id: NodeId }");
        let err = dsl_check_attrs(&v).expect_err("no judgement rejected");
        assert!(
            err.to_string().contains("at least one of `requires`"),
            "message = {err}"
        );
    }

    #[test]
    fn a_duplicate_key_is_rejected() {
        let v = variant(
            "#[dsl_check(requires = \"state(A)\", requires = \"state(B)\")] X { id: NodeId }",
        );
        let err = dsl_check_attrs(&v).expect_err("duplicate rejected");
        assert!(err.to_string().contains("duplicate `requires`"), "{err}");
    }

    #[test]
    fn variables_and_nested_constructors_parse() {
        let v = variant(
            "#[dsl_check(produces = \"state(ServiceRunning(Named($name, v1)))\")] \
             X { id: NodeId, name: String }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        let produces = attrs.produces.as_ref().expect("produces parsed");
        assert_eq!(
            produces.args,
            [CheckTermDecl::Ctor(
                "ServiceRunning".into(),
                vec![CheckTermDecl::Ctor(
                    "Named".into(),
                    vec![
                        CheckTermDecl::Var("name".into()),
                        CheckTermDecl::Atom("v1".into()),
                    ],
                )],
            )]
        );
        assert_eq!(attrs.mentioned_vars(), ["name"]);
    }

    #[test]
    fn bind_wires_a_variable_to_a_payload_field() {
        let v = variant(
            "#[dsl_check(requires = \"state(ServiceRunning($target))\", \
             bind(target = \"target\"))] \
             Readiness { id: NodeId, target: String, port: u16 }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        assert_eq!(attrs.binds.len(), 1);
        assert_eq!(attrs.binds[0].var, "target");
        assert_eq!(attrs.binds[0].field, "target");
        validate_check_attrs(&v, &attrs, &enum_name()).expect("shape agrees");
    }

    #[test]
    fn bind_must_name_a_payload_field_a_judgement_uses() {
        // No such field.
        let v = variant(
            "#[dsl_check(requires = \"state(Running($n))\", bind(n = \"nmae\"))] \
             X { id: NodeId, name: String }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        let err = validate_check_attrs(&v, &attrs, &enum_name()).expect_err("typo rejected");
        assert!(err.to_string().contains("does not declare"), "{err}");

        // The field is a child slot, not a payload value.
        let v = variant(
            "#[dsl_check(requires = \"state(Running($n))\", bind(n = \"steps\"))] \
             X { id: NodeId, steps: Vec<Phase> }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        let err = validate_check_attrs(&v, &attrs, &enum_name()).expect_err("child slot rejected");
        assert!(
            err.to_string().contains("child slot, not a payload"),
            "{err}"
        );

        // Bound, but nothing mentions `$n` — dead wiring.
        let v = variant(
            "#[dsl_check(requires = \"state(Ready)\", bind(n = \"name\"))] \
             X { id: NodeId, name: String }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        let err = validate_check_attrs(&v, &attrs, &enum_name()).expect_err("unused bind rejected");
        assert!(err.to_string().contains("no judgement"), "{err}");
    }

    #[test]
    fn requires_with_slots_becomes_child_premises() {
        let v = variant(
            "#[dsl_check(requires(cond = \"type(Bool)\", then_branch = \"type($a)\"), \
             concludes = \"type($a)\")] \
             If { id: NodeId, cond: Box<Phase>, then_branch: Box<Phase> }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        assert!(state_premise(&attrs).is_none());
        let children = child_premises(&attrs);
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].0, "cond");
        assert_eq!(children[0].1.args, [CheckTermDecl::Atom("Bool".into())]);
        assert_eq!(children[1].0, "then_branch");
        assert_eq!(children[1].1.args, [CheckTermDecl::Var("a".into())]);
        let concludes = attrs.concludes.as_ref().expect("concludes parsed");
        assert_eq!(concludes.pred, "type");
        validate_check_attrs(&v, &attrs, &enum_name()).expect("shape agrees");
    }

    #[test]
    fn a_child_premise_must_name_a_child_slot() {
        let v = variant(
            "#[dsl_check(requires(conde = \"type(Bool)\"))] If { id: NodeId, cond: Box<Phase> }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        let err = validate_check_attrs(&v, &attrs, &enum_name()).expect_err("typo rejected");
        assert!(err.to_string().contains("does not declare"), "{err}");

        let v = variant(
            "#[dsl_check(requires(version = \"type(Bool)\"))] X { id: NodeId, version: String }",
        );
        let attrs = dsl_check_attrs(&v).expect("annotation parses");
        let err = validate_check_attrs(&v, &attrs, &enum_name()).expect_err("payload rejected");
        assert!(
            err.to_string().contains("payload field, not a child slot"),
            "{err}"
        );
    }

    #[test]
    fn a_fold_seed_cannot_carry_a_variable() {
        let f = field("#[dsl_check(fold(initial = \"state($x)\"))] steps: Vec<Phase>");
        let err = dsl_check_fold_attr(&f).expect_err("`$var` seed rejected");
        assert!(
            err.to_string()
                .contains("cannot appear in a fold's initial"),
            "{err}"
        );
    }

    #[test]
    fn a_malformed_fact_is_rejected() {
        let v = variant("#[dsl_check(requires = \"state(Ready\")] X { id: NodeId }");
        let err = dsl_check_attrs(&v).expect_err("unbalanced parens rejected");
        assert!(err.to_string().contains("unbalanced parentheses"), "{err}");

        let v = variant("#[dsl_check(requires = \"1state(Ready)\")] X { id: NodeId }");
        let err = dsl_check_attrs(&v).expect_err("bad predicate rejected");
        assert!(err.to_string().contains("invalid predicate name"), "{err}");

        let v = variant("#[dsl_check(requires = \"state(not an atom)\")] X { id: NodeId }");
        let err = dsl_check_attrs(&v).expect_err("bad atom rejected");
        assert!(err.to_string().contains("invalid argument"), "{err}");
    }

    #[test]
    fn fold_declares_the_initial_state() {
        let f = field("#[dsl_check(fold(initial = \"state(Raw)\"))] steps: Vec<Phase>");
        let initial = dsl_check_fold_attr(&f)
            .expect("annotation parses")
            .expect("initial present");
        assert_eq!(initial.pred, "state");
        assert_eq!(initial.args, [CheckTermDecl::Atom("Raw".into())]);

        let plain = field("steps: Vec<Phase>");
        assert!(
            dsl_check_fold_attr(&plain)
                .expect("no annotation")
                .is_none()
        );
    }

    #[test]
    fn an_unknown_fold_key_is_a_compile_error() {
        let f = field("#[dsl_check(fold(start = \"state(Raw)\"))] steps: Vec<Phase>");
        let err = dsl_check_fold_attr(&f).expect_err("unknown inner key rejected");
        assert!(
            err.to_string().contains("unsupported `fold(...)` key"),
            "{err}"
        );

        let f = field("#[dsl_check(requires = \"state(Raw)\")] steps: Vec<Phase>");
        let err = dsl_check_fold_attr(&f).expect_err("variant key on a field rejected");
        assert!(
            err.to_string().contains("on a child slot"),
            "message = {err}"
        );
    }

    #[test]
    fn normalize_collapses_token_stream_whitespace() {
        assert_eq!(normalize_type_str("Option < String >"), "Option<String>");
        assert_eq!(normalize_type_str("Vec < String >"), "Vec<String>");
        assert_eq!(
            normalize_type_str("HashMap < String , u32 >"),
            "HashMap<String, u32>"
        );
        assert_eq!(
            normalize_type_str("Option < Vec < String > >"),
            "Option<Vec<String>>"
        );
        assert_eq!(
            normalize_type_str("std :: string :: String"),
            "std::string::String"
        );
        assert_eq!(normalize_type_str("& str"), "&str");
        assert_eq!(normalize_type_str("& 'a str"), "&'a str");
        assert_eq!(normalize_type_str("String"), "String");
    }
}
