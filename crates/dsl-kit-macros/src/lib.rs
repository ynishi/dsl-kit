//! Derive macros for `dsl-kit`.
//!
//! ## Design
//!
//! One input shape, four derives. Every macro accepts the same enum
//! form (named-field variants, one `id: NodeId` each), so a DSL opts
//! into traversal, schema reflection, parse-tree building, and engine
//! execution by adding derives — never by restating its shape.
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
//! `#[derive(DslExec)]` accepts the same shape and emits an
//! `impl DslExec` — the mechanical half of an engine `Ast`. Every
//! variant names its engine `NodeKind` through a `#[dsl_exec(...)]`
//! annotation (`value` / `read(field)` / `apply = "op"` / `bind(field)`
//! / `branch` / `repeat` / `seq` / `scope(field)` / `maybe` /
//! `call(field)`); recursive child fields are picked up in declaration
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
#[proc_macro_derive(DslSchema)]
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
                    }
                });
            } else if let Some(value_ty) = detect_scalar_map(&f.ty, &name) {
                // Shape 1: `BTreeMap<String, ScalarType>` — keyed
                // slot whose values are non-recursive payloads.
                // Reported as `Multiplicity::Map` with
                // `ChildValueShape::Scalar { ty }`; the value type is
                // stored as Rust source text so schema consumers can
                // dispatch on it without a semantic type system.
                let value_ty_src = normalize_type_str(&value_ty.to_token_stream().to_string());
                child_ctors.push(quote! {
                    ::dsl_kit_schema::ChildSchema {
                        name: #ident_str.to_string(),
                        multiplicity: ::dsl_kit_schema::Multiplicity::Map,
                        value_shape: ::dsl_kit_schema::ChildValueShape::Scalar {
                            ty: #value_ty_src.to_string(),
                        },
                    }
                });
            } else {
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
///    caller-supplied `IdGen`.
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

        variant_arms.push(quote! {
            #variant_name_str => {
                #(#let_bindings)*
                ::std::result::Result::Ok(Self::#variant_ident {
                    id: ids.node(),
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
    /// `#[dsl_exec(call(FIELD))]` — effect leaf labelled by `FIELD`.
    Call { label_field: Ident },
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
                spec = Some(ExecSpec::Call {
                    label_field: field_of(&meta)?,
                });
            } else {
                return Err(meta.error(
                    "unknown #[dsl_exec(...)] form; expected one of value / read(field) / \
                     apply = \"op\" / bind(field) / branch / repeat / seq / scope(field) / \
                     maybe / call(field)",
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
            ExecSpec::Call { label_field } => {
                if !recursive.is_empty() {
                    return err("#[dsl_exec(call(..))] variants must have no child fields".into());
                }
                kind_arms.push(quote! {
                    Self::#variant_ident { #label_field, .. } =>
                        ::dsl_kit_core::NodeKind::Call {
                            label: #label_field.clone(),
                            payload: ::dsl_kit_core::serde_json::Value::Null,
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

/// Parses a field's `#[dsl_build(with = path)]` annotation, if present.
/// `Ok(None)` when the field carries no `dsl_build` attribute.
fn dsl_build_with_attr(f: &syn::Field) -> syn::Result<Option<syn::Path>> {
    let mut with: Option<syn::Path> = None;
    for attr in &f.attrs {
        if !attr.path().is_ident("dsl_build") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("with") {
                with = Some(meta.value()?.parse::<syn::Path>()?);
                Ok(())
            } else {
                Err(meta.error("unsupported #[dsl_build(...)] key; expected `with = <path>`"))
            }
        })?;
        if with.is_none() {
            return Err(syn::Error::new_spanned(
                attr,
                "#[dsl_build] requires `with = <path>`",
            ));
        }
    }
    Ok(with)
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
    use super::normalize_type_str;

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
