//! Derive macros for `dsl-kit`.
//!
//! ## Design
//!
//! One input shape, three derives. Every macro accepts the same enum
//! form (named-field variants, one `id: NodeId` each), so a DSL opts
//! into traversal, schema reflection, and parse-tree building by adding
//! derives — never by restating its shape.
//!
//! `#[derive(DslNode)]` accepts an `enum` whose every variant uses named
//! fields, exactly one of which is called `id` and typed `NodeId`. The
//! macro generates three impls in one shot:
//!
//! - `DslNode` — returns the `id` field for each variant.
//! - `Walk` — returns direct children by inspecting each variant's
//!   field types. Any field of type `T`, `Box<T>`, `Option<T>`, or
//!   `Vec<T>`, where `T` is the enum itself, is treated as a child.
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
//! / `build_child_many`. Types deriving `DslBuild` must also derive
//! `DslSchema` (used for the level-scoped conformance check). Payload
//! fields whose type has no `FromStr` route (e.g. `Option<T>`, which
//! the orphan rule bars downstream crates from implementing) opt out of
//! `build_field` with `#[dsl_build(with = path)]` — the named function
//! converts the field itself. See [`derive_dsl_build`].
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

/// Returns true if `ty` is a `Type::Path` whose last segment matches
/// `enum_name` and carries no generic arguments.
fn matches_enum(ty: &Type, enum_name: &Ident) -> bool {
    match last_segment(ty) {
        Some(seg) => seg.ident == *enum_name && matches!(seg.arguments, PathArguments::None),
        None => false,
    }
}

fn detect_recursion(ty: &Type, enum_name: &Ident) -> Option<Recursion> {
    if matches_enum(ty, enum_name) {
        return Some(Recursion::Direct);
    }

    let seg = last_segment(ty)?;
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
/// `Option<T>` / `Option<Box<T>>` / `Vec<T>` / `Vec<Box<T>>` where `T`
/// is the enum itself) become `ChildSchema` entries; every other named
/// field becomes a `FieldSchema` carrying the Rust type source text.
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
                };
                child_ctors.push(quote! {
                    ::dsl_kit_schema::ChildSchema {
                        name: #ident_str.to_string(),
                        multiplicity: ::dsl_kit_schema::Multiplicity::#mult,
                    }
                });
            } else {
                let ty_src = f.ty.to_token_stream().to_string();
                field_ctors.push(quote! {
                    ::dsl_kit_schema::FieldSchema {
                        name: #ident_str.to_string(),
                        ty: #ty_src.to_string(),
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
///    (`build_child_one` / `_optional` / `_many`) and re-wraps the
///    result in `Box` where the source field is boxed.
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
                let_bindings.push(quote! { let #ident = #path(tree, #ident_str)?; });
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
                };
                let_bindings.push(quote! { let #ident = #helper_call; });
                ctor_fields.push(quote! { #ident });
            } else {
                let ty = &f.ty;
                let_bindings.push(quote! {
                    let #ident: #ty = ::dsl_kit_parse::build_field(tree, #ident_str)?;
                });
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
