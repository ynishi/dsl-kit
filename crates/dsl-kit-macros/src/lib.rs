//! Derive macros for `dsl-kit`.
//!
//! `#[derive(DslNode)]` accepts an `enum` whose every variant uses named
//! fields, exactly one of which is called `id` and typed `NodeId`. The
//! macro generates three impls in one shot:
//!
//! - [`DslNode`] — returns the `id` field for each variant.
//! - [`Walk`] — returns direct children by inspecting each variant's
//!   field types. Any field of type `T`, `Box<T>`, `Option<T>`, or
//!   `Vec<T>`, where `T` is the enum itself, is treated as a child.
//! - [`WalkMut`] — mutable counterpart of `Walk`.
//!
//! Variants may carry additional fields of unrelated types (payload); those
//! fields are ignored by the traversal.
//!
//! Advanced shapes (indirect recursion through a struct, mixed tuple /
//! named variants, generic ASTs) can implement the traits by hand.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
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
fn last_segment<'a>(ty: &'a Type) -> Option<&'a syn::PathSegment> {
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

#[proc_macro_derive(DslNode)]
pub fn derive_dsl_node(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(
            &input,
            "#[derive(DslNode)] currently supports enums only",
        )
        .to_compile_error()
        .into();
    };

    let mut node_arms = Vec::new();
    let mut child_arms = Vec::new();
    let mut child_mut_arms = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;

        let Fields::Named(fields) = &variant.fields else {
            return syn::Error::new_spanned(
                variant,
                "#[derive(DslNode)] requires every variant to use named fields",
            )
            .to_compile_error()
            .into();
        };

        // Locate the `id` field.
        let has_id = fields.named.iter().any(|f| {
            f.ident
                .as_ref()
                .is_some_and(|ident| ident == "id")
        });
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
