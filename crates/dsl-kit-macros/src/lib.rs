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
    Direct,
    Boxed,
    Optional,
    Many,
}

fn detect_recursion(ty: &Type, enum_name: &Ident) -> Option<Recursion> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return None;
    };
    let last = path.segments.last()?;
    if last.ident == *enum_name && matches!(last.arguments, PathArguments::None) {
        return Some(Recursion::Direct);
    }
    let PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    let GenericArgument::Type(inner) = args.args.first()? else {
        return None;
    };
    let Type::Path(TypePath { path: inner_path, .. }) = inner else {
        return None;
    };
    let inner_last = inner_path.segments.last()?;
    if inner_last.ident != *enum_name {
        return None;
    }
    match last.ident.to_string().as_str() {
        "Box" => Some(Recursion::Boxed),
        "Option" => Some(Recursion::Optional),
        "Vec" => Some(Recursion::Many),
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
            Recursion::Many => quote! { _v.extend(#field_ident.iter()); },
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
            Recursion::Many => quote! { _v.extend(#field_ident.iter_mut()); },
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
