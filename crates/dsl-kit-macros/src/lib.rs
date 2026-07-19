//! Derive macros for `dsl-kit`.
//!
//! At this stage only the skeleton of `#[derive(DslNode)]` is implemented:
//! it accepts an enum whose every variant carries at least one field, and
//! generates a `DslNode` impl that delegates `node_id()` to the first
//! field (which must itself implement `DslNode`).
//!
//! The macro will grow to cover visitor generation, schema derivation, and
//! stepper state-machine construction; the current shape is intentionally
//! narrow so that the wiring end to end can be validated first.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(DslNode)]
pub fn derive_dsl_node(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(
            &input,
            "#[derive(DslNode)] currently supports enums only",
        )
        .to_compile_error()
        .into();
    };

    let arms = data.variants.iter().map(|variant| {
        let variant_ident = &variant.ident;
        match &variant.fields {
            Fields::Unnamed(fields) if !fields.unnamed.is_empty() => {
                let bindings = (0..fields.unnamed.len()).map(|i| {
                    if i == 0 {
                        quote!(inner)
                    } else {
                        quote!(_)
                    }
                });
                quote! {
                    Self::#variant_ident(#(#bindings),*) => inner.node_id(),
                }
            }
            Fields::Named(fields) => {
                let first = fields.named.first().and_then(|f| f.ident.as_ref());
                if let Some(first_ident) = first {
                    quote! {
                        Self::#variant_ident { #first_ident, .. } => #first_ident.node_id(),
                    }
                } else {
                    quote! {
                        Self::#variant_ident { .. } => ::dsl_kit_core::NodeId(0),
                    }
                }
            }
            _ => quote! {
                Self::#variant_ident => ::dsl_kit_core::NodeId(0),
            },
        }
    });

    let expanded = quote! {
        impl #impl_generics ::dsl_kit_core::DslNode for #name #ty_generics #where_clause {
            fn node_id(&self) -> ::dsl_kit_core::NodeId {
                match self {
                    #(#arms)*
                }
            }
        }
    };

    expanded.into()
}
