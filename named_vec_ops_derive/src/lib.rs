
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields};

#[proc_macro_derive(NamedVecOps)]
pub fn named_vec_ops_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    // let generics = &input.generics;

    let generics = input.generics.clone();

    let (impl_generics, ty_generics, original_where_clause) = generics.split_for_impl();

    let mut where_clause = original_where_clause.cloned().unwrap_or(syn::WhereClause {
        where_token: Default::default(),
        predicates: syn::punctuated::Punctuated::new(),
    });

    where_clause.predicates.push(parse_quote! {
        T: Copy
         + Clone
         + ::nalgebra::Scalar
         + std::ops::Add<T, Output = T>
         + std::ops::Sub<T, Output = T>
         + std::ops::Mul<T, Output = T>
         + std::ops::AddAssign<T>
         + std::ops::SubAssign<T>
         //  + ::num_dual::DualNum<F>
    });

    let fields = match &input.data {
        Data::Struct(data_struct) => {
            if let Fields::Named(fields_named) = &data_struct.fields {
                &fields_named.named
            } else {
                panic!("NamedVecOps can only be derived for structs with named fields");
            }
        }
        _ => panic!("NamedVecOps can only be derived for structs")
    };

    let n_fields = fields.len();
    let n_literal = syn::Index::from(n_fields);
    let field_names: Vec<_> = fields.iter().map(|f| &f.ident).collect();
    let field_indexes: Vec<_> = (0..n_fields).map(syn::Index::from).collect();
    
    let add_fields = fields.iter().map(|f| {
        let field = &f.ident;
        quote! { #field: self.#field + rhs.#field }
    });

    let add_assign_fields = fields.iter().map(|f| {
        let field = &f.ident;
        quote! { self.#field += rhs.#field; }
    });

    let sub_assign_fields = fields.iter().map(|f| {
        let field = &f.ident;
        quote! { self.#field -= rhs.#field; }
    });

    let sub_fields = fields.iter().map(|f| {
        let field = &f.ident;
        quote! { #field: self.#field - rhs.#field }
    });

    let mul_fields: Vec<_> = fields.iter().map(|f| {
        let field = &f.ident;
        quote! { #field: self.#field * rhs }
    }).collect();

    let expanded = quote! {
        impl #impl_generics named_vec_ops::NamedVecOps<T, #n_literal> for #name #ty_generics #where_clause {
            fn to_svector(&self) -> ::nalgebra::SVector<T, #n_literal> {
                ::nalgebra::SVector::from([
                    #(self.#field_names.clone()),*
                ])
            }
            fn from_svector(v: &::nalgebra::SVector<T, #n_literal>) -> Self {
                Self {
                    #(#field_names: v[#field_indexes].clone()),*
                }
            }
        }

        impl #impl_generics std::ops::AddAssign for #name #ty_generics #where_clause {
            fn add_assign(&mut self, rhs: Self) {
                #(#add_assign_fields)*
            }
        }

        impl #impl_generics std::ops::SubAssign for #name #ty_generics #where_clause {
            fn sub_assign(&mut self, rhs: Self) {
                #(#sub_assign_fields)*
            }
        }

        impl #impl_generics std::ops::Mul<T> for #name #ty_generics #where_clause {
            type Output = Self;

            fn mul(self, rhs: T) -> Self {
                Self {
                    #(#mul_fields),*
                }
            }
        }

        impl #impl_generics std::ops::Sub<Self> for #name #ty_generics #where_clause {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                Self {
                    #(#sub_fields),*
                }
            }
        }

        impl #impl_generics std::ops::Add<Self> for #name #ty_generics #where_clause {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                Self {
                    #(#add_fields),*
                }
            }
        }

        // orphan rule
        // impl #impl_generics std::ops::Mul<#name #ty_generics> for T #where_clause {
        //     type Output = Self;

        //     fn mul(self, rhs: Self) -> Self {
        //         rhs * self
        //     }
        // }

        // not a struct
        // impl #impl_generics std::ops::Mul<T> for & #name #ty_generics #where_clause {
        //     type Output = Self;

        //     fn mul(self, rhs: T) -> Self {
        //         Self {
        //             #(#mul_fields),*
        //         }
        //     }
        // }

        // orphan rule
        // impl #impl_generics std::ops::Mul<& #name #ty_generics> for T #where_clause {
        //     type Output = Self;

        //     fn mul(self, rhs: & Self) -> Self {
        //         rhs * self
        //     }
        // }
    };

    TokenStream::from(expanded)
}
