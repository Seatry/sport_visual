
pub(crate) mod number_trait;
pub(crate) mod number_impl_big_rational;
pub(crate) mod number_impl_gmp;

pub mod number;
pub mod point;
pub mod vector;
pub mod mesh;
pub mod triangle;

pub(crate) mod zero_trait;
pub(crate) use self::zero_trait::Zero;

pub(crate) mod signed_trait;
pub(crate) use self::signed_trait::Signed;




