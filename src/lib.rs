mod core_trait;
pub use core_trait::{Model, RollingPredictOpt};

mod enums;
pub use enums::*;

pub mod models;

mod series_ext;
#[allow(unused_imports)]
pub(crate) use series_ext::SeriesExt;
