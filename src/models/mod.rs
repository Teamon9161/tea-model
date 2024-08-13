mod average;
pub use average::AverageModel;

#[cfg(feature = "linear")]
mod linear;
#[cfg(feature = "linear")]
pub use linear::LinearModel;

#[cfg(feature = "lgbm")]
mod lgbm;
#[cfg(feature = "lgbm")]
pub use lgbm::{
    LgbmModel, LgbmOpt, Objective as LgbmObjective, ParameterValue as LgbmParameterValue,
    Verbosity as LgbmVerbosity,
};
