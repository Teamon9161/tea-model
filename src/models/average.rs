use crate::Model;
use anyhow::Result;
use polars::frame::NullStrategy;
use polars::prelude::*;

pub struct AverageModel();

// impl AverageModel {
//     #[inline]
//     pub fn new() -> Self {
//         Self {}
//     }
// }

impl Model for AverageModel {
    #[inline]
    fn name(&self) -> &str {
        "average"
    }

    #[inline]
    fn fit(&mut self, _df: &DataFrame, _y: &str) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn predict(&self, df: &DataFrame) -> Result<Series> {
        let out = df.mean_horizontal(NullStrategy::Ignore)?;
        out.ok_or_else(|| anyhow::Error::msg("Dataframe should have at least one feature."))
    }
}