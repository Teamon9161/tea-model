use crate::{Model, SeriesExt};
use anyhow::Result;
use linfa::prelude::*;
pub use linfa_linear::{FittedLinearRegression, LinearRegression};
use polars::prelude::*;

enum InnerLinearModel {
    Unfitted(LinearRegression),
    Fitted(FittedLinearRegression<f64>),
}

impl InnerLinearModel {
    #[inline]
    fn new(constant: bool) -> Self {
        if constant {
            InnerLinearModel::Unfitted(LinearRegression::new().with_intercept(true))
        } else {
            InnerLinearModel::Unfitted(LinearRegression::new().with_intercept(false))
        }
    }
}

pub struct LinearModel {
    model: InnerLinearModel,
    pub constant: bool,
}

impl LinearModel {
    pub fn new(constant: bool) -> Self {
        let inner = InnerLinearModel::new(constant);
        LinearModel {
            model: inner,
            constant,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        match &mut self.model {
            InnerLinearModel::Fitted(_) => {
                self.model = InnerLinearModel::new(self.constant);
            }
            InnerLinearModel::Unfitted(_) => {}
        }
    }

    #[inline]
    pub fn is_fitted(&self) -> bool {
        match &self.model {
            InnerLinearModel::Fitted(_) => true,
            InnerLinearModel::Unfitted(_) => false,
        }
    }

    #[inline]
    pub fn as_unfitted(&self) -> &LinearRegression {
        match &self.model {
            InnerLinearModel::Unfitted(m) => m,
            InnerLinearModel::Fitted(_) => panic!("Model is already fitted."),
        }
    }

    #[inline]
    pub fn as_fitted(&self) -> &FittedLinearRegression<f64> {
        match &self.model {
            InnerLinearModel::Fitted(m) => m,
            InnerLinearModel::Unfitted(_) => panic!("Model is not fitted."),
        }
    }
}

impl Model for LinearModel {
    #[inline]
    fn name(&self) -> &str {
        "linear"
    }

    #[inline]
    fn fit(&mut self, df: &DataFrame, y: &str) -> Result<()> {
        self.reset();
        let unfitted_model = self.as_unfitted();
        let features = Self::get_features(df, y);
        let y = df[y].f64_array()?;
        let vec_x = features
            .into_iter()
            .map(|s| df[s].f64_array().unwrap())
            .collect::<Vec<_>>();
        let x_view: Vec<_> = vec_x.iter().map(|a| a.view()).collect();
        let x = ndarray::stack(ndarray::Axis(1), &x_view)?;
        let data_base = DatasetBase::new(x, y);
        let fitted_model = unfitted_model.fit(&data_base)?;
        self.model = InnerLinearModel::Fitted(fitted_model);
        Ok(())
    }

    #[inline]
    fn predict(&self, df: &DataFrame) -> Result<Series> {
        let vec_x = df
            .get_columns()
            .iter()
            .map(|s| s.f64_array().unwrap())
            .collect::<Vec<_>>();
        let x_view: Vec<_> = vec_x.iter().map(|a| a.view()).collect();
        let x = ndarray::stack(ndarray::Axis(1), &x_view)?;
        let predict = self.as_fitted().predict(&x);
        Ok(Float64Chunked::from_vec(self.name(), predict.into_raw_vec()).into_series())
    }
}
