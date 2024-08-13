use std::ops::{Deref, DerefMut, Index};

use crate::{Boosting, Device, Model, SeriesExt};
use anyhow::Result;
pub use lgbm::{
    mat::MatLayouts,
    parameters::{
        Boosting as LgbmBoosting, DeviceType as LgbmDeviceType, Objective, ParameterValue,
        Verbosity,
    },
    Booster, Dataset, Field, MatBuf, Parameters, PredictType,
};
use polars::prelude::*;

#[derive(Debug, Default, Clone)]
pub struct LgbmOpt(pub Parameters);

impl From<Device> for LgbmDeviceType {
    fn from(d: Device) -> Self {
        match d {
            Device::Cpu => LgbmDeviceType::Cpu,
            Device::Gpu => LgbmDeviceType::Gpu,
            Device::Cuda => LgbmDeviceType::Cuda,
        }
    }
}

impl From<Device> for ParameterValue {
    #[inline]
    fn from(value: Device) -> Self {
        let d: LgbmDeviceType = value.into();
        d.into()
    }
}

impl From<Boosting> for LgbmBoosting {
    fn from(b: Boosting) -> Self {
        match b {
            Boosting::Gbdt => LgbmBoosting::Gbdt,
            Boosting::Rf => LgbmBoosting::Rf,
            Boosting::Dart => LgbmBoosting::Dart,
            Boosting::Goss => LgbmBoosting::Goss,
        }
    }
}

impl From<Boosting> for ParameterValue {
    #[inline]
    fn from(value: Boosting) -> Self {
        let b: LgbmBoosting = value.into();
        b.into()
    }
}

impl Deref for LgbmOpt {
    type Target = Parameters;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LgbmOpt {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Index<&str> for LgbmOpt {
    type Output = ParameterValue;

    fn index(&self, key: &str) -> &Self::Output {
        self.get(key).unwrap()
    }
}

impl LgbmOpt {
    #[inline]
    pub fn new() -> Self {
        LgbmOpt(Parameters::new())
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<&ParameterValue> {
        self.0 .0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn with_num_iterations(mut self, num_iterations: usize) -> Self {
        self.0.push("num_iterations", num_iterations);
        self
    }

    pub fn with_objective(mut self, objective: Objective) -> Self {
        self.0.push("objective", objective);
        self
    }

    pub fn with_num_leaves(mut self, num_leaves: i32) -> Self {
        self.0.push("num_leaves", num_leaves);
        self
    }

    pub fn with_max_depth(mut self, max_depth: i32) -> Self {
        self.0.push("max_depth", max_depth);
        self
    }

    pub fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.0.push("verbosity", verbosity);
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.0.push("device_type", device);
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.0.push("learning_rate", learning_rate);
        self
    }

    pub fn with_seed(mut self, seed: i32) -> Self {
        self.0.push("seed", seed);
        self
    }

    pub fn with_num_threads(mut self, num_threads: i32) -> Self {
        self.0.push("num_threads", num_threads);
        self
    }

    pub fn with_boosting(mut self, boosting: Boosting) -> Self {
        self.0.push("boosting_type", boosting);
        self
    }
}

pub struct LgbmModel {
    pub model: Option<Booster>,
    pub opt: LgbmOpt,
}

impl LgbmModel {
    pub fn new(opt: LgbmOpt) -> Self {
        LgbmModel { model: None, opt }
    }

    pub fn train(&mut self, dataset: Arc<Dataset>) -> Result<()> {
        let num_iterations: usize = self
            .opt
            .get("num_iterations")
            .map(|v| match v {
                ParameterValue::Int(v) => (*v) as usize,
                ParameterValue::USize(v) => *v,
                _ => unreachable!(),
            })
            .unwrap_or(100);
        let mut b = Booster::new(dataset, &self.opt.0)?;
        for _ in 0..num_iterations {
            if b.update_one_iter()? {
                break;
            }
        }
        self.model = Some(b);
        Ok(())
    }
}

impl Model for LgbmModel {
    #[inline]
    fn name(&self) -> &str {
        "lgbm"
    }

    #[inline]
    fn fit(&mut self, df: &DataFrame, y: &str) -> Result<()> {
        let features = Self::get_features(df, y);
        let y = df[y].f32_array()?.into_raw_vec();
        let vec_x = features
            .into_iter()
            .map(|s| df[s].f64_array().unwrap())
            .collect::<Vec<_>>();
        let x_view: Vec<_> = vec_x.iter().map(|a| a.view()).collect();
        let x = ndarray::stack(ndarray::Axis(1), &x_view)?;
        let layout = if x.is_standard_layout() {
            MatLayouts::RowMajor
        } else {
            MatLayouts::ColMajor
        };
        let nrow = x.shape()[0];
        let ncol = x.shape()[1];
        let mat = MatBuf::from_vec(x.into_raw_vec(), nrow, ncol, layout);
        let mut train_dataset = Dataset::from_mat(&mat, None, &self.opt.0)?;
        train_dataset.set_field(Field::LABEL, &y)?;
        self.train(Arc::new(train_dataset))
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
        let layout = if x.is_standard_layout() {
            MatLayouts::RowMajor
        } else {
            MatLayouts::ColMajor
        };
        let nrow = x.shape()[0];
        let ncol = x.shape()[1];
        let mat = MatBuf::from_vec(x.into_raw_vec(), nrow, ncol, layout);
        let res = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model is not fitted."))?
            .predict_for_mat(mat, PredictType::Normal, 0, None, &Parameters::new())?;
        let ca: Float64Chunked = res.values().iter().cloned().collect_ca_trusted(self.name());
        Ok(ca.into_series())
    }
}
