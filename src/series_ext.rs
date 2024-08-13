use anyhow::Result;
use ndarray::Array1;
use polars::prelude::*;

#[allow(dead_code)]
pub(crate) trait SeriesExt {
    fn cast_f64(&self) -> Result<Series>;
    fn cast_bool(&self) -> Result<Series>;
    fn cast_f32(&self) -> Result<Series>;

    fn f32_array(&self) -> Result<Array1<f32>>;
    fn f64_array(&self) -> Result<Array1<f64>>;
}

impl SeriesExt for Series {
    #[inline]
    fn cast_f64(&self) -> Result<Series> {
        if let DataType::Float64 = self.dtype() {
            Ok(self.clone())
        } else {
            Ok(Series::cast(self, &DataType::Float64)?)
        }
    }

    #[inline]
    fn cast_bool(&self) -> Result<Series> {
        if let DataType::Boolean = self.dtype() {
            Ok(self.clone())
        } else {
            Ok(Series::cast(self, &DataType::Boolean)?)
        }
    }

    #[inline]
    fn cast_f32(&self) -> Result<Series> {
        if let DataType::Float32 = self.dtype() {
            Ok(self.clone())
        } else {
            Ok(Series::cast(self, &DataType::Float32)?)
        }
    }

    #[inline]
    fn f32_array(&self) -> Result<Array1<f32>> {
        let s = self.cast_f32()?;
        Ok(s.f32()?
            .iter()
            .map(|v| if v.is_none() { f32::NAN } else { v.unwrap() })
            .collect::<Array1<f32>>())
    }

    #[inline]
    fn f64_array(&self) -> Result<Array1<f64>> {
        let s = self.cast_f64()?;
        Ok(s.f64()?
            .iter()
            .map(|v| if v.is_none() { f64::NAN } else { v.unwrap() })
            .collect::<Array1<f64>>())
    }
}
