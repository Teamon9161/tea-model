use anyhow::{bail, ensure, Result};
use polars::export::arrow::legacy::utils::CustomIterTools;
use polars::prelude::*;
use tea_time::{
    unit::{Microsecond, Millisecond, Nanosecond},
    DateTime, TimeDelta, TimeUnitTrait,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrOrUsize<'a> {
    Str(&'a str),
    Usize(usize),
}

impl<'a> From<&'a str> for StrOrUsize<'a> {
    #[inline]
    fn from(s: &'a str) -> Self {
        StrOrUsize::Str(s)
    }
}

impl From<usize> for StrOrUsize<'_> {
    #[inline]
    fn from(s: usize) -> Self {
        StrOrUsize::Usize(s)
    }
}

impl<'a> StrOrUsize<'a> {
    #[inline]
    pub fn is_str(&self) -> bool {
        matches!(self, StrOrUsize::Str(_))
    }

    #[inline]
    pub fn is_usize(&self) -> bool {
        matches!(self, StrOrUsize::Usize(_))
    }
}

/// if predict period n is not 1, we should cut off the last n-1 period
/// so that there is no future information in the training set
#[derive(Clone)]
pub struct RollingPredictOpt<'a> {
    pub window: &'a str, // train window
    pub adjust: &'a str, // window to adjust model
    pub y: &'a str,
    // TODO: use a fixed cut off period may not be a good idea, as it does not consider the holidays
    pub cut_off: StrOrUsize<'a>, // cut off time
    pub time: &'a str,
    pub min_window: Option<&'a str>,
    pub fit_train: bool,
    pub expanding: bool,
    pub verbose: bool,
    pub time_fmt: &'a str,
    pub useful: &'a str, // boolean column indicating whether the row is useful
}

impl Default for RollingPredictOpt<'_> {
    fn default() -> Self {
        RollingPredictOpt {
            window: "3y",
            adjust: "6mo",
            y: "y",
            cut_off: "0ns".into(),
            time: "trading_date",
            min_window: None,
            fit_train: true,
            expanding: false,
            verbose: true,
            time_fmt: "%Y-%m-%d %H:%M:%S",
            useful: "useful",
        }
    }
}

fn find_idx<U: TimeUnitTrait>(
    time_series: &Series,
    start: DateTime<U>,
    end: DateTime<U>,
    cut_off_end: DateTime<U>,
    predict_end: DateTime<U>,
    begin_at: Option<usize>,
) -> Result<(usize, usize, usize, usize)> {
    ensure!(
        cut_off_end > start,
        "cut off end time should be greater than start time"
    );
    let time_series = if let Some(begin_at) = begin_at {
        ensure!(
            begin_at <= time_series.len(),
            "begin_at should be less than length of time series"
        );
        time_series.slice(begin_at as i64, time_series.len() - begin_at)
    } else {
        time_series.clone()
    };
    let begin_at = begin_at.unwrap_or(0);
    let mut start_idx = None;
    let mut end_idx = None;
    let mut cut_off_end_idx = None;
    let mut predict_end_idx = None;
    // TODO: this should be faster if we judge whether cut_off_end is equal to end and
    // implement the loop separately for special case
    match time_series.dtype() {
        DataType::Datetime(TimeUnit::Nanoseconds, None) => {
            let start: DateTime<Nanosecond> = start.into_unit();
            let end: DateTime<Nanosecond> = end.into_unit();
            let cut_off_end: DateTime<Nanosecond> = cut_off_end.into_unit();
            let predict_end: DateTime<Nanosecond> = predict_end.into_unit();
            for (idx, dt) in time_series.datetime()?.into_iter().enumerate() {
                if let Some(dt) = dt {
                    let dt: DateTime<Nanosecond> = dt.into();
                    if start_idx.is_none() && dt >= start {
                        start_idx = Some(idx);
                    }
                    if cut_off_end_idx.is_none() && dt > cut_off_end {
                        cut_off_end_idx = Some(idx);
                    }
                    if end_idx.is_none() && dt > end {
                        end_idx = Some(idx);
                    }
                    if predict_end_idx.is_none() && dt > predict_end {
                        predict_end_idx = Some(idx);
                        break;
                    }
                }
            }
        }
        DataType::Datetime(TimeUnit::Microseconds, None) => {
            let start: DateTime<Microsecond> = start.into_unit();
            let end: DateTime<Microsecond> = end.into_unit();
            let cut_off_end: DateTime<Microsecond> = cut_off_end.into_unit();
            let predict_end: DateTime<Microsecond> = predict_end.into_unit();
            for (idx, dt) in time_series.datetime()?.into_iter().enumerate() {
                if let Some(dt) = dt {
                    let dt: DateTime<Microsecond> = dt.into();
                    if start_idx.is_none() && dt >= start {
                        start_idx = Some(idx);
                    }
                    if cut_off_end_idx.is_none() && dt > cut_off_end {
                        cut_off_end_idx = Some(idx);
                    }
                    if end_idx.is_none() && dt > end {
                        end_idx = Some(idx);
                    }
                    if predict_end_idx.is_none() && dt > predict_end {
                        predict_end_idx = Some(idx);
                        break;
                    }
                }
            }
        }
        DataType::Datetime(TimeUnit::Milliseconds, None) => {
            let start: DateTime<Millisecond> = start.into_unit();
            let end: DateTime<Millisecond> = end.into_unit();
            let cut_off_end: DateTime<Millisecond> = cut_off_end.into_unit();
            let predict_end: DateTime<Millisecond> = predict_end.into_unit();
            for (idx, dt) in time_series.datetime()?.into_iter().enumerate() {
                if let Some(dt) = dt {
                    let dt: DateTime<Millisecond> = dt.into();
                    if start_idx.is_none() && dt >= start {
                        start_idx = Some(idx);
                    }
                    if cut_off_end_idx.is_none() && dt > cut_off_end {
                        cut_off_end_idx = Some(idx);
                    }
                    if end_idx.is_none() && dt > end {
                        end_idx = Some(idx);
                    }
                    if predict_end_idx.is_none() && dt > predict_end {
                        predict_end_idx = Some(idx);
                        break;
                    }
                }
            }
        }
        _ => bail!("time_series should be datetime"),
    }
    ensure!(
        (start_idx.is_some() && end_idx.is_some() && cut_off_end_idx.is_some()),
        "start, end, cut_off_end can not be found successfully in time series"
    );
    Ok((
        start_idx.unwrap() + begin_at,
        end_idx.unwrap() + begin_at,
        cut_off_end_idx.unwrap() + begin_at,
        predict_end_idx.unwrap_or(time_series.len()) + begin_at,
    ))
}

pub trait Model {
    fn name(&self) -> &str;

    fn fit(&mut self, df: &DataFrame, y: &str) -> Result<()>;

    fn predict(&self, df: &DataFrame) -> Result<Series>;

    #[inline]
    fn fit_useful(&mut self, df: &DataFrame, y: &str, useful: &str) -> Result<()> {
        let df = Self::train_preprocess(df, useful)?;
        self.fit(&df, y)
    }

    #[inline]
    fn train_preprocess(df: &DataFrame, useful: &str) -> Result<DataFrame> {
        if let Ok(useful_se) = df.column(useful) {
            let useful_ca = useful_se.bool()?;
            Ok(df.filter(&useful_ca)?)
        } else {
            Ok(df.clone())
        }
    }

    fn rolling_predict(&mut self, df: &DataFrame, opt: RollingPredictOpt) -> Result<Series> {
        let y = opt.y;
        let window = TimeDelta::parse(opt.window)?;
        let adjust = TimeDelta::parse(opt.adjust)?;
        let min_window = opt.min_window.map(|s| s.parse().unwrap()).unwrap_or(window);
        let features = df
            .get_column_names()
            .into_iter()
            .filter(|c| *c != y)
            .collect::<Vec<_>>();
        if df.is_empty() {
            return Ok(Series::new_empty("", &DataType::Float64));
        }
        let mut start_time: DateTime = df[opt.time]
            .get(0)
            .unwrap()
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .try_extract::<i64>()?
            .into();
        let mut current_time = start_time + min_window;
        let mut init_flag = true;
        let end_time: DateTime = df[opt.time]
            .get(df.height() - 1)
            .unwrap()
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .try_extract::<i64>()?
            .into();
        let mut out = vec![f64::NAN; df.height()];
        if current_time > end_time {
            eprintln!("The length of the data is less than the minimum window size");
        }

        let unique_time_series = if opt.cut_off.is_usize() {
            Some(
                df[opt.time]
                    .unique()?
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?,
            )
        } else {
            None
        };
        let mut last_start_idx = 0;
        while current_time <= end_time {
            let current_dt_for_train = match opt.cut_off {
                StrOrUsize::Str(cut_off) => {
                    if cut_off != "0ns" {
                        current_time - TimeDelta::parse(cut_off)?
                    } else {
                        current_time
                    }
                }
                StrOrUsize::Usize(cut_off) => {
                    let se = unique_time_series.as_ref().unwrap().datetime()?;
                    let ca = se.filter(&se.lt_eq(current_time.into_i64()))?;
                    let time = ca.get(ca.len() - cut_off - 1);
                    time.map(|t| t.into()).unwrap_or(current_time)
                }
            };
            // dbg!(current_dt_for_train.strftime(None), current_time);

            let predict_time = current_time + adjust;

            let (start_idx, end_idx, cut_off_end_idx, predict_idx) = find_idx(
                &df[opt.time],
                start_time,
                current_time,
                current_dt_for_train,
                predict_time,
                Some(last_start_idx),
            )?;
            last_start_idx = start_idx;
            let current_train_df = df.slice(start_idx as i64, cut_off_end_idx - start_idx);

            self.fit_useful(&current_train_df, y, opt.useful)?;
            if init_flag && opt.fit_train {
                init_flag = false;
                let start_df = df.slice(0, end_idx).select(&features)?;
                let predict = self.predict(&start_df)?;
                assert_eq!(end_idx, predict.len());
                out[0..end_idx]
                    .iter_mut()
                    .zip(predict.f64()?)
                    .for_each(|(o, pv)| *o = pv.unwrap_or(f64::NAN));
            }
            let df_to_predict = df
                .slice(end_idx as i64, predict_idx - end_idx)
                .select(&features)?;
            if opt.verbose {
                let last_train_time: DateTime = current_train_df[opt.time]
                    .get(current_train_df.height() - 1)
                    .unwrap()
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .try_extract::<i64>()?
                    .into();
                println!(
                    "train_time: {} -> {}, last_train_time: {:?}, last_predict_time: {:?}",
                    start_time.strftime(Some(opt.time_fmt)),
                    current_time.strftime(Some(opt.time_fmt)),
                    last_train_time.strftime(Some(opt.time_fmt)),
                    predict_time.strftime(Some(opt.time_fmt)),
                );
                let first_predict_time: DateTime = df_to_predict[opt.time]
                    .get(0)
                    .unwrap()
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .try_extract::<i64>()?
                    .into();
                let last_predict_time: DateTime = df_to_predict[opt.time]
                    .get(df_to_predict.height() - 1)
                    .unwrap()
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .try_extract::<i64>()?
                    .into();
                println!(
                    "predict_df: {} -> {}",
                    first_predict_time.strftime(Some(opt.time_fmt)),
                    last_predict_time.strftime(Some(opt.time_fmt))
                )
            }
            let predict = self.predict(&df_to_predict)?;
            assert_eq!(predict_idx - end_idx, predict.len());
            out[end_idx..predict_idx]
                .iter_mut()
                .zip(predict.f64()?)
                .for_each(|(o, pv)| *o = pv.unwrap_or(f64::NAN));
            if !opt.expanding && start_time + window <= current_time {
                start_time = start_time + adjust;
            }
            current_time = current_time + adjust;
        }
        let ca: Float64Chunked = out
            .into_iter()
            .map(|v| if v.is_nan() { None } else { Some(v) })
            .collect_trusted();
        Ok(ca.into_series().with_name(self.name()))
    }
}
