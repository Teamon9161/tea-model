#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Boosting {
    #[default]
    Gbdt,
    Rf,
    Dart,
    Goss,
}
