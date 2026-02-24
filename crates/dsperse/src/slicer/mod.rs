pub mod analyzer;
pub mod autotiler;
pub mod onnx_proto;
pub mod onnx_slicer;
pub mod tensor_graph;

pub use onnx_slicer::slice_model;

pub(crate) const ELEMENTWISE_OPS: &[&str] = &[
    "Sigmoid", "Mul", "Add", "Sub", "Div", "Relu", "LeakyRelu", "PRelu",
    "Tanh", "Clip", "Neg", "Abs", "Sqrt", "Exp", "Log", "Pow", "Sin", "Cos",
];

pub(crate) const SHAPE_PRESERVING_OPS: &[&str] = &[
    "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh", "Clip", "Neg",
    "Abs", "Sqrt", "Exp", "Log", "Sin", "Cos", "BatchNormalization",
    "Dropout", "Identity",
];
