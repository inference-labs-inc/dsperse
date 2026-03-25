pub mod analyzer;
pub mod autotiler;
pub mod combiner;
pub mod materializer;
pub mod onnx_proto;
pub mod onnx_slicer;

pub use onnx_slicer::slice_model;

pub(crate) const UNARY_ACTIVATIONS: &[&str] = &[
    "Relu",
    "LeakyRelu",
    "PRelu",
    "Sigmoid",
    "Tanh",
    "Clip",
    "Neg",
    "Abs",
    "Sqrt",
    "Exp",
    "Log",
    "Sin",
    "Cos",
    "Erf",
];

pub(crate) const UNARY_STRUCTURAL: &[&str] = &["Cast", "Not", "Identity", "Dropout"];

pub(crate) const BINARY_ARITHMETIC: &[&str] = &["Add", "Sub", "Mul", "Div", "Pow", "Max", "Min"];

pub(crate) const NORMALIZATION_OPS: &[&str] =
    &["BatchNormalization", "Softmax", "LayerNormalization"];

pub(crate) fn is_shape_preserving(op: &str) -> bool {
    UNARY_ACTIVATIONS.contains(&op)
        || UNARY_STRUCTURAL.contains(&op)
        || NORMALIZATION_OPS.contains(&op)
}

pub(crate) fn is_elementwise(op: &str) -> bool {
    UNARY_ACTIVATIONS.contains(&op) || BINARY_ARITHMETIC.contains(&op)
}

pub(crate) fn is_binary_arithmetic(op: &str) -> bool {
    BINARY_ARITHMETIC.contains(&op)
}

pub(crate) fn build_segment_ranges(
    slice_points: &[usize],
    total_nodes: Option<usize>,
) -> Vec<(usize, usize)> {
    let mut points = slice_points.to_vec();
    if let Some(total) = total_nodes
        && !points.contains(&total)
    {
        points.push(total);
    }
    points.sort();
    points.dedup();

    let mut ranges = Vec::new();
    for i in 0..points.len() {
        let start = if i > 0 { points[i - 1] } else { 0 };
        let end = points[i];
        if start < end {
            ranges.push((start, end));
        }
    }
    ranges
}
