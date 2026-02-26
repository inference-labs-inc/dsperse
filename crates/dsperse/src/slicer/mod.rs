pub mod analyzer;
pub mod autotiler;
pub mod materializer;
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

pub(crate) fn build_segment_ranges(
    slice_points: &[usize],
    total_nodes: Option<usize>,
) -> Vec<(usize, usize)> {
    let mut points = slice_points.to_vec();
    if let Some(total) = total_nodes {
        if !points.contains(&total) {
            points.push(total);
        }
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
