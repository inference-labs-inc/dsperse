pub mod analyzer;
pub mod autotiler;
pub mod combiner;
pub mod materializer;
pub(crate) mod layernorm_fuse;
pub(crate) mod self_div_rewrite;
pub(crate) mod onnx_fold;
pub mod onnx_proto;
pub(crate) mod onnx_shapes;
pub mod onnx_slicer;
pub(crate) mod trace;

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

pub(crate) const CONTROL_FLOW_OPS: &[&str] = &["Loop", "If", "Scan"];

pub(crate) fn is_control_flow(op: &str) -> bool {
    CONTROL_FLOW_OPS.contains(&op)
}

pub(crate) fn collect_subgraph_outer_refs(
    node: &onnx_proto::NodeProto,
    graph: &onnx_proto::GraphProto,
) -> Vec<String> {
    let mut outer_refs = Vec::new();
    for attr in &node.attribute {
        let subgraphs: Vec<&onnx_proto::GraphProto> =
            attr.g.iter().chain(attr.graphs.iter()).collect();
        for sg in subgraphs {
            collect_outer_refs_recursive(sg, graph, &mut outer_refs);
        }
    }
    outer_refs.sort();
    outer_refs.dedup();
    outer_refs
}

fn collect_outer_refs_recursive(
    subgraph: &onnx_proto::GraphProto,
    outer_graph: &onnx_proto::GraphProto,
    outer_refs: &mut Vec<String>,
) {
    let local_names: std::collections::HashSet<String> = subgraph
        .input
        .iter()
        .map(|vi| vi.name.clone())
        .chain(subgraph.initializer.iter().map(|i| i.name.clone()))
        .chain(subgraph.node.iter().flat_map(|n| n.output.iter().cloned()))
        .collect();

    let outer_names: std::collections::HashSet<&str> = outer_graph
        .input
        .iter()
        .map(|vi| vi.name.as_str())
        .chain(outer_graph.initializer.iter().map(|i| i.name.as_str()))
        .chain(
            outer_graph
                .node
                .iter()
                .flat_map(|n| n.output.iter().map(|s| s.as_str())),
        )
        .chain(outer_graph.value_info.iter().map(|vi| vi.name.as_str()))
        .collect();

    for sg_node in &subgraph.node {
        for inp in &sg_node.input {
            if !inp.is_empty() && !local_names.contains(inp) && outer_names.contains(inp.as_str()) {
                outer_refs.push(inp.clone());
            }
        }
        for attr in &sg_node.attribute {
            let nested: Vec<&onnx_proto::GraphProto> =
                attr.g.iter().chain(attr.graphs.iter()).collect();
            for nested_sg in nested {
                collect_outer_refs_recursive(nested_sg, outer_graph, outer_refs);
            }
        }
    }
}

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
