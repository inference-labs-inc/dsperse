use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::analyzer::{self, AnalysisResult, NodeAnalysis, TiledResult};
use super::autotiler::{self, JSTPROVE_SUPPORTED_OPS};
use super::onnx_proto::{
    self, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
};
use super::tensor_graph::TensorGraph;
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;

pub fn slice_model(
    onnx_path: &Path,
    output_path: Option<&Path>,
    tile_size: Option<usize>,
) -> Result<ModelMetadata> {
    let model = onnx_proto::load_model(onnx_path)?;

    tracing::info!("tracing shapes via tract");
    let traced_shapes = trace_shapes_tract(onnx_path)?;

    let analysis = analyzer::analyze(&model, Some(onnx_path));

    let output_dir = output_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| {
            onnx_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("slices")
        });
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| DsperseError::io(e, &output_dir))?;

    let slice_points = determine_slice_points(&analysis, tile_size);
    tracing::info!(points = ?slice_points, "determined slice points");

    if slice_points.is_empty() {
        return Err(DsperseError::Slicer("no slice points determined".into()));
    }

    let model_with_shapes = apply_traced_shapes(model, &traced_shapes);

    let (slices_paths, tg) = slice_graph(
        &model_with_shapes,
        &analysis,
        &slice_points,
        &output_dir,
        &traced_shapes,
    )?;

    let mut tiled_info: HashMap<usize, TiledResult> = HashMap::new();
    if let Some(ts) = tile_size {
        tracing::info!(tile_size = ts, "applying tiling transform");
        tiled_info = autotiler::apply_tiling(&slices_paths, ts)?;
    }

    let metadata = analyzer::generate_slices_metadata(
        &analysis,
        &slice_points,
        &slices_paths,
        &output_dir,
        &tiled_info,
    )?;

    tracing::info!(
        slices = slices_paths.len(),
        tiled = tiled_info.len(),
        tensor_graph = %tg,
        "slicing complete"
    );

    Ok(metadata)
}

fn determine_slice_points(analysis: &AnalysisResult, tile_size: Option<usize>) -> Vec<usize> {
    let mut points: HashSet<usize> = HashSet::new();
    let max_idx = analysis
        .nodes
        .values()
        .map(|n| n.index)
        .max()
        .unwrap_or(0);

    for node in analysis.nodes.values() {
        if !node.parameter_details.is_empty() {
            points.insert(node.index);
            if node.node_type == "Conv" && node.index + 1 <= max_idx {
                points.insert(node.index + 1);
            }
        }
    }

    let mut sorted_points: Vec<usize> = points.into_iter().collect();
    sorted_points.sort();

    sorted_points = isolate_conv(&sorted_points, analysis);
    sorted_points = optimize_jstprove_slices(&sorted_points, analysis);

    if tile_size.is_some() {
        sorted_points = optimize_for_tiling(&sorted_points, analysis);
    }

    sorted_points = filter_constant_only_slices(&sorted_points, analysis);
    sorted_points.sort();
    sorted_points.dedup();

    complete_slice_points(&mut sorted_points, analysis);
    sorted_points
}

fn isolate_conv(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    let mut updated: HashSet<usize> = points.iter().copied().collect();
    let max_idx = analysis.nodes.values().map(|n| n.index).max().unwrap_or(0);

    for node in analysis.nodes.values() {
        if node.node_type == "Conv" {
            updated.insert(node.index);
            if node.index + 1 <= max_idx {
                updated.insert(node.index + 1);
            }
        }
    }
    let mut v: Vec<usize> = updated.into_iter().collect();
    v.sort();
    v
}

fn optimize_jstprove_slices(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    let mut updated: HashSet<usize> = points.iter().copied().collect();
    let mut sorted_nodes: Vec<&NodeAnalysis> = analysis.nodes.values().collect();
    sorted_nodes.sort_by_key(|n| n.index);
    let max_idx = sorted_nodes.last().map(|n| n.index).unwrap_or(0);

    let is_supported = |n: &NodeAnalysis| JSTPROVE_SUPPORTED_OPS.contains(&n.node_type.as_str());

    for i in 0..sorted_nodes.len().saturating_sub(1) {
        if is_supported(sorted_nodes[i]) != is_supported(sorted_nodes[i + 1]) {
            updated.insert(sorted_nodes[i + 1].index);
        }
    }

    let mut v: Vec<usize> = updated.into_iter().filter(|&p| p <= max_idx).collect();
    v.sort();
    v
}

fn optimize_for_tiling(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    let elementwise_ops: HashSet<&str> = [
        "Sigmoid", "Mul", "Add", "Sub", "Div", "Relu", "LeakyRelu", "PRelu",
        "Tanh", "Clip", "Neg", "Abs", "Sqrt", "Exp", "Log", "Pow", "Sin", "Cos",
    ]
    .into_iter()
    .collect();

    let mut updated: HashSet<usize> = points.iter().copied().collect();
    let mut sorted_nodes: Vec<&NodeAnalysis> = analysis.nodes.values().collect();
    sorted_nodes.sort_by_key(|n| n.index);
    let max_idx = sorted_nodes.last().map(|n| n.index).unwrap_or(0);

    let is_tileable = |n: &NodeAnalysis| {
        n.node_type == "Conv" || elementwise_ops.contains(n.node_type.as_str())
    };

    let mut skip_next = false;
    for i in 0..sorted_nodes.len().saturating_sub(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        let curr = sorted_nodes[i];
        let next = sorted_nodes[i + 1];
        if !is_tileable(curr) && next.node_type == "Relu" {
            skip_next = true;
            continue;
        }
        if is_tileable(curr) != is_tileable(next) {
            updated.insert(next.index);
        }
    }

    let mut v: Vec<usize> = updated.into_iter().filter(|&p| p <= max_idx).collect();
    v.sort();
    v
}

fn filter_constant_only_slices(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    if points.is_empty() {
        return points.to_vec();
    }
    let nodes_by_idx: HashMap<usize, &NodeAnalysis> =
        analysis.nodes.values().map(|n| (n.index, n)).collect();

    let mut to_remove: HashSet<usize> = HashSet::new();
    for (i, &end_idx) in points.iter().enumerate() {
        let start_idx = if i > 0 { points[i - 1] } else { 0 };
        if start_idx == end_idx {
            continue;
        }
        let all_constant = (start_idx..end_idx).all(|idx| {
            nodes_by_idx
                .get(&idx)
                .map(|n| n.node_type == "Constant")
                .unwrap_or(true)
        });
        if all_constant {
            to_remove.insert(end_idx);
        }
    }
    if !to_remove.is_empty() {
        tracing::info!(count = to_remove.len(), "merged constant-only slices");
    }
    points.iter().filter(|p| !to_remove.contains(p)).copied().collect()
}

fn complete_slice_points(points: &mut Vec<usize>, analysis: &AnalysisResult) {
    let max_index = analysis.nodes.values().map(|n| n.index).max().unwrap_or(0);
    let end = max_index + 1;
    if !points.contains(&0) {
        points.push(0);
    }
    if !points.contains(&end) {
        points.push(end);
    }
    points.sort();
    points.dedup();
}

fn trace_shapes_tract(onnx_path: &Path) -> Result<HashMap<String, Vec<i64>>> {
    use tract_onnx::prelude::*;

    let mut model = tract_onnx::onnx()
        .model_for_path(onnx_path)
        .map_err(|e| DsperseError::Slicer(format!("tract load: {e}")))?;

    for i in 0..model.inputs.len() {
        let input_fact = model
            .input_fact(i)
            .map_err(|e| DsperseError::Slicer(format!("input fact {i}: {e}")))?
            .clone();
        if let Ok(tf) = input_fact.to_typed_fact() {
            let concrete: Vec<usize> = tf
                .shape
                .iter()
                .map(|d| d.to_i64().unwrap_or(1) as usize)
                .collect();
            let _ = model.set_input_fact(
                i,
                InferenceFact::dt_shape(tf.datum_type, &concrete),
            );
        }
    }

    let typed = model
        .into_typed()
        .map_err(|e| DsperseError::Slicer(format!("tract type inference: {e}")))?;

    let mut shapes = HashMap::new();
    let mut tract_names_to_shapes: Vec<(String, Vec<i64>)> = Vec::new();

    for node_id in 0..typed.nodes().len() {
        let node_obj = typed.node(node_id);
        for (ix, outlet) in node_obj.outputs.iter().enumerate() {
            let fact = &outlet.fact;
            if let Some(shape) = fact.shape.as_concrete() {
                let shape_vec: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                let name = if ix == 0 && !node_obj.name.is_empty() {
                    node_obj.name.clone()
                } else {
                    format!("{}:{}", node_obj.name, ix)
                };
                tract_names_to_shapes.push((name.clone(), shape_vec.clone()));
                shapes.insert(name, shape_vec);
            }
        }
    }

    let proto_model = onnx_proto::load_model(onnx_path)?;
    if let Some(graph) = &proto_model.graph {
        let onnx_node_outputs: Vec<(String, Vec<String>)> = graph
            .node
            .iter()
            .enumerate()
            .map(|(idx, n)| {
                let name = if n.name.is_empty() {
                    format!("{}_{}", n.op_type, idx)
                } else {
                    n.name.clone()
                };
                (name, n.output.clone())
            })
            .collect();

        for (onnx_name, onnx_outputs) in &onnx_node_outputs {
            let mut matched_shape: Option<&Vec<i64>> = None;
            for (tract_name, shape) in &tract_names_to_shapes {
                if tract_name == onnx_name {
                    matched_shape = Some(shape);
                    break;
                }
            }
            if matched_shape.is_none() {
                let prefix = format!("{onnx_name}.");
                for (tract_name, shape) in &tract_names_to_shapes {
                    if tract_name.starts_with(&prefix) {
                        if matched_shape.map_or(true, |s| shape.len() > s.len()) {
                            matched_shape = Some(shape);
                        }
                    }
                }
            }
            if let Some(shape) = matched_shape {
                for out in onnx_outputs {
                    if !out.is_empty() && !shapes.contains_key(out) {
                        shapes.insert(out.clone(), shape.clone());
                    }
                }
            }
        }

        let shape_preserving: HashSet<&str> = [
            "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh", "Clip", "Neg",
            "Abs", "Sqrt", "Exp", "Log", "Sin", "Cos", "BatchNormalization",
            "Dropout", "Identity",
        ].into_iter().collect();
        for node in &graph.node {
            if shape_preserving.contains(node.op_type.as_str()) {
                if let Some(inp) = node.input.first() {
                    if let Some(in_shape) = shapes.get(inp).cloned() {
                        for out in &node.output {
                            if !out.is_empty() {
                                shapes.insert(out.clone(), in_shape.clone());
                            }
                        }
                    }
                }
            }
        }

        let binary_ops: HashSet<&str> = [
            "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min",
        ].into_iter().collect();
        for node in &graph.node {
            if binary_ops.contains(node.op_type.as_str()) {
                let best = node.input.iter()
                    .filter_map(|inp| shapes.get(inp))
                    .max_by_key(|s| s.len())
                    .cloned();
                if let Some(s) = best {
                    for out in &node.output {
                        if !out.is_empty() && !shapes.contains_key(out) {
                            shapes.insert(out.clone(), s.clone());
                        }
                    }
                }
            }
        }

        for node in &graph.node {
            if node.op_type == "MaxPool" {
                if let Some(inp) = node.input.first() {
                    if let Some(in_shape) = shapes.get(inp).cloned() {
                        if in_shape.len() == 4 {
                            let kernel = onnx_proto::get_attribute_ints(node, "kernel_shape").unwrap_or_default();
                            let strides = onnx_proto::get_attribute_ints(node, "strides").unwrap_or_default();
                            let pads = onnx_proto::get_attribute_ints(node, "pads").unwrap_or_default();
                            if kernel.len() >= 2 && strides.len() >= 2 {
                                let pad_h = if pads.len() >= 4 { pads[0] + pads[2] } else { 0 };
                                let pad_w = if pads.len() >= 4 { pads[1] + pads[3] } else { 0 };
                                let h = ((in_shape[2] + pad_h).saturating_sub(kernel[0])) / strides[0] + 1;
                                let w = ((in_shape[3] + pad_w).saturating_sub(kernel[1])) / strides[1] + 1;
                                let out_shape = vec![in_shape[0], in_shape[1], h, w];
                                for out in &node.output {
                                    if !out.is_empty() && !shapes.contains_key(out) {
                                        shapes.insert(out.clone(), out_shape.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for vi in graph.input.iter().chain(graph.output.iter()).chain(graph.value_info.iter()) {
            if !shapes.contains_key(&vi.name) {
                let dims = onnx_proto::vi_shape(vi);
                if !dims.is_empty() {
                    let concrete: Vec<i64> = dims.iter().map(|&d| if d == 0 { 1 } else { d }).collect();
                    shapes.insert(vi.name.clone(), concrete);
                }
            }
        }
    }

    tracing::info!(tensors = shapes.len(), "shape trace complete");
    Ok(shapes)
}

fn apply_traced_shapes(
    mut model: ModelProto,
    shapes: &HashMap<String, Vec<i64>>,
) -> ModelProto {
    fn set_shape(vi: &mut ValueInfoProto, shape: &[i64]) {
        if let Some(ref mut tp) = vi.r#type {
            if let Some(onnx_proto::onnx::type_proto::Value::TensorType(ref mut tt)) = tp.value {
                tt.shape = Some(onnx_proto::onnx::TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&d| onnx_proto::onnx::tensor_shape_proto::Dimension {
                            denotation: String::new(),
                            value: Some(
                                onnx_proto::onnx::tensor_shape_proto::dimension::Value::DimValue(d),
                            ),
                        })
                        .collect(),
                });
            }
        }
    }

    if let Some(ref mut graph) = model.graph {
        for inp in &mut graph.input {
            if let Some(shape) = shapes.get(&inp.name) {
                set_shape(inp, shape);
            }
        }
        for out in &mut graph.output {
            if let Some(shape) = shapes.get(&out.name) {
                set_shape(out, shape);
            }
        }
        for vi in &mut graph.value_info {
            if let Some(shape) = shapes.get(&vi.name) {
                set_shape(vi, shape);
            }
        }

        let existing: HashSet<String> = graph
            .input
            .iter()
            .chain(graph.output.iter())
            .chain(graph.value_info.iter())
            .map(|vi| vi.name.clone())
            .collect();

        let init_types: HashMap<&str, i32> = graph
            .initializer
            .iter()
            .map(|i| (i.name.as_str(), i.data_type))
            .collect();

        let mut node_output_types: HashMap<String, i32> = HashMap::new();
        for node in &graph.node {
            match node.op_type.as_str() {
                "Cast" => {
                    if let Some(to) = onnx_proto::get_attribute_int(node, "to") {
                        for out in &node.output {
                            if !out.is_empty() {
                                node_output_types.insert(out.clone(), to as i32);
                            }
                        }
                    }
                }
                "MaxPool" => {
                    if node.output.len() > 1 {
                        if let Some(idx_out) = node.output.get(1) {
                            if !idx_out.is_empty() {
                                node_output_types.insert(idx_out.clone(), TensorProto::INT64);
                            }
                        }
                    }
                }
                "Shape" | "NonZero" | "ArgMax" | "ArgMin" => {
                    for out in &node.output {
                        if !out.is_empty() {
                            node_output_types.insert(out.clone(), TensorProto::INT64);
                        }
                    }
                }
                _ => {}
            }
        }

        for (name, shape) in shapes {
            if !existing.contains(name) {
                let elem_type = init_types
                    .get(name.as_str())
                    .copied()
                    .or_else(|| node_output_types.get(name).copied())
                    .unwrap_or(TensorProto::FLOAT);
                graph.value_info.push(onnx_proto::make_tensor_value_info(
                    name,
                    elem_type,
                    shape,
                ));
            }
        }
    }
    model
}

fn slice_graph(
    model: &ModelProto,
    _analysis: &AnalysisResult,
    slice_points: &[usize],
    output_dir: &Path,
    traced_shapes: &HashMap<String, Vec<i64>>,
) -> Result<(HashMap<usize, String>, TensorGraph)> {
    let graph = model.graph.as_ref().unwrap();
    let tensor_graph = TensorGraph::new(graph);

    let init_map: HashMap<&str, &TensorProto> = graph
        .initializer
        .iter()
        .map(|i| (i.name.as_str(), i))
        .collect();

    let vi_map = onnx_proto::build_value_info_map(graph);

    let segment_ranges = build_segment_ranges(slice_points);

    let future_deps = compute_future_dependencies(graph, &segment_ranges, &init_map);

    let opset_version = model
        .opset_import
        .first()
        .map(|o| o.version)
        .unwrap_or(13);

    let mut slices_paths = HashMap::new();
    let mut failures = Vec::new();

    for (seg_idx, &(start, end)) in segment_ranges.iter().enumerate() {
        let nodes: Vec<NodeProto> = graph.node[start..end].to_vec();
        if nodes.is_empty() {
            continue;
        }

        let seg_outputs: HashSet<String> = nodes
            .iter()
            .flat_map(|n| n.output.iter().cloned())
            .collect();

        let seg_inputs_set: HashSet<String> = nodes
            .iter()
            .flat_map(|n| n.input.iter().filter(|s| !s.is_empty()).cloned())
            .collect();

        let future = future_deps.get(&seg_idx).cloned().unwrap_or_default();

        let (inputs, outputs, initializers) = get_segment_details(
            &nodes,
            graph,
            &init_map,
            &vi_map,
            &seg_outputs,
            &seg_inputs_set,
            &future,
            traced_shapes,
        );

        let seg_graph = onnx_proto::make_graph(
            &format!("segment_{seg_idx}_graph"),
            nodes,
            inputs,
            outputs,
            initializers,
        );
        let seg_model = onnx_proto::make_model(seg_graph, opset_version);

        let save_path = output_dir.join(format!("slice_{seg_idx}"));
        let payload_dir = save_path.join("payload");
        std::fs::create_dir_all(&payload_dir)
            .map_err(|e| DsperseError::io(e, &payload_dir))?;
        let file_path = payload_dir.join(format!("slice_{seg_idx}.onnx"));

        match onnx_proto::save_model(&seg_model, &file_path) {
            Ok(()) => {
                tracing::info!(slice = seg_idx, "built slice");
                slices_paths
                    .insert(seg_idx, file_path.to_string_lossy().to_string());
            }
            Err(e) => {
                tracing::error!(slice = seg_idx, err = %e, "failed to build slice");
                failures.push((seg_idx, e));
            }
        }
    }

    if !failures.is_empty() {
        let indices: Vec<usize> = failures.iter().map(|(i, _)| *i).collect();
        return Err(DsperseError::Slicer(format!(
            "failed to build {} slice(s): {:?}",
            failures.len(),
            indices
        )));
    }

    Ok((slices_paths, tensor_graph))
}

fn build_segment_ranges(slice_points: &[usize]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for i in 0..slice_points.len() {
        let start = if i > 0 { slice_points[i - 1] } else { 0 };
        let end = slice_points[i];
        if start < end {
            ranges.push((start, end));
        }
    }
    ranges
}

fn compute_future_dependencies(
    graph: &GraphProto,
    segment_ranges: &[(usize, usize)],
    init_map: &HashMap<&str, &TensorProto>,
) -> HashMap<usize, HashSet<String>> {
    let mut seg_inputs: HashMap<usize, HashSet<String>> = HashMap::new();

    for (seg_idx, &(start, end)) in segment_ranges.iter().enumerate() {
        let seg_outputs: HashSet<String> = graph.node[start..end]
            .iter()
            .flat_map(|n| n.output.iter().cloned())
            .collect();

        let inputs: HashSet<String> = graph.node[start..end]
            .iter()
            .flat_map(|n| n.input.iter())
            .filter(|inp| !inp.is_empty() && !seg_outputs.contains(inp.as_str()) && !init_map.contains_key(inp.as_str()))
            .cloned()
            .collect();

        seg_inputs.insert(seg_idx, inputs);
    }

    let mut future: HashMap<usize, HashSet<String>> = HashMap::new();
    for seg_idx in 0..segment_ranges.len() {
        let mut deps = HashSet::new();
        for future_idx in (seg_idx + 1)..segment_ranges.len() {
            if let Some(inputs) = seg_inputs.get(&future_idx) {
                deps.extend(inputs.iter().cloned());
            }
        }
        future.insert(seg_idx, deps);
    }
    future
}

#[allow(clippy::too_many_arguments)]
fn get_segment_details(
    nodes: &[NodeProto],
    graph: &GraphProto,
    init_map: &HashMap<&str, &TensorProto>,
    vi_map: &HashMap<String, &ValueInfoProto>,
    seg_outputs: &HashSet<String>,
    seg_inputs_set: &HashSet<String>,
    future_inputs: &HashSet<String>,
    traced_shapes: &HashMap<String, Vec<i64>>,
) -> (Vec<ValueInfoProto>, Vec<ValueInfoProto>, Vec<TensorProto>) {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut initializers = Vec::new();

    let constant_producers: HashMap<String, &TensorProto> = graph
        .node
        .iter()
        .filter(|n| n.op_type == "Constant")
        .flat_map(|n| {
            n.output.iter().filter_map(|out| {
                n.attribute
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.t.as_ref())
                    .map(|t| (out.clone(), t))
            })
        })
        .collect();

    let model_output_names: HashSet<String> =
        graph.output.iter().map(|o| o.name.clone()).collect();

    let mut added_inputs: HashSet<String> = HashSet::new();
    for inp_name in seg_inputs_set {
        if seg_outputs.contains(inp_name) {
            continue;
        }
        if init_map.contains_key(inp_name.as_str()) {
            initializers.push((*init_map[inp_name.as_str()]).clone());
        } else if constant_producers.contains_key(inp_name) {
            let mut tensor = constant_producers[inp_name].clone();
            tensor.name = inp_name.clone();
            initializers.push(tensor);
        } else if !added_inputs.contains(inp_name) {
            if let Some(vi) = vi_map.get(inp_name) {
                inputs.push((*vi).clone());
            } else {
                let shape = traced_shapes
                    .get(inp_name)
                    .cloned()
                    .unwrap_or_else(|| {
                        tracing::warn!(tensor = %inp_name, "using fallback shape [1] for segment input");
                        vec![1]
                    });
                inputs.push(onnx_proto::make_tensor_value_info(
                    inp_name,
                    TensorProto::FLOAT,
                    &shape,
                ));
            }
            added_inputs.insert(inp_name.clone());
        }
    }

    for out_name in seg_outputs {
        let consumed_internally = nodes.iter().any(|n| n.input.contains(out_name));
        let needed_externally =
            future_inputs.contains(out_name) || model_output_names.contains(out_name);

        if !consumed_internally || needed_externally {
            if let Some(vi) = vi_map.get(out_name) {
                outputs.push((*vi).clone());
            } else {
                let shape = traced_shapes
                    .get(out_name)
                    .cloned()
                    .unwrap_or_else(|| {
                        tracing::warn!(tensor = %out_name, "using fallback shape [1] for segment output");
                        vec![1]
                    });
                outputs.push(onnx_proto::make_tensor_value_info(
                    out_name,
                    TensorProto::FLOAT,
                    &shape,
                ));
            }
        }
    }

    (inputs, outputs, initializers)
}
