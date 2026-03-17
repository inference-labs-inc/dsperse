use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::analyzer::{self, AnalysisResult, NodeAnalysis};
use super::autotiler;
use super::materializer;
use super::onnx_proto::{self, ModelProto};
use super::{ELEMENTWISE_OPS, SHAPE_PRESERVING_OPS};
use crate::error::{DsperseError, Result};
use crate::schema::metadata::{
    Compilation, Dependencies, ModelMetadata, SliceMetadata, SliceShapeWrapper, TensorShape,
};

pub fn slice_model(
    onnx_path: &Path,
    output_path: Option<&Path>,
    tile_size: Option<usize>,
    jstprove_ops: &[&str],
) -> Result<ModelMetadata> {
    let mut model = onnx_proto::load_model(onnx_path)?;
    onnx_proto::normalize_opset(&mut model);

    let tmp_dir = tempfile::tempdir().map_err(|e| {
        DsperseError::Slicer(format!("create tempdir for opset-normalized model: {e}"))
    })?;
    let normalized_path = tmp_dir.path().join("model.onnx");
    onnx_proto::save_model(&model, &normalized_path)?;

    tracing::info!("tracing shapes via tract");
    let traced_shapes = trace_shapes_tract(&normalized_path, &model)?;

    let analysis = analyzer::analyze(&model, Some(onnx_path))?;

    let output_dir = output_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        onnx_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("slices")
    });
    std::fs::create_dir_all(&output_dir).map_err(|e| DsperseError::io(e, &output_dir))?;

    let slice_points = determine_slice_points(&analysis, tile_size, jstprove_ops);
    tracing::info!(points = ?slice_points, "determined slice points");
    debug_assert!(
        !slice_points.is_empty(),
        "complete_slice_points guarantees at least [0, end]"
    );

    let model_dest = output_dir.join("model.onnx");
    onnx_proto::save_model(&model, &model_dest)?;

    let segment_ranges = super::build_segment_ranges(&slice_points, None);

    let mut tiled_info = HashMap::new();
    if tile_size.is_some() {
        let trimmed_points = &slice_points[..slice_points.len().saturating_sub(1)];
        for (seg_idx, _) in segment_ranges.iter().enumerate() {
            let slice_model = materializer::materialize_slice_model(
                &model,
                trimmed_points,
                &traced_shapes,
                seg_idx,
            )?;
            if let Some(detection) = autotiler::detect_tiling_needs(&slice_model, tile_size) {
                tiled_info.insert(seg_idx, detection);
            }
        }
    }

    let slices = build_slice_metadata(
        &analysis,
        &slice_points,
        &segment_ranges,
        &traced_shapes,
        &tiled_info,
    );

    let mut metadata = ModelMetadata {
        original_model: analysis.original_model.clone().unwrap_or_default(),
        model_type: analysis.model_type.clone(),
        input_shape: analysis.input_shape.clone(),
        output_shapes: analysis.output_shapes.clone(),
        output_names: analysis.output_names.clone(),
        slice_points: slice_points[..slice_points.len().saturating_sub(1)].to_vec(),
        slices,
        dsperse_version: None,
        dsperse_rev: None,
        jstprove_version: None,
        jstprove_rev: None,
        traced_shapes: Some(traced_shapes),
        original_model_path: Some("model.onnx".to_string()),
    };
    metadata.stamp_version();
    metadata.save(&output_dir.join(crate::utils::paths::METADATA_FILE))?;

    tracing::info!(
        slices = metadata.slices.len(),
        tiled = tiled_info.len(),
        "slicing complete"
    );

    Ok(metadata)
}

fn build_slice_metadata(
    analysis: &AnalysisResult,
    _slice_points: &[usize],
    segment_ranges: &[(usize, usize)],
    traced_shapes: &HashMap<String, Vec<i64>>,
    tiled_info: &HashMap<usize, autotiler::TilingDetection>,
) -> Vec<SliceMetadata> {
    let mut slices = Vec::new();

    for (seg_idx, &(start, end)) in segment_ranges.iter().enumerate() {
        let dependencies = analyzer::get_segment_dependencies(analysis, start, end);

        let shape = build_shape_from_traced(analysis, start, end, &dependencies, traced_shapes);

        let filename = format!("slice_{seg_idx}.onnx");
        let relative_path = format!("slice_{seg_idx}/payload/{filename}");

        let mut tiling = None;
        let mut channel_split = None;
        if let Some(detection) = tiled_info.get(&seg_idx) {
            match detection {
                autotiler::TilingDetection::Spatial {
                    input_name,
                    output_name,
                    c_in,
                    c_out,
                    h: _,
                    w: _,
                    tile_size: actual_tile,
                    halo,
                    tiles_y,
                    tiles_x,
                    out_tile,
                    stride,
                } => {
                    tiling = Some(crate::schema::tiling::TilingInfo {
                        slice_idx: seg_idx,
                        tile_size: *actual_tile as usize,
                        num_tiles: (*tiles_y * *tiles_x) as usize,
                        tiles_y: *tiles_y as usize,
                        tiles_x: *tiles_x as usize,
                        halo: *halo,
                        out_tile: *out_tile,
                        stride: *stride,
                        c_in: *c_in as usize,
                        c_out: *c_out as usize,
                        input_name: input_name.clone(),
                        output_name: output_name.clone(),
                        tile: Some(crate::schema::tiling::TileInfo {
                            path: format!("slice_{seg_idx}/payload/tiles/tile.onnx"),
                            conv_out: *out_tile,
                            jstprove_circuit_path: None,
                        }),
                        tiles: None,
                    });
                }
                autotiler::TilingDetection::ChannelSplit {
                    input_name,
                    output_name,
                    c_in,
                    c_out,
                    h,
                    w,
                    num_groups,
                    channels_per_group,
                } => {
                    channel_split = Some(crate::schema::tiling::ChannelSplitInfo {
                        slice_idx: seg_idx,
                        c_in: *c_in as usize,
                        c_out: *c_out as usize,
                        num_groups: *num_groups as usize,
                        channels_per_group: *channels_per_group as usize,
                        input_name: input_name.clone(),
                        output_name: output_name.clone(),
                        h: *h as usize,
                        w: *w as usize,
                        out_h: 0,
                        out_w: 0,
                        groups: Vec::new(),
                        bias_path: None,
                    });
                }
            }
        }

        slices.push(SliceMetadata {
            index: seg_idx,
            filename: filename.clone(),
            path: format!("payload/{filename}"),
            relative_path,
            shape: SliceShapeWrapper {
                tensor_shape: shape,
            },
            dependencies,
            tiling,
            channel_split,
            compilation: Compilation::default(),
            slice_metadata: None,
            slice_metadata_relative_path: None,
        });
    }

    slices
}

fn build_shape_from_traced(
    _analysis: &AnalysisResult,
    _start: usize,
    _end: usize,
    dependencies: &Dependencies,
    traced_shapes: &HashMap<String, Vec<i64>>,
) -> TensorShape {
    let input_shapes: Vec<Vec<i64>> = dependencies
        .filtered_inputs
        .iter()
        .filter_map(|name| traced_shapes.get(name).cloned())
        .collect();

    let output_shapes: Vec<Vec<i64>> = dependencies
        .output
        .iter()
        .filter_map(|name| traced_shapes.get(name).cloned())
        .collect();

    TensorShape {
        input: input_shapes,
        output: output_shapes,
    }
}

fn determine_slice_points(
    analysis: &AnalysisResult,
    tile_size: Option<usize>,
    jstprove_ops: &[&str],
) -> Vec<usize> {
    let mut points: HashSet<usize> = HashSet::new();

    for node in analysis.nodes.values() {
        if !node.parameter_details.is_empty() {
            points.insert(node.index);
        }
    }

    let mut sorted_points: Vec<usize> = points.into_iter().collect();
    sorted_points.sort();

    sorted_points = isolate_conv(&sorted_points, analysis);
    sorted_points = optimize_jstprove_slices(&sorted_points, analysis, jstprove_ops);

    if tile_size.is_some() {
        sorted_points = optimize_for_tiling(&sorted_points, analysis);
    }

    sorted_points = filter_constant_only_slices(&sorted_points, analysis);
    sorted_points.sort();
    sorted_points.dedup();

    complete_slice_points(&mut sorted_points, analysis);
    sorted_points
}

fn optimize_points(
    points: &[usize],
    analysis: &AnalysisResult,
    mutate: impl FnOnce(&mut HashSet<usize>, &[&NodeAnalysis], usize),
) -> Vec<usize> {
    let mut updated: HashSet<usize> = points.iter().copied().collect();
    let mut sorted_nodes: Vec<&NodeAnalysis> = analysis.nodes.values().collect();
    sorted_nodes.sort_by_key(|n| n.index);
    let max_idx = sorted_nodes.last().map(|n| n.index).unwrap_or(0);
    mutate(&mut updated, &sorted_nodes, max_idx);
    let mut v: Vec<usize> = updated.into_iter().filter(|&p| p <= max_idx).collect();
    v.sort();
    v
}

fn isolate_conv(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    optimize_points(points, analysis, |updated, sorted_nodes, max_idx| {
        for (pos, node) in sorted_nodes.iter().enumerate() {
            if node.node_type == "Conv" {
                updated.insert(node.index);
                let mut produced: HashSet<&str> = node
                    .dependencies
                    .output
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                let mut end = pos + 1;
                while end < sorted_nodes.len() {
                    let candidate = sorted_nodes[end];
                    if !SHAPE_PRESERVING_OPS.contains(&candidate.node_type.as_str()) {
                        break;
                    }
                    let consumes_produced = candidate.dependencies.input.iter().any(|inp| {
                        !analysis.initializer_names.contains(inp) && produced.contains(inp.as_str())
                    });
                    if !consumes_produced {
                        break;
                    }
                    for out in &candidate.dependencies.output {
                        produced.insert(out.as_str());
                    }
                    end += 1;
                }
                if end < sorted_nodes.len() && sorted_nodes[end].index <= max_idx {
                    updated.insert(sorted_nodes[end].index);
                }
            }
        }
    })
}

fn optimize_jstprove_slices(
    points: &[usize],
    analysis: &AnalysisResult,
    jstprove_ops: &[&str],
) -> Vec<usize> {
    optimize_points(points, analysis, |updated, sorted_nodes, _max_idx| {
        let is_supported = |n: &NodeAnalysis| jstprove_ops.contains(&n.node_type.as_str());
        for i in 0..sorted_nodes.len().saturating_sub(1) {
            if is_supported(sorted_nodes[i]) != is_supported(sorted_nodes[i + 1]) {
                updated.insert(sorted_nodes[i + 1].index);
            }
        }
    })
}

fn optimize_for_tiling(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    optimize_points(points, analysis, |updated, sorted_nodes, _max_idx| {
        let is_tileable = |n: &NodeAnalysis| {
            n.node_type == "Conv" || ELEMENTWISE_OPS.contains(&n.node_type.as_str())
        };
        for i in 0..sorted_nodes.len().saturating_sub(1) {
            let curr = sorted_nodes[i];
            let next = sorted_nodes[i + 1];
            if !is_tileable(curr) && next.node_type == "Relu" {
                continue;
            }
            if is_tileable(curr) != is_tileable(next) {
                updated.insert(next.index);
            }
        }
    })
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
    points
        .iter()
        .filter(|p| !to_remove.contains(p))
        .copied()
        .collect()
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

fn trace_shapes_tract(
    onnx_path: &Path,
    proto_model: &ModelProto,
) -> Result<HashMap<String, Vec<i64>>> {
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
            model
                .set_input_fact(i, InferenceFact::dt_shape(tf.datum_type, &concrete))
                .map_err(|e| {
                    DsperseError::Slicer(format!("set_input_fact({i}, shape={concrete:?}): {e}"))
                })?;
        }
    }

    let typed_result = model.into_typed();

    let mut shapes = HashMap::new();
    let mut tract_names_to_shapes: Vec<(String, Vec<i64>)> = Vec::new();

    match typed_result {
        Err(e) => {
            tracing::warn!(
                error = %e,
                "tract type inference failed; falling back to value_info shapes"
            );
            if let Some(graph) = &proto_model.graph {
                for vi in graph
                    .input
                    .iter()
                    .chain(graph.output.iter())
                    .chain(graph.value_info.iter())
                {
                    let shape = onnx_proto::vi_shape(vi);
                    if !shape.is_empty() {
                        tract_names_to_shapes.push((vi.name.clone(), shape.clone()));
                        shapes.insert(vi.name.clone(), shape);
                    }
                }
                for init in &graph.initializer {
                    let shape: Vec<i64> = init.dims.to_vec();
                    tract_names_to_shapes.push((init.name.clone(), shape.clone()));
                    shapes.insert(init.name.clone(), shape);
                }
            }
        }
        Ok(typed) => {
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
        }
    }

    if let Some(graph) = &proto_model.graph {
        for init in &graph.initializer {
            if !shapes.contains_key(&init.name) {
                let shape: Vec<i64> = init.dims.to_vec();
                shapes.insert(init.name.clone(), shape);
            }
        }

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
                    if tract_name.starts_with(&prefix)
                        && matched_shape.is_none_or(|s| shape.len() > s.len())
                    {
                        matched_shape = Some(shape);
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

        let binary_ops: HashSet<&str> = ["Add", "Sub", "Mul", "Div", "Pow", "Max", "Min"]
            .into_iter()
            .collect();

        let mut prev_len = 0;
        while shapes.len() != prev_len {
            prev_len = shapes.len();

            for node in &graph.node {
                if super::SHAPE_PRESERVING_OPS.contains(&node.op_type.as_str())
                    && let Some(inp) = node.input.first()
                    && let Some(in_shape) = shapes.get(inp).cloned()
                {
                    for out in &node.output {
                        if !out.is_empty() && !shapes.contains_key(out) {
                            shapes.insert(out.clone(), in_shape.clone());
                        }
                    }
                }
            }

            for node in &graph.node {
                if binary_ops.contains(node.op_type.as_str()) {
                    let input_shapes: Vec<&Vec<i64>> = node
                        .input
                        .iter()
                        .filter_map(|inp| shapes.get(inp))
                        .collect();
                    if let Some(broadcasted) = broadcast_shapes(&input_shapes) {
                        for out in &node.output {
                            if !out.is_empty() && !shapes.contains_key(out) {
                                shapes.insert(out.clone(), broadcasted.clone());
                            }
                        }
                    }
                }
            }

            for node in &graph.node {
                if node.op_type == "MaxPool"
                    && let Some(inp) = node.input.first()
                    && let Some(in_shape) = shapes.get(inp).cloned()
                    && in_shape.len() == 4
                {
                    let kernel =
                        onnx_proto::get_attribute_ints(node, "kernel_shape").unwrap_or_default();
                    let strides =
                        onnx_proto::get_attribute_ints(node, "strides").unwrap_or_default();
                    let pads = onnx_proto::get_attribute_ints(node, "pads").unwrap_or_default();
                    let dilations =
                        onnx_proto::get_attribute_ints(node, "dilations").unwrap_or_default();
                    let ceil_mode = onnx_proto::get_attribute_int(node, "ceil_mode").unwrap_or(0);
                    if kernel.len() >= 2 && strides.len() >= 2 && strides[0] > 0 && strides[1] > 0 {
                        let pad_h = if pads.len() >= 4 {
                            pads[0] + pads[2]
                        } else {
                            0
                        };
                        let pad_w = if pads.len() >= 4 {
                            pads[1] + pads[3]
                        } else {
                            0
                        };
                        let dil_h = dilations.first().copied().unwrap_or(1);
                        let dil_w = dilations.get(1).copied().unwrap_or(1);
                        let eff_k_h = (kernel[0] - 1) * dil_h + 1;
                        let eff_k_w = (kernel[1] - 1) * dil_w + 1;
                        let (h, w) = if ceil_mode != 0 {
                            (
                                (in_shape[2] + pad_h - eff_k_h + strides[0] - 1) / strides[0] + 1,
                                (in_shape[3] + pad_w - eff_k_w + strides[1] - 1) / strides[1] + 1,
                            )
                        } else {
                            (
                                (in_shape[2] + pad_h).saturating_sub(eff_k_h) / strides[0] + 1,
                                (in_shape[3] + pad_w).saturating_sub(eff_k_w) / strides[1] + 1,
                            )
                        };
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

        for vi in graph
            .input
            .iter()
            .chain(graph.output.iter())
            .chain(graph.value_info.iter())
        {
            if !shapes.contains_key(&vi.name) {
                let dims = onnx_proto::vi_shape(vi);
                if !dims.is_empty() {
                    let concrete: Vec<i64> =
                        dims.iter().map(|&d| if d <= 0 { 1 } else { d }).collect();
                    shapes.insert(vi.name.clone(), concrete);
                }
            }
        }
    }

    tracing::info!(tensors = shapes.len(), "shape trace complete");
    Ok(shapes)
}

fn broadcast_shapes(shapes: &[&Vec<i64>]) -> Option<Vec<i64>> {
    if shapes.is_empty() {
        return None;
    }
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = vec![1i64; max_rank];
    for shape in shapes {
        let offset = max_rank - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let ri = offset + i;
            if result[ri] == 1 {
                result[ri] = dim;
            } else if dim != 1 && dim != result[ri] {
                return None;
            }
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use analyzer::NodeDependencies;

    fn make_analysis_with_params(nodes: Vec<(&str, usize, &str, bool)>) -> AnalysisResult {
        let mut node_map = HashMap::new();
        for (name, index, op_type, has_params) in &nodes {
            let mut parameter_details = HashMap::new();
            if *has_params {
                parameter_details.insert(
                    format!("{}_weight", name),
                    analyzer::ParameterDetail {
                        shape: vec![3, 3],
                        size: 9,
                    },
                );
            }
            node_map.insert(
                name.to_string(),
                NodeAnalysis {
                    index: *index,
                    slice_name: format!("{}_{}", op_type, index),
                    node_type: op_type.to_string(),
                    parameter_details,
                    dependencies: NodeDependencies {
                        input: vec![],
                        output: vec![],
                    },
                },
            );
        }
        AnalysisResult {
            original_model: None,
            model_type: "ONNX".to_string(),
            node_count: nodes.len(),
            initializer_count: 0,
            input_shape: vec![],
            output_shapes: vec![],
            output_names: vec![],
            opset_version: Some(18),
            nodes: node_map,
            initializer_names: HashSet::new(),
        }
    }

    const TEST_OPS: &[&str] = &["Conv", "Gemm", "MatMul"];

    #[test]
    fn complete_slice_points_adds_boundaries() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "Conv", false),
            ("b", 1, "Relu", false),
            ("c", 2, "Conv", false),
        ]);
        let mut points = vec![1];
        complete_slice_points(&mut points, &analysis);
        assert!(points.contains(&0));
        assert!(points.contains(&3));
        assert!(points.contains(&1));
    }

    #[test]
    fn complete_slice_points_already_complete() {
        let analysis =
            make_analysis_with_params(vec![("a", 0, "Conv", false), ("b", 1, "Relu", false)]);
        let mut points = vec![0, 2];
        complete_slice_points(&mut points, &analysis);
        assert_eq!(points, vec![0, 2]);
    }

    #[test]
    fn complete_slice_points_deduplicates() {
        let analysis = make_analysis_with_params(vec![("a", 0, "Conv", false)]);
        let mut points = vec![0, 0, 1, 1];
        complete_slice_points(&mut points, &analysis);
        assert_eq!(points, vec![0, 1]);
    }

    #[test]
    fn isolate_conv_inserts_boundaries() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "Conv", false),
            ("b", 1, "Relu", false),
            ("c", 2, "MaxPool", false),
            ("d", 3, "Conv", false),
            ("e", 4, "Relu", false),
        ]);
        let points = vec![0, 3];
        let result = isolate_conv(&points, &analysis);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(result.contains(&3));
        assert!(result.contains(&4));
    }

    #[test]
    fn isolate_conv_no_convs() {
        let analysis =
            make_analysis_with_params(vec![("a", 0, "Relu", false), ("b", 1, "MaxPool", false)]);
        let points = vec![0];
        let result = isolate_conv(&points, &analysis);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn optimize_jstprove_slices_splits_at_boundary() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "Conv", false),
            ("b", 1, "Relu", false),
            ("c", 2, "Conv", false),
        ]);
        let points = vec![0];
        let result = optimize_jstprove_slices(&points, &analysis, TEST_OPS);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }

    #[test]
    fn optimize_jstprove_slices_all_supported() {
        let analysis =
            make_analysis_with_params(vec![("a", 0, "Conv", false), ("b", 1, "Conv", false)]);
        let points = vec![0, 1];
        let result = optimize_jstprove_slices(&points, &analysis, TEST_OPS);
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn optimize_for_tiling_splits_tileable_boundary() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "Conv", false),
            ("b", 1, "Relu", false),
            ("c", 2, "MaxPool", false),
            ("d", 3, "Conv", false),
        ]);
        let points = vec![0, 3];
        let result = optimize_for_tiling(&points, &analysis);
        assert!(result.contains(&2));
    }

    #[test]
    fn optimize_for_tiling_relu_after_non_tileable_kept() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "MaxPool", false),
            ("b", 1, "Relu", false),
            ("c", 2, "Conv", false),
        ]);
        let points = vec![0, 2];
        let result = optimize_for_tiling(&points, &analysis);
        assert!(!result.contains(&1));
    }

    #[test]
    fn filter_constant_only_slices_removes_constant_segments() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "Constant", false),
            ("b", 1, "Constant", false),
            ("c", 2, "Conv", false),
            ("d", 3, "Relu", false),
        ]);
        let points = vec![2, 4];
        let result = filter_constant_only_slices(&points, &analysis);
        assert!(!result.contains(&2));
        assert!(result.contains(&4));
    }

    #[test]
    fn filter_constant_only_slices_keeps_non_constant() {
        let analysis =
            make_analysis_with_params(vec![("a", 0, "Conv", false), ("b", 1, "Relu", false)]);
        let points = vec![1, 2];
        let result = filter_constant_only_slices(&points, &analysis);
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn filter_constant_only_slices_empty_points() {
        let analysis = make_analysis_with_params(vec![("a", 0, "Conv", false)]);
        let result = filter_constant_only_slices(&[], &analysis);
        assert!(result.is_empty());
    }

    #[test]
    fn determine_slice_points_includes_parameterized_nodes() {
        let analysis = make_analysis_with_params(vec![
            ("conv0", 0, "Conv", true),
            ("relu0", 1, "Relu", false),
            ("conv1", 2, "Conv", true),
            ("relu1", 3, "Relu", false),
        ]);
        let points = determine_slice_points(&analysis, None, TEST_OPS);
        assert!(points.contains(&0));
        assert!(points.contains(&2));
        let max = *points.last().unwrap();
        assert_eq!(max, 4);
    }

    #[test]
    fn determine_slice_points_with_tile_size() {
        let analysis = make_analysis_with_params(vec![
            ("conv0", 0, "Conv", true),
            ("relu0", 1, "Relu", false),
            ("pool", 2, "MaxPool", false),
            ("conv1", 3, "Conv", true),
        ]);
        let points = determine_slice_points(&analysis, Some(1024), TEST_OPS);
        assert!(points.contains(&0));
        assert!(points.len() >= 3);
    }
}
