use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::analyzer::{self, AnalysisResult, NodeAnalysis};
use super::autotiler;
use super::materializer;
use super::onnx_proto::{self, GraphProto, ModelProto, NodeProto};
use crate::error::{DsperseError, Result};
use crate::schema::metadata::{
    Dependencies, ModelMetadata, SliceMetadata, SliceShapeWrapper, TensorShape,
};
use crate::schema::tiling::DimSplitInfo;

pub fn slice_model(
    onnx_path: &Path,
    output_path: Option<&Path>,
    tile_size: Option<usize>,
    jstprove_ops: &[&str],
    input_shape: Option<&[i64]>,
) -> Result<ModelMetadata> {
    let mut model = onnx_proto::load_model(onnx_path)?;
    onnx_proto::normalize_opset(&mut model);
    onnx_proto::normalize_resize_modes(&mut model);
    onnx_proto::resolve_dynamic_input_shapes(&mut model, input_shape)?;

    onnx_proto::strip_symbolic_value_info(&mut model);
    onnx_proto::fold_constant_nodes(&mut model);

    let tmp_dir = tempfile::tempdir().map_err(|e| DsperseError::io(e, onnx_path))?;
    let tract_path = tmp_dir.path().join("tract_model.onnx");
    onnx_proto::save_model(&model, &tract_path)?;

    tracing::info!("folding constants and tracing shapes via tract");
    let (folded_constants, mut traced_shapes) = fold_and_trace_via_tract(&tract_path, &mut model)?;

    if let Some(graph) = &model.graph {
        super::const_prop::fill_shapes_from_graph(graph, &mut traced_shapes);
    }

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

    let trimmed_points = &slice_points[..slice_points.len().saturating_sub(1)];

    let mut tiled_info = HashMap::new();
    let mut dim_split_info: HashMap<usize, autotiler::DimSplitDetection> = HashMap::new();
    for (seg_idx, _) in segment_ranges.iter().enumerate() {
        let slice_model =
            materializer::materialize_slice_model(&model, trimmed_points, &traced_shapes, seg_idx)?;
        if let Some(detection) = autotiler::detect_tiling_needs(&slice_model, tile_size) {
            tiled_info.insert(seg_idx, detection);
            continue;
        }
        if let Some(graph) = slice_model.graph.as_ref() {
            let init_names: HashSet<String> =
                graph.initializer.iter().map(|t| t.name.clone()).collect();
            let mut slice_shapes: HashMap<String, Vec<i64>> = HashMap::new();
            for vi in graph
                .input
                .iter()
                .chain(graph.output.iter())
                .chain(graph.value_info.iter())
            {
                let dims = onnx_proto::vi_shape(vi);
                if !dims.is_empty() {
                    slice_shapes.insert(vi.name.clone(), dims);
                }
            }
            for init in &graph.initializer {
                slice_shapes
                    .entry(init.name.clone())
                    .or_insert_with(|| init.dims.to_vec());
            }
            for (name, shape) in &traced_shapes {
                slice_shapes
                    .entry(name.clone())
                    .or_insert_with(|| shape.clone());
            }
            if let Some(detection) =
                autotiler::detect_dim_split(&graph.node, &slice_shapes, &init_names)
            {
                tracing::info!(
                    slice = seg_idx,
                    estimated = detection.estimated_constraints,
                    num_groups = detection.num_groups,
                    "dim-split candidate detected"
                );
                tracing::warn!(
                    slice = seg_idx,
                    "dim-split detected but compilation support pending"
                );
                dim_split_info.insert(seg_idx, detection);
            }
        }
    }

    let slices = build_slice_metadata(
        &analysis,
        &slice_points,
        &segment_ranges,
        &traced_shapes,
        &tiled_info,
        &dim_split_info,
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
        folded_constant_names: folded_constants.into_iter().collect(),
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
    dim_split_info: &HashMap<usize, autotiler::DimSplitDetection>,
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
                    input_names,
                    ndim,
                    c_in,
                    c_out,
                    h,
                    w,
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
                        input_names: input_names.clone(),
                        ndim: *ndim as usize,
                        h: *h as usize,
                        w: *w as usize,
                        tile: Some(crate::schema::tiling::TileInfo {
                            path: format!("slice_{seg_idx}/payload/tiles/tile.onnx"),
                            conv_out: *out_tile,
                            jstprove_circuit_path: None,
                        }),
                        tiles: None,
                        segment_size: None,
                        total_elements: None,
                        original_shape: vec![],
                    });
                }
                autotiler::TilingDetection::FixedSegment {
                    input_name,
                    output_name,
                    input_names,
                    total_elements,
                    segment_size,
                    num_segments,
                    original_shape,
                } => {
                    tiling = Some(crate::schema::tiling::TilingInfo {
                        slice_idx: seg_idx,
                        tile_size: *segment_size as usize,
                        num_tiles: *num_segments as usize,
                        tiles_y: *num_segments as usize,
                        tiles_x: 1,
                        halo: [0, 0, 0, 0],
                        out_tile: [*segment_size, 1],
                        stride: [1, 1],
                        c_in: 1,
                        c_out: 1,
                        input_name: input_name.clone(),
                        output_name: output_name.clone(),
                        input_names: input_names.clone(),
                        ndim: 1,
                        h: *total_elements as usize,
                        w: 1,
                        tile: Some(crate::schema::tiling::TileInfo {
                            path: format!("slice_{seg_idx}/payload/tiles/tile.onnx"),
                            conv_out: [*segment_size, 1],
                            jstprove_circuit_path: None,
                        }),
                        tiles: None,
                        segment_size: Some(*segment_size as usize),
                        total_elements: Some(*total_elements as usize),
                        original_shape: original_shape.clone(),
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

        let dim_split = dim_split_info.get(&seg_idx).map(|d| DimSplitInfo {
            slice_idx: seg_idx,
            split_kind: d.split_kind.clone(),
            split_dim: d.split_dim,
            dim_size: d.dim_size,
            num_groups: d.num_groups,
            elements_per_group: d.elements_per_group,
            input_name: d.input_name.clone(),
            output_name: d.output_name.clone(),
            concat_axis: d.concat_axis,
            groups: Vec::new(),
        });

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
            dim_split,
            compilation: Default::default(),
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
    sorted_points = merge_control_flow_segments(&sorted_points, analysis);
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

fn is_spatial_primary(op: &str) -> bool {
    op == "Conv" || op == "MaxPool"
}

fn isolate_conv(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    optimize_points(points, analysis, |updated, sorted_nodes, max_idx| {
        for (pos, node) in sorted_nodes.iter().enumerate() {
            if is_spatial_primary(&node.node_type) {
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
                    if !super::is_shape_preserving(&candidate.node_type) {
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
            n.node_type == "Conv" || n.node_type == "MaxPool" || super::is_elementwise(&n.node_type)
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

fn merge_control_flow_segments(points: &[usize], analysis: &AnalysisResult) -> Vec<usize> {
    let output_to_node_idx: HashMap<&str, usize> = analysis
        .nodes
        .values()
        .flat_map(|n| {
            n.dependencies
                .output
                .iter()
                .map(move |o| (o.as_str(), n.index))
        })
        .collect();

    let mut to_remove: HashSet<usize> = HashSet::new();
    for node in analysis.nodes.values() {
        if !super::is_control_flow(&node.node_type) {
            continue;
        }
        for inp in &node.dependencies.input {
            if let Some(&producer_idx) = output_to_node_idx.get(inp.as_str()) {
                for &pt in points {
                    if pt > producer_idx && pt <= node.index {
                        to_remove.insert(pt);
                    }
                }
            }
        }
    }

    if !to_remove.is_empty() {
        tracing::info!(
            count = to_remove.len(),
            "removed slice points to preserve control flow node dependencies"
        );
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

#[allow(clippy::type_complexity)]
fn fold_and_trace_via_tract(
    onnx_path: &Path,
    model: &mut ModelProto,
) -> Result<(HashSet<String>, HashMap<String, Vec<i64>>)> {
    use tract_onnx::prelude::*;

    let tract_model = tract_onnx::onnx()
        .model_for_path(onnx_path)
        .map_err(|e| DsperseError::Slicer(format!("tract load: {e}")))?;

    let mut values: HashMap<usize, TVec<TValue>> = HashMap::new();
    let mut folded_names = HashSet::new();

    for node_id in 0..tract_model.nodes().len() {
        let node = tract_model.node(node_id);
        if node.inputs.is_empty() {
            if node.op.is_stateless() {
                if let Ok(outputs) = node.op.eval(tvec![]) {
                    values.insert(node_id, outputs);
                }
            } else if node.op.name().as_ref() == "Source"
                && let Ok(fact) = tract_model.outlet_fact(OutletId::new(node_id, 0))
                && let Ok(tf) = fact.to_typed_fact()
                && tf.shape.is_concrete()
            {
                let shape: Vec<usize> = tf
                    .shape
                    .iter()
                    .map(|d| d.to_i64().unwrap_or(1) as usize)
                    .collect();
                let tensor = match Tensor::zero_dt(tf.datum_type, &shape) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                values.insert(node_id, tvec![tensor.into_tvalue()]);
            }
        }
    }

    if let Ok(eval_order) = tract_model.eval_order() {
        for &node_id in &eval_order {
            if values.contains_key(&node_id) {
                continue;
            }
            let node = tract_model.node(node_id);
            if !node.op.is_stateless() || node.inputs.is_empty() {
                continue;
            }
            if !node.inputs.iter().all(|inp| values.contains_key(&inp.node)) {
                continue;
            }
            let inputs: TVec<TValue> = node
                .inputs
                .iter()
                .filter_map(|inp| {
                    values
                        .get(&inp.node)
                        .and_then(|outs| outs.get(inp.slot).cloned())
                })
                .collect();
            if inputs.len() != node.inputs.len() {
                continue;
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| node.op.eval(inputs))) {
                Ok(Ok(outputs)) => {
                    let is_shape_tensor = outputs.iter().all(|t| t.len() <= 1024);
                    if is_shape_tensor {
                        values.insert(node_id, outputs);
                    }
                }
                Ok(Err(e)) => {
                    tracing::debug!(node = %node.name, error = %e, "eval error");
                }
                Err(_) => {
                    tracing::debug!(node = %node.name, "eval panic");
                }
            }
        }
    }

    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| DsperseError::Slicer("model has no graph".into()))?;
    let existing_inits: HashSet<String> =
        graph.initializer.iter().map(|i| i.name.clone()).collect();

    let onnx_node_outputs: HashMap<&str, &[String]> = graph
        .node
        .iter()
        .map(|n| (n.name.as_str(), n.output.as_slice()))
        .collect();

    for (&node_id, outputs) in &values {
        let tract_node = tract_model.node(node_id);
        for (slot, tv) in outputs.iter().enumerate() {
            let outlet = OutletId::new(node_id, slot);
            let label = tract_model.outlet_label(outlet).map(String::from);
            let onnx_name = onnx_node_outputs
                .get(tract_node.name.as_str())
                .and_then(|outs| outs.get(slot))
                .cloned();
            let name = match label.or(onnx_name) {
                Some(n) if !n.is_empty() && !existing_inits.contains(&n) => n,
                _ => continue,
            };
            let tensor = tv.clone().into_tensor();
            if tensor.datum_type() == DatumType::TDim {
                let view = unsafe { tensor.as_slice_unchecked::<TDim>() };
                let i64_vals: Vec<i64> = view.iter().map(|d| d.to_i64().unwrap_or(0)).collect();
                let dims: Vec<i64> = tensor.shape().iter().map(|&d| d as i64).collect();
                graph.initializer.push(onnx_proto::TensorProto {
                    name: name.clone(),
                    data_type: onnx_proto::TensorProto::INT64,
                    dims,
                    int64_data: i64_vals,
                    ..Default::default()
                });
                folded_names.insert(name);
                continue;
            }
            let dims: Vec<i64> = tensor.shape().iter().map(|&d| d as i64).collect();
            let data_type = match tensor.datum_type() {
                DatumType::F32 => onnx_proto::TensorProto::FLOAT,
                DatumType::I64 => onnx_proto::TensorProto::INT64,
                DatumType::I32 => onnx_proto::TensorProto::INT32,
                DatumType::F64 => onnx_proto::TensorProto::DOUBLE,
                DatumType::Bool => onnx_proto::TensorProto::BOOL,
                _ => continue,
            };
            graph.initializer.push(onnx_proto::TensorProto {
                name: name.clone(),
                data_type,
                dims,
                raw_data: tensor.as_bytes().to_vec(),
                ..Default::default()
            });
            folded_names.insert(name);
        }
    }

    let init_names: HashSet<String> = graph.initializer.iter().map(|i| i.name.clone()).collect();
    graph.node.retain(|n| {
        !n.output
            .iter()
            .filter(|o| !o.is_empty())
            .all(|o| init_names.contains(o))
    });
    tracing::info!(
        folded = folded_names.len(),
        evaluated = values.len(),
        "constant folding complete"
    );

    let folded_path = onnx_path.with_extension("folded.onnx");
    onnx_proto::save_model(model, &folded_path)?;

    let mut tract2 = tract_onnx::onnx()
        .model_for_path(&folded_path)
        .map_err(|e| DsperseError::Slicer(format!("tract reload after folding: {e}")))?;

    for i in 0..tract2.inputs.len() {
        if let Ok(fact) = tract2.input_fact(i).cloned()
            && let Ok(tf) = fact.to_typed_fact()
        {
            let concrete: Vec<usize> = tf
                .shape
                .iter()
                .map(|d| d.to_i64().unwrap_or(1) as usize)
                .collect();
            let _ = tract2.set_input_fact(i, InferenceFact::dt_shape(tf.datum_type, &concrete));
        }
    }

    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();

    match tract2.into_typed() {
        Ok(typed) => {
            for node_id in 0..typed.nodes().len() {
                let node = typed.node(node_id);
                for (ix, outlet) in node.outputs.iter().enumerate() {
                    if let Some(shape) = outlet.fact.shape.as_concrete() {
                        let v: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                        let name = if ix == 0 && !node.name.is_empty() {
                            node.name.clone()
                        } else {
                            format!("{}:{}", node.name, ix)
                        };
                        shapes.insert(name, v.clone());
                        let outlet_id = OutletId::new(node_id, ix);
                        if let Some(label) = typed.outlet_label(outlet_id) {
                            shapes.entry(label.to_string()).or_insert(v);
                        }
                    }
                }
            }
            tracing::info!(
                tensors = shapes.len(),
                "typed model shape extraction succeeded"
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "into_typed failed after constant folding");
        }
    }

    {
        use tract_onnx::tract_hir::infer::Factoid;
        let mut tract3 = tract_onnx::onnx()
            .model_for_path(&folded_path)
            .map_err(|e| DsperseError::Slicer(format!("tract reload for analysis: {e}")))?;
        for i in 0..tract3.inputs.len() {
            if let Ok(fact) = tract3.input_fact(i).cloned()
                && let Ok(tf) = fact.to_typed_fact()
            {
                let concrete: Vec<usize> = tf
                    .shape
                    .iter()
                    .map(|d| d.to_i64().unwrap_or(1) as usize)
                    .collect();
                let _ = tract3.set_input_fact(i, InferenceFact::dt_shape(tf.datum_type, &concrete));
            }
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tract3.analyse(true)));
        for node_id in 0..tract3.nodes().len() {
            let node = tract3.node(node_id);
            for (ix, outlet) in node.outputs.iter().enumerate() {
                if let Some(dims) = outlet.fact.shape.concretize() {
                    let v: Vec<i64> = dims.iter().map(|d| d.to_i64().unwrap_or(0)).collect();
                    if v.iter().all(|&d| d > 0) {
                        let outlet_id = OutletId::new(node_id, ix);
                        if let Some(label) = tract3.outlet_label(outlet_id) {
                            shapes.entry(label.to_string()).or_insert(v.clone());
                        }
                        let name = if ix == 0 && !node.name.is_empty() {
                            node.name.clone()
                        } else {
                            format!("{}:{}", node.name, ix)
                        };
                        shapes.entry(name).or_insert(v);
                    }
                }
            }
        }
        tracing::info!(tensors = shapes.len(), "after obstinate analysis");
    }

    if let Some(graph) = &model.graph {
        let mut extra: Vec<(String, Vec<i64>)> = Vec::new();
        for n in &graph.node {
            if let Some(shape) = shapes.get(&n.name) {
                for out in &n.output {
                    if !out.is_empty() && !shapes.contains_key(out) {
                        extra.push((out.clone(), shape.clone()));
                    }
                }
            }
        }
        for (name, shape) in extra {
            shapes.insert(name, shape);
        }
        for init in &graph.initializer {
            if !init.dims.is_empty() {
                shapes
                    .entry(init.name.clone())
                    .or_insert_with(|| init.dims.clone());
            }
        }
    }

    tracing::info!(tensors = shapes.len(), "shape trace complete");
    Ok((folded_names, shapes))
}

#[allow(dead_code)]
fn conv_pool_spatial(
    in_shape: &[i64],
    kernel: &[i64],
    node: &NodeProto,
    ceil_mode: bool,
) -> Option<Vec<i64>> {
    if in_shape.len() != 4 || kernel.len() < 2 {
        return None;
    }
    let strides = onnx_proto::get_attribute_ints(node, "strides").unwrap_or_default();
    let pads = onnx_proto::get_attribute_ints(node, "pads").unwrap_or_default();
    let dilations = onnx_proto::get_attribute_ints(node, "dilations").unwrap_or_default();
    let sh = strides.first().copied().unwrap_or(1);
    let sw = strides.get(1).copied().unwrap_or(1);
    if sh == 0 || sw == 0 {
        return None;
    }
    let ph = if pads.len() >= 4 {
        pads[0] + pads[2]
    } else {
        0
    };
    let pw = if pads.len() >= 4 {
        pads[1] + pads[3]
    } else {
        0
    };
    let dh = dilations.first().copied().unwrap_or(1);
    let dw = dilations.get(1).copied().unwrap_or(1);
    let kh = (kernel[0] - 1) * dh + 1;
    let kw = (kernel[1] - 1) * dw + 1;
    let (h, w) = if ceil_mode {
        (
            (in_shape[2] + ph - kh + sh - 1) / sh + 1,
            (in_shape[3] + pw - kw + sw - 1) / sw + 1,
        )
    } else {
        (
            (in_shape[2] + ph).saturating_sub(kh) / sh + 1,
            (in_shape[3] + pw).saturating_sub(kw) / sw + 1,
        )
    };
    Some(vec![h, w])
}

#[allow(dead_code)]
fn infer_conv_pool_shape(
    node: &NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
    graph: &GraphProto,
) -> Option<Vec<i64>> {
    let inp = node.input.first()?;
    let in_shape = shapes.get(inp)?;
    match node.op_type.as_str() {
        "Conv" => {
            let w_name = node.input.get(1)?;
            let w_dims: Vec<i64> = graph
                .initializer
                .iter()
                .find(|i| &i.name == w_name)
                .map(|i| i.dims.clone())?;
            if w_dims.len() != 4 {
                return None;
            }
            let hw = conv_pool_spatial(in_shape, &w_dims[2..], node, false)?;
            Some(vec![in_shape[0], w_dims[0], hw[0], hw[1]])
        }
        "MaxPool" | "AveragePool" => {
            let kernel = onnx_proto::get_attribute_ints(node, "kernel_shape").unwrap_or_default();
            let ceil = onnx_proto::get_attribute_int(node, "ceil_mode").unwrap_or(0) != 0;
            let hw = conv_pool_spatial(in_shape, &kernel, node, ceil)?;
            Some(vec![in_shape[0], in_shape[1], hw[0], hw[1]])
        }
        "Transpose" => {
            let perm = onnx_proto::get_attribute_ints(node, "perm")?;
            if perm.len() != in_shape.len() {
                return None;
            }
            Some(perm.iter().map(|&p| in_shape[p as usize]).collect())
        }
        "MatMul" => {
            let b = shapes.get(node.input.get(1)?)?;
            if in_shape.len() < 2 || b.len() < 2 {
                return None;
            }
            let mut out = in_shape[..in_shape.len() - 1].to_vec();
            out.push(*b.last().unwrap());
            Some(out)
        }
        "Concat" => {
            let axis = onnx_proto::get_attribute_int(node, "axis")?;
            let all: Vec<&Vec<i64>> = node.input.iter().filter_map(|n| shapes.get(n)).collect();
            if all.len() != node.input.len() || all.is_empty() {
                return None;
            }
            let rank = all[0].len();
            if rank == 0 {
                return None;
            }
            let axis = if axis < 0 { rank as i64 + axis } else { axis } as usize;
            if axis >= rank || all.iter().any(|s| s.len() != rank) {
                return None;
            }
            let mut out = all[0].clone();
            for s in &all[1..] {
                out[axis] += s[axis];
            }
            Some(out)
        }
        "Reshape" => {
            let target_name = node.input.get(1)?;
            let target_tensor = graph.initializer.iter().find(|i| &i.name == target_name)?;
            let target = onnx_proto::tensor_to_i64(target_tensor);
            let vol: i64 = in_shape.iter().product();
            let mut out = target;
            let neg_idx = out.iter().position(|&v| v == -1);
            let known_vol: i64 = out.iter().filter(|&&v| v != -1).product();
            if let Some(idx) = neg_idx {
                out[idx] = if known_vol != 0 { vol / known_vol } else { 0 };
            }
            for (i, d) in out.iter_mut().enumerate() {
                if *d == 0 && i < in_shape.len() {
                    *d = in_shape[i];
                }
            }
            Some(out)
        }
        "Flatten" => {
            let axis = onnx_proto::get_attribute_int(node, "axis").unwrap_or(1) as usize;
            let d0: i64 = in_shape[..axis].iter().product();
            let d1: i64 = in_shape[axis..].iter().product();
            Some(vec![d0, d1])
        }
        "LayerNormalization" | "BatchNormalization" | "Resize" => Some(in_shape.clone()),
        "Tile" => {
            let repeats_name = node.input.get(1)?;
            let repeats_tensor = graph.initializer.iter().find(|i| &i.name == repeats_name)?;
            let repeats = onnx_proto::tensor_to_i64(repeats_tensor);
            if repeats.len() != in_shape.len() {
                return None;
            }
            Some(
                in_shape
                    .iter()
                    .zip(repeats.iter())
                    .map(|(&d, &r)| d * r)
                    .collect(),
            )
        }
        _ => None,
    }
}

#[allow(dead_code)]
fn trace_shapes_tract(
    onnx_path: &Path,
    proto_model: &ModelProto,
) -> Result<HashMap<String, Vec<i64>>> {
    use tract_onnx::prelude::*;
    use tract_onnx::tract_hir::infer::Factoid;

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

    let mut shapes = HashMap::new();
    let mut tract_names_to_shapes: Vec<(String, Vec<i64>)> = Vec::new();

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.analyse(true))) {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "tract obstinate analysis encountered errors");
        }
        Err(_) => {
            tracing::warn!("tract analysis panicked; extracting partial results");
        }
    }

    for node_id in 0..model.nodes().len() {
        let node_obj = model.node(node_id);
        for (ix, outlet) in node_obj.outputs.iter().enumerate() {
            let concrete = match outlet.fact.shape.concretize() {
                Some(dims) => dims,
                None => continue,
            };
            let shape_vec: Vec<i64> = concrete.iter().map(|d| d.to_i64().unwrap_or(0)).collect();
            if shape_vec.iter().all(|&d| d > 0) {
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
                let volume = |s: &[i64]| -> i64 { s.iter().copied().product() };
                for (tract_name, shape) in &tract_names_to_shapes {
                    if tract_name.starts_with(&prefix)
                        && matched_shape.is_none_or(|s| {
                            shape.len() > s.len()
                                || (shape.len() == s.len() && volume(shape) > volume(s))
                        })
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

        let mut prev_len = 0;
        while shapes.len() != prev_len {
            prev_len = shapes.len();

            for node in &graph.node {
                if super::is_shape_preserving(&node.op_type)
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
                if node.op_type == "Shape"
                    && let Some(inp) = node.input.first()
                    && let Some(in_shape) = shapes.get(inp)
                {
                    let rank = in_shape.len() as i64;
                    let start = onnx_proto::get_attribute_int(node, "start").unwrap_or(0);
                    let end = onnx_proto::get_attribute_int(node, "end").unwrap_or(rank);
                    let normalize = |idx: i64| {
                        if idx < 0 {
                            (rank + idx).max(0)
                        } else {
                            idx.min(rank)
                        }
                    };
                    let len = (normalize(end) - normalize(start)).max(0);
                    for out in &node.output {
                        if !out.is_empty() && !shapes.contains_key(out) {
                            shapes.insert(out.clone(), vec![len]);
                        }
                    }
                }
            }

            for node in &graph.node {
                for out in &node.output {
                    if out.is_empty() || shapes.contains_key(out) {
                        continue;
                    }
                    if let Some(vi) = graph.value_info.iter().find(|v| v.name == *out)
                        && let Some(shape) = onnx_proto::shape_from_value_info(vi)
                        && shape.iter().all(|&d| d > 0)
                    {
                        shapes.insert(out.clone(), shape);
                    }
                }
            }

            for node in &graph.node {
                if super::is_binary_arithmetic(&node.op_type) {
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
                if let Some(out_shape) = infer_conv_pool_shape(node, &shapes, graph) {
                    for out in &node.output {
                        if !out.is_empty() && !shapes.contains_key(out) {
                            shapes.insert(out.clone(), out_shape.clone());
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

    for shape in shapes.values_mut() {
        for d in shape.iter_mut() {
            if *d <= 0 {
                *d = 1;
            }
        }
    }

    tracing::info!(tensors = shapes.len(), "shape trace complete");
    Ok(shapes)
}

pub(crate) fn broadcast_shapes(shapes: &[&Vec<i64>]) -> Option<Vec<i64>> {
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
            make_analysis_with_params(vec![("a", 0, "Relu", false), ("b", 1, "Reshape", false)]);
        let points = vec![0];
        let result = isolate_conv(&points, &analysis);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn isolate_maxpool_gets_boundary() {
        let analysis =
            make_analysis_with_params(vec![("a", 0, "Relu", false), ("b", 1, "MaxPool", false)]);
        let points = vec![0];
        let result = isolate_conv(&points, &analysis);
        assert_eq!(result, vec![0, 1]);
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
    fn optimize_for_tiling_maxpool_stays_grouped() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "Conv", false),
            ("b", 1, "Relu", false),
            ("c", 2, "MaxPool", false),
            ("d", 3, "Conv", false),
        ]);
        let points = vec![0, 3];
        let result = optimize_for_tiling(&points, &analysis);
        assert!(!result.contains(&2));
    }

    #[test]
    fn optimize_for_tiling_splits_at_non_tileable() {
        let analysis = make_analysis_with_params(vec![
            ("a", 0, "Conv", false),
            ("b", 1, "Relu", false),
            ("c", 2, "Reshape", false),
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

    type NodeSpec<'a> = (&'a str, usize, &'a str, bool, Vec<&'a str>, Vec<&'a str>);

    fn make_analysis_with_deps(nodes: Vec<NodeSpec<'_>>) -> AnalysisResult {
        let mut node_map = HashMap::new();
        for (name, index, op_type, has_params, inputs, outputs) in &nodes {
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
                        input: inputs.iter().map(|s| s.to_string()).collect(),
                        output: outputs.iter().map(|s| s.to_string()).collect(),
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

    #[test]
    fn merge_control_flow_removes_boundary_between_producer_and_loop() {
        let analysis = make_analysis_with_deps(vec![
            ("conv0", 0, "Conv", true, vec!["x"], vec!["conv_out"]),
            (
                "relu0",
                1,
                "Relu",
                false,
                vec!["conv_out"],
                vec!["relu_out"],
            ),
            (
                "matmul0",
                2,
                "MatMul",
                true,
                vec!["relu_out"],
                vec!["mm_out"],
            ),
            (
                "loop0",
                3,
                "Loop",
                false,
                vec!["trip", "cond", "init", "relu_out"],
                vec!["loop_out"],
            ),
        ]);
        let points = vec![0, 2, 4];
        let result = merge_control_flow_segments(&points, &analysis);
        assert!(
            !result.contains(&2),
            "slice point 2 separates relu0 (producer of relu_out at idx 1) from Loop (idx 3); must be removed: {:?}",
            result
        );
    }

    #[test]
    fn merge_control_flow_preserves_unrelated_boundaries() {
        let analysis = make_analysis_with_deps(vec![
            ("conv0", 0, "Conv", true, vec!["x"], vec!["conv_out"]),
            (
                "relu0",
                1,
                "Relu",
                false,
                vec!["conv_out"],
                vec!["relu_out"],
            ),
            (
                "conv1",
                2,
                "Conv",
                true,
                vec!["relu_out"],
                vec!["conv1_out"],
            ),
            (
                "relu1",
                3,
                "Relu",
                false,
                vec!["conv1_out"],
                vec!["relu1_out"],
            ),
            (
                "loop0",
                4,
                "Loop",
                false,
                vec!["trip", "cond", "relu1_out"],
                vec!["loop_out"],
            ),
        ]);
        let points = vec![0, 2, 5];
        let result = merge_control_flow_segments(&points, &analysis);
        assert!(
            result.contains(&2),
            "boundary at 2 is between conv0/relu0 and conv1/relu1, should be preserved since Loop only depends on relu1_out (idx 3): {:?}",
            result
        );
    }

    #[test]
    fn merge_control_flow_no_control_flow_ops() {
        let analysis = make_analysis_with_deps(vec![
            ("conv0", 0, "Conv", true, vec!["x"], vec!["conv_out"]),
            ("relu0", 1, "Relu", false, vec!["conv_out"], vec!["y"]),
        ]);
        let points = vec![0, 1, 2];
        let result = merge_control_flow_segments(&points, &analysis);
        assert_eq!(result, vec![0, 1, 2]);
    }
}
