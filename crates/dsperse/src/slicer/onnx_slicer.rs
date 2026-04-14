use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::analyzer::{self, AnalysisResult, NodeAnalysis};
use super::autotiler;
use super::materializer;
use super::onnx_proto;
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
    onnx_proto::resolve_dynamic_input_shapes(&mut model, input_shape)?;

    onnx_proto::strip_symbolic_value_info(&mut model);
    let folded_constants = super::onnx_fold::fold_constant_nodes(&mut model);

    let tmp_dir = tempfile::tempdir().map_err(|e| DsperseError::io(e, onnx_path))?;
    let tract_path = tmp_dir.path().join("tract_model.onnx");
    onnx_proto::save_model(&model, &tract_path)?;

    tracing::info!("folding constants and tracing shapes via tract");
    let trace_result = super::trace::fold_and_trace_via_tract(&tract_path, &model)?;
    let mut traced_shapes = trace_result.shapes;
    let traced_types = trace_result.types;

    if let Some(graph) = model.graph.as_mut() {
        // Chains of shape-dependent ops (Shape -> Gather -> Reshape,
        // or nested ConstantOfShape pyramids) expose constants only
        // after earlier rounds have folded their producers, so run
        // propagate_constants_with_shapes to a fixpoint.  A small
        // safety cap prevents an unexpected non-monotonic evaluator
        // from spinning indefinitely; propagation is monotone by
        // construction so the loop is expected to converge in O(1)
        // iterations even for the deepest chains we have observed.
        const SHAPE_CONST_PROP_ITERATION_CAP: usize = 16;
        let mut total_folded = 0usize;
        for pass in 0..SHAPE_CONST_PROP_ITERATION_CAP {
            let folded = super::onnx_fold::propagate_constants_with_shapes(graph, &traced_shapes);
            if folded == 0 {
                break;
            }
            total_folded += folded;
            tracing::info!(pass, folded, "shape-constant propagation pass");
        }
        if total_folded > 0 {
            tracing::info!(
                total_folded,
                "propagated shape-derived constants in parent graph"
            );
        }
    }

    let fused_ln = super::layernorm_fuse::fuse_inline_layernorms(&mut model, &mut traced_shapes);
    if fused_ln > 0 {
        tracing::info!(fused_ln, "fused inline LayerNorm patterns");
    }

    let self_div_rewrites =
        super::self_div_rewrite::rewrite_self_div_to_one(&mut model, &mut traced_shapes);
    if self_div_rewrites > 0 {
        tracing::info!(self_div_rewrites, "rewrote degenerate Div(X, X) nodes");
    }

    let missing: Vec<String> = if let Some(graph) = &model.graph {
        let mut missing = Vec::new();
        for n in &graph.node {
            for out in &n.output {
                if !out.is_empty() && !traced_shapes.contains_key(out) {
                    missing.push(out.clone());
                }
            }
        }
        missing
    } else {
        Vec::new()
    };
    if !missing.is_empty() {
        tracing::warn!(count = missing.len(), first_few = ?&missing[..missing.len().min(5)], "unresolved tensor shapes after all inference passes");
    }

    let analysis = analyzer::analyze(&model, Some(onnx_path))?;

    let output_dir = output_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        onnx_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("slices")
    });
    std::fs::create_dir_all(&output_dir).map_err(|e| DsperseError::io(e, &output_dir))?;

    let slice_points =
        determine_slice_points(&analysis, tile_size, jstprove_ops, &model, &traced_shapes);
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
    let mut dim_split_info: HashMap<usize, (autotiler::DimSplitDetection, Option<String>)> =
        HashMap::new();
    for (seg_idx, _) in segment_ranges.iter().enumerate() {
        let slice_model = materializer::materialize_slice_model(
            &model,
            trimmed_points,
            &traced_shapes,
            &traced_types,
            seg_idx,
        )?;
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
            if let Some(detection) = autotiler::detect_dim_split(
                &graph.node,
                &slice_shapes,
                &init_names,
                autotiler::model_opset(&model),
            ) {
                // Build a tentative DimSplitInfo to attempt template creation.
                // Only record the detection if the template materializes
                // successfully, so the metadata never carries dim_split
                // entries that can't be fulfilled at runtime.
                let tentative_info = DimSplitInfo::from_detection(&detection, seg_idx, None);
                let slice_dir = output_dir.join(format!("slice_{seg_idx}")).join("payload");
                std::fs::create_dir_all(&slice_dir).map_err(|e| DsperseError::io(e, &slice_dir))?;
                match autotiler::create_dim_split_template(
                    &slice_model,
                    &tentative_info,
                    &slice_dir,
                    Some(&traced_shapes),
                ) {
                    Ok(tmpl_path) => {
                        let tmpl_rel = tmpl_path
                            .strip_prefix(&output_dir)
                            .map_err(|_| {
                                DsperseError::Slicer(format!(
                                    "dim-split template path {} is not under output dir {}",
                                    tmpl_path.display(),
                                    output_dir.display()
                                ))
                            })?
                            .to_string_lossy()
                            .into_owned();
                        tracing::info!(
                            slice = seg_idx,
                            estimated = detection.estimated_constraints,
                            num_groups = detection.num_groups,
                            split_kind = ?detection.split_kind,
                            "dim-split detected and template created"
                        );
                        dim_split_info.insert(seg_idx, (detection, Some(tmpl_rel)));
                    }
                    Err(e) => {
                        tracing::warn!(
                            slice = seg_idx,
                            estimated = detection.estimated_constraints,
                            error = %e,
                            "dim-split detected but template creation failed; \
                             slice will be skipped during compilation"
                        );
                        // Record detection with no template path so the
                        // compiler knows this slice was over-budget and
                        // should be skipped rather than falling through
                        // to monolithic compilation.
                        dim_split_info.insert(seg_idx, (detection, None));
                    }
                }
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
        traced_types: Some(traced_types),
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
    dim_split_info: &HashMap<usize, (autotiler::DimSplitDetection, Option<String>)>,
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

        let dim_split = dim_split_info
            .get(&seg_idx)
            .map(|(d, tmpl_rel)| DimSplitInfo::from_detection(d, seg_idx, tmpl_rel.clone()));

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
    model: &onnx_proto::ModelProto,
    traced_shapes: &HashMap<String, Vec<i64>>,
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
    sorted_points = isolate_expensive_ops(&sorted_points, analysis, model, traced_shapes);
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

/// Insert slice points before AND after every ONNX node whose
/// estimated constraint count exceeds
/// [`autotiler::MAX_ESTIMATED_CONSTRAINTS`].  Each "expensive" op
/// (large MatMul, LayerNormalization, Softmax, etc.) becomes a
/// single-node slice so the dim-split detector sees an unambiguous
/// shape and the runner doesn't need to trace which axis lives where
/// through Transpose / Reshape neighbours.  Small ops keep their
/// existing grouping for circuit catalog reuse.
fn isolate_expensive_ops(
    points: &[usize],
    analysis: &AnalysisResult,
    model: &onnx_proto::ModelProto,
    traced_shapes: &HashMap<String, Vec<i64>>,
) -> Vec<usize> {
    use jstprove_circuits::api::{EstimationConfig, estimate_op_constraints};
    let cfg = EstimationConfig::bn254_defaults();
    let threshold = autotiler::MAX_ESTIMATED_CONSTRAINTS;

    // Build a parallel index: ONNX-node-index -> &NodeProto so we can
    // resolve input/output tensor names per slicer-node.
    let onnx_nodes: Vec<&onnx_proto::NodeProto> = model
        .graph
        .as_ref()
        .map(|g| g.node.iter().collect())
        .unwrap_or_default();
    // Resolve a tensor's traced shape strictly: every dim must be a
    // concrete positive value.  Coercing dynamic / -1 / 0 dims to 1
    // would silently drive the cost estimate to ~zero and let the
    // very nodes this pass exists to isolate sneak through.  Returning
    // `None` for an unresolved tensor is the signal to pessimistically
    // isolate the node anyway.
    let to_usize_shape = |name: &String| -> Option<Vec<usize>> {
        let shape = traced_shapes.get(name)?;
        let mut out = Vec::with_capacity(shape.len());
        for &d in shape {
            if d <= 0 {
                return None;
            }
            out.push(d as usize);
        }
        Some(out)
    };

    // Pure elementwise binary ops (Add / Sub / Mul / Div / Pow) are
    // never isolated.  This is a coupling to jstprove_circuits's
    // single-op-slice invariants: when an isolated slice contains
    // exactly one Div with a runtime divisor, one Mul / Sub between
    // operands of broadcast-incompatible shapes, or one Pow whose
    // exponent is a non-constant tensor, the per-op layer builder
    // rejects the slice with a strict-mode error.  When the same
    // pattern appears inside a larger multi-op slice the
    // dim-split / LayerNorm fusion machinery rewrites the
    // surrounding subgraph and the strict check passes.  These ops
    // are also cheap to compile in absolute terms, so isolating them
    // buys little proving wall-clock and surfaces the strict-mode
    // failure more often.
    //
    // TODO: revisit when jstprove_circuits relaxes the single-op
    // invariants (or exposes a "permissive" mode) so we can drop
    // this exemption and let the autotiler decide based on cost.
    let elementwise_skip: HashSet<&str> = ["Add", "Sub", "Mul", "Div", "Pow"].into_iter().collect();
    optimize_points(points, analysis, |updated, sorted_nodes, max_idx| {
        for node in sorted_nodes {
            if elementwise_skip.contains(node.node_type.as_str()) {
                continue;
            }
            let Some(onnx_node) = onnx_nodes.get(node.index) else {
                continue;
            };
            // ONNX node inputs / outputs use "" to denote an
            // unbound optional slot (e.g. Conv with no bias, GRU
            // with no initial_h).  Treating those as unresolved
            // boundary tensors makes every node carrying an empty
            // slot pessimistically isolate, even when the real
            // boundary tensors are fully shape-resolved.  Skip the
            // empty entries so estimate_op_constraints sees only
            // the real boundary tensors.
            let in_shapes: Option<Vec<Vec<usize>>> = onnx_node
                .input
                .iter()
                .filter(|name| !name.is_empty())
                .map(&to_usize_shape)
                .collect();
            let out_shapes: Option<Vec<Vec<usize>>> = onnx_node
                .output
                .iter()
                .filter(|name| !name.is_empty())
                .map(&to_usize_shape)
                .collect();
            // If any boundary tensor is unresolved we cannot give an
            // honest cost estimate; isolate pessimistically so the
            // downstream compile path sees a single-op slice and can
            // either compile it successfully or skip it cleanly,
            // rather than silently grouping an unbounded op.
            let isolate = match (in_shapes, out_shapes) {
                (Some(ins), Some(outs)) => {
                    estimate_op_constraints(&node.node_type, &ins, &outs, &cfg) > threshold
                }
                _ => true,
            };
            if isolate {
                updated.insert(node.index);
                if node.index < max_idx {
                    updated.insert(node.index + 1);
                }
            }
        }
    })
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
                    if !super::is_slice_passthrough(&candidate.node_type) {
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
        let model = onnx_proto::ModelProto::default();
        let traced = HashMap::new();
        let points = determine_slice_points(&analysis, None, TEST_OPS, &model, &traced);
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
        let model = onnx_proto::ModelProto::default();
        let traced = HashMap::new();
        let points = determine_slice_points(&analysis, Some(1024), TEST_OPS, &model, &traced);
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

    /// Regression for PR #183: isolate_conv's inner grouping walk
    /// must treat the LAYOUT_OPS set (Reshape / Transpose /
    /// Flatten / Squeeze / Unsqueeze / Gather) as passthroughs so
    /// that Conv -> Reshape -> MatMul places the trailing compile
    /// boundary on the heavy MatMul rather than on the Reshape
    /// that sits between them.  Before the is_slice_passthrough
    /// split these ops were absent from is_shape_preserving and
    /// the walk terminated on the Reshape, isolating it into its
    /// own slice.
    #[test]
    fn isolate_conv_absorbs_reshape_then_boundaries_on_matmul() {
        let analysis = make_analysis_with_deps(vec![
            ("conv0", 0, "Conv", true, vec!["x"], vec!["conv_out"]),
            (
                "reshape0",
                1,
                "Reshape",
                false,
                vec!["conv_out", "shape"],
                vec!["reshape_out"],
            ),
            (
                "matmul0",
                2,
                "MatMul",
                true,
                vec!["reshape_out", "matmul0_weight"],
                vec!["matmul_out"],
            ),
        ]);
        let points = vec![0, 3];
        let result = isolate_conv(&points, &analysis);
        assert!(
            result.contains(&0),
            "isolate_conv should insert a boundary at the Conv itself: {result:?}"
        );
        assert!(
            result.contains(&2),
            "is_slice_passthrough should absorb Reshape into the Conv slice and place the trailing boundary on MatMul at index 2: {result:?}"
        );
        assert!(
            !result.contains(&1),
            "Reshape at index 1 must not become its own slice boundary when it sits between a Conv and a heavy op: {result:?}"
        );
    }

    /// Transpose + Squeeze variant so we also cover the other
    /// layout ops added to LAYOUT_OPS.
    #[test]
    fn isolate_conv_absorbs_transpose_chain_then_boundaries_on_matmul() {
        let analysis = make_analysis_with_deps(vec![
            ("conv0", 0, "Conv", true, vec!["x"], vec!["conv_out"]),
            (
                "transpose0",
                1,
                "Transpose",
                false,
                vec!["conv_out"],
                vec!["trans_out"],
            ),
            (
                "squeeze0",
                2,
                "Squeeze",
                false,
                vec!["trans_out"],
                vec!["sq_out"],
            ),
            (
                "matmul0",
                3,
                "MatMul",
                true,
                vec!["sq_out", "matmul0_weight"],
                vec!["matmul_out"],
            ),
        ]);
        let points = vec![0, 4];
        let result = isolate_conv(&points, &analysis);
        assert!(result.contains(&0));
        assert!(
            result.contains(&3),
            "Transpose + Squeeze chain should absorb into the Conv slice, leaving MatMul at index 3 as the boundary: {result:?}"
        );
        assert!(!result.contains(&1));
        assert!(!result.contains(&2));
    }

    /// Counter-case: a layout op whose input is NOT produced by
    /// the preceding Conv slice must still break the walk, so the
    /// consumes_produced guard is exercised.
    #[test]
    fn isolate_conv_stops_when_passthrough_consumes_external_input() {
        let analysis = make_analysis_with_deps(vec![
            ("conv0", 0, "Conv", true, vec!["x"], vec!["conv_out"]),
            (
                "reshape0",
                1,
                "Reshape",
                false,
                // Reshape consumes an external tensor, not
                // conv_out, so is_slice_passthrough being true is
                // not sufficient to absorb it.
                vec!["external_y", "shape"],
                vec!["reshape_out"],
            ),
        ]);
        let points = vec![0, 2];
        let result = isolate_conv(&points, &analysis);
        assert!(result.contains(&0));
        assert!(
            result.contains(&1),
            "Reshape that doesn't consume any conv-produced tensor should remain the trailing boundary: {result:?}"
        );
    }
}
