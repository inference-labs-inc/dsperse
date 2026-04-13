use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::onnx_proto::{self, GraphProto, ModelProto, NodeProto, TensorProto};
use crate::error::Result;
use crate::schema::tiling::{ChannelGroupInfo, ChannelSplitInfo, DimSplitKind};

fn try_pair(v: &[i64]) -> Option<[i64; 2]> {
    if v.len() == 2 {
        Some([v[0], v[1]])
    } else {
        None
    }
}

fn try_quad(v: &[i64]) -> Option<[i64; 4]> {
    if v.len() == 4 {
        Some([v[0], v[1], v[2], v[3]])
    } else {
        None
    }
}

fn model_opset(model: &ModelProto) -> i64 {
    model
        .opset_import
        .iter()
        .filter(|o| o.domain.is_empty())
        .map(|o| o.version)
        .max()
        .unwrap_or(13)
}

fn is_elementwise(op: &str) -> bool {
    super::is_elementwise(op)
}

#[derive(Debug, Clone)]
pub struct ChannelSplitParams {
    pub c_in: i64,
    pub c_out: i64,
    pub num_groups: i64,
    pub channels_per_group: i64,
    pub h: i64,
    pub w: i64,
    pub slice_idx: usize,
}

struct PoolParams {
    node_idx: usize,
    kernel: [i64; 2],
    stride: [i64; 2],
    dilation: [i64; 2],
    pads: [i64; 4],
}

impl PoolParams {
    fn from_node(node: &NodeProto, node_idx: usize) -> Option<PoolParams> {
        if node.op_type != "MaxPool" {
            return None;
        }
        let kernel = try_pair(&onnx_proto::get_attribute_ints(node, "kernel_shape")?)?;
        let stride = match onnx_proto::get_attribute_ints(node, "strides") {
            None => [1, 1],
            Some(v) => try_pair(&v)?,
        };
        let dilation = match onnx_proto::get_attribute_ints(node, "dilations") {
            None => [1, 1],
            Some(v) => try_pair(&v)?,
        };
        let auto_pad = node
            .attribute
            .iter()
            .find(|a| a.name == "auto_pad")
            .map(|a| a.s.as_slice());
        if matches!(auto_pad, Some(v) if !v.is_empty() && v != b"NOTSET") {
            return None;
        }
        let pads = match onnx_proto::get_attribute_ints(node, "pads") {
            None => [0, 0, 0, 0],
            Some(v) => try_quad(&v)?,
        };
        let ceil_mode = onnx_proto::get_attribute_int(node, "ceil_mode").unwrap_or(0);
        if ceil_mode != 0 {
            return None;
        }
        if kernel.iter().any(|&v| v <= 0) || stride.iter().any(|&v| v <= 0) {
            return None;
        }
        if dilation.iter().any(|&v| v <= 0) || pads.iter().any(|&v| v < 0) {
            return None;
        }
        Some(PoolParams {
            node_idx,
            kernel,
            stride,
            dilation,
            pads,
        })
    }
}

fn get_pool_params(graph: &GraphProto) -> Option<PoolParams> {
    for (idx, node) in graph.node.iter().enumerate() {
        if let Some(pp) = PoolParams::from_node(node, idx) {
            return Some(pp);
        }
    }
    None
}

struct ConvParams {
    node_idx: usize,
    kernel: [i64; 2],
    stride: [i64; 2],
    dilation: [i64; 2],
    pads: [i64; 4],
    group: i64,
    c_out: i64,
    c_in: i64,
}

impl ConvParams {
    fn from_node(node: &NodeProto, node_idx: usize, graph: &GraphProto) -> Option<ConvParams> {
        if node.op_type != "Conv" {
            return None;
        }
        let w_name = node.input.get(1)?;
        let w = graph.initializer.iter().find(|t| &t.name == w_name)?;
        if w.dims.len() != 4 {
            return None;
        }
        let c_out = w.dims[0];
        let c_in = w.dims[1];
        if c_out <= 0 || c_in <= 0 {
            return None;
        }

        let inferred_kernel = [w.dims[2], w.dims[3]];
        let kernel = match onnx_proto::get_attribute_ints(node, "kernel_shape") {
            Some(v) => {
                let k = try_pair(&v)?;
                if k != inferred_kernel {
                    return None;
                }
                k
            }
            None => inferred_kernel,
        };
        let stride = match onnx_proto::get_attribute_ints(node, "strides") {
            None => [1, 1],
            Some(v) => try_pair(&v)?,
        };
        let dilation = match onnx_proto::get_attribute_ints(node, "dilations") {
            None => [1, 1],
            Some(v) => try_pair(&v)?,
        };
        let auto_pad = node
            .attribute
            .iter()
            .find(|a| a.name == "auto_pad")
            .map(|a| a.s.as_slice());
        if matches!(auto_pad, Some(v) if !v.is_empty() && v != b"NOTSET") {
            return None;
        }
        let pads = match onnx_proto::get_attribute_ints(node, "pads") {
            None => [0, 0, 0, 0],
            Some(v) => try_quad(&v)?,
        };
        if kernel.iter().any(|&v| v <= 0) {
            return None;
        }
        if stride.iter().any(|&v| v <= 0) {
            return None;
        }
        if dilation.iter().any(|&v| v <= 0) {
            return None;
        }
        if pads.iter().any(|&v| v < 0) {
            return None;
        }
        let group = onnx_proto::get_attribute_int(node, "group").unwrap_or(1);
        if group <= 0 {
            return None;
        }

        Some(ConvParams {
            node_idx,
            kernel,
            stride,
            dilation,
            pads,
            group,
            c_out,
            c_in,
        })
    }
}

fn get_conv_params(graph: &GraphProto) -> Option<ConvParams> {
    for (idx, node) in graph.node.iter().enumerate() {
        if let Some(cp) = ConvParams::from_node(node, idx, graph) {
            return Some(cp);
        }
    }
    None
}

fn effective_kernel(kernel: [i64; 2], dilation: [i64; 2]) -> Option<[i64; 2]> {
    let ek0 = kernel[0]
        .checked_sub(1)?
        .checked_mul(dilation[0])?
        .checked_add(1)?;
    let ek1 = kernel[1]
        .checked_sub(1)?
        .checked_mul(dilation[1])?
        .checked_add(1)?;
    Some([ek0, ek1])
}

fn conv_output_hw(
    h_in: i64,
    w_in: i64,
    pads: [i64; 4],
    kernel: [i64; 2],
    dilation: [i64; 2],
    stride: [i64; 2],
) -> Option<(i64, i64)> {
    if stride[0] <= 0 || stride[1] <= 0 {
        return None;
    }
    let eff = effective_kernel(kernel, dilation)?;
    let num_h = h_in
        .checked_add(pads[0])?
        .checked_add(pads[2])?
        .checked_sub(eff[0])?;
    let num_w = w_in
        .checked_add(pads[1])?
        .checked_add(pads[3])?
        .checked_sub(eff[1])?;
    let out_h = num_h.div_euclid(stride[0]).checked_add(1)?;
    let out_w = num_w.div_euclid(stride[1]).checked_add(1)?;
    if out_h <= 0 || out_w <= 0 {
        return None;
    }
    Some((out_h, out_w))
}

fn compute_halo_size(pads: [i64; 4]) -> Option<[i64; 4]> {
    if pads.iter().any(|&v| v < 0) {
        return None;
    }
    Some(pads)
}

fn compute_min_spatial_tile(kernel: [i64; 2], dilation: [i64; 2]) -> Option<i64> {
    let eff = effective_kernel(kernel, dilation)?;
    eff[0].max(eff[1]).checked_add(1)
}

struct SpatialKernelParams {
    kernel: [i64; 2],
    stride: [i64; 2],
    dilation: [i64; 2],
    pads: [i64; 4],
}

fn extract_spatial_kernel_params(
    graph: &GraphProto,
    primary_op: &str,
) -> Option<SpatialKernelParams> {
    if graph.input.len() > 1 {
        return None;
    }
    let op_count = graph
        .node
        .iter()
        .filter(|n| n.op_type == primary_op)
        .count();
    if op_count != 1 {
        return None;
    }
    let (node_idx, kernel, stride, dilation, pads) = if primary_op == "Conv" {
        let cp = get_conv_params(graph)?;
        (cp.node_idx, cp.kernel, cp.stride, cp.dilation, cp.pads)
    } else if primary_op == "MaxPool" {
        let pp = get_pool_params(graph)?;
        (pp.node_idx, pp.kernel, pp.stride, pp.dilation, pp.pads)
    } else {
        return None;
    };
    if node_idx != 0 {
        return None;
    }
    let ops: HashSet<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
    if ops.iter().any(|&o| o != primary_op && !is_elementwise(o)) {
        return None;
    }
    Some(SpatialKernelParams {
        kernel,
        stride,
        dilation,
        pads,
    })
}

fn is_spatial_tileable(graph: &GraphProto, primary_op: &str) -> bool {
    let Some(sp) = extract_spatial_kernel_params(graph, primary_op) else {
        return false;
    };
    let Some(eff) = effective_kernel(sp.kernel, sp.dilation) else {
        return false;
    };
    let total_pad_h = sp.pads[0] + sp.pads[2];
    let total_pad_w = sp.pads[1] + sp.pads[3];
    total_pad_h >= eff[0] - sp.stride[0] && total_pad_w >= eff[1] - sp.stride[1]
}

fn is_standard_conv_slice(graph: &GraphProto) -> Option<ConvParams> {
    extract_spatial_kernel_params(graph, "Conv")?;
    get_conv_params(graph)
}

fn is_tileable(graph: &GraphProto) -> bool {
    is_spatial_tileable(graph, "Conv")
}

fn is_channel_splittable(graph: &GraphProto) -> bool {
    let Some(cp) = is_standard_conv_slice(graph) else {
        return false;
    };
    cp.group == 1
}

fn get_model_dimensions(graph: &GraphProto) -> Option<(String, String, i64, i64, i64)> {
    let inp = graph.input.first()?;
    let out = graph.output.first()?;
    let dims = onnx_proto::vi_shape(inp);
    if dims.len() != 4 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0 {
        return None;
    }
    Some((
        inp.name.clone(),
        out.name.clone(),
        dims[1],
        dims[2],
        dims[3],
    ))
}

fn is_elementwise_only_slice(graph: &GraphProto) -> bool {
    if graph.node.is_empty() || graph.input.is_empty() {
        return false;
    }
    graph.node.iter().all(|n| is_elementwise(&n.op_type))
}

fn find_weights_and_bias(
    graph: &GraphProto,
    conv_node: &NodeProto,
) -> (Option<WeightInfo>, Option<Vec<f32>>) {
    let mut weights: Option<WeightInfo> = None;
    let mut bias: Option<Vec<f32>> = None;

    for init in &graph.initializer {
        if conv_node.input.len() > 1 && init.name == conv_node.input[1] {
            let data = onnx_proto::tensor_to_f32(init);
            weights = Some(WeightInfo {
                data,
                dims: init.dims.clone(),
            });
        }
        if conv_node.input.len() > 2 && init.name == conv_node.input[2] {
            bias = Some(onnx_proto::tensor_to_f32(init));
        }
    }
    (weights, bias)
}

struct WeightInfo {
    data: Vec<f32>,
    dims: Vec<i64>,
}

struct SlicePrologue<'a> {
    graph: &'a GraphProto,
    cp: ConvParams,
    weights: Option<WeightInfo>,
    bias: Option<Vec<f32>>,
}

fn extract_slice_prologue(model: &ModelProto) -> Option<SlicePrologue<'_>> {
    let graph = model.graph.as_ref()?;
    let cp = get_conv_params(graph)?;
    let conv_node = &graph.node[cp.node_idx];
    let (weights, bias) = find_weights_and_bias(graph, conv_node);
    if let Some(ref w) = weights {
        if w.dims.len() != 4 {
            return None;
        }
        let c_out = usize::try_from(w.dims[0]).ok()?;
        let c_in = usize::try_from(w.dims[1]).ok()?;
        let kh = usize::try_from(w.dims[2]).ok()?;
        let kw = usize::try_from(w.dims[3]).ok()?;
        let expected = c_out.checked_mul(c_in)?.checked_mul(kh)?.checked_mul(kw)?;
        if w.data.len() != expected {
            return None;
        }
        if let Some(ref b) = bias
            && b.len() != c_out
        {
            return None;
        }
    }
    Some(SlicePrologue {
        graph,
        cp,
        weights,
        bias,
    })
}

fn find_optimal_tile_size(
    spatial_dim: i64,
    target: i64,
    min_tile: i64,
    stride: i64,
) -> Option<i64> {
    if min_tile <= target && target < spatial_dim {
        for tile in (min_tile..=target).rev() {
            if spatial_dim % tile == 0 && tile % stride == 0 {
                return Some(tile);
            }
        }
    }
    None
}

fn calculate_spatial_tile_config(
    channels: i64,
    h: i64,
    w: i64,
    tile_size: i64,
    min_tile: i64,
    stride: i64,
) -> (Option<i64>, Option<&'static str>) {
    let total = channels * h * w;
    if total <= tile_size {
        return (None, Some("already_fits"));
    }
    let max_tile = ((tile_size as f64) / (channels as f64)).sqrt() as i64;
    if max_tile < min_tile {
        return (None, Some("min_tile_too_large"));
    }
    let target_tile = max_tile.min(h).min(w);
    match find_optimal_tile_size(h, target_tile, min_tile, stride) {
        Some(t) => (Some(t), None),
        None => (None, Some("no_divisor")),
    }
}

fn calculate_channel_split_config(
    c_in: i64,
    _c_out: i64,
    h: i64,
    w: i64,
    tile_size: i64,
) -> Option<(i64, i64)> {
    if h == 0 || w == 0 {
        return None;
    }
    let max_ch = tile_size / (h * w);
    if max_ch >= 1 && max_ch < c_in {
        let mut num_groups = (c_in + max_ch - 1) / max_ch;
        if num_groups > 1 {
            let mut cpg = (c_in + num_groups - 1) / num_groups;
            while cpg * (num_groups - 1) >= c_in && num_groups > 1 {
                num_groups -= 1;
                cpg = (c_in + num_groups - 1) / num_groups;
            }
            if num_groups > 1 {
                return Some((num_groups, cpg));
            }
        }
    }
    None
}

pub const CONV_TILE_BUDGET: i64 = 512;
pub const POOL_TILE_BUDGET: i64 = 1024;

pub fn detect_tiling_needs(
    model: &ModelProto,
    tile_size: Option<usize>,
) -> Option<TilingDetection> {
    let graph = model.graph.as_ref()?;
    tile_size?;

    let dims_4d = get_model_dimensions(graph);

    if let Some((ref inp_name, ref out_name, c_in, h, w)) = dims_4d
        && let Some(cp) = get_conv_params(graph)
    {
        let budget = CONV_TILE_BUDGET;
        let c_out = cp.c_out;

        if is_tileable(graph) {
            let min_tile = compute_min_spatial_tile(cp.kernel, cp.dilation)?;
            let (actual_tile, _skip_reason) =
                calculate_spatial_tile_config(c_in, h, w, budget, min_tile, cp.stride[0]);

            if let Some(actual_tile) = actual_tile
                && h % actual_tile == 0
                && w % actual_tile == 0
                && actual_tile % cp.stride[0] == 0
                && actual_tile % cp.stride[1] == 0
            {
                let tiles_y = h / actual_tile;
                let tiles_x = w / actual_tile;
                if tiles_y * tiles_x >= 2 {
                    let halo = compute_halo_size(cp.pads)?;
                    return Some(TilingDetection::Spatial {
                        input_name: inp_name.clone(),
                        output_name: out_name.clone(),
                        input_names: vec![inp_name.clone()],
                        ndim: 4,
                        c_in,
                        c_out,
                        h,
                        w,
                        tile_size: actual_tile,
                        halo,
                        tiles_y,
                        tiles_x,
                        out_tile: [actual_tile / cp.stride[0], actual_tile / cp.stride[1]],
                        stride: cp.stride,
                    });
                }
            }
        }

        if is_channel_splittable(graph)
            && let Some((num_groups, cpg)) =
                calculate_channel_split_config(c_in, c_out, h, w, budget)
        {
            return Some(TilingDetection::ChannelSplit {
                input_name: inp_name.clone(),
                output_name: out_name.clone(),
                c_in,
                c_out,
                h,
                w,
                num_groups,
                channels_per_group: cpg,
            });
        }
    }

    if let Some((ref inp_name, ref out_name, c_in, h, w)) = dims_4d
        && is_spatial_tileable(graph, "MaxPool")
        && let Some(pp) = get_pool_params(graph)
    {
        let budget = POOL_TILE_BUDGET;
        let min_tile = compute_min_spatial_tile(pp.kernel, pp.dilation)?;
        let (actual_tile, _skip_reason) =
            calculate_spatial_tile_config(c_in, h, w, budget, min_tile, pp.stride[0]);

        if let Some(actual_tile) = actual_tile
            && h % actual_tile == 0
            && w % actual_tile == 0
            && actual_tile % pp.stride[0] == 0
            && actual_tile % pp.stride[1] == 0
        {
            let tiles_y = h / actual_tile;
            let tiles_x = w / actual_tile;
            if tiles_y * tiles_x >= 2 {
                let halo = compute_halo_size(pp.pads)?;
                return Some(TilingDetection::Spatial {
                    input_name: inp_name.clone(),
                    output_name: out_name.clone(),
                    input_names: vec![inp_name.clone()],
                    ndim: 4,
                    c_in,
                    c_out: c_in,
                    h,
                    w,
                    tile_size: actual_tile,
                    halo,
                    tiles_y,
                    tiles_x,
                    out_tile: [actual_tile / pp.stride[0], actual_tile / pp.stride[1]],
                    stride: pp.stride,
                });
            }
        }
    }

    if let Some(detection) = detect_elementwise_fixed_segments(graph) {
        return Some(detection);
    }

    None
}

pub const ELEMENTWISE_SEGMENT_SIZE: i64 = 1024;

fn elementwise_segment_size() -> i64 {
    std::env::var("DSPERSE_EW_SEGMENT_SIZE")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(ELEMENTWISE_SEGMENT_SIZE)
}

fn detect_elementwise_fixed_segments(graph: &GraphProto) -> Option<TilingDetection> {
    if !is_elementwise_only_slice(graph) {
        return None;
    }
    let seg_size = elementwise_segment_size();
    let out = graph.output.first()?;
    let first_inp = graph.input.first()?;
    let first_dims = onnx_proto::vi_shape(first_inp);
    if first_dims.is_empty() || first_dims.iter().any(|&d| d <= 0) {
        return None;
    }
    let total_elements = first_dims
        .iter()
        .try_fold(1i64, |acc, &d| acc.checked_mul(d))?;
    if total_elements <= seg_size {
        return None;
    }
    let last_dim = *first_dims.last().unwrap_or(&0);
    let mut effective_seg_size = seg_size;
    for init in &graph.initializer {
        let vol: i64 = init.dims.iter().product();
        if vol <= 1 || vol == seg_size {
            continue;
        }
        if init.dims.len() == 1 && init.dims[0] == last_dim && last_dim > 0 {
            effective_seg_size = last_dim;
            continue;
        }
        return None;
    }
    let seg_size = effective_seg_size;
    let mut input_names = Vec::with_capacity(graph.input.len());
    for inp in &graph.input {
        let d = onnx_proto::vi_shape(inp);
        if d != first_dims || d.iter().any(|&v| v <= 0) {
            return None;
        }
        input_names.push(inp.name.clone());
    }
    #[allow(clippy::manual_div_ceil)]
    let num_segments = (total_elements + seg_size - 1) / seg_size;
    if num_segments < 2 {
        return None;
    }
    let primary_name = input_names[0].clone();
    Some(TilingDetection::FixedSegment {
        input_name: primary_name,
        output_name: out.name.clone(),
        input_names,
        total_elements,
        segment_size: seg_size,
        num_segments,
        original_shape: first_dims,
    })
}

pub const MAX_ESTIMATED_CONSTRAINTS: u64 = 750_000;

/// Return the smallest divisor of `dim` that is >= `target`.  Returns
/// `None` if no such divisor exists in `(0, dim]`, which is the
/// signal to refuse the dim-split: pad-then-trim on the last group
/// would inject zeros into reductions on non-split axes (Softmax,
/// LayerNorm, ReduceMean, etc.) and contaminate the unpadded
/// region's outputs.
fn smallest_divisor_at_least(dim: usize, target: usize) -> Option<usize> {
    if dim == 0 || target == 0 {
        return None;
    }
    let target = target.min(dim);
    (target..=dim).find(|&g| dim.is_multiple_of(g))
}

#[derive(Debug, Clone)]
pub struct DimSplitDetection {
    pub split_kind: DimSplitKind,
    pub split_dim: usize,
    pub dim_size: usize,
    pub num_groups: usize,
    pub elements_per_group: usize,
    pub input_name: String,
    pub output_name: String,
    pub concat_axis: usize,
    pub estimated_constraints: u64,
    pub weight_name: Option<String>,
    pub k_dim: usize,
    pub n_dim: usize,
    pub k_chunks: usize,
}

pub fn estimate_slice_constraints(nodes: &[NodeProto], shapes: &HashMap<String, Vec<i64>>) -> u64 {
    let config = jstprove_circuits::api::EstimationConfig::bn254_defaults();
    let mut total: u64 = 0;

    let to_usize_shape = |name: &String| -> Vec<usize> {
        shapes
            .get(name)
            .map(|s| s.iter().map(|&d| d.max(1) as usize).collect())
            .unwrap_or_default()
    };

    for node in nodes {
        let input_shapes: Vec<Vec<usize>> = node.input.iter().map(&to_usize_shape).collect();
        let output_shapes: Vec<Vec<usize>> = node.output.iter().map(&to_usize_shape).collect();

        let cost = jstprove_circuits::api::estimate_op_constraints(
            &node.op_type,
            &input_shapes,
            &output_shapes,
            &config,
        );
        total = total.saturating_add(cost);
    }
    total
}

pub fn detect_dim_split(
    nodes: &[NodeProto],
    shapes: &HashMap<String, Vec<i64>>,
    initializer_names: &HashSet<String>,
) -> Option<DimSplitDetection> {
    let estimated = estimate_slice_constraints(nodes, shapes);
    if estimated <= MAX_ESTIMATED_CONSTRAINTS {
        return None;
    }

    let target_groups = estimated.div_ceil(MAX_ESTIMATED_CONSTRAINTS) as usize;

    for (idx, node) in nodes.iter().enumerate() {
        if matches!(node.op_type.as_str(), "MatMul" | "Gemm") {
            // Gemm with a bias (input C) is not yet supported by the dim-split
            // template builder; skip so the template construction downstream
            // stays in sync with the detector.
            if node.op_type == "Gemm" && node.input.get(2).is_some_and(|s: &String| !s.is_empty()) {
                continue;
            }
            // The dim-split runner replaces the entire slice execution with
            // the patched MatMul template and only writes ds.output_name to
            // the tensor cache. If this MatMul/Gemm output is consumed by a
            // later node in the same slice, those downstream ops would never
            // execute and the slice would publish the wrong tensor. Decline
            // and let the search continue or fall through to other paths.
            let Some(node_out) = node.output.first().filter(|s| !s.is_empty()) else {
                continue;
            };
            let consumed_downstream = nodes
                .iter()
                .skip(idx + 1)
                .any(|later| later.input.iter().any(|i| i == node_out));
            if consumed_downstream {
                continue;
            }
            let Some(weight_name) = node.input.get(1) else {
                continue;
            };
            if !initializer_names.contains(weight_name) {
                continue;
            }
            let Some(weight_shape) = shapes.get(weight_name) else {
                continue;
            };
            if weight_shape.len() != 2 {
                continue;
            }
            // Gemm with transA=1 transposes the activation matrix, which the
            // single-row sequence tile and the rank-2 template do not model.
            // Skip so detection stays consistent with the template builder.
            if node.op_type == "Gemm"
                && super::onnx_proto::get_attribute_int(node, "transA").unwrap_or(0) == 1
            {
                continue;
            }
            let trans_b = node.op_type == "Gemm"
                && super::onnx_proto::get_attribute_int(node, "transB").unwrap_or(0) == 1;
            let k_dim = if trans_b {
                weight_shape[1] as usize
            } else {
                weight_shape[0] as usize
            };
            let n_dim = if trans_b {
                weight_shape[0] as usize
            } else {
                weight_shape[1] as usize
            };
            let Some(inp_shape) = node.input.first().and_then(|name| shapes.get(name)) else {
                continue;
            };
            let total_rows: usize = inp_shape
                .iter()
                .take(inp_shape.len().saturating_sub(1))
                .map(|&d| d.max(1) as usize)
                .product();
            if total_rows == 0 || k_dim == 0 || n_dim == 0 {
                continue;
            }
            let row_cost = k_dim.saturating_mul(n_dim).saturating_mul(2);
            let max_per_chunk = MAX_ESTIMATED_CONSTRAINTS as usize;
            // Even with k_chunks == k_dim (chunk_size == 1), the per-chunk
            // cost is at minimum n_dim * 2. If that alone exceeds the budget
            // the split is infeasible; let the caller fall through to other
            // detection paths.
            if n_dim.saturating_mul(2) > max_per_chunk {
                continue;
            }
            let mut k_chunks = if row_cost > max_per_chunk {
                row_cost.div_ceil(max_per_chunk).max(1)
            } else {
                1
            };
            k_chunks = k_chunks.min(k_dim);
            while k_chunks < k_dim
                && k_dim
                    .div_ceil(k_chunks)
                    .saturating_mul(n_dim)
                    .saturating_mul(2)
                    > max_per_chunk
            {
                k_chunks += 1;
            }
            if total_rows == 1 && k_chunks == 1 {
                continue;
            }
            let Some(input_name) = node.input.first().filter(|s| !s.is_empty()).cloned() else {
                continue;
            };
            let Some(output_name) = node.output.first().filter(|s| !s.is_empty()).cloned() else {
                continue;
            };
            return Some(DimSplitDetection {
                split_kind: DimSplitKind::MatMulOutputDim,
                split_dim: 0,
                dim_size: total_rows,
                num_groups: total_rows,
                elements_per_group: 1,
                input_name,
                output_name,
                concat_axis: 0,
                estimated_constraints: estimated,
                weight_name: Some(weight_name.clone()),
                k_dim,
                n_dim,
                k_chunks,
            });
        }
    }

    for node in nodes {
        if node.op_type == "Softmax" {
            let Some(softmax_in) = node.input.first().and_then(|name| shapes.get(name)) else {
                continue;
            };
            if softmax_in.len() != 4 {
                continue;
            }
            let softmax_axis = onnx_proto::get_attribute_int(node, "axis").unwrap_or(-1);
            let softmax_axis_abs = if softmax_axis < 0 {
                (softmax_in.len() as i64 + softmax_axis).max(0) as usize
            } else {
                softmax_axis as usize
            };
            // Find the attention-block input among the slice inputs: the
            // first non-init tensor whose rank matches the softmax input
            // rank (Q/V-like activation).
            let attn_input = nodes.iter().flat_map(|n| n.input.iter()).find(|name| {
                !name.is_empty()
                    && !initializer_names.contains(name.as_str())
                    && shapes.get(*name).is_some_and(|s| s.len() == 4 && s[0] > 0)
            });
            let Some(attn_input_name) = attn_input.cloned() else {
                continue;
            };
            let Some(attn_shape) = shapes.get(&attn_input_name) else {
                continue;
            };
            // Choose the dim (among 0..rank) that is not the softmax-reduction
            // axis and yields the highest axis size; that axis gives the
            // most groups and the lowest per-group cost.
            let mut best: Option<(usize, usize, DimSplitKind)> = None;
            for (d, &axis_len) in attn_shape.iter().enumerate() {
                if d == softmax_axis_abs {
                    continue;
                }
                let dim_size = axis_len.max(1) as usize;
                if dim_size < 2 {
                    continue;
                }
                let kind = if d == 1 {
                    DimSplitKind::HeadDim
                } else {
                    DimSplitKind::BatchDim
                };
                let better = best.as_ref().is_none_or(|(_, sz, _)| dim_size > *sz);
                if better {
                    best = Some((d, dim_size, kind));
                }
            }
            let Some((split_dim, dim_size, split_kind)) = best else {
                continue;
            };
            let num_groups = match smallest_divisor_at_least(dim_size, target_groups) {
                Some(g) => g,
                None => continue,
            };
            let elements_per_group = dim_size / num_groups;
            let output_name = nodes
                .last()
                .and_then(|n| n.output.first())
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| node.output.first().cloned().unwrap_or_default());
            if output_name.is_empty() {
                continue;
            }
            // Reject the split when axis tracing through the slice cannot
            // prove the split axis lands at the same position (and size)
            // in the final output.  Shape-reordering ops (Reshape,
            // Transpose, Flatten, Squeeze, Unsqueeze, Concat on the
            // split axis) are non-trivial to follow here, so we require
            // the output shape to match the attention input at split_dim.
            let Some(out_shape) = shapes.get(&output_name) else {
                continue;
            };
            if out_shape.len() != attn_shape.len() || out_shape[split_dim] != attn_shape[split_dim]
            {
                continue;
            }
            return Some(DimSplitDetection {
                split_kind,
                split_dim,
                dim_size,
                num_groups,
                elements_per_group,
                input_name: attn_input_name,
                output_name,
                concat_axis: split_dim,
                estimated_constraints: estimated,
                weight_name: None,
                k_dim: 0,
                n_dim: 0,
                k_chunks: 1,
            });
        }
    }

    let first_non_init_input = nodes.first().and_then(|n| {
        n.input
            .iter()
            .find(|name| !name.is_empty() && !initializer_names.contains(name.as_str()))
    });
    let first_input_shape = first_non_init_input.and_then(|name| shapes.get(name))?;
    if first_input_shape.is_empty() {
        return None;
    }

    // Conv / ConvTranspose / Pooling are not separable along arbitrary
    // input axes: splitting the input channel or the spatial dimensions
    // produces semantically incorrect per-group outputs. The dedicated
    // detection paths (conv spatial tiling, channel splitting) handle
    // these ops correctly; this generic fallback refuses to emit a
    // split for them.  MatMul / Gemm are *not* listed here: their
    // dedicated dim-split-k path handles the K-axis split when the
    // weight is an initializer, but non-terminal MatMul/Gemm slices or
    // slices whose weight is a runtime tensor still benefit from the
    // generic axis-0 (batch) fallback, which is always semantically
    // sound because the batch dimension is independent across rows.
    for node in nodes {
        if matches!(
            node.op_type.as_str(),
            "Conv"
                | "ConvTranspose"
                | "AveragePool"
                | "MaxPool"
                | "GlobalAveragePool"
                | "GlobalMaxPool"
                | "LRN"
        ) {
            return None;
        }
    }

    // Find the deepest split_dim that is still compatible with every
    // normalization-style op in the slice.  Splitting a later axis produces
    // more groups and a smaller per-group cost without violating op semantics.
    let rank = first_input_shape.len();
    // If the slice contains any axis-reordering op (Transpose) AND any
    // axis-sensitive normalization op (LayerNormalization / Softmax),
    // we can no longer cheaply trace which axis the normalization
    // really runs on after the reorder.  Restrict the split to axis 0
    // (always the batch dim, always semantically sound) so we never
    // emit a split that lands on the post-Transpose normalization axis.
    let has_transpose = nodes.iter().any(|n| n.op_type == "Transpose");
    let has_norm = nodes.iter().any(|n| {
        matches!(
            n.op_type.as_str(),
            "LayerNormalization" | "Softmax" | "LogSoftmax"
        )
    });
    let mut max_allowed = if has_transpose && has_norm { 1 } else { rank };
    for node in nodes {
        match node.op_type.as_str() {
            "LayerNormalization" => {
                let axis = onnx_proto::get_attribute_int(node, "axis").unwrap_or(-1);
                let resolved = if axis < 0 {
                    (rank as i64 + axis).max(0) as usize
                } else {
                    (axis as usize).min(rank)
                };
                if resolved < max_allowed {
                    max_allowed = resolved;
                }
            }
            "Softmax" | "LogSoftmax" => {
                let axis = onnx_proto::get_attribute_int(node, "axis").unwrap_or(-1);
                let resolved = if axis < 0 {
                    (rank as i64 + axis).max(0) as usize
                } else {
                    (axis as usize).min(rank.saturating_sub(1))
                };
                if resolved < max_allowed {
                    max_allowed = resolved;
                }
            }
            "BatchNormalization" => {
                if max_allowed > 0 {
                    max_allowed = 0;
                }
            }
            _ => {}
        }
    }
    if max_allowed == 0 {
        return None;
    }

    let mut best: Option<(usize, usize)> = None;
    for (d, &axis_len) in first_input_shape.iter().enumerate().take(max_allowed) {
        let dim = axis_len.max(1) as usize;
        if dim <= 1 {
            continue;
        }
        if best.map(|(_, size)| dim > size).unwrap_or(true) {
            best = Some((d, dim));
        }
    }
    let (split_dim, dim_size) = best?;

    let num_groups = smallest_divisor_at_least(dim_size, target_groups)?;
    let elements_per_group = dim_size / num_groups;
    let input_name = first_non_init_input.cloned()?;
    let output_name = nodes
        .last()
        .and_then(|n| n.output.first())
        .filter(|s| !s.is_empty())
        .cloned()?;
    // Require the final output shape to preserve rank and the split
    // axis size; otherwise an intermediate op (Reshape, Transpose,
    // Flatten, Squeeze, Unsqueeze) has reordered the axes and
    // concat_axis=split_dim would splice the groups into the wrong
    // output dimension.  Tracing the axis through an arbitrary chain
    // of shape ops is out of scope here, so we conservatively reject.
    let out_shape = shapes.get(&output_name)?;
    if out_shape.len() != first_input_shape.len()
        || out_shape[split_dim] != first_input_shape[split_dim]
    {
        return None;
    }
    Some(DimSplitDetection {
        split_kind: DimSplitKind::BatchDim,
        split_dim,
        dim_size,
        num_groups,
        elements_per_group,
        input_name,
        output_name,
        concat_axis: split_dim,
        estimated_constraints: estimated,
        weight_name: None,
        k_dim: 0,
        n_dim: 0,
        k_chunks: 1,
    })
}

#[derive(Debug, Clone)]
pub enum TilingDetection {
    Spatial {
        input_name: String,
        output_name: String,
        input_names: Vec<String>,
        ndim: i64,
        c_in: i64,
        c_out: i64,
        h: i64,
        w: i64,
        tile_size: i64,
        halo: [i64; 4],
        tiles_y: i64,
        tiles_x: i64,
        out_tile: [i64; 2],
        stride: [i64; 2],
    },
    ChannelSplit {
        input_name: String,
        output_name: String,
        c_in: i64,
        c_out: i64,
        h: i64,
        w: i64,
        num_groups: i64,
        channels_per_group: i64,
    },
    FixedSegment {
        input_name: String,
        output_name: String,
        input_names: Vec<String>,
        total_elements: i64,
        segment_size: i64,
        num_segments: i64,
        original_shape: Vec<i64>,
    },
}

struct SpatialTileGeometry {
    c_in: i64,
    c_out: i64,
    tile_h: i64,
    tile_w: i64,
    out_h: i64,
    out_w: i64,
}

fn compute_spatial_tile_geometry(
    graph: &GraphProto,
    pads: [i64; 4],
    kernel: [i64; 2],
    dilation: [i64; 2],
    stride: [i64; 2],
    tile_size: i64,
    c_out_override: Option<i64>,
) -> Result<SpatialTileGeometry> {
    let halo = compute_halo_size(pads).ok_or_else(|| {
        crate::error::DsperseError::Slicer("spatial tile: invalid pad values".to_string())
    })?;
    let tile_h = tile_size
        .checked_add(halo[0])
        .and_then(|v| v.checked_add(halo[2]))
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "spatial tile: tile_h overflow (tile_size={tile_size}, halo={:?})",
                halo
            ))
        })?;
    let tile_w = tile_size
        .checked_add(halo[1])
        .and_then(|v| v.checked_add(halo[3]))
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "spatial tile: tile_w overflow (tile_size={tile_size}, halo={:?})",
                halo
            ))
        })?;
    let (out_h, out_w) = conv_output_hw(tile_h, tile_w, [0, 0, 0, 0], kernel, dilation, stride)
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "spatial tile: invalid output dims for tile_h={tile_h}, tile_w={tile_w}, stride={stride:?}, kernel={kernel:?}"
            ))
        })?;
    let c_in = graph
        .input
        .first()
        .map(onnx_proto::vi_shape)
        .and_then(|s| (s.len() == 4 && s[1] > 0).then_some(s[1]))
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(
                "spatial tile: unable to determine input channels".to_string(),
            )
        })?;
    let c_out = c_out_override.unwrap_or(c_in);
    Ok(SpatialTileGeometry {
        c_in,
        c_out,
        tile_h,
        tile_w,
        out_h,
        out_w,
    })
}

struct TileModelSpec {
    nodes: Vec<NodeProto>,
    input: onnx_proto::ValueInfoProto,
    output: onnx_proto::ValueInfoProto,
    initializers: Vec<onnx_proto::TensorProto>,
    out_hw: [i64; 2],
}

fn save_tile_model(
    model: &ModelProto,
    spec: TileModelSpec,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<TileSliceResult> {
    let graph = onnx_proto::make_graph(
        &format!("tile_{slice_idx}"),
        spec.nodes,
        vec![spec.input],
        vec![spec.output],
        spec.initializers,
    );
    let tile_model = onnx_proto::make_model(graph, model_opset(model));
    let tiles_dir = output_dir.join("tiles");
    std::fs::create_dir_all(&tiles_dir)
        .map_err(|e| crate::error::DsperseError::io(e, &tiles_dir))?;
    let onnx_path = tiles_dir.join("tile.onnx");
    onnx_proto::save_model(&tile_model, &onnx_path)?;
    Ok(TileSliceResult {
        path: format!("slice_{slice_idx}/payload/tiles/tile.onnx"),
        conv_out: spec.out_hw,
    })
}

pub fn create_tile_slice(
    model: &ModelProto,
    tile_size: i64,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<TileSliceResult> {
    if tile_size <= 0 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_tile_slice: tile_size must be > 0, got {tile_size}"
        )));
    }
    let SlicePrologue {
        graph,
        cp,
        weights,
        bias,
    } = extract_slice_prologue(model).ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_tile_slice: failed to extract slice prologue".to_string(),
        )
    })?;
    let conv_node = &graph.node[cp.node_idx];
    let weights = weights.ok_or_else(|| {
        crate::error::DsperseError::Slicer("create_tile_slice: conv weights not found".to_string())
    })?;

    let cfg_c_in = cp.c_in.checked_mul(cp.group).filter(|&v| v > 0);
    let geom = compute_spatial_tile_geometry(
        graph,
        cp.pads,
        cp.kernel,
        cp.dilation,
        cp.stride,
        tile_size,
        Some(weights.dims[0]),
    )?;
    if let Some(c) = cfg_c_in
        && geom.c_in != c
    {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_tile_slice: graph c_in ({}) != weight c_in*group ({c})",
            geom.c_in
        )));
    }

    let x = onnx_proto::make_tensor_value_info(
        "tile_in",
        TensorProto::FLOAT,
        &[1, geom.c_in, geom.tile_h, geom.tile_w],
    );
    let y = onnx_proto::make_tensor_value_info(
        "tile_out",
        TensorProto::FLOAT,
        &[1, geom.c_out, geom.out_h, geom.out_w],
    );

    let mut initializers = vec![onnx_proto::make_tensor(
        "W",
        TensorProto::FLOAT,
        &weights.dims,
        weights.data,
    )];
    let mut conv_inputs = vec!["tile_in".to_string(), "W".to_string()];

    if let Some(bias_data) = &bias {
        let bias_dims = [geom.c_out];
        initializers.push(onnx_proto::make_tensor(
            "B",
            TensorProto::FLOAT,
            &bias_dims,
            bias_data.clone(),
        ));
        conv_inputs.push("B".to_string());
    }

    let mut conv_attrs = vec![
        onnx_proto::make_attribute_ints("kernel_shape", &cp.kernel),
        onnx_proto::make_attribute_ints("strides", &cp.stride),
        onnx_proto::make_attribute_ints("pads", &[0, 0, 0, 0]),
        onnx_proto::make_attribute_ints("dilations", &cp.dilation),
    ];
    if cp.group != 1 {
        conv_attrs.push(onnx_proto::make_attribute_int("group", cp.group));
    }

    let mut nodes = vec![onnx_proto::make_node(
        "Conv",
        conv_inputs,
        vec!["conv_out".to_string()],
        conv_attrs,
    )];

    integrate_extra_ops(graph, conv_node, &mut initializers, &mut nodes)?;

    save_tile_model(
        model,
        TileModelSpec {
            nodes,
            input: x,
            output: y,
            initializers,
            out_hw: [geom.out_h, geom.out_w],
        },
        slice_idx,
        output_dir,
    )
}

pub fn create_pool_tile_slice(
    model: &ModelProto,
    tile_size: i64,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<TileSliceResult> {
    if tile_size <= 0 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_pool_tile_slice: tile_size must be > 0, got {tile_size}"
        )));
    }
    let graph = model.graph.as_ref().ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_pool_tile_slice: model.graph is None".to_string(),
        )
    })?;
    let pp = get_pool_params(graph).ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_pool_tile_slice: no MaxPool node found".to_string(),
        )
    })?;
    let pool_node = &graph.node[pp.node_idx];

    let geom = compute_spatial_tile_geometry(
        graph,
        pp.pads,
        pp.kernel,
        pp.dilation,
        pp.stride,
        tile_size,
        None,
    )?;

    let x = onnx_proto::make_tensor_value_info(
        "tile_in",
        TensorProto::FLOAT,
        &[1, geom.c_in, geom.tile_h, geom.tile_w],
    );
    let y = onnx_proto::make_tensor_value_info(
        "tile_out",
        TensorProto::FLOAT,
        &[1, geom.c_out, geom.out_h, geom.out_w],
    );

    let pool_attrs = vec![
        onnx_proto::make_attribute_ints("kernel_shape", &pp.kernel),
        onnx_proto::make_attribute_ints("strides", &pp.stride),
        onnx_proto::make_attribute_ints("pads", &[0, 0, 0, 0]),
        onnx_proto::make_attribute_ints("dilations", &pp.dilation),
    ];
    let mut nodes = vec![onnx_proto::make_node(
        "MaxPool",
        vec!["tile_in".to_string()],
        vec!["pool_out".to_string()],
        pool_attrs,
    )];

    let mut initializers = Vec::new();
    integrate_extra_ops(graph, pool_node, &mut initializers, &mut nodes)?;

    save_tile_model(
        model,
        TileModelSpec {
            nodes,
            input: x,
            output: y,
            initializers,
            out_hw: [geom.out_h, geom.out_w],
        },
        slice_idx,
        output_dir,
    )
}

fn integrate_extra_ops(
    graph: &GraphProto,
    primary_node: &NodeProto,
    initializers: &mut Vec<onnx_proto::TensorProto>,
    nodes: &mut Vec<NodeProto>,
) -> crate::error::Result<()> {
    let primary_op = primary_node.op_type.as_str();
    let orig_input_name = graph.input.first().map(|i| i.name.as_str()).unwrap_or("");

    let extra: Vec<&NodeProto> = graph
        .node
        .iter()
        .filter(|n| n.op_type != primary_op)
        .collect();

    if extra.is_empty() {
        let last = nodes.last_mut().ok_or_else(|| {
            crate::error::DsperseError::Slicer(
                "integrate_extra_ops: no nodes to set output on".into(),
            )
        })?;
        let out = last.output.get_mut(0).ok_or_else(|| {
            crate::error::DsperseError::Slicer(
                "integrate_extra_ops: last node has no outputs".into(),
            )
        })?;
        *out = "tile_out".to_string();
        return Ok(());
    }

    let mut primary_weight_names: HashSet<String> = HashSet::new();
    for inp in primary_node.input.iter().skip(1) {
        primary_weight_names.insert(inp.clone());
    }

    for init in &graph.initializer {
        if !primary_weight_names.contains(&init.name) {
            initializers.push(init.clone());
        }
    }

    let primary_outputs: HashSet<String> = graph
        .node
        .iter()
        .filter(|n| n.op_type == primary_op)
        .flat_map(|n| n.output.iter().cloned())
        .collect();

    let primary_out_wire = nodes
        .last()
        .and_then(|n| n.output.first())
        .cloned()
        .unwrap_or_else(|| format!("{}_out", primary_op.to_lowercase()));

    for (i, orig_node) in extra.iter().enumerate() {
        let new_inputs: Vec<String> = orig_node
            .input
            .iter()
            .map(|inp| {
                if primary_outputs.contains(inp) {
                    primary_out_wire.clone()
                } else if inp == orig_input_name {
                    "tile_in".to_string()
                } else {
                    inp.clone()
                }
            })
            .collect();

        let is_last = i == extra.len() - 1;
        let new_outputs = if is_last {
            vec!["tile_out".to_string()]
        } else {
            orig_node.output.clone()
        };

        nodes.push(NodeProto {
            op_type: orig_node.op_type.clone(),
            input: new_inputs,
            output: new_outputs,
            attribute: orig_node.attribute.clone(),
            name: String::new(),
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn create_channel_group_slice(
    model: &ModelProto,
    prologue: &SlicePrologue<'_>,
    group_idx: usize,
    c_start: i64,
    c_end: i64,
    h_in: i64,
    w_in: i64,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<ChannelGroupInfo> {
    let cp = &prologue.cp;
    if c_start < 0 || c_end < 0 || c_start >= c_end {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_channel_group_slice: invalid channel range c_start={c_start}, c_end={c_end}"
        )));
    }
    let weights = prologue.weights.as_ref().ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_channel_group_slice: conv weights not found".to_string(),
        )
    })?;

    let c_group = c_end - c_start;
    let (h_out, w_out) = conv_output_hw(h_in, w_in, cp.pads, cp.kernel, cp.dilation, cp.stride)
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "create_channel_group_slice: invalid output dims for h_in={h_in}, w_in={w_in}"
            ))
        })?;
    let c_out = cp.c_out;

    let input_name = format!("group_{group_idx}_in");
    let output_name = format!("group_{group_idx}_out");

    let x = onnx_proto::make_tensor_value_info(
        &input_name,
        TensorProto::FLOAT,
        &[1, c_group, h_in, w_in],
    );
    let y = onnx_proto::make_tensor_value_info(
        &output_name,
        TensorProto::FLOAT,
        &[1, c_out, h_out, w_out],
    );

    let c_start_uz = i64_to_usize(c_start, "create_channel_group_slice", "c_start")?;
    let c_end_uz = i64_to_usize(c_end, "create_channel_group_slice", "c_end")?;
    let sliced_weights = slice_weights(weights, c_start_uz, c_end_uz)?;

    let w_tensor = onnx_proto::make_tensor(
        "W",
        TensorProto::FLOAT,
        &sliced_weights.dims,
        sliced_weights.data,
    );

    let mut conv_attrs = vec![
        onnx_proto::make_attribute_ints("kernel_shape", &cp.kernel),
        onnx_proto::make_attribute_ints("strides", &cp.stride),
        onnx_proto::make_attribute_ints("pads", &cp.pads),
        onnx_proto::make_attribute_ints("dilations", &cp.dilation),
    ];
    if cp.group != 1 {
        conv_attrs.push(onnx_proto::make_attribute_int("group", cp.group));
    }

    let node = onnx_proto::make_node(
        "Conv",
        vec![input_name, "W".to_string()],
        vec![output_name],
        conv_attrs,
    );

    let graph_proto = onnx_proto::make_graph(
        &format!("channel_group_{slice_idx}_{group_idx}"),
        vec![node],
        vec![x],
        vec![y],
        vec![w_tensor],
    );
    let group_model = onnx_proto::make_model(graph_proto, model_opset(model));

    let groups_dir = output_dir.join("channel_groups");
    std::fs::create_dir_all(&groups_dir)
        .map_err(|e| crate::error::DsperseError::io(e, &groups_dir))?;
    let onnx_path = groups_dir.join(format!("group_{group_idx}.onnx"));
    onnx_proto::save_model(&group_model, &onnx_path)?;

    Ok(ChannelGroupInfo {
        group_idx,
        c_start: c_start_uz,
        c_end: c_end_uz,
        path: format!("slice_{slice_idx}/payload/channel_groups/group_{group_idx}.onnx"),
        jstprove_circuit_path: None,
        jstprove_settings_path: None,
    })
}

fn i64_to_usize(val: i64, ctx: &str, name: &str) -> Result<usize> {
    usize::try_from(val).map_err(|_| {
        crate::error::DsperseError::Slicer(format!("{ctx}: {name} ({val}) out of range for usize"))
    })
}

fn checked_dim_product(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |acc, &f| {
        acc.checked_mul(f).ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "slice_weights: dimension product overflow (factors={factors:?})"
            ))
        })
    })
}

fn slice_weights(weights: &WeightInfo, c_start: usize, c_end: usize) -> Result<WeightInfo> {
    if weights.dims.len() < 4 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "slice_weights: expected >= 4 dims, got {}",
            weights.dims.len()
        )));
    }
    let to_usize = |dim: i64, name: &str| -> Result<usize> {
        usize::try_from(dim).map_err(|_| {
            crate::error::DsperseError::Slicer(format!(
                "slice_weights: {name} dimension {dim} is negative or too large"
            ))
        })
    };
    let c_out = to_usize(weights.dims[0], "c_out")?;
    let c_in = to_usize(weights.dims[1], "c_in")?;
    let kh = to_usize(weights.dims[2], "kh")?;
    let kw = to_usize(weights.dims[3], "kw")?;
    let expected_len = checked_dim_product(&[c_out, c_in, kh, kw])?;
    if weights.data.len() != expected_len {
        return Err(crate::error::DsperseError::Slicer(format!(
            "slice_weights: data length {} != expected {} (dims={:?})",
            weights.data.len(),
            expected_len,
            weights.dims
        )));
    }
    if c_start >= c_end {
        return Err(crate::error::DsperseError::Slicer(format!(
            "slice_weights: c_start ({c_start}) >= c_end ({c_end})"
        )));
    }
    if c_end > c_in {
        return Err(crate::error::DsperseError::Slicer(format!(
            "slice_weights: c_end ({c_end}) exceeds c_in ({c_in})"
        )));
    }
    let c_group = c_end - c_start;
    let capacity = checked_dim_product(&[c_out, c_group, kh, kw])?;
    let stride_cin = checked_dim_product(&[c_in, kh, kw])?;
    let stride_kh = checked_dim_product(&[kh, kw])?;

    let mut sliced = Vec::with_capacity(capacity);
    for o in 0..c_out {
        for c in c_start..c_end {
            for h in 0..kh {
                for w_idx in 0..kw {
                    let idx = o * stride_cin + c * stride_kh + h * kw + w_idx;
                    sliced.push(weights.data[idx]);
                }
            }
        }
    }

    Ok(WeightInfo {
        data: sliced,
        dims: vec![c_out as i64, c_group as i64, kh as i64, kw as i64],
    })
}

fn save_conv_bias(
    prologue: &SlicePrologue<'_>,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<Option<String>> {
    let Some(bias_data) = &prologue.bias else {
        return Ok(None);
    };

    let groups_dir = output_dir.join("channel_groups");
    std::fs::create_dir_all(&groups_dir)
        .map_err(|e| crate::error::DsperseError::io(e, &groups_dir))?;

    let bias_bytes = rmp_serde::to_vec_named(&bias_data)?;
    let bias_path = groups_dir.join("bias.msgpack");
    std::fs::write(&bias_path, bias_bytes)
        .map_err(|e| crate::error::DsperseError::io(e, &bias_path))?;

    Ok(Some(format!(
        "slice_{slice_idx}/payload/channel_groups/bias.msgpack"
    )))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_channel_splitting(
    model: &ModelProto,
    cfg: &ChannelSplitParams,
    input_name: &str,
    output_name: &str,
    output_dir: &Path,
) -> Result<ChannelSplitInfo> {
    let &ChannelSplitParams {
        c_in,
        c_out,
        num_groups,
        channels_per_group,
        h,
        w,
        slice_idx,
    } = cfg;
    if c_in <= 0 || c_out <= 0 || num_groups <= 0 || channels_per_group <= 0 || h <= 0 || w <= 0 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "apply_channel_splitting: invalid ChannelSplitParams (c_in={c_in}, c_out={c_out}, num_groups={num_groups}, channels_per_group={channels_per_group}, h={h}, w={w})"
        )));
    }
    let covered = num_groups.checked_mul(channels_per_group).ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "apply_channel_splitting: num_groups * channels_per_group overflow".to_string(),
        )
    })?;
    if covered < c_in {
        return Err(crate::error::DsperseError::Slicer(format!(
            "apply_channel_splitting: cfg covers only {covered} input channels, expected at least {c_in}",
        )));
    }
    let last_group_start = (num_groups - 1)
        .checked_mul(channels_per_group)
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(
                "apply_channel_splitting: group start computation overflow".to_string(),
            )
        })?;
    if last_group_start >= c_in {
        return Err(crate::error::DsperseError::Slicer(format!(
            "apply_channel_splitting: cfg creates empty trailing groups (last_start={last_group_start}, c_in={c_in})"
        )));
    }
    let prologue = extract_slice_prologue(model).ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "apply_channel_splitting: failed to extract slice prologue from model".to_string(),
        )
    })?;

    let (_, _, model_c_in, model_h, model_w) =
        get_model_dimensions(prologue.graph).ok_or_else(|| {
            crate::error::DsperseError::Slicer(
                "apply_channel_splitting: unable to determine model dimensions".to_string(),
            )
        })?;
    let model_c_out = prologue.cp.c_out;
    if prologue.cp.group != 1 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "apply_channel_splitting: unsupported Conv group={}, expected 1",
            prologue.cp.group
        )));
    }
    if prologue.cp.c_in != model_c_in {
        return Err(crate::error::DsperseError::Slicer(format!(
            "apply_channel_splitting: weight/model c_in mismatch (weights c_in={}, model c_in={})",
            prologue.cp.c_in, model_c_in
        )));
    }
    if model_c_in != c_in || model_c_out != c_out || model_h != h || model_w != w {
        return Err(crate::error::DsperseError::Slicer(format!(
            "apply_channel_splitting: cfg dims (c_in={c_in}, c_out={c_out}, h={h}, w={w}) mismatch model dims (c_in={model_c_in}, c_out={model_c_out}, h={model_h}, w={model_w})"
        )));
    }

    let (out_h, out_w) = conv_output_hw(
        h,
        w,
        prologue.cp.pads,
        prologue.cp.kernel,
        prologue.cp.dilation,
        prologue.cp.stride,
    )
    .ok_or_else(|| {
        crate::error::DsperseError::Slicer(format!(
            "apply_channel_splitting: invalid conv output dimensions for h={h}, w={w}, stride={:?}, kernel={:?}",
            prologue.cp.stride, prologue.cp.kernel
        ))
    })?;

    let groups_dir = output_dir.join("channel_groups");
    let cleanup = || {
        if groups_dir.exists() {
            let _ = std::fs::remove_dir_all(&groups_dir);
        }
    };

    let mut groups = Vec::new();
    for g in 0..num_groups {
        let c_start = g * channels_per_group;
        let c_end = ((g + 1) * channels_per_group).min(c_in);

        let g_uz = i64_to_usize(g, "apply_channel_splitting", "group_idx").inspect_err(|_| {
            cleanup();
        })?;
        let group_info = match create_channel_group_slice(
            model, &prologue, g_uz, c_start, c_end, h, w, slice_idx, output_dir,
        ) {
            Ok(info) => info,
            Err(e) => {
                cleanup();
                return Err(e);
            }
        };
        groups.push(group_info);
    }

    let bias_path = match save_conv_bias(&prologue, slice_idx, output_dir) {
        Ok(p) => p,
        Err(e) => {
            cleanup();
            return Err(e);
        }
    };

    let ctx = "apply_channel_splitting";
    let c_in_uz = i64_to_usize(c_in, ctx, "c_in").inspect_err(|_| cleanup())?;
    let c_out_uz = i64_to_usize(c_out, ctx, "c_out").inspect_err(|_| cleanup())?;
    let num_groups_uz = i64_to_usize(num_groups, ctx, "num_groups").inspect_err(|_| cleanup())?;
    let cpg_uz =
        i64_to_usize(channels_per_group, ctx, "channels_per_group").inspect_err(|_| cleanup())?;
    let h_uz = i64_to_usize(h, ctx, "h").inspect_err(|_| cleanup())?;
    let w_uz = i64_to_usize(w, ctx, "w").inspect_err(|_| cleanup())?;
    let out_h_uz = i64_to_usize(out_h, ctx, "out_h").inspect_err(|_| cleanup())?;
    let out_w_uz = i64_to_usize(out_w, ctx, "out_w").inspect_err(|_| cleanup())?;
    Ok(ChannelSplitInfo {
        slice_idx,
        c_in: c_in_uz,
        c_out: c_out_uz,
        num_groups: num_groups_uz,
        channels_per_group: cpg_uz,
        input_name: input_name.to_string(),
        output_name: output_name.to_string(),
        h: h_uz,
        w: w_uz,
        out_h: out_h_uz,
        out_w: out_w_uz,
        groups,
        bias_path,
    })
}

pub fn create_dim_split_template(
    model: &ModelProto,
    info: &crate::schema::tiling::DimSplitInfo,
    output_dir: &Path,
    traced_shapes: Option<&HashMap<String, Vec<i64>>>,
) -> Result<std::path::PathBuf> {
    let graph = model.graph.as_ref().ok_or_else(|| {
        crate::error::DsperseError::Slicer("create_dim_split_template: model has no graph".into())
    })?;

    match info.split_kind {
        crate::schema::tiling::DimSplitKind::MatMulOutputDim => {
            create_matmul_dim_template(model, graph, info, output_dir)
        }
        crate::schema::tiling::DimSplitKind::HeadDim
        | crate::schema::tiling::DimSplitKind::BatchDim => {
            create_generic_dim_template(model, graph, info, output_dir, traced_shapes)
        }
    }
}

fn create_matmul_dim_template(
    model: &ModelProto,
    graph: &GraphProto,
    info: &crate::schema::tiling::DimSplitInfo,
    output_dir: &Path,
) -> Result<std::path::PathBuf> {
    let weight_name = info.weight_name.as_ref().ok_or_else(|| {
        crate::error::DsperseError::Slicer(format!(
            "create_matmul_dim_template: slice {} DimSplitInfo missing weight_name",
            info.slice_idx
        ))
    })?;

    // Match the exact split node by weight, activation input, and output
    // name. A graph may reuse the same weight initializer in multiple
    // MatMul/Gemm ops (tied weights, weight sharing across heads); without
    // checking IO we could bind the wrong op and emit a template that
    // doesn't match the slice the runner will execute.
    let matmul_node = graph
        .node
        .iter()
        .find(|n| {
            matches!(n.op_type.as_str(), "MatMul" | "Gemm")
                && n.input.iter().any(|i| i == weight_name)
                && n.input.iter().any(|i| i == &info.input_name)
                && n.output.iter().any(|o| o == &info.output_name)
        })
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "create_matmul_dim_template: slice {} no MatMul/Gemm matches weight={weight_name:?} input={:?} output={:?}",
                info.slice_idx, info.input_name, info.output_name
            ))
        })?;

    if matmul_node.op_type == "Gemm" && matmul_node.input.get(2).is_some_and(|s| !s.is_empty()) {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_matmul_dim_template: slice {} Gemm with bias not supported for dim-split",
            info.slice_idx
        )));
    }

    let weight_tensor = graph
        .initializer
        .iter()
        .find(|i| i.name == *weight_name)
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "create_matmul_dim_template: weight {weight_name:?} not in initializers"
            ))
        })?;
    if weight_tensor.dims.len() != 2 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_matmul_dim_template: expected 2D weights, got {:?}",
            weight_tensor.dims
        )));
    }

    if matmul_node.op_type == "Gemm"
        && onnx_proto::get_attribute_int(matmul_node, "transA").unwrap_or(0) == 1
    {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_matmul_dim_template: slice {} Gemm with transA=1 is not supported for dim-split",
            info.slice_idx
        )));
    }

    let trans_b = matmul_node.op_type == "Gemm"
        && onnx_proto::get_attribute_int(matmul_node, "transB").unwrap_or(0) == 1;

    let (rows, cols) = (
        weight_tensor.dims[0] as usize,
        weight_tensor.dims[1] as usize,
    );
    let (k_dim, n_dim) = if trans_b { (cols, rows) } else { (rows, cols) };
    let k_chunk_size = k_dim.div_ceil(info.k_chunks.max(1));

    let tmpl_input_name = "dim_tmpl_in".to_string();
    let tmpl_output_name = "dim_tmpl_out".to_string();
    let tmpl_weight_name = "W".to_string();

    let tmpl_input_shape: Vec<i64> = vec![1, k_chunk_size as i64];
    let output_shape: Vec<i64> = vec![1, n_dim as i64];

    let x =
        onnx_proto::make_tensor_value_info(&tmpl_input_name, TensorProto::FLOAT, &tmpl_input_shape);
    let y =
        onnx_proto::make_tensor_value_info(&tmpl_output_name, TensorProto::FLOAT, &output_shape);
    let tmpl_weight_dims: Vec<i64> = if trans_b {
        vec![n_dim as i64, k_chunk_size as i64]
    } else {
        vec![k_chunk_size as i64, n_dim as i64]
    };
    let w = onnx_proto::make_tensor(
        &tmpl_weight_name,
        TensorProto::FLOAT,
        &tmpl_weight_dims,
        vec![0.0f32; k_chunk_size * n_dim],
    );

    let mut attrs = Vec::new();
    let node_inputs = vec![tmpl_input_name, tmpl_weight_name];
    let initializers = vec![w];

    if matmul_node.op_type == "Gemm" {
        if let Some(alpha) = onnx_proto::get_attribute_float(matmul_node, "alpha") {
            attrs.push(onnx_proto::make_attribute_float("alpha", alpha));
        }
        if let Some(beta) = onnx_proto::get_attribute_float(matmul_node, "beta") {
            attrs.push(onnx_proto::make_attribute_float("beta", beta));
        }
        // transA is rejected above; the template always uses A non-transposed.
        if trans_b {
            attrs.push(onnx_proto::make_attribute_int("transB", 1));
        }
        // Biased Gemm is rejected above, so no C initializer is ever folded
        // into the template.
    }

    let node = onnx_proto::make_node(
        &matmul_node.op_type,
        node_inputs,
        vec![tmpl_output_name],
        attrs,
    );

    let graph_proto = onnx_proto::make_graph(
        &format!("dim_template_{}", info.slice_idx),
        vec![node],
        vec![x],
        vec![y],
        initializers,
    );
    let tmpl_model = onnx_proto::make_model(graph_proto, model_opset(model));

    let tmpl_path = output_dir.join("dim_template.onnx");
    onnx_proto::save_model(&tmpl_model, &tmpl_path)?;
    Ok(tmpl_path)
}

fn check_axis_separable(graph: &GraphProto, split_dim: usize, slice_idx: usize) -> Result<()> {
    let resolve_axis = |axis: i64| -> usize {
        let ndim = graph
            .input
            .first()
            .and_then(onnx_proto::shape_from_value_info)
            .map(|s| s.len() as i64)
            .unwrap_or(4);
        if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        }
    };

    for node in &graph.node {
        match node.op_type.as_str() {
            "Flatten" => {
                let axis = resolve_axis(onnx_proto::get_attribute_int(node, "axis").unwrap_or(1));
                if split_dim < axis {
                    return Err(crate::error::DsperseError::Slicer(format!(
                        "create_generic_dim_template: slice {slice_idx} Flatten axis \
                         {axis} > split_dim {split_dim}; split dimension falls in the merged leading group"
                    )));
                }
            }
            "Softmax" | "LogSoftmax" => {
                let resolved =
                    resolve_axis(onnx_proto::get_attribute_int(node, "axis").unwrap_or(-1));
                if resolved == split_dim {
                    return Err(crate::error::DsperseError::Slicer(format!(
                        "create_generic_dim_template: slice {slice_idx} {} axis {resolved} \
                         equals split_dim {split_dim}; normalization spans the split dimension",
                        node.op_type
                    )));
                }
            }
            "LayerNormalization" => {
                let resolved =
                    resolve_axis(onnx_proto::get_attribute_int(node, "axis").unwrap_or(-1));
                if resolved <= split_dim {
                    return Err(crate::error::DsperseError::Slicer(format!(
                        "create_generic_dim_template: slice {slice_idx} LayerNormalization axis \
                         {resolved} <= split_dim {split_dim}; normalization spans the split dimension",
                    )));
                }
            }
            "BatchNormalization" if split_dim == 0 => {
                return Err(crate::error::DsperseError::Slicer(format!(
                    "create_generic_dim_template: slice {slice_idx} BatchNormalization requires \
                     full batch statistics; cannot split at dim 0"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

fn create_generic_dim_template(
    model: &ModelProto,
    graph: &GraphProto,
    info: &crate::schema::tiling::DimSplitInfo,
    output_dir: &Path,
    traced_shapes: Option<&HashMap<String, Vec<i64>>>,
) -> Result<std::path::PathBuf> {
    if info.elements_per_group == 0 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_generic_dim_template: slice {} elements_per_group is 0",
            info.slice_idx
        )));
    }

    check_axis_separable(graph, info.split_dim, info.slice_idx)?;

    // Rewrite the template so the split axis carries elements_per_group
    // instead of the full dim_size.  The runner only ever feeds a single
    // group's worth of activations to the compiled circuit, so the
    // *compile* cost should match the per-group cost rather than the
    // whole-slice cost.  Catalog reuse is preserved at per-group
    // granularity: any two slices that share (split_dim, epg, surrounding
    // op shapes) hash identically.
    //
    // The strategy is: rewrite only the boundary shapes (graph inputs +
    // shape-input initializers consumed by Reshape / Expand / Tile /
    // ConstantOfShape) and a fresh shape inference pass derives every
    // intermediate value_info from those.  Per-feature initializers
    // (gamma, beta, weights) are never touched, and there are no ad-hoc
    // cases for individual op patterns -- the rule is "rewrite the
    // boundary, let inference do the rest".
    let mut tmpl_model = model.clone();
    let tmpl_graph = tmpl_model.graph.as_mut().ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_generic_dim_template: cloned model has no graph".into(),
        )
    })?;
    let dim_size = info.dim_size as i64;
    let epg = info.elements_per_group as i64;
    let split_dim = info.split_dim;

    // 1. Decide which graph inputs must be rewritten at split_dim.
    //
    //    The runner always slices every cached tensor whose shape has
    //    dim_size at split_dim, so the *compile-time* template needs
    //    every such input declared with epg, otherwise jstprove's
    //    type checker rejects the op (e.g. Mul broadcast 150 vs 300,
    //    or MatMul A.K vs B.K mismatch).  But for ops where two
    //    inputs reference dim_size at the *same* split_dim with
    //    different semantic meanings (the canonical case is the
    //    second attention MatMul: attn[B,H,M,N] @ V[B,H,N,D] with
    //    M == N at split_dim=2) blanket rewriting both inputs
    //    produces a real mismatch.
    //
    //    Heuristic:
    //      * Elementwise / broadcast ops (Add, Sub, Mul, Div, Pow, Min,
    //        Max, Where, Equal, Greater, Less): rewrite every input
    //        whose shape has dim_size at split_dim.  All inputs share
    //        a logical broadcast axis, so all must shrink together.
    //      * MatMul / Gemm: rewrite only `info.input_name`.  The other
    //        operand's split_dim is a contraction axis; touching it
    //        produces an inner-dim mismatch.
    //      * Everything else (the single-op slices we get after
    //        isolate_expensive_ops): rewrite only `info.input_name`,
    //        which is the safe default for ops with one primary
    //        activation and a handful of scalar / per-feature
    //        initializer inputs.
    let elementwise_ops: HashSet<&str> = [
        "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Where", "Equal", "Greater", "Less",
    ]
    .into_iter()
    .collect();
    let rewrite_all_matching = tmpl_graph
        .node
        .iter()
        .all(|n| elementwise_ops.contains(n.op_type.as_str()));

    let rewrite_input_at_split_dim = |vi: &mut super::onnx_proto::ValueInfoProto| {
        if let Some(t) = vi.r#type.as_mut()
            && let Some(super::onnx_proto::onnx::type_proto::Value::TensorType(tt)) =
                t.value.as_mut()
            && let Some(shape) = tt.shape.as_mut()
            && let Some(d) = shape.dim.get_mut(split_dim)
            && let Some(super::onnx_proto::onnx::tensor_shape_proto::dimension::Value::DimValue(v)) =
                d.value.as_mut()
            && *v == dim_size
        {
            *v = epg;
        }
    };

    if rewrite_all_matching {
        for vi in tmpl_graph
            .input
            .iter_mut()
            .chain(tmpl_graph.output.iter_mut())
        {
            rewrite_input_at_split_dim(vi);
        }
    } else {
        for vi in tmpl_graph
            .input
            .iter_mut()
            .filter(|vi| vi.name == info.input_name)
            .chain(
                tmpl_graph
                    .output
                    .iter_mut()
                    .filter(|vi| vi.name == info.output_name),
            )
        {
            rewrite_input_at_split_dim(vi);
        }
    }

    // 2. Rewrite shape-input initializers (Reshape / Expand / Tile /
    //    ConstantOfShape).  These are explicit shape descriptors; if
    //    the input shape changes their dim_size entry must change too.
    let shape_input_initializers: HashSet<String> = tmpl_graph
        .node
        .iter()
        .filter_map(|n| match n.op_type.as_str() {
            "Reshape" | "Expand" | "Tile" => n.input.get(1).cloned(),
            "ConstantOfShape" => n.input.first().cloned(),
            _ => None,
        })
        .filter(|name| !name.is_empty())
        .collect();
    for init in &mut tmpl_graph.initializer {
        if init.data_type == TensorProto::INT64 && shape_input_initializers.contains(&init.name) {
            for v in &mut init.int64_data {
                if *v == dim_size {
                    *v = epg;
                }
            }
        }
    }

    // 3. Drop every intermediate value_info; it will be re-derived.
    tmpl_graph.value_info.clear();

    let _ = traced_shapes; // intentionally unused: we re-trace after rewriting.

    let tmpl_path = output_dir.join("dim_template.onnx");
    onnx_proto::save_model(&tmpl_model, &tmpl_path)?;

    // 4. Re-run shape inference on the rewritten template and inject
    //    the derived shapes back as value_info.  This replaces the old
    //    ad-hoc per-op rewrites (which had to special-case every shape
    //    op).  If re-trace fails the template is uncompilable -- the
    //    circuit compiler downstream will see no value_info for the
    //    intermediate tensors and produce hard-to-diagnose shape
    //    errors at compile time.  Refuse to emit the template instead.
    let trace = super::trace::fold_and_trace_via_tract(&tmpl_path, &tmpl_model).map_err(
        |e| {
            crate::error::DsperseError::Slicer(format!(
                "create_generic_dim_template: slice {} re-trace failed (template input shape {:?}, split_dim {}): {e}",
                info.slice_idx, info.input_name, split_dim
            ))
        },
    )?;
    {
        let mut model_after = onnx_proto::load_model(&tmpl_path)?;
        if let Some(graph_after) = model_after.graph.as_mut() {
            let existing: HashSet<String> = graph_after
                .input
                .iter()
                .chain(graph_after.output.iter())
                .chain(graph_after.value_info.iter())
                .map(|vi| vi.name.clone())
                .collect();
            let init_names: HashSet<&str> = graph_after
                .initializer
                .iter()
                .map(|i| i.name.as_str())
                .collect();
            for node in &graph_after.node {
                for out_name in &node.output {
                    if out_name.is_empty()
                        || existing.contains(out_name)
                        || init_names.contains(out_name.as_str())
                    {
                        continue;
                    }
                    if let Some(shape) = trace.shapes.get(out_name) {
                        let elem_type = trace
                            .types
                            .get(out_name)
                            .copied()
                            .unwrap_or(TensorProto::FLOAT);
                        graph_after
                            .value_info
                            .push(onnx_proto::make_tensor_value_info(
                                out_name, elem_type, shape,
                            ));
                    }
                }
            }
            // Promote output_name to graph output if it now exists in
            // value_info but not in graph.output.
            if !graph_after
                .output
                .iter()
                .any(|o| o.name == info.output_name)
                && let Some(vi) = graph_after
                    .value_info
                    .iter()
                    .find(|v| v.name == info.output_name)
                    .cloned()
            {
                graph_after.output.push(vi);
            }
        }
        onnx_proto::save_model(&model_after, &tmpl_path)?;
    }

    Ok(tmpl_path)
}

pub fn create_elementwise_tile_slice(
    model: &ModelProto,
    segment_size: i64,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<TileSliceResult> {
    if segment_size <= 0 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_elementwise_tile_slice: segment_size must be > 0, got {segment_size}"
        )));
    }
    let graph = model.graph.as_ref().ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_elementwise_tile_slice: model.graph is None".to_string(),
        )
    })?;
    if graph.input.is_empty() {
        return Err(crate::error::DsperseError::Slicer(
            "create_elementwise_tile_slice: no graph inputs".to_string(),
        ));
    }
    let out = graph.output.first().ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_elementwise_tile_slice: no graph outputs".to_string(),
        )
    })?;
    let orig_output_name = &out.name;

    let tile_shape: Vec<i64> = vec![segment_size];

    let init_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|i| i.name.as_str()).collect();

    let mut orig_to_tile: Vec<(String, String)> = Vec::with_capacity(graph.input.len());
    let mut tile_inputs = Vec::with_capacity(graph.input.len());
    let mut tile_idx = 0usize;
    for inp in &graph.input {
        let inp_shape = onnx_proto::shape_from_value_info(inp);
        let is_broadcast = init_names.contains(inp.name.as_str())
            || inp_shape
                .as_ref()
                .is_some_and(|s| s.iter().product::<i64>() < segment_size);
        if is_broadcast {
            tile_inputs.push(inp.clone());
        } else {
            let tile_name = format!("tile_in_{tile_idx}");
            tile_idx += 1;
            tile_inputs.push(onnx_proto::make_tensor_value_info(
                &tile_name,
                onnx_proto::elem_type_from_value_info(inp).unwrap_or(TensorProto::FLOAT),
                &tile_shape,
            ));
            orig_to_tile.push((inp.name.clone(), tile_name));
        }
    }
    if tile_idx == 1
        && let Some((_, tile_name)) = orig_to_tile.first_mut()
    {
        let old = tile_name.clone();
        *tile_name = "tile_in".to_string();
        for ti in &mut tile_inputs {
            if ti.name == old {
                ti.name = "tile_in".to_string();
            }
        }
    }

    let y = onnx_proto::make_tensor_value_info("tile_out", TensorProto::FLOAT, &tile_shape);

    let initializers: Vec<_> = graph.initializer.to_vec();

    let input_remap: std::collections::HashMap<&str, &str> = orig_to_tile
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let mut nodes = Vec::new();
    for orig_node in &graph.node {
        let new_inputs: Vec<String> = orig_node
            .input
            .iter()
            .map(|name| {
                input_remap
                    .get(name.as_str())
                    .map(|s| (*s).to_string())
                    .unwrap_or_else(|| name.clone())
            })
            .collect();
        let produces_output = orig_node.output.contains(orig_output_name);
        let new_outputs = if produces_output {
            orig_node
                .output
                .iter()
                .map(|o| {
                    if o == orig_output_name {
                        "tile_out".to_string()
                    } else {
                        o.clone()
                    }
                })
                .collect()
        } else {
            orig_node.output.clone()
        };

        nodes.push(NodeProto {
            op_type: orig_node.op_type.clone(),
            input: new_inputs,
            output: new_outputs,
            attribute: orig_node.attribute.clone(),
            name: String::new(),
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        });
    }

    let tile_graph = onnx_proto::make_graph(
        &format!("tile_{slice_idx}"),
        nodes,
        tile_inputs,
        vec![y],
        initializers,
    );
    let tile_model = onnx_proto::make_model(tile_graph, model_opset(model));

    let tiles_dir = output_dir.join("tiles");
    std::fs::create_dir_all(&tiles_dir)
        .map_err(|e| crate::error::DsperseError::io(e, &tiles_dir))?;
    let onnx_path = tiles_dir.join("tile.onnx");
    onnx_proto::save_model(&tile_model, &onnx_path)?;

    Ok(TileSliceResult {
        path: format!("slice_{slice_idx}/payload/tiles/tile.onnx"),
        conv_out: [segment_size, 1],
    })
}

#[derive(Debug)]
pub struct TileSliceResult {
    pub path: String,
    pub conv_out: [i64; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halo_symmetric_pads() {
        assert_eq!(compute_halo_size([1, 1, 1, 1]), Some([1, 1, 1, 1]));
    }

    #[test]
    fn halo_asymmetric_pads() {
        assert_eq!(compute_halo_size([6, 6, 7, 7]), Some([6, 6, 7, 7]));
    }

    #[test]
    fn halo_zero_pads() {
        assert_eq!(compute_halo_size([0, 0, 0, 0]), Some([0, 0, 0, 0]));
    }

    #[test]
    fn halo_negative_pads_rejected() {
        assert_eq!(compute_halo_size([-1, 0, 0, 0]), None);
    }

    #[test]
    fn halo_mixed_pads() {
        assert_eq!(compute_halo_size([1, 2, 1, 2]), Some([1, 2, 1, 2]));
    }

    #[test]
    fn min_tile_3x3_no_dilation() {
        assert_eq!(compute_min_spatial_tile([3, 3], [1, 1]), Some(4));
    }

    #[test]
    fn min_tile_5x5_no_dilation() {
        assert_eq!(compute_min_spatial_tile([5, 5], [1, 1]), Some(6));
    }

    #[test]
    fn min_tile_3x3_dilation_2() {
        let eff = (3 - 1) * 2 + 1;
        assert_eq!(compute_min_spatial_tile([3, 3], [2, 2]), Some(eff + 1));
    }

    #[test]
    fn min_tile_1x1() {
        assert_eq!(compute_min_spatial_tile([1, 1], [1, 1]), Some(2));
    }

    #[test]
    fn optimal_tile_exact_divisor() {
        assert_eq!(find_optimal_tile_size(64, 32, 4, 1), Some(32));
    }

    #[test]
    fn optimal_tile_no_exact_divisor_falls_back() {
        assert_eq!(find_optimal_tile_size(64, 30, 4, 1), Some(16));
    }

    #[test]
    fn optimal_tile_target_equals_spatial() {
        assert_eq!(find_optimal_tile_size(32, 32, 4, 1), None);
    }

    #[test]
    fn optimal_tile_min_exceeds_target() {
        assert_eq!(find_optimal_tile_size(64, 3, 4, 1), None);
    }

    #[test]
    fn optimal_tile_stride_constraint() {
        assert_eq!(find_optimal_tile_size(64, 32, 4, 2), Some(32));
        assert_eq!(find_optimal_tile_size(12, 8, 2, 4), Some(4));
    }

    #[test]
    fn optimal_tile_no_valid_stride_divisor() {
        assert_eq!(find_optimal_tile_size(15, 10, 2, 4), None);
    }

    #[test]
    fn checked_dim_product_normal() {
        assert_eq!(checked_dim_product(&[2, 3, 4]).unwrap(), 24);
    }

    #[test]
    fn checked_dim_product_empty() {
        assert_eq!(checked_dim_product(&[]).unwrap(), 1);
    }

    #[test]
    fn checked_dim_product_overflow() {
        assert!(checked_dim_product(&[usize::MAX, 2]).is_err());
    }

    #[test]
    fn checked_dim_product_single() {
        assert_eq!(checked_dim_product(&[42]).unwrap(), 42);
    }

    #[test]
    fn slice_weights_basic() {
        let weights = WeightInfo {
            data: (0..24).map(|i| i as f32).collect(),
            dims: vec![2, 3, 2, 2],
        };
        let sliced = slice_weights(&weights, 0, 2).unwrap();
        assert_eq!(sliced.dims, vec![2, 2, 2, 2]);
        assert_eq!(sliced.data.len(), 16);
        assert_eq!(sliced.data[0], 0.0);
        assert_eq!(sliced.data[1], 1.0);
        assert_eq!(sliced.data[2], 2.0);
        assert_eq!(sliced.data[3], 3.0);
    }

    #[test]
    fn slice_weights_single_channel() {
        let weights = WeightInfo {
            data: (0..24).map(|i| i as f32).collect(),
            dims: vec![2, 3, 2, 2],
        };
        let sliced = slice_weights(&weights, 1, 2).unwrap();
        assert_eq!(sliced.dims, vec![2, 1, 2, 2]);
        assert_eq!(sliced.data.len(), 8);
    }

    #[test]
    fn slice_weights_start_ge_end() {
        let weights = WeightInfo {
            data: vec![1.0; 16],
            dims: vec![1, 4, 2, 2],
        };
        assert!(slice_weights(&weights, 3, 2).is_err());
    }

    #[test]
    fn slice_weights_end_exceeds_c_in() {
        let weights = WeightInfo {
            data: vec![1.0; 16],
            dims: vec![1, 4, 2, 2],
        };
        assert!(slice_weights(&weights, 0, 5).is_err());
    }

    #[test]
    fn slice_weights_insufficient_dims() {
        let weights = WeightInfo {
            data: vec![1.0; 6],
            dims: vec![2, 3],
        };
        assert!(slice_weights(&weights, 0, 1).is_err());
    }

    #[test]
    fn slice_weights_data_length_mismatch() {
        let weights = WeightInfo {
            data: vec![1.0; 10],
            dims: vec![2, 3, 2, 2],
        };
        assert!(slice_weights(&weights, 0, 2).is_err());
    }

    #[test]
    fn elementwise_ops_recognized() {
        assert!(is_elementwise("Relu"));
        assert!(is_elementwise("Sigmoid"));
        assert!(is_elementwise("Add"));
        assert!(is_elementwise("Mul"));
    }

    #[test]
    fn non_elementwise_ops_rejected() {
        assert!(!is_elementwise("Conv"));
        assert!(!is_elementwise("MaxPool"));
        assert!(!is_elementwise("Gemm"));
        assert!(!is_elementwise("BatchNormalization"));
    }

    #[test]
    fn spatial_tile_config_already_fits() {
        let (tile, reason) = calculate_spatial_tile_config(3, 4, 4, 64, 4, 1);
        assert!(tile.is_none());
        assert_eq!(reason, Some("already_fits"));
    }

    #[test]
    fn spatial_tile_config_min_tile_too_large() {
        let (tile, reason) = calculate_spatial_tile_config(64, 8, 8, 100, 8, 1);
        assert!(tile.is_none());
        assert_eq!(reason, Some("min_tile_too_large"));
    }

    #[test]
    fn spatial_tile_config_finds_tile() {
        let (tile, reason) = calculate_spatial_tile_config(3, 64, 64, 3 * 32 * 32, 4, 1);
        assert!(tile.is_some());
        assert!(reason.is_none());
        let t = tile.unwrap();
        assert!(64 % t == 0);
        assert!(t >= 4);
    }

    #[test]
    fn channel_split_config_basic() {
        let result = calculate_channel_split_config(64, 32, 4, 4, 32);
        assert!(result.is_some());
        let (num_groups, cpg) = result.unwrap();
        assert!(num_groups > 1);
        assert!(cpg > 0);
        assert!(cpg * (num_groups - 1) < 64);
    }

    #[test]
    fn channel_split_config_zero_dims() {
        assert!(calculate_channel_split_config(64, 32, 0, 4, 32).is_none());
        assert!(calculate_channel_split_config(64, 32, 4, 0, 32).is_none());
    }

    #[test]
    fn channel_split_config_fits_without_splitting() {
        assert!(calculate_channel_split_config(4, 32, 2, 2, 100).is_none());
    }

    #[test]
    fn detect_tiling_none_without_tile_size() {
        let model = onnx_proto::make_model(
            onnx_proto::make_graph("test", vec![], vec![], vec![], vec![]),
            13,
        );
        assert!(detect_tiling_needs(&model, None).is_none());
    }

    #[test]
    fn detect_tiling_none_empty_graph() {
        let model = onnx_proto::make_model(
            onnx_proto::make_graph("test", vec![], vec![], vec![], vec![]),
            13,
        );
        assert!(detect_tiling_needs(&model, Some(1024)).is_none());
    }

    #[test]
    fn effective_kernel_overflow() {
        assert_eq!(effective_kernel([i64::MAX, 1], [2, 1]), None);
        assert_eq!(effective_kernel([1, i64::MAX], [1, 2]), None);
    }

    #[test]
    fn effective_kernel_sub_underflow() {
        assert_eq!(effective_kernel([i64::MIN, 3], [1, 1]), None);
    }

    #[test]
    fn effective_kernel_valid() {
        assert_eq!(effective_kernel([3, 3], [1, 1]), Some([3, 3]));
        assert_eq!(effective_kernel([3, 3], [2, 2]), Some([5, 5]));
        assert_eq!(effective_kernel([1, 1], [1, 1]), Some([1, 1]));
    }

    #[test]
    fn conv_output_hw_zero_stride() {
        assert_eq!(
            conv_output_hw(8, 8, [0, 0, 0, 0], [3, 3], [1, 1], [0, 1]),
            None
        );
        assert_eq!(
            conv_output_hw(8, 8, [0, 0, 0, 0], [3, 3], [1, 1], [1, 0]),
            None
        );
    }

    #[test]
    fn conv_output_hw_kernel_exceeds_input() {
        assert_eq!(
            conv_output_hw(2, 2, [0, 0, 0, 0], [5, 5], [1, 1], [1, 1]),
            None
        );
    }

    #[test]
    fn conv_output_hw_overflow_pads() {
        assert_eq!(
            conv_output_hw(i64::MAX, 8, [1, 0, 0, 0], [3, 3], [1, 1], [1, 1]),
            None
        );
    }

    #[test]
    fn conv_output_hw_valid() {
        assert_eq!(
            conv_output_hw(8, 8, [1, 1, 1, 1], [3, 3], [1, 1], [1, 1]),
            Some((8, 8))
        );
        assert_eq!(
            conv_output_hw(8, 8, [0, 0, 0, 0], [3, 3], [1, 1], [2, 2]),
            Some((3, 3))
        );
    }

    #[test]
    fn compute_halo_size_negative_rejected() {
        assert_eq!(compute_halo_size([0, 0, -1, 0]), None);
    }

    #[test]
    fn compute_min_spatial_tile_overflow() {
        assert_eq!(compute_min_spatial_tile([i64::MAX, 1], [2, 1]), None);
    }

    #[test]
    fn slice_weights_full_range_is_identity() {
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let weights = WeightInfo {
            data: data.clone(),
            dims: vec![2, 3, 2, 4],
        };
        let sliced = slice_weights(&weights, 0, 3).unwrap();
        assert_eq!(sliced.dims, vec![2, 3, 2, 4]);
        assert_eq!(sliced.data, data);
    }

    #[test]
    fn detect_dim_split_gemm_trans_b() {
        use super::onnx_proto::{NodeProto, make_attribute_int};

        // Unbiased Gemm with transB=1. Biased Gemm is rejected upstream by
        // create_matmul_dim_template, so the detector now skips it as well.
        let node = NodeProto {
            op_type: "Gemm".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            attribute: vec![make_attribute_int("transB", 1)],
            ..Default::default()
        };

        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![4, 145, 384]);
        shapes.insert("weight".to_string(), vec![1536, 384]);
        shapes.insert("output".to_string(), vec![4, 145, 1536]);

        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());

        let detection = detect_dim_split(&[node], &shapes, &init_names);
        assert!(detection.is_some());
        let d = detection.unwrap();
        assert_eq!(d.split_dim, 0);
        assert_eq!(d.dim_size, 580);
        assert_eq!(d.num_groups, 580);
        assert_eq!(d.elements_per_group, 1);
        assert_eq!(d.k_dim, 384);
        assert_eq!(d.n_dim, 1536);
        assert!(matches!(d.split_kind, DimSplitKind::MatMulOutputDim));
    }

    #[test]
    fn detect_dim_split_matmul_no_trans() {
        let node = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };

        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![4, 145, 384]);
        shapes.insert("weight".to_string(), vec![384, 1536]);
        shapes.insert("output".to_string(), vec![4, 145, 1536]);

        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());

        let detection = detect_dim_split(&[node], &shapes, &init_names);
        assert!(detection.is_some());
        let d = detection.unwrap();
        assert_eq!(d.split_dim, 0);
        assert_eq!(d.dim_size, 580);
        assert_eq!(d.num_groups, 580);
        assert_eq!(d.elements_per_group, 1);
        assert_eq!(d.k_dim, 384);
        assert_eq!(d.n_dim, 1536);
        assert!(matches!(d.split_kind, DimSplitKind::MatMulOutputDim));
    }

    #[test]
    fn detect_dim_split_k_chunks_saturate_budget() {
        // k_dim=10, n_dim=300_000: row_cost=6M. Naive k_chunks=ceil(6M/2M)=3
        // yields chunk_size=ceil(10/3)=4 -> per-chunk=4*300_000*2=2.4M > 2M
        // (MAX_ESTIMATED_CONSTRAINTS). Loop bumps k_chunks to 4 giving
        // chunk_size=3 -> per-chunk=1.8M which fits.
        let node = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![4, 10]);
        shapes.insert("weight".to_string(), vec![10, 300_000]);
        shapes.insert("output".to_string(), vec![4, 300_000]);
        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());

        let d = detect_dim_split(&[node], &shapes, &init_names).unwrap();
        assert_eq!(d.k_dim, 10);
        assert_eq!(d.n_dim, 300_000);
        let chunk_size = d.k_dim.div_ceil(d.k_chunks);
        assert!(
            chunk_size * d.n_dim * 2 <= MAX_ESTIMATED_CONSTRAINTS as usize,
            "per-chunk cost {} exceeds MAX {}",
            chunk_size * d.n_dim * 2,
            MAX_ESTIMATED_CONSTRAINTS
        );
    }

    #[test]
    fn detect_dim_split_single_row_with_k_chunking() {
        // total_rows=1 but k*n*2 > MAX: still detect, K-chunk it.
        let node = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![1, 2048]);
        shapes.insert("weight".to_string(), vec![2048, 2048]);
        shapes.insert("output".to_string(), vec![1, 2048]);
        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());

        let d = detect_dim_split(&[node], &shapes, &init_names).unwrap();
        assert_eq!(d.dim_size, 1);
        assert_eq!(d.num_groups, 1);
        assert!(d.k_chunks > 1, "expected K-chunking for single row");
        let chunk_size = d.k_dim.div_ceil(d.k_chunks);
        assert!(chunk_size * d.n_dim * 2 <= MAX_ESTIMATED_CONSTRAINTS as usize);
    }

    #[test]
    fn detect_dim_split_skips_single_row_single_chunk() {
        // total_rows=1 and k*n*2 <= MAX: nothing to split via MatMul path.
        // The slice is still over budget (forced via a second MatMul), but
        // dim-split should decline and let the caller fall through.
        let node1 = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "w1".to_string()],
            output: vec!["mid".to_string()],
            ..Default::default()
        };
        let node2 = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["mid".to_string(), "w2".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![1, 64]);
        shapes.insert("w1".to_string(), vec![64, 64]);
        shapes.insert("mid".to_string(), vec![1, 64]);
        shapes.insert("w2".to_string(), vec![64, 64]);
        shapes.insert("output".to_string(), vec![1, 64]);
        let mut init_names = HashSet::new();
        init_names.insert("w1".to_string());
        init_names.insert("w2".to_string());
        // Tiny per-op cost; slice estimate stays under MAX so detect_dim_split
        // returns None at the outer gate, which is what we want for a
        // single-row single-chunk MatMul.
        assert!(detect_dim_split(&[node1, node2], &shapes, &init_names).is_none());
    }

    #[test]
    fn detect_dim_split_declines_infeasible_n() {
        // n_dim * 2 > MAX means even k_chunks == k_dim (chunk_size = 1)
        // cannot fit inside the per-chunk budget, so the MatMul branch must
        // decline. Use batch=1 so the BatchDim fallback path is not taken.
        let node = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        // n_dim = 1_500_000 -> n*2 = 3_000_000 > MAX (2_000_000)
        shapes.insert("input".to_string(), vec![1, 4]);
        shapes.insert("weight".to_string(), vec![4, 1_500_000]);
        shapes.insert("output".to_string(), vec![1, 1_500_000]);
        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());

        let got = detect_dim_split(&[node], &shapes, &init_names);
        assert!(
            got.as_ref()
                .is_none_or(|d| !matches!(d.split_kind, DimSplitKind::MatMulOutputDim)),
            "expected MatMul dim-split to decline, got {got:?}"
        );
    }

    #[test]
    fn detect_dim_split_skips_non_terminal_matmul() {
        // MatMul output is consumed by a later Add inside the same slice.
        // The dim-split runner only writes MatMul output to the cache, so
        // the Add would never run; detection must decline this MatMul and
        // either pick a later terminal MatMul or fall through.
        let matmul = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["mid".to_string()],
            ..Default::default()
        };
        let add = NodeProto {
            op_type: "Add".to_string(),
            input: vec!["mid".to_string(), "bias".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![1, 145, 384]);
        shapes.insert("weight".to_string(), vec![384, 1536]);
        shapes.insert("bias".to_string(), vec![1536]);
        shapes.insert("mid".to_string(), vec![1, 145, 1536]);
        shapes.insert("output".to_string(), vec![1, 145, 1536]);
        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());
        init_names.insert("bias".to_string());

        let got = detect_dim_split(&[matmul, add], &shapes, &init_names);
        assert!(
            got.as_ref()
                .is_none_or(|d| !matches!(d.split_kind, DimSplitKind::MatMulOutputDim)),
            "expected non-terminal MatMul to be declined, got {got:?}"
        );
    }

    #[test]
    fn detect_dim_split_picks_terminal_matmul_after_consumed_one() {
        // First MatMul feeds a second MatMul; only the second is terminal,
        // so detection must skip the first and select the second when both
        // are otherwise eligible.
        let m1 = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "w1".to_string()],
            output: vec!["mid".to_string()],
            ..Default::default()
        };
        let m2 = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["mid".to_string(), "w2".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        shapes.insert("input".to_string(), vec![4, 145, 384]);
        shapes.insert("w1".to_string(), vec![384, 1536]);
        shapes.insert("mid".to_string(), vec![4, 145, 1536]);
        shapes.insert("w2".to_string(), vec![1536, 384]);
        shapes.insert("output".to_string(), vec![4, 145, 384]);
        let mut init_names = HashSet::new();
        init_names.insert("w1".to_string());
        init_names.insert("w2".to_string());

        let d = detect_dim_split(&[m1, m2], &shapes, &init_names).unwrap();
        assert_eq!(d.weight_name.as_deref(), Some("w2"));
        assert_eq!(d.output_name, "output");
        assert_eq!(d.k_dim, 1536);
        assert_eq!(d.n_dim, 384);
    }

    #[test]
    fn detect_dim_split_skips_gemm_trans_a() {
        use super::onnx_proto::make_attribute_int;
        let node = NodeProto {
            op_type: "Gemm".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            attribute: vec![make_attribute_int("transA", 1)],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        // Use batch=1 so the BatchDim fallback path does not mask the
        // MatMul-branch decline we want to assert.
        shapes.insert("input".to_string(), vec![1, 384, 145]);
        shapes.insert("weight".to_string(), vec![384, 1536]);
        shapes.insert("output".to_string(), vec![1, 145, 1536]);
        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());

        let got = detect_dim_split(&[node], &shapes, &init_names);
        assert!(
            got.as_ref()
                .is_none_or(|d| !matches!(d.split_kind, DimSplitKind::MatMulOutputDim)),
            "expected Gemm transA=1 MatMul decline, got {got:?}"
        );
    }

    #[test]
    fn detect_dim_split_skips_gemm_with_bias() {
        use super::onnx_proto::make_attribute_int;
        let node = NodeProto {
            op_type: "Gemm".to_string(),
            input: vec![
                "input".to_string(),
                "weight".to_string(),
                "bias".to_string(),
            ],
            output: vec!["output".to_string()],
            attribute: vec![make_attribute_int("transB", 1)],
            ..Default::default()
        };
        let mut shapes = HashMap::new();
        // Use batch=1 so the BatchDim fallback path does not mask the
        // MatMul-branch decline we want to assert.
        shapes.insert("input".to_string(), vec![1, 145, 384]);
        shapes.insert("weight".to_string(), vec![1536, 384]);
        shapes.insert("bias".to_string(), vec![1536]);
        shapes.insert("output".to_string(), vec![1, 145, 1536]);
        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());
        init_names.insert("bias".to_string());

        // Detector should decline the MatMul branch since the template
        // builder cannot handle biased Gemm, forcing fall-through.
        let got = detect_dim_split(&[node], &shapes, &init_names);
        assert!(
            got.as_ref()
                .is_none_or(|d| !matches!(d.split_kind, DimSplitKind::MatMulOutputDim)),
            "expected Gemm-with-bias MatMul decline, got {got:?}"
        );
    }

    #[test]
    fn create_matmul_dim_template_uses_info_weight_name() {
        // Graph has two MatMul nodes referencing different weights. The
        // template builder must pick the node whose input is info.weight_name,
        // not the first MatMul encountered.
        let x = onnx_proto::make_tensor_value_info("input", TensorProto::FLOAT, &[4, 64]);
        let y = onnx_proto::make_tensor_value_info("output", TensorProto::FLOAT, &[4, 2048]);

        let w_small = onnx_proto::make_tensor(
            "w_small",
            TensorProto::FLOAT,
            &[64, 64],
            vec![0.0f32; 64 * 64],
        );
        let w_big = onnx_proto::make_tensor(
            "w_big",
            TensorProto::FLOAT,
            &[64, 2048],
            vec![0.0f32; 64 * 2048],
        );

        let n1 = onnx_proto::make_node(
            "MatMul",
            vec!["input".into(), "w_small".into()],
            vec!["mid".into()],
            vec![],
        );
        let n2 = onnx_proto::make_node(
            "MatMul",
            vec!["mid".into(), "w_big".into()],
            vec!["output".into()],
            vec![],
        );

        let graph = onnx_proto::make_graph(
            "two_matmul",
            vec![n1, n2],
            vec![x],
            vec![y],
            vec![w_small, w_big],
        );
        let model = onnx_proto::make_model(graph, 13);

        let info = crate::schema::tiling::DimSplitInfo {
            slice_idx: 0,
            weight_name: Some("w_big".to_string()),
            input_name: "mid".to_string(),
            output_name: "output".to_string(),
            k_dim: 64,
            n_dim: 2048,
            k_chunks: 1,
            ..Default::default()
        };

        let tmp = tempfile::tempdir().unwrap();
        let tmpl_path = create_dim_split_template(&model, &info, tmp.path(), None).unwrap();
        let tmpl_model = onnx_proto::load_model(&tmpl_path).unwrap();
        let g = tmpl_model.graph.as_ref().unwrap();
        let w = g.initializer.iter().find(|i| i.name == "W").unwrap();
        // Template weight shape must reflect w_big (64, 2048), not w_small.
        assert_eq!(w.dims, vec![64, 2048]);
    }

    #[test]
    fn create_matmul_dim_template_disambiguates_shared_weight() {
        // Two MatMul ops share the same weight initializer (e.g. tied
        // weights). The template builder must select the op whose
        // input/output names match info, not the first node that happens
        // to reference the initializer.
        let x = onnx_proto::make_tensor_value_info("input", TensorProto::FLOAT, &[4, 64]);
        let y_a = onnx_proto::make_tensor_value_info("out_a", TensorProto::FLOAT, &[4, 32]);
        let y_b = onnx_proto::make_tensor_value_info("out_b", TensorProto::FLOAT, &[1, 32]);

        let shared_w = onnx_proto::make_tensor(
            "tied_w",
            TensorProto::FLOAT,
            &[64, 32],
            vec![0.0f32; 64 * 32],
        );

        // First op: input -> tied_w -> out_a (shape [4, 32])
        let n_a = onnx_proto::make_node(
            "MatMul",
            vec!["input".into(), "tied_w".into()],
            vec!["out_a".into()],
            vec![],
        );
        // Second op: alt_in -> tied_w -> out_b (shape [1, 32])
        let alt_in = onnx_proto::make_tensor_value_info("alt_in", TensorProto::FLOAT, &[1, 64]);
        let n_b = onnx_proto::make_node(
            "MatMul",
            vec!["alt_in".into(), "tied_w".into()],
            vec!["out_b".into()],
            vec![],
        );

        let graph = onnx_proto::make_graph(
            "shared_weight",
            vec![n_a, n_b],
            vec![x, alt_in],
            vec![y_a, y_b],
            vec![shared_w],
        );
        let model = onnx_proto::make_model(graph, 13);

        // Target the second op explicitly via input_name/output_name.
        let info = crate::schema::tiling::DimSplitInfo {
            slice_idx: 0,
            weight_name: Some("tied_w".to_string()),
            input_name: "alt_in".to_string(),
            output_name: "out_b".to_string(),
            k_dim: 64,
            n_dim: 32,
            k_chunks: 1,
            ..Default::default()
        };

        let tmp = tempfile::tempdir().unwrap();
        // Builder should succeed by binding the second op (the one whose
        // IO matches info), even though the first op also references the
        // same weight initializer.
        let tmpl_path = create_dim_split_template(&model, &info, tmp.path(), None).unwrap();
        let tmpl_model = onnx_proto::load_model(&tmpl_path).unwrap();
        let g = tmpl_model.graph.as_ref().unwrap();
        let w = g.initializer.iter().find(|i| i.name == "W").unwrap();
        assert_eq!(w.dims, vec![64, 32]);
    }

    fn make_maxpool_node(
        kernel: i64,
        stride: i64,
        pads: [i64; 4],
        ceil_mode: Option<i64>,
    ) -> NodeProto {
        let mut attrs = vec![
            onnx_proto::make_attribute_ints("kernel_shape", &[kernel, kernel]),
            onnx_proto::make_attribute_ints("strides", &[stride, stride]),
            onnx_proto::make_attribute_ints("pads", &pads),
        ];
        if let Some(cm) = ceil_mode {
            attrs.push(onnx_proto::make_attribute_int("ceil_mode", cm));
        }
        onnx_proto::make_node(
            "MaxPool",
            vec!["input".into()],
            vec!["output".into()],
            attrs,
        )
    }

    #[test]
    fn pool_params_valid() {
        let node = make_maxpool_node(2, 2, [0, 0, 0, 0], None);
        let pp = PoolParams::from_node(&node, 0);
        assert!(pp.is_some());
        let pp = pp.unwrap();
        assert_eq!(pp.kernel, [2, 2]);
        assert_eq!(pp.stride, [2, 2]);
    }

    #[test]
    fn pool_params_rejects_ceil_mode() {
        let node = make_maxpool_node(2, 2, [0, 0, 0, 0], Some(1));
        assert!(PoolParams::from_node(&node, 0).is_none());
    }

    #[test]
    fn pool_params_accepts_ceil_mode_zero() {
        let node = make_maxpool_node(2, 2, [0, 0, 0, 0], Some(0));
        assert!(PoolParams::from_node(&node, 0).is_some());
    }

    #[test]
    fn pool_params_rejects_auto_pad() {
        let mut attrs = vec![
            onnx_proto::make_attribute_ints("kernel_shape", &[2, 2]),
            onnx_proto::make_attribute_ints("strides", &[2, 2]),
        ];
        attrs.push(onnx_proto::AttributeProto {
            name: "auto_pad".into(),
            s: b"SAME_UPPER".to_vec(),
            ..Default::default()
        });
        let node = onnx_proto::make_node(
            "MaxPool",
            vec!["input".into()],
            vec!["output".into()],
            attrs,
        );
        assert!(PoolParams::from_node(&node, 0).is_none());
    }

    #[test]
    fn pool_params_rejects_non_maxpool() {
        let node = onnx_proto::make_node(
            "Conv",
            vec!["input".into()],
            vec!["output".into()],
            vec![onnx_proto::make_attribute_ints("kernel_shape", &[3, 3])],
        );
        assert!(PoolParams::from_node(&node, 0).is_none());
    }

    fn make_elementwise_model(op: &str, shape: &[i64]) -> ModelProto {
        let x = onnx_proto::make_tensor_value_info("input", TensorProto::FLOAT, shape);
        let y = onnx_proto::make_tensor_value_info("output", TensorProto::FLOAT, shape);
        let node = onnx_proto::make_node(op, vec!["input".into()], vec!["output".into()], vec![]);
        let graph = onnx_proto::make_graph("test", vec![node], vec![x], vec![y], vec![]);
        onnx_proto::make_model(graph, 13)
    }

    #[test]
    fn fixed_segments_too_small_returns_none() {
        let model = make_elementwise_model("Relu", &[1, 3, 8, 8]);
        assert!(detect_elementwise_fixed_segments(model.graph.as_ref().unwrap()).is_none());
    }

    #[test]
    fn fixed_segments_detects_large_tensor() {
        let model = make_elementwise_model("Relu", &[1, 16, 64, 64]);
        let graph = model.graph.as_ref().unwrap();
        let det = detect_elementwise_fixed_segments(graph);
        assert!(det.is_some());
        if let Some(TilingDetection::FixedSegment {
            segment_size,
            total_elements,
            num_segments,
            ..
        }) = det
        {
            assert_eq!(total_elements, 16 * 64 * 64);
            assert_eq!(segment_size, ELEMENTWISE_SEGMENT_SIZE);
            assert_eq!(
                num_segments,
                (total_elements + segment_size - 1) / segment_size
            );
        } else {
            panic!("expected FixedSegment variant");
        }
    }

    #[test]
    fn fixed_segments_rejects_zero_dim() {
        let model = make_elementwise_model("Relu", &[1, 0, 64, 64]);
        assert!(detect_elementwise_fixed_segments(model.graph.as_ref().unwrap()).is_none());
    }

    #[test]
    fn fixed_segments_rejects_non_elementwise() {
        let x = onnx_proto::make_tensor_value_info("input", TensorProto::FLOAT, &[1, 16, 64, 64]);
        let y = onnx_proto::make_tensor_value_info("output", TensorProto::FLOAT, &[1, 16, 64, 64]);
        let node = onnx_proto::make_node(
            "Softmax",
            vec!["input".into()],
            vec!["output".into()],
            vec![],
        );
        let graph = onnx_proto::make_graph("test", vec![node], vec![x], vec![y], vec![]);
        let model = onnx_proto::make_model(graph, 13);
        assert!(detect_elementwise_fixed_segments(model.graph.as_ref().unwrap()).is_none());
    }

    #[test]
    fn create_pool_tile_slice_valid() {
        let x = onnx_proto::make_tensor_value_info("input", TensorProto::FLOAT, &[1, 3, 64, 64]);
        let y = onnx_proto::make_tensor_value_info("output", TensorProto::FLOAT, &[1, 3, 32, 32]);
        let node = make_maxpool_node(2, 2, [0, 0, 0, 0], None);
        let graph = onnx_proto::make_graph("pool", vec![node], vec![x], vec![y], vec![]);
        let model = onnx_proto::make_model(graph, 13);
        let tmp = tempfile::tempdir().unwrap();
        let result = create_pool_tile_slice(&model, 16, 0, tmp.path());
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.path.contains("tile.onnx"));
    }

    #[test]
    fn create_pool_tile_slice_rejects_zero_tile() {
        let x = onnx_proto::make_tensor_value_info("input", TensorProto::FLOAT, &[1, 3, 64, 64]);
        let y = onnx_proto::make_tensor_value_info("output", TensorProto::FLOAT, &[1, 3, 32, 32]);
        let node = make_maxpool_node(2, 2, [0, 0, 0, 0], None);
        let graph = onnx_proto::make_graph("pool", vec![node], vec![x], vec![y], vec![]);
        let model = onnx_proto::make_model(graph, 13);
        let tmp = tempfile::tempdir().unwrap();
        assert!(create_pool_tile_slice(&model, 0, 0, tmp.path()).is_err());
    }

    #[test]
    fn create_pool_tile_slice_no_pool_node() {
        let x = onnx_proto::make_tensor_value_info("input", TensorProto::FLOAT, &[1, 3, 64, 64]);
        let y = onnx_proto::make_tensor_value_info("output", TensorProto::FLOAT, &[1, 3, 64, 64]);
        let node =
            onnx_proto::make_node("Relu", vec!["input".into()], vec!["output".into()], vec![]);
        let graph = onnx_proto::make_graph("no_pool", vec![node], vec![x], vec![y], vec![]);
        let model = onnx_proto::make_model(graph, 13);
        let tmp = tempfile::tempdir().unwrap();
        assert!(create_pool_tile_slice(&model, 16, 0, tmp.path()).is_err());
    }

    #[test]
    fn estimate_slice_constraints_clamps_symbolic_dimensions() {
        // ONNX serializes dynamic axes as -1 and placeholder axes as 0.
        // Both must be clamped to 1 before forwarding to the jstprove
        // estimator, otherwise product(shape) multiplies by zero and
        // collapses the op's cost contribution to 0.
        let node = NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };

        let mut symbolic_shapes = HashMap::new();
        symbolic_shapes.insert("input".to_string(), vec![-1, 64]);
        symbolic_shapes.insert("weight".to_string(), vec![64, 128]);
        symbolic_shapes.insert("output".to_string(), vec![0, 128]);

        let mut concrete_shapes = HashMap::new();
        concrete_shapes.insert("input".to_string(), vec![1, 64]);
        concrete_shapes.insert("weight".to_string(), vec![64, 128]);
        concrete_shapes.insert("output".to_string(), vec![1, 128]);

        let nodes = [node];
        let symbolic_cost = estimate_slice_constraints(&nodes, &symbolic_shapes);
        let concrete_cost = estimate_slice_constraints(&nodes, &concrete_shapes);

        assert!(
            symbolic_cost > 0,
            "symbolic dims must not collapse cost to zero"
        );
        assert_eq!(
            symbolic_cost, concrete_cost,
            "batch -1 and batch 0 must clamp to 1 and match concrete batch 1"
        );
    }
}
