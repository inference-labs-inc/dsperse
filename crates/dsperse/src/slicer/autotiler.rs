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
    for init in &graph.initializer {
        let vol: i64 = init.dims.iter().product();
        if vol > 1 && seg_size % vol != 0 {
            return None;
        }
    }
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

pub const MAX_ESTIMATED_CONSTRAINTS: u64 = 500_000;

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
}

pub fn estimate_slice_constraints(nodes: &[NodeProto], shapes: &HashMap<String, Vec<i64>>) -> u64 {
    let mut total: u64 = 0;
    for node in nodes {
        let output_elements: u64 = node
            .output
            .first()
            .and_then(|name| shapes.get(name))
            .map(|s| s.iter().filter(|&&d| d > 0).map(|&d| d as u64).product())
            .unwrap_or(0);

        let cost = match node.op_type.as_str() {
            "MatMul" | "Gemm" => {
                let input_last_dim: u64 = node
                    .input
                    .first()
                    .and_then(|name| shapes.get(name))
                    .and_then(|s| s.last())
                    .map(|&d| d.max(0) as u64)
                    .unwrap_or(1);
                output_elements
                    .saturating_mul(input_last_dim)
                    .saturating_mul(2)
            }
            "Softmax" => output_elements.saturating_mul(4),
            "Conv" => output_elements.saturating_mul(3),
            _ => output_elements.saturating_mul(2),
        };
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

    for node in nodes {
        if matches!(node.op_type.as_str(), "MatMul" | "Gemm") {
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
            let trans_b = node.op_type == "Gemm"
                && super::onnx_proto::get_attribute_int(node, "transB").unwrap_or(0) == 1;
            let (n_dim, split_dim) = if trans_b {
                (weight_shape[0] as usize, 0)
            } else {
                (weight_shape[1] as usize, 1)
            };
            if n_dim <= 1 {
                continue;
            }
            let num_groups = target_groups.min(n_dim);
            let elements_per_group = n_dim.div_ceil(num_groups);
            let Some(output_shape) = node.output.first().and_then(|name| shapes.get(name)) else {
                continue;
            };
            let concat_axis = output_shape.len().saturating_sub(1);
            let Some(input_name) = node.input.first().filter(|s| !s.is_empty()).cloned() else {
                continue;
            };
            let Some(output_name) = node.output.first().filter(|s| !s.is_empty()).cloned() else {
                continue;
            };
            return Some(DimSplitDetection {
                split_kind: DimSplitKind::MatMulOutputDim,
                split_dim,
                dim_size: n_dim,
                num_groups,
                elements_per_group,
                input_name,
                output_name,
                concat_axis,
                estimated_constraints: estimated,
                weight_name: Some(weight_name.clone()),
            });
        }
    }

    for node in nodes {
        if node.op_type == "Softmax" {
            let Some(input_shape) = node.input.first().and_then(|name| shapes.get(name)) else {
                continue;
            };
            if input_shape.len() == 4 && input_shape[1] > 1 {
                let head_dim = input_shape[1] as usize;
                let num_groups = target_groups.min(head_dim);
                let elements_per_group = head_dim.div_ceil(num_groups);
                let Some(input_name) = node.input.first().filter(|s| !s.is_empty()).cloned() else {
                    continue;
                };
                let Some(output_name) = node.output.first().filter(|s| !s.is_empty()).cloned()
                else {
                    continue;
                };
                return Some(DimSplitDetection {
                    split_kind: DimSplitKind::HeadDim,
                    split_dim: 1,
                    dim_size: head_dim,
                    num_groups,
                    elements_per_group,
                    input_name,
                    output_name,
                    concat_axis: 1,
                    estimated_constraints: estimated,
                    weight_name: None,
                });
            }
        }
    }

    let first_input_shape = nodes
        .first()
        .and_then(|n| n.input.first())
        .and_then(|name| shapes.get(name))?;
    if !first_input_shape.is_empty() && first_input_shape[0] > 1 {
        let batch_dim = first_input_shape[0] as usize;
        let num_groups = target_groups.min(batch_dim);
        let elements_per_group = batch_dim.div_ceil(num_groups);
        let input_name = nodes
            .first()
            .and_then(|n| n.input.first())
            .filter(|s| !s.is_empty())
            .cloned()?;
        let output_name = nodes
            .last()
            .and_then(|n| n.output.first())
            .filter(|s| !s.is_empty())
            .cloned()?;
        return Some(DimSplitDetection {
            split_kind: DimSplitKind::BatchDim,
            split_dim: 0,
            dim_size: batch_dim,
            num_groups,
            elements_per_group,
            input_name,
            output_name,
            concat_axis: 0,
            estimated_constraints: estimated,
            weight_name: None,
        });
    }

    None
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

    let mut nodes = Vec::new();
    for orig_node in &graph.node {
        let new_inputs: Vec<String> = orig_node
            .input
            .iter()
            .map(|name| {
                for (orig, tile) in &orig_to_tile {
                    if name == orig {
                        return tile.clone();
                    }
                }
                name.clone()
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
        shapes.insert("input".to_string(), vec![4, 145, 384]);
        shapes.insert("weight".to_string(), vec![1536, 384]);
        shapes.insert("output".to_string(), vec![4, 145, 1536]);

        let mut init_names = HashSet::new();
        init_names.insert("weight".to_string());

        let detection = detect_dim_split(&[node], &shapes, &init_names);
        assert!(detection.is_some());
        let d = detection.unwrap();
        assert_eq!(d.split_dim, 0);
        assert_eq!(d.dim_size, 1536);
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
        assert_eq!(d.split_dim, 1);
        assert_eq!(d.dim_size, 1536);
        assert!(matches!(d.split_kind, DimSplitKind::MatMulOutputDim));
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
}
