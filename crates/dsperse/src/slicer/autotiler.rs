use std::collections::HashSet;
use std::path::Path;

use super::ELEMENTWISE_OPS;
use super::onnx_proto::{self, GraphProto, ModelProto, NodeProto, TensorProto};
use crate::error::Result;
use crate::schema::tiling::{ChannelGroupInfo, ChannelSplitInfo};

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
    ELEMENTWISE_OPS.contains(&op)
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

fn is_standard_conv_slice(graph: &GraphProto) -> Option<ConvParams> {
    if graph.input.len() > 1 {
        return None;
    }
    let conv_count = graph.node.iter().filter(|n| n.op_type == "Conv").count();
    if conv_count != 1 {
        return None;
    }
    let conv_params = get_conv_params(graph)?;
    if conv_params.node_idx != 0 {
        return None;
    }
    let ops: HashSet<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
    let non_conv: HashSet<&&str> = ops.iter().filter(|&&o| o != "Conv").collect();
    if !non_conv.iter().all(|&&o| is_elementwise(o)) {
        return None;
    }
    Some(conv_params)
}

fn is_tileable(graph: &GraphProto) -> bool {
    let Some(cp) = is_standard_conv_slice(graph) else {
        return false;
    };
    let Some(eff) = effective_kernel(cp.kernel, cp.dilation) else {
        return false;
    };
    let total_pad_h = cp.pads[0] + cp.pads[2];
    let total_pad_w = cp.pads[1] + cp.pads[3];
    total_pad_h >= eff[0] - cp.stride[0] && total_pad_w >= eff[1] - cp.stride[1]
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

fn get_elementwise_dimensions_3d(graph: &GraphProto) -> Option<(Vec<String>, String, i64, i64)> {
    if graph.input.is_empty() {
        return None;
    }
    let out = graph.output.first()?;
    let first = graph.input.first()?;
    let dims = onnx_proto::vi_shape(first);
    if dims.len() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 {
        return None;
    }
    let (seq, hidden) = (dims[1], dims[2]);
    let mut input_names = Vec::with_capacity(graph.input.len());
    for inp in &graph.input {
        let d = onnx_proto::vi_shape(inp);
        if d.len() != 3 || d[0] != 1 || d[1] != seq || d[2] != hidden {
            return None;
        }
        input_names.push(inp.name.clone());
    }
    Some((input_names, out.name.clone(), hidden, seq))
}

fn get_elementwise_dimensions(graph: &GraphProto) -> Option<(Vec<String>, String, i64, i64, i64)> {
    if graph.input.is_empty() {
        return None;
    }
    let out = graph.output.first()?;
    let first = graph.input.first()?;
    let dims = onnx_proto::vi_shape(first);
    if dims.len() != 4 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0 {
        return None;
    }
    let (c, h, w) = (dims[1], dims[2], dims[3]);
    let mut input_names = Vec::with_capacity(graph.input.len());
    for inp in &graph.input {
        let d = onnx_proto::vi_shape(inp);
        if d.len() != 4 || d[1] != c || d[2] != h || d[3] != w {
            return None;
        }
        input_names.push(inp.name.clone());
    }
    Some((input_names, out.name.clone(), c, h, w))
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

pub fn detect_tiling_needs(
    model: &ModelProto,
    tile_size: Option<usize>,
) -> Option<TilingDetection> {
    let graph = model.graph.as_ref()?;
    let tile_size = tile_size? as i64;

    let dims_4d = get_model_dimensions(graph);

    if let Some((ref inp_name, ref out_name, c_in, h, w)) = dims_4d
        && let Some(cp) = get_conv_params(graph)
    {
        let c_out = cp.c_out;

        if is_tileable(graph) {
            let min_tile = compute_min_spatial_tile(cp.kernel, cp.dilation)?;
            let (actual_tile, _skip_reason) =
                calculate_spatial_tile_config(c_in, h, w, tile_size, min_tile, cp.stride[0]);

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
                calculate_channel_split_config(c_in, c_out, h, w, tile_size)
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

    if is_elementwise_only_slice(graph)
        && let Some((ew_input_names, ew_out_name, ew_c, ew_h, ew_w)) =
            get_elementwise_dimensions(graph)
    {
        let num_inputs = i64::try_from(ew_input_names.len()).ok()?;
        let per_pixel = ew_c.checked_mul(num_inputs)?;
        let total = per_pixel.checked_mul(ew_h)?.checked_mul(ew_w)?;
        if total <= tile_size {
            return None;
        }
        if per_pixel > tile_size {
            return None;
        }
        let max_spatial = ((tile_size / per_pixel) as f64).sqrt() as i64;
        if max_spatial < 1 {
            return None;
        }
        let actual_tile = max_spatial.min(ew_h).min(ew_w);
        let tiles_y = (ew_h + actual_tile - 1) / actual_tile;
        let tiles_x = (ew_w + actual_tile - 1) / actual_tile;
        if tiles_y * tiles_x >= 2 {
            let c_out = graph
                .output
                .first()
                .map(onnx_proto::vi_shape)
                .and_then(|s| (s.len() == 4).then(|| s[1]))
                .unwrap_or(ew_c);
            let primary_name = ew_input_names[0].clone();
            return Some(TilingDetection::Spatial {
                input_name: primary_name,
                output_name: ew_out_name,
                input_names: ew_input_names,
                ndim: 4,
                c_in: ew_c,
                c_out,
                h: ew_h,
                w: ew_w,
                tile_size: actual_tile,
                halo: [0, 0, 0, 0],
                tiles_y,
                tiles_x,
                out_tile: [actual_tile, actual_tile],
                stride: [1, 1],
            });
        }
    }

    if is_elementwise_only_slice(graph)
        && let Some((ew_input_names, ew_out_name, ew_hidden, ew_seq)) =
            get_elementwise_dimensions_3d(graph)
    {
        let num_inputs = i64::try_from(ew_input_names.len()).ok()?;
        let per_tile_cost = ew_hidden.checked_mul(num_inputs)?;
        let total = per_tile_cost.checked_mul(ew_seq)?;
        if total <= tile_size {
            return None;
        }
        if per_tile_cost > tile_size {
            return None;
        }
        let max_tile = tile_size / per_tile_cost;
        let actual_tile = max_tile.min(ew_seq);
        let tiles_y = (ew_seq + actual_tile - 1) / actual_tile;
        if tiles_y >= 2 {
            let c_out = graph
                .output
                .first()
                .map(onnx_proto::vi_shape)
                .and_then(|s| (s.len() == 3).then(|| s[2]))
                .unwrap_or(ew_hidden);
            let primary_name = ew_input_names[0].clone();
            return Some(TilingDetection::Spatial {
                input_name: primary_name,
                output_name: ew_out_name,
                input_names: ew_input_names,
                ndim: 3,
                c_in: ew_hidden,
                c_out,
                h: ew_seq,
                w: 1,
                tile_size: actual_tile,
                halo: [0, 0, 0, 0],
                tiles_y,
                tiles_x: 1,
                out_tile: [actual_tile, 1],
                stride: [1, 1],
            });
        }
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
            "create_tile_slice: failed to extract slice prologue from model".to_string(),
        )
    })?;
    let conv_node = &graph.node[cp.node_idx];
    let weights = weights.ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_tile_slice: conv weights not found in model initializers".to_string(),
        )
    })?;

    let halo = compute_halo_size(cp.pads).ok_or_else(|| {
        crate::error::DsperseError::Slicer("create_tile_slice: invalid pad values".to_string())
    })?;
    let tile_h = tile_size
        .checked_add(halo[0])
        .and_then(|v| v.checked_add(halo[2]))
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "create_tile_slice: tile_h overflow (tile_size={tile_size}, halo_top={}, halo_bottom={})",
                halo[0], halo[2]
            ))
        })?;
    let tile_w = tile_size
        .checked_add(halo[1])
        .and_then(|v| v.checked_add(halo[3]))
        .ok_or_else(|| {
            crate::error::DsperseError::Slicer(format!(
                "create_tile_slice: tile_w overflow (tile_size={tile_size}, halo_left={}, halo_right={})",
                halo[1], halo[3]
            ))
        })?;
    let (out_h, out_w) = conv_output_hw(
        tile_h,
        tile_w,
        [0, 0, 0, 0],
        cp.kernel,
        cp.dilation,
        cp.stride,
    )
    .ok_or_else(|| {
        crate::error::DsperseError::Slicer(format!(
            "create_tile_slice: invalid conv output dimensions for tile_h={tile_h}, tile_w={tile_w}, stride={:?}, kernel={:?}",
            cp.stride, cp.kernel
        ))
    })?;

    let graph_c_in = graph
        .input
        .first()
        .map(onnx_proto::vi_shape)
        .and_then(|s| (s.len() == 4 && s[1] > 0).then_some(s[1]));
    let cfg_c_in = cp.c_in.checked_mul(cp.group).filter(|&v| v > 0);
    if let (Some(g), Some(c)) = (graph_c_in, cfg_c_in)
        && g != c
    {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_tile_slice: graph c_in ({g}) != weight c_in*group ({c})"
        )));
    }
    let c_in = graph_c_in.or(cfg_c_in).ok_or_else(|| {
        crate::error::DsperseError::Slicer(
            "create_tile_slice: unable to determine input channels".to_string(),
        )
    })?;

    let x = onnx_proto::make_tensor_value_info(
        "tile_in",
        TensorProto::FLOAT,
        &[1, c_in, tile_h, tile_w],
    );
    let y = onnx_proto::make_tensor_value_info(
        "tile_out",
        TensorProto::FLOAT,
        &[1, weights.dims[0], out_h, out_w],
    );

    let mut initializers = vec![onnx_proto::make_tensor(
        "W",
        TensorProto::FLOAT,
        &weights.dims,
        weights.data,
    )];
    let mut conv_inputs = vec!["tile_in".to_string(), "W".to_string()];

    if let Some(bias_data) = &bias {
        let bias_dims = [weights.dims[0]];
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

    let graph = onnx_proto::make_graph(
        &format!("tile_{slice_idx}"),
        nodes,
        vec![x],
        vec![y],
        initializers,
    );
    let tile_model = onnx_proto::make_model(graph, model_opset(model));

    let tiles_dir = output_dir.join("tiles");
    std::fs::create_dir_all(&tiles_dir)
        .map_err(|e| crate::error::DsperseError::io(e, &tiles_dir))?;
    let onnx_path = tiles_dir.join("tile.onnx");
    onnx_proto::save_model(&tile_model, &onnx_path)?;

    Ok(TileSliceResult {
        path: format!("slice_{slice_idx}/payload/tiles/tile.onnx"),
        conv_out: [out_h, out_w],
    })
}

fn integrate_extra_ops(
    graph: &GraphProto,
    conv_node: &NodeProto,
    initializers: &mut Vec<onnx_proto::TensorProto>,
    nodes: &mut Vec<NodeProto>,
) -> crate::error::Result<()> {
    let orig_input_name = graph.input.first().map(|i| i.name.as_str()).unwrap_or("");

    let non_conv: Vec<&NodeProto> = graph.node.iter().filter(|n| n.op_type != "Conv").collect();

    if non_conv.is_empty() {
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

    let mut conv_weight_names: HashSet<String> = HashSet::new();
    if conv_node.input.len() > 1 {
        conv_weight_names.insert(conv_node.input[1].clone());
    }
    if conv_node.input.len() > 2 {
        conv_weight_names.insert(conv_node.input[2].clone());
    }

    for init in &graph.initializer {
        if !conv_weight_names.contains(&init.name) {
            initializers.push(init.clone());
        }
    }

    let conv_outputs: HashSet<String> = graph
        .node
        .iter()
        .filter(|n| n.op_type == "Conv")
        .flat_map(|n| n.output.iter().cloned())
        .collect();

    for (i, orig_node) in non_conv.iter().enumerate() {
        let new_inputs: Vec<String> = orig_node
            .input
            .iter()
            .map(|inp| {
                if conv_outputs.contains(inp) {
                    "conv_out".to_string()
                } else if inp == orig_input_name {
                    "tile_in".to_string()
                } else {
                    inp.clone()
                }
            })
            .collect();

        let is_last = i == non_conv.len() - 1;
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
    tile_size: i64,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<TileSliceResult> {
    if tile_size <= 0 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_elementwise_tile_slice: tile_size must be > 0, got {tile_size}"
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
    let first_dims = onnx_proto::vi_shape(&graph.input[0]);
    let rank = first_dims.len();
    if rank != 3 && rank != 4 {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_elementwise_tile_slice: unsupported input rank {rank}"
        )));
    }
    let orig_output_name = &out.name;

    let mut orig_to_tile: Vec<(String, String)> = Vec::with_capacity(graph.input.len());
    let mut tile_inputs = Vec::with_capacity(graph.input.len());
    for (idx, inp) in graph.input.iter().enumerate() {
        let tile_name = if graph.input.len() == 1 {
            "tile_in".to_string()
        } else {
            format!("tile_in_{idx}")
        };
        let inp_dims = onnx_proto::vi_shape(inp);
        if inp_dims.len() != rank {
            return Err(crate::error::DsperseError::Slicer(format!(
                "create_elementwise_tile_slice: rank mismatch for input '{}' (expected {rank}, got {})",
                inp.name,
                inp_dims.len()
            )));
        }
        let tile_shape: Vec<i64> = match rank {
            3 => {
                let hidden = inp_dims.get(2).copied().filter(|&v| v > 0).ok_or_else(|| {
                    crate::error::DsperseError::Slicer(format!(
                        "create_elementwise_tile_slice: invalid hidden dim for input '{}'",
                        inp.name
                    ))
                })?;
                vec![1, tile_size, hidden]
            }
            4 => {
                let inp_c = inp_dims.get(1).copied().filter(|&v| v > 0).ok_or_else(|| {
                    crate::error::DsperseError::Slicer(format!(
                        "create_elementwise_tile_slice: invalid c_in for input '{}'",
                        inp.name
                    ))
                })?;
                vec![1, inp_c, tile_size, tile_size]
            }
            _ => {
                return Err(crate::error::DsperseError::Slicer(format!(
                    "create_elementwise_tile_slice: unsupported input rank {rank}"
                )));
            }
        };
        tile_inputs.push(onnx_proto::make_tensor_value_info(
            &tile_name,
            TensorProto::FLOAT,
            &tile_shape,
        ));
        orig_to_tile.push((inp.name.clone(), tile_name));
    }

    let out_dims = onnx_proto::vi_shape(out);
    if out_dims.len() != rank {
        return Err(crate::error::DsperseError::Slicer(format!(
            "create_elementwise_tile_slice: output rank mismatch (expected {rank}, got {})",
            out_dims.len()
        )));
    }
    let out_tile_shape: Vec<i64> = match rank {
        3 => {
            let hidden = out_dims
                .get(2)
                .copied()
                .filter(|&v| v > 0)
                .unwrap_or(first_dims.get(2).copied().unwrap_or(1));
            vec![1, tile_size, hidden]
        }
        _ => {
            let c_in = first_dims.get(1).copied().unwrap_or(1);
            let c_out = out_dims.get(1).copied().filter(|&v| v > 0).unwrap_or(c_in);
            vec![1, c_out, tile_size, tile_size]
        }
    };
    let y = onnx_proto::make_tensor_value_info("tile_out", TensorProto::FLOAT, &out_tile_shape);

    let mut initializers = Vec::new();
    for init in &graph.initializer {
        initializers.push(init.clone());
    }

    let mut nodes = Vec::new();
    for (i, orig_node) in graph.node.iter().enumerate() {
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
        let is_last = i == graph.node.len() - 1;
        let new_outputs = if is_last {
            let mut remapped = orig_node.output.clone();
            let mut mapped = false;
            for out_name in &mut remapped {
                if out_name == orig_output_name {
                    *out_name = "tile_out".to_string();
                    mapped = true;
                }
            }
            if !mapped {
                return Err(crate::error::DsperseError::Slicer(
                    "create_elementwise_tile_slice: last node does not produce selected graph output".to_string(),
                ));
            }
            remapped
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

    let conv_out = if rank == 3 {
        [tile_size, 1]
    } else {
        [tile_size, tile_size]
    };
    Ok(TileSliceResult {
        path: format!("slice_{slice_idx}/payload/tiles/tile.onnx"),
        conv_out,
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
}
