use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::analyzer::TiledResult;
use super::onnx_proto::{
    self, GraphProto, ModelProto, NodeProto, TensorProto,
};
use crate::error::Result;
use crate::schema::tiling::{ChannelGroupInfo, ChannelSplitInfo, TileInfo, TilingInfo};

const ELEMENTWISE_OPS: &[&str] = &[
    "Sigmoid", "Mul", "Add", "Sub", "Div", "Relu", "LeakyRelu", "PRelu",
    "Tanh", "Clip", "Neg", "Abs", "Sqrt", "Exp", "Log", "Pow", "Sin", "Cos",
];

pub const JSTPROVE_SUPPORTED_OPS: &[&str] = &[
    "Add", "Clip", "BatchNormalization", "Div", "Sub",
    "Mul", "Constant", "Flatten", "Gemm",
    "MaxPool", "Max", "Min", "Relu", "Reshape", "Conv",
];

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

struct ConvParams {
    node_idx: usize,
    kernel: [i64; 2],
    stride: [i64; 2],
    dilation: [i64; 2],
    pads: [i64; 4],
    group: i64,
}

fn get_conv_params(graph: &GraphProto) -> Option<ConvParams> {
    for (idx, node) in graph.node.iter().enumerate() {
        if node.op_type == "Conv" {
            let kernel = onnx_proto::get_attribute_ints(node, "kernel_shape")
                .and_then(|v| if v.len() >= 2 { Some([v[0], v[1]]) } else { None })
                .unwrap_or([3, 3]);
            let stride = onnx_proto::get_attribute_ints(node, "strides")
                .and_then(|v| if v.len() >= 2 { Some([v[0], v[1]]) } else { None })
                .unwrap_or([1, 1]);
            let dilation = onnx_proto::get_attribute_ints(node, "dilations")
                .and_then(|v| if v.len() >= 2 { Some([v[0], v[1]]) } else { None })
                .unwrap_or([1, 1]);
            let pads = onnx_proto::get_attribute_ints(node, "pads")
                .and_then(|v| if v.len() >= 4 { Some([v[0], v[1], v[2], v[3]]) } else { None })
                .unwrap_or([0, 0, 0, 0]);
            let group = onnx_proto::get_attribute_int(node, "group").unwrap_or(1);

            return Some(ConvParams {
                node_idx: idx,
                kernel,
                stride,
                dilation,
                pads,
                group,
            });
        }
    }
    None
}

fn compute_halo_size(kernel: [i64; 2], dilation: [i64; 2]) -> [i64; 2] {
    let eff_kh = (kernel[0] - 1) * dilation[0] + 1;
    let eff_kw = (kernel[1] - 1) * dilation[1] + 1;
    [eff_kh / 2, eff_kw / 2]
}

fn compute_min_spatial_tile(kernel: [i64; 2], dilation: [i64; 2]) -> i64 {
    let eff_kh = (kernel[0] - 1) * dilation[0] + 1;
    let eff_kw = (kernel[1] - 1) * dilation[1] + 1;
    eff_kh.max(eff_kw) + 1
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
    if cp.kernel[0] % 2 == 0 || cp.kernel[1] % 2 == 0 {
        return false;
    }
    let halo = compute_halo_size(cp.kernel, cp.dilation);
    cp.pads == [halo[0], halo[1], halo[0], halo[1]]
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
    if dims.len() != 4 || dims[2] != dims[3] {
        return None;
    }
    Some((inp.name.clone(), out.name.clone(), dims[1], dims[2], dims[3]))
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

fn find_optimal_tile_size(spatial_dim: i64, target: i64, min_tile: i64, stride: i64) -> Option<i64> {
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
    if !is_tileable(graph) {
        return None;
    }
    let (inp_name, out_name, c_in, h, w) = get_model_dimensions(graph)?;
    let cp = get_conv_params(graph)?;
    if cp.stride[0] == 0 || cp.stride[1] == 0 {
        return None;
    }
    let (wi, _) = find_weights_and_bias(graph, &graph.node[cp.node_idx]);
    let weights = wi?;
    if weights.dims.is_empty() {
        return None;
    }
    let c_out = weights.dims[0];
    let tile_size = tile_size? as i64;
    let min_tile = compute_min_spatial_tile(cp.kernel, cp.dilation);

    let (actual_tile, skip_reason) =
        calculate_spatial_tile_config(c_in, h, w, tile_size, min_tile, cp.stride[0]);

    if let Some(actual_tile) = actual_tile {
        if h % actual_tile != 0 || w % actual_tile != 0 {
            return None;
        }
        let tiles_y = h / actual_tile;
        let tiles_x = w / actual_tile;
        if tiles_y * tiles_x < 2 {
            return None;
        }
        let halo = compute_halo_size(cp.kernel, cp.dilation);
        return Some(TilingDetection::Spatial {
            input_name: inp_name,
            output_name: out_name,
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

    if matches!(skip_reason, Some("min_tile_too_large" | "no_divisor")) && is_channel_splittable(graph) {
        if let Some((num_groups, cpg)) =
            calculate_channel_split_config(c_in, c_out, h, w, tile_size)
        {
            return Some(TilingDetection::ChannelSplit {
                input_name: inp_name,
                output_name: out_name,
                c_in,
                c_out,
                h,
                w,
                num_groups,
                channels_per_group: cpg,
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
        c_in: i64,
        c_out: i64,
        h: i64,
        w: i64,
        tile_size: i64,
        halo: [i64; 2],
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
) -> Result<Option<TileSliceResult>> {
    let graph = match model.graph.as_ref() {
        Some(g) => g,
        None => return Ok(None),
    };
    let cp = match get_conv_params(graph) {
        Some(c) => c,
        None => return Ok(None),
    };
    if cp.stride[0] == 0 || cp.stride[1] == 0 {
        return Ok(None);
    }
    let conv_node = &graph.node[cp.node_idx];
    let (wi, bias) = find_weights_and_bias(graph, conv_node);
    let weights = match wi {
        Some(w) => w,
        None => return Ok(None),
    };
    if weights.dims.len() < 4 {
        return Ok(None);
    }

    let halo = compute_halo_size(cp.kernel, cp.dilation);
    let eff_kh = (cp.kernel[0] - 1) * cp.dilation[0] + 1;
    let eff_kw = (cp.kernel[1] - 1) * cp.dilation[1] + 1;
    let tile_h = tile_size + 2 * halo[0];
    let tile_w = tile_size + 2 * halo[1];
    let out_h = (tile_h - eff_kh) / cp.stride[0] + 1;
    let out_w = (tile_w - eff_kw) / cp.stride[1] + 1;
    if out_h <= 0 || out_w <= 0 {
        return Ok(None);
    }

    let c_in = graph
        .input
        .first()
        .map(|i| onnx_proto::vi_shape(i))
        .and_then(|s| if s.len() == 4 { Some(s[1]) } else { None })
        .unwrap_or(weights.dims.get(1).copied().unwrap_or(1));

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

    integrate_extra_ops(graph, conv_node, &mut initializers, &mut nodes);

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

    Ok(Some(TileSliceResult {
        path: format!("slice_{slice_idx}/payload/tiles/tile.onnx"),
        conv_out: [out_h, out_w],
    }))
}

fn integrate_extra_ops(
    graph: &GraphProto,
    conv_node: &NodeProto,
    initializers: &mut Vec<onnx_proto::TensorProto>,
    nodes: &mut Vec<NodeProto>,
) {
    let orig_input_name = graph
        .input
        .first()
        .map(|i| i.name.as_str())
        .unwrap_or("");

    let non_conv: Vec<&NodeProto> = graph
        .node
        .iter()
        .filter(|n| n.op_type != "Conv")
        .collect();

    if non_conv.is_empty() {
        assert!(!nodes.is_empty(), "integrate_extra_ops requires at least one node");
        nodes.last_mut().unwrap().output[0] = "tile_out".to_string();
        return;
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
}

pub fn create_channel_group_slice(
    model: &ModelProto,
    group_idx: usize,
    c_start: i64,
    c_end: i64,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<Option<ChannelGroupInfo>> {
    let graph = match model.graph.as_ref() {
        Some(g) => g,
        None => return Ok(None),
    };
    let cp = match get_conv_params(graph) {
        Some(c) => c,
        None => return Ok(None),
    };
    if c_start < 0 || c_end < 0 || c_start >= c_end {
        return Ok(None);
    }
    if cp.stride[0] == 0 || cp.stride[1] == 0 {
        return Ok(None);
    }
    let conv_node = &graph.node[cp.node_idx];
    let (wi, _) = find_weights_and_bias(graph, conv_node);
    let weights = match wi {
        Some(w) => w,
        None => return Ok(None),
    };
    if weights.dims.len() < 4 {
        return Ok(None);
    }

    let dims = match get_model_dimensions(graph) {
        Some(d) => d,
        None => return Ok(None),
    };
    let (_inp_name, _out_name, _c_in, h_in, w_in) = dims;

    let c_group = c_end - c_start;
    let eff_kh = (cp.kernel[0] - 1) * cp.dilation[0] + 1;
    let eff_kw = (cp.kernel[1] - 1) * cp.dilation[1] + 1;
    let h_out = (h_in + cp.pads[0] + cp.pads[2] - eff_kh) / cp.stride[0] + 1;
    let w_out = (w_in + cp.pads[1] + cp.pads[3] - eff_kw) / cp.stride[1] + 1;
    if h_out <= 0 || w_out <= 0 {
        return Ok(None);
    }
    let c_out = weights.dims[0];

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

    let sliced_weights = slice_weights(&weights, c_start as usize, c_end as usize);

    let w_tensor = onnx_proto::make_tensor(
        "W",
        TensorProto::FLOAT,
        &sliced_weights.dims,
        sliced_weights.data,
    );

    let node = onnx_proto::make_node(
        "Conv",
        vec![input_name, "W".to_string()],
        vec![output_name],
        vec![
            onnx_proto::make_attribute_ints("kernel_shape", &cp.kernel),
            onnx_proto::make_attribute_ints("strides", &cp.stride),
            onnx_proto::make_attribute_ints("pads", &cp.pads),
            onnx_proto::make_attribute_ints("dilations", &cp.dilation),
        ],
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

    Ok(Some(ChannelGroupInfo {
        group_idx,
        c_start: c_start as usize,
        c_end: c_end as usize,
        path: format!("slice_{slice_idx}/payload/channel_groups/group_{group_idx}.onnx"),
        jstprove_circuit_path: None,
        settings_path: None,
        vk_path: None,
        pk_path: None,
        jstprove_settings_path: None,
        ezkl_circuit_path: None,
        ezkl_settings_path: None,
        ezkl_pk_path: None,
        ezkl_vk_path: None,
    }))
}

fn slice_weights(weights: &WeightInfo, c_start: usize, c_end: usize) -> WeightInfo {
    assert!(
        weights.dims.len() >= 4,
        "slice_weights: expected >= 4 dims, got {}",
        weights.dims.len()
    );
    let c_out = weights.dims[0] as usize;
    let c_in = weights.dims[1] as usize;
    let kh = weights.dims[2] as usize;
    let kw = weights.dims[3] as usize;
    let expected_len = c_out * c_in * kh * kw;
    assert!(
        weights.data.len() == expected_len,
        "slice_weights: data length {} != expected {} (dims={:?})",
        weights.data.len(),
        expected_len,
        weights.dims
    );
    assert!(
        c_end <= c_in,
        "slice_weights: c_end ({c_end}) exceeds c_in ({c_in})"
    );
    let c_group = c_end - c_start;

    let mut sliced = Vec::with_capacity(c_out * c_group * kh * kw);
    for o in 0..c_out {
        for c in c_start..c_end {
            for h in 0..kh {
                for w_idx in 0..kw {
                    let idx = o * c_in * kh * kw + c * kh * kw + h * kw + w_idx;
                    sliced.push(weights.data[idx]);
                }
            }
        }
    }

    WeightInfo {
        data: sliced,
        dims: vec![c_out as i64, c_group as i64, kh as i64, kw as i64],
    }
}

pub fn save_conv_bias(
    model: &ModelProto,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<Option<String>> {
    let graph = match model.graph.as_ref() {
        Some(g) => g,
        None => return Ok(None),
    };
    let cp = match get_conv_params(graph) {
        Some(c) => c,
        None => return Ok(None),
    };
    let conv_node = &graph.node[cp.node_idx];
    let (_, bias) = find_weights_and_bias(graph, conv_node);
    let Some(bias_data) = bias else {
        return Ok(None);
    };

    let groups_dir = output_dir.join("channel_groups");
    std::fs::create_dir_all(&groups_dir)
        .map_err(|e| crate::error::DsperseError::io(e, &groups_dir))?;

    let bias_json = serde_json::to_string(&bias_data)?;
    let bias_path = groups_dir.join("bias.json");
    std::fs::write(&bias_path, bias_json)
        .map_err(|e| crate::error::DsperseError::io(e, &bias_path))?;

    Ok(Some(format!(
        "slice_{slice_idx}/payload/channel_groups/bias.json"
    )))
}

pub fn apply_channel_splitting(
    model: &ModelProto,
    c_in: i64,
    c_out: i64,
    num_groups: i64,
    channels_per_group: i64,
    input_name: &str,
    output_name: &str,
    h: i64,
    w: i64,
    slice_idx: usize,
    output_dir: &Path,
) -> Result<Option<ChannelSplitInfo>> {
    let graph = match model.graph.as_ref() {
        Some(g) => g,
        None => return Ok(None),
    };
    let cp = match get_conv_params(graph) {
        Some(c) => c,
        None => return Ok(None),
    };
    if cp.stride[0] == 0 || cp.stride[1] == 0 {
        return Ok(None);
    }
    let eff_kh = (cp.kernel[0] - 1) * cp.dilation[0] + 1;
    let eff_kw = (cp.kernel[1] - 1) * cp.dilation[1] + 1;
    let out_h = (h + cp.pads[0] + cp.pads[2] - eff_kh) / cp.stride[0] + 1;
    let out_w = (w + cp.pads[1] + cp.pads[3] - eff_kw) / cp.stride[1] + 1;
    if out_h <= 0 || out_w <= 0 {
        return Ok(None);
    }

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

        let group_info = match create_channel_group_slice(
            model,
            g as usize,
            c_start,
            c_end,
            slice_idx,
            output_dir,
        ) {
            Ok(info) => info,
            Err(e) => {
                cleanup();
                return Err(e);
            }
        };

        match group_info {
            Some(gi) => groups.push(gi),
            None => {
                cleanup();
                return Ok(None);
            }
        }
    }

    let bias_path = match save_conv_bias(model, slice_idx, output_dir) {
        Ok(p) => p,
        Err(e) => {
            cleanup();
            return Err(e);
        }
    };

    Ok(Some(ChannelSplitInfo {
        slice_idx,
        c_in: c_in as usize,
        c_out: c_out as usize,
        num_groups: num_groups as usize,
        channels_per_group: channels_per_group as usize,
        input_name: input_name.to_string(),
        output_name: output_name.to_string(),
        h: h as usize,
        w: w as usize,
        out_h: out_h as usize,
        out_w: out_w as usize,
        groups,
        bias_path,
    }))
}

pub fn apply_tiling(
    slices_paths: &HashMap<usize, String>,
    tile_size: usize,
) -> Result<HashMap<usize, TiledResult>> {
    let mut results = HashMap::new();

    for (&idx, onnx_path) in slices_paths {
        let path = Path::new(onnx_path);
        if !path.exists() {
            tracing::warn!(slice = idx, path = %onnx_path, "ONNX file not found, skipping tiling");
            continue;
        }
        let model = onnx_proto::load_model(path)?;
        let detection = match detect_tiling_needs(&model, Some(tile_size)) {
            Some(d) => d,
            None => continue,
        };

        let output_dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        match detection {
            TilingDetection::ChannelSplit {
                input_name,
                output_name,
                c_in,
                c_out,
                h,
                w,
                num_groups,
                channels_per_group,
            } => {
                tracing::info!(
                    slice = idx,
                    c_in,
                    h,
                    w,
                    num_groups,
                    "channel splitting Conv slice"
                );
                if let Some(info) = apply_channel_splitting(
                    &model,
                    c_in,
                    c_out,
                    num_groups,
                    channels_per_group,
                    &input_name,
                    &output_name,
                    h,
                    w,
                    idx,
                    output_dir,
                )? {
                    results.insert(
                        idx,
                        TiledResult {
                            channel_split: Some(info),
                            tiling: None,
                        },
                    );
                }
            }
            TilingDetection::Spatial {
                input_name,
                output_name,
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
                tracing::info!(
                    slice = idx,
                    c_in,
                    h,
                    w,
                    tile_size = actual_tile,
                    tiles = tiles_y * tiles_x,
                    "tiling Conv slice"
                );
                if let Some(tile_result) = create_tile_slice(&model, actual_tile, idx, output_dir)?
                {
                    let info = TilingInfo {
                        slice_idx: idx,
                        tile_size: actual_tile as usize,
                        num_tiles: (tiles_y * tiles_x) as usize,
                        tiles_y: tiles_y as usize,
                        tiles_x: tiles_x as usize,
                        halo,
                        out_tile,
                        stride,
                        c_in: c_in as usize,
                        c_out: c_out as usize,
                        input_name,
                        output_name,
                        tile: Some(TileInfo {
                            path: tile_result.path,
                            conv_out: tile_result.conv_out,
                            jstprove_circuit_path: None,
                        }),
                        tiles: None,
                    };
                    results.insert(
                        idx,
                        TiledResult {
                            tiling: Some(info),
                            channel_split: None,
                        },
                    );
                }
            }
        }
    }

    tracing::info!(count = results.len(), "tiled Conv slices");
    Ok(results)
}

pub struct TileSliceResult {
    pub path: String,
    pub conv_out: [i64; 2],
}
