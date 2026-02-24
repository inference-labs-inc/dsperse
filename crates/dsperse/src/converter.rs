use std::collections::{HashMap, HashSet};
use std::path::Path;

use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use jstprove_circuits::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};
use serde_json::json;

use crate::error::{DsperseError, Result};
use crate::slicer::onnx_proto::{
    self, build_initializer_map, get_attribute_int, get_attribute_ints, tensor_to_f32,
    vi_elem_type, vi_shape,
};

const SCALE_BASE: u32 = 2;
const SCALE_EXPONENT: u32 = 18;
const ALPHA: f64 = 262144.0;
const WEIGHT_SCALE: f64 = 262144.0;
const BIAS_SCALE: f64 = 68719476736.0;
const MIN_N_BITS: usize = 16;

pub fn prepare_jstprove_artifacts(
    onnx_path: &Path,
    weights_as_inputs: bool,
) -> Result<(CircuitParams, Architecture, WANDB)> {
    let model = onnx_proto::load_model(onnx_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Pipeline(format!("no graph in {}", onnx_path.display())))?;

    let init_map = build_initializer_map(graph);
    let opset_version = model.opset_import.first().map_or(13, |o| o.version) as i16;

    let init_names: HashSet<&str> = init_map.keys().map(|k| k.as_str()).collect();

    let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();

    for vi in &graph.input {
        let s = vi_shape(vi);
        shape_map.insert(vi.name.clone(), s.iter().map(|&d| d as usize).collect());
    }

    for init in &graph.initializer {
        shape_map.insert(
            init.name.clone(),
            init.dims.iter().map(|&d| d as usize).collect(),
        );
    }

    for vi in &graph.value_info {
        let s = vi_shape(vi);
        shape_map.insert(vi.name.clone(), s.iter().map(|&d| d as usize).collect());
    }

    for vi in &graph.output {
        let s = vi_shape(vi);
        if !s.is_empty() {
            shape_map.insert(vi.name.clone(), s.iter().map(|&d| d as usize).collect());
        }
    }

    let mut arch_layers: Vec<ONNXLayer> = Vec::new();
    let mut wandb_layers: Vec<ONNXLayer> = Vec::new();
    let mut n_bits_config: HashMap<String, usize> = HashMap::new();
    let mut wandb_id = 0usize;
    let mut input_max = 1.0f64;

    for (idx, node) in graph.node.iter().enumerate() {
        if node.op_type != "Conv" {
            continue;
        }

        let node_name = if node.name.is_empty() {
            node.output
                .first()
                .cloned()
                .unwrap_or_else(|| format!("conv_{idx}"))
        } else {
            node.name.clone()
        };

        let input_name = node
            .input
            .first()
            .ok_or_else(|| DsperseError::Pipeline(format!("Conv node {node_name} has no input")))?;
        let input_shape = shape_map.get(input_name).ok_or_else(|| {
            DsperseError::Pipeline(format!("missing shape for Conv input {input_name}"))
        })?;

        let weight_name = node.input.get(1).ok_or_else(|| {
            DsperseError::Pipeline(format!("Conv node {node_name} has no weight input"))
        })?;
        let weight_tensor = init_map.get(weight_name).ok_or_else(|| {
            DsperseError::Pipeline(format!("weight initializer {weight_name} not found"))
        })?;
        let weight_shape: Vec<usize> = weight_tensor.dims.iter().map(|&d| d as usize).collect();
        let weight_floats = tensor_to_f32(weight_tensor);

        let (bias_name, bias_floats, bias_shape) =
            resolve_bias(node, &node_name, &init_map, weight_shape[0])?;

        let kernel_shape = get_attribute_ints(node, "kernel_shape")
            .unwrap_or_else(|| vec![weight_shape[2] as i64, weight_shape[3] as i64]);
        let strides = get_attribute_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_attribute_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let dilations = get_attribute_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
        let group = get_attribute_int(node, "group").unwrap_or(1);

        let (n, h_in, w_in) = parse_spatial_dims(input_shape);
        let c_out = weight_shape[0];

        let h_out = (h_in as i64 + pads[0] + pads[2] - dilations[0] * (kernel_shape[0] - 1) - 1)
            / strides[0]
            + 1;
        let w_out = (w_in as i64 + pads[1] + pads[3] - dilations[1] * (kernel_shape[1] - 1) - 1)
            / strides[1]
            + 1;

        let output_name = node.output.first().ok_or_else(|| {
            DsperseError::Pipeline(format!("Conv node {node_name} has no output"))
        })?;
        let output_shape = vec![n, c_out, h_out as usize, w_out as usize];
        shape_map.insert(output_name.clone(), output_shape.clone());

        let mut node_inputs = node.input.clone();
        if node_inputs.len() == 2 {
            node_inputs.push(bias_name.clone());
        } else if node_inputs.len() == 3 && node_inputs[2].is_empty() {
            node_inputs[2] = bias_name.clone();
        }

        arch_layers.push(ONNXLayer {
            id: idx,
            name: node_name.clone(),
            op_type: "Conv".to_string(),
            inputs: node_inputs,
            outputs: node.output.clone(),
            shape: HashMap::from([(output_name.clone(), output_shape)]),
            tensor: None,
            params: Some(json!({
                "kernel_shape": kernel_shape,
                "strides": strides,
                "pads": pads,
                "dilations": dilations,
                "group": group,
            })),
            opset_version_number: opset_version,
        });

        let quantized_weights: Vec<i64> = weight_floats
            .iter()
            .map(|&v| (v as f64 * WEIGHT_SCALE) as i64)
            .collect();
        wandb_layers.push(ONNXLayer {
            id: wandb_id,
            name: weight_name.clone(),
            op_type: "Const".to_string(),
            inputs: vec![],
            outputs: vec![],
            shape: HashMap::from([(weight_name.clone(), weight_shape.clone())]),
            tensor: Some(build_nested_array(&quantized_weights, &weight_shape)),
            params: None,
            opset_version_number: opset_version,
        });
        wandb_id += 1;

        let quantized_bias: Vec<i64> = bias_floats
            .iter()
            .map(|&v| (v as f64 * BIAS_SCALE) as i64)
            .collect();
        wandb_layers.push(ONNXLayer {
            id: wandb_id,
            name: bias_name.clone(),
            op_type: "Const".to_string(),
            inputs: vec![],
            outputs: vec![],
            shape: HashMap::from([(bias_name, bias_shape.clone())]),
            tensor: Some(build_nested_array(&quantized_bias, &bias_shape)),
            params: None,
            opset_version_number: opset_version,
        });
        wandb_id += 1;

        let out_channels = weight_shape[0];
        let weights_per_filter: usize = weight_shape[1..].iter().product();

        let mut real_out_max = 0.0f64;
        for c in 0..out_channels {
            let start = c * weights_per_filter;
            let end = start + weights_per_filter;
            let l1_norm: f64 = weight_floats[start..end]
                .iter()
                .map(|&w| (w as f64).abs())
                .sum();
            let bias_abs = (bias_floats.get(c).copied().unwrap_or(0.0) as f64).abs();
            let out_max_c = l1_norm * input_max + bias_abs;
            if out_max_c > real_out_max {
                real_out_max = out_max_c;
            }
        }

        let n_bits = if real_out_max > 0.0 {
            let val = ALPHA * real_out_max + 1.0;
            ((val.log2().ceil() as usize) + 1).max(MIN_N_BITS)
        } else {
            MIN_N_BITS
        };

        n_bits_config.insert(node_name, n_bits);
        input_max = real_out_max;
    }

    let mut inputs: Vec<ONNXIO> = Vec::new();
    for vi in &graph.input {
        if !init_names.contains(vi.name.as_str()) {
            let s = vi_shape(vi);
            inputs.push(ONNXIO {
                name: vi.name.clone(),
                elem_type: vi_elem_type(vi) as i16,
                shape: s.iter().map(|&d| d as usize).collect(),
            });
        }
    }

    if weights_as_inputs {
        for init in &graph.initializer {
            inputs.push(ONNXIO {
                name: init.name.clone(),
                elem_type: init.data_type as i16,
                shape: init.dims.iter().map(|&d| d as usize).collect(),
            });
        }
    }

    let outputs: Vec<ONNXIO> = graph
        .output
        .iter()
        .map(|vi| {
            let s = vi_shape(vi);
            ONNXIO {
                name: vi.name.clone(),
                elem_type: vi_elem_type(vi) as i16,
                shape: s.iter().map(|&d| d as usize).collect(),
            }
        })
        .collect();

    let params = CircuitParams {
        scale_base: SCALE_BASE,
        scale_exponent: SCALE_EXPONENT,
        rescale_config: HashMap::new(),
        inputs,
        outputs,
        freivalds_reps: 1,
        n_bits_config,
        weights_as_inputs,
    };

    Ok((
        params,
        Architecture {
            architecture: arch_layers,
        },
        WANDB {
            w_and_b: wandb_layers,
        },
    ))
}

fn resolve_bias(
    node: &onnx_proto::NodeProto,
    node_name: &str,
    init_map: &HashMap<String, &onnx_proto::TensorProto>,
    out_channels: usize,
) -> Result<(String, Vec<f32>, Vec<usize>)> {
    match node.input.get(2) {
        Some(bname) if !bname.is_empty() => {
            let bt = init_map.get(bname).ok_or_else(|| {
                DsperseError::Pipeline(format!("bias initializer {bname} not found"))
            })?;
            let bs: Vec<usize> = bt.dims.iter().map(|&d| d as usize).collect();
            Ok((bname.clone(), tensor_to_f32(bt), bs))
        }
        _ => Ok((
            format!("{node_name}_zero_bias"),
            vec![0.0f32; out_channels],
            vec![out_channels],
        )),
    }
}

fn parse_spatial_dims(shape: &[usize]) -> (usize, usize, usize) {
    match shape.len() {
        4 => (shape[0], shape[2], shape[3]),
        3 => (1, shape[1], shape[2]),
        _ => (1, 1, 1),
    }
}

fn build_nested_array(flat: &[i64], shape: &[usize]) -> serde_json::Value {
    if shape.len() <= 1 {
        return serde_json::Value::Array(
            flat.iter().map(|&v| serde_json::Value::from(v)).collect(),
        );
    }

    let inner_size: usize = shape[1..].iter().product();
    serde_json::Value::Array(
        flat.chunks(inner_size)
            .map(|chunk| build_nested_array(chunk, &shape[1..]))
            .collect(),
    )
}
