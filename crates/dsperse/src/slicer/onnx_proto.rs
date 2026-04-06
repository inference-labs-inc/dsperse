#[allow(clippy::doc_overindented_list_items)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use std::collections::{HashMap, HashSet};
use std::path::Path;

use prost::Message;

use crate::error::{DsperseError, Result};

pub use onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TypeProto,
    ValueInfoProto,
};

pub fn load_model(path: &Path) -> Result<ModelProto> {
    let bytes = crate::utils::limits::read_checked(path)?;
    ModelProto::decode(bytes.as_slice())
        .map_err(|e| DsperseError::Slicer(format!("decode {}: {e}", path.display())))
}

fn canonicalize_node_attributes(nodes: &mut [NodeProto]) {
    for node in nodes {
        node.attribute.sort_by(|a, b| a.name.cmp(&b.name));
        for attr in &mut node.attribute {
            if let Some(g) = attr.g.as_mut() {
                canonicalize_node_attributes(&mut g.node);
            }
            for g in &mut attr.graphs {
                canonicalize_node_attributes(&mut g.node);
            }
        }
    }
}

pub fn save_model(model: &ModelProto, path: &Path) -> Result<()> {
    let mut model = model.clone();
    if let Some(graph) = model.graph.as_mut() {
        canonicalize_node_attributes(&mut graph.node);
    }
    for func in &mut model.functions {
        canonicalize_node_attributes(&mut func.node);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let bytes = model.encode_to_vec();
    std::fs::write(path, bytes).map_err(|e| DsperseError::io(e, path))
}

pub fn shape_from_value_info(vi: &ValueInfoProto) -> Option<Vec<i64>> {
    let tp = vi.r#type.as_ref()?;
    let onnx::type_proto::Value::TensorType(tensor) = tp.value.as_ref()? else {
        return None;
    };
    let shape_proto = tensor.shape.as_ref()?;
    let mut dims = Vec::new();
    for d in &shape_proto.dim {
        match &d.value {
            Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => dims.push(*v),
            _ => return None,
        }
    }
    Some(dims)
}

pub fn elem_type_from_value_info(vi: &ValueInfoProto) -> Option<i32> {
    let tp = vi.r#type.as_ref()?;
    let onnx::type_proto::Value::TensorType(tensor) = tp.value.as_ref()? else {
        return None;
    };
    Some(tensor.elem_type)
}

pub fn make_tensor_value_info(name: &str, elem_type: i32, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type,
                    shape: Some(onnx::TensorShapeProto {
                        dim: shape
                            .iter()
                            .map(|&d| onnx::tensor_shape_proto::Dimension {
                                denotation: String::new(),
                                value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(
                                    d,
                                )),
                            })
                            .collect(),
                    }),
                },
            )),
        }),
        doc_string: String::new(),
        metadata_props: vec![],
    }
}

pub fn make_tensor(name: &str, elem_type: i32, dims: &[i64], float_data: Vec<f32>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: elem_type,
        dims: dims.to_vec(),
        float_data,
        ..Default::default()
    }
}

pub fn make_node(
    op_type: &str,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: inputs,
        output: outputs,
        attribute: attributes,
        name: String::new(),
        domain: String::new(),
        doc_string: String::new(),
        overload: String::new(),
        metadata_props: vec![],
        device_configurations: vec![],
    }
}

pub fn make_graph(
    name: &str,
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
) -> GraphProto {
    GraphProto {
        name: name.to_string(),
        node: nodes,
        input: inputs,
        output: outputs,
        initializer: initializers,
        ..Default::default()
    }
}

pub fn make_model(graph: GraphProto, opset_version: i64) -> ModelProto {
    ModelProto {
        ir_version: 8,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: opset_version,
        }],
        ..Default::default()
    }
}

pub fn make_attribute_ints(name: &str, ints: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Ints as i32,
        ints: ints.to_vec(),
        ..Default::default()
    }
}

pub fn make_attribute_int(name: &str, val: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        i: val,
        ..Default::default()
    }
}

pub fn get_attribute_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

pub fn get_attribute_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

pub fn vi_shape(vi: &ValueInfoProto) -> Vec<i64> {
    vi.r#type
        .as_ref()
        .and_then(|t| match &t.value {
            Some(onnx::type_proto::Value::TensorType(tt)) => tt.shape.as_ref(),
            _ => None,
        })
        .map(|s| {
            s.dim
                .iter()
                .map(|d| match &d.value {
                    Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                    _ => 0,
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn tensor_to_i64(tensor: &TensorProto) -> Vec<i64> {
    if !tensor.int64_data.is_empty() {
        return tensor.int64_data.clone();
    }
    if !tensor.raw_data.is_empty() && tensor.data_type == TensorProto::INT64 {
        return tensor
            .raw_data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
    }
    if !tensor.int32_data.is_empty() {
        return tensor.int32_data.iter().map(|&v| v as i64).collect();
    }
    Vec::new()
}

pub fn tensor_to_f32(tensor: &TensorProto) -> Vec<f32> {
    if !tensor.float_data.is_empty() {
        return tensor.float_data.clone();
    }
    if !tensor.raw_data.is_empty() && tensor.data_type == TensorProto::FLOAT {
        let chunks = tensor.raw_data.chunks_exact(4);
        if !chunks.remainder().is_empty() {
            return Vec::new();
        }
        return chunks
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
    }
    if !tensor.int64_data.is_empty() {
        return tensor.int64_data.iter().map(|&v| v as f32).collect();
    }
    if !tensor.int32_data.is_empty() {
        return tensor.int32_data.iter().map(|&v| v as f32).collect();
    }
    if !tensor.double_data.is_empty() {
        return tensor.double_data.iter().map(|&v| v as f32).collect();
    }
    if !tensor.raw_data.is_empty() {
        match tensor.data_type {
            TensorProto::INT64 => {
                let chunks = tensor.raw_data.chunks_exact(8);
                if !chunks.remainder().is_empty() {
                    return Vec::new();
                }
                return chunks
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect();
            }
            TensorProto::INT32 => {
                let chunks = tensor.raw_data.chunks_exact(4);
                if !chunks.remainder().is_empty() {
                    return Vec::new();
                }
                return chunks
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect();
            }
            TensorProto::DOUBLE => {
                let chunks = tensor.raw_data.chunks_exact(8);
                if !chunks.remainder().is_empty() {
                    return Vec::new();
                }
                return chunks
                    .map(|c| {
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect();
            }
            _ => {}
        }
    }
    Vec::new()
}

pub fn build_initializer_map(graph: &GraphProto) -> HashMap<String, &TensorProto> {
    graph
        .initializer
        .iter()
        .map(|i| (i.name.clone(), i))
        .collect()
}

pub fn build_value_info_map(graph: &GraphProto) -> HashMap<String, &ValueInfoProto> {
    let mut map: HashMap<String, &ValueInfoProto> = HashMap::new();
    for vi in &graph.input {
        map.insert(vi.name.clone(), vi);
    }
    for vi in &graph.output {
        map.insert(vi.name.clone(), vi);
    }
    for vi in &graph.value_info {
        map.insert(vi.name.clone(), vi);
    }
    map
}

impl TensorProto {
    pub const FLOAT: i32 = 1;
    pub const INT64: i32 = 7;
    pub const DOUBLE: i32 = 11;
    pub const INT32: i32 = 6;
    pub const FLOAT16: i32 = 10;
    pub const BOOL: i32 = 9;
}

fn is_paddable_shape(target: &[i64], donor: &[i64]) -> bool {
    if target.len() != donor.len() || target.is_empty() {
        return false;
    }
    let last = target.len() - 1;
    target[..last] == donor[..last] && donor[last] < target[last] && donor[last] > 0
}

pub fn validate_initializer_compatibility(
    initializers: &[TensorProto],
    donor_init_map: &HashMap<String, &TensorProto>,
    context: &str,
) -> Result<()> {
    for init in initializers {
        if let Some(donor) = donor_init_map.get(&init.name) {
            if init.data_type != donor.data_type {
                return Err(DsperseError::Pipeline(format!(
                    "dtype mismatch for initializer '{}' in {context}: slice has dtype {}, consumer has dtype {}",
                    init.name, init.data_type, donor.data_type
                )));
            }
            if init.dims != donor.dims {
                if is_paddable_shape(&init.dims, &donor.dims) {
                    tracing::info!(
                        name = %init.name,
                        target = ?init.dims,
                        donor = ?donor.dims,
                        "donor initializer will be zero-padded on last axis"
                    );
                } else {
                    return Err(DsperseError::Pipeline(format!(
                        "shape mismatch for initializer '{}' in {context}: slice expects {:?}, consumer provides {:?}",
                        init.name, init.dims, donor.dims
                    )));
                }
            }
        } else {
            tracing::debug!(
                name = %init.name,
                context,
                "initializer not in donor weights, retaining slice value"
            );
        }
    }
    Ok(())
}

fn pad_float_data(
    donor_data: &[f32],
    target_dims: &[i64],
    donor_dims: &[i64],
    pad_val: f32,
) -> Vec<f32> {
    let last = target_dims.len() - 1;
    let target_last = target_dims[last] as usize;
    let donor_last = donor_dims[last] as usize;
    let rows = donor_data.len() / donor_last.max(1);
    let mut padded = Vec::with_capacity(rows * target_last);
    for row in 0..rows {
        let start = row * donor_last;
        let end = start + donor_last;
        padded.extend_from_slice(&donor_data[start..end.min(donor_data.len())]);
        padded.resize(padded.len() + (target_last - donor_last), pad_val);
    }
    padded
}

fn pad_raw_data_f32(raw: &[u8], target_dims: &[i64], donor_dims: &[i64], pad_val: f32) -> Vec<u8> {
    let donor_floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let padded = pad_float_data(&donor_floats, target_dims, donor_dims, pad_val);
    padded.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn replace_initializers(
    model: &mut ModelProto,
    donor_init_map: &HashMap<String, &TensorProto>,
) -> Result<usize> {
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| DsperseError::Pipeline("ONNX model missing graph".into()))?;
    let mut replaced = 0;
    for init in &mut graph.initializer {
        if let Some(donor) = donor_init_map.get(&init.name) {
            if init.data_type != donor.data_type {
                return Err(DsperseError::Pipeline(format!(
                    "dtype mismatch for initializer '{}' in replace_initializers: slice has dtype {}, consumer has dtype {}",
                    init.name, init.data_type, donor.data_type
                )));
            }
            let needs_pad = init.dims != donor.dims && is_paddable_shape(&init.dims, &donor.dims);
            if init.dims != donor.dims && !needs_pad {
                return Err(DsperseError::Pipeline(format!(
                    "shape mismatch for initializer '{}' in replace_initializers: slice expects {:?}, consumer provides {:?}",
                    init.name, init.dims, donor.dims
                )));
            }
            if needs_pad {
                let is_bias = donor.dims.len() == 1;
                let pad_val: f32 = if is_bias { -10.0 } else { 0.0 };
                if !donor.float_data.is_empty() {
                    init.float_data =
                        pad_float_data(&donor.float_data, &init.dims, &donor.dims, pad_val);
                    init.raw_data.clear();
                } else if !donor.raw_data.is_empty() && donor.data_type == TensorProto::FLOAT {
                    init.raw_data =
                        pad_raw_data_f32(&donor.raw_data, &init.dims, &donor.dims, pad_val);
                    init.float_data.clear();
                }
                tracing::info!(
                    name = %init.name,
                    from = ?donor.dims,
                    to = ?init.dims,
                    "padded donor initializer"
                );
            } else {
                init.float_data = donor.float_data.clone();
                init.raw_data = donor.raw_data.clone();
                init.double_data = donor.double_data.clone();
                init.int32_data = donor.int32_data.clone();
                init.int64_data = donor.int64_data.clone();
            }
            replaced += 1;
        }
    }
    Ok(replaced)
}

pub fn build_patched_onnx(
    slice_onnx: &Path,
    donor_init_map: &HashMap<String, &TensorProto>,
) -> Result<tempfile::NamedTempFile> {
    let mut model = load_model(slice_onnx)?;
    replace_initializers(&mut model, donor_init_map)?;
    let tmp = tempfile::NamedTempFile::with_suffix(".onnx")
        .map_err(|e| DsperseError::Pipeline(format!("create temp file: {e}")))?;
    save_model(&model, tmp.path())?;
    Ok(tmp)
}

fn model_opset_version(model: &ModelProto) -> i64 {
    model
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .map(|o| o.version)
        .unwrap_or(1)
}

fn min_opset_for_op(op_type: &str) -> Option<i64> {
    match op_type {
        "GridSample" => Some(16),
        "ScatterND" => Some(16),
        "ScatterElements" => Some(16),
        _ => None,
    }
}

pub fn normalize_opset(model: &mut ModelProto) -> usize {
    let opset = model_opset_version(model);
    if opset < 13 {
        return 0;
    }
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };
    let mut required_opset = opset;
    for node in graph.node.iter() {
        if let Some(min) = min_opset_for_op(&node.op_type) {
            required_opset = required_opset.max(min);
        }
    }
    let mut new_initializers: Vec<TensorProto> = Vec::new();
    let mut count = 0;
    for node in &mut graph.node {
        match node.op_type.as_str() {
            "Unsqueeze" | "Squeeze" if node.input.len() == 1 => {
                if let Some(axes) = get_attribute_ints(node, "axes") {
                    let axes_name = format!("{}_axes_const", node.name);
                    new_initializers.push(TensorProto {
                        name: axes_name.clone(),
                        data_type: TensorProto::INT64,
                        dims: vec![axes.len() as i64],
                        int64_data: axes,
                        ..Default::default()
                    });
                    node.input.push(axes_name);
                    node.attribute.retain(|a| a.name != "axes");
                    count += 1;
                }
            }
            "Reshape" if opset < 14 => {
                let had = node.attribute.iter().any(|a| a.name == "allowzero");
                if had {
                    node.attribute.retain(|a| a.name != "allowzero");
                    count += 1;
                }
            }
            _ => {}
        }
    }
    graph.initializer.extend(new_initializers);
    if required_opset > opset {
        if let Some(entry) = model
            .opset_import
            .iter_mut()
            .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        {
            entry.version = required_opset;
        }
        tracing::info!(
            from = opset,
            to = required_opset,
            "bumped declared opset to match op requirements"
        );
        count += 1;
    }
    if count > 0 {
        tracing::info!(
            opset = required_opset,
            fixes = count,
            "normalized ONNX opset conventions"
        );
    }
    count
}

pub fn fold_constant_nodes(model: &mut ModelProto) -> std::collections::HashSet<String> {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return std::collections::HashSet::new(),
    };

    let mut folded_tensors: Vec<TensorProto> = Vec::new();
    let mut folded_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    for node in &graph.node {
        if node.op_type != "Constant" {
            continue;
        }
        let out_name = match node.output.first() {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let tensor = match node.attribute.iter().find(|a| a.name == "value") {
            Some(a) => match a.t.as_ref() {
                Some(t) => t,
                None => continue,
            },
            None => continue,
        };
        let mut t = tensor.clone();
        t.name = out_name.clone();
        folded_tensors.push(t);
        folded_names.insert(out_name.clone());
    }

    if folded_names.is_empty() {
        return folded_names;
    }

    graph
        .node
        .retain(|n| n.op_type != "Constant" || !n.output.iter().any(|o| folded_names.contains(o)));

    let count = folded_tensors.len();
    graph.initializer.extend(folded_tensors);

    tracing::info!(count, "folded Constant ops into initializers");
    folded_names
}

pub fn concretize_symbolic_dims(model: &mut ModelProto) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };

    let mut count = 0;
    let all_vi = graph.value_info.iter_mut().chain(graph.output.iter_mut());
    for vi in all_vi {
        let tp = match vi.r#type.as_mut() {
            Some(t) => t,
            None => continue,
        };
        let tensor = match &mut tp.value {
            Some(onnx::type_proto::Value::TensorType(tt)) => tt,
            _ => continue,
        };
        let shape = match tensor.shape.as_mut() {
            Some(s) => s,
            None => continue,
        };
        for d in &mut shape.dim {
            if matches!(
                &d.value,
                Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_))
            ) {
                d.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(0));
                count += 1;
            }
        }
    }

    if count > 0 {
        tracing::info!(
            count,
            "replaced symbolic dimension parameters with placeholder values"
        );
    }
    count
}

pub fn resolve_dynamic_input_shapes(
    model: &mut ModelProto,
    explicit_shape: Option<&[i64]>,
) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };
    let inferred_spatial = infer_spatial_from_graph(graph);
    let mut resolved = 0;
    for inp in &mut graph.input {
        let tp = match inp.r#type.as_mut() {
            Some(t) => t,
            None => continue,
        };
        let tensor = match &mut tp.value {
            Some(onnx::type_proto::Value::TensorType(tt)) => tt,
            _ => continue,
        };
        let shape = match tensor.shape.as_mut() {
            Some(s) => s,
            None => continue,
        };
        let has_symbolic = shape.dim.iter().any(|d| {
            matches!(
                &d.value,
                Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None
            )
        });
        if !has_symbolic {
            continue;
        }
        if let Some(explicit) = explicit_shape {
            if explicit.len() == shape.dim.len() {
                for (d, &v) in shape.dim.iter_mut().zip(explicit.iter()) {
                    d.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v));
                }
                tracing::info!(
                    input = %inp.name,
                    shape = ?explicit,
                    "applied explicit input shape"
                );
                resolved += 1;
                continue;
            }
            tracing::warn!(
                input = %inp.name,
                expected_rank = shape.dim.len(),
                provided_rank = explicit.len(),
                "explicit shape rank mismatch; falling back to heuristic resolution"
            );
        }
        let rank = shape.dim.len();
        for (i, d) in shape.dim.iter_mut().enumerate() {
            let is_symbolic = matches!(
                &d.value,
                Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None
            );
            if !is_symbolic {
                continue;
            }
            let dim_name = match &d.value {
                Some(onnx::tensor_shape_proto::dimension::Value::DimParam(s)) => s.clone(),
                _ => String::new(),
            };
            let inferred = if i == 0 {
                1
            } else if rank == 4
                && (i == 2 || i == 3)
                && let Some(spatial) = inferred_spatial
            {
                spatial
            } else {
                tracing::warn!(
                    input = %inp.name,
                    dim = i,
                    dim_name = %dim_name,
                    "could not infer symbolic dimension; defaulting to 1"
                );
                1
            };
            d.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(
                inferred,
            ));
        }
        let final_shape: Vec<i64> = shape
            .dim
            .iter()
            .filter_map(|d| match &d.value {
                Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => Some(*v),
                _ => None,
            })
            .collect();
        tracing::info!(
            input = %inp.name,
            shape = ?final_shape,
            "resolved dynamic input dimensions"
        );
        resolved += 1;
    }
    resolved
}

fn infer_spatial_from_graph(graph: &GraphProto) -> Option<i64> {
    let init_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|i| i.name.as_str()).collect();

    for node in &graph.node {
        if node.op_type != "Conv" {
            continue;
        }
        let weight_name = match node.input.get(1) {
            Some(n) if init_names.contains(n.as_str()) => n,
            _ => continue,
        };
        let weight = match graph.initializer.iter().find(|t| &t.name == weight_name) {
            Some(t) => t,
            None => continue,
        };
        if weight.dims.len() == 4 {
            let kernel_h = weight.dims[2];
            let kernel_w = weight.dims[3];
            let strides = get_attribute_ints(node, "strides").unwrap_or_default();
            let stride = strides.first().copied().unwrap_or(1);
            let spatial = if stride > 1 {
                kernel_h.max(kernel_w) * stride * 14
            } else {
                kernel_h.max(kernel_w) * 224
            };
            tracing::info!(
                kernel = ?[kernel_h, kernel_w],
                stride,
                inferred_spatial = spatial,
                "inferred spatial dimensions from first Conv"
            );
            return Some(spatial);
        }
    }
    None
}

pub fn normalize_for_circuit_backend(model: &mut ModelProto) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };
    let count = flatten_matmul_inputs(graph) + materialize_reshape_targets(graph);
    if count > 0 {
        tracing::info!(count, "normalized graph for circuit backend compatibility");
    }
    count
}

fn flatten_matmul_inputs(graph: &mut GraphProto) -> usize {
    let vi_shapes: HashMap<String, Vec<i64>> = graph
        .input
        .iter()
        .chain(graph.value_info.iter())
        .chain(graph.output.iter())
        .filter_map(|vi| shape_from_value_info(vi).map(|s| (vi.name.clone(), s)))
        .collect();

    let init_shapes: HashMap<String, Vec<i64>> = graph
        .initializer
        .iter()
        .map(|i| (i.name.clone(), i.dims.clone()))
        .collect();

    let shapes: HashMap<String, Vec<i64>> = vi_shapes.into_iter().chain(init_shapes).collect();

    let mut new_nodes: Vec<(usize, Vec<NodeProto>)> = Vec::new();
    let mut new_inits: Vec<TensorProto> = Vec::new();
    let mut new_vis: Vec<ValueInfoProto> = Vec::new();
    let mut count = 0;

    for (idx, node) in graph.node.iter().enumerate() {
        if node.op_type != "MatMul" {
            continue;
        }
        let a_name = match node.input.first() {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let b_name = match node.input.get(1) {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let a_shape = match shapes.get(a_name) {
            Some(s) if s.len() > 2 => s.clone(),
            _ => continue,
        };
        let b_shape = match shapes.get(b_name) {
            Some(s) => s.clone(),
            None => continue,
        };
        let out_name = match node.output.first() {
            Some(n) if !n.is_empty() => n.clone(),
            _ => continue,
        };

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let batch_vol: i64 = batch_dims.iter().product();
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];

        let a_2d_name = format!("{a_name}__flat2d");
        let a_2d_shape_name = format!("{a_name}__flat2d_shape");
        let a_2d = vec![batch_vol * m, k];

        let mut b_2d_name = b_name.clone();
        let mut needs_b_reshape = false;
        let n_dim;
        if b_shape.len() > 2 {
            let b_m = b_shape[b_shape.len() - 2];
            n_dim = b_shape[b_shape.len() - 1];
            b_2d_name = format!("{b_name}__flat2d");
            let b_2d_shape_name = format!("{b_name}__flat2d_shape");
            let b_batch: i64 = b_shape[..b_shape.len() - 2].iter().product();
            let b_2d = vec![b_batch * b_m, n_dim];
            new_inits.push(TensorProto {
                name: b_2d_shape_name.clone(),
                data_type: TensorProto::INT64,
                dims: vec![2],
                int64_data: b_2d.clone(),
                ..Default::default()
            });
            new_vis.push(make_tensor_value_info(&b_2d_name, 1, &b_2d));
            needs_b_reshape = true;
        } else {
            n_dim = *b_shape.last().unwrap_or(&1);
        }

        let matmul_out_name = format!("{out_name}__matmul2d");
        let matmul_2d_shape = vec![batch_vol * m, n_dim];

        let restore_shape_name = format!("{out_name}__restore_shape");
        let mut restored: Vec<i64> = batch_dims.to_vec();
        restored.push(m);
        restored.push(n_dim);

        new_inits.push(TensorProto {
            name: a_2d_shape_name.clone(),
            data_type: TensorProto::INT64,
            dims: vec![2],
            int64_data: a_2d.clone(),
            ..Default::default()
        });
        new_inits.push(TensorProto {
            name: restore_shape_name.clone(),
            data_type: TensorProto::INT64,
            dims: vec![restored.len() as i64],
            int64_data: restored.clone(),
            ..Default::default()
        });

        new_vis.push(make_tensor_value_info(&a_2d_name, 1, &a_2d));
        new_vis.push(make_tensor_value_info(
            &matmul_out_name,
            1,
            &matmul_2d_shape,
        ));

        let mut inserted = Vec::new();

        inserted.push(NodeProto {
            op_type: "Reshape".into(),
            name: format!("{}_flatten_a", node.name),
            input: vec![a_name.clone(), a_2d_shape_name],
            output: vec![a_2d_name.clone()],
            ..Default::default()
        });

        if needs_b_reshape {
            let b_2d_shape_name = format!("{b_name}__flat2d_shape");
            inserted.push(NodeProto {
                op_type: "Reshape".into(),
                name: format!("{}_flatten_b", node.name),
                input: vec![b_name.clone(), b_2d_shape_name],
                output: vec![b_2d_name.clone()],
                ..Default::default()
            });
        }

        inserted.push(NodeProto {
            op_type: "MatMul".into(),
            name: node.name.clone(),
            input: vec![a_2d_name, b_2d_name],
            output: vec![matmul_out_name.clone()],
            attribute: node.attribute.clone(),
            ..Default::default()
        });

        inserted.push(NodeProto {
            op_type: "Reshape".into(),
            name: format!("{}_restore", node.name),
            input: vec![matmul_out_name, restore_shape_name],
            output: vec![out_name],
            ..Default::default()
        });

        new_nodes.push((idx, inserted));
        count += 1;
    }

    for (offset, (idx, nodes)) in new_nodes.into_iter().enumerate() {
        let pos = idx + offset * 2;
        graph.node.remove(pos);
        for (i, n) in nodes.into_iter().enumerate() {
            graph.node.insert(pos + i, n);
        }
    }
    graph.initializer.extend(new_inits);
    graph.value_info.extend(new_vis);
    count
}

fn materialize_reshape_targets(graph: &mut GraphProto) -> usize {
    let init_names: HashSet<String> = graph.initializer.iter().map(|i| i.name.clone()).collect();
    let input_names: HashSet<String> = graph.input.iter().map(|i| i.name.clone()).collect();

    let vi_shapes: HashMap<String, Vec<i64>> = graph
        .value_info
        .iter()
        .chain(graph.output.iter())
        .filter_map(|vi| shape_from_value_info(vi).map(|s| (vi.name.clone(), s)))
        .collect();

    let mut new_inits: Vec<TensorProto> = Vec::new();
    let mut count = 0;

    for node in &graph.node {
        if node.op_type != "Reshape" {
            continue;
        }
        let shape_input = match node.input.get(1) {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        if init_names.contains(shape_input) || input_names.contains(shape_input) {
            continue;
        }
        let out_name = match node.output.first() {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let out_shape = match vi_shapes.get(out_name) {
            Some(s) if !s.is_empty() && s.iter().all(|&d| d > 0) => s,
            _ => continue,
        };
        new_inits.push(TensorProto {
            name: shape_input.clone(),
            data_type: TensorProto::INT64,
            dims: vec![out_shape.len() as i64],
            int64_data: out_shape.clone(),
            ..Default::default()
        });
        count += 1;
    }

    graph.initializer.extend(new_inits);
    count
}

pub fn normalize_resize_modes(model: &mut ModelProto) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };
    let mut count = 0;
    for node in &mut graph.node {
        if node.op_type != "Resize" {
            continue;
        }
        let is_cubic = node
            .attribute
            .iter()
            .any(|a| a.name == "mode" && a.s == b"cubic");
        if is_cubic {
            if let Some(attr) = node.attribute.iter_mut().find(|a| a.name == "mode") {
                attr.s = b"linear".to_vec();
            }
            node.attribute.retain(|a| a.name != "cubic_coeff_a");
            tracing::info!(
                node = %node.name,
                "downgraded Resize interpolation from cubic to linear"
            );
            count += 1;
        }
    }
    count
}
