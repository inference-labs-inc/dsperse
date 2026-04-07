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
        if !tensor.raw_data.len().is_multiple_of(8) {
            tracing::warn!(
                tensor = %tensor.name,
                raw_len = tensor.raw_data.len(),
                "misaligned INT64 raw_data, skipping"
            );
            return Vec::new();
        }
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

    let propagated_names = propagate_constants(graph);
    if !propagated_names.is_empty() {
        tracing::info!(
            propagated = propagated_names.len(),
            "propagated constants after Constant-node folding"
        );
    }
    folded_names.extend(propagated_names);

    folded_names
}

pub fn strip_symbolic_value_info(model: &mut ModelProto) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };

    let has_symbolic = |vi: &ValueInfoProto| -> bool {
        vi.r#type
            .as_ref()
            .and_then(|t| match &t.value {
                Some(onnx::type_proto::Value::TensorType(tt)) => tt.shape.as_ref(),
                _ => None,
            })
            .is_some_and(|s| {
                s.dim.iter().any(|d| {
                    matches!(
                        &d.value,
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_))
                    )
                })
            })
    };

    let before = graph.value_info.len();
    graph.value_info.retain(|vi| !has_symbolic(vi));
    let removed = before - graph.value_info.len();

    for out in &mut graph.output {
        if let Some(ref mut tp) = out.r#type
            && let Some(onnx::type_proto::Value::TensorType(ref mut tt)) = tp.value
            && let Some(ref mut shape) = tt.shape
        {
            for d in &mut shape.dim {
                if matches!(
                    &d.value,
                    Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_))
                ) {
                    d.value = None;
                }
            }
        }
    }

    if removed > 0 {
        tracing::info!(
            removed,
            "stripped value_info entries with symbolic dimensions"
        );
    }
    removed
}

pub fn resolve_dynamic_input_shapes(
    model: &mut ModelProto,
    explicit_shape: Option<&[i64]>,
) -> crate::error::Result<usize> {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return Ok(0),
    };
    let symbolic_count = graph
        .input
        .iter()
        .filter(|inp| {
            inp.r#type
                .as_ref()
                .and_then(|t| match &t.value {
                    Some(onnx::type_proto::Value::TensorType(tt)) => tt.shape.as_ref(),
                    _ => None,
                })
                .is_some_and(|s| {
                    s.dim.iter().any(|d| {
                        matches!(
                            &d.value,
                            Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None
                        )
                    })
                })
        })
        .count();
    if symbolic_count > 1 && explicit_shape.is_some() {
        return Err(crate::error::DsperseError::Slicer(format!(
            "model has {symbolic_count} inputs with dynamic dimensions; \
             --input-shape applies to a single input. Per-input shapes not yet supported."
        )));
    }

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
            if explicit.len() != shape.dim.len() {
                return Err(crate::error::DsperseError::Slicer(format!(
                    "input '{}' has rank {} but --input-shape provides {} dims",
                    inp.name,
                    shape.dim.len(),
                    explicit.len()
                )));
            }
            for (d, &v) in shape.dim.iter_mut().zip(explicit.iter()) {
                if let Some(onnx::tensor_shape_proto::dimension::Value::DimValue(existing)) =
                    &d.value
                {
                    if *existing != v {
                        return Err(crate::error::DsperseError::Slicer(format!(
                            "input '{}': --input-shape dim {} conflicts with fixed dim {}",
                            inp.name, v, existing
                        )));
                    }
                } else {
                    d.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v));
                }
            }
            tracing::info!(input = %inp.name, shape = ?explicit, "applied explicit input shape");
            resolved += 1;
            continue;
        }
        let non_batch_symbolic = shape.dim.iter().skip(1).any(|d| {
            matches!(
                &d.value,
                Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None
            )
        });
        if non_batch_symbolic {
            let dim_names: Vec<String> = shape
                .dim
                .iter()
                .map(|d| match &d.value {
                    Some(onnx::tensor_shape_proto::dimension::Value::DimParam(s)) => s.clone(),
                    Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => v.to_string(),
                    None => "?".into(),
                })
                .collect();
            return Err(crate::error::DsperseError::Slicer(format!(
                "model input '{}' has dynamic dimensions [{}]; provide --input-shape to set concrete values",
                inp.name,
                dim_names.join(", ")
            )));
        }
        shape.dim[0].value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(1));
        tracing::info!(input = %inp.name, "defaulted batch dimension to 1");
        resolved += 1;
    }
    Ok(resolved)
}

pub fn normalize_for_circuit_backend(model: &mut ModelProto) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };
    let folded_names = propagate_constants(graph);
    let folded = folded_names.len();
    let fixed = fix_zero_dims(graph);
    let count = flatten_matmul_inputs(graph) + materialize_reshape_targets(graph) + fixed;
    let total = folded + count;
    if total > 0 {
        tracing::info!(
            total,
            folded,
            "normalized graph for circuit backend compatibility"
        );
    }
    total
}

pub fn propagate_constants_with_shapes(
    graph: &mut GraphProto,
    traced_shapes: &HashMap<String, Vec<i64>>,
) -> usize {
    for node in &graph.node {
        if node.op_type == "Shape"
            && let Some(inp_name) = node.input.first()
            && let Some(full_shape) = traced_shapes.get(inp_name)
            && let Some(out_name) = node.output.first()
            && !out_name.is_empty()
            && !graph.initializer.iter().any(|i| i.name == *out_name)
        {
            let ndim = full_shape.len() as i64;
            let start_attr = node
                .attribute
                .iter()
                .find(|a| a.name == "start")
                .map(|a| a.i)
                .unwrap_or(0);
            let end_attr = node
                .attribute
                .iter()
                .find(|a| a.name == "end")
                .map(|a| a.i)
                .unwrap_or(ndim);
            let start = if start_attr < 0 {
                (ndim + start_attr).max(0) as usize
            } else {
                (start_attr as usize).min(full_shape.len())
            };
            let end = if end_attr < 0 {
                (ndim + end_attr).max(0) as usize
            } else {
                (end_attr as usize).min(full_shape.len())
            };
            let sliced: Vec<i64> = if start < end {
                full_shape[start..end].to_vec()
            } else {
                vec![]
            };
            graph.initializer.push(TensorProto {
                name: out_name.clone(),
                data_type: TensorProto::INT64,
                dims: vec![sliced.len() as i64],
                int64_data: sliced,
                ..Default::default()
            });
        }
    }
    graph.node.retain(|n| {
        n.op_type != "Shape" || !graph.initializer.iter().any(|i| n.output.contains(&i.name))
    });
    let folded = propagate_constants(graph);
    folded.len()
}

fn propagate_constants(graph: &mut GraphProto) -> HashSet<String> {
    let mut constants: HashMap<String, TensorProto> = graph
        .initializer
        .iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();

    let mut folded_node_indices: HashSet<usize> = HashSet::new();

    loop {
        let mut progress = false;
        for (idx, node) in graph.node.iter().enumerate() {
            if folded_node_indices.contains(&idx) {
                continue;
            }
            let inputs: Vec<&str> = node
                .input
                .iter()
                .filter(|s| !s.is_empty())
                .map(String::as_str)
                .collect();
            if inputs.is_empty() {
                continue;
            }
            if !inputs.iter().all(|name| constants.contains_key(*name)) {
                continue;
            }
            let input_tensors: Vec<&TensorProto> = inputs.iter().map(|n| &constants[*n]).collect();
            if let Some(outputs) = eval_const_node(node, &input_tensors) {
                for (out_name, tensor) in outputs {
                    constants.insert(out_name, tensor);
                }
                folded_node_indices.insert(idx);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }

    if folded_node_indices.is_empty() {
        return HashSet::new();
    }

    let mut new_init_names: HashSet<String> = HashSet::new();
    for idx in &folded_node_indices {
        for out in &graph.node[*idx].output {
            if !out.is_empty() && constants.contains_key(out) {
                new_init_names.insert(out.clone());
            }
        }
    }

    let consumed_by_remaining: HashSet<String> = graph
        .node
        .iter()
        .enumerate()
        .filter(|(i, _)| !folded_node_indices.contains(i))
        .flat_map(|(_, n)| n.input.iter().cloned())
        .collect();
    let output_names: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    for name in &new_init_names {
        if (consumed_by_remaining.contains(name) || output_names.contains(name))
            && let Some(t) = constants.get(name)
            && !graph.initializer.iter().any(|i| i.name == *name)
        {
            graph.initializer.push(t.clone());
        }
    }

    let removed_outputs: HashSet<String> = folded_node_indices
        .iter()
        .flat_map(|idx| graph.node[*idx].output.iter().cloned())
        .collect();
    graph
        .input
        .retain(|vi| !removed_outputs.contains(&vi.name) || output_names.contains(&vi.name));

    let count = folded_node_indices.len();
    let mut kept = Vec::with_capacity(graph.node.len() - count);
    for (idx, node) in graph.node.drain(..).enumerate() {
        if !folded_node_indices.contains(&idx) {
            kept.push(node);
        }
    }
    graph.node = kept;

    tracing::info!(count, "propagated constant subgraphs into initializers");
    new_init_names
}

fn eval_const_node(
    node: &NodeProto,
    inputs: &[&TensorProto],
) -> Option<Vec<(String, TensorProto)>> {
    let out_name = node.output.first()?.clone();
    if out_name.is_empty() {
        return None;
    }
    match node.op_type.as_str() {
        "Identity" => {
            let mut t = inputs[0].clone();
            t.name = out_name.clone();
            Some(vec![(out_name, t)])
        }
        "Cast" => eval_cast(node, inputs[0], &out_name),
        "Sqrt" => eval_unary_f32(inputs[0], &out_name, f32::sqrt),
        "Neg" => eval_unary_f32(inputs[0], &out_name, |x| -x),
        "Abs" => eval_unary_f32(inputs[0], &out_name, f32::abs),
        "Exp" => eval_unary_f32(inputs[0], &out_name, f32::exp),
        "Log" => eval_unary_f32(inputs[0], &out_name, f32::ln),
        "Ceil" => eval_unary_f32(inputs[0], &out_name, f32::ceil),
        "Floor" => eval_unary_f32(inputs[0], &out_name, f32::floor),
        "Reciprocal" => eval_unary_f32(inputs[0], &out_name, |x| 1.0 / x),
        "Relu" => eval_unary_f32(inputs[0], &out_name, |x| x.max(0.0)),
        "Sigmoid" => eval_unary_f32(inputs[0], &out_name, |x| 1.0 / (1.0 + (-x).exp())),
        "Tanh" => eval_unary_f32(inputs[0], &out_name, f32::tanh),
        "Add" => eval_binary_f32(inputs, &out_name, |a, b| a + b),
        "Sub" => eval_binary_f32(inputs, &out_name, |a, b| a - b),
        "Mul" => eval_binary_f32(inputs, &out_name, |a, b| a * b),
        "Div" => eval_binary_f32(inputs, &out_name, |a, b| a / b),
        "Pow" => eval_binary_f32(inputs, &out_name, f32::powf),
        "Reshape" => eval_reshape(node, inputs, &out_name),
        "Squeeze" => eval_squeeze(node, inputs, &out_name),
        "Unsqueeze" => eval_unsqueeze(node, inputs, &out_name),
        "Shape" => eval_shape(node, inputs[0], &out_name),
        "Gather" if inputs.len() >= 2 => eval_gather(node, inputs, &out_name),
        "Slice" if inputs.len() >= 3 => eval_slice(inputs, &out_name),
        "Concat" => eval_concat(node, inputs, &out_name),
        _ => None,
    }
}

fn eval_cast(
    node: &NodeProto,
    input: &TensorProto,
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let target_type = node
        .attribute
        .iter()
        .find(|a| a.name == "to")
        .map(|a| a.i as i32)?;
    match target_type {
        TensorProto::INT64 => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT64,
                dims: input.dims.clone(),
                int64_data: vals.iter().map(|&v| v as i64).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::INT32 => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT32,
                dims: input.dims.clone(),
                int32_data: vals.iter().map(|&v| v as i32).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::FLOAT => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::FLOAT,
                dims: input.dims.clone(),
                float_data: vals,
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::DOUBLE => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::DOUBLE,
                dims: input.dims.clone(),
                double_data: vals.iter().map(|&v| v as f64).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::BOOL => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::BOOL,
                dims: input.dims.clone(),
                int32_data: vals.iter().map(|&v| (v != 0.0) as i32).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        _ => None,
    }
}

fn eval_unary_f32(
    input: &TensorProto,
    out_name: &str,
    f: fn(f32) -> f32,
) -> Option<Vec<(String, TensorProto)>> {
    let vals: Vec<f32> = tensor_to_f32(input).into_iter().map(f).collect();
    if vals.is_empty() {
        return None;
    }
    let t = make_f32_tensor(out_name, &input.dims, &vals, TensorProto::FLOAT);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_binary_f32(
    inputs: &[&TensorProto],
    out_name: &str,
    f: fn(f32, f32) -> f32,
) -> Option<Vec<(String, TensorProto)>> {
    if inputs.len() < 2 {
        return None;
    }
    let a = tensor_to_f32(inputs[0]);
    let b = tensor_to_f32(inputs[1]);
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (result, dims) = broadcast_binary(&a, &inputs[0].dims, &b, &inputs[1].dims, f)?;
    let t = make_f32_tensor(out_name, &dims, &result, TensorProto::FLOAT);
    Some(vec![(out_name.to_string(), t)])
}

fn broadcast_binary(
    a: &[f32],
    a_dims: &[i64],
    b: &[f32],
    b_dims: &[i64],
    f: fn(f32, f32) -> f32,
) -> Option<(Vec<f32>, Vec<i64>)> {
    if a_dims == b_dims {
        let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
        return Some((result, a_dims.to_vec()));
    }
    if a.len() == 1 {
        let result: Vec<f32> = b.iter().map(|&y| f(a[0], y)).collect();
        return Some((result, b_dims.to_vec()));
    }
    if b.len() == 1 {
        let result: Vec<f32> = a.iter().map(|&x| f(x, b[0])).collect();
        return Some((result, a_dims.to_vec()));
    }
    None
}

fn eval_reshape(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    if inputs.len() < 2 {
        return None;
    }
    let vals = tensor_to_f32(inputs[0]);
    let shape = tensor_to_i64(inputs[1]);
    if vals.is_empty() || shape.is_empty() {
        return None;
    }
    let allowzero = node
        .attribute
        .iter()
        .find(|a| a.name == "allowzero")
        .map(|a| a.i != 0)
        .unwrap_or(false);
    let mut new_dims: Vec<i64> = shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 {
                if allowzero {
                    0
                } else {
                    *inputs[0].dims.get(i).unwrap_or(&1)
                }
            } else {
                d
            }
        })
        .collect();
    if let Some(neg_idx) = new_dims.iter().position(|&d| d == -1) {
        let known: i64 = new_dims
            .iter()
            .enumerate()
            .filter(|&(i, &d)| i != neg_idx && d > 0)
            .map(|(_, &d)| d)
            .product();
        let total: i64 = vals.len() as i64;
        if known > 0 {
            new_dims[neg_idx] = total / known;
        }
    }
    let t = make_f32_tensor(out_name, &new_dims, &vals, inputs[0].data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_squeeze(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let input = inputs[0];
    let ndim = input.dims.len() as i64;
    let raw_axes: Vec<i64> = if inputs.len() >= 2 {
        tensor_to_i64(inputs[1])
    } else {
        node.attribute
            .iter()
            .find(|a| a.name == "axes")
            .map(|a| a.ints.clone())
            .unwrap_or_default()
    };
    let axes: Vec<usize> = raw_axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    if axes.is_empty() {
        let new_dims: Vec<i64> = input.dims.iter().copied().filter(|&d| d != 1).collect();
        let vals = tensor_to_f32(input);
        if vals.is_empty() {
            return None;
        }
        let t = make_f32_tensor(out_name, &new_dims, &vals, input.data_type);
        return Some(vec![(out_name.to_string(), t)]);
    }
    for &ax in &axes {
        if ax >= input.dims.len() || input.dims[ax] != 1 {
            return None;
        }
    }
    let new_dims: Vec<i64> = input
        .dims
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, &d)| d)
        .collect();
    let vals = tensor_to_f32(input);
    if vals.is_empty() {
        return None;
    }
    let t = make_f32_tensor(out_name, &new_dims, &vals, input.data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_unsqueeze(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let axes: Vec<i64> = if inputs.len() >= 2 {
        tensor_to_i64(inputs[1])
    } else {
        node.attribute
            .iter()
            .find(|a| a.name == "axes")
            .map(|a| a.ints.clone())
            .unwrap_or_default()
    };
    let ndim = inputs[0].dims.len() + axes.len();
    let mut new_dims = inputs[0].dims.clone();
    let mut sorted_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    sorted_axes.sort();
    for &ax in &sorted_axes {
        if ax <= new_dims.len() {
            new_dims.insert(ax, 1);
        }
    }
    let vals = tensor_to_f32(inputs[0]);
    if vals.is_empty() {
        return None;
    }
    let t = make_f32_tensor(out_name, &new_dims, &vals, TensorProto::FLOAT);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_shape(
    node: &NodeProto,
    input: &TensorProto,
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let dims = &input.dims;
    if dims.is_empty() {
        return None;
    }
    let ndim = dims.len() as i64;
    let start_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "start")
        .map(|a| a.i)
        .unwrap_or(0);
    let end_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "end")
        .map(|a| a.i)
        .unwrap_or(ndim);
    let start = if start_attr < 0 {
        (ndim + start_attr).max(0) as usize
    } else {
        (start_attr as usize).min(dims.len())
    };
    let end = if end_attr < 0 {
        (ndim + end_attr).max(0) as usize
    } else {
        (end_attr as usize).min(dims.len())
    };
    let sliced: Vec<i64> = if start < end {
        dims[start..end].to_vec()
    } else {
        vec![]
    };
    let t = TensorProto {
        name: out_name.to_string(),
        data_type: TensorProto::INT64,
        dims: vec![sliced.len() as i64],
        int64_data: sliced,
        ..Default::default()
    };
    Some(vec![(out_name.to_string(), t)])
}

fn eval_gather(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let axis = node
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
        .unwrap_or(0);
    let data = inputs[0];
    let indices = tensor_to_i64(inputs[1]);
    if indices.is_empty() || data.dims.is_empty() {
        return None;
    }
    if data.dims.len() == 1 && axis == 0 {
        let data_vals = tensor_to_f32(data);
        if data_vals.is_empty() {
            let data_i64 = tensor_to_i64(data);
            if data_i64.is_empty() {
                return None;
            }
            let result: Vec<i64> = indices
                .iter()
                .map(|&i| {
                    let idx = if i < 0 {
                        (data.dims[0] + i) as usize
                    } else {
                        i as usize
                    };
                    data_i64.get(idx).copied().unwrap_or(0)
                })
                .collect();
            let out_dims = if inputs[1].dims.is_empty() {
                vec![]
            } else {
                inputs[1].dims.clone()
            };
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT64,
                dims: out_dims,
                int64_data: result,
                ..Default::default()
            };
            return Some(vec![(out_name.to_string(), t)]);
        }
        let result: Vec<f32> = indices
            .iter()
            .map(|&i| {
                let idx = if i < 0 {
                    (data.dims[0] + i) as usize
                } else {
                    i as usize
                };
                data_vals.get(idx).copied().unwrap_or(0.0)
            })
            .collect();
        let out_dims = if inputs[1].dims.is_empty() {
            vec![]
        } else {
            inputs[1].dims.clone()
        };
        let t = make_f32_tensor(out_name, &out_dims, &result, TensorProto::FLOAT);
        return Some(vec![(out_name.to_string(), t)]);
    }
    None
}

fn eval_slice(inputs: &[&TensorProto], out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let data = inputs[0];
    let starts = tensor_to_i64(inputs[1]);
    let ends = tensor_to_i64(inputs[2]);
    if starts.is_empty() || ends.is_empty() {
        return None;
    }
    let axes: Vec<i64> = if inputs.len() > 3 {
        tensor_to_i64(inputs[3])
    } else {
        (0..starts.len() as i64).collect()
    };
    let steps: Vec<i64> = if inputs.len() > 4 {
        tensor_to_i64(inputs[4])
    } else {
        vec![1; starts.len()]
    };
    if data.dims.len() == 1 && axes == [0] && steps.iter().all(|&s| s == 1) {
        let dim = data.dims[0];
        let start = if starts[0] < 0 {
            (dim + starts[0]).max(0) as usize
        } else {
            (starts[0] as usize).min(dim as usize)
        };
        let end = if ends[0] < 0 {
            (dim + ends[0]).max(0) as usize
        } else {
            (ends[0] as usize).min(dim as usize)
        };
        if start >= end {
            return None;
        }
        if data.data_type == TensorProto::INT64 {
            let vals = tensor_to_i64(data);
            let sliced: Vec<i64> = vals.get(start..end)?.to_vec();
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT64,
                dims: vec![(end - start) as i64],
                int64_data: sliced,
                ..Default::default()
            };
            return Some(vec![(out_name.to_string(), t)]);
        }
        let vals = tensor_to_f32(data);
        let sliced: Vec<f32> = vals.get(start..end)?.to_vec();
        let t = make_f32_tensor(out_name, &[(end - start) as i64], &sliced, data.data_type);
        return Some(vec![(out_name.to_string(), t)]);
    }
    None
}

fn eval_concat(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let _axis = node
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
        .unwrap_or(0);
    if inputs.is_empty() {
        return None;
    }
    let all_1d = inputs.iter().all(|t| t.dims.len() <= 1);
    if !all_1d {
        return None;
    }
    if inputs[0].data_type == TensorProto::INT64
        || inputs.iter().all(|t| !tensor_to_i64(t).is_empty())
    {
        let mut result = Vec::new();
        for t in inputs {
            result.extend(tensor_to_i64(t));
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: vec![result.len() as i64],
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let mut result = Vec::new();
    for t in inputs {
        let vals = tensor_to_f32(t);
        if vals.is_empty() {
            return None;
        }
        result.extend(vals);
    }
    let t = make_f32_tensor(
        out_name,
        &[result.len() as i64],
        &result,
        inputs[0].data_type,
    );
    Some(vec![(out_name.to_string(), t)])
}

fn make_f32_tensor(name: &str, dims: &[i64], vals: &[f32], target_type: i32) -> TensorProto {
    match target_type {
        TensorProto::INT64 => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::INT64,
            dims: dims.to_vec(),
            int64_data: vals.iter().map(|&v| v as i64).collect(),
            ..Default::default()
        },
        TensorProto::INT32 => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::INT32,
            dims: dims.to_vec(),
            int32_data: vals.iter().map(|&v| v as i32).collect(),
            ..Default::default()
        },
        TensorProto::DOUBLE => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::DOUBLE,
            dims: dims.to_vec(),
            double_data: vals.iter().map(|&v| v as f64).collect(),
            ..Default::default()
        },
        _ => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::FLOAT,
            dims: dims.to_vec(),
            float_data: vals.to_vec(),
            ..Default::default()
        },
    }
}

fn fix_zero_dims(graph: &mut GraphProto) -> usize {
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    for inp in &graph.input {
        if let Some(s) = shape_from_value_info(inp)
            && s.iter().all(|&d| d > 0)
        {
            shapes.insert(inp.name.clone(), s);
        }
    }
    for init in &graph.initializer {
        if !init.dims.is_empty() {
            shapes.insert(init.name.clone(), init.dims.clone());
        }
    }
    for vi in &graph.value_info {
        if let Some(s) = shape_from_value_info(vi)
            && s.iter().all(|&d| d > 0)
            && !shapes.contains_key(&vi.name)
        {
            shapes.insert(vi.name.clone(), s);
        }
    }

    let mut count = 0;
    for vi in graph.value_info.iter_mut().chain(graph.output.iter_mut()) {
        if let Some(new_shape) = shapes.get(&vi.name)
            && let Some(existing) = shape_from_value_info(vi)
            && existing.contains(&0)
        {
            set_vi_shape(vi, new_shape);
            count += 1;
        }
    }

    if count > 0 {
        tracing::info!(count, "resolved zero-valued placeholder dimensions");
    }
    count
}

fn set_vi_shape(vi: &mut ValueInfoProto, shape: &[i64]) {
    if let Some(ref mut tp) = vi.r#type
        && let Some(onnx::type_proto::Value::TensorType(ref mut tt)) = tp.value
    {
        tt.shape = Some(onnx::TensorShapeProto {
            dim: shape
                .iter()
                .map(|&d| onnx::tensor_shape_proto::Dimension {
                    denotation: String::new(),
                    value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(d)),
                })
                .collect(),
        });
    }
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

    let elem_types: HashMap<String, i32> = graph
        .input
        .iter()
        .chain(graph.value_info.iter())
        .chain(graph.output.iter())
        .filter_map(|vi| elem_type_from_value_info(vi).map(|t| (vi.name.clone(), t)))
        .chain(
            graph
                .initializer
                .iter()
                .map(|i| (i.name.clone(), i.data_type)),
        )
        .collect();

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
            Some(s) if s.len() > 3 => s.clone(),
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

        let node_tag = if node.name.is_empty() {
            format!("matmul_{idx}")
        } else {
            node.name.clone()
        };
        let a_2d_name = format!("{a_name}__flat2d_{node_tag}");
        let a_2d_shape_name = format!("{a_name}__flat2d_shape_{node_tag}");
        let a_2d = vec![batch_vol * m, k];

        let mut b_2d_name = b_name.clone();
        let mut needs_b_reshape = false;
        let n_dim;
        if b_shape.len() > 2 {
            let b_m = b_shape[b_shape.len() - 2];
            n_dim = b_shape[b_shape.len() - 1];
            let b_batch: i64 = b_shape[..b_shape.len() - 2].iter().product();
            if b_batch == 1 {
                b_2d_name = format!("{b_name}__flat2d_{node_tag}");
                let b_2d_shape_name = format!("{b_name}__flat2d_shape_{node_tag}");
                let b_2d = vec![b_batch * b_m, n_dim];
                new_inits.push(TensorProto {
                    name: b_2d_shape_name.clone(),
                    data_type: TensorProto::INT64,
                    dims: vec![2],
                    int64_data: b_2d.clone(),
                    ..Default::default()
                });
                let b_elem = elem_types
                    .get(b_name)
                    .copied()
                    .unwrap_or(TensorProto::FLOAT);
                new_vis.push(make_tensor_value_info(&b_2d_name, b_elem, &b_2d));
                needs_b_reshape = true;
            }
        } else {
            n_dim = *b_shape.last().unwrap_or(&1);
        }

        let matmul_out_name = format!("{out_name}__matmul2d_{node_tag}");
        let matmul_2d_shape = vec![batch_vol * m, n_dim];

        let restore_shape_name = format!("{out_name}__restore_shape_{node_tag}");
        let mut restored: Vec<i64> = batch_dims.to_vec();
        restored.push(m);
        if b_shape.len() > 1 {
            restored.push(n_dim);
        }

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

        let a_elem = elem_types
            .get(a_name)
            .copied()
            .unwrap_or(TensorProto::FLOAT);
        new_vis.push(make_tensor_value_info(&a_2d_name, a_elem, &a_2d));
        new_vis.push(make_tensor_value_info(
            &matmul_out_name,
            a_elem,
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
            let b_2d_shape_name = format!("{b_name}__flat2d_shape_{node_tag}");
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

    let mut cumulative_offset: usize = 0;
    for (idx, nodes) in new_nodes {
        let pos = idx + cumulative_offset;
        graph.node.remove(pos);
        let inserted = nodes.len();
        for (i, n) in nodes.into_iter().enumerate() {
            graph.node.insert(pos + i, n);
        }
        cumulative_offset += inserted - 1;
    }
    graph.initializer.extend(new_inits);
    graph.value_info.extend(new_vis);
    count
}

fn materialize_reshape_targets(graph: &mut GraphProto) -> usize {
    let mut init_names: HashSet<String> =
        graph.initializer.iter().map(|i| i.name.clone()).collect();
    let input_names: HashSet<String> = graph.input.iter().map(|i| i.name.clone()).collect();
    let produced_names: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|n| n.output.iter().cloned())
        .collect();

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
        if init_names.contains(shape_input)
            || input_names.contains(shape_input)
            || produced_names.contains(shape_input)
        {
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
        init_names.insert(shape_input.clone());
        count += 1;
    }

    graph.initializer.extend(new_inits);
    count
}
