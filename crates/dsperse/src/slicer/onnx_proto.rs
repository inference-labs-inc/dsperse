#[allow(clippy::doc_overindented_list_items)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use std::collections::HashMap;
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

pub fn save_model(model: &ModelProto, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let bytes = model.encode_to_vec();
    std::fs::write(path, bytes).map_err(|e| DsperseError::io(e, path))
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

pub fn tensor_to_f32(tensor: &TensorProto) -> Vec<f32> {
    if !tensor.float_data.is_empty() {
        return tensor.float_data.clone();
    }
    if !tensor.raw_data.is_empty() && tensor.data_type == TensorProto::FLOAT {
        return tensor
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
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
                return Err(DsperseError::Pipeline(format!(
                    "shape mismatch for initializer '{}' in {context}: slice expects {:?}, consumer provides {:?}",
                    init.name, init.dims, donor.dims
                )));
            }
        } else {
            return Err(DsperseError::Pipeline(format!(
                "consumer weights missing initializer '{}' required by {context}",
                init.name
            )));
        }
    }
    Ok(())
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
                    "dtype mismatch for initializer '{}': slice has dtype {}, consumer has dtype {}",
                    init.name, init.data_type, donor.data_type
                )));
            }
            if init.dims != donor.dims {
                return Err(DsperseError::Pipeline(format!(
                    "shape mismatch for initializer '{}': slice expects {:?}, consumer provides {:?}",
                    init.name, init.dims, donor.dims
                )));
            }
            init.float_data = donor.float_data.clone();
            init.raw_data = donor.raw_data.clone();
            init.double_data = donor.double_data.clone();
            init.int32_data = donor.int32_data.clone();
            init.int64_data = donor.int64_data.clone();
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
