use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::onnx_proto::{self, ModelProto, TensorProto, ValueInfoProto};
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;

pub fn materialize_combined_model(
    model: &ModelProto,
    metadata: &ModelMetadata,
    traced_shapes: &HashMap<String, Vec<i64>>,
    traced_types: Option<&HashMap<String, i32>>,
) -> Result<ModelProto> {
    let mut combined = model.clone();
    let graph = combined
        .graph
        .as_mut()
        .ok_or_else(|| DsperseError::Slicer("model.graph is None".into()))?;

    let existing_outputs: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    let all_node_outputs: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|n| n.output.iter().cloned())
        .collect();

    let mut new_outputs: Vec<ValueInfoProto> = Vec::new();
    let mut added: HashSet<String> = HashSet::new();

    {
        let vi_map = onnx_proto::build_value_info_map(graph);

        for slice in &metadata.slices {
            for output_name in &slice.dependencies.output {
                if existing_outputs.contains(output_name) || added.contains(output_name) {
                    continue;
                }
                if !all_node_outputs.contains(output_name) {
                    tracing::warn!(
                        tensor = %output_name,
                        slice = slice.index,
                        "slice output not produced by any node in original graph, skipping"
                    );
                    continue;
                }

                if let Some(vi) =
                    resolve_value_info(output_name, &vi_map, traced_shapes, traced_types)?
                {
                    new_outputs.push(vi);
                    added.insert(output_name.clone());
                }
            }

            for input_name in &slice.dependencies.filtered_inputs {
                if existing_outputs.contains(input_name) || added.contains(input_name) {
                    continue;
                }
                if !all_node_outputs.contains(input_name) {
                    tracing::debug!(
                        tensor = %input_name,
                        slice = slice.index,
                        "slice filtered_input not produced by any node in original graph, skipping"
                    );
                    continue;
                }

                if let Some(vi) =
                    resolve_value_info(input_name, &vi_map, traced_shapes, traced_types)?
                {
                    new_outputs.push(vi);
                    added.insert(input_name.clone());
                }
            }
        }
    }

    graph.output.extend(new_outputs);

    tracing::info!(
        intermediate_outputs = added.len(),
        total_outputs = graph.output.len(),
        "combined model with slice boundary outputs"
    );

    Ok(combined)
}

const ONNX_STRING_DATATYPE: i32 = 8;
const NON_NUMERIC_TENSOR_TYPES: &[i32] = &[ONNX_STRING_DATATYPE];

fn resolve_value_info(
    name: &str,
    vi_map: &HashMap<String, &ValueInfoProto>,
    traced_shapes: &HashMap<String, Vec<i64>>,
    traced_types: Option<&HashMap<String, i32>>,
) -> Result<Option<ValueInfoProto>> {
    if let Some(vi) = vi_map.get(name) {
        let elem_type = onnx_proto::elem_type_from_value_info(vi).unwrap_or(TensorProto::FLOAT);
        if NON_NUMERIC_TENSOR_TYPES.contains(&elem_type) {
            return Ok(None);
        }
        return Ok(Some((*vi).clone()));
    }

    let shape = traced_shapes.get(name).ok_or_else(|| {
        DsperseError::Slicer(format!(
            "no shape info for combined model output tensor '{name}'"
        ))
    })?;

    let elem_type = traced_types
        .and_then(|t| t.get(name).copied())
        .unwrap_or(TensorProto::FLOAT);

    if NON_NUMERIC_TENSOR_TYPES.contains(&elem_type) {
        return Ok(None);
    }

    Ok(Some(onnx_proto::make_tensor_value_info(
        name, elem_type, shape,
    )))
}

pub fn ensure_combined_materialized(
    slices_dir: &Path,
    metadata: &ModelMetadata,
) -> Result<PathBuf> {
    let output_path = slices_dir.join("combined.onnx");
    if output_path.exists() {
        return Ok(output_path);
    }
    materialize_combined_to_disk(slices_dir, metadata)
}

pub fn materialize_combined_to_disk(
    slices_dir: &Path,
    metadata: &ModelMetadata,
) -> Result<PathBuf> {
    let traced_shapes = metadata.traced_shapes.as_ref().ok_or_else(|| {
        DsperseError::Slicer("metadata missing traced_shapes for combined model".into())
    })?;
    let traced_types = metadata.traced_types.as_ref();
    let original_path = metadata.original_model_path.as_ref().ok_or_else(|| {
        DsperseError::Slicer("metadata missing original_model_path for combined model".into())
    })?;

    let model_path = if Path::new(original_path).is_absolute() {
        std::path::PathBuf::from(original_path)
    } else {
        slices_dir.join(original_path)
    };

    let mut model = onnx_proto::load_model(&model_path)?;
    onnx_proto::normalize_opset(&mut model);

    let combined = materialize_combined_model(&model, metadata, traced_shapes, traced_types)?;

    let dest = slices_dir.join("combined.onnx");
    onnx_proto::save_model(&combined, &dest)?;
    tracing::info!(path = %dest.display(), "materialized combined ONNX");

    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::metadata::{
        Dependencies, ModelMetadata, SliceMetadata, SliceShapeWrapper, TensorShape,
    };

    fn make_test_model(
        node_output_types: HashMap<String, i32>,
        traced_shapes: HashMap<String, Vec<i64>>,
    ) -> (ModelProto, ModelMetadata) {
        let graph = onnx_proto::GraphProto {
            node: vec![onnx_proto::NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["input".to_string()],
                output: vec![
                    "float_tensor".to_string(),
                    "bool_tensor".to_string(),
                    "string_tensor".to_string(),
                    "int_tensor".to_string(),
                ],
                ..Default::default()
            }],
            input: vec![onnx_proto::make_tensor_value_info(
                "input",
                TensorProto::FLOAT,
                &[1, 3, 8, 8],
            )],
            output: vec![onnx_proto::make_tensor_value_info(
                "model_output",
                TensorProto::FLOAT,
                &[1, 3, 8, 8],
            )],
            ..Default::default()
        };
        let model = onnx_proto::make_model(graph, 13);

        let metadata = ModelMetadata {
            slices: vec![SliceMetadata {
                index: 0,
                filename: "s0.onnx".to_string(),
                path: "s0.onnx".to_string(),
                relative_path: "s0.onnx".to_string(),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![],
                        output: vec![],
                    },
                },
                dependencies: Dependencies {
                    input: vec![],
                    filtered_inputs: vec![],
                    output: vec![
                        "float_tensor".to_string(),
                        "bool_tensor".to_string(),
                        "string_tensor".to_string(),
                        "int_tensor".to_string(),
                    ],
                },
                ..Default::default()
            }],
            traced_shapes: Some(traced_shapes.clone()),
            traced_types: Some(node_output_types),
            ..Default::default()
        };

        (model, metadata)
    }

    #[test]
    fn bool_outputs_included_in_combined_model() {
        let mut node_output_types = HashMap::new();
        node_output_types.insert("float_tensor".to_string(), TensorProto::FLOAT);
        node_output_types.insert("bool_tensor".to_string(), TensorProto::BOOL);
        node_output_types.insert("string_tensor".to_string(), ONNX_STRING_DATATYPE);
        node_output_types.insert("int_tensor".to_string(), TensorProto::INT64);

        let mut traced_shapes = HashMap::new();
        traced_shapes.insert("float_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("bool_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("string_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("int_tensor".to_string(), vec![1, 3, 8, 8]);

        let (model, metadata) = make_test_model(node_output_types, traced_shapes.clone());

        let traced_types = metadata.traced_types.as_ref();
        let combined =
            materialize_combined_model(&model, &metadata, &traced_shapes, traced_types).unwrap();

        let graph = combined.graph.as_ref().unwrap();

        let float_vi = graph.output.iter().find(|o| o.name == "float_tensor");
        assert!(float_vi.is_some());

        let bool_vi = graph.output.iter().find(|o| o.name == "bool_tensor");
        assert!(bool_vi.is_some());

        let string_vi = graph.output.iter().find(|o| o.name == "string_tensor");
        assert!(
            string_vi.is_none(),
            "string tensors should be excluded from combined outputs"
        );

        let int_vi = graph.output.iter().find(|o| o.name == "int_tensor");
        assert!(int_vi.is_some());
    }

    #[test]
    fn combined_model_has_intermediate_outputs() {
        let mut traced_shapes = HashMap::new();
        traced_shapes.insert("float_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("bool_tensor".to_string(), vec![1]);
        traced_shapes.insert("string_tensor".to_string(), vec![1]);
        traced_shapes.insert("int_tensor".to_string(), vec![2, 4]);

        let mut types = HashMap::new();
        types.insert("float_tensor".to_string(), TensorProto::FLOAT);
        types.insert("bool_tensor".to_string(), TensorProto::BOOL);
        types.insert("int_tensor".to_string(), TensorProto::INT64);

        let (model, metadata) = make_test_model(types, traced_shapes.clone());
        let traced_types = metadata.traced_types.as_ref();
        let combined =
            materialize_combined_model(&model, &metadata, &traced_shapes, traced_types).unwrap();

        let graph = combined.graph.as_ref().unwrap();
        assert!(
            graph.output.len() > 1,
            "combined model should have intermediate outputs"
        );
    }

    #[test]
    fn combined_model_to_disk_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let slices_dir = dir.path();

        let mut traced_shapes = HashMap::new();
        traced_shapes.insert("float_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("bool_tensor".to_string(), vec![1]);
        traced_shapes.insert("string_tensor".to_string(), vec![1]);
        traced_shapes.insert("int_tensor".to_string(), vec![2, 4]);

        let mut types = HashMap::new();
        types.insert("float_tensor".to_string(), TensorProto::FLOAT);
        types.insert("bool_tensor".to_string(), TensorProto::BOOL);
        types.insert("int_tensor".to_string(), TensorProto::INT64);

        let (model, mut metadata) = make_test_model(types, traced_shapes);
        metadata.original_model_path = Some("model.onnx".to_string());

        let model_path = slices_dir.join("model.onnx");
        onnx_proto::save_model(&model, &model_path).unwrap();
        let meta_path = slices_dir.join("metadata.msgpack");
        metadata.save(&meta_path).unwrap();

        let dest = materialize_combined_to_disk(slices_dir, &metadata).unwrap();
        assert!(dest.exists());

        let loaded = onnx_proto::load_model(&dest).unwrap();
        let graph = loaded.graph.as_ref().unwrap();
        assert!(
            graph.output.len() > 1,
            "reloaded combined model should have intermediate outputs"
        );
    }

    #[test]
    fn ensure_combined_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let slices_dir = dir.path();

        let mut traced_shapes = HashMap::new();
        traced_shapes.insert("float_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("bool_tensor".to_string(), vec![1]);
        traced_shapes.insert("string_tensor".to_string(), vec![1]);
        traced_shapes.insert("int_tensor".to_string(), vec![2, 4]);

        let mut types = HashMap::new();
        types.insert("float_tensor".to_string(), TensorProto::FLOAT);
        types.insert("bool_tensor".to_string(), TensorProto::BOOL);
        types.insert("int_tensor".to_string(), TensorProto::INT64);

        let (model, mut metadata) = make_test_model(types, traced_shapes);
        metadata.original_model_path = Some("model.onnx".to_string());

        let model_path = slices_dir.join("model.onnx");
        onnx_proto::save_model(&model, &model_path).unwrap();
        let meta_path = slices_dir.join("metadata.msgpack");
        metadata.save(&meta_path).unwrap();

        let dest1 = materialize_combined_to_disk(slices_dir, &metadata).unwrap();
        let dest2 = materialize_combined_to_disk(slices_dir, &metadata).unwrap();
        assert_eq!(dest1, dest2);
    }
}
