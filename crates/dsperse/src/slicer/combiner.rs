use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::materializer::build_node_output_types;
use super::onnx_proto::{self, ModelProto, TensorProto, ValueInfoProto};
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;

pub fn materialize_combined_model(
    model: &ModelProto,
    metadata: &ModelMetadata,
    traced_shapes: &HashMap<String, Vec<i64>>,
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
        let init_types: HashMap<&str, i32> = graph
            .initializer
            .iter()
            .map(|i| (i.name.as_str(), i.data_type))
            .collect();
        let node_output_types = build_node_output_types(graph);

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

                if let Some(vi) = resolve_value_info(
                    output_name,
                    &vi_map,
                    traced_shapes,
                    &init_types,
                    &node_output_types,
                )? {
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

                if let Some(vi) = resolve_value_info(
                    input_name,
                    &vi_map,
                    traced_shapes,
                    &init_types,
                    &node_output_types,
                )? {
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
    init_types: &HashMap<&str, i32>,
    node_output_types: &HashMap<String, i32>,
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

    let elem_type = init_types
        .get(name)
        .copied()
        .or_else(|| node_output_types.get(name).copied())
        .unwrap_or(TensorProto::FLOAT);

    if NON_NUMERIC_TENSOR_TYPES.contains(&elem_type) {
        return Ok(None);
    }

    Ok(Some(onnx_proto::make_tensor_value_info(
        name, elem_type, shape,
    )))
}

pub fn materialize_combined_to_disk(
    slices_dir: &Path,
    metadata: &ModelMetadata,
) -> Result<PathBuf> {
    let traced_shapes = metadata.traced_shapes.as_ref().ok_or_else(|| {
        DsperseError::Slicer("metadata missing traced_shapes for combined model".into())
    })?;
    let original_path = metadata.original_model_path.as_ref().ok_or_else(|| {
        DsperseError::Slicer("metadata missing original_model_path for combined model".into())
    })?;

    let model_path = if Path::new(original_path).is_absolute() {
        PathBuf::from(original_path)
    } else {
        slices_dir.join(original_path)
    };

    let mut model = onnx_proto::load_model(&model_path)?;
    onnx_proto::normalize_opset(&mut model);
    let mut combined = materialize_combined_model(&model, metadata, traced_shapes)?;

    let output_path = slices_dir.join("combined.onnx");
    onnx_proto::save_model(&mut combined, &output_path)?;

    tracing::info!(path = %output_path.display(), "materialized combined ONNX");

    Ok(output_path)
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_OPS: &[&str] = &["Conv", "Gemm", "MatMul"];

    #[test]
    fn combined_model_has_intermediate_outputs() {
        let models_dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/models/net"
        ));
        let model_path = models_dir.join("model.onnx");
        if !model_path.exists() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let meta = crate::slicer::slice_model(&model_path, Some(tmp.path()), None, TEST_OPS, None)
            .unwrap();

        if meta.slices.len() <= 1 {
            return;
        }

        let traced = meta.traced_shapes.as_ref().unwrap();
        let model = onnx_proto::load_model(&tmp.path().join("model.onnx")).unwrap();
        let original_output_count = model.graph.as_ref().unwrap().output.len();

        let combined = materialize_combined_model(&model, &meta, traced).unwrap();
        let combined_output_count = combined.graph.as_ref().unwrap().output.len();

        assert!(
            combined_output_count > original_output_count,
            "combined model should have more outputs than original: {combined_output_count} vs {original_output_count}"
        );

        let output_names: HashSet<String> = combined
            .graph
            .as_ref()
            .unwrap()
            .output
            .iter()
            .map(|o| o.name.clone())
            .collect();

        for slice in &meta.slices {
            for out in &slice.dependencies.output {
                let node_produces = combined
                    .graph
                    .as_ref()
                    .unwrap()
                    .node
                    .iter()
                    .any(|n| n.output.contains(out));
                if node_produces {
                    assert!(
                        output_names.contains(out),
                        "slice {} output '{out}' should be in combined model outputs",
                        slice.index
                    );
                }
            }
        }
    }

    #[test]
    fn combined_model_to_disk_roundtrip() {
        let models_dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/models/net"
        ));
        let model_path = models_dir.join("model.onnx");
        if !model_path.exists() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let meta = crate::slicer::slice_model(&model_path, Some(tmp.path()), None, TEST_OPS, None)
            .unwrap();

        let combined_path = materialize_combined_to_disk(tmp.path(), &meta).unwrap();
        assert!(combined_path.exists());

        let reloaded = onnx_proto::load_model(&combined_path).unwrap();
        assert!(reloaded.graph.is_some());
    }

    #[test]
    fn ensure_combined_is_idempotent() {
        let models_dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/models/net"
        ));
        let model_path = models_dir.join("model.onnx");
        if !model_path.exists() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let meta = crate::slicer::slice_model(&model_path, Some(tmp.path()), None, TEST_OPS, None)
            .unwrap();

        let p1 = ensure_combined_materialized(tmp.path(), &meta).unwrap();
        let p2 = ensure_combined_materialized(tmp.path(), &meta).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn bool_outputs_included_in_combined_model() {
        let vi_map: HashMap<String, &ValueInfoProto> = HashMap::new();
        let init_types: HashMap<&str, i32> = HashMap::new();

        let mut node_output_types = HashMap::new();
        node_output_types.insert("float_tensor".to_string(), TensorProto::FLOAT);
        node_output_types.insert("bool_tensor".to_string(), TensorProto::BOOL);
        node_output_types.insert("string_tensor".to_string(), ONNX_STRING_DATATYPE);
        node_output_types.insert("int_tensor".to_string(), TensorProto::INT64);

        let mut traced_shapes = HashMap::new();
        traced_shapes.insert("float_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("bool_tensor".to_string(), vec![1, 3, 8, 8]);
        traced_shapes.insert("string_tensor".to_string(), vec![4]);
        traced_shapes.insert("int_tensor".to_string(), vec![1, 300]);

        let float_vi = resolve_value_info(
            "float_tensor",
            &vi_map,
            &traced_shapes,
            &init_types,
            &node_output_types,
        )
        .unwrap();
        assert!(float_vi.is_some());

        let bool_vi = resolve_value_info(
            "bool_tensor",
            &vi_map,
            &traced_shapes,
            &init_types,
            &node_output_types,
        )
        .unwrap();
        assert!(bool_vi.is_some());

        let string_vi = resolve_value_info(
            "string_tensor",
            &vi_map,
            &traced_shapes,
            &init_types,
            &node_output_types,
        )
        .unwrap();
        assert!(string_vi.is_none());

        let int_vi = resolve_value_info(
            "int_tensor",
            &vi_map,
            &traced_shapes,
            &init_types,
            &node_output_types,
        )
        .unwrap();
        assert!(int_vi.is_some());
    }
}
