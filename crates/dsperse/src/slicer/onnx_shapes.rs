use super::onnx_proto::{ModelProto, ValueInfoProto, onnx};

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

pub fn set_vi_shape(vi: &mut ValueInfoProto, shape: &[i64]) {
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
