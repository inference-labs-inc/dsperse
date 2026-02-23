use std::collections::HashMap;
use std::path::Path;

use ndarray::{ArrayD, Axis, IxDyn};

use crate::error::{DsperseError, Result};

pub fn read_input_json(path: &Path) -> Result<serde_json::Value> {
    let data = std::fs::read_to_string(path).map_err(|e| DsperseError::io(e, path))?;
    serde_json::from_str(&data).map_err(Into::into)
}

pub fn write_input_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = serde_json::to_string(value)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}

pub fn extract_input_data(value: &serde_json::Value) -> Option<&serde_json::Value> {
    value
        .get("input_data")
        .or_else(|| value.get("input"))
        .or_else(|| value.get("data"))
        .or_else(|| value.get("inputs"))
}

pub fn flatten_nested_list(value: &serde_json::Value) -> Vec<f64> {
    let mut result = Vec::new();
    flatten_recursive(value, &mut result);
    result
}

fn flatten_recursive(value: &serde_json::Value, out: &mut Vec<f64>) {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.push(f);
            } else {
                tracing::warn!(number = %n, "dropping non-f64 representable number");
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                flatten_recursive(item, out);
            }
        }
        _ => {}
    }
}

pub fn infer_shape(value: &serde_json::Value) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut current = value;
    loop {
        match current {
            serde_json::Value::Array(arr) => {
                shape.push(arr.len());
                if let Some(first) = arr.first() {
                    current = first;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    shape
}

pub fn json_to_arrayd(value: &serde_json::Value) -> Result<ArrayD<f64>> {
    let flat = flatten_nested_list(value);
    let shape = infer_shape(value);
    if flat.is_empty() {
        return Ok(ArrayD::from_shape_vec(IxDyn(&[0]), vec![])
            .map_err(|e| DsperseError::Pipeline(format!("empty arrayd: {e}")))?);
    }
    let product: usize = shape.iter().product();
    if product != flat.len() || shape.is_empty() {
        tracing::warn!(
            flat_len = flat.len(),
            ?shape,
            product,
            "shape mismatch, falling back to 1D"
        );
        return ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat)
            .map_err(|e| DsperseError::Pipeline(format!("arrayd reshape fallback: {e}")));
    }
    ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .map_err(|e| DsperseError::Pipeline(format!("arrayd reshape: {e}")))
}

pub fn arrayd_to_json(arr: &ArrayD<f64>) -> serde_json::Value {
    match arr.ndim() {
        0 => serde_json::json!(arr[IxDyn(&[])]),
        1 => {
            let vals: Vec<serde_json::Value> = arr
                .iter()
                .map(|&v| serde_json::json!(v))
                .collect();
            serde_json::Value::Array(vals)
        }
        _ => {
            let vals: Vec<serde_json::Value> = (0..arr.shape()[0])
                .map(|i| {
                    let sub = arr.index_axis(Axis(0), i).to_owned();
                    arrayd_to_json(&sub)
                })
                .collect();
            serde_json::Value::Array(vals)
        }
    }
}

pub fn extract_output_tensor(data: &serde_json::Value) -> serde_json::Value {
    data.get("output_data")
        .or_else(|| data.get("output"))
        .cloned()
        .unwrap_or_else(|| data.clone())
}

pub fn extract_input_tensor(data: &serde_json::Value) -> serde_json::Value {
    extract_input_data(data)
        .cloned()
        .unwrap_or_else(|| data.clone())
}

pub fn gather_inputs_from_cache(
    cache: &HashMap<String, ArrayD<f64>>,
    inputs: &[String],
) -> Result<ArrayD<f64>> {
    let mut collected = Vec::new();
    let mut missing = Vec::new();
    for name in inputs {
        if let Some(val) = cache.get(name) {
            collected.push(val.clone());
        } else {
            missing.push(name.clone());
        }
    }
    if collected.is_empty() {
        return Err(DsperseError::Pipeline(format!(
            "no cached tensor found for inputs: {inputs:?}"
        )));
    }
    if !missing.is_empty() {
        return Err(DsperseError::Pipeline(format!(
            "missing tensors in cache: {missing:?} (found {} of {})",
            collected.len(),
            inputs.len()
        )));
    }
    if collected.len() == 1 {
        return Ok(collected.into_iter().next().unwrap());
    }
    let ref_shape = &collected[0].shape()[1..];
    for (i, arr) in collected.iter().enumerate().skip(1) {
        if arr.shape()[1..] != *ref_shape {
            return Err(DsperseError::Pipeline(format!(
                "shape mismatch at input {}: expected trailing dims {:?}, got {:?}",
                i, ref_shape, &arr.shape()[1..]
            )));
        }
    }
    ndarray::concatenate(
        ndarray::Axis(0),
        &collected.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .map_err(|e| DsperseError::Pipeline(format!("concat inputs: {e}")))
}
