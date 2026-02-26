use std::collections::HashMap;
use std::path::Path;

use ndarray::{ArrayD, Axis, IxDyn};
use rmpv::Value;

use crate::error::{DsperseError, Result};

pub fn read_msgpack(path: &Path) -> Result<Value> {
    let data = std::fs::read(path).map_err(|e| DsperseError::io(e, path))?;
    rmp_serde::from_slice(&data).map_err(Into::into)
}

pub fn write_msgpack(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = rmp_serde::to_vec_named(value)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}

pub fn extract_input_data(value: &Value) -> Option<&Value> {
    map_get_ref(value, "input_data")
        .or_else(|| map_get_ref(value, "input"))
        .or_else(|| map_get_ref(value, "data"))
        .or_else(|| map_get_ref(value, "inputs"))
}

pub fn flatten_nested_list(value: &Value) -> Vec<f64> {
    let mut result = Vec::new();
    flatten_recursive(value, &mut result);
    result
}

fn flatten_recursive(value: &Value, out: &mut Vec<f64>) {
    match value {
        Value::F64(f) => out.push(*f),
        Value::F32(f) => out.push(*f as f64),
        Value::Integer(n) => {
            if let Some(f) = n.as_f64() {
                out.push(f);
            } else {
                tracing::warn!(number = ?n, "flatten_recursive: dropping non-f64 representable integer");
            }
        }
        Value::Array(arr) => {
            for item in arr {
                flatten_recursive(item, out);
            }
        }
        other => {
            tracing::warn!(variant = %other, "flatten_recursive: dropping non-numeric value during flattening");
        }
    }
}

pub fn infer_shape(value: &Value) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut current = value;
    while let Value::Array(arr) = current {
        shape.push(arr.len());
        if let Some(first) = arr.first() {
            current = first;
        } else {
            break;
        }
    }
    shape
}

pub fn value_to_arrayd(value: &Value) -> Result<ArrayD<f64>> {
    let flat = flatten_nested_list(value);
    let shape = infer_shape(value);
    if flat.is_empty() {
        return ArrayD::from_shape_vec(IxDyn(&shape), vec![])
            .map_err(|e| DsperseError::Pipeline(format!("empty arrayd: {e}")));
    }
    if shape.is_empty() && flat.len() == 1 {
        return ArrayD::from_shape_vec(IxDyn(&[]), flat)
            .map_err(|e| DsperseError::Pipeline(format!("scalar arrayd: {e}")));
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

pub fn arrayd_to_value(arr: &ArrayD<f64>) -> Value {
    match arr.ndim() {
        0 => Value::F64(arr[IxDyn(&[])]),
        1 => {
            let vals: Vec<Value> = arr.iter().map(|&v| Value::F64(v)).collect();
            Value::Array(vals)
        }
        _ => {
            let vals: Vec<Value> = (0..arr.shape()[0])
                .map(|i| {
                    let sub = arr.index_axis(Axis(0), i).to_owned();
                    arrayd_to_value(&sub)
                })
                .collect();
            Value::Array(vals)
        }
    }
}

pub fn extract_output_tensor(data: &Value) -> Value {
    map_get_ref(data, "output_data")
        .or_else(|| map_get_ref(data, "output"))
        .cloned()
        .unwrap_or_else(|| data.clone())
}

pub fn extract_input_tensor(data: &Value) -> Value {
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
                i,
                ref_shape,
                &arr.shape()[1..]
            )));
        }
    }
    ndarray::concatenate(
        ndarray::Axis(0),
        &collected.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .map_err(|e| DsperseError::Pipeline(format!("concat inputs: {e}")))
}

pub fn build_msgpack_map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (Value::String(k.into()), v))
            .collect(),
    )
}

pub fn map_get_ref<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match value {
        Value::Map(entries) => entries.iter().find_map(|(k, v)| {
            if k.as_str().is_some_and(|s| s == key) {
                Some(v)
            } else {
                None
            }
        }),
        _ => None,
    }
}
