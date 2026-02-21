use std::path::Path;

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
