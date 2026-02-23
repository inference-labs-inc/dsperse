use std::path::Path;

use crate::error::{DsperseError, Result};
use crate::schema::{ModelMetadata, RunMetadata};

pub fn load_model_metadata(path: &Path) -> Result<ModelMetadata> {
    let data = std::fs::read_to_string(path).map_err(|e| DsperseError::io(e, path))?;
    serde_json::from_str(&data).map_err(Into::into)
}

pub fn save_model_metadata(path: &Path, meta: &ModelMetadata) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = serde_json::to_string_pretty(meta)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}

pub fn load_run_metadata(path: &Path) -> Result<RunMetadata> {
    let data = std::fs::read_to_string(path).map_err(|e| DsperseError::io(e, path))?;
    serde_json::from_str(&data).map_err(Into::into)
}

pub fn save_run_metadata(path: &Path, meta: &RunMetadata) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = serde_json::to_string_pretty(meta)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}

pub fn load_run_results(path: &Path) -> Result<serde_json::Value> {
    let data = std::fs::read_to_string(path).map_err(|e| DsperseError::io(e, path))?;
    serde_json::from_str(&data).map_err(Into::into)
}

pub fn save_run_results(path: &Path, results: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = serde_json::to_string_pretty(results)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}
