use std::path::Path;

use rmpv::Value;

use crate::error::{DsperseError, Result};
use crate::schema::{ModelMetadata, RunMetadata};

pub fn load_model_metadata(path: &Path) -> Result<ModelMetadata> {
    let data = std::fs::read(path).map_err(|e| DsperseError::io(e, path))?;
    rmp_serde::from_slice(&data).map_err(Into::into)
}

pub fn save_model_metadata(path: &Path, meta: &ModelMetadata) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = rmp_serde::to_vec_named(meta)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}

pub fn load_run_metadata(path: &Path) -> Result<RunMetadata> {
    let data = std::fs::read(path).map_err(|e| DsperseError::io(e, path))?;
    rmp_serde::from_slice(&data).map_err(Into::into)
}

pub fn save_run_metadata(path: &Path, meta: &RunMetadata) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = rmp_serde::to_vec_named(meta)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}

pub fn load_run_results(path: &Path) -> Result<Value> {
    let data = std::fs::read(path).map_err(|e| DsperseError::io(e, path))?;
    rmp_serde::from_slice(&data).map_err(Into::into)
}

pub fn save_run_results(path: &Path, results: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = rmp_serde::to_vec_named(results)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}
