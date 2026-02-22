use std::path::Path;

use crate::backend::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;
use crate::utils::paths::{find_metadata_path, resolve_relative_path, slice_dir_path};

pub fn compile_slices(
    slices_dir: &Path,
    backend: &JstproveBackend,
    _parallel: usize,
    weights_as_inputs: bool,
    layers: Option<&[usize]>,
) -> Result<()> {
    let meta_path =
        find_metadata_path(slices_dir).ok_or_else(|| DsperseError::Metadata("no metadata.json found in slices directory".into()))?;
    let metadata = ModelMetadata::load(&meta_path)?;

    let slices: Vec<_> = metadata
        .slices
        .iter()
        .filter(|s| layers.is_none() || layers.unwrap().contains(&s.index))
        .collect();

    tracing::info!(total = slices.len(), "compiling slices");

    let mut errors: Vec<(usize, DsperseError)> = Vec::new();

    for slice in &slices {
        match compile_single_slice(slices_dir, slice, backend, weights_as_inputs) {
            Ok(()) => {
                tracing::info!(slice = slice.index, "compiled");
            }
            Err(e) => {
                tracing::error!(slice = slice.index, error = %e, "compilation failed");
                errors.push((slice.index, e));
            }
        }
    }

    if errors.is_empty() {
        tracing::info!("all slices compiled");
        Ok(())
    } else {
        let msg = errors
            .iter()
            .map(|(idx, e)| format!("slice {idx}: {e}"))
            .collect::<Vec<_>>()
            .join("; ");
        Err(DsperseError::Pipeline(format!(
            "{} slices failed: {msg}",
            errors.len()
        )))
    }
}

fn compile_single_slice(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    backend: &JstproveBackend,
    weights_as_inputs: bool,
) -> Result<()> {
    let slice_dir = slice_dir_path(slices_dir, slice.index);
    if !slice_dir.exists() {
        return Err(DsperseError::Pipeline(format!(
            "slice directory not found: {}",
            slice_dir.display()
        )));
    }

    let onnx_path = resolve_relative_path(&slice_dir, &slice.path);
    if !onnx_path.exists() {
        return Err(DsperseError::Pipeline(format!(
            "ONNX model not found for slice {}: {}",
            slice.index,
            onnx_path.display()
        )));
    }

    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let circuit_path = jst_dir.join("circuit.bin");
    let metadata_path = jst_dir.join("metadata.json");
    let architecture_path = jst_dir.join("architecture.json");

    let wandb_path = if weights_as_inputs {
        let p = jst_dir.join("wandb.json");
        Some(p)
    } else {
        None
    };

    backend.compile(
        &circuit_path,
        &metadata_path,
        &architecture_path,
        wandb_path.as_deref(),
    )?;

    Ok(())
}
