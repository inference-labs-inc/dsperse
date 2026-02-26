use std::path::Path;

use rayon::prelude::*;

use crate::backend::jstprove::JstproveBackend;
use crate::converter;
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;
use crate::slicer::autotiler::JSTPROVE_SUPPORTED_OPS;
use crate::slicer::onnx_proto;
use crate::utils::paths::{find_metadata_path, slice_dir_path};

pub fn compile_slices(
    slices_dir: &Path,
    backend: &JstproveBackend,
    parallel: usize,
    weights_as_inputs: bool,
    layers: Option<&[usize]>,
) -> Result<()> {
    let meta_path = find_metadata_path(slices_dir).ok_or_else(|| {
        DsperseError::Metadata(format!("no {} found in slices directory", crate::utils::paths::METADATA_FILE))
    })?;
    let metadata = ModelMetadata::load(&meta_path)?;

    if metadata.original_model_path.is_some() {
        crate::slicer::materializer::ensure_all_slices_materialized(slices_dir, &metadata)?;
    }

    let slices: Vec<_> = metadata
        .slices
        .iter()
        .filter(|s| layers.is_none_or(|l| l.contains(&s.index)))
        .collect();

    tracing::info!(total = slices.len(), "compiling slices");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let errors: Vec<_> = pool.install(|| {
        slices
            .par_iter()
            .filter_map(|slice| {
                match compile_single_slice(slices_dir, slice, backend, weights_as_inputs) {
                    Ok(true) => {
                        tracing::info!(slice = slice.index, "compiled");
                        None
                    }
                    Ok(false) => {
                        tracing::info!(slice = slice.index, "skipped (unsupported ops)");
                        None
                    }
                    Err(e) => {
                        tracing::error!(slice = slice.index, error = %e, "compilation failed");
                        Some((slice.index, e))
                    }
                }
            })
            .collect()
    });

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

fn is_jstprove_compatible(onnx_path: &Path) -> Result<bool> {
    let model = onnx_proto::load_model(onnx_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Slicer(format!("no graph in {}", onnx_path.display())))?;
    Ok(graph
        .node
        .iter()
        .all(|n| JSTPROVE_SUPPORTED_OPS.contains(&n.op_type.as_str())))
}

fn compile_single_slice(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    backend: &JstproveBackend,
    weights_as_inputs: bool,
) -> Result<bool> {
    let slice_dir = slice_dir_path(slices_dir, slice.index);
    if !slice_dir.exists() {
        return Err(DsperseError::Pipeline(format!(
            "slice directory not found: {}",
            slice_dir.display()
        )));
    }

    let onnx_path = resolve_compile_onnx(slices_dir, slice)?;
    if !onnx_path.exists() {
        return Err(DsperseError::Pipeline(format!(
            "ONNX model not found for slice {}: {}",
            slice.index,
            onnx_path.display()
        )));
    }

    if !is_jstprove_compatible(&onnx_path)? {
        return Ok(false);
    }

    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let circuit_path = jst_dir.join("circuit.bin");

    let (params, architecture, wandb) =
        converter::prepare_jstprove_artifacts(&onnx_path, weights_as_inputs)?;

    backend.compile(&circuit_path, params, architecture, wandb)?;

    Ok(true)
}

fn resolve_compile_onnx(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
) -> Result<std::path::PathBuf> {
    if let Some(ref tiling) = slice.tiling {
        if let Some(ref tile) = tiling.tile {
            let tile_path = slices_dir.join(&tile.path);
            if tile_path.exists() {
                tracing::info!(
                    slice = slice.index,
                    path = %tile_path.display(),
                    "using tile ONNX"
                );
                return Ok(tile_path);
            }
        }
    }

    Ok(slice.resolve_onnx(slices_dir))
}
