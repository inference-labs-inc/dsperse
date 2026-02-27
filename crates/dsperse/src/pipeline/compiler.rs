use std::path::Path;

use rayon::prelude::*;

use crate::backend::jstprove::JstproveBackend;
use crate::converter;
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;
use crate::slicer::onnx_proto;
use crate::utils::paths::{find_metadata_path, slice_dir_path};

pub fn compile_slices(
    slices_dir: &Path,
    backend: &JstproveBackend,
    parallel: usize,
    weights_as_inputs: bool,
    layers: Option<&[usize]>,
    jstprove_ops: &[&str],
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
                match compile_single_slice(slices_dir, slice, backend, weights_as_inputs, jstprove_ops) {
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

fn is_jstprove_compatible(onnx_path: &Path, jstprove_ops: &[&str]) -> Result<bool> {
    let model = onnx_proto::load_model(onnx_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Slicer(format!("no graph in {}", onnx_path.display())))?;
    Ok(graph
        .node
        .iter()
        .all(|n| jstprove_ops.contains(&n.op_type.as_str())))
}

fn compile_single_slice(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    backend: &JstproveBackend,
    weights_as_inputs: bool,
    jstprove_ops: &[&str],
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

    if !is_jstprove_compatible(&onnx_path, jstprove_ops)? {
        return Ok(false);
    }

    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let circuit_path = jst_dir.join("circuit.msgpack");

    if circuit_path.exists() {
        match backend.load_params(&circuit_path) {
            Ok(_) => {
                tracing::info!(slice = slice.index, "already compiled, skipping");
                return Ok(true);
            }
            Err(e) => {
                tracing::warn!(slice = slice.index, error = %e, "cached circuit invalid, recompiling");
                let _ = std::fs::remove_file(&circuit_path);
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::metadata::{
        Compilation, Dependencies, SliceMetadata, SliceShapeWrapper, TensorShape,
    };
    use crate::schema::tiling::{TileInfo, TilingInfo};

    fn test_models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models"))
    }

    fn make_slice_metadata(index: usize, path: &str) -> SliceMetadata {
        SliceMetadata {
            index,
            filename: format!("slice_{index}.onnx"),
            path: path.to_string(),
            relative_path: path.to_string(),
            shape: SliceShapeWrapper {
                tensor_shape: TensorShape::default(),
            },
            dependencies: Dependencies {
                input: vec![],
                output: vec![],
                filtered_inputs: vec![],
            },
            tiling: None,
            channel_split: None,
            compilation: Compilation::default(),
            slice_metadata: None,
            slice_metadata_relative_path: None,
        }
    }

    const TEST_OPS: &[&str] = &["Conv", "Gemm", "MatMul"];

    #[test]
    fn is_jstprove_compatible_nonexistent() {
        let result = is_jstprove_compatible(Path::new("/nonexistent.onnx"), TEST_OPS);
        assert!(result.is_err());
    }

    #[test]
    fn is_jstprove_compatible_test_model() {
        let model_path = test_models_dir().join("net/model.onnx");
        assert!(model_path.exists(), "fixture missing: {}", model_path.display());
        let result = is_jstprove_compatible(&model_path, TEST_OPS).unwrap();
        assert!(!result);
    }

    #[test]
    fn resolve_compile_onnx_no_tiling() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let slice_dir = slices_dir.join("slice_0");
        std::fs::create_dir_all(&slice_dir).unwrap();

        let meta = make_slice_metadata(0, "slice_0.onnx");
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("slice_0.onnx"));
    }

    #[test]
    fn resolve_compile_onnx_with_tile() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let tile_path = slices_dir.join("slice_0/payload/tiles/tile.onnx");
        std::fs::create_dir_all(tile_path.parent().unwrap()).unwrap();
        std::fs::write(&tile_path, b"dummy").unwrap();

        let mut meta = make_slice_metadata(0, "slice_0.onnx");
        meta.tiling = Some(TilingInfo {
            slice_idx: 0,
            tile_size: 8,
            num_tiles: 4,
            tiles_y: 2,
            tiles_x: 2,
            halo: [1, 1],
            out_tile: [4, 4],
            stride: [1, 1],
            c_in: 3,
            c_out: 16,
            input_name: "input".into(),
            output_name: "output".into(),
            tile: Some(TileInfo {
                path: "slice_0/payload/tiles/tile.onnx".into(),
                conv_out: [4, 4],
                jstprove_circuit_path: None,
            }),
            tiles: None,
        });
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("tile.onnx"));
    }

    #[test]
    fn resolve_compile_onnx_tile_missing_falls_back() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let slice_dir = slices_dir.join("slice_0");
        std::fs::create_dir_all(&slice_dir).unwrap();

        let mut meta = make_slice_metadata(0, "slice_0.onnx");
        meta.tiling = Some(TilingInfo {
            slice_idx: 0,
            tile_size: 8,
            num_tiles: 4,
            tiles_y: 2,
            tiles_x: 2,
            halo: [1, 1],
            out_tile: [4, 4],
            stride: [1, 1],
            c_in: 3,
            c_out: 16,
            input_name: "input".into(),
            output_name: "output".into(),
            tile: Some(TileInfo {
                path: "slice_0/payload/tiles/nonexistent.onnx".into(),
                conv_out: [4, 4],
                jstprove_circuit_path: None,
            }),
            tiles: None,
        });
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("slice_0.onnx"));
    }
}
