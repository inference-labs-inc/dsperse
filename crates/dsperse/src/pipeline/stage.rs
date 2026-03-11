use std::path::Path;

use rayon::prelude::*;

use crate::backend::ProofBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{ExecutionMethod, RunMetadata, SliceResult, TileResult};
use crate::schema::metadata::RunSliceMetadata;
use crate::schema::tiling::TilingInfo;
use crate::utils::paths::resolve_relative_path;

use super::tile_executor::resolve_tile_circuit;

#[derive(Debug, Clone, Copy)]
pub enum PipelineStage {
    Prove,
    Verify,
}

impl PipelineStage {
    fn execution_method(&self) -> ExecutionMethod {
        match self {
            Self::Prove => ExecutionMethod::JstproveProve,
            Self::Verify => ExecutionMethod::JstproveVerify,
        }
    }

    fn action_label(&self) -> &'static str {
        match self {
            Self::Prove => "proving",
            Self::Verify => "verifying",
        }
    }

    fn past_label(&self) -> &'static str {
        match self {
            Self::Prove => "proved",
            Self::Verify => "verified",
        }
    }

    fn error_label(&self) -> &'static str {
        match self {
            Self::Prove => "proof",
            Self::Verify => "verification",
        }
    }
}

pub fn run_pipeline_stage(
    stage: PipelineStage,
    run_dir: &Path,
    slices_dir: &Path,
    backend: &dyn ProofBackend,
    parallel: usize,
) -> Result<RunMetadata> {
    let meta_path = run_dir.join(crate::utils::paths::METADATA_FILE);
    let data = crate::utils::limits::read_checked(&meta_path)?;
    let mut run_meta: RunMetadata = rmp_serde::from_slice(&data)?;

    let circuit_slices: Vec<(String, _)> = run_meta
        .iter_circuit_slices()
        .map(|(id, meta)| (id.to_string(), meta.clone()))
        .collect();

    tracing::info!(
        total = circuit_slices.len(),
        "{} circuit slices",
        stage.action_label()
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let results: Vec<_> = pool.install(|| {
        circuit_slices
            .par_iter()
            .map(|(slice_id, meta)| {
                if slice_id.strip_prefix("slice_").and_then(|s| s.parse::<usize>().ok()).is_none() {
                    return (
                        slice_id.clone(),
                        Err(DsperseError::Pipeline(format!(
                            "invalid slice_id format: {slice_id:?}"
                        ))),
                    );
                }
                let slice_run_dir = run_dir.join(slice_id);

                let result =
                    execute_single_slice(stage, slices_dir, &slice_run_dir, slice_id, meta, backend);

                match &result {
                    Ok(r) if r.success => tracing::info!(slice = %slice_id, "{}", stage.past_label()),
                    Ok(r) => tracing::error!(
                        slice = %slice_id,
                        error = r.error.as_deref().unwrap_or("unknown"),
                        "{} failed", stage.error_label()
                    ),
                    Err(e) => tracing::error!(slice = %slice_id, error = %e, "{} error", stage.error_label()),
                }

                (slice_id.clone(), result)
            })
            .collect()
    });

    let method = stage.execution_method();
    let mut succeeded = 0;
    for (slice_id, result) in results {
        let slice_result = match result {
            Ok(r) => {
                if r.success {
                    succeeded += 1;
                }
                r
            }
            Err(e) => SliceResult::failure(slice_id.clone(), method, e.to_string(), 0.0),
        };

        if let Some(entry) = run_meta
            .execution_chain
            .execution_results
            .iter_mut()
            .find(|e| e.slice_id == slice_id)
        {
            match stage {
                PipelineStage::Prove => entry.proof_execution = Some(slice_result),
                PipelineStage::Verify => entry.verification_execution = Some(slice_result),
            }
        } else {
            tracing::warn!(
                slice = %slice_id,
                stage = ?stage,
                success = slice_result.success,
                error = slice_result.error.as_deref().unwrap_or("none"),
                "no matching execution_results entry, result dropped"
            );
        }
    }

    match stage {
        PipelineStage::Prove => run_meta.execution_chain.jstprove_proved_slices = succeeded,
        PipelineStage::Verify => run_meta.execution_chain.jstprove_verified_slices = succeeded,
    }

    let meta_bytes = rmp_serde::to_vec_named(&run_meta)?;
    std::fs::write(&meta_path, meta_bytes).map_err(|e| DsperseError::io(e, &meta_path))?;

    tracing::info!(
        succeeded,
        total = circuit_slices.len(),
        "{} complete",
        stage.action_label()
    );
    Ok(run_meta)
}

fn execute_single_slice(
    stage: PipelineStage,
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    meta: &RunSliceMetadata,
    backend: &dyn ProofBackend,
) -> Result<SliceResult> {
    if let Some(ref tiling) = meta.tiling {
        let default_circuit_path = meta
            .jstprove_circuit_path
            .as_deref()
            .map(|p| resolve_relative_path(slices_dir, p))
            .transpose()?;
        return execute_tiled_stage(
            stage,
            slice_id,
            default_circuit_path.as_deref(),
            slice_run_dir,
            tiling,
            slices_dir,
            backend,
        );
    }

    let circuit_path = meta
        .jstprove_circuit_path
        .as_deref()
        .map(|p| resolve_relative_path(slices_dir, p))
        .transpose()?
        .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {slice_id}")))?;

    let start = std::time::Instant::now();
    let method = stage.execution_method();
    let witness_path = slice_run_dir.join(crate::utils::paths::WITNESS_FILE);
    let witness_bytes = match crate::utils::limits::read_checked(&witness_path) {
        Ok(b) => b,
        Err(e) => {
            return Ok(SliceResult::failure(
                slice_id,
                method,
                format!("witness file read error: {}: {e}", witness_path.display()),
                start.elapsed().as_secs_f64(),
            ));
        }
    };

    execute_stage_operation(
        stage,
        slice_id,
        &circuit_path,
        &witness_bytes,
        slice_run_dir,
        backend,
        start,
        method,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_stage_operation(
    stage: PipelineStage,
    slice_id: &str,
    circuit_path: &Path,
    witness_bytes: &[u8],
    output_dir: &Path,
    backend: &dyn ProofBackend,
    start: std::time::Instant,
    method: ExecutionMethod,
) -> Result<SliceResult> {
    match stage {
        PipelineStage::Prove => {
            let proof_bytes = backend.prove(circuit_path, witness_bytes)?;
            let proof_path = output_dir.join(crate::utils::paths::PROOF_FILE);
            std::fs::write(&proof_path, &proof_bytes)
                .map_err(|e| DsperseError::io(e, &proof_path))?;

            let mut result = SliceResult::success(slice_id, method, start.elapsed().as_secs_f64());
            result.proof_path = Some(proof_path.to_string_lossy().into_owned());
            Ok(result)
        }
        PipelineStage::Verify => {
            let proof_path = output_dir.join(crate::utils::paths::PROOF_FILE);
            let proof_bytes = match crate::utils::limits::read_checked(&proof_path) {
                Ok(b) => b,
                Err(e) => {
                    return Ok(SliceResult::failure(
                        slice_id,
                        method,
                        format!("proof file read error: {}: {e}", proof_path.display()),
                        start.elapsed().as_secs_f64(),
                    ));
                }
            };

            let valid = backend.verify(circuit_path, witness_bytes, &proof_bytes)?;

            let elapsed = start.elapsed().as_secs_f64();
            let mut result = if valid {
                SliceResult::success(slice_id, method, elapsed)
            } else {
                SliceResult::failure(
                    slice_id,
                    method,
                    "proof verification failed".into(),
                    elapsed,
                )
            };
            result.proof_path = Some(proof_path.to_string_lossy().into_owned());
            Ok(result)
        }
    }
}

fn execute_tiled_stage(
    stage: PipelineStage,
    slice_id: &str,
    default_circuit_path: Option<&Path>,
    slice_run_dir: &Path,
    tiling: &TilingInfo,
    slices_dir: &Path,
    backend: &dyn ProofBackend,
) -> Result<SliceResult> {
    if tiling.num_tiles == 0 {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: tiling.num_tiles is 0"
        )));
    }

    let start = std::time::Instant::now();
    let method = stage.execution_method();

    let tile_results: Vec<TileResult> = (0..tiling.num_tiles)
        .into_par_iter()
        .map(|tile_idx| {
            let tile_start = std::time::Instant::now();
            let fail = |error: String| {
                TileResult::failure(
                    tile_idx,
                    error,
                    Some(method),
                    tile_start.elapsed().as_secs_f64(),
                )
            };
            let tile_dir = slice_run_dir.join(format!("tile_{tile_idx}"));

            let tile_circuit_path =
                match resolve_tile_circuit(tiling, tile_idx, slices_dir, default_circuit_path) {
                    Ok(Some(p)) => p,
                    Ok(None) => return fail(format!("no circuit path for tile {tile_idx}")),
                    Err(e) => return fail(e),
                };

            let witness_path = tile_dir.join(crate::utils::paths::WITNESS_FILE);
            let witness_bytes = match crate::utils::limits::read_checked(&witness_path) {
                Ok(b) => b,
                Err(e) => {
                    return fail(format!(
                        "witness read error: {}: {e}",
                        witness_path.display()
                    ));
                }
            };

            execute_tile_stage_operation(
                stage,
                tile_idx,
                &tile_circuit_path,
                &witness_bytes,
                &tile_dir,
                backend,
                method,
                tile_start,
            )
        })
        .collect();

    let failed = tile_results.iter().filter(|t| !t.success).count();
    let all_success = failed == 0;

    let elapsed = start.elapsed().as_secs_f64();
    let mut result = if all_success {
        SliceResult::success(slice_id, method, elapsed)
    } else {
        SliceResult::failure(
            slice_id,
            method,
            format!("{failed} of {} tiles failed", tiling.num_tiles),
            elapsed,
        )
    };
    result.tiles = tile_results;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn execute_tile_stage_operation(
    stage: PipelineStage,
    tile_idx: usize,
    circuit_path: &Path,
    witness_bytes: &[u8],
    tile_dir: &Path,
    backend: &dyn ProofBackend,
    method: ExecutionMethod,
    tile_start: std::time::Instant,
) -> TileResult {
    let fail = |error: String| {
        TileResult::failure(
            tile_idx,
            error,
            Some(method),
            tile_start.elapsed().as_secs_f64(),
        )
    };

    match stage {
        PipelineStage::Prove => {
            let proof_bytes = match backend.prove(circuit_path, witness_bytes) {
                Ok(b) => b,
                Err(e) => return fail(e.to_string()),
            };
            let proof_path = tile_dir.join(crate::utils::paths::PROOF_FILE);
            if let Err(e) = std::fs::write(&proof_path, &proof_bytes) {
                return fail(format!("write proof: {}: {e}", proof_path.display()));
            }
            let mut result =
                TileResult::success(tile_idx, Some(method), tile_start.elapsed().as_secs_f64());
            result.proof_path = Some(proof_path.to_string_lossy().into_owned());
            result
        }
        PipelineStage::Verify => {
            let proof_path = tile_dir.join(crate::utils::paths::PROOF_FILE);
            let proof_bytes = match crate::utils::limits::read_checked(&proof_path) {
                Ok(b) => b,
                Err(e) => {
                    return fail(format!("proof read error: {}: {e}", proof_path.display()));
                }
            };
            let valid = match backend.verify(circuit_path, witness_bytes, &proof_bytes) {
                Ok(v) => v,
                Err(e) => return fail(e.to_string()),
            };
            let elapsed = tile_start.elapsed().as_secs_f64();
            let mut result = if valid {
                TileResult::success(tile_idx, Some(method), elapsed)
            } else {
                TileResult::failure(
                    tile_idx,
                    "proof verification failed".into(),
                    Some(method),
                    elapsed,
                )
            };
            result.proof_path = Some(proof_path.to_string_lossy().into_owned());
            result
        }
    }
}
