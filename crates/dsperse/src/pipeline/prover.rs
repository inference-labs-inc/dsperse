use std::path::Path;

use rayon::prelude::*;

use crate::backend::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{ExecutionMethod, RunMetadata, SliceResult};
use crate::utils::paths::resolve_relative_path;

pub fn prove_run(
    run_dir: &Path,
    slices_dir: &Path,
    backend: &JstproveBackend,
    parallel: usize,
    tiles: Option<&[usize]>,
) -> Result<RunMetadata> {
    let meta_path = run_dir.join("metadata.json");
    let data = std::fs::read_to_string(&meta_path).map_err(|e| DsperseError::io(e, &meta_path))?;
    let mut run_meta: RunMetadata = serde_json::from_str(&data)?;

    let circuit_slices: Vec<(String, _)> = run_meta
        .iter_circuit_slices()
        .map(|(id, meta)| (id.to_string(), meta.clone()))
        .collect();

    tracing::info!(total = circuit_slices.len(), "proving circuit slices");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let results: Vec<_> = pool.install(|| {
        circuit_slices
            .par_iter()
            .map(|(slice_id, meta)| {
                let slice_idx: usize = slice_id
                    .strip_prefix("slice_")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let slice_dir = slices_dir.join(format!("slice_{slice_idx}"));
                let slice_run_dir = run_dir.join(slice_id);

                let result = prove_single_slice(
                    &slice_dir,
                    &slice_run_dir,
                    slice_id,
                    meta,
                    backend,
                    tiles,
                );

                match &result {
                    Ok(r) if r.success => tracing::info!(slice = %slice_id, "proved"),
                    Ok(r) => tracing::error!(
                        slice = %slice_id,
                        error = r.error.as_deref().unwrap_or("unknown"),
                        "prove failed"
                    ),
                    Err(e) => tracing::error!(slice = %slice_id, error = %e, "prove error"),
                }

                (slice_id.clone(), result)
            })
            .collect()
    });

    let mut proved = 0;
    for (slice_id, result) in results {
        let proof_result = match result {
            Ok(r) => {
                if r.success {
                    proved += 1;
                }
                r
            }
            Err(e) => SliceResult {
                slice_id: slice_id.clone(),
                success: false,
                method: Some(ExecutionMethod::JstproveProve.to_string()),
                error: Some(e.to_string()),
                proof_path: None,
                time_sec: 0.0,
                tiles: Vec::new(),
            },
        };

        if let Some(entry) = run_meta
            .execution_chain
            .execution_results
            .iter_mut()
            .find(|e| e.slice_id == slice_id)
        {
            entry.proof_execution = Some(proof_result);
        }
    }

    run_meta.execution_chain.jstprove_proved_slices = proved;

    let meta_json = serde_json::to_string_pretty(&run_meta)?;
    std::fs::write(&meta_path, meta_json).map_err(|e| DsperseError::io(e, &meta_path))?;

    tracing::info!(proved, total = circuit_slices.len(), "proving complete");
    Ok(run_meta)
}

fn prove_single_slice(
    slice_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    meta: &crate::schema::metadata::RunSliceMetadata,
    backend: &JstproveBackend,
    _tiles: Option<&[usize]>,
) -> Result<SliceResult> {
    let start = std::time::Instant::now();

    let circuit_path = meta
        .jstprove_circuit_path
        .as_deref()
        .or(meta.circuit_path.as_deref())
        .map(|p| resolve_relative_path(slice_dir, p))
        .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {slice_id}")))?;

    let witness_path = slice_run_dir.join("witness.bin");

    if !witness_path.exists() {
        return Ok(SliceResult {
            slice_id: slice_id.into(),
            success: false,
            method: Some(ExecutionMethod::JstproveProve.to_string()),
            error: Some("witness file not found".into()),
            proof_path: None,
            time_sec: 0.0,
            tiles: Vec::new(),
        });
    }

    let witness_bytes =
        std::fs::read(&witness_path).map_err(|e| DsperseError::io(e, &witness_path))?;

    let proof_bytes = backend.prove(&circuit_path, &witness_bytes)?;

    let proof_path = slice_run_dir.join("proof.bin");
    std::fs::write(&proof_path, &proof_bytes).map_err(|e| DsperseError::io(e, &proof_path))?;

    Ok(SliceResult {
        slice_id: slice_id.into(),
        success: true,
        method: Some(ExecutionMethod::JstproveProve.to_string()),
        error: None,
        proof_path: Some(proof_path.to_string_lossy().into_owned()),
        time_sec: start.elapsed().as_secs_f64(),
        tiles: Vec::new(),
    })
}
