use std::path::Path;

use rayon::prelude::*;

use crate::backend::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{ExecutionMethod, RunMetadata, SliceResult};
use crate::utils::paths::resolve_relative_path;

pub fn verify_run(
    run_dir: &Path,
    slices_dir: &Path,
    backend: &JstproveBackend,
    parallel: usize,
) -> Result<RunMetadata> {
    let meta_path = run_dir.join("metadata.json");
    let data = std::fs::read_to_string(&meta_path).map_err(|e| DsperseError::io(e, &meta_path))?;
    let mut run_meta: RunMetadata = serde_json::from_str(&data)?;

    let circuit_slices: Vec<(String, _)> = run_meta
        .iter_circuit_slices()
        .map(|(id, meta)| (id.to_string(), meta.clone()))
        .collect();

    tracing::info!(total = circuit_slices.len(), "verifying circuit slices");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let results: Vec<_> = pool.install(|| {
        circuit_slices
            .par_iter()
            .map(|(slice_id, meta)| {
                let slice_idx: usize =
                    match slice_id.strip_prefix("slice_").and_then(|s| s.parse().ok()) {
                        Some(idx) => idx,
                        None => {
                            return (
                                slice_id.clone(),
                                Err(DsperseError::Pipeline(format!(
                                    "invalid slice_id format: {slice_id:?}"
                                ))),
                            );
                        }
                    };
                let slice_dir = slices_dir.join(format!("slice_{slice_idx}"));
                let slice_run_dir = run_dir.join(slice_id);

                let result =
                    verify_single_slice(&slice_dir, &slice_run_dir, slice_id, meta, backend);

                match &result {
                    Ok(r) if r.success => tracing::info!(slice = %slice_id, "verified"),
                    Ok(r) => tracing::error!(
                        slice = %slice_id,
                        error = r.error.as_deref().unwrap_or("unknown"),
                        "verification failed"
                    ),
                    Err(e) => tracing::error!(slice = %slice_id, error = %e, "verify error"),
                }

                (slice_id.clone(), result)
            })
            .collect()
    });

    let mut verified = 0;
    for (slice_id, result) in results {
        let verify_result = match result {
            Ok(r) => {
                if r.success {
                    verified += 1;
                }
                r
            }
            Err(e) => SliceResult {
                slice_id: slice_id.clone(),
                success: false,
                method: Some(ExecutionMethod::JstproveVerify.to_string()),
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
            entry.verification_execution = Some(verify_result);
        }
    }

    run_meta.execution_chain.jstprove_verified_slices = verified;

    let meta_json = serde_json::to_string_pretty(&run_meta)?;
    std::fs::write(&meta_path, meta_json).map_err(|e| DsperseError::io(e, &meta_path))?;

    tracing::info!(
        verified,
        total = circuit_slices.len(),
        "verification complete"
    );
    Ok(run_meta)
}

fn verify_single_slice(
    slice_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    meta: &crate::schema::metadata::RunSliceMetadata,
    backend: &JstproveBackend,
) -> Result<SliceResult> {
    let start = std::time::Instant::now();

    let circuit_path = meta
        .jstprove_circuit_path
        .as_deref()
        .or(meta.circuit_path.as_deref())
        .map(|p| resolve_relative_path(slice_dir, p))
        .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {slice_id}")))?;

    let witness_path = slice_run_dir.join("witness.bin");
    let proof_path = slice_run_dir.join("proof.bin");

    for (label, path) in [("witness", &witness_path), ("proof", &proof_path)] {
        if !path.exists() {
            return Ok(SliceResult {
                slice_id: slice_id.into(),
                success: false,
                method: Some(ExecutionMethod::JstproveVerify.to_string()),
                error: Some(format!("{label} file not found: {}", path.display())),
                proof_path: None,
                time_sec: 0.0,
                tiles: Vec::new(),
            });
        }
    }

    let witness_bytes =
        std::fs::read(&witness_path).map_err(|e| DsperseError::io(e, &witness_path))?;
    let proof_bytes = std::fs::read(&proof_path).map_err(|e| DsperseError::io(e, &proof_path))?;

    let valid = backend.verify(&circuit_path, &witness_bytes, &proof_bytes)?;

    if !valid {
        return Ok(SliceResult {
            slice_id: slice_id.into(),
            success: false,
            method: Some(ExecutionMethod::JstproveVerify.to_string()),
            error: Some("proof verification failed".into()),
            proof_path: Some(proof_path.to_string_lossy().into_owned()),
            time_sec: start.elapsed().as_secs_f64(),
            tiles: Vec::new(),
        });
    }

    Ok(SliceResult {
        slice_id: slice_id.into(),
        success: true,
        method: Some(ExecutionMethod::JstproveVerify.to_string()),
        error: None,
        proof_path: Some(proof_path.to_string_lossy().into_owned()),
        time_sec: start.elapsed().as_secs_f64(),
        tiles: Vec::new(),
    })
}
