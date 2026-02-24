use std::path::Path;

use rayon::prelude::*;

use crate::backend::jstprove::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{ExecutionMethod, RunMetadata, SliceResult};
use crate::schema::metadata::RunSliceMetadata;
use crate::utils::paths::resolve_relative_path;

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
            Self::Prove => "prove",
            Self::Verify => "verification",
        }
    }
}

pub fn run_pipeline_stage(
    stage: PipelineStage,
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

    tracing::info!(total = circuit_slices.len(), "{} circuit slices", stage.action_label());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let results: Vec<_> = pool.install(|| {
        circuit_slices
            .par_iter()
            .map(|(slice_id, meta)| {
                let _slice_idx: usize =
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
            Err(e) => SliceResult {
                slice_id: slice_id.clone(),
                success: false,
                method: Some(method.to_string()),
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
            match stage {
                PipelineStage::Prove => entry.proof_execution = Some(slice_result),
                PipelineStage::Verify => entry.verification_execution = Some(slice_result),
            }
        }
    }

    match stage {
        PipelineStage::Prove => run_meta.execution_chain.jstprove_proved_slices = succeeded,
        PipelineStage::Verify => run_meta.execution_chain.jstprove_verified_slices = succeeded,
    }

    let meta_json = serde_json::to_string_pretty(&run_meta)?;
    std::fs::write(&meta_path, meta_json).map_err(|e| DsperseError::io(e, &meta_path))?;

    tracing::info!(succeeded, total = circuit_slices.len(), "{} complete", stage.action_label());
    Ok(run_meta)
}

fn execute_single_slice(
    stage: PipelineStage,
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    meta: &RunSliceMetadata,
    backend: &JstproveBackend,
) -> Result<SliceResult> {
    let start = std::time::Instant::now();
    let method = stage.execution_method();

    let circuit_path = meta
        .jstprove_circuit_path
        .as_deref()
        .map(|p| resolve_relative_path(slices_dir, p))
        .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {slice_id}")))?;

    let witness_path = slice_run_dir.join("witness.bin");

    let required_files: Vec<(&str, std::path::PathBuf)> = match stage {
        PipelineStage::Prove => vec![("witness", witness_path.clone())],
        PipelineStage::Verify => vec![
            ("witness", witness_path.clone()),
            ("proof", slice_run_dir.join("proof.bin")),
        ],
    };

    for (label, path) in &required_files {
        if !path.exists() {
            return Ok(SliceResult {
                slice_id: slice_id.into(),
                success: false,
                method: Some(method.to_string()),
                error: Some(format!("{label} file not found: {}", path.display())),
                proof_path: None,
                time_sec: 0.0,
                tiles: Vec::new(),
            });
        }
    }

    let witness_bytes =
        std::fs::read(&witness_path).map_err(|e| DsperseError::io(e, &witness_path))?;

    match stage {
        PipelineStage::Prove => {
            let proof_bytes = backend.prove(&circuit_path, &witness_bytes)?;
            let proof_path = slice_run_dir.join("proof.bin");
            std::fs::write(&proof_path, &proof_bytes)
                .map_err(|e| DsperseError::io(e, &proof_path))?;

            Ok(SliceResult {
                slice_id: slice_id.into(),
                success: true,
                method: Some(method.to_string()),
                error: None,
                proof_path: Some(proof_path.to_string_lossy().into_owned()),
                time_sec: start.elapsed().as_secs_f64(),
                tiles: Vec::new(),
            })
        }
        PipelineStage::Verify => {
            let proof_path = slice_run_dir.join("proof.bin");
            let proof_bytes =
                std::fs::read(&proof_path).map_err(|e| DsperseError::io(e, &proof_path))?;

            let valid = backend.verify(&circuit_path, &witness_bytes, &proof_bytes)?;

            Ok(SliceResult {
                slice_id: slice_id.into(),
                success: valid,
                method: Some(method.to_string()),
                error: if valid {
                    None
                } else {
                    Some("proof verification failed".into())
                },
                proof_path: Some(proof_path.to_string_lossy().into_owned()),
                time_sec: start.elapsed().as_secs_f64(),
                tiles: Vec::new(),
            })
        }
    }
}
