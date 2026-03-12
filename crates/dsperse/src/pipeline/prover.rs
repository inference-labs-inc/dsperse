use std::path::Path;

use crate::backend::ProofBackend;
use crate::error::Result;
use crate::schema::execution::RunMetadata;

use super::stage::{PipelineStage, run_pipeline_stage};

pub fn prove_run(
    run_dir: &Path,
    slices_dir: &Path,
    backend: &dyn ProofBackend,
    parallel: usize,
) -> Result<RunMetadata> {
    run_pipeline_stage(PipelineStage::Prove, run_dir, slices_dir, backend, parallel)
}
