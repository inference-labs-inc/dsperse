use std::path::Path;

use crate::backend::jstprove::JstproveBackend;
use crate::error::Result;
use crate::schema::execution::RunMetadata;

use super::stage::{PipelineStage, run_pipeline_stage};

pub fn verify_run(
    run_dir: &Path,
    slices_dir: &Path,
    backend: &JstproveBackend,
    parallel: usize,
) -> Result<RunMetadata> {
    run_pipeline_stage(
        PipelineStage::Verify,
        run_dir,
        slices_dir,
        backend,
        parallel,
    )
}
