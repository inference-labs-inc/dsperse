use std::path::Path;

use crate::error::{DsperseError, Result};
use crate::schema::RunMetadata;

pub fn load_run_metadata(path: &Path) -> Result<RunMetadata> {
    let data =
        crate::utils::limits::read_limited(path, crate::utils::limits::MAX_METADATA_JSON_BYTES)?;
    rmp_serde::from_slice(&data).map_err(Into::into)
}

pub fn save_run_metadata(path: &Path, meta: &RunMetadata) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    let data = rmp_serde::to_vec_named(meta)?;
    std::fs::write(path, data).map_err(|e| DsperseError::io(e, path))
}
