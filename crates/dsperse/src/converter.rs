use std::collections::HashSet;
use std::path::Path;

use jstprove_circuits::api::{
    self, ArchitectureType as Architecture, CircuitParamsType as CircuitParams, WANDBType as WANDB,
};

use crate::error::{DsperseError, Result};

pub fn prepare_jstprove_artifacts(
    onnx_path: &Path,
    weights_as_inputs: bool,
) -> Result<(CircuitParams, Architecture, WANDB)> {
    prepare_jstprove_artifacts_filtered(onnx_path, weights_as_inputs, &HashSet::new())
}

pub fn prepare_jstprove_artifacts_filtered(
    onnx_path: &Path,
    weights_as_inputs: bool,
    exclude_from_wai: &HashSet<String>,
) -> Result<(CircuitParams, Architecture, WANDB)> {
    let meta = api::generate_metadata(onnx_path)
        .map_err(|e| DsperseError::Pipeline(format!("ONNX metadata generation: {e:#}")))?;

    let mut params = meta.circuit_params;
    if weights_as_inputs {
        api::populate_wai_inputs(&mut params, &meta.wandb, exclude_from_wai);
    }

    Ok((params, meta.architecture, meta.wandb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_jstprove_artifacts_nonexistent_model() {
        let result = prepare_jstprove_artifacts(Path::new("/nonexistent.onnx"), false);
        assert!(result.is_err());
    }

    #[test]
    fn prepare_jstprove_artifacts_with_weights_as_inputs() {
        let result = prepare_jstprove_artifacts(Path::new("/nonexistent.onnx"), true);
        assert!(result.is_err());
    }
}
