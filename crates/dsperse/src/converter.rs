use std::path::Path;

use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use jstprove_circuits::expander_metadata;

use crate::error::{DsperseError, Result};

pub fn prepare_jstprove_artifacts(
    onnx_path: &Path,
    weights_as_inputs: bool,
) -> Result<(CircuitParams, Architecture, WANDB)> {
    let meta = expander_metadata::generate_from_onnx(onnx_path)
        .map_err(|e| DsperseError::Pipeline(format!("ONNX metadata generation: {e:#}")))?;

    let mut params = meta.circuit_params;
    if weights_as_inputs {
        params.weights_as_inputs = true;
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
