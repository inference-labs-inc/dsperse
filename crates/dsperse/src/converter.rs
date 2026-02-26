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
