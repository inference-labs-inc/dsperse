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
        for wb in &meta.wandb.w_and_b {
            let shape = wb.shape.get(&wb.name).cloned().unwrap_or_default();
            params.inputs.push(
                jstprove_circuits::circuit_functions::utils::onnx_types::ONNXIO {
                    name: wb.name.clone(),
                    elem_type: 1,
                    shape,
                },
            );
        }
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
