use std::path::Path;

use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_bn254, extract_outputs_bn254, prove_bn254, verify_bn254, witness_bn254,
    witness_bn254_from_f64,
};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;
use jstprove_circuits::runner::schema::{CompiledCircuit, WitnessRequest};

use crate::error::{DsperseError, Result};

#[derive(Debug)]
pub struct JstproveBackend {
    compress: bool,
}

impl Default for JstproveBackend {
    fn default() -> Self {
        Self { compress: true }
    }
}

impl JstproveBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    pub fn compress(&self) -> bool {
        self.compress
    }

    pub fn compile(
        &self,
        circuit_path: &Path,
        params: CircuitParams,
        architecture: Architecture,
        wandb: WANDB,
    ) -> Result<()> {
        OnnxContext::set_params(params.clone());
        OnnxContext::set_architecture(architecture);
        OnnxContext::set_wandb(wandb);

        let circuit_path_str = circuit_path
            .to_str()
            .ok_or_else(|| DsperseError::Backend("non-UTF8 circuit path".into()))?;

        compile_bn254(circuit_path_str, self.compress, Some(params))
            .map_err(|e| DsperseError::Backend(format!("compile: {e}")))
    }

    pub fn witness(
        &self,
        circuit_path: &Path,
        input_json: &[u8],
        output_json: &[u8],
    ) -> Result<Vec<u8>> {
        let bundle = load_bundle(circuit_path)?;

        if let Some(ref params) = bundle.metadata {
            OnnxContext::set_params(params.clone());
        }

        let req = WitnessRequest {
            circuit: bundle.circuit,
            witness_solver: bundle.witness_solver,
            inputs: input_json.to_vec(),
            outputs: output_json.to_vec(),
            metadata: bundle.metadata.clone(),
        };

        let result = witness_bn254(&req, self.compress)
            .map_err(|e| DsperseError::Backend(format!("witness: {e}")))?;

        Ok(result.witness)
    }

    pub fn witness_f64(
        &self,
        circuit_path: &Path,
        activations: &[f64],
        initializers: &[(Vec<f64>, Vec<usize>)],
    ) -> Result<Vec<u8>> {
        let bundle = load_bundle(circuit_path)?;
        let params = bundle.metadata.as_ref().ok_or_else(|| {
            DsperseError::Backend("circuit bundle missing metadata (required for quantization)".into())
        })?;

        let result = witness_bn254_from_f64(
            &bundle.circuit,
            &bundle.witness_solver,
            params,
            activations,
            initializers,
            self.compress,
        )
        .map_err(|e| DsperseError::Backend(format!("witness_f64: {e}")))?;

        Ok(result.witness)
    }

    pub fn load_params(&self, circuit_path: &Path) -> Result<Option<CircuitParams>> {
        let bundle = load_bundle(circuit_path)?;
        Ok(bundle.metadata)
    }

    pub fn prove(&self, circuit_path: &Path, witness_bytes: &[u8]) -> Result<Vec<u8>> {
        let bundle = load_bundle(circuit_path)?;

        prove_bn254(&bundle.circuit, witness_bytes, self.compress)
            .map_err(|e| DsperseError::Backend(format!("prove: {e}")))
    }

    pub fn extract_outputs(
        &self,
        witness_bytes: &[u8],
        num_model_inputs: usize,
    ) -> Result<Vec<f64>> {
        if num_model_inputs == 0 {
            return Err(DsperseError::Backend(
                "extract_outputs: num_model_inputs must be > 0".into(),
            ));
        }
        let result = extract_outputs_bn254(witness_bytes, num_model_inputs)
            .map_err(|e| DsperseError::Backend(format!("extract_outputs: {e}")))?;
        Ok(result.outputs)
    }

    pub fn verify(
        &self,
        circuit_path: &Path,
        witness_bytes: &[u8],
        proof_bytes: &[u8],
    ) -> Result<bool> {
        let bundle = load_bundle(circuit_path)?;

        verify_bn254(&bundle.circuit, witness_bytes, proof_bytes)
            .map_err(|e| DsperseError::Backend(format!("verify: {e}")))
    }
}

fn load_bundle(circuit_path: &Path) -> Result<CompiledCircuit> {
    let path_str = circuit_path
        .to_str()
        .ok_or_else(|| DsperseError::Backend("non-UTF8 circuit path".into()))?;

    read_circuit_msgpack(path_str)
        .map_err(|e| DsperseError::Backend(format!("read circuit msgpack: {e}")))
}

pub struct WarmCircuit {
    bundle: CompiledCircuit,
    pub params: CircuitParams,
    initializers: Vec<(Vec<f64>, Vec<usize>)>,
    compress: bool,
}

impl WarmCircuit {
    pub fn load(
        circuit_path: &Path,
        initializers: Vec<(Vec<f64>, Vec<usize>)>,
        compress: bool,
    ) -> Result<Self> {
        let bundle = load_bundle(circuit_path)?;
        let params = bundle
            .metadata
            .clone()
            .ok_or_else(|| DsperseError::Backend("circuit bundle missing metadata".into()))?;
        Ok(Self {
            bundle,
            params,
            initializers,
            compress,
        })
    }

    pub fn witness_f64(&self, activations: &[f64]) -> Result<Vec<u8>> {
        let result = witness_bn254_from_f64(
            &self.bundle.circuit,
            &self.bundle.witness_solver,
            &self.params,
            activations,
            &self.initializers,
            self.compress,
        )
        .map_err(|e| DsperseError::Backend(format!("witness_f64: {e}")))?;

        Ok(result.witness)
    }
}
