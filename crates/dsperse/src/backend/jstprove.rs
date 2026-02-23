use std::path::Path;

use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{compile_bn254, prove_bn254, verify_bn254, witness_bn254};
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

    pub fn prove(&self, circuit_path: &Path, witness_bytes: &[u8]) -> Result<Vec<u8>> {
        let bundle = load_bundle(circuit_path)?;

        prove_bn254(&bundle.circuit, witness_bytes, self.compress)
            .map_err(|e| DsperseError::Backend(format!("prove: {e}")))
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
    let msgpack_path = circuit_path.with_extension("msgpack");
    let msgpack_str = msgpack_path
        .to_str()
        .ok_or_else(|| DsperseError::Backend("non-UTF8 msgpack path".into()))?;

    read_circuit_msgpack(msgpack_str)
        .map_err(|e| DsperseError::Backend(format!("read circuit msgpack: {e}")))
}
