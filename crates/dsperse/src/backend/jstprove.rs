use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lru::LruCache;

use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{compile_bn254, prove_bn254, verify_bn254, witness_bn254};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;
use jstprove_circuits::runner::schema::{CompiledCircuit, WitnessRequest};

use crate::error::{DsperseError, Result};

#[derive(Debug, Clone)]
pub struct JstproveBackend {
    compress: bool,
    cache: Arc<Mutex<LruCache<PathBuf, Arc<CompiledCircuit>>>>,
}

impl Default for JstproveBackend {
    fn default() -> Self {
        Self {
            compress: true,
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(16).unwrap()))),
        }
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

    fn get_bundle(&self, circuit_path: &Path) -> Result<CompiledCircuit> {
        let msgpack_path = circuit_path.with_extension("msgpack");
        let canonical = msgpack_path
            .canonicalize()
            .map_err(|e| DsperseError::io(e, &msgpack_path))?;

        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(bundle) = cache.get(&canonical) {
            return Ok(Arc::unwrap_or_clone(bundle.clone()));
        }

        let msgpack_str = canonical
            .to_str()
            .ok_or_else(|| DsperseError::Backend("non-UTF8 msgpack path".into()))?;
        let bundle = read_circuit_msgpack(msgpack_str)
            .map_err(|e| DsperseError::Backend(format!("read circuit msgpack: {e}")))?;

        let bundle = Arc::new(bundle);
        cache.put(canonical, Arc::clone(&bundle));
        Ok(Arc::unwrap_or_clone(bundle))
    }

    pub fn compile(
        &self,
        circuit_path: &Path,
        metadata_path: &Path,
        architecture_path: &Path,
        wandb_path: Option<&Path>,
    ) -> Result<()> {
        let meta_json = std::fs::read_to_string(metadata_path)
            .map_err(|e| DsperseError::io(e, metadata_path))?;
        let params: CircuitParams =
            serde_json::from_str(&meta_json).map_err(|e| DsperseError::Backend(format!("metadata json: {e}")))?;

        let arch_json = std::fs::read_to_string(architecture_path)
            .map_err(|e| DsperseError::io(e, architecture_path))?;
        let arch: Architecture =
            serde_json::from_str(&arch_json).map_err(|e| DsperseError::Backend(format!("architecture json: {e}")))?;

        OnnxContext::set_params(params.clone());
        OnnxContext::set_architecture(arch);

        if let Some(wandb) = wandb_path {
            let wandb_json =
                std::fs::read_to_string(wandb).map_err(|e| DsperseError::io(e, wandb))?;
            let wandb_data: WANDB = serde_json::from_str(&wandb_json)
                .map_err(|e| DsperseError::Backend(format!("wandb json: {e}")))?;
            OnnxContext::set_wandb(wandb_data);
        }

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
        let bundle = self.get_bundle(circuit_path)?;

        if let Some(ref params) = bundle.metadata {
            OnnxContext::set_params(params.clone());
        }

        let req = WitnessRequest {
            circuit: bundle.circuit,
            witness_solver: bundle.witness_solver,
            inputs: input_json.to_vec(),
            outputs: output_json.to_vec(),
        };

        let result = witness_bn254(&req, self.compress)
            .map_err(|e| DsperseError::Backend(format!("witness: {e}")))?;

        Ok(result.witness)
    }

    pub fn prove(&self, circuit_path: &Path, witness_bytes: &[u8]) -> Result<Vec<u8>> {
        let bundle = self.get_bundle(circuit_path)?;

        prove_bn254(&bundle.circuit, witness_bytes, self.compress)
            .map_err(|e| DsperseError::Backend(format!("prove: {e}")))
    }

    pub fn verify(
        &self,
        circuit_path: &Path,
        witness_bytes: &[u8],
        proof_bytes: &[u8],
    ) -> Result<bool> {
        let bundle = self.get_bundle(circuit_path)?;

        verify_bn254(&bundle.circuit, witness_bytes, proof_bytes)
            .map_err(|e| DsperseError::Backend(format!("verify: {e}")))
    }
}
