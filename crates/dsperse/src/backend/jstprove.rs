use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
    fast_compile: bool,
    bundle_cache: Mutex<HashMap<PathBuf, Arc<CompiledCircuit>>>,
}

impl Default for JstproveBackend {
    fn default() -> Self {
        Self {
            compress: true,
            fast_compile: false,
            bundle_cache: Mutex::new(HashMap::new()),
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

    pub fn with_fast_compile(mut self, fast_compile: bool) -> Self {
        self.fast_compile = fast_compile;
        self
    }

    pub fn compress(&self) -> bool {
        self.compress
    }

    pub fn load_bundle_cached(&self, path: &Path) -> Result<Arc<CompiledCircuit>> {
        let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        {
            let cache = self
                .bundle_cache
                .lock()
                .map_err(|e| DsperseError::Backend(format!("bundle cache lock poisoned: {e}")))?;
            if let Some(bundle) = cache.get(&key) {
                return Ok(Arc::clone(bundle));
            }
        }

        let bundle = Arc::new(load_bundle(path)?);

        let mut cache = self
            .bundle_cache
            .lock()
            .map_err(|e| DsperseError::Backend(format!("bundle cache lock poisoned: {e}")))?;
        if let Some(existing) = cache.get(&key) {
            return Ok(Arc::clone(existing));
        }
        cache.insert(key, Arc::clone(&bundle));

        Ok(bundle)
    }

    pub fn clear_cache(&self) {
        match self.bundle_cache.lock() {
            Ok(mut cache) => {
                let count = cache.len();
                cache.clear();
                tracing::debug!(cleared = count, "bundle cache cleared");
            }
            Err(e) => {
                tracing::warn!("bundle cache lock poisoned on clear: {e}");
            }
        }
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

        compile_bn254(
            circuit_path_str,
            self.compress,
            Some(params),
            self.fast_compile,
        )
        .map_err(|e| DsperseError::Backend(format!("compile: {e}")))?;

        let key = circuit_path
            .canonicalize()
            .unwrap_or_else(|_| circuit_path.to_path_buf());
        self.bundle_cache
            .lock()
            .map_err(|e| DsperseError::Backend(format!("bundle cache lock poisoned: {e}")))?
            .remove(&key);

        Ok(())
    }

    pub fn witness(
        &self,
        circuit_path: &Path,
        input_json: &[u8],
        output_json: &[u8],
    ) -> Result<Vec<u8>> {
        let bundle = self.load_bundle_cached(circuit_path)?;

        if let Some(ref params) = bundle.metadata {
            OnnxContext::set_params(params.clone());
        }

        let req = WitnessRequest {
            circuit: bundle.circuit.clone(),
            witness_solver: bundle.witness_solver.clone(),
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
        let bundle = self.load_bundle_cached(circuit_path)?;
        let params = bundle.metadata.as_ref().ok_or_else(|| {
            DsperseError::Backend(
                "circuit bundle missing metadata (required for quantization)".into(),
            )
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
        let bundle = self.load_bundle_cached(circuit_path)?;
        Ok(bundle.metadata.clone())
    }

    pub fn prove(&self, circuit_path: &Path, witness_bytes: &[u8]) -> Result<Vec<u8>> {
        let bundle = self.load_bundle_cached(circuit_path)?;

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
        let bundle = self.load_bundle_cached(circuit_path)?;

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
    bundle: Arc<CompiledCircuit>,
    pub params: CircuitParams,
    initializers: Vec<(Vec<f64>, Vec<usize>)>,
    compress: bool,
}

impl WarmCircuit {
    pub fn load(
        circuit_path: &Path,
        initializers: Vec<(Vec<f64>, Vec<usize>)>,
        backend: &JstproveBackend,
    ) -> Result<Self> {
        let bundle = backend.load_bundle_cached(circuit_path)?;
        let params = bundle
            .metadata
            .clone()
            .ok_or_else(|| DsperseError::Backend("circuit bundle missing metadata".into()))?;
        Ok(Self {
            bundle,
            params,
            initializers,
            compress: backend.compress(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fast_compile_is_false() {
        let backend = JstproveBackend::default();
        assert!(!backend.fast_compile);
    }

    #[test]
    fn with_fast_compile_sets_flag() {
        let backend = JstproveBackend::default().with_fast_compile(true);
        assert!(backend.fast_compile);
    }

    #[test]
    fn with_fast_compile_false_preserves_default() {
        let backend = JstproveBackend::default().with_fast_compile(false);
        assert!(!backend.fast_compile);
    }

    #[test]
    fn builder_chain_compress_and_fast_compile() {
        let backend = JstproveBackend::default()
            .with_compress(false)
            .with_fast_compile(true);
        assert!(!backend.compress());
        assert!(backend.fast_compile);
    }

    #[test]
    fn bundle_cache_starts_empty() {
        let backend = JstproveBackend::default();
        let cache = backend.bundle_cache.lock().unwrap();
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_cache_on_empty_succeeds() {
        let backend = JstproveBackend::default();
        backend.clear_cache();
        let cache = backend.bundle_cache.lock().unwrap();
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_cache_removes_entries() {
        let backend = JstproveBackend::default();
        let dummy = Arc::new(CompiledCircuit {
            circuit: vec![1, 2, 3],
            witness_solver: vec![],
            metadata: None,
            version: None,
        });
        backend
            .bundle_cache
            .lock()
            .unwrap()
            .insert(PathBuf::from("/tmp/test-circuit"), dummy);
        assert_eq!(backend.bundle_cache.lock().unwrap().len(), 1);
        backend.clear_cache();
        assert!(backend.bundle_cache.lock().unwrap().is_empty());
    }

    #[test]
    fn load_bundle_cached_returns_error_for_missing_path() {
        let backend = JstproveBackend::default();
        let result = backend.load_bundle_cached(Path::new("/nonexistent/circuit/path"));
        assert!(result.is_err());
        assert!(backend.bundle_cache.lock().unwrap().is_empty());
    }
}
