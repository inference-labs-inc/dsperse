use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub use jstprove_circuits::api::ExtractedOutputType as ExtractedOutput;
pub use jstprove_circuits::api::ProofConfigType as ProofConfig;
pub use jstprove_circuits::api::StampedProofConfigType as StampedProofConfig;
pub use jstprove_circuits::api::VerifiedOutputType as VerifiedOutput;
use jstprove_circuits::api::{
    self, ArchitectureType as Architecture, CircuitParamsType as CircuitParams,
    CompiledCircuitType as CompiledCircuit, WANDBType as WANDB,
};
use jstprove_circuits::runner::schema::WitnessRequest;

use crate::error::{DsperseError, Result};

use super::traits::ProofBackend;

#[derive(Debug)]
pub struct JstproveBackend {
    compress: bool,
    bundle_cache: Mutex<HashMap<PathBuf, Arc<CompiledCircuit>>>,
}

impl Default for JstproveBackend {
    fn default() -> Self {
        Self {
            compress: true,
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

    pub fn compress(&self) -> bool {
        self.compress
    }

    pub fn load_bundle_cached(&self, path: &Path) -> Result<Arc<CompiledCircuit>> {
        let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        let mut cache = self
            .bundle_cache
            .lock()
            .map_err(|e| DsperseError::Backend(format!("bundle cache lock poisoned: {e}")))?;
        if let Some(bundle) = cache.get(&key) {
            return Ok(Arc::clone(bundle));
        }
        let bundle = Arc::new(load_bundle(path)?);
        cache.insert(key, Arc::clone(&bundle));

        Ok(bundle)
    }

    pub fn clear_cache(&self) {
        let mut cache = match self.bundle_cache.lock() {
            Ok(cache) => cache,
            Err(e) => {
                tracing::warn!("bundle cache lock poisoned on clear: {e}");
                e.into_inner()
            }
        };
        let count = cache.len();
        cache.clear();
        tracing::debug!(cleared = count, "bundle cache cleared");
    }

    /// Evict cached bundles whose canonical path starts with the given
    /// prefix. Used by callers that want to drop a model's entries
    /// without clearing the entire cache.
    pub fn evict_cache_by_prefix(&self, prefix: &Path) {
        let mut cache = match self.bundle_cache.lock() {
            Ok(cache) => cache,
            Err(e) => {
                tracing::warn!("bundle cache lock poisoned on evict: {e}");
                e.into_inner()
            }
        };
        let before = cache.len();
        cache.retain(|k, _| !k.starts_with(prefix));
        let evicted = before - cache.len();
        if evicted > 0 {
            tracing::info!(
                prefix = %prefix.display(),
                evicted,
                remaining = cache.len(),
                "evicted bundle cache entries"
            );
        }
    }

    /// Resolve the proof config for a freshly loaded bundle. Errors if
    /// the bundle does not carry a stamped proof config or if the
    /// stamped version does not match the current spec, so callers can
    /// fail fast on legacy or incompatible bundles instead of running
    /// the wrong prover.
    fn resolve_proof_config(bundle: &CompiledCircuit) -> Result<ProofConfig> {
        let stamped = bundle
            .metadata
            .as_ref()
            .and_then(|m| m.proof_config)
            .ok_or_else(|| {
                DsperseError::Backend(
                    "circuit bundle has no stamped proof_config; recompile with a stamping prover"
                        .into(),
                )
            })?;
        stamped
            .ensure_current()
            .map_err(|e| DsperseError::Backend(format!("incompatible bundle: {e}")))?;
        Ok(stamped.config)
    }

    /// Resolve the proof config without touching the circuit or
    /// witness-solver blobs. Reads only `manifest.msgpack`, which is
    /// kilobytes versus the tens of megabytes a full bundle load
    /// pulls in. Falls back to `resolve_proof_config` on a full
    /// bundle load if the manifest is missing the stamp so callers
    /// still get the same "no stamped proof_config" error path for
    /// legacy bundles rather than a confusing deserialization
    /// failure.
    fn resolve_proof_config_from_manifest(&self, circuit_path: &Path) -> Result<ProofConfig> {
        match jstprove_io::bundle::read_bundle_metadata::<CircuitParams>(circuit_path) {
            Ok((Some(params), _)) => {
                let stamped = params.proof_config.ok_or_else(|| {
                    DsperseError::Backend(
                        "circuit bundle has no stamped proof_config; recompile with a stamping prover"
                            .into(),
                    )
                })?;
                stamped
                    .ensure_current()
                    .map_err(|e| DsperseError::Backend(format!("incompatible bundle: {e}")))?;
                Ok(stamped.config)
            }
            Ok((None, _)) => {
                let bundle = self.load_bundle_cached(circuit_path)?;
                Self::resolve_proof_config(&bundle)
            }
            Err(e) => {
                // Surface the manifest-read failure so operators
                // investigating a slow verify path or a legacy
                // bundle layout can tell the fast path missed
                // rather than silently eating a parse / IO error.
                tracing::debug!(
                    path = %circuit_path.display(),
                    error = %e,
                    "manifest-only proof_config read failed; falling back to full bundle load"
                );
                let bundle = self.load_bundle_cached(circuit_path)?;
                Self::resolve_proof_config(&bundle)
            }
        }
    }

    pub fn compile(
        &self,
        circuit_path: &Path,
        config: ProofConfig,
        params: CircuitParams,
        architecture: Architecture,
        wandb: WANDB,
    ) -> Result<()> {
        let circuit_path_str = circuit_path
            .to_str()
            .ok_or_else(|| DsperseError::Backend("non-UTF8 circuit path".into()))?;

        api::compile(
            circuit_path_str,
            config,
            params,
            architecture,
            wandb,
            self.compress,
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
        let config = Self::resolve_proof_config(&bundle)?;

        let req = WitnessRequest {
            circuit: bundle.circuit.clone(),
            witness_solver: bundle.witness_solver.clone(),
            inputs: input_json.to_vec(),
            outputs: output_json.to_vec(),
            metadata: bundle.metadata.clone(),
        };

        let result = api::witness(config, &req, self.compress)
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
        let config = Self::resolve_proof_config(&bundle)?;
        let params = bundle.metadata.as_ref().ok_or_else(|| {
            DsperseError::Backend(
                "circuit bundle missing metadata (required for quantization)".into(),
            )
        })?;

        let result = api::witness_f64(
            config,
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
        let config = Self::resolve_proof_config(&bundle)?;

        api::prove(config, &bundle.circuit, witness_bytes, self.compress)
            .map_err(|e| DsperseError::Backend(format!("prove: {e}")))
    }

    pub fn extract_outputs(
        &self,
        witness_bytes: &[u8],
        num_model_inputs: usize,
    ) -> Result<Vec<f64>> {
        Ok(self
            .extract_outputs_full(witness_bytes, num_model_inputs)?
            .outputs)
    }

    /// Full extracted output bundle: inputs, outputs, and the
    /// witness-stamped scale parameters. Holographic verifiers call
    /// this after `verify_holographic` because their soundness path
    /// does not reach through `verify_and_extract`, yet they still
    /// need the scale fields to report the same `VerifyResult` shape
    /// the non-holographic path produces.
    pub fn extract_outputs_full(
        &self,
        witness_bytes: &[u8],
        num_model_inputs: usize,
    ) -> Result<ExtractedOutput> {
        if num_model_inputs == 0 {
            return Err(DsperseError::Backend(
                "extract_outputs: num_model_inputs must be > 0".into(),
            ));
        }
        api::extract_outputs(witness_bytes, num_model_inputs)
            .map_err(|e| DsperseError::Backend(format!("extract_outputs: {e}")))
    }

    pub fn verify(
        &self,
        circuit_path: &Path,
        witness_bytes: &[u8],
        proof_bytes: &[u8],
    ) -> Result<bool> {
        let bundle = self.load_bundle_cached(circuit_path)?;
        let config = Self::resolve_proof_config(&bundle)?;

        api::verify(config, &bundle.circuit, witness_bytes, proof_bytes)
            .map_err(|e| DsperseError::Backend(format!("verify: {e}")))
    }

    pub fn verify_and_extract(
        &self,
        circuit_path: &Path,
        witness_bytes: &[u8],
        proof_bytes: &[u8],
        num_inputs: usize,
        expected_inputs: Option<&[f64]>,
    ) -> Result<VerifiedOutput> {
        let bundle = self.load_bundle_cached(circuit_path)?;
        let config = Self::resolve_proof_config(&bundle)?;

        api::verify_and_extract(
            config,
            &bundle.circuit,
            witness_bytes,
            proof_bytes,
            num_inputs,
            expected_inputs,
        )
        .map_err(|e| DsperseError::Backend(format!("verify_and_extract: {e}")))
    }

    /// Run holographic GKR setup against the compiled circuit at
    /// `circuit_path` and persist the resulting verifying key as
    /// `vk.bin` inside the bundle directory. The bundle is read from
    /// the cache, so callers that just compiled the bundle through
    /// [`Self::compile`] pay only the holographic setup cost on top.
    ///
    /// `setup_holographic_vk` only succeeds when the bundle was
    /// compiled with `ProofConfig::GoldilocksExt4Whir`; the underlying
    /// jstprove API rejects every other config.
    ///
    /// The vk blob is written using the same compression mode as the
    /// rest of the bundle (`Self::compress`) so
    /// `jstprove_io::bundle::read_vk_only` can decode it via the
    /// shared auto-detecting reader.
    pub fn setup_holographic_vk(&self, circuit_path: &Path) -> Result<()> {
        let bundle = self.load_bundle_cached(circuit_path)?;
        let config = Self::resolve_proof_config(&bundle)?;

        let vk_bytes = api::setup_holographic_vk(config, &bundle.circuit)
            .map_err(|e| DsperseError::Backend(format!("setup_holographic_vk: {e}")))?;

        let vk_path = circuit_path.join("vk.bin");
        let payload = if self.compress {
            jstprove_io::compress_bytes(&vk_bytes)
                .map_err(|e| DsperseError::Backend(format!("compress vk: {e}")))?
        } else {
            vk_bytes
        };
        std::fs::write(&vk_path, &payload).map_err(|e| DsperseError::io(e, &vk_path))?;
        Ok(())
    }

    /// Generate a holographic GKR proof for an existing bundle and
    /// witness. Like [`Self::setup_holographic_vk`] this requires the
    /// bundle to have been compiled with
    /// `ProofConfig::GoldilocksExt4Whir`.
    pub fn prove_holographic(&self, circuit_path: &Path, witness_bytes: &[u8]) -> Result<Vec<u8>> {
        let bundle = self.load_bundle_cached(circuit_path)?;
        let config = Self::resolve_proof_config(&bundle)?;

        api::prove_holographic(config, &bundle.circuit, witness_bytes)
            .map_err(|e| DsperseError::Backend(format!("prove_holographic: {e}")))
    }

    /// Verify a holographic GKR proof against the bundle's vk.bin.
    /// The vk is read independently of the (much larger) circuit
    /// blob, mirroring the validator-side flow where the verifying
    /// party only ever ships the vk.
    pub fn verify_holographic(&self, circuit_path: &Path, proof_bytes: &[u8]) -> Result<bool> {
        // Verifiers only need the vk and the proof config — the
        // circuit and witness solver blobs are not used downstream.
        // Skip load_bundle_cached here so validators that only ever
        // hold vk.bin + manifest.msgpack (the intended light-weight
        // deployment shape) don't fail with a missing circuit.bin
        // and don't pay the tens-of-megabytes read cost.
        let config = self.resolve_proof_config_from_manifest(circuit_path)?;
        let vk_bytes = jstprove_io::bundle::read_vk_only(circuit_path)
            .map_err(|e| DsperseError::Backend(format!("read vk: {e}")))?;

        api::verify_holographic(config, &vk_bytes, proof_bytes)
            .map_err(|e| DsperseError::Backend(format!("verify_holographic: {e}")))
    }
}

impl ProofBackend for JstproveBackend {
    fn prove(&self, circuit_path: &Path, witness_bytes: &[u8]) -> Result<Vec<u8>> {
        self.prove(circuit_path, witness_bytes)
    }

    fn verify(
        &self,
        circuit_path: &Path,
        witness_bytes: &[u8],
        proof_bytes: &[u8],
    ) -> Result<bool> {
        self.verify(circuit_path, witness_bytes, proof_bytes)
    }

    fn witness_f64(
        &self,
        circuit_path: &Path,
        activations: &[f64],
        initializers: &[(Vec<f64>, Vec<usize>)],
    ) -> Result<Vec<u8>> {
        self.witness_f64(circuit_path, activations, initializers)
    }
}

fn load_bundle(circuit_path: &Path) -> Result<CompiledCircuit> {
    let path_str = circuit_path
        .to_str()
        .ok_or_else(|| DsperseError::Backend("non-UTF8 circuit path".into()))?;

    api::read_circuit_bundle(path_str)
        .map_err(|e| DsperseError::Backend(format!("read circuit bundle: {e}")))
}

pub struct WarmCircuit {
    bundle: Arc<CompiledCircuit>,
    pub params: CircuitParams,
    initializers: Vec<(Vec<f64>, Vec<usize>)>,
    compress: bool,
    config: ProofConfig,
}

impl WarmCircuit {
    pub fn load(
        circuit_path: &Path,
        initializers: Vec<(Vec<f64>, Vec<usize>)>,
        backend: &JstproveBackend,
    ) -> Result<Self> {
        let bundle = backend.load_bundle_cached(circuit_path)?;
        let config = JstproveBackend::resolve_proof_config(&bundle)?;
        let params = bundle
            .metadata
            .clone()
            .ok_or_else(|| DsperseError::Backend("circuit bundle missing metadata".into()))?;
        Ok(Self {
            bundle,
            params,
            initializers,
            compress: backend.compress(),
            config,
        })
    }

    pub fn witness_f64(&self, activations: &[f64]) -> Result<Vec<u8>> {
        let result = api::witness_f64(
            self.config,
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
    fn bundle_cache_starts_empty() {
        let backend = JstproveBackend::default();
        let cache = backend.bundle_cache.lock().unwrap();
        assert!(cache.is_empty());
    }

    #[test]
    fn backend_constructs_without_proof_config_state() {
        let backend = JstproveBackend::default();
        assert!(backend.compress());
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

    #[test]
    fn resolve_proof_config_rejects_unstamped_bundle() {
        let bundle = CompiledCircuit {
            circuit: vec![],
            witness_solver: vec![],
            metadata: None,
            version: None,
        };
        let err = JstproveBackend::resolve_proof_config(&bundle).unwrap_err();
        match err {
            DsperseError::Backend(msg) => {
                assert!(msg.contains("no stamped proof_config"), "{msg}")
            }
            other => panic!("expected Backend error, got {other:?}"),
        }
    }
}
