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
struct CachedBundle {
    bundle: Arc<CompiledCircuit>,
    touched: std::time::Instant,
    /// In-memory size of the compiled bundle's byte vectors, used for
    /// byte-capped eviction. Measured from the loaded form itself rather
    /// than the path, which may be a directory whose metadata length says
    /// nothing about content size.
    approx_bytes: u64,
}

#[derive(Debug)]
pub struct JstproveBackend {
    compress: bool,
    bundle_cache: Mutex<HashMap<PathBuf, CachedBundle>>,
    /// Total approx bytes the cache may hold; zero means uncapped. Provers
    /// leave this uncapped because their reuse is bursty and a hot bundle is
    /// worth its memory. Verifiers that sample across many distinct circuits
    /// see near-zero reuse and set a cap so the cache cannot occupy a large
    /// fraction of host memory holding bundles that will never be hit again.
    cache_byte_cap: std::sync::atomic::AtomicU64,
}

impl Default for JstproveBackend {
    fn default() -> Self {
        Self {
            compress: true,
            bundle_cache: Mutex::new(HashMap::new()),
            cache_byte_cap: std::sync::atomic::AtomicU64::new(0),
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

    /// Bound the cache's total approx bytes; `None` removes the bound. The
    /// bound is enforced at insert time by evicting least-recently-touched
    /// entries, never the entry being inserted, so a single oversized bundle
    /// still loads and serves.
    pub fn set_cache_byte_cap(&self, cap: Option<u64>) {
        self.cache_byte_cap
            .store(cap.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    }

    /// Current cache occupancy as (entry count, total approx bytes).
    pub fn cache_stats(&self) -> (usize, u64) {
        let cache = match self.bundle_cache.lock() {
            Ok(cache) => cache,
            Err(e) => e.into_inner(),
        };
        let bytes = cache.values().map(|c| c.approx_bytes).sum();
        (cache.len(), bytes)
    }

    pub fn load_bundle_cached(&self, path: &Path) -> Result<Arc<CompiledCircuit>> {
        let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        let mut cache = self
            .bundle_cache
            .lock()
            .map_err(|e| DsperseError::Backend(format!("bundle cache lock poisoned: {e}")))?;
        if let Some(cached) = cache.get_mut(&key) {
            cached.touched = std::time::Instant::now();
            return Ok(Arc::clone(&cached.bundle));
        }
        let bundle = Arc::new(load_bundle(path)?);
        let approx_bytes = (bundle.circuit.len() + bundle.witness_solver.len()) as u64;
        cache.insert(
            key.clone(),
            CachedBundle {
                bundle: Arc::clone(&bundle),
                touched: std::time::Instant::now(),
                approx_bytes,
            },
        );

        let cap = self
            .cache_byte_cap
            .load(std::sync::atomic::Ordering::Relaxed);
        if cap > 0 {
            let evict = plan_lru_evictions(
                cache
                    .iter()
                    .map(|(k, c)| (k.clone(), c.touched, c.approx_bytes)),
                cap,
                &key,
            );
            if !evict.is_empty() {
                for k in &evict {
                    cache.remove(k);
                }
                tracing::debug!(
                    evicted = evict.len(),
                    remaining = cache.len(),
                    cap_bytes = cap,
                    "bundle cache byte cap enforced"
                );
            }
        }

        Ok(bundle)
    }

    pub fn evict_idle(&self, ttl: std::time::Duration) -> usize {
        let mut cache = match self.bundle_cache.lock() {
            Ok(cache) => cache,
            Err(e) => e.into_inner(),
        };
        let now = std::time::Instant::now();
        let before = cache.len();
        cache.retain(|_, c| now.duration_since(c.touched) < ttl);
        before - cache.len()
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
    /// this after `verify_holographic` because the holographic
    /// verify path does not reach through `verify_and_extract`, yet
    /// the validator still needs the declared inputs (to cross-check
    /// against what it sent) and the scale fields (to report the
    /// same `VerifiedOutput` shape the non-holographic path
    /// produces). Keeping `extract_outputs` as a thin wrapper
    /// preserves the existing `Vec<f64>` contract for callers that
    /// only want the outputs.
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

/// Least-recently-touched entries to evict so total approx bytes fit under
/// `cap`. The entry at `keep` (the one just inserted) is never selected, so a
/// single bundle larger than the whole cap still loads and serves.
fn plan_lru_evictions(
    entries: impl Iterator<Item = (PathBuf, std::time::Instant, u64)>,
    cap: u64,
    keep: &Path,
) -> Vec<PathBuf> {
    let entries: Vec<(PathBuf, std::time::Instant, u64)> = entries.collect();
    let total: u64 = entries.iter().map(|(_, _, b)| *b).sum();
    let mut over = total.saturating_sub(cap);
    if over == 0 {
        return Vec::new();
    }
    let mut candidates: Vec<(PathBuf, std::time::Instant, u64)> =
        entries.into_iter().filter(|(k, _, _)| k != keep).collect();
    candidates.sort_by_key(|(_, touched, _)| *touched);
    let mut evict = Vec::new();
    for (key, _, bytes) in candidates {
        if over == 0 {
            break;
        }
        over = over.saturating_sub(bytes);
        evict.push(key);
    }
    evict
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

    fn entry(name: &str, age_secs: u64, bytes: u64) -> (PathBuf, std::time::Instant, u64) {
        (
            PathBuf::from(name),
            std::time::Instant::now() - std::time::Duration::from_secs(age_secs),
            bytes,
        )
    }

    #[test]
    fn lru_evictions_respect_cap_and_age_order() {
        let keep = PathBuf::from("new");
        let entries = vec![
            entry("oldest", 300, 400),
            entry("middle", 200, 400),
            entry("recent", 100, 400),
            entry("new", 0, 400),
        ];
        // Total 1600, cap 1000: must shed 600, oldest-first, so exactly the
        // two oldest go and the newer pair stays.
        let evict = plan_lru_evictions(entries.into_iter(), 1000, &keep);
        assert_eq!(
            evict,
            vec![PathBuf::from("oldest"), PathBuf::from("middle")]
        );
    }

    #[test]
    fn lru_evictions_never_select_the_inserted_entry() {
        let keep = PathBuf::from("huge");
        let entries = vec![entry("small", 100, 10), entry("huge", 0, 10_000)];
        // Even a bundle larger than the whole cap loads and serves; only the
        // other entries are shed.
        let evict = plan_lru_evictions(entries.into_iter(), 100, &keep);
        assert_eq!(evict, vec![PathBuf::from("small")]);
    }

    #[test]
    fn lru_evictions_noop_under_cap() {
        let keep = PathBuf::from("b");
        let entries = vec![entry("a", 100, 100), entry("b", 0, 100)];
        assert!(plan_lru_evictions(entries.into_iter(), 1000, &keep).is_empty());
    }

    #[test]
    fn cached_bundle_size_reflects_loaded_bytes() {
        let backend = JstproveBackend::default();
        let dummy = Arc::new(CompiledCircuit {
            circuit: vec![0u8; 700],
            witness_solver: vec![0u8; 300],
            metadata: None,
            version: None,
        });
        backend.bundle_cache.lock().unwrap().insert(
            PathBuf::from("/tmp/sized-circuit"),
            CachedBundle {
                approx_bytes: (dummy.circuit.len() + dummy.witness_solver.len()) as u64,
                bundle: dummy,
                touched: std::time::Instant::now(),
            },
        );
        assert_eq!(backend.cache_stats(), (1, 1000));
    }

    #[test]
    fn cache_stats_and_cap_roundtrip() {
        let backend = JstproveBackend::default();
        assert_eq!(backend.cache_stats(), (0, 0));
        backend.set_cache_byte_cap(Some(2 * 1024 * 1024 * 1024));
        backend.set_cache_byte_cap(None);
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
        backend.bundle_cache.lock().unwrap().insert(
            PathBuf::from("/tmp/test-circuit"),
            CachedBundle {
                bundle: dummy,
                touched: std::time::Instant::now(),
                approx_bytes: 3,
            },
        );
        assert_eq!(backend.bundle_cache.lock().unwrap().len(), 1);
        assert_eq!(backend.cache_stats(), (1, 3));
        backend.clear_cache();
        assert!(backend.bundle_cache.lock().unwrap().is_empty());
    }

    #[test]
    fn evict_idle_drops_only_stale_entries() {
        let backend = JstproveBackend::default();
        assert_eq!(backend.evict_idle(std::time::Duration::from_secs(0)), 0);
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
