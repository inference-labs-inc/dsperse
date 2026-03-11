use std::path::Path;

use jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams;

use crate::error::Result;

pub trait ProofBackend: Send + Sync {
    fn prove(&self, circuit_path: &Path, witness_bytes: &[u8]) -> Result<Vec<u8>>;

    fn verify(&self, circuit_path: &Path, witness_bytes: &[u8], proof_bytes: &[u8])
    -> Result<bool>;

    fn witness_f64(
        &self,
        circuit_path: &Path,
        activations: &[f64],
        initializers: &[(Vec<f64>, Vec<usize>)],
    ) -> Result<Vec<u8>>;

    fn load_params(&self, circuit_path: &Path) -> Result<Option<CircuitParams>>;
}
