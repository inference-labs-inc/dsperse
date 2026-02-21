mod compiler;
mod prover;
mod runner;
mod verifier;

pub use compiler::compile_slices;
pub use prover::prove_run;
pub use runner::{run_inference, RunConfig};
pub use verifier::verify_run;
