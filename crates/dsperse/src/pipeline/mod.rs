mod compiler;
mod incremental;
mod prover;
mod runner;
mod verifier;

pub use compiler::compile_slices;
pub use incremental::{IncrementalRun, SliceExecutionResult, SliceWork};
pub use prover::prove_run;
pub use runner::{RunConfig, run_inference};
pub use verifier::verify_run;
