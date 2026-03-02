mod compiler;
mod incremental;
mod prover;
mod runner;
mod stage;
mod verifier;

pub use compiler::compile_slices;
pub use incremental::{IncrementalRun, SliceExecutionResult, SliceWork};
pub use prover::prove_run;
pub use runner::{RunConfig, extract_onnx_initializers, reconstruct_from_tiles, run_inference, split_into_tiles};
pub use verifier::verify_run;
