mod compiler;
mod incremental;
pub mod packager;
mod prover;
pub mod runner;
mod stage;
pub mod strategy;
pub mod tensor_store;
pub mod tile_executor;
mod verifier;

pub use compiler::compile_slices;
pub use incremental::{IncrementalRun, SliceExecutionResult, SliceWork};
pub use prover::prove_run;
pub use runner::{
    RunConfig, extract_onnx_initializers, reconstruct_from_tiles, run_inference, split_into_tiles,
};
pub use strategy::ExecutionStrategy;
pub use tensor_store::TensorStore;
pub use verifier::verify_run;
