pub mod activations;
mod channel_split;
mod combined;
mod compiler;
mod dim_split;
mod incremental;
pub mod packager;
mod prover;
pub mod publisher;
pub mod runner;
pub mod slice_cache;
mod stage;
pub mod strategy;
pub mod tensor_store;
pub mod tile_executor;
mod tiled;
mod verifier;

pub use combined::CombinedRun;
pub use compiler::{
    CompileReport, HolographicSetupReport, SliceAnalysisReport, analyze_slices, compile_slices,
    setup_holographic_for_slices,
};
pub use dim_split::{
    GroupPayloadPart, dim_split_bound_inputs, dim_split_group_payloads_planned,
    dim_split_unit_count, dim_split_weight_and_transb, dim_split_weight_chunk, plan_group_payload,
    split_for_dim_split_dispatch,
};
pub use incremental::{IncrementalRun, SliceExecutionResult, SliceWork};
pub use prover::prove_run;
pub use runner::{RunConfig, extract_onnx_initializers, run_inference, split_inline_wai_inputs};
pub use slice_cache::SliceAssets;
pub use strategy::ExecutionStrategy;
pub use tensor_store::TensorStore;
pub use tiled::{
    reconstruct_from_tiles, split_for_multi_input_dispatch, split_for_tiling, split_into_tiles,
};
pub use verifier::verify_run;
