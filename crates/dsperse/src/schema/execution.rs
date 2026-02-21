use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::metadata::{Backend, RunSliceMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMethod {
    JstproveGenWitness,
    OnnxOnly,
    OnnxMultiInput,
    Tiled,
    JstproveFallbackOnnx,
    JstproveProve,
    JstproveVerify,
}

impl std::fmt::Display for ExecutionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JstproveGenWitness => write!(f, "jstprove_gen_witness"),
            Self::OnnxOnly => write!(f, "onnx_only"),
            Self::OnnxMultiInput => write!(f, "onnx_multi_input"),
            Self::Tiled => write!(f, "tiled"),
            Self::JstproveFallbackOnnx => write!(f, "jstprove_fallback_onnx"),
            Self::JstproveProve => write!(f, "jstprove_prove"),
            Self::JstproveVerify => write!(f, "jstprove_verify"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileResult {
    pub tile_idx: usize,
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub time_sec: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionInfo {
    pub method: String,
    #[serde(default)]
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub witness_file: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        alias = "tiles",
        rename = "tile_exec_infos"
    )]
    pub tile_exec_infos: Vec<TileResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceResult {
    pub slice_id: String,
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_path: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub time_sec: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiles: Vec<TileResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionNode {
    pub slice_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<String>,
    #[serde(default)]
    pub fallbacks: Vec<String>,
    #[serde(default)]
    pub use_circuit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub circuit_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_path: Option<String>,
    #[serde(default)]
    pub backend: String,
}

impl Default for ExecutionNode {
    fn default() -> Self {
        Self {
            slice_id: String::new(),
            primary: None,
            fallbacks: Vec::new(),
            use_circuit: false,
            next: None,
            circuit_path: None,
            onnx_path: None,
            backend: Backend::Onnx.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResultEntry {
    pub slice_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub witness_execution: Option<ExecutionInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_execution: Option<SliceResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_execution: Option<SliceResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionChain {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    #[serde(default)]
    pub nodes: HashMap<String, ExecutionNode>,
    #[serde(default)]
    pub fallback_map: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub execution_results: Vec<ExecutionResultEntry>,
    #[serde(default)]
    pub jstprove_proved_slices: usize,
    #[serde(default)]
    pub jstprove_verified_slices: usize,
}

impl ExecutionChain {
    pub fn get_result_for_slice(&self, slice_id: &str) -> Option<&ExecutionResultEntry> {
        self.execution_results
            .iter()
            .find(|e| e.slice_id == slice_id)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunMetadata {
    #[serde(default)]
    pub slices: HashMap<String, RunSliceMetadata>,
    #[serde(default)]
    pub execution_chain: ExecutionChain,
    #[serde(default)]
    pub circuit_slices: HashMap<String, bool>,
    #[serde(default)]
    pub overall_security: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packaging_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_directory: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
}

impl RunMetadata {
    pub fn get_slice(&self, slice_id: &str) -> Option<&RunSliceMetadata> {
        self.slices.get(slice_id)
    }

    pub fn iter_circuit_slices(&self) -> impl Iterator<Item = (&str, &RunSliceMetadata)> {
        self.execution_chain
            .nodes
            .iter()
            .filter(|(_, node)| node.use_circuit)
            .filter_map(|(slice_id, _)| {
                self.slices
                    .get(slice_id)
                    .map(|meta| (slice_id.as_str(), meta))
            })
    }
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}
