use serde::{Deserialize, Serialize};

use super::tiling::{ChannelSplitInfo, TilingInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[serde(alias = "JSTPROVE")]
    Jstprove,
    Onnx,
    Auto,
}

impl Default for Backend {
    fn default() -> Self {
        Self::Onnx
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Jstprove => write!(f, "jstprove"),
            Self::Onnx => write!(f, "onnx"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorShape {
    #[serde(default)]
    pub input: Vec<Vec<i64>>,
    #[serde(default)]
    pub output: Vec<Vec<i64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dependencies {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
    #[serde(default)]
    pub filtered_inputs: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompilationFiles {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "compiled_circuit",
        alias = "circuit"
    )]
    pub compiled: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pk_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vk_key: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendCompilation {
    #[serde(default)]
    pub compiled: bool,
    #[serde(default)]
    pub tiled: bool,
    #[serde(default)]
    pub weights_as_inputs: bool,
    #[serde(default)]
    pub files: CompilationFiles,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compilation_timestamp: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Compilation {
    #[serde(default)]
    pub jstprove: BackendCompilation,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SliceShapeWrapper {
    #[serde(default)]
    pub tensor_shape: TensorShape,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceMetadata {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub relative_path: String,
    #[serde(default)]
    pub shape: SliceShapeWrapper,
    #[serde(default)]
    pub dependencies: Dependencies,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiling: Option<TilingInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_split: Option<ChannelSplitInfo>,
    #[serde(default)]
    pub compilation: Compilation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_metadata: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_metadata_relative_path: Option<String>,
}

impl SliceMetadata {
    pub fn output_names(&self) -> &[String] {
        &self.dependencies.output
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSliceMetadata {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub input_shape: Vec<Vec<i64>>,
    #[serde(default)]
    pub output_shape: Vec<Vec<i64>>,
    #[serde(default)]
    pub dependencies: Dependencies,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiling: Option<TilingInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_split: Option<ChannelSplitInfo>,
    #[serde(default)]
    pub backend: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "circuit_path"
    )]
    pub jstprove_circuit_path: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "settings_path"
    )]
    pub jstprove_settings_path: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    #[serde(default)]
    pub original_model: String,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub input_shape: Vec<Vec<i64>>,
    #[serde(default)]
    pub output_shapes: Vec<Vec<i64>>,
    #[serde(default)]
    pub slice_points: Vec<usize>,
    #[serde(default)]
    pub slices: Vec<SliceMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsperse_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsperse_rev: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jstprove_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jstprove_rev: Option<String>,
}

impl ModelMetadata {
    pub fn load(path: &std::path::Path) -> crate::error::Result<Self> {
        let data =
            std::fs::read(path).map_err(|e| crate::error::DsperseError::io(e, path))?;
        rmp_serde::from_slice(&data).map_err(Into::into)
    }

    pub fn save(&self, path: &std::path::Path) -> crate::error::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| crate::error::DsperseError::io(e, parent))?;
        }
        let data = rmp_serde::to_vec_named(self)?;
        std::fs::write(path, data).map_err(|e| crate::error::DsperseError::io(e, path))
    }

    pub fn stamp_version(&mut self) {
        let ver = crate::version::dsperse_artifact_version();
        self.dsperse_version = Some(ver.dsperse_version);
        self.dsperse_rev = ver.dsperse_rev;
        self.jstprove_version = Some(ver.jstprove_version);
        self.jstprove_rev = ver.jstprove_rev;
    }
}
