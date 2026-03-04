use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, DsperseError>;

#[derive(Debug, thiserror::Error)]
pub enum DsperseError {
    #[error("I/O error at {}: {source}", .path.file_name().and_then(|n| n.to_str()).unwrap_or("<unknown>"))]
    Io {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("msgpack encode error: {0}")]
    MsgpackEncode(#[from] rmp_serde::encode::Error),

    #[error("msgpack decode error: {0}")]
    MsgpackDecode(#[from] rmp_serde::decode::Error),

    #[error("ONNX error: {0}")]
    Onnx(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("slicer error: {0}")]
    Slicer(String),

    #[error("archive error: {0}")]
    Archive(String),

    #[error("metadata error: {0}")]
    Metadata(String),

    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error("{0}")]
    Other(String),
}

impl DsperseError {
    pub fn io(source: std::io::Error, path: impl Into<PathBuf>) -> Self {
        Self::Io {
            source,
            path: path.into(),
        }
    }
}
