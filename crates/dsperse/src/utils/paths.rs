use std::path::{Component, Path, PathBuf};

use crate::error::{DsperseError, Result};

pub const METADATA_FILE: &str = "metadata.msgpack";
pub const INPUT_FILE: &str = "input.msgpack";
pub const OUTPUT_FILE: &str = "output.msgpack";
pub const WITNESS_FILE: &str = "witness.bin";
pub const PROOF_FILE: &str = "proof.bin";

pub fn resolve_relative_path(base: &Path, relative: &str) -> Result<PathBuf> {
    let rel = Path::new(relative);
    if rel.is_absolute() {
        return Err(DsperseError::Archive(format!(
            "absolute path in metadata is not permitted: {relative}"
        )));
    }
    for component in rel.components() {
        match component {
            Component::ParentDir => {
                return Err(DsperseError::Archive(format!(
                    "path traversal component in metadata is not permitted: {relative}"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(DsperseError::Archive(format!(
                    "invalid path component in metadata: {relative}"
                )));
            }
            _ => {}
        }
    }
    Ok(base.join(rel))
}

pub fn relativize_path(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

pub fn slice_dir_path(root: &Path, index: usize) -> PathBuf {
    root.join(format!("slice_{index}"))
}

pub fn find_metadata_path(dir: &Path) -> Option<PathBuf> {
    let direct = dir.join(METADATA_FILE);
    if direct.exists() {
        return Some(direct);
    }
    let slices = dir.join("slices").join(METADATA_FILE);
    if slices.exists() {
        return Some(slices);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_relative_normal_path() {
        let base = Path::new("/tmp/slices");
        let result = resolve_relative_path(base, "payload/model.onnx").unwrap();
        assert_eq!(result, PathBuf::from("/tmp/slices/payload/model.onnx"));
    }

    #[test]
    fn resolve_relative_rejects_absolute() {
        let base = Path::new("/tmp/slices");
        assert!(resolve_relative_path(base, "/etc/passwd").is_err());
    }

    #[test]
    fn resolve_relative_rejects_parent_dir() {
        let base = Path::new("/tmp/slices");
        assert!(resolve_relative_path(base, "../../../etc/passwd").is_err());
    }

    #[test]
    fn resolve_relative_rejects_embedded_parent() {
        let base = Path::new("/tmp/slices");
        assert!(resolve_relative_path(base, "payload/../../../etc/passwd").is_err());
    }

    #[test]
    fn resolve_relative_allows_current_dir() {
        let base = Path::new("/tmp/slices");
        let result = resolve_relative_path(base, "./model.onnx").unwrap();
        assert_eq!(result, PathBuf::from("/tmp/slices/./model.onnx"));
    }

    #[test]
    fn resolve_relative_empty_string() {
        let base = Path::new("/tmp/slices");
        let result = resolve_relative_path(base, "").unwrap();
        assert_eq!(result, PathBuf::from("/tmp/slices/"));
    }
}
