use std::path::Path;

use crate::error::{DsperseError, Result};

pub const MAX_ONNX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_METADATA_JSON_BYTES: u64 = 10 * 1024 * 1024;
pub const MAX_INPUT_JSON_BYTES: u64 = 100 * 1024 * 1024;
pub const MAX_WITNESS_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_PROOF_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_ZIP_ENTRY_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_ZIP_TOTAL_BYTES: u64 = 10 * 1024 * 1024 * 1024;

pub fn check_file_size(path: &Path, max_bytes: u64) -> Result<()> {
    let metadata = std::fs::metadata(path).map_err(|e| DsperseError::io(e, path))?;
    let size = metadata.len();
    if size > max_bytes {
        return Err(DsperseError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("file size {size} exceeds limit {max_bytes}"),
            ),
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

pub fn reject_symlink(path: &Path) -> Result<()> {
    let m = std::fs::symlink_metadata(path).map_err(|e| DsperseError::io(e, path))?;
    if m.is_symlink() {
        return Err(DsperseError::Archive(format!(
            "symlink not permitted: {}",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<unknown>")
        )));
    }
    Ok(())
}

pub fn read_limited(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    reject_symlink(path)?;
    check_file_size(path, max_bytes)?;
    std::fs::read(path).map_err(|e| DsperseError::io(e, path))
}

pub fn read_to_string_limited(path: &Path, max_bytes: u64) -> Result<String> {
    reject_symlink(path)?;
    check_file_size(path, max_bytes)?;
    std::fs::read_to_string(path).map_err(|e| DsperseError::io(e, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_file_size_within_limit() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        assert!(check_file_size(tmp.path(), 1024).is_ok());
    }

    #[test]
    fn check_file_size_exceeds_limit() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        assert!(check_file_size(tmp.path(), 2).is_err());
    }

    #[test]
    fn reject_symlink_on_regular_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        assert!(reject_symlink(tmp.path()).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn reject_symlink_on_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        std::fs::write(&target, b"data").unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(reject_symlink(&link).is_err());
    }

    #[test]
    fn read_limited_normal() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        let data = read_limited(tmp.path(), 1024).unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn read_limited_exceeds() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        assert!(read_limited(tmp.path(), 2).is_err());
    }

    #[test]
    fn read_to_string_limited_normal() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello world").unwrap();
        let s = read_to_string_limited(tmp.path(), 1024).unwrap();
        assert_eq!(s, "hello world");
    }
}
