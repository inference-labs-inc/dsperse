use std::path::Path;

use crate::error::{DsperseError, Result};

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

pub fn read_checked(path: &Path) -> Result<Vec<u8>> {
    reject_symlink(path)?;
    std::fs::read(path).map_err(|e| DsperseError::io(e, path))
}

pub fn read_to_string_checked(path: &Path) -> Result<String> {
    reject_symlink(path)?;
    std::fs::read_to_string(path).map_err(|e| DsperseError::io(e, path))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn read_checked_normal() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        let data = read_checked(tmp.path()).unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn read_to_string_checked_normal() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello world").unwrap();
        let s = read_to_string_checked(tmp.path()).unwrap();
        assert_eq!(s, "hello world");
    }
}
