use std::path::{Path, PathBuf};

pub const METADATA_FILE: &str = "metadata.msgpack";
pub const INPUT_FILE: &str = "input.msgpack";
pub const OUTPUT_FILE: &str = "output.msgpack";
pub const WITNESS_FILE: &str = "witness.bin";
pub const PROOF_FILE: &str = "proof.bin";

pub fn resolve_relative_path(base: &Path, relative: &str) -> PathBuf {
    if Path::new(relative).is_absolute() {
        PathBuf::from(relative)
    } else {
        base.join(relative)
    }
}

pub fn relativize_path(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

pub fn dirs_root_from(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if name.starts_with("slice_") {
        path.parent().unwrap_or(path).to_path_buf()
    } else {
        path.to_path_buf()
    }
}

pub fn slice_dir_path(root: &Path, index: usize) -> PathBuf {
    root.join(format!("slice_{index}"))
}

pub fn normalize_path(path: &str) -> PathBuf {
    if path.starts_with('~') {
        if let Some(home) = dirs_home() {
            PathBuf::from(path.replacen('~', &home.to_string_lossy(), 1))
        } else {
            PathBuf::from(path)
        }
    } else {
        PathBuf::from(path)
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
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
