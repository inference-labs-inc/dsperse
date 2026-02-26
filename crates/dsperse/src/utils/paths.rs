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
