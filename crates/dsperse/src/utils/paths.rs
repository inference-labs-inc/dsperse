use std::path::{Path, PathBuf};

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
    let expanded = if path.starts_with('~') {
        if let Some(home) = dirs_home() {
            PathBuf::from(path.replacen('~', &home.to_string_lossy(), 1))
        } else {
            PathBuf::from(path)
        }
    } else {
        PathBuf::from(path)
    };
    expanded
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub fn find_metadata_path(dir: &Path) -> Option<PathBuf> {
    let direct = dir.join("metadata.json");
    if direct.exists() {
        return Some(direct);
    }
    let slices = dir.join("slices").join("metadata.json");
    if slices.exists() {
        return Some(slices);
    }
    None
}
