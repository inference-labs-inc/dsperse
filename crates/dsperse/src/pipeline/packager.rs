use std::fs;
use std::path::Path;

use walkdir::WalkDir;
use zip::ZipWriter;
use zip::write::FileOptions;

use crate::error::{DsperseError, Result};

pub struct PackageResult {
    pub count: usize,
    pub total_size: u64,
}

pub fn package_slices(slices_dir: &Path, cleanup: bool) -> Result<PackageResult> {
    if !slices_dir.is_dir() {
        return Err(DsperseError::Other(format!(
            "slices directory not found: {}",
            slices_dir.display()
        )));
    }

    let mut slice_dirs: Vec<_> = fs::read_dir(slices_dir)
        .map_err(|e| DsperseError::io(e, slices_dir))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            name_str.starts_with("slice_")
                && entry.path().is_dir()
                && name_str["slice_".len()..].parse::<usize>().is_ok()
        })
        .collect();

    slice_dirs.sort_by_key(|entry| {
        let name = entry.file_name();
        let name_str = name.to_string_lossy().to_string();
        name_str["slice_".len()..].parse::<usize>().unwrap_or(0)
    });

    let mut dslice_names: Vec<String> = Vec::with_capacity(slice_dirs.len());
    let mut total_size: u64 = 0;

    for (i, entry) in slice_dirs.iter().enumerate() {
        let slice_path = entry.path();
        let dir_name = entry.file_name();
        let archive_name = format!("{}.dslice", dir_name.to_string_lossy());
        let archive_path = slices_dir.join(&archive_name);

        create_zip_archive(&slice_path, &archive_path)?;

        let archive_size = fs::metadata(&archive_path)
            .map_err(|e| DsperseError::io(e, &archive_path))?
            .len();
        total_size += archive_size;

        dslice_names.push(archive_name);

        if cleanup {
            fs::remove_dir_all(&slice_path).map_err(|e| DsperseError::io(e, &slice_path))?;
        }

        if (i + 1) % 50 == 0 {
            tracing::info!(
                progress = i + 1,
                total = slice_dirs.len(),
                "packaging slices"
            );
        }
    }

    let parent_name = slices_dir
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let metadata = serde_json::json!({
        "name": parent_name,
        "dslices": dslice_names,
    });

    let metadata_path = slices_dir.join("metadata.json");
    let metadata_bytes =
        serde_json::to_string_pretty(&metadata).map_err(|e| DsperseError::Other(e.to_string()))?;
    fs::write(&metadata_path, metadata_bytes).map_err(|e| DsperseError::io(e, &metadata_path))?;

    let msgpack_src = slices_dir.join("metadata.msgpack");
    if msgpack_src.exists() {
        tracing::info!("metadata.msgpack already present in output directory");
    }

    tracing::info!(
        count = slice_dirs.len(),
        total_size_bytes = total_size,
        "packaging complete"
    );

    Ok(PackageResult {
        count: slice_dirs.len(),
        total_size,
    })
}

fn create_zip_archive(source_dir: &Path, archive_path: &Path) -> Result<()> {
    let file = fs::File::create(archive_path).map_err(|e| DsperseError::io(e, archive_path))?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::<zip::write::ExtendedFileOptions>::default()
        .compression_method(zip::CompressionMethod::Deflated);

    for entry in WalkDir::new(source_dir) {
        let entry = entry.map_err(|e| DsperseError::Other(e.to_string()))?;
        let abs_path = entry.path();
        let relative = abs_path
            .strip_prefix(source_dir)
            .map_err(|e| DsperseError::Other(e.to_string()))?;

        if relative.as_os_str().is_empty() {
            continue;
        }

        if abs_path.is_dir() {
            zip.add_directory(relative.to_string_lossy(), options.clone())
                .map_err(|e| DsperseError::Archive(e.to_string()))?;
        } else {
            zip.start_file(relative.to_string_lossy(), options.clone())
                .map_err(|e| DsperseError::Archive(e.to_string()))?;
            let mut f = fs::File::open(abs_path).map_err(|e| DsperseError::io(e, abs_path))?;
            std::io::copy(&mut f, &mut zip).map_err(|e| DsperseError::io(e, abs_path))?;
        }
    }

    zip.finish()
        .map_err(|e| DsperseError::Archive(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_slice_dirs(tmp: &TempDir, count: usize) -> std::path::PathBuf {
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        for i in 0..count {
            let slice_dir = slices_dir.join(format!("slice_{}", i));
            let payload_dir = slice_dir.join("payload");
            fs::create_dir_all(&payload_dir).unwrap();
            fs::write(
                payload_dir.join(format!("slice_{}.onnx", i)),
                format!("onnx_data_{}", i),
            )
            .unwrap();

            let circuit_dir = slice_dir.join("jstprove").join("circuit.bundle");
            fs::create_dir_all(&circuit_dir).unwrap();
            fs::write(circuit_dir.join("circuit.bin"), format!("circuit_{}", i)).unwrap();
        }

        slices_dir
    }

    #[test]
    fn test_package_creates_archives() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = setup_slice_dirs(&tmp, 3);

        let result = package_slices(&slices_dir, false).unwrap();

        assert_eq!(result.count, 3);
        assert!(result.total_size > 0);

        for i in 0..3 {
            let archive = slices_dir.join(format!("slice_{}.dslice", i));
            assert!(archive.exists(), "archive slice_{}.dslice should exist", i);
        }

        for i in 0..3 {
            let dir = slices_dir.join(format!("slice_{}", i));
            assert!(dir.exists(), "slice dir should still exist with no cleanup");
        }
    }

    #[test]
    fn test_package_cleanup_removes_dirs() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = setup_slice_dirs(&tmp, 2);

        package_slices(&slices_dir, true).unwrap();

        for i in 0..2 {
            let dir = slices_dir.join(format!("slice_{}", i));
            assert!(!dir.exists(), "slice dir should be removed after cleanup");
        }

        for i in 0..2 {
            let archive = slices_dir.join(format!("slice_{}.dslice", i));
            assert!(archive.exists());
        }
    }

    #[test]
    fn test_metadata_json_generated() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = setup_slice_dirs(&tmp, 2);

        package_slices(&slices_dir, false).unwrap();

        let metadata_path = slices_dir.join("metadata.json");
        assert!(metadata_path.exists());

        let content = fs::read_to_string(&metadata_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(parsed["name"], "model");
        let dslices = parsed["dslices"].as_array().unwrap();
        assert_eq!(dslices.len(), 2);
        assert_eq!(dslices[0], "slice_0.dslice");
        assert_eq!(dslices[1], "slice_1.dslice");
    }

    #[test]
    fn test_zip_contains_correct_paths() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = setup_slice_dirs(&tmp, 1);

        package_slices(&slices_dir, false).unwrap();

        let archive_path = slices_dir.join("slice_0.dslice");
        let file = fs::File::open(&archive_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        let mut names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .filter(|n| !n.ends_with('/'))
            .collect();
        names.sort();

        assert!(names.contains(&"jstprove/circuit.bundle/circuit.bin".to_string()));
        assert!(names.contains(&"payload/slice_0.onnx".to_string()));
    }

    #[test]
    fn test_package_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        let result = package_slices(&slices_dir, false).unwrap();
        assert_eq!(result.count, 0);
        assert_eq!(result.total_size, 0);

        let metadata_path = slices_dir.join("metadata.json");
        assert!(metadata_path.exists());
        let parsed: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&metadata_path).unwrap()).unwrap();
        assert!(parsed["dslices"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_package_nonexistent_dir() {
        let result = package_slices(Path::new("/nonexistent/path"), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_numeric_sorting() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        for i in [10, 2, 0, 1, 20] {
            let d = slices_dir.join(format!("slice_{}", i));
            fs::create_dir_all(d.join("payload")).unwrap();
            fs::write(d.join("payload").join("data.bin"), "x").unwrap();
        }

        package_slices(&slices_dir, false).unwrap();

        let metadata: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(slices_dir.join("metadata.json")).unwrap())
                .unwrap();
        let dslices: Vec<&str> = metadata["dslices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();

        assert_eq!(
            dslices,
            vec![
                "slice_0.dslice",
                "slice_1.dslice",
                "slice_2.dslice",
                "slice_10.dslice",
                "slice_20.dslice",
            ]
        );
    }
}
