use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use zip::write::SimpleFileOptions;

use crate::error::{DsperseError, Result};

const EXTRACTED_SENTINEL: &str = ".extracted";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatType {
    Dirs,
    Dslice,
    Dsperse,
}

impl FormatType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dirs => "dirs",
            Self::Dslice => "dslice",
            Self::Dsperse => "dsperse",
        }
    }
}

impl std::str::FromStr for FormatType {
    type Err = DsperseError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "dirs" => Ok(Self::Dirs),
            "dslice" => Ok(Self::Dslice),
            "dsperse" => Ok(Self::Dsperse),
            _ => Err(DsperseError::Archive(format!("unknown format: {s}"))),
        }
    }
}

pub fn detect_type(path: &Path) -> Result<FormatType> {
    if path.is_file() {
        return match path.extension().and_then(|e| e.to_str()) {
            Some("dsperse") => Ok(FormatType::Dsperse),
            Some("dslice") => Ok(FormatType::Dslice),
            other => Err(DsperseError::Archive(format!(
                "unknown file extension: {}",
                other.unwrap_or("none")
            ))),
        };
    }

    if path.is_dir() {
        if has_slice_dirs(path) || is_slice_dir(path) {
            return Ok(FormatType::Dirs);
        }
        if has_dslice_files(path) {
            return Ok(FormatType::Dslice);
        }
        return Err(DsperseError::Archive(format!(
            "cannot determine format of directory: {}",
            path.display()
        )));
    }

    Err(DsperseError::Archive(format!(
        "path does not exist: {}",
        path.display()
    )))
}

pub fn convert(
    path: &Path,
    output_type: FormatType,
    output_path: Option<&Path>,
    cleanup: bool,
) -> Result<PathBuf> {
    if !path.exists() {
        return Err(DsperseError::Archive(format!(
            "path does not exist: {}",
            path.display()
        )));
    }

    let current = detect_type(path)?;
    if current == output_type {
        tracing::info!("already in desired format: {}", output_type.as_str());
        return Ok(path.to_path_buf());
    }

    tracing::info!(
        "converting from {} to {}",
        current.as_str(),
        output_type.as_str()
    );

    match (current, output_type) {
        (FormatType::Dirs, FormatType::Dslice) => dirs_to_dslice(path, output_path, cleanup),
        (FormatType::Dirs, FormatType::Dsperse) => dirs_to_dsperse(path, output_path, cleanup),
        (FormatType::Dslice, FormatType::Dirs) => dslice_to_dirs(path, output_path, cleanup),
        (FormatType::Dslice, FormatType::Dsperse) => {
            let temp = tempfile::tempdir()
                .map_err(|e| DsperseError::io(e, path))?;
            let expanded = dslice_to_dirs(path, Some(temp.path()), false)?;
            let result = dirs_to_dsperse(&expanded, output_path, true)?;
            if cleanup {
                if path.is_file() {
                    let _ = fs::remove_file(path);
                } else if path.is_dir() {
                    let _ = fs::remove_dir_all(path);
                }
            }
            Ok(result)
        }
        (FormatType::Dsperse, FormatType::Dirs) => {
            let result = dsperse_to_dirs(path, output_path, true)?;
            if cleanup {
                let _ = fs::remove_file(path);
            }
            Ok(result)
        }
        (FormatType::Dsperse, FormatType::Dslice) => {
            let result = dsperse_to_dirs(path, output_path, false)?;
            if cleanup {
                let _ = fs::remove_file(path);
            }
            Ok(result)
        }
        (FormatType::Dirs, FormatType::Dirs)
        | (FormatType::Dslice, FormatType::Dslice)
        | (FormatType::Dsperse, FormatType::Dsperse) => unreachable!(),
    }
}

pub fn dirs_to_dslice(path: &Path, output_path: Option<&Path>, cleanup: bool) -> Result<PathBuf> {
    let slice_dirs = find_slice_dirs(path);

    if slice_dirs.is_empty() {
        if is_slice_dir(path) {
            let dslice_out = if let Some(out) = output_path {
                if out.is_dir() || out.extension().is_none() {
                    out.join(format!("{}.dslice", path_name(path)?))
                } else {
                    out.to_path_buf()
                }
            } else {
                path.parent()
                    .unwrap_or(path)
                    .join(format!("{}.dslice", path_name(path)?))
            };
            ensure_parent(&dslice_out)?;
            zip_directory(path, &dslice_out, &[])?;
            return Ok(dslice_out);
        }
        return Err(DsperseError::Archive(format!(
            "no slice_* directories found in {}",
            path.display()
        )));
    }

    let output_dir = output_path.map(PathBuf::from).unwrap_or(path.to_path_buf());
    fs::create_dir_all(&output_dir).map_err(|e| DsperseError::io(e, &output_dir))?;

    for slice_dir in &slice_dirs {
        let dslice_path = output_dir.join(format!("{}.dslice", path_name(slice_dir)?));
        zip_directory(slice_dir, &dslice_path, &[])?;
        tracing::info!("created {}", dslice_path.display());
    }

    if cleanup {
        for slice_dir in &slice_dirs {
            let _ = fs::remove_dir_all(slice_dir);
        }
    }

    Ok(output_dir)
}

pub fn dirs_to_dsperse(path: &Path, output_path: Option<&Path>, cleanup: bool) -> Result<PathBuf> {
    let slice_dirs = find_slice_dirs(path);
    if slice_dirs.is_empty() {
        return Err(DsperseError::Archive(format!(
            "no slice_* directories found in {}",
            path.display()
        )));
    }

    let mut dslice_files = Vec::new();
    for slice_dir in &slice_dirs {
        let dslice_out = path.join(format!("{}.dslice", path_name(slice_dir)?));
        zip_directory(slice_dir, &dslice_out, &[])?;
        dslice_files.push(dslice_out);
    }

    let dsperse_out = if let Some(out) = output_path {
        if out.is_dir() || out.extension().is_none() {
            out.join(format!("{}.dsperse", path_name(path)?))
        } else {
            out.to_path_buf()
        }
    } else {
        path.parent()
            .unwrap_or(path)
            .join(format!("{}.dsperse", path_name(path)?))
    };
    ensure_parent(&dsperse_out)?;

    zip_directory(path, &dsperse_out, &["slice_"])?;

    verify_archive(&dsperse_out)?;

    for f in &dslice_files {
        let _ = fs::remove_file(f);
    }
    if cleanup {
        for d in &slice_dirs {
            let _ = fs::remove_dir_all(d);
        }
        let metadata_file = path.join("metadata.json");
        if metadata_file.exists() {
            let _ = fs::remove_file(&metadata_file);
        }
        let _ = fs::remove_dir(path);
    }

    Ok(dsperse_out)
}

pub fn dslice_to_dirs(path: &Path, output_path: Option<&Path>, cleanup: bool) -> Result<PathBuf> {
    if path.is_file() {
        let output_dir = match output_path {
            Some(out) => PathBuf::from(out),
            None => path.parent().unwrap_or(path).join(path_stem(path)?),
        };
        fs::create_dir_all(&output_dir).map_err(|e| DsperseError::io(e, &output_dir))?;
        unzip_file(path, &output_dir)?;
        if cleanup {
            let _ = fs::remove_file(path);
        }
        return Ok(output_dir);
    }

    if path.is_dir() {
        let dslice_files = find_dslice_files(path);
        if dslice_files.is_empty() {
            return Err(DsperseError::Archive(format!(
                "no .dslice files found in {}",
                path.display()
            )));
        }
        let output_dir = output_path.map(PathBuf::from).unwrap_or(path.to_path_buf());

        for dslice_file in &dslice_files {
            let slice_dir = output_dir.join(path_stem(dslice_file)?);
            fs::create_dir_all(&slice_dir).map_err(|e| DsperseError::io(e, &slice_dir))?;
            unzip_file(dslice_file, &slice_dir)?;

            if cleanup {
                let _ = fs::remove_file(dslice_file);
            }
        }
        return Ok(output_dir);
    }

    Err(DsperseError::Archive(format!(
        "invalid dslice path: {}",
        path.display()
    )))
}

pub fn dsperse_to_dirs(
    path: &Path,
    output_path: Option<&Path>,
    expand_slices: bool,
) -> Result<PathBuf> {
    if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("dsperse") {
        return Err(DsperseError::Archive(format!(
            "expected .dsperse file, got {}",
            path.display()
        )));
    }

    let output_dir = match output_path {
        Some(out) => PathBuf::from(out),
        None => path.parent().unwrap_or(path).join(path_stem(path)?),
    };
    fs::create_dir_all(&output_dir).map_err(|e| DsperseError::io(e, &output_dir))?;

    unzip_file(path, &output_dir)?;

    if expand_slices {
        let dslice_files = find_dslice_files(&output_dir);
        for dslice_file in &dslice_files {
            let slice_dir = output_dir.join(path_stem(dslice_file)?);
            fs::create_dir_all(&slice_dir).map_err(|e| DsperseError::io(e, &slice_dir))?;
            unzip_file(dslice_file, &slice_dir)?;
            let _ = fs::remove_file(dslice_file);
        }
    }

    Ok(output_dir)
}

pub fn extract_metadata_only(archive_path: &Path, output_dir: Option<&Path>) -> Result<PathBuf> {
    if !archive_path.is_file() || archive_path.extension().and_then(|e| e.to_str()) != Some("dsperse") {
        return Err(DsperseError::Archive(format!(
            "expected .dsperse file, got {}",
            archive_path.display()
        )));
    }

    let out = match output_dir {
        Some(dir) => PathBuf::from(dir),
        None => archive_path
            .parent()
            .unwrap_or(archive_path)
            .join(path_stem(archive_path)?),
    };
    fs::create_dir_all(&out).map_err(|e| DsperseError::io(e, &out))?;

    let file =
        fs::File::open(archive_path).map_err(|e| DsperseError::io(e, archive_path))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| DsperseError::Archive(e.to_string()))?;

    let mut found = false;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| DsperseError::Archive(e.to_string()))?;
        let name = entry.name().to_string();
        if name == "metadata.json" || name.ends_with("/metadata.json") {
            let dest = out.join("metadata.json");
            let mut out_file =
                fs::File::create(&dest).map_err(|e| DsperseError::io(e, &dest))?;
            io::copy(&mut entry, &mut out_file)
                .map_err(|e| DsperseError::io(e, &dest))?;
            found = true;
            break;
        }
    }

    if !found {
        return Err(DsperseError::Archive(format!(
            "no metadata.json found in {}",
            archive_path.display()
        )));
    }

    Ok(out)
}

pub fn extract_single_slice(
    archive_path: &Path,
    slice_id: &str,
    output_dir: Option<&Path>,
) -> Result<PathBuf> {
    validate_slice_id(slice_id)?;
    let dslice_name = format!("{slice_id}.dslice");

    let out = match output_dir {
        Some(dir) => PathBuf::from(dir),
        None if archive_path.is_file() => archive_path
            .parent()
            .unwrap_or(archive_path)
            .join(path_stem(archive_path)?),
        None => archive_path.to_path_buf(),
    };
    fs::create_dir_all(&out).map_err(|e| DsperseError::io(e, &out))?;

    let slice_dir = out.join(slice_id);
    let sentinel = slice_dir.join(EXTRACTED_SENTINEL);
    if sentinel.exists() && slice_dir.join("payload").exists() {
        return Ok(slice_dir);
    }

    if slice_dir.exists() {
        let _ = fs::remove_dir_all(&slice_dir);
    }

    if archive_path.is_file()
        && archive_path.extension().and_then(|e| e.to_str()) == Some("dsperse")
    {
        let file =
            fs::File::open(archive_path).map_err(|e| DsperseError::io(e, archive_path))?;
        let mut archive =
            zip::ZipArchive::new(file).map_err(|e| DsperseError::Archive(e.to_string()))?;

        let mut dslice_data = None;
        for i in 0..archive.len() {
            let mut entry = archive
                .by_index(i)
                .map_err(|e| DsperseError::Archive(e.to_string()))?;
            let name = entry.name().to_string();
            if name == dslice_name || name.ends_with(&format!("/{dslice_name}")) {
                let mut buf = Vec::new();
                entry
                    .read_to_end(&mut buf)
                    .map_err(|e| DsperseError::io(e, archive_path))?;
                dslice_data = Some(buf);
                break;
            }
        }

        let data = dslice_data.ok_or_else(|| {
            DsperseError::Archive(format!(
                "slice {slice_id} not found in {}",
                archive_path.display()
            ))
        })?;

        fs::create_dir_all(&slice_dir).map_err(|e| DsperseError::io(e, &slice_dir))?;
        let cursor = io::Cursor::new(data);
        let mut inner =
            zip::ZipArchive::new(cursor).map_err(|e| DsperseError::Archive(e.to_string()))?;
        if let Err(e) = inner.extract(&slice_dir) {
            let _ = fs::remove_dir_all(&slice_dir);
            return Err(DsperseError::Archive(e.to_string()));
        }
    } else if archive_path.is_dir() {
        let dslice_file = archive_path.join(&dslice_name);
        let dslice_file = if dslice_file.exists() {
            dslice_file
        } else {
            let alt = out.join(&dslice_name);
            if alt.exists() {
                alt
            } else {
                return Err(DsperseError::Archive(format!(
                    "slice file {dslice_name} not found in {}",
                    archive_path.display()
                )));
            }
        };
        fs::create_dir_all(&slice_dir).map_err(|e| DsperseError::io(e, &slice_dir))?;
        if let Err(e) = unzip_file(&dslice_file, &slice_dir) {
            let _ = fs::remove_dir_all(&slice_dir);
            return Err(e);
        }
    } else {
        return Err(DsperseError::Archive(format!(
            "cannot extract slice from {}",
            archive_path.display()
        )));
    }

    write_sentinel(&sentinel)?;

    Ok(slice_dir)
}

pub fn cleanup_extracted_slice(slices_dir: &Path, slice_id: &str) {
    if validate_slice_id(slice_id).is_err() {
        return;
    }
    let slice_dir = slices_dir.join(slice_id);
    if slice_dir.exists() && slice_dir.is_dir() {
        let _ = fs::remove_dir_all(&slice_dir);
    }
}

pub fn read_dslice_slice_metadata(
    dslice_path: &Path,
) -> Result<crate::schema::metadata::SliceMetadata> {
    let file = fs::File::open(dslice_path).map_err(|e| DsperseError::io(e, dslice_path))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| DsperseError::Archive(e.to_string()))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| DsperseError::Archive(e.to_string()))?;
        if entry.name() == "metadata.json" {
            let mut buf = String::new();
            entry
                .read_to_string(&mut buf)
                .map_err(|e| DsperseError::io(e, dslice_path))?;
            let model_meta: crate::schema::metadata::ModelMetadata =
                serde_json::from_str(&buf)?;
            return model_meta.slices.into_iter().next().ok_or_else(|| {
                DsperseError::Metadata(format!(
                    "no slices in metadata inside {}",
                    dslice_path.display()
                ))
            });
        }
    }

    Err(DsperseError::Metadata(format!(
        "no metadata.json found in {}",
        dslice_path.display()
    )))
}

fn validate_slice_id(slice_id: &str) -> Result<()> {
    if slice_id.contains('/') || slice_id.contains('\\') || slice_id.contains("..") || slice_id.is_empty() {
        return Err(DsperseError::Archive(format!(
            "invalid slice_id contains path separators or traversal: {slice_id:?}"
        )));
    }
    Ok(())
}

fn write_sentinel(path: &Path) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, b"").map_err(|e| DsperseError::io(e, &tmp))?;
    fs::rename(&tmp, path).map_err(|e| DsperseError::io(e, path))?;
    Ok(())
}

fn verify_archive(path: &Path) -> Result<()> {
    let file = fs::File::open(path).map_err(|e| DsperseError::io(e, path))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| DsperseError::Archive(format!("archive verification: {e}")))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| DsperseError::Archive(format!("archive entry {i}: {e}")))?;
        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .map_err(|e| DsperseError::Archive(format!("read entry {}: {e}", entry.name())))?;
    }
    Ok(())
}

fn zip_directory(source: &Path, output: &Path, exclude_dir_prefixes: &[&str]) -> Result<()> {
    let file = fs::File::create(output).map_err(|e| DsperseError::io(e, output))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    for entry in walkdir::WalkDir::new(source).into_iter().filter_map(|e| match e {
        Ok(entry) => Some(entry),
        Err(err) => {
            tracing::warn!(path = ?err.path(), error = %err, "skipping unreadable entry during archive");
            None
        }
    }) {
        let entry_path = entry.path();
        let rel = entry_path
            .strip_prefix(source)
            .unwrap_or(entry_path);

        if rel.as_os_str().is_empty() {
            continue;
        }

        let rel_str = rel.to_str().ok_or_else(|| {
            DsperseError::Archive(format!("non-UTF-8 path: {}", rel.display()))
        })?;

        let first_component = rel
            .components()
            .next()
            .and_then(|c| c.as_os_str().to_str())
            .unwrap_or_default();
        if entry_path.is_dir()
            && exclude_dir_prefixes
                .iter()
                .any(|p| first_component.starts_with(p))
        {
            continue;
        }
        if !entry_path.is_dir()
            && exclude_dir_prefixes
                .iter()
                .any(|p| first_component.starts_with(p) && rel.components().count() > 1)
        {
            continue;
        }

        if entry_path.is_dir() {
            let dir_name = format!("{rel_str}/");
            zip.add_directory(&dir_name, options)
                .map_err(|e| DsperseError::Archive(e.to_string()))?;
        } else if entry_path.is_file() {
            zip.start_file(rel_str, options)
                .map_err(|e| DsperseError::Archive(e.to_string()))?;
            let mut f =
                fs::File::open(entry_path).map_err(|e| DsperseError::io(e, entry_path))?;
            io::copy(&mut f, &mut zip).map_err(|e| DsperseError::io(e, entry_path))?;
        }
    }

    zip.finish()
        .map_err(|e| DsperseError::Archive(e.to_string()))?;
    Ok(())
}

fn unzip_file(zip_path: &Path, output_dir: &Path) -> Result<()> {
    let file = fs::File::open(zip_path).map_err(|e| DsperseError::io(e, zip_path))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| DsperseError::Archive(e.to_string()))?;
    archive
        .extract(output_dir)
        .map_err(|e| DsperseError::Archive(e.to_string()))?;
    Ok(())
}

fn has_slice_dirs(path: &Path) -> bool {
    path.is_dir()
        && fs::read_dir(path)
            .ok()
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path().is_dir()
                        && e.file_name()
                            .to_str()
                            .is_some_and(|n| n.starts_with("slice_"))
                })
            })
            .unwrap_or(false)
}

fn is_slice_dir(path: &Path) -> bool {
    path.is_dir() && path.join("metadata.json").exists() && path.join("payload").exists()
}

fn has_dslice_files(path: &Path) -> bool {
    path.is_dir()
        && fs::read_dir(path)
            .ok()
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path().is_file()
                        && e.path()
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .is_some_and(|ext| ext == "dslice")
                })
            })
            .unwrap_or(false)
}

fn find_slice_dirs(path: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().is_dir()
                        && e.file_name()
                            .to_str()
                            .is_some_and(|n| n.starts_with("slice_"))
                })
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    dirs
}

pub fn find_dslice_files(path: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().is_file()
                        && e.path()
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .is_some_and(|ext| ext == "dslice")
                })
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

fn path_name(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| DsperseError::Archive(format!("missing file name for path: {}", path.display())))
}

fn path_stem(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| DsperseError::Archive(format!("missing file stem for path: {}", path.display())))
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_dirs_to_dslice_to_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path().join("slices");

        for i in 0..2 {
            let slice_dir = slices_dir.join(format!("slice_{i}"));
            let payload_dir = slice_dir.join("payload");
            fs::create_dir_all(&payload_dir).unwrap();
            fs::write(
                slice_dir.join("metadata.json"),
                format!(r#"{{"index": {i}}}"#),
            )
            .unwrap();
            fs::write(payload_dir.join("model.onnx"), b"fake onnx data").unwrap();
        }
        fs::write(
            slices_dir.join("metadata.json"),
            r#"{"original_model": "test.onnx"}"#,
        )
        .unwrap();

        assert_eq!(detect_type(&slices_dir).unwrap(), FormatType::Dirs);

        let dslice_out = tmp.path().join("dslice_output");
        let result = dirs_to_dslice(&slices_dir, Some(&dslice_out), false).unwrap();
        assert!(result.join("slice_0.dslice").exists());
        assert!(result.join("slice_1.dslice").exists());
        assert_eq!(detect_type(&result).unwrap(), FormatType::Dslice);

        let dirs_out = tmp.path().join("dirs_output");
        let result2 = dslice_to_dirs(&result, Some(&dirs_out), false).unwrap();
        assert!(result2.join("slice_0").join("metadata.json").exists());
        assert!(result2.join("slice_1").join("payload").join("model.onnx").exists());
    }

    #[test]
    fn roundtrip_dirs_to_dsperse_to_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path().join("slices");

        for i in 0..2 {
            let slice_dir = slices_dir.join(format!("slice_{i}"));
            let payload_dir = slice_dir.join("payload");
            fs::create_dir_all(&payload_dir).unwrap();
            fs::write(
                slice_dir.join("metadata.json"),
                format!(r#"{{"index": {i}}}"#),
            )
            .unwrap();
            fs::write(payload_dir.join("model.onnx"), b"fake onnx data").unwrap();
        }
        fs::write(
            slices_dir.join("metadata.json"),
            r#"{"original_model": "test.onnx"}"#,
        )
        .unwrap();

        let dsperse_file = tmp.path().join("test.dsperse");
        let result = dirs_to_dsperse(&slices_dir, Some(&dsperse_file), false).unwrap();
        assert!(result.exists());
        assert_eq!(detect_type(&result).unwrap(), FormatType::Dsperse);

        let dirs_out = tmp.path().join("expanded");
        let result2 = dsperse_to_dirs(&result, Some(&dirs_out), true).unwrap();
        assert!(result2.join("slice_0").join("metadata.json").exists());
        assert!(result2.join("slice_1").join("payload").join("model.onnx").exists());
        assert!(result2.join("metadata.json").exists());

        let meta_content = fs::read_to_string(result2.join("metadata.json")).unwrap();
        assert!(meta_content.contains("test.onnx"));
    }

    #[test]
    fn extract_single_slice_from_dsperse() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path().join("slices");

        for i in 0..3 {
            let slice_dir = slices_dir.join(format!("slice_{i}"));
            let payload_dir = slice_dir.join("payload");
            fs::create_dir_all(&payload_dir).unwrap();
            fs::write(
                slice_dir.join("metadata.json"),
                format!(r#"{{"index": {i}}}"#),
            )
            .unwrap();
            fs::write(
                payload_dir.join("model.onnx"),
                format!("onnx data for slice {i}"),
            )
            .unwrap();
        }
        fs::write(slices_dir.join("metadata.json"), "{}").unwrap();

        let dsperse_file = tmp.path().join("test.dsperse");
        dirs_to_dsperse(&slices_dir, Some(&dsperse_file), false).unwrap();

        let extract_dir = tmp.path().join("extracted");
        let slice_dir =
            extract_single_slice(&dsperse_file, "slice_1", Some(&extract_dir)).unwrap();
        assert!(slice_dir.join("metadata.json").exists());
        assert!(slice_dir.join("payload").join("model.onnx").exists());
        assert!(slice_dir.join(EXTRACTED_SENTINEL).exists());

        let onnx_data = fs::read_to_string(slice_dir.join("payload").join("model.onnx")).unwrap();
        assert_eq!(onnx_data, "onnx data for slice 1");

        let slice_dir2 =
            extract_single_slice(&dsperse_file, "slice_1", Some(&extract_dir)).unwrap();
        assert_eq!(slice_dir, slice_dir2);
    }

    #[test]
    fn extract_metadata_only_from_dsperse() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path().join("slices");

        let slice_dir = slices_dir.join("slice_0");
        let payload_dir = slice_dir.join("payload");
        fs::create_dir_all(&payload_dir).unwrap();
        fs::write(slice_dir.join("metadata.json"), r#"{"index": 0}"#).unwrap();
        fs::write(payload_dir.join("model.onnx"), b"data").unwrap();
        fs::write(
            slices_dir.join("metadata.json"),
            r#"{"original_model": "test.onnx"}"#,
        )
        .unwrap();

        let dsperse_file = tmp.path().join("test.dsperse");
        dirs_to_dsperse(&slices_dir, Some(&dsperse_file), false).unwrap();

        let out_dir = tmp.path().join("meta_only");
        let result = extract_metadata_only(&dsperse_file, Some(&out_dir)).unwrap();
        assert!(result.join("metadata.json").exists());
        assert!(!result.join("slice_0").exists());
    }
}
