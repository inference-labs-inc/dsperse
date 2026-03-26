use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::error::{DsperseError, Result};
use crate::pipeline::runner::load_model_metadata;
use crate::schema::metadata::SliceMetadata;
use crate::utils::paths::resolve_relative_path;

pub struct PackageConfig {
    pub output_dir: PathBuf,
    pub cleanup: bool,
    pub author: Option<String>,
    pub model_version: Option<String>,
    pub model_name: Option<String>,
    pub timeout: Option<u64>,
    pub curve: Option<String>,
}

#[derive(Debug)]
pub struct PackageResult {
    pub component_count: usize,
    pub wb_count: usize,
    pub manifest_path: PathBuf,
    pub total_size: u64,
}

#[derive(Serialize)]
struct Manifest {
    version: u32,
    model: ModelInfo,
    components: Vec<ComponentEntry>,
    dag: Vec<DagNode>,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    curve: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<u64>,
    input_schema: InputSchema,
    #[serde(skip_serializing_if = "Option::is_none")]
    dsperse_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    jstprove_version: Option<String>,
}

#[derive(Serialize)]
struct InputSchema {
    shape: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    output_names: Vec<String>,
}

#[derive(Serialize)]
struct ComponentEntry {
    index: usize,
    name: String,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    curve: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_system: Option<String>,
    files: Vec<String>,
    weights: Vec<WeightRef>,
}

#[derive(Serialize)]
struct WeightRef {
    sha256: String,
    role: String,
    filename: String,
    size_bytes: u64,
}

#[derive(Serialize)]
struct DagNode {
    component_index: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
    input_shape: Vec<Vec<i64>>,
    output_shape: Vec<Vec<i64>>,
}

const VALID_CURVES: &[&str] = &["bn254", "goldilocks", "goldilocks_basefold"];

fn normalize_curve(curve: Option<&str>) -> Result<Option<String>> {
    let Some(c) = curve else { return Ok(None) };
    let c = c.trim().to_ascii_lowercase();
    if c.is_empty() {
        return Err(DsperseError::Other("curve must not be empty".into()));
    }
    if !VALID_CURVES.contains(&c.as_str()) {
        return Err(DsperseError::Other(format!(
            "unsupported curve {c:?}; expected one of: {}",
            VALID_CURVES.join(", ")
        )));
    }
    Ok(Some(c))
}

pub fn package_content_addressed(
    slices_dir: &Path,
    config: &PackageConfig,
) -> Result<PackageResult> {
    if !slices_dir.is_dir() {
        return Err(DsperseError::Other(format!(
            "slices directory not found: {}",
            slices_dir.display()
        )));
    }

    let curve = normalize_curve(config.curve.as_deref())?;

    if config.cleanup {
        validate_output_dir_not_under_slice(&config.output_dir)?;
    }

    let model_meta = load_model_metadata(slices_dir)?;

    let components_dir = config.output_dir.join("components");
    let wb_dir = config.output_dir.join("wb");
    fs::create_dir_all(&components_dir).map_err(|e| DsperseError::io(e, &components_dir))?;
    fs::create_dir_all(&wb_dir).map_err(|e| DsperseError::io(e, &wb_dir))?;

    let mut components: Vec<ComponentEntry> = Vec::new();
    let mut dag_nodes: Vec<DagNode> = Vec::new();
    let mut written_components: HashSet<String> = HashSet::new();
    let mut written_wbs: HashSet<String> = HashSet::new();
    let mut total_size: u64 = 0;

    for slice in &model_meta.slices {
        let slice_dir = slices_dir.join(format!("slice_{}", slice.index));

        let (component_hash, component_files, proof_system, source) =
            extract_component(slices_dir, slice, &slice_dir, curve.as_deref())?;

        if !written_components.contains(&component_hash) {
            let dest = components_dir.join(&component_hash);
            fs::create_dir_all(&dest).map_err(|e| DsperseError::io(e, &dest))?;

            match &source {
                ComponentSource::CircuitBundle(circuit_dir) => {
                    total_size += copy_files_flat(circuit_dir, &dest)?;
                }
                ComponentSource::OnnxFile(onnx_path) => {
                    if let Some(filename) = component_files.first() {
                        let dest_file = dest.join(filename);
                        fs::copy(onnx_path, &dest_file)
                            .map_err(|e| DsperseError::io(e, onnx_path))?;
                        total_size += onnx_path
                            .metadata()
                            .map_err(|e| DsperseError::io(e, onnx_path))?
                            .len();
                    }
                }
            }
            written_components.insert(component_hash.clone());
        }

        let mut weights: Vec<WeightRef> = Vec::new();
        let payload_blobs = collect_payload_blobs(slices_dir, slice, &slice_dir)?;
        for (role, filename, data) in &payload_blobs {
            let hash = sha256_bytes(data);
            if !written_wbs.contains(&hash) {
                let wb_path = wb_dir.join(&hash);
                fs::write(&wb_path, data).map_err(|e| DsperseError::io(e, &wb_path))?;
                total_size += data.len() as u64;
                written_wbs.insert(hash.clone());
            }
            weights.push(WeightRef {
                sha256: hash,
                role: role.clone(),
                filename: filename.clone(),
                size_bytes: data.len() as u64,
            });
        }

        components.push(ComponentEntry {
            index: slice.index,
            name: format!("slice_{}", slice.index),
            sha256: component_hash,
            curve: curve.clone(),
            proof_system,
            files: component_files,
            weights,
        });

        dag_nodes.push(DagNode {
            component_index: slice.index,
            inputs: slice.dependencies.input.clone(),
            outputs: slice.dependencies.output.clone(),
            input_shape: slice.shape.tensor_shape.input.clone(),
            output_shape: slice.shape.tensor_shape.output.clone(),
        });

        if (slice.index + 1) % 50 == 0 {
            tracing::info!(
                progress = slice.index + 1,
                total = model_meta.slices.len(),
                "packaging slices"
            );
        }
    }

    let model_name = config
        .model_name
        .clone()
        .or_else(|| {
            slices_dir
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .map(String::from)
        })
        .unwrap_or_else(|| "unknown".to_string());

    let manifest = Manifest {
        version: 1,
        model: ModelInfo {
            name: model_name,
            curve: curve.clone(),
            author: config.author.clone(),
            version: config.model_version.clone(),
            timeout: config.timeout,
            input_schema: InputSchema {
                shape: model_meta.input_shape,
                output_shapes: model_meta.output_shapes,
                output_names: model_meta.output_names,
            },
            dsperse_version: model_meta.dsperse_version,
            jstprove_version: model_meta.jstprove_version,
        },
        components,
        dag: dag_nodes,
    };

    let manifest_path = config.output_dir.join("manifest.msgpack");
    let manifest_bytes = rmp_serde::to_vec_named(&manifest)
        .map_err(|e| DsperseError::Other(format!("failed to serialize manifest: {e}")))?;
    fs::write(&manifest_path, &manifest_bytes).map_err(|e| DsperseError::io(e, &manifest_path))?;
    total_size += manifest_bytes.len() as u64;

    if config.cleanup {
        for slice in &model_meta.slices {
            let slice_dir = slices_dir.join(format!("slice_{}", slice.index));
            if slice_dir.is_dir() {
                fs::remove_dir_all(&slice_dir).map_err(|e| DsperseError::io(e, &slice_dir))?;
            }
        }
    }

    Ok(PackageResult {
        component_count: written_components.len(),
        wb_count: written_wbs.len(),
        manifest_path,
        total_size,
    })
}

fn resolve_circuit_dir(slices_dir: &Path, slice: &SliceMetadata) -> Result<Option<PathBuf>> {
    if let Some(ref compiled_path) = slice.compilation.jstprove.files.compiled {
        let abs = resolve_relative_path(slices_dir, compiled_path)?;
        if abs.is_dir() {
            return Ok(Some(abs));
        }
    }
    if let Some(ref cs) = slice.channel_split
        && let Some(group) = cs.groups.first()
        && let Some(ref circuit_path) = group.jstprove_circuit_path
    {
        let abs = resolve_relative_path(slices_dir, circuit_path)?;
        if abs.is_dir() {
            return Ok(Some(abs));
        }
    }
    Ok(None)
}

enum ComponentSource {
    CircuitBundle(PathBuf),
    OnnxFile(PathBuf),
}

fn extract_component(
    slices_dir: &Path,
    slice: &SliceMetadata,
    slice_dir: &Path,
    curve: Option<&str>,
) -> Result<(String, Vec<String>, Option<String>, ComponentSource)> {
    if slice.compilation.jstprove.compiled {
        let circuit_dir = resolve_circuit_dir(slices_dir, slice)?;
        return match circuit_dir {
            Some(dir) => {
                let (hash, files) = hash_directory(&dir, curve)?;
                Ok((
                    hash,
                    files,
                    Some("jstprove".to_string()),
                    ComponentSource::CircuitBundle(dir),
                ))
            }
            None => Err(DsperseError::Other(format!(
                "slice {} marked compiled but circuit directory not found",
                slice.index
            ))),
        };
    }

    let onnx_path = slice.resolve_onnx(slices_dir).unwrap_or_else(|_| {
        slice_dir
            .join("payload")
            .join(format!("slice_{}.onnx", slice.index))
    });
    reject_symlink_path(&onnx_path)?;
    if onnx_path.is_file() {
        let filename = onnx_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.onnx")
            .to_string();
        let hash = hash_named_file(&onnx_path, &filename, curve)?;
        return Ok((
            hash,
            vec![filename],
            None,
            ComponentSource::OnnxFile(onnx_path),
        ));
    }

    Err(DsperseError::Other(format!(
        "slice {} has no circuit directory or ONNX artifact to package",
        slice.index
    )))
}

fn collect_payload_blobs(
    slices_dir: &Path,
    slice: &SliceMetadata,
    slice_dir: &Path,
) -> Result<Vec<(String, String, Vec<u8>)>> {
    let mut blobs: Vec<(String, String, Vec<u8>)> = Vec::new();

    let onnx_path = slice.resolve_onnx(slices_dir).unwrap_or_else(|_| {
        slice_dir
            .join("payload")
            .join(format!("slice_{}.onnx", slice.index))
    });
    reject_symlink_path(&onnx_path)?;
    if onnx_path.is_file() {
        let data = fs::read(&onnx_path).map_err(|e| DsperseError::io(e, &onnx_path))?;
        let filename = onnx_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.onnx")
            .to_string();
        blobs.push(("payload".to_string(), filename, data));
    }

    if let Some(ref cs) = slice.channel_split {
        for group in &cs.groups {
            let group_path = resolve_relative_path(slices_dir, &group.path)?;
            reject_symlink_path(&group_path)?;
            if group_path.is_file() {
                let data = fs::read(&group_path).map_err(|e| DsperseError::io(e, &group_path))?;
                let filename = group_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("group.onnx")
                    .to_string();
                blobs.push(("channel_group".to_string(), filename, data));
            }
        }
        if let Some(ref bias_path) = cs.bias_path {
            let abs = resolve_relative_path(slices_dir, bias_path)?;
            reject_symlink_path(&abs)?;
            if abs.is_file() {
                let data = fs::read(&abs).map_err(|e| DsperseError::io(e, &abs))?;
                blobs.push(("bias".to_string(), "bias.msgpack".to_string(), data));
            }
        }
    }

    Ok(blobs)
}

fn validate_output_dir_not_under_slice(output_dir: &Path) -> Result<()> {
    for ancestor in output_dir.ancestors() {
        if let Some(name) = ancestor.file_name()
            && name.to_string_lossy().starts_with("slice_")
        {
            return Err(DsperseError::Other(format!(
                "output directory {} is inside a slice directory that would be removed by cleanup",
                output_dir.display()
            )));
        }
    }
    Ok(())
}

fn reject_symlink_path(path: &Path) -> Result<()> {
    if path
        .symlink_metadata()
        .is_ok_and(|m| m.file_type().is_symlink())
    {
        return Err(DsperseError::Other(format!(
            "symlinked file is not allowed: {}",
            path.display()
        )));
    }
    Ok(())
}

fn reject_symlink(entry: &walkdir::DirEntry) -> Result<()> {
    if entry.file_type().is_symlink() {
        return Err(DsperseError::Other(format!(
            "symlinked bundle entry is not allowed: {}",
            entry.path().display()
        )));
    }
    Ok(())
}

fn hash_named_file(path: &Path, filename: &str, curve: Option<&str>) -> Result<String> {
    let mut hasher = Sha256::new();
    if let Some(c) = curve {
        let c_bytes = c.as_bytes();
        hasher.update((c_bytes.len() as u64).to_le_bytes());
        hasher.update(c_bytes);
    }
    let name_bytes = filename.as_bytes();
    hasher.update((name_bytes.len() as u64).to_le_bytes());
    hasher.update(name_bytes);
    let mut file = fs::File::open(path).map_err(|e| DsperseError::io(e, path))?;
    let file_len = file
        .metadata()
        .map_err(|e| DsperseError::io(e, path))?
        .len();
    hasher.update(file_len.to_le_bytes());
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).map_err(|e| DsperseError::io(e, path))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(encode_hex(&hasher.finalize()))
}

fn hash_directory(dir: &Path, curve: Option<&str>) -> Result<(String, Vec<String>)> {
    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    for entry in WalkDir::new(dir) {
        let entry = entry.map_err(|e| DsperseError::Other(e.to_string()))?;
        reject_symlink(&entry)?;
        if entry.file_type().is_file() {
            let relative = entry
                .path()
                .strip_prefix(dir)
                .map_err(|e| DsperseError::Other(e.to_string()))?
                .to_string_lossy()
                .to_string();
            entries.push((relative, entry.path().to_path_buf()));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    if let Some(c) = curve {
        let c_bytes = c.as_bytes();
        hasher.update((c_bytes.len() as u64).to_le_bytes());
        hasher.update(c_bytes);
    }
    let file_names: Vec<String> = entries.iter().map(|(name, _)| name.clone()).collect();

    for (name, path) in &entries {
        let name_bytes = name.as_bytes();
        hasher.update((name_bytes.len() as u64).to_le_bytes());
        hasher.update(name_bytes);

        let mut file = fs::File::open(path).map_err(|e| DsperseError::io(e, path))?;
        let file_len = file
            .metadata()
            .map_err(|e| DsperseError::io(e, path))?
            .len();
        hasher.update(file_len.to_le_bytes());
        let mut buf = [0u8; 8192];
        loop {
            let n = file.read(&mut buf).map_err(|e| DsperseError::io(e, path))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }

    let hash = encode_hex(&hasher.finalize());
    Ok((hash, file_names))
}

fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    encode_hex(&hasher.finalize())
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        write!(s, "{:02x}", b).unwrap();
    }
    s
}

fn copy_files_flat(source_dir: &Path, dest_dir: &Path) -> Result<u64> {
    let mut total: u64 = 0;
    for entry in WalkDir::new(source_dir) {
        let entry = entry.map_err(|e| DsperseError::Other(e.to_string()))?;
        reject_symlink(&entry)?;
        if entry.file_type().is_file() {
            let relative = entry
                .path()
                .strip_prefix(source_dir)
                .map_err(|e| DsperseError::Other(e.to_string()))?;
            let dest_path = dest_dir.join(relative);
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
            }
            fs::copy(entry.path(), &dest_path).map_err(|e| DsperseError::io(e, entry.path()))?;
            total += entry
                .path()
                .metadata()
                .map_err(|e| DsperseError::io(e, entry.path()))?
                .len();
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    use crate::schema::metadata::{
        BackendCompilation, Compilation, CompilationFiles, Dependencies, ModelMetadata,
        SliceShapeWrapper, TensorShape,
    };

    fn create_test_model_metadata(slices_dir: &Path, count: usize) {
        let mut slices = Vec::new();
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
            fs::write(
                circuit_dir.join("settings.json"),
                format!("{{\"idx\":{}}}", i),
            )
            .unwrap();

            let inputs = if i == 0 {
                vec!["model_input".to_string()]
            } else {
                vec![format!("tensor_{}", i - 1)]
            };
            let outputs = vec![format!("tensor_{}", i)];

            slices.push(SliceMetadata {
                index: i,
                filename: format!("slice_{}.onnx", i),
                path: slice_dir.to_string_lossy().to_string(),
                relative_path: format!("slice_{}/payload/slice_{}.onnx", i, i),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![vec![1, 3, 224, 224]],
                        output: vec![vec![1, 64, 112, 112]],
                    },
                },
                dependencies: Dependencies {
                    input: inputs,
                    output: outputs,
                    filtered_inputs: vec![],
                },
                tiling: None,
                channel_split: None,
                compilation: Compilation {
                    jstprove: BackendCompilation {
                        compiled: true,
                        tiled: false,
                        weights_as_inputs: false,
                        files: CompilationFiles {
                            compiled: Some(format!("slice_{}/jstprove/circuit.bundle", i)),
                            settings: None,
                            pk_key: None,
                            vk_key: None,
                        },
                        compilation_timestamp: None,
                    },
                },
                slice_metadata: None,
                slice_metadata_relative_path: None,
            });
        }

        let meta = ModelMetadata {
            original_model: "test_model".to_string(),
            model_type: "onnx".to_string(),
            input_shape: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            output_names: vec!["output".to_string()],
            slice_points: (0..count).collect(),
            slices,
            dsperse_version: Some("0.0.1-test".to_string()),
            dsperse_rev: None,
            jstprove_version: Some("0.1.0-test".to_string()),
            jstprove_rev: None,
            traced_shapes: None,
            original_model_path: None,
        };

        meta.save(&slices_dir.join("metadata.msgpack")).unwrap();
    }

    #[test]
    fn test_content_addressed_output_structure() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 3);

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir: output_dir.clone(),
            cleanup: false,
            author: Some("test-author".to_string()),
            model_version: Some("1.0.0".to_string()),
            model_name: Some("test-model".to_string()),
            timeout: Some(300),
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config).unwrap();

        assert_eq!(result.component_count, 3);
        assert_eq!(result.wb_count, 3);
        assert!(result.total_size > 0);
        assert!(output_dir.join("components").is_dir());
        assert!(output_dir.join("wb").is_dir());
        assert!(output_dir.join("manifest.msgpack").is_file());
    }

    #[test]
    fn test_manifest_structure() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 2);

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir: output_dir.clone(),
            cleanup: false,
            author: Some("test-author".to_string()),
            model_version: Some("1.0.0".to_string()),
            model_name: Some("test-model".to_string()),
            timeout: Some(300),
            curve: None,
        };

        package_content_addressed(&slices_dir, &config).unwrap();

        let manifest: serde_json::Value =
            rmp_serde::from_slice(&fs::read(output_dir.join("manifest.msgpack")).unwrap()).unwrap();

        assert_eq!(manifest["version"], 1);
        assert_eq!(manifest["model"]["name"], "test-model");
        assert_eq!(manifest["model"]["author"], "test-author");
        assert_eq!(manifest["model"]["version"], "1.0.0");
        assert_eq!(manifest["model"]["timeout"], 300);

        let components = manifest["components"].as_array().unwrap();
        assert_eq!(components.len(), 2);
        for comp in components {
            let sha = comp["sha256"].as_str().unwrap();
            assert_eq!(sha.len(), 64);
            assert!(!comp["files"].as_array().unwrap().is_empty());
            assert_eq!(comp["proof_system"], "jstprove");
            assert!(!comp["weights"].as_array().unwrap().is_empty());
        }

        let dag = manifest["dag"].as_array().unwrap();
        assert_eq!(dag.len(), 2);
        assert_eq!(dag[0]["inputs"][0], "model_input");
        assert_eq!(dag[0]["outputs"][0], "tensor_0");
        assert_eq!(dag[1]["inputs"][0], "tensor_0");
    }

    #[test]
    fn test_component_files_exist() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 1);

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir: output_dir.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        package_content_addressed(&slices_dir, &config).unwrap();

        let manifest: serde_json::Value =
            rmp_serde::from_slice(&fs::read(output_dir.join("manifest.msgpack")).unwrap()).unwrap();

        let comp = &manifest["components"][0];
        let sha = comp["sha256"].as_str().unwrap();
        let comp_dir = output_dir.join("components").join(sha);
        assert!(comp_dir.is_dir());
        assert!(comp_dir.join("circuit.bin").is_file());
        assert!(comp_dir.join("settings.json").is_file());
    }

    #[test]
    fn test_wb_files_exist() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 1);

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir: output_dir.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        package_content_addressed(&slices_dir, &config).unwrap();

        let manifest: serde_json::Value =
            rmp_serde::from_slice(&fs::read(output_dir.join("manifest.msgpack")).unwrap()).unwrap();

        let weight = &manifest["components"][0]["weights"][0];
        let sha = weight["sha256"].as_str().unwrap();
        let wb_path = output_dir.join("wb").join(sha);
        assert!(wb_path.is_file());

        let size = weight["size_bytes"].as_u64().unwrap();
        assert_eq!(fs::metadata(&wb_path).unwrap().len(), size);
    }

    #[test]
    fn test_hash_determinism() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 2);

        let out1 = tmp.path().join("out1");
        let out2 = tmp.path().join("out2");

        let config1 = PackageConfig {
            output_dir: out1.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };
        let config2 = PackageConfig {
            output_dir: out2.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        package_content_addressed(&slices_dir, &config1).unwrap();
        package_content_addressed(&slices_dir, &config2).unwrap();

        let m1: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out1.join("manifest.msgpack")).unwrap()).unwrap();
        let m2: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out2.join("manifest.msgpack")).unwrap()).unwrap();

        for i in 0..2 {
            assert_eq!(m1["components"][i]["sha256"], m2["components"][i]["sha256"]);
        }
    }

    #[test]
    fn test_curve_changes_hash() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 2);

        let out_none = tmp.path().join("out_none");
        let out_bn = tmp.path().join("out_bn");
        let out_gl = tmp.path().join("out_gl");

        let config_none = PackageConfig {
            output_dir: out_none.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };
        let config_bn = PackageConfig {
            output_dir: out_bn.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: Some("bn254".to_string()),
        };
        let config_gl = PackageConfig {
            output_dir: out_gl.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: Some("goldilocks".to_string()),
        };

        package_content_addressed(&slices_dir, &config_none).unwrap();
        package_content_addressed(&slices_dir, &config_bn).unwrap();
        package_content_addressed(&slices_dir, &config_gl).unwrap();

        let m_none: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out_none.join("manifest.msgpack")).unwrap()).unwrap();
        let m_bn: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out_bn.join("manifest.msgpack")).unwrap()).unwrap();
        let m_gl: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out_gl.join("manifest.msgpack")).unwrap()).unwrap();

        for i in 0..2 {
            let h_none = m_none["components"][i]["sha256"].as_str().unwrap();
            let h_bn = m_bn["components"][i]["sha256"].as_str().unwrap();
            let h_gl = m_gl["components"][i]["sha256"].as_str().unwrap();
            assert_ne!(h_none, h_bn, "curve=None vs bn254 should differ");
            assert_ne!(h_none, h_gl, "curve=None vs goldilocks should differ");
            assert_ne!(h_bn, h_gl, "bn254 vs goldilocks should differ");
        }
    }

    #[test]
    fn test_invalid_curve_rejected() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 1);

        let config_typo = PackageConfig {
            output_dir: tmp.path().join("output"),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: Some("bm254".to_string()),
        };
        let result = package_content_addressed(&slices_dir, &config_typo);
        assert!(result.is_err());

        let config_empty = PackageConfig {
            output_dir: tmp.path().join("output2"),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: Some("".to_string()),
        };
        let result = package_content_addressed(&slices_dir, &config_empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_curve_normalization() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 1);

        let out1 = tmp.path().join("out1");
        let out2 = tmp.path().join("out2");
        let out3 = tmp.path().join("out3");

        let config1 = PackageConfig {
            output_dir: out1.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: Some("bn254".to_string()),
        };
        let config2 = PackageConfig {
            output_dir: out2.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: Some(" bn254 ".to_string()),
        };
        let config3 = PackageConfig {
            output_dir: out3.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: Some("BN254".to_string()),
        };

        package_content_addressed(&slices_dir, &config1).unwrap();
        package_content_addressed(&slices_dir, &config2).unwrap();
        package_content_addressed(&slices_dir, &config3).unwrap();

        let m1: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out1.join("manifest.msgpack")).unwrap()).unwrap();
        let m2: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out2.join("manifest.msgpack")).unwrap()).unwrap();
        let m3: serde_json::Value =
            rmp_serde::from_slice(&fs::read(out3.join("manifest.msgpack")).unwrap()).unwrap();

        assert_eq!(m1["components"][0]["sha256"], m2["components"][0]["sha256"]);
        assert_eq!(m1["components"][0]["sha256"], m3["components"][0]["sha256"]);
    }

    #[test]
    fn test_cleanup_removes_slice_dirs() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 2);

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir,
            cleanup: true,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        package_content_addressed(&slices_dir, &config).unwrap();

        for i in 0..2 {
            assert!(!slices_dir.join(format!("slice_{}", i)).exists());
        }
    }

    #[test]
    fn test_deduplication_shared_circuits() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        let mut slices = Vec::new();
        let shared_circuit_dir = slices_dir.join("shared_circuit").join("circuit.bundle");
        fs::create_dir_all(&shared_circuit_dir).unwrap();
        fs::write(
            shared_circuit_dir.join("circuit.bin"),
            "shared_circuit_data",
        )
        .unwrap();

        for i in 0..3 {
            let slice_dir = slices_dir.join(format!("slice_{}", i));
            let payload_dir = slice_dir.join("payload");
            fs::create_dir_all(&payload_dir).unwrap();
            fs::write(
                payload_dir.join(format!("slice_{}.onnx", i)),
                format!("onnx_data_{}", i),
            )
            .unwrap();

            slices.push(SliceMetadata {
                index: i,
                filename: format!("slice_{}.onnx", i),
                path: slice_dir.to_string_lossy().to_string(),
                relative_path: format!("slice_{}/payload/slice_{}.onnx", i, i),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![vec![1, 64]],
                        output: vec![vec![1, 64]],
                    },
                },
                dependencies: Dependencies {
                    input: vec![format!("t_{}", i)],
                    output: vec![format!("t_{}", i + 1)],
                    filtered_inputs: vec![],
                },
                tiling: None,
                channel_split: None,
                compilation: Compilation {
                    jstprove: BackendCompilation {
                        compiled: true,
                        tiled: false,
                        weights_as_inputs: false,
                        files: CompilationFiles {
                            compiled: Some("shared_circuit/circuit.bundle".to_string()),
                            settings: None,
                            pk_key: None,
                            vk_key: None,
                        },
                        compilation_timestamp: None,
                    },
                },
                slice_metadata: None,
                slice_metadata_relative_path: None,
            });
        }

        let meta = ModelMetadata {
            original_model: "shared_test".to_string(),
            model_type: "onnx".to_string(),
            input_shape: vec![vec![1, 64]],
            output_shapes: vec![vec![1, 64]],
            output_names: vec!["out".to_string()],
            slice_points: vec![0, 1, 2],
            slices,
            dsperse_version: None,
            dsperse_rev: None,
            jstprove_version: None,
            jstprove_rev: None,
            traced_shapes: None,
            original_model_path: None,
        };
        meta.save(&slices_dir.join("metadata.msgpack")).unwrap();

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir: output_dir.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config).unwrap();

        assert_eq!(result.component_count, 1);
        assert_eq!(result.wb_count, 3);

        let manifest: serde_json::Value =
            rmp_serde::from_slice(&fs::read(output_dir.join("manifest.msgpack")).unwrap()).unwrap();
        let components = manifest["components"].as_array().unwrap();
        let hash0 = components[0]["sha256"].as_str().unwrap();
        let hash1 = components[1]["sha256"].as_str().unwrap();
        let hash2 = components[2]["sha256"].as_str().unwrap();
        assert_eq!(hash0, hash1);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_uncompiled_onnx_only_slice() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        let slice_dir = slices_dir.join("slice_0");
        let payload_dir = slice_dir.join("payload");
        fs::create_dir_all(&payload_dir).unwrap();
        fs::write(payload_dir.join("slice_0.onnx"), "onnx_payload_data").unwrap();

        let meta = ModelMetadata {
            original_model: "test".to_string(),
            model_type: "onnx".to_string(),
            input_shape: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            output_names: vec!["output".to_string()],
            slice_points: vec![0],
            slices: vec![SliceMetadata {
                index: 0,
                filename: "slice_0.onnx".to_string(),
                path: slice_dir.to_string_lossy().to_string(),
                relative_path: "slice_0/payload/slice_0.onnx".to_string(),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![vec![1, 3, 224, 224]],
                        output: vec![vec![1, 1000]],
                    },
                },
                dependencies: Dependencies {
                    input: vec!["input".to_string()],
                    output: vec!["output".to_string()],
                    filtered_inputs: vec![],
                },
                tiling: None,
                channel_split: None,
                compilation: Compilation {
                    jstprove: BackendCompilation {
                        compiled: false,
                        tiled: false,
                        weights_as_inputs: false,
                        files: CompilationFiles::default(),
                        compilation_timestamp: None,
                    },
                },
                slice_metadata: None,
                slice_metadata_relative_path: None,
            }],
            dsperse_version: None,
            dsperse_rev: None,
            jstprove_version: None,
            jstprove_rev: None,
            traced_shapes: None,
            original_model_path: None,
        };
        meta.save(&slices_dir.join("metadata.msgpack")).unwrap();

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir: output_dir.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config).unwrap();
        assert_eq!(result.component_count, 1);

        let manifest: serde_json::Value =
            rmp_serde::from_slice(&fs::read(output_dir.join("manifest.msgpack")).unwrap()).unwrap();

        let comp = &manifest["components"][0];
        assert!(comp["proof_system"].is_null());
        let sha = comp["sha256"].as_str().unwrap();
        let files = comp["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], "slice_0.onnx");

        let comp_dir = output_dir.join("components").join(sha);
        assert!(comp_dir.join("slice_0.onnx").is_file());
    }

    #[test]
    fn test_missing_artifact_errors() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        let slice_dir = slices_dir.join("slice_0");
        fs::create_dir_all(&slice_dir).unwrap();

        let meta = ModelMetadata {
            original_model: "test".to_string(),
            model_type: "onnx".to_string(),
            input_shape: vec![vec![1]],
            output_shapes: vec![vec![1]],
            output_names: vec!["out".to_string()],
            slice_points: vec![0],
            slices: vec![SliceMetadata {
                index: 0,
                filename: "slice_0.onnx".to_string(),
                path: slice_dir.to_string_lossy().to_string(),
                relative_path: "slice_0/payload/slice_0.onnx".to_string(),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![vec![1]],
                        output: vec![vec![1]],
                    },
                },
                dependencies: Dependencies {
                    input: vec!["in".to_string()],
                    output: vec!["out".to_string()],
                    filtered_inputs: vec![],
                },
                tiling: None,
                channel_split: None,
                compilation: Compilation {
                    jstprove: BackendCompilation {
                        compiled: false,
                        tiled: false,
                        weights_as_inputs: false,
                        files: CompilationFiles::default(),
                        compilation_timestamp: None,
                    },
                },
                slice_metadata: None,
                slice_metadata_relative_path: None,
            }],
            dsperse_version: None,
            dsperse_rev: None,
            jstprove_version: None,
            jstprove_rev: None,
            traced_shapes: None,
            original_model_path: None,
        };
        meta.save(&slices_dir.join("metadata.msgpack")).unwrap();

        let config = PackageConfig {
            output_dir: tmp.path().join("output"),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no circuit directory or ONNX artifact"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_path_traversal_rejected() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        let slice_dir = slices_dir.join("slice_0");
        let payload_dir = slice_dir.join("payload");
        fs::create_dir_all(&payload_dir).unwrap();
        fs::write(payload_dir.join("slice_0.onnx"), "data").unwrap();

        let meta = ModelMetadata {
            original_model: "test".to_string(),
            model_type: "onnx".to_string(),
            input_shape: vec![vec![1]],
            output_shapes: vec![vec![1]],
            output_names: vec!["out".to_string()],
            slice_points: vec![0],
            slices: vec![SliceMetadata {
                index: 0,
                filename: "slice_0.onnx".to_string(),
                path: slice_dir.to_string_lossy().to_string(),
                relative_path: "slice_0/payload/slice_0.onnx".to_string(),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![vec![1]],
                        output: vec![vec![1]],
                    },
                },
                dependencies: Dependencies {
                    input: vec!["in".to_string()],
                    output: vec!["out".to_string()],
                    filtered_inputs: vec![],
                },
                tiling: None,
                channel_split: None,
                compilation: Compilation {
                    jstprove: BackendCompilation {
                        compiled: true,
                        tiled: false,
                        weights_as_inputs: false,
                        files: CompilationFiles {
                            compiled: Some("../../etc/passwd".to_string()),
                            settings: None,
                            pk_key: None,
                            vk_key: None,
                        },
                        compilation_timestamp: None,
                    },
                },
                slice_metadata: None,
                slice_metadata_relative_path: None,
            }],
            dsperse_version: None,
            dsperse_rev: None,
            jstprove_version: None,
            jstprove_rev: None,
            traced_shapes: None,
            original_model_path: None,
        };
        meta.save(&slices_dir.join("metadata.msgpack")).unwrap();

        let config = PackageConfig {
            output_dir: tmp.path().join("output"),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("path traversal"),
            "expected path traversal error, got: {err}"
        );
    }

    #[test]
    fn test_nonexistent_dir() {
        let config = PackageConfig {
            output_dir: PathBuf::from("/tmp/nonexistent_output"),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };
        let result = package_content_addressed(Path::new("/nonexistent/path"), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_cleanup_rejects_output_under_slice_dir() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();
        create_test_model_metadata(&slices_dir, 1);

        let config = PackageConfig {
            output_dir: slices_dir.join("slice_0").join("output"),
            cleanup: true,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("slice directory"),
            "expected slice dir error, got: {err}"
        );
    }

    #[test]
    fn test_identical_bytes_different_filenames_distinct_hashes() {
        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        let identical_data = "identical_onnx_content";

        let mut slices = Vec::new();
        for i in 0..2 {
            let slice_dir = slices_dir.join(format!("slice_{}", i));
            let payload_dir = slice_dir.join("payload");
            fs::create_dir_all(&payload_dir).unwrap();
            fs::write(
                payload_dir.join(format!("slice_{}.onnx", i)),
                identical_data,
            )
            .unwrap();

            slices.push(SliceMetadata {
                index: i,
                filename: format!("slice_{}.onnx", i),
                path: slice_dir.to_string_lossy().to_string(),
                relative_path: format!("slice_{}/payload/slice_{}.onnx", i, i),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![vec![1]],
                        output: vec![vec![1]],
                    },
                },
                dependencies: Dependencies {
                    input: vec![format!("t_{}", i)],
                    output: vec![format!("t_{}", i + 1)],
                    filtered_inputs: vec![],
                },
                tiling: None,
                channel_split: None,
                compilation: Compilation {
                    jstprove: BackendCompilation {
                        compiled: false,
                        tiled: false,
                        weights_as_inputs: false,
                        files: CompilationFiles::default(),
                        compilation_timestamp: None,
                    },
                },
                slice_metadata: None,
                slice_metadata_relative_path: None,
            });
        }

        let meta = ModelMetadata {
            original_model: "test".to_string(),
            model_type: "onnx".to_string(),
            input_shape: vec![vec![1]],
            output_shapes: vec![vec![1]],
            output_names: vec!["out".to_string()],
            slice_points: vec![0, 1],
            slices,
            dsperse_version: None,
            dsperse_rev: None,
            jstprove_version: None,
            jstprove_rev: None,
            traced_shapes: None,
            original_model_path: None,
        };
        meta.save(&slices_dir.join("metadata.msgpack")).unwrap();

        let output_dir = tmp.path().join("output");
        let config = PackageConfig {
            output_dir: output_dir.clone(),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config).unwrap();
        assert_eq!(result.component_count, 2);

        let manifest: serde_json::Value =
            rmp_serde::from_slice(&fs::read(output_dir.join("manifest.msgpack")).unwrap()).unwrap();
        let c0 = &manifest["components"][0];
        let c1 = &manifest["components"][1];
        assert_ne!(c0["sha256"], c1["sha256"]);

        let dir0 = output_dir
            .join("components")
            .join(c0["sha256"].as_str().unwrap());
        let dir1 = output_dir
            .join("components")
            .join(c1["sha256"].as_str().unwrap());
        assert!(dir0.join("slice_0.onnx").is_file());
        assert!(dir1.join("slice_1.onnx").is_file());
    }

    #[test]
    #[cfg(unix)]
    fn test_symlink_payload_rejected() {
        use std::os::unix::fs::symlink;

        let tmp = TempDir::new().unwrap();
        let slices_dir = tmp.path().join("model").join("slices");
        fs::create_dir_all(&slices_dir).unwrap();

        let external = tmp.path().join("external_secret.bin");
        fs::write(&external, "sensitive data").unwrap();

        let slice_dir = slices_dir.join("slice_0");
        let payload_dir = slice_dir.join("payload");
        fs::create_dir_all(&payload_dir).unwrap();
        symlink(&external, payload_dir.join("slice_0.onnx")).unwrap();

        let meta = ModelMetadata {
            original_model: "test".to_string(),
            model_type: "onnx".to_string(),
            input_shape: vec![vec![1]],
            output_shapes: vec![vec![1]],
            output_names: vec!["out".to_string()],
            slice_points: vec![0],
            slices: vec![SliceMetadata {
                index: 0,
                filename: "slice_0.onnx".to_string(),
                path: slice_dir.to_string_lossy().to_string(),
                relative_path: "slice_0/payload/slice_0.onnx".to_string(),
                shape: SliceShapeWrapper {
                    tensor_shape: TensorShape {
                        input: vec![vec![1]],
                        output: vec![vec![1]],
                    },
                },
                dependencies: Dependencies {
                    input: vec!["in".to_string()],
                    output: vec!["out".to_string()],
                    filtered_inputs: vec![],
                },
                tiling: None,
                channel_split: None,
                compilation: Compilation {
                    jstprove: BackendCompilation {
                        compiled: false,
                        tiled: false,
                        weights_as_inputs: false,
                        files: CompilationFiles::default(),
                        compilation_timestamp: None,
                    },
                },
                slice_metadata: None,
                slice_metadata_relative_path: None,
            }],
            dsperse_version: None,
            dsperse_rev: None,
            jstprove_version: None,
            jstprove_rev: None,
            traced_shapes: None,
            original_model_path: None,
        };
        meta.save(&slices_dir.join("metadata.msgpack")).unwrap();

        let config = PackageConfig {
            output_dir: tmp.path().join("output"),
            cleanup: false,
            author: None,
            model_version: None,
            model_name: None,
            timeout: None,
            curve: None,
        };

        let result = package_content_addressed(&slices_dir, &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("symlink"),
            "expected symlink error, got: {err}"
        );
    }
}
