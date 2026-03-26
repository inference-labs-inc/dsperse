use std::path::Path;

use rayon::prelude::*;

use crate::backend::jstprove::JstproveBackend;
use crate::converter;
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;
use crate::slicer::onnx_proto;
use crate::utils::paths::{find_metadata_path, slice_dir_path};

enum CompileOutcome {
    Compiled,
    CompiledChannelSplit {
        group_circuits: Vec<(usize, String)>,
    },
    Skipped,
}

pub fn compile_slices(
    slices_dir: &Path,
    backend: &JstproveBackend,
    parallel: usize,
    weights_as_inputs: bool,
    layers: Option<&[usize]>,
    jstprove_ops: &[&str],
) -> Result<()> {
    let meta_path = find_metadata_path(slices_dir).ok_or_else(|| {
        DsperseError::Metadata(format!(
            "no {} found in slices directory",
            crate::utils::paths::METADATA_FILE
        ))
    })?;
    let mut metadata = ModelMetadata::load(&meta_path)?;

    if metadata.original_model_path.is_some() {
        crate::slicer::materializer::ensure_all_slices_materialized(slices_dir, &metadata)?;
    }

    let mut metadata_dirty = false;
    for slice in &mut metadata.slices {
        if let Some(ref mut cs) = slice.channel_split
            && cs.groups.is_empty()
        {
            let populated = populate_channel_split_groups(slices_dir, slice.index, cs)?;
            if populated {
                metadata_dirty = true;
            }
        }
    }
    if metadata_dirty {
        metadata.save(&meta_path)?;
        tracing::info!("persisted materialized channel split groups to metadata");
    }

    let slices: Vec<_> = metadata
        .slices
        .iter()
        .filter(|s| layers.is_none_or(|l| l.contains(&s.index)))
        .collect();

    tracing::info!(total = slices.len(), "compiling slices");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let results: Vec<(usize, Result<CompileOutcome>)> = pool.install(|| {
        slices
            .par_iter()
            .map(|slice| {
                let r = compile_single_slice(
                    slices_dir,
                    slice,
                    backend,
                    weights_as_inputs,
                    jstprove_ops,
                );
                match &r {
                    Ok(CompileOutcome::Compiled) => {
                        tracing::info!(slice = slice.index, "compiled")
                    }
                    Ok(CompileOutcome::CompiledChannelSplit { group_circuits }) => {
                        tracing::info!(
                            slice = slice.index,
                            groups = group_circuits.len(),
                            "compiled channel split groups"
                        )
                    }
                    Ok(CompileOutcome::Skipped) => {
                        tracing::info!(slice = slice.index, "skipped (unsupported ops)")
                    }
                    Err(e) => {
                        tracing::error!(slice = slice.index, error = %e, "compilation failed")
                    }
                }
                (slice.index, r)
            })
            .collect()
    });

    let mut compiled_indices: Vec<usize> = Vec::new();
    let mut channel_split_results: Vec<(usize, Vec<(usize, String)>)> = Vec::new();
    let mut errors: Vec<(usize, DsperseError)> = Vec::new();
    for (idx, result) in results {
        match result {
            Ok(CompileOutcome::Compiled) => compiled_indices.push(idx),
            Ok(CompileOutcome::CompiledChannelSplit { group_circuits }) => {
                compiled_indices.push(idx);
                channel_split_results.push((idx, group_circuits));
            }
            Ok(CompileOutcome::Skipped) => {}
            Err(e) => errors.push((idx, e)),
        }
    }

    if !compiled_indices.is_empty() {
        drop(slices);
        let mut compiled_set = std::collections::HashSet::with_capacity(compiled_indices.len());
        compiled_set.extend(compiled_indices.iter().copied());

        let cs_map: std::collections::HashMap<usize, &Vec<(usize, String)>> = channel_split_results
            .iter()
            .map(|(idx, gc)| (*idx, gc))
            .collect();

        for slice in &mut metadata.slices {
            if compiled_set.contains(&slice.index) {
                slice.compilation.jstprove.compiled = true;
                if let Some(group_circuits) = cs_map.get(&slice.index) {
                    if let Some(ref mut cs) = slice.channel_split {
                        for (group_idx, circuit_path) in *group_circuits {
                            if let Some(group) =
                                cs.groups.iter_mut().find(|g| g.group_idx == *group_idx)
                            {
                                group.jstprove_circuit_path = Some(circuit_path.clone());
                            } else {
                                tracing::warn!(
                                    slice = slice.index,
                                    group = group_idx,
                                    "compiled group not found in metadata"
                                );
                            }
                        }
                    }
                } else {
                    slice.compilation.jstprove.files.compiled =
                        Some(format!("slice_{}/jstprove/circuit.bundle", slice.index));
                }
            }
        }
        metadata.save(&meta_path)?;
        tracing::info!(
            count = compiled_indices.len(),
            "persisted compiled flags to metadata"
        );
    }

    if errors.is_empty() {
        tracing::info!("all slices compiled");
        Ok(())
    } else {
        let msg = errors
            .iter()
            .map(|(idx, e)| format!("slice {idx}: {e}"))
            .collect::<Vec<_>>()
            .join("; ");
        Err(DsperseError::Pipeline(format!(
            "{} slices failed: {msg}",
            errors.len()
        )))
    }
}

struct SliceAnalysis {
    compatible: bool,
}

fn analyze_slice_onnx(onnx_path: &Path, jstprove_ops: &[&str]) -> Result<SliceAnalysis> {
    let model = onnx_proto::load_model(onnx_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Slicer(format!("no graph in {}", onnx_path.display())))?;
    Ok(SliceAnalysis {
        compatible: graph
            .node
            .iter()
            .all(|n| jstprove_ops.contains(&n.op_type.as_str())),
    })
}

fn compile_single_slice(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    backend: &JstproveBackend,
    weights_as_inputs: bool,
    jstprove_ops: &[&str],
) -> Result<CompileOutcome> {
    let slice_dir = slice_dir_path(slices_dir, slice.index);
    if !slice_dir.exists() {
        return Err(DsperseError::Pipeline(format!(
            "slice directory not found: {}",
            slice_dir.display()
        )));
    }

    if let Some(ref cs) = slice.channel_split
        && !cs.groups.is_empty()
    {
        return compile_channel_split_slice(slices_dir, slice, cs, backend, jstprove_ops);
    }

    let onnx_path = resolve_compile_onnx(slices_dir, slice)?;
    if !onnx_path.exists() {
        return Err(DsperseError::Pipeline(format!(
            "ONNX model not found for slice {}: {}",
            slice.index,
            onnx_path.display()
        )));
    }

    let analysis = analyze_slice_onnx(&onnx_path, jstprove_ops)?;
    if !analysis.compatible {
        return Ok(CompileOutcome::Skipped);
    }

    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let circuit_path = jst_dir.join("circuit.bundle");

    if circuit_path.is_dir() {
        match backend.load_params(&circuit_path) {
            Ok(_) => {
                tracing::info!(slice = slice.index, "already compiled, skipping");
                return Ok(CompileOutcome::Compiled);
            }
            Err(e) => {
                tracing::warn!(slice = slice.index, error = %e, "cached circuit invalid, recompiling");
                std::fs::remove_dir_all(&circuit_path)
                    .map_err(|e| DsperseError::io(e, &circuit_path))?;
            }
        }
    }

    let effective_wai = weights_as_inputs;

    let (params, architecture, wandb) =
        converter::prepare_jstprove_artifacts(&onnx_path, effective_wai)?;

    std::panic::catch_unwind(|| backend.compile(&circuit_path, params, architecture, wandb))
        .map_err(|p| {
            let msg = p
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| p.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            DsperseError::Backend(format!("jstprove panicked: {msg}"))
        })??;

    Ok(CompileOutcome::Compiled)
}

fn populate_channel_split_groups(
    slices_dir: &Path,
    slice_idx: usize,
    cs: &mut crate::schema::tiling::ChannelSplitInfo,
) -> Result<bool> {
    let groups_dir = slices_dir
        .join(format!("slice_{slice_idx}"))
        .join("payload")
        .join("channel_groups");
    if !groups_dir.exists() {
        return Ok(false);
    }

    let cpg = cs.channels_per_group;
    let mut groups = Vec::with_capacity(cs.num_groups);
    for g in 0..cs.num_groups {
        let c_start = g.checked_mul(cpg).ok_or_else(|| {
            DsperseError::Slicer(format!("overflow computing c_start for group {g}"))
        })?;
        let c_end = (g + 1)
            .checked_mul(cpg)
            .map(|v| v.min(cs.c_in))
            .ok_or_else(|| {
                DsperseError::Slicer(format!("overflow computing c_end for group {g}"))
            })?;
        let rel_path = format!("slice_{slice_idx}/payload/channel_groups/group_{g}.onnx");
        let abs_path = slices_dir.join(&rel_path);
        if !abs_path.exists() {
            tracing::warn!(
                slice = slice_idx,
                group = g,
                "expected group ONNX not found, skipping population"
            );
            return Ok(false);
        }
        groups.push(crate::schema::tiling::ChannelGroupInfo {
            group_idx: g,
            c_start,
            c_end,
            path: rel_path,
            jstprove_circuit_path: None,
            jstprove_settings_path: None,
        });
    }

    let bias_rel = format!("slice_{slice_idx}/payload/channel_groups/bias.msgpack");
    if slices_dir.join(&bias_rel).exists() {
        cs.bias_path = Some(bias_rel);
    }

    tracing::info!(
        slice = slice_idx,
        groups = groups.len(),
        "populated channel split groups from materialized files"
    );
    cs.groups = groups;
    Ok(true)
}

fn compile_channel_split_slice(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    cs: &crate::schema::tiling::ChannelSplitInfo,
    backend: &JstproveBackend,
    jstprove_ops: &[&str],
) -> Result<CompileOutcome> {
    let slice_dir = slice_dir_path(slices_dir, slice.index);
    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let shared_circuit_rel = format!("slice_{}/jstprove/shared/circuit.bundle", slice.index);
    let shared_circuit_path = jst_dir.join("shared").join("circuit.bundle");

    if !shared_circuit_path.is_dir() {
        let first_group = cs.groups.first().ok_or_else(|| {
            DsperseError::Pipeline(format!("slice {} channel_split has no groups", slice.index))
        })?;
        let onnx_path = slices_dir.join(&first_group.path);
        if !onnx_path.exists() {
            return Err(DsperseError::Pipeline(format!(
                "channel group ONNX not found: {}",
                onnx_path.display()
            )));
        }

        let analysis = analyze_slice_onnx(&onnx_path, jstprove_ops)?;
        if !analysis.compatible {
            return Err(DsperseError::Pipeline(format!(
                "slice {} group 0 has unsupported ops for circuit compilation",
                slice.index
            )));
        }

        let shared_dir = shared_circuit_path
            .parent()
            .ok_or_else(|| DsperseError::Pipeline("shared circuit path has no parent".into()))?;
        std::fs::create_dir_all(shared_dir).map_err(|e| DsperseError::io(e, shared_dir))?;

        tracing::info!(
            slice = slice.index,
            groups = cs.groups.len(),
            "compiling shared channel group circuit (weights-as-inputs)"
        );

        let (params, architecture, wandb) =
            converter::prepare_jstprove_artifacts(&onnx_path, true)?;

        std::panic::catch_unwind(|| {
            backend.compile(&shared_circuit_path, params, architecture, wandb)
        })
        .map_err(|p| {
            let msg = p
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| p.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            DsperseError::Backend(format!(
                "jstprove panicked on slice {} shared circuit: {msg}",
                slice.index
            ))
        })??;

        tracing::info!(slice = slice.index, "shared circuit compiled");
    } else {
        backend.load_params(&shared_circuit_path).map_err(|e| {
            DsperseError::Pipeline(format!(
                "slice {} cached shared circuit invalid: {e}",
                slice.index
            ))
        })?;
        tracing::info!(
            slice = slice.index,
            "shared circuit already compiled, reusing"
        );
    }

    let group_circuits: Vec<(usize, String)> = cs
        .groups
        .iter()
        .map(|g| (g.group_idx, shared_circuit_rel.clone()))
        .collect();

    Ok(CompileOutcome::CompiledChannelSplit { group_circuits })
}

fn resolve_compile_onnx(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
) -> Result<std::path::PathBuf> {
    if let Some(ref tiling) = slice.tiling
        && let Some(ref tile) = tiling.tile
    {
        let tile_path = slices_dir.join(&tile.path);
        if tile_path.exists() {
            tracing::info!(
                slice = slice.index,
                path = %tile_path.display(),
                "using tile ONNX"
            );
            return Ok(tile_path);
        }
    }

    slice.resolve_onnx(slices_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::metadata::{
        Compilation, Dependencies, SliceMetadata, SliceShapeWrapper, TensorShape,
    };
    use crate::schema::tiling::{TileInfo, TilingInfo};

    fn test_models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models"))
    }

    fn make_slice_metadata(index: usize, path: &str) -> SliceMetadata {
        SliceMetadata {
            index,
            filename: format!("slice_{index}.onnx"),
            path: path.to_string(),
            relative_path: path.to_string(),
            shape: SliceShapeWrapper {
                tensor_shape: TensorShape::default(),
            },
            dependencies: Dependencies {
                input: vec![],
                output: vec![],
                filtered_inputs: vec![],
            },
            tiling: None,
            channel_split: None,
            compilation: Compilation::default(),
            slice_metadata: None,
            slice_metadata_relative_path: None,
        }
    }

    const TEST_OPS: &[&str] = &["Conv", "Gemm", "MatMul"];

    #[test]
    fn analyze_slice_onnx_nonexistent() {
        let result = analyze_slice_onnx(Path::new("/nonexistent.onnx"), TEST_OPS);
        assert!(result.is_err());
    }

    #[test]
    fn analyze_slice_onnx_test_model() {
        let model_path = test_models_dir().join("net/model.onnx");
        assert!(
            model_path.exists(),
            "fixture missing: {}",
            model_path.display()
        );
        let analysis = analyze_slice_onnx(&model_path, TEST_OPS).unwrap();
        assert!(!analysis.compatible);
    }

    #[test]
    fn analyze_slice_onnx_with_initializers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("with_init.onnx");
        let model = onnx_proto::ModelProto {
            graph: Some(onnx_proto::GraphProto {
                node: vec![onnx_proto::make_node("Conv", vec![], vec![], vec![])],
                initializer: vec![onnx_proto::make_tensor(
                    "weight",
                    1,
                    &[3, 3, 3, 3],
                    vec![0.0; 81],
                )],
                ..Default::default()
            }),
            ..Default::default()
        };
        onnx_proto::save_model(&model, &path).unwrap();
        let analysis = analyze_slice_onnx(&path, &["Conv"]).unwrap();
        assert!(analysis.compatible);
    }

    #[test]
    fn analyze_slice_onnx_without_initializers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("no_init.onnx");
        let model = onnx_proto::ModelProto {
            graph: Some(onnx_proto::GraphProto {
                node: vec![onnx_proto::make_node("Relu", vec![], vec![], vec![])],
                initializer: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };
        onnx_proto::save_model(&model, &path).unwrap();
        let analysis = analyze_slice_onnx(&path, &["Relu"]).unwrap();
        assert!(analysis.compatible);
    }

    #[test]
    fn resolve_compile_onnx_no_tiling() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let slice_dir = slices_dir.join("slice_0");
        std::fs::create_dir_all(&slice_dir).unwrap();

        let meta = make_slice_metadata(0, "slice_0.onnx");
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("slice_0.onnx"));
    }

    #[test]
    fn resolve_compile_onnx_with_tile() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let tile_path = slices_dir.join("slice_0/payload/tiles/tile.onnx");
        std::fs::create_dir_all(tile_path.parent().unwrap()).unwrap();
        std::fs::write(&tile_path, b"dummy").unwrap();

        let mut meta = make_slice_metadata(0, "slice_0.onnx");
        meta.tiling = Some(TilingInfo {
            slice_idx: 0,
            tile_size: 8,
            num_tiles: 4,
            tiles_y: 2,
            tiles_x: 2,
            halo: [1, 1, 1, 1],
            out_tile: [4, 4],
            stride: [1, 1],
            c_in: 3,
            c_out: 16,
            input_name: "input".into(),
            output_name: "output".into(),
            input_names: vec![],
            ndim: 4,
            h: 16,
            w: 16,
            tile: Some(TileInfo {
                path: "slice_0/payload/tiles/tile.onnx".into(),
                conv_out: [4, 4],
                jstprove_circuit_path: None,
            }),
            tiles: None,
        });
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("tile.onnx"));
    }

    #[test]
    fn resolve_compile_onnx_tile_missing_falls_back() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let slice_dir = slices_dir.join("slice_0");
        std::fs::create_dir_all(&slice_dir).unwrap();

        let mut meta = make_slice_metadata(0, "slice_0.onnx");
        meta.tiling = Some(TilingInfo {
            slice_idx: 0,
            tile_size: 8,
            num_tiles: 4,
            tiles_y: 2,
            tiles_x: 2,
            halo: [1, 1, 1, 1],
            out_tile: [4, 4],
            stride: [1, 1],
            c_in: 3,
            c_out: 16,
            input_name: "input".into(),
            output_name: "output".into(),
            input_names: vec![],
            ndim: 4,
            h: 16,
            w: 16,
            tile: Some(TileInfo {
                path: "slice_0/payload/tiles/nonexistent.onnx".into(),
                conv_out: [4, 4],
                jstprove_circuit_path: None,
            }),
            tiles: None,
        });
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("slice_0.onnx"));
    }
}
