use std::collections::HashMap;
use std::path::Path;

use ndarray::{s, Array4, ArrayD, Axis, IxDyn};

use crate::backend::{JstproveBackend, PipeWitnessJob};
use crate::error::{DsperseError, Result};
use crate::schema::execution::{
    ExecutionChain, ExecutionInfo, ExecutionMethod, ExecutionNode, ExecutionResultEntry,
    RunMetadata, TileResult,
};
use crate::schema::metadata::{ModelMetadata, RunSliceMetadata};
use crate::schema::tiling::{ChannelGroupInfo, ChannelSplitInfo, TilingInfo};
use crate::utils::io::{flatten_nested_list, read_input_json, write_input_json};
use crate::utils::paths::{find_metadata_path, resolve_relative_path, slice_dir_path};

pub struct RunConfig {
    pub parallel: usize,
    pub batch: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            parallel: 1,
            batch: false,
        }
    }
}

pub fn run_inference(
    slices_dir: &Path,
    input_path: &Path,
    run_dir: &Path,
    backend: &JstproveBackend,
    config: &RunConfig,
) -> Result<RunMetadata> {
    let meta_path = find_metadata_path(slices_dir)
        .ok_or_else(|| DsperseError::Metadata("no metadata.json in slices".into()))?;
    let model_meta = ModelMetadata::load(&meta_path)?;

    std::fs::create_dir_all(run_dir).map_err(|e| DsperseError::io(e, run_dir))?;

    let input_data = read_input_json(input_path)?;

    let chain = build_execution_chain(&model_meta, slices_dir);
    let run_meta = build_run_metadata(&model_meta, slices_dir, run_dir, &chain);

    let mut tensor_cache: HashMap<String, serde_json::Value> = HashMap::new();

    if let Some(input_val) = input_data.get("input_data").or_else(|| input_data.get("input")) {
        if let Some(first_input) = model_meta.slices.first() {
            if let Some(name) = first_input.dependencies.input.first() {
                tensor_cache.insert(name.clone(), input_val.clone());
            }
        }
    }

    let input_copy = run_dir.join("input.json");
    write_input_json(&input_copy, &input_data)?;

    let mut results: Vec<ExecutionResultEntry> = Vec::new();

    let mut current = chain.head.clone();
    while let Some(slice_id) = current.take() {
        let node = chain
            .nodes
            .get(&slice_id)
            .ok_or_else(|| DsperseError::Pipeline(format!("missing node {slice_id}")))?;

        let slice_meta = run_meta
            .slices
            .get(&slice_id)
            .ok_or_else(|| {
                DsperseError::Pipeline(format!("missing run slice metadata {slice_id}"))
            })?;

        let slice_run_dir = run_dir.join(&slice_id);
        std::fs::create_dir_all(&slice_run_dir)
            .map_err(|e| DsperseError::io(e, &slice_run_dir))?;

        tracing::info!(slice = %slice_id, circuit = node.use_circuit, "executing");

        let exec_result = execute_slice(
            slices_dir,
            &slice_run_dir,
            &slice_id,
            node,
            slice_meta,
            &mut tensor_cache,
            backend,
            config,
        );

        let exec_info = match exec_result {
            Ok(info) => info,
            Err(e) => {
                tracing::error!(slice = %slice_id, error = %e, "execution failed");
                ExecutionInfo {
                    method: ExecutionMethod::OnnxOnly.to_string(),
                    success: false,
                    error: Some(e.to_string()),
                    witness_file: None,
                    tile_exec_infos: Vec::new(),
                }
            }
        };

        results.push(ExecutionResultEntry {
            slice_id: slice_id.clone(),
            witness_execution: Some(exec_info),
            proof_execution: None,
            verification_execution: None,
        });

        current = node.next.clone();
    }

    let output_path = run_dir.join("output.json");
    if let Some(last_slice) = model_meta.slices.last() {
        if let Some(output_name) = last_slice.dependencies.output.first() {
            if let Some(output_val) = tensor_cache.get(output_name) {
                let output_json = serde_json::json!({ "output_data": output_val });
                write_input_json(&output_path, &output_json)?;
            }
        }
    }

    let mut final_meta = run_meta;
    final_meta.execution_chain.execution_results = results;
    final_meta.run_directory = Some(run_dir.to_string_lossy().into_owned());

    let meta_out = run_dir.join("metadata.json");
    let meta_json = serde_json::to_string_pretty(&final_meta)?;
    std::fs::write(&meta_out, meta_json).map_err(|e| DsperseError::io(e, &meta_out))?;

    Ok(final_meta)
}

fn execute_slice(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    node: &ExecutionNode,
    meta: &RunSliceMetadata,
    tensor_cache: &mut HashMap<String, serde_json::Value>,
    backend: &JstproveBackend,
    config: &RunConfig,
) -> Result<ExecutionInfo> {
    if let Some(ref cs) = meta.channel_split {
        return execute_channel_split(slices_dir, slice_run_dir, slice_id, cs, tensor_cache, backend);
    }

    if let Some(ref tiling) = meta.tiling {
        return execute_tiled(slices_dir, slice_run_dir, slice_id, tiling, tensor_cache, backend, config);
    }

    execute_single(slices_dir, slice_run_dir, slice_id, node, meta, tensor_cache, backend)
}

fn execute_single(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    node: &ExecutionNode,
    meta: &RunSliceMetadata,
    tensor_cache: &mut HashMap<String, serde_json::Value>,
    backend: &JstproveBackend,
) -> Result<ExecutionInfo> {
    let slice_idx: usize = slice_id
        .strip_prefix("slice_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let slice_dir = slice_dir_path(slices_dir, slice_idx);

    let input_tensor = gather_inputs(tensor_cache, &meta.dependencies.input)?;

    let input_path = slice_run_dir.join("input.json");
    write_input_json(
        &input_path,
        &serde_json::json!({ "input_data": input_tensor }),
    )?;

    if node.use_circuit {
        let circuit_path = meta
            .jstprove_circuit_path
            .as_deref()
            .or(meta.circuit_path.as_deref())
            .map(|p| resolve_relative_path(&slice_dir, p))
            .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {slice_id}")))?;

        let metadata_path = meta
            .jstprove_settings_path
            .as_deref()
            .or(meta.settings_path.as_deref())
            .map(|p| resolve_relative_path(&slice_dir, p))
            .unwrap_or_else(|| circuit_path.clone());

        let output_path = slice_run_dir.join("output.json");
        let witness_path = slice_run_dir.join("witness.bin");

        backend.witness(
            &circuit_path,
            &input_path,
            &output_path,
            &witness_path,
            &metadata_path,
            None,
        )?;

        let output_data = read_input_json(&output_path)?;
        store_outputs(tensor_cache, &meta.dependencies.output, &output_data)?;

        Ok(ExecutionInfo {
            method: ExecutionMethod::JstproveGenWitness.to_string(),
            success: true,
            error: None,
            witness_file: Some(witness_path.to_string_lossy().into_owned()),
            tile_exec_infos: Vec::new(),
        })
    } else {
        let onnx_path = resolve_relative_path(&slice_dir, &meta.path);
        let output_data = run_onnx_inference(&onnx_path, &input_tensor)?;
        store_outputs(tensor_cache, &meta.dependencies.output, &output_data)?;

        Ok(ExecutionInfo {
            method: ExecutionMethod::OnnxOnly.to_string(),
            success: true,
            error: None,
            witness_file: None,
            tile_exec_infos: Vec::new(),
        })
    }
}

fn execute_tiled(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    tiling: &TilingInfo,
    tensor_cache: &mut HashMap<String, serde_json::Value>,
    backend: &JstproveBackend,
    config: &RunConfig,
) -> Result<ExecutionInfo> {
    let slice_idx: usize = slice_id
        .strip_prefix("slice_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let slice_dir = slice_dir_path(slices_dir, slice_idx);

    let input_val = tensor_cache
        .get(&tiling.input_name)
        .ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "tiling input '{}' not in cache for {slice_id}",
                tiling.input_name
            ))
        })?
        .clone();

    let input_flat = flatten_nested_list(&input_val);
    let input_4d = reshape_to_4d(&input_flat, tiling.c_in, tiling.tile_size)?;

    let tiles = split_into_tiles(&input_4d, tiling);

    tracing::info!(
        slice = %slice_id,
        num_tiles = tiles.len(),
        tile_size = tiling.tile_size,
        "splitting into tiles"
    );

    let tile_infos = tiling.tiles.as_deref().unwrap_or(&[]);
    let single_tile = tiling.tile.as_ref();

    let mut tile_results: Vec<TileResult> = Vec::new();
    let mut tile_outputs: Vec<serde_json::Value> = Vec::new();

    if config.batch {
        let batch_result = execute_tiles_batch(
            &slice_dir,
            slice_run_dir,
            &tiles,
            tiling,
            backend,
        )?;
        tile_results = batch_result.0;
        tile_outputs = batch_result.1;
    } else {
        for (tile_idx, tile_data) in tiles.iter().enumerate() {
            let start = std::time::Instant::now();
            let tile_dir = slice_run_dir.join(format!("tile_{tile_idx}"));
            std::fs::create_dir_all(&tile_dir)
                .map_err(|e| DsperseError::io(e, &tile_dir))?;
            let tile_run_dir = &tile_dir;

            let tile_info = tile_infos.get(tile_idx).or(single_tile);
            let tile_input_json = ndarray_to_nested_json(&tile_data.clone().into_dyn());

            let input_path = tile_run_dir.join("input.json");
            write_input_json(
                &input_path,
                &serde_json::json!({ "input_data": tile_input_json }),
            )?;

            let result = if let Some(ti) = tile_info {
                let tile_circuit =
                    resolve_relative_path(&slice_dir, &ti.path);
                let tile_circuit_parent = tile_circuit.parent().unwrap_or(&slice_dir);
                let metadata_file = tile_circuit_parent.join("metadata.json");
                let metadata_path = if metadata_file.exists() {
                    metadata_file
                } else {
                    tile_circuit.clone()
                };

                let output_path = tile_run_dir.join("output.json");
                let witness_path = tile_run_dir.join("witness.bin");

                match backend.witness(
                    &tile_circuit,
                    &input_path,
                    &output_path,
                    &witness_path,
                    &metadata_path,
                    None,
                ) {
                    Ok(()) => {
                        let output_data = read_input_json(&output_path)?;
                        tile_outputs.push(extract_output_tensor(&output_data));
                        TileResult {
                            tile_idx,
                            success: true,
                            error: None,
                            method: Some("jstprove".into()),
                            time_sec: start.elapsed().as_secs_f64(),
                            proof_path: None,
                        }
                    }
                    Err(e) => TileResult {
                        tile_idx,
                        success: false,
                        error: Some(e.to_string()),
                        method: Some("jstprove".into()),
                        time_sec: start.elapsed().as_secs_f64(),
                        proof_path: None,
                    },
                }
            } else {
                TileResult {
                    tile_idx,
                    success: false,
                    error: Some("no tile circuit info".into()),
                    method: None,
                    time_sec: 0.0,
                    proof_path: None,
                }
            };

            tile_results.push(result);
        }
    }

    let all_success = tile_results.iter().all(|r| r.success);

    if all_success && !tile_outputs.is_empty() {
        let reconstructed = reconstruct_from_tiles(&tile_outputs, tiling)?;
        tensor_cache.insert(tiling.output_name.clone(), reconstructed);
    }

    Ok(ExecutionInfo {
        method: ExecutionMethod::Tiled.to_string(),
        success: all_success,
        error: if all_success {
            None
        } else {
            let failed: Vec<_> = tile_results
                .iter()
                .filter(|r| !r.success)
                .map(|r| format!("tile {}: {}", r.tile_idx, r.error.as_deref().unwrap_or("?")))
                .collect();
            Some(failed.join("; "))
        },
        witness_file: None,
        tile_exec_infos: tile_results,
    })
}

fn execute_tiles_batch(
    slice_dir: &Path,
    slice_run_dir: &Path,
    tiles: &[Array4<f64>],
    tiling: &TilingInfo,
    backend: &JstproveBackend,
) -> Result<(Vec<TileResult>, Vec<serde_json::Value>)> {
    let tile_info = tiling.tile.as_ref().or_else(|| {
        tiling.tiles.as_ref().and_then(|t| t.first())
    });
    let ti = tile_info.ok_or_else(|| DsperseError::Pipeline("no tile info for batch".into()))?;

    let tile_circuit = resolve_relative_path(slice_dir, &ti.path);
    let tile_circuit_parent = tile_circuit.parent().unwrap_or(slice_dir);
    let metadata_file = tile_circuit_parent.join("metadata.json");
    let metadata_path = if metadata_file.exists() {
        metadata_file
    } else {
        tile_circuit.clone()
    };

    let mut jobs: Vec<PipeWitnessJob> = Vec::new();
    for (tile_idx, tile_data) in tiles.iter().enumerate() {
        let tile_input_json = ndarray_to_nested_json(&tile_data.clone().into_dyn());
        let tile_dir = slice_run_dir.join(format!("tile_{tile_idx}"));
        std::fs::create_dir_all(&tile_dir)
            .map_err(|e| DsperseError::io(e, &tile_dir))?;
        let witness_path = tile_dir.join("witness.bin");

        jobs.push(PipeWitnessJob {
            input: serde_json::json!({ "input_data": tile_input_json }),
            output: serde_json::json!({ "output_data": [] }),
            witness: witness_path.to_string_lossy().into_owned(),
        });
    }

    let batch_result = backend.witness_piped(
        &tile_circuit,
        &metadata_path,
        &jobs,
        None,
    )?;

    let mut tile_results = Vec::new();
    let mut tile_outputs = Vec::new();

    for (tile_idx, _) in tiles.iter().enumerate() {
        let output_path = slice_run_dir
            .join(format!("tile_{tile_idx}"))
            .join("output.json");

        let failed = batch_result
            .errors
            .iter()
            .any(|(idx, _)| *idx == tile_idx);

        if failed {
            let err_msg = batch_result
                .errors
                .iter()
                .find(|(idx, _)| *idx == tile_idx)
                .map(|(_, msg)| msg.clone())
                .unwrap_or_default();
            tile_results.push(TileResult {
                tile_idx,
                success: false,
                error: Some(err_msg),
                method: Some("jstprove_batch".into()),
                time_sec: 0.0,
                proof_path: None,
            });
        } else if output_path.exists() {
            let output_data = read_input_json(&output_path)?;
            tile_outputs.push(extract_output_tensor(&output_data));
            tile_results.push(TileResult {
                tile_idx,
                success: true,
                error: None,
                method: Some("jstprove_batch".into()),
                time_sec: 0.0,
                proof_path: None,
            });
        } else {
            tile_results.push(TileResult {
                tile_idx,
                success: false,
                error: Some(format!("output file missing: {}", output_path.display())),
                method: Some("jstprove_batch".into()),
                time_sec: 0.0,
                proof_path: None,
            });
        }
    }

    Ok((tile_results, tile_outputs))
}

fn execute_channel_split(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    cs: &ChannelSplitInfo,
    tensor_cache: &mut HashMap<String, serde_json::Value>,
    backend: &JstproveBackend,
) -> Result<ExecutionInfo> {
    let slice_idx: usize = slice_id
        .strip_prefix("slice_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let slice_dir = slice_dir_path(slices_dir, slice_idx);

    let input_val = tensor_cache
        .get(&cs.input_name)
        .ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "channel split input '{}' not in cache for {slice_id}",
                cs.input_name
            ))
        })?
        .clone();

    let input_flat = flatten_nested_list(&input_val);
    let n = 1usize;
    let total_elements = input_flat.len();
    let spatial = if cs.c_in > 0 && total_elements > 0 {
        total_elements / (n * cs.c_in)
    } else {
        cs.h * cs.w
    };
    let h = cs.h.max(1);
    let w = if spatial > 0 && h > 0 { spatial / h } else { cs.w.max(1) };

    let input_4d = Array4::from_shape_vec(
        (n, cs.c_in, h, w),
        input_flat,
    )
    .map_err(|e| DsperseError::Pipeline(format!("channel split reshape: {e}")))?;

    let mut accumulated: Option<Array4<f64>> = None;

    tracing::info!(
        slice = %slice_id,
        num_groups = cs.groups.len(),
        "channel split execution"
    );

    for group in &cs.groups {
        let group_input = input_4d.slice(s![.., group.c_start..group.c_end, .., ..]);
        let group_input_json = ndarray_to_nested_json(&group_input.to_owned().into_dyn());

        let group_dir = slice_run_dir.join(format!("group_{}", group.group_idx));
        std::fs::create_dir_all(&group_dir).map_err(|e| DsperseError::io(e, &group_dir))?;

        let input_path = group_dir.join("input.json");
        write_input_json(
            &input_path,
            &serde_json::json!({ "input_data": group_input_json }),
        )?;

        let group_output = execute_channel_group(
            &slice_dir,
            &group_dir,
            group,
            &input_path,
            backend,
        )?;

        let group_flat = flatten_nested_list(&group_output);
        let group_4d = Array4::from_shape_vec(
            (n, cs.c_out, h, w),
            group_flat,
        )
        .map_err(|e| DsperseError::Pipeline(format!("group output reshape: {e}")))?;

        accumulated = Some(match accumulated {
            Some(acc) => acc + &group_4d,
            None => group_4d,
        });
    }

    if let Some(ref bias_path_str) = cs.bias_path {
        let bias_file = resolve_relative_path(&slice_dir, bias_path_str);
        if bias_file.exists() {
            let bias_data = read_input_json(&bias_file)?;
            let bias_flat = flatten_nested_list(&bias_data);
            if let Some(ref mut acc) = accumulated {
                for ((_, c, _, _), val) in acc.indexed_iter_mut() {
                    if c < bias_flat.len() {
                        *val += bias_flat[c];
                    }
                }
            }
        }
    }

    if let Some(acc) = accumulated {
        let output_json = ndarray_to_nested_json(&acc.into_dyn());
        tensor_cache.insert(cs.output_name.clone(), output_json);
    }

    Ok(ExecutionInfo {
        method: "channel_split".to_string(),
        success: true,
        error: None,
        witness_file: None,
        tile_exec_infos: Vec::new(),
    })
}

fn execute_channel_group(
    slice_dir: &Path,
    group_dir: &Path,
    group: &ChannelGroupInfo,
    input_path: &Path,
    backend: &JstproveBackend,
) -> Result<serde_json::Value> {
    if let Some(ref circuit_path_str) = group.jstprove_circuit_path {
        let circuit_path = resolve_relative_path(slice_dir, circuit_path_str);
        let metadata_path = group
            .jstprove_settings_path
            .as_deref()
            .or(group.settings_path.as_deref())
            .map(|p| resolve_relative_path(slice_dir, p))
            .unwrap_or_else(|| circuit_path.clone());

        let output_path = group_dir.join("output.json");
        let witness_path = group_dir.join("witness.bin");

        backend.witness(
            &circuit_path,
            input_path,
            &output_path,
            &witness_path,
            &metadata_path,
            None,
        )?;

        let output_data = read_input_json(&output_path)?;
        Ok(extract_output_tensor(&output_data))
    } else {
        let onnx_path = resolve_relative_path(slice_dir, &group.path);
        let input_data = read_input_json(input_path)?;
        let tensor = extract_output_tensor(&input_data);
        run_onnx_inference(&onnx_path, &tensor)
    }
}

fn split_into_tiles(input: &Array4<f64>, tiling: &TilingInfo) -> Vec<Array4<f64>> {
    let (n, c, h, w) = input.dim();
    let halo_h = tiling.halo[0].unsigned_abs() as usize;
    let halo_w = tiling.halo[1].unsigned_abs() as usize;
    let stride_h = tiling.stride[0].max(1) as usize;
    let stride_w = tiling.stride[1].max(1) as usize;
    let tile_h = tiling.tile_size + 2 * halo_h;
    let tile_w = tiling.tile_size + 2 * halo_w;

    let padded_h = h + 2 * halo_h;
    let padded_w = w + 2 * halo_w;
    let mut padded = Array4::<f64>::zeros((n, c, padded_h, padded_w));
    padded
        .slice_mut(s![.., .., halo_h..halo_h + h, halo_w..halo_w + w])
        .assign(input);

    let mut tiles = Vec::new();
    for ty in 0..tiling.tiles_y {
        for tx in 0..tiling.tiles_x {
            let y_start = ty * stride_h;
            let x_start = tx * stride_w;
            let y_end = (y_start + tile_h).min(padded_h);
            let x_end = (x_start + tile_w).min(padded_w);

            let tile = padded
                .slice(s![.., .., y_start..y_end, x_start..x_end])
                .to_owned();
            tiles.push(tile);
        }
    }

    tiles
}

fn reconstruct_from_tiles(
    tile_outputs: &[serde_json::Value],
    tiling: &TilingInfo,
) -> Result<serde_json::Value> {
    if tile_outputs.is_empty() {
        return Err(DsperseError::Pipeline("no tile outputs to reconstruct".into()));
    }

    let out_h = tiling.out_tile[0].max(1) as usize;
    let out_w = tiling.out_tile[1].max(1) as usize;
    let c_out = tiling.c_out;
    let total_h = out_h * tiling.tiles_y;
    let total_w = out_w * tiling.tiles_x;

    let mut output = Array4::<f64>::zeros((1, c_out, total_h, total_w));

    for (idx, tile_val) in tile_outputs.iter().enumerate() {
        let ty = idx / tiling.tiles_x;
        let tx = idx % tiling.tiles_x;

        let tile_flat = flatten_nested_list(tile_val);
        if tile_flat.is_empty() {
            continue;
        }

        let tile_elements = c_out * out_h * out_w;
        let tile_flat = if tile_flat.len() >= tile_elements {
            &tile_flat[..tile_elements]
        } else {
            &tile_flat
        };

        if let Ok(tile_arr) = Array4::from_shape_vec((1, c_out, out_h, out_w), tile_flat.to_vec())
        {
            let y_start = ty * out_h;
            let x_start = tx * out_w;
            output
                .slice_mut(s![
                    ..,
                    ..,
                    y_start..y_start + out_h,
                    x_start..x_start + out_w
                ])
                .assign(&tile_arr);
        }
    }

    Ok(ndarray_to_nested_json(&output.into_dyn()))
}

fn reshape_to_4d(flat: &[f64], c: usize, tile_size: usize) -> Result<Array4<f64>> {
    let n = 1usize;
    let total = flat.len();
    let spatial = if c > 0 { total / (n * c) } else { 0 };
    let h_sqrt = (spatial as f64).sqrt() as usize;
    let (h, w) = if h_sqrt > 0 && h_sqrt * h_sqrt == spatial {
        (h_sqrt, h_sqrt)
    } else if tile_size > 0 && spatial > 0 {
        (tile_size, spatial / tile_size)
    } else {
        (0, 0)
    };

    if n * c * h * w != total {
        let h = tile_size;
        let w = if c > 0 && h > 0 { total / (n * c * h) } else { 0 };
        if n * c * h * w != total {
            return Err(DsperseError::Pipeline(format!(
                "cannot reshape {total} elements to 4D (c={c})"
            )));
        }
        Array4::from_shape_vec((n, c, h, w), flat.to_vec())
            .map_err(|e| DsperseError::Pipeline(format!("reshape: {e}")))
    } else {
        Array4::from_shape_vec((n, c, h, w), flat.to_vec())
            .map_err(|e| DsperseError::Pipeline(format!("reshape: {e}")))
    }
}

fn ndarray_to_nested_json(arr: &ArrayD<f64>) -> serde_json::Value {
    match arr.ndim() {
        0 => serde_json::json!(arr[IxDyn(&[])]),
        1 => {
            let vals: Vec<serde_json::Value> = arr
                .iter()
                .map(|&v| serde_json::json!(v))
                .collect();
            serde_json::Value::Array(vals)
        }
        _ => {
            let vals: Vec<serde_json::Value> = (0..arr.shape()[0])
                .map(|i| {
                    let sub = arr.index_axis(Axis(0), i).to_owned();
                    ndarray_to_nested_json(&sub)
                })
                .collect();
            serde_json::Value::Array(vals)
        }
    }
}

fn extract_output_tensor(data: &serde_json::Value) -> serde_json::Value {
    data.get("output_data")
        .or_else(|| data.get("output"))
        .cloned()
        .unwrap_or_else(|| data.clone())
}

fn gather_inputs(
    tensor_cache: &HashMap<String, serde_json::Value>,
    inputs: &[String],
) -> Result<serde_json::Value> {
    for name in inputs {
        if let Some(val) = tensor_cache.get(name) {
            return Ok(val.clone());
        }
    }
    Err(DsperseError::Pipeline(format!(
        "no cached tensor found for inputs: {inputs:?}"
    )))
}

fn store_outputs(
    tensor_cache: &mut HashMap<String, serde_json::Value>,
    output_names: &[String],
    output_data: &serde_json::Value,
) -> Result<()> {
    let data = extract_output_tensor(output_data);
    for name in output_names {
        tensor_cache.insert(name.clone(), data.clone());
    }
    Ok(())
}

fn infer_shape(value: &serde_json::Value) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut current = value;
    loop {
        match current {
            serde_json::Value::Array(arr) => {
                shape.push(arr.len());
                if let Some(first) = arr.first() {
                    current = first;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    shape
}

fn run_onnx_inference(
    onnx_path: &Path,
    input: &serde_json::Value,
) -> Result<serde_json::Value> {
    let input_flat = flatten_nested_list(input);
    let input_shape = infer_shape(input);
    let shape_ref = if input_shape.iter().product::<usize>() == input_flat.len() && !input_shape.is_empty() {
        &input_shape[..]
    } else {
        &[]
    };
    let (output_data, output_shape) =
        crate::backend::onnx::run_inference(onnx_path, &input_flat, shape_ref)?;

    let output_arr = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&output_shape),
        output_data,
    )
    .map_err(|e| DsperseError::Pipeline(format!("output reshape: {e}")))?;

    let output_json = ndarray_to_nested_json(&output_arr);
    Ok(serde_json::json!({ "output_data": output_json }))
}

fn build_execution_chain(model_meta: &ModelMetadata, slices_dir: &Path) -> ExecutionChain {
    let mut nodes = HashMap::new();
    let mut head = None;

    for (i, slice) in model_meta.slices.iter().enumerate() {
        let slice_id = format!("slice_{}", slice.index);
        let slice_dir = slice_dir_path(slices_dir, slice.index);

        if i == 0 {
            head = Some(slice_id.clone());
        }

        let has_circuit = slice.compilation.jstprove.compiled;
        let next = model_meta
            .slices
            .get(i + 1)
            .map(|s| format!("slice_{}", s.index));

        let circuit_path = if has_circuit {
            slice
                .compilation
                .jstprove
                .files
                .compiled
                .as_ref()
                .map(|p| {
                    slice_dir
                        .join("jstprove")
                        .join(p)
                        .to_string_lossy()
                        .into_owned()
                })
        } else {
            None
        };

        let onnx_path = Some(
            slice_dir
                .join(&slice.path)
                .to_string_lossy()
                .into_owned(),
        );

        let backend = if has_circuit { "jstprove" } else { "onnx" };

        nodes.insert(
            slice_id.clone(),
            ExecutionNode {
                slice_id: slice_id.clone(),
                primary: if has_circuit {
                    Some("jstprove".into())
                } else {
                    Some("onnx".into())
                },
                fallbacks: if has_circuit {
                    vec!["onnx".into()]
                } else {
                    Vec::new()
                },
                use_circuit: has_circuit,
                next,
                circuit_path,
                onnx_path,
                backend: backend.into(),
            },
        );
    }

    ExecutionChain {
        head,
        nodes,
        fallback_map: HashMap::new(),
        execution_results: Vec::new(),
        jstprove_proved_slices: 0,
        jstprove_verified_slices: 0,
    }
}

fn build_run_metadata(
    model_meta: &ModelMetadata,
    slices_dir: &Path,
    _run_dir: &Path,
    chain: &ExecutionChain,
) -> RunMetadata {
    let mut slices = HashMap::new();
    let mut circuit_slices = HashMap::new();

    for slice in &model_meta.slices {
        let slice_id = format!("slice_{}", slice.index);
        let slice_dir = slice_dir_path(slices_dir, slice.index);
        let node = chain.nodes.get(&slice_id);
        let has_circuit = node.is_some_and(|n| n.use_circuit);

        let run_slice = RunSliceMetadata {
            path: slice_dir
                .join(&slice.path)
                .to_string_lossy()
                .into_owned(),
            input_shape: slice.shape.tensor_shape.input.clone(),
            output_shape: slice.shape.tensor_shape.output.clone(),
            dependencies: slice.dependencies.clone(),
            tiling: slice.tiling.clone(),
            channel_split: slice.channel_split.clone(),
            backend: if has_circuit {
                "jstprove".into()
            } else {
                "onnx".into()
            },
            circuit_path: node.and_then(|n| n.circuit_path.clone()),
            settings_path: None,
            vk_path: None,
            pk_path: None,
            jstprove_circuit_path: node.and_then(|n| n.circuit_path.clone()),
            jstprove_settings_path: None,
        };

        if has_circuit {
            circuit_slices.insert(slice_id.clone(), true);
        }

        slices.insert(slice_id, run_slice);
    }

    RunMetadata {
        slices,
        execution_chain: chain.clone(),
        circuit_slices,
        overall_security: 0.0,
        packaging_type: None,
        source_path: Some(slices_dir.to_string_lossy().into_owned()),
        run_directory: None,
        model_path: None,
    }
}
