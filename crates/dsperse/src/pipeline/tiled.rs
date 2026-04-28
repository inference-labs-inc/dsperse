use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array4, ArrayD, IxDyn, s};
use rayon::prelude::*;

use super::tensor_store::TensorStore;
use crate::backend::jstprove::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{ExecutionInfo, ExecutionMethod, TileResult};
use crate::schema::tiling::TilingInfo;
use crate::slicer::onnx_proto::TensorProto;
use crate::utils::paths::resolve_relative_path;

use super::runner::{
    RunConfig, extract_initializers_from_map, extract_onnx_initializers,
    resolve_circuit_path_optional, run_onnx_inference,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_tiled(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    tiling: &TilingInfo,
    slice_circuit_path: Option<&Path>,
    tensor_cache: &TensorStore,
    backend: &JstproveBackend,
    config: &RunConfig,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<crate::schema::execution::StrategyOutput> {
    let all_names = tiling.all_input_names();
    let multi_input = all_names.len() > 1;
    let is_fixed_segment = tiling.ndim == 1;
    let is_1d = tiling.ndim == 3;

    let all_tiles_dyn = if is_fixed_segment {
        prepare_fixed_segments_from_cache(tiling, tensor_cache)?
    } else {
        prepare_tiles_from_cache(tiling, tensor_cache, is_1d)?
    };

    let num_tiles = all_tiles_dyn[0].len();

    tracing::info!(
        slice = %slice_id,
        num_tiles,
        tile_size = tiling.tile_size,
        ndim = tiling.ndim,
        "splitting into tiles"
    );

    let tile_infos = tiling.tiles.as_deref().unwrap_or(&[]);
    let single_tile = tiling.tile.as_ref();

    if tile_infos.is_empty() && single_tile.is_none() {
        return Err(DsperseError::Pipeline(format!(
            "tiling for '{}' has neither tile list nor single tile template",
            tiling.output_name
        )));
    }

    let first_tile_info = tile_infos.first().or(single_tile);
    let first_tile_onnx = first_tile_info
        .map(|ti| resolve_relative_path(slices_dir, &ti.path))
        .transpose()?;

    let warm_model = if multi_input || is_1d || is_fixed_segment {
        None
    } else {
        match (first_tile_onnx.as_deref(), all_tiles_dyn[0].first()) {
            (Some(onnx_path), Some(sample)) => {
                let shape = sample.shape().to_vec();
                let model = crate::backend::onnx::WarmModel::load(onnx_path, &shape)?;
                tracing::info!(slice = %slice_id, "loaded ONNX model");
                Some(model)
            }
            _ => None,
        }
    };

    let circuit_path = resolve_circuit_path_optional(
        slices_dir,
        first_tile_info.and_then(|ti| ti.jstprove_circuit_path.as_deref()),
    )?
    .or_else(|| slice_circuit_path.map(|p| p.to_path_buf()));

    let warm_circuit = match (&circuit_path, &first_tile_onnx) {
        (Some(cp), Some(onnx_path)) => {
            let params = backend.load_params(cp)?;
            let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

            if donor_init_map.is_some() && !is_wai {
                return Err(DsperseError::Pipeline(format!(
                    "{slice_id}: consumer weights require circuits compiled with --weights-as-inputs"
                )));
            }

            let initializers = if is_wai {
                if let Some(map) = donor_init_map {
                    extract_initializers_from_map(map, params.as_ref().unwrap())?
                } else {
                    extract_onnx_initializers(onnx_path, params.as_ref().unwrap())?
                }
            } else {
                vec![]
            };
            let wc = crate::backend::jstprove::WarmCircuit::load(cp, initializers, backend)?;
            tracing::info!(slice = %slice_id, wai = is_wai, "loaded circuit bundle");
            Some(wc)
        }
        _ => None,
    };

    let warm_model = warm_model.map(Arc::new);
    let warm_circuit = warm_circuit.map(Arc::new);
    let circuit_path = circuit_path.map(Arc::from);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let tile_input_names: Vec<String> = if all_names.len() > 1 {
        (0..all_names.len())
            .map(|i| format!("tile_in_{i}"))
            .collect()
    } else {
        vec!["tile_in".to_string()]
    };

    let collected: Vec<(TileResult, Option<ArrayD<f64>>)> = pool.install(|| {
        (0..num_tiles)
            .into_par_iter()
            .map(|tile_idx| {
                let start = std::time::Instant::now();
                let tile_dir = slice_run_dir.join(format!("tile_{tile_idx}"));
                if let Err(e) = std::fs::create_dir_all(&tile_dir) {
                    return (
                        TileResult::failure(
                            tile_idx,
                            format!("mkdir: {e}"),
                            None,
                            start.elapsed().as_secs_f64(),
                        ),
                        None,
                    );
                }

                let tile_info = tile_infos.get(tile_idx).or(single_tile);
                let tile_dyn = all_tiles_dyn[0][tile_idx].clone();

                let per_tile_onnx = tile_info
                    .map(|ti| resolve_relative_path(slices_dir, &ti.path))
                    .transpose();
                let per_tile_onnx = match per_tile_onnx {
                    Ok(p) => p,
                    Err(e) => {
                        return (
                            TileResult::failure(
                                tile_idx,
                                format!("resolve tile path: {e}"),
                                None,
                                start.elapsed().as_secs_f64(),
                            ),
                            None,
                        );
                    }
                };
                let effective_tile_onnx_ref = per_tile_onnx.as_deref();

                if tile_info.is_none() {
                    return (
                        TileResult::failure(
                            tile_idx,
                            "no tile circuit info".into(),
                            None,
                            start.elapsed().as_secs_f64(),
                        ),
                        None,
                    );
                }

                let tile_output = if multi_input || is_1d || is_fixed_segment {
                    if let Some(onnx) = effective_tile_onnx_ref {
                        let inputs: Vec<(&str, Vec<f64>, Vec<usize>)> = all_tiles_dyn
                            .iter()
                            .zip(tile_input_names.iter())
                            .map(|(input_tiles, tile_name)| {
                                let t = &input_tiles[tile_idx];
                                let shape: Vec<usize> = t.shape().to_vec();
                                let data: Vec<f64> = t.iter().copied().collect();
                                (tile_name.as_str(), data, shape)
                            })
                            .collect();
                        crate::backend::onnx::run_inference_multi_named(onnx, &inputs).and_then(
                            |named| {
                                let (data, shape) =
                                    named.into_values().next().ok_or_else(|| {
                                        DsperseError::Pipeline(
                                            "multi-input tile produced no output".into(),
                                        )
                                    })?;
                                ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
                                    DsperseError::Pipeline(format!(
                                        "multi-input tile output reshape: {e}"
                                    ))
                                })
                            },
                        )
                    } else {
                        Err(DsperseError::Pipeline(format!(
                            "tile {tile_idx}: no ONNX model available for inference"
                        )))
                    }
                } else if let Some(ref wm) = warm_model {
                    let input_flat: Vec<f64> = tile_dyn.iter().copied().collect();
                    wm.run(&input_flat).and_then(|(data, shape)| {
                        ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
                            crate::error::DsperseError::Pipeline(format!(
                                "warm model output reshape: {e}"
                            ))
                        })
                    })
                } else if let Some(onnx) = effective_tile_onnx_ref {
                    run_onnx_inference(onnx, &tile_dyn)
                } else {
                    Err(DsperseError::Pipeline(format!(
                        "tile {tile_idx}: no ONNX model available for inference"
                    )))
                };

                let output_tensor = match tile_output {
                    Ok(t) => t,
                    Err(e) => {
                        return (
                            TileResult::failure(
                                tile_idx,
                                format!("onnx inference: {e}"),
                                Some(ExecutionMethod::OnnxOnly),
                                start.elapsed().as_secs_f64(),
                            ),
                            None,
                        );
                    }
                };

                if circuit_path.is_none() {
                    return (
                        TileResult::success(
                            tile_idx,
                            Some(ExecutionMethod::OnnxOnly),
                            start.elapsed().as_secs_f64(),
                        ),
                        Some(output_tensor),
                    );
                }

                let flat: Vec<f64> = flatten_tile_inputs(&all_tiles_dyn, tile_idx);
                let witness_result = if let Some(ref wc) = warm_circuit {
                    wc.witness_f64(&flat)
                } else {
                    let cp = circuit_path
                        .as_ref()
                        .expect("circuit_path is Some: guarded by early return");
                    backend.witness_f64(cp, &flat, &[])
                };

                match witness_result {
                    Ok(witness_bytes) => {
                        let witness_path = tile_dir.join(crate::utils::paths::WITNESS_FILE);
                        if let Err(e) = std::fs::write(&witness_path, &witness_bytes) {
                            return (
                                TileResult::failure(
                                    tile_idx,
                                    format!("write witness: {e}"),
                                    Some(ExecutionMethod::JstproveGenWitness),
                                    start.elapsed().as_secs_f64(),
                                ),
                                None,
                            );
                        }
                        (
                            TileResult::success(
                                tile_idx,
                                Some(ExecutionMethod::JstproveGenWitness),
                                start.elapsed().as_secs_f64(),
                            ),
                            Some(output_tensor),
                        )
                    }
                    Err(e) => (
                        TileResult::failure(
                            tile_idx,
                            e.to_string(),
                            Some(ExecutionMethod::JstproveGenWitness),
                            start.elapsed().as_secs_f64(),
                        ),
                        None,
                    ),
                }
            })
            .collect()
    });

    let mut tile_results: Vec<TileResult> = Vec::with_capacity(collected.len());
    let mut tile_outputs: Vec<ArrayD<f64>> = Vec::with_capacity(collected.len());
    for (result, output) in collected {
        if let Some(o) = output {
            tile_outputs.push(o);
        }
        tile_results.push(result);
    }

    if tile_results.is_empty() {
        return Err(DsperseError::Pipeline(format!(
            "tiling produced zero tiles for '{}'",
            tiling.output_name
        )));
    }

    let all_success = tile_results.iter().all(|r| r.success);

    if !all_success {
        let failed: Vec<_> = tile_results
            .iter()
            .filter(|r| !r.success)
            .map(|r| format!("tile {}: {}", r.tile_idx, r.error.as_deref().unwrap_or("?")))
            .collect();
        return Err(DsperseError::Pipeline(format!(
            "tiled execution failed for '{}': {}",
            tiling.output_name,
            failed.join("; ")
        )));
    }

    debug_assert!(
        !tile_outputs.is_empty(),
        "all tiles reported success but no outputs for '{}'",
        tiling.output_name
    );
    let reconstructed = if is_fixed_segment {
        reconstruct_from_fixed_segments(&tile_outputs, tiling)?
    } else if is_1d {
        let r = reconstruct_from_tiles_1d(&tile_outputs, tiling)?;
        trim_to_original_seq(r, tiling)?
    } else {
        let r = reconstruct_from_tiles(&tile_outputs, tiling)?;
        trim_to_original_dims(r, tiling)?
    };
    Ok(crate::schema::execution::StrategyOutput {
        info: ExecutionInfo {
            method: ExecutionMethod::Tiled,
            success: true,
            error: None,
            witness_file: None,
            tile_exec_infos: tile_results,
        },
        outputs: vec![(tiling.output_name.clone(), reconstructed)],
    })
}

/// Witness-only tiled execution for combined inference mode.
///
/// The full-model ONNX inference has already run and populated the tensor
/// cache with all intermediate activations. This function splits those
/// cached activations into tiles, generates per-tile ZK witnesses via the
/// circuit backend, and returns tile-level execution results. It does NOT
/// reconstruct output tensors — those already exist in the cache from the
/// monolithic inference pass — hence the empty `outputs` vec in the
/// returned `StrategyOutput`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_combined_tiled(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    tiling: &TilingInfo,
    slice_circuit_path: Option<&str>,
    tensor_cache: &TensorStore,
    backend: &JstproveBackend,
    config: &RunConfig,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<crate::schema::execution::StrategyOutput> {
    let is_fixed_segment = tiling.ndim == 1;
    let is_1d = tiling.ndim == 3;
    let all_tiles_dyn = if is_fixed_segment {
        prepare_fixed_segments_from_cache(tiling, tensor_cache)?
    } else {
        prepare_tiles_from_cache(tiling, tensor_cache, is_1d)?
    };

    let num_tiles = all_tiles_dyn[0].len();

    tracing::info!(
        slice = %slice_id,
        num_tiles,
        tile_size = tiling.tile_size,
        "splitting combined activations into tiles for witness generation"
    );

    let tile_infos = tiling.tiles.as_deref().unwrap_or(&[]);
    let single_tile = tiling.tile.as_ref();
    let first_tile_info = tile_infos.first().or(single_tile);

    let circuit_path = resolve_circuit_path_optional(
        slices_dir,
        first_tile_info
            .and_then(|ti| ti.jstprove_circuit_path.as_deref())
            .or(slice_circuit_path),
    )?;

    let circuit_path = match circuit_path {
        Some(p) => p,
        None => {
            return Ok(crate::schema::execution::StrategyOutput {
                info: ExecutionInfo {
                    method: ExecutionMethod::Tiled,
                    success: true,
                    error: None,
                    witness_file: None,
                    tile_exec_infos: (0..num_tiles)
                        .map(|i| TileResult::success(i, Some(ExecutionMethod::OnnxOnly), 0.0))
                        .collect(),
                },
                outputs: vec![],
            });
        }
    };

    let first_tile_onnx = first_tile_info
        .map(|ti| resolve_relative_path(slices_dir, &ti.path))
        .transpose()?;

    let patched_tile_onnx = match (&first_tile_onnx, donor_init_map) {
        (Some(onnx_path), Some(map)) => Some(crate::slicer::onnx_proto::build_patched_onnx(
            onnx_path, map,
        )?),
        _ => None,
    };
    let effective_tile_onnx = patched_tile_onnx.as_ref().map(|t| t.path().to_path_buf());
    let effective_tile_onnx_ref = effective_tile_onnx
        .as_deref()
        .or(first_tile_onnx.as_deref());

    let params = backend.load_params(&circuit_path)?;
    let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

    if donor_init_map.is_some() && !is_wai {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: consumer weights require circuits compiled with --weights-as-inputs"
        )));
    }

    let warm_circuit = match effective_tile_onnx_ref {
        Some(onnx_path) => {
            let initializers = if is_wai {
                if let Some(map) = donor_init_map {
                    extract_initializers_from_map(map, params.as_ref().unwrap())?
                } else {
                    extract_onnx_initializers(onnx_path, params.as_ref().unwrap())?
                }
            } else {
                vec![]
            };
            let wc =
                crate::backend::jstprove::WarmCircuit::load(&circuit_path, initializers, backend)?;
            tracing::info!(slice = %slice_id, wai = is_wai, "loaded tile circuit for combined tiling");
            Some(wc)
        }
        None => None,
    };

    let warm_circuit = warm_circuit.map(Arc::new);
    let circuit_path = Arc::from(circuit_path);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let collected: Vec<TileResult> = pool.install(|| {
        (0..num_tiles)
            .into_par_iter()
            .map(|tile_idx| {
                let start = std::time::Instant::now();
                let tile_dir = slice_run_dir.join(format!("tile_{tile_idx}"));
                if let Err(e) = std::fs::create_dir_all(&tile_dir) {
                    return TileResult::failure(
                        tile_idx,
                        format!("mkdir: {e}"),
                        None,
                        start.elapsed().as_secs_f64(),
                    );
                }

                let flat: Vec<f64> = flatten_tile_inputs(&all_tiles_dyn, tile_idx);

                let witness_result = if let Some(ref wc) = warm_circuit {
                    wc.witness_f64(&flat)
                } else {
                    backend.witness_f64(&circuit_path, &flat, &[])
                };

                match witness_result {
                    Ok(witness_bytes) => {
                        let witness_path = tile_dir.join(crate::utils::paths::WITNESS_FILE);
                        if let Err(e) = std::fs::write(&witness_path, &witness_bytes) {
                            return TileResult::failure(
                                tile_idx,
                                format!("write witness: {e}"),
                                Some(ExecutionMethod::JstproveGenWitness),
                                start.elapsed().as_secs_f64(),
                            );
                        }
                        TileResult::success(
                            tile_idx,
                            Some(ExecutionMethod::JstproveGenWitness),
                            start.elapsed().as_secs_f64(),
                        )
                    }
                    Err(e) => TileResult::failure(
                        tile_idx,
                        e.to_string(),
                        Some(ExecutionMethod::JstproveGenWitness),
                        start.elapsed().as_secs_f64(),
                    ),
                }
            })
            .collect()
    });

    let all_success = collected.iter().all(|r| r.success);
    if !all_success {
        let failed: Vec<_> = collected
            .iter()
            .filter(|r| !r.success)
            .map(|r| format!("tile {}: {}", r.tile_idx, r.error.as_deref().unwrap_or("?")))
            .collect();
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: tiled witness generation failed: {}",
            failed.join("; ")
        )));
    }

    tracing::info!(
        slice = %slice_id,
        num_tiles,
        "tiled witness generation from combined outputs complete"
    );

    // No output tensors: combined mode already has activations in cache
    // from the monolithic ONNX run. Only witness artifacts are produced here.
    Ok(crate::schema::execution::StrategyOutput {
        info: ExecutionInfo {
            method: ExecutionMethod::Tiled,
            success: true,
            error: None,
            witness_file: None,
            tile_exec_infos: collected,
        },
        outputs: vec![],
    })
}

pub(crate) fn prepare_tiles_from_cache(
    tiling: &TilingInfo,
    tensor_cache: &TensorStore,
    is_1d: bool,
) -> Result<Vec<Vec<ArrayD<f64>>>> {
    let all_names = tiling.all_input_names();
    let mut all_tiles: Vec<Vec<ArrayD<f64>>> = Vec::with_capacity(all_names.len());
    for name in &all_names {
        let input_arr = tensor_cache.get(name)?.clone();
        if is_1d {
            let tiles = split_into_tiles_1d(&input_arr, tiling)?;
            all_tiles.push(tiles);
        } else {
            let input_4d = if input_arr.ndim() == 4 {
                let s = input_arr.shape();
                Array4::from_shape_vec(
                    (s[0], s[1], s[2], s[3]),
                    input_arr.iter().copied().collect(),
                )
                .map_err(|e| DsperseError::Pipeline(format!("tiling input reshape: {e}")))?
            } else {
                let input_flat: Vec<f64> = input_arr.iter().copied().collect();
                let h = if tiling.h > 0 {
                    tiling.h
                } else {
                    tiling.tiles_y * tiling.tile_size
                };
                let w = if tiling.w > 0 {
                    tiling.w
                } else {
                    tiling.tiles_x * tiling.tile_size
                };
                reshape_to_4d(&input_flat, tiling.c_in, h, w)?
            };
            let tiles = split_into_tiles(&input_4d, tiling)?;
            all_tiles.push(tiles.into_iter().map(|t| t.into_dyn()).collect());
        }
    }
    Ok(all_tiles)
}

/// Build per-tile dispatch payloads for a tiled slice that consumes multiple
/// upstream activation tensors per witness. Returns one entry per tile; each
/// entry is the concatenation, in `tiling.all_input_names()` order, of that
/// tile's segment from each input tensor. The element count of every returned
/// entry equals `N * per_input_tile_size`, matching what the slice's compiled
/// witness solver consumes per request.
pub fn split_for_multi_input_dispatch(
    named_inputs: &[(String, ArrayD<f64>)],
    tiling: &TilingInfo,
) -> Result<Vec<Vec<f64>>> {
    if named_inputs.is_empty() {
        return Err(DsperseError::Pipeline(
            "split_for_multi_input_dispatch: named_inputs is empty".into(),
        ));
    }
    let expected_names = tiling.all_input_names();
    let provided: std::collections::HashSet<&str> =
        named_inputs.iter().map(|(n, _)| n.as_str()).collect();
    let missing: Vec<&str> = expected_names
        .iter()
        .copied()
        .filter(|n| !provided.contains(n))
        .collect();
    if !missing.is_empty() {
        return Err(DsperseError::Pipeline(format!(
            "split_for_multi_input_dispatch: missing input tensors required by tiling.all_input_names(): {missing:?}"
        )));
    }
    let mut cache = TensorStore::new();
    for (name, arr) in named_inputs {
        cache.put(name.clone(), arr.clone());
    }
    let is_fixed_segment = tiling.ndim == 1;
    let is_1d = tiling.ndim == 3;
    let all_tiles = if is_fixed_segment {
        prepare_fixed_segments_from_cache(tiling, &cache)?
    } else {
        prepare_tiles_from_cache(tiling, &cache, is_1d)?
    };
    if all_tiles.is_empty() {
        return Err(DsperseError::Pipeline(
            "split_for_multi_input_dispatch: empty tile set".into(),
        ));
    }
    let num_tiles = all_tiles[0].len();
    for (idx, per_input) in all_tiles.iter().enumerate() {
        if per_input.len() != num_tiles {
            return Err(DsperseError::Pipeline(format!(
                "split_for_multi_input_dispatch: input {idx} produced {} tiles, expected {num_tiles}",
                per_input.len()
            )));
        }
    }
    let mut per_tile = Vec::with_capacity(num_tiles);
    for tile_idx in 0..num_tiles {
        per_tile.push(flatten_tile_inputs(&all_tiles, tile_idx));
    }
    Ok(per_tile)
}

pub fn split_for_tiling(input: &ArrayD<f64>, tiling: &TilingInfo) -> Result<Vec<ArrayD<f64>>> {
    let is_fixed_segment = tiling.ndim == 1;
    if is_fixed_segment {
        let segment_size = tiling.segment_size.ok_or_else(|| {
            DsperseError::Pipeline("split_for_tiling: fixed segment missing segment_size".into())
        })?;
        if segment_size == 0 {
            return Err(DsperseError::Pipeline(
                "split_for_tiling: segment_size must be > 0".into(),
            ));
        }
        let total_elements = tiling.total_elements.ok_or_else(|| {
            DsperseError::Pipeline("split_for_tiling: fixed segment missing total_elements".into())
        })?;
        let flat: Vec<f64> = input.iter().copied().collect();
        if flat.len() < total_elements {
            return Err(DsperseError::Pipeline(format!(
                "split_for_tiling: input has {} elements, expected at least {}",
                flat.len(),
                total_elements
            )));
        }
        let num_segments = total_elements.div_ceil(segment_size);
        let mut segments = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let start = i * segment_size;
            if start >= flat.len() {
                break;
            }
            let end = (start + segment_size).min(total_elements);
            let mut seg_data = vec![0.0f64; segment_size];
            seg_data[..end - start].copy_from_slice(&flat[start..end]);
            segments.push(
                ArrayD::from_shape_vec(IxDyn(&[segment_size]), seg_data)
                    .map_err(|e| DsperseError::Pipeline(format!("segment reshape: {e}")))?,
            );
        }
        return Ok(segments);
    }
    let is_1d = tiling.ndim == 3;
    if is_1d {
        return split_into_tiles_1d(input, tiling);
    }
    let input_4d = if input.ndim() == 4 {
        let s = input.shape();
        Array4::from_shape_vec((s[0], s[1], s[2], s[3]), input.iter().copied().collect())
            .map_err(|e| DsperseError::Pipeline(format!("tiling input reshape: {e}")))?
    } else {
        let flat: Vec<f64> = input.iter().copied().collect();
        let h = if tiling.h > 0 {
            tiling.h
        } else {
            tiling.tiles_y * tiling.tile_size
        };
        let w = if tiling.w > 0 {
            tiling.w
        } else {
            tiling.tiles_x * tiling.tile_size
        };
        reshape_to_4d(&flat, tiling.c_in, h, w)?
    };
    let tiles = split_into_tiles(&input_4d, tiling)?;
    Ok(tiles.into_iter().map(|t| t.into_dyn()).collect())
}

pub fn split_into_tiles(input: &Array4<f64>, tiling: &TilingInfo) -> Result<Vec<Array4<f64>>> {
    if tiling.halo.iter().any(|&v| v < 0) {
        return Err(DsperseError::Pipeline(format!(
            "negative halo values not supported: halo={:?}",
            tiling.halo
        )));
    }
    let (n, c, h, w) = input.dim();
    if n != 1 {
        return Err(DsperseError::Pipeline(format!(
            "split_into_tiles: batch size {n} not supported, expected 1"
        )));
    }
    let halo_top = tiling.halo[0] as usize;
    let halo_left = tiling.halo[1] as usize;
    let halo_bottom = tiling.halo[2] as usize;
    let halo_right = tiling.halo[3] as usize;
    let tile_h = tiling.tile_size + halo_top + halo_bottom;
    let tile_w = tiling.tile_size + halo_left + halo_right;

    let padded_h = tiling.tiles_y * tiling.tile_size + halo_top + halo_bottom;
    let padded_w = tiling.tiles_x * tiling.tile_size + halo_left + halo_right;
    if halo_top + h > padded_h || halo_left + w > padded_w {
        return Err(DsperseError::Pipeline(format!(
            "split_into_tiles: input spatial ({h}x{w}) exceeds padded grid ({padded_h}x{padded_w})"
        )));
    }
    let mut padded = Array4::<f64>::zeros((n, c, padded_h, padded_w));
    padded
        .slice_mut(s![.., .., halo_top..halo_top + h, halo_left..halo_left + w])
        .assign(input);

    let mut tiles = Vec::new();
    for ty in 0..tiling.tiles_y {
        for tx in 0..tiling.tiles_x {
            let y_start = ty * tiling.tile_size;
            let x_start = tx * tiling.tile_size;
            let tile = padded
                .slice(s![
                    ..,
                    ..,
                    y_start..y_start + tile_h,
                    x_start..x_start + tile_w
                ])
                .to_owned();
            tiles.push(tile);
        }
    }

    Ok(tiles)
}

pub fn reconstruct_from_tiles(
    tile_outputs: &[ArrayD<f64>],
    tiling: &TilingInfo,
) -> Result<ArrayD<f64>> {
    let expected_tiles = tiling.tiles_y * tiling.tiles_x;
    if tile_outputs.len() != expected_tiles {
        return Err(DsperseError::Pipeline(format!(
            "reconstruct: expected {} tiles ({}x{}), got {}",
            expected_tiles,
            tiling.tiles_y,
            tiling.tiles_x,
            tile_outputs.len()
        )));
    }

    let out_h = tiling.out_tile[0].max(1) as usize;
    let out_w = tiling.out_tile[1].max(1) as usize;
    let c_out = tiling.c_out;
    let total_h = out_h * tiling.tiles_y;
    let total_w = out_w * tiling.tiles_x;

    let mut output = Array4::<f64>::zeros((1, c_out, total_h, total_w));

    for (idx, tile_arr) in tile_outputs.iter().enumerate() {
        let ty = idx / tiling.tiles_x;
        let tx = idx % tiling.tiles_x;

        let tile_flat: Vec<f64> = tile_arr.iter().copied().collect();
        if tile_flat.is_empty() {
            return Err(DsperseError::Pipeline(format!(
                "tile ({},{}) marked successful but produced no data",
                ty, tx
            )));
        }

        let tile_elements = c_out * out_h * out_w;
        if tile_flat.len() != tile_elements {
            return Err(DsperseError::Pipeline(format!(
                "tile ({},{}) has {} elements, expected {} (c_out={}, out_h={}, out_w={})",
                ty,
                tx,
                tile_flat.len(),
                tile_elements,
                c_out,
                out_h,
                out_w
            )));
        }

        let tile_4d = Array4::from_shape_vec((1, c_out, out_h, out_w), tile_flat.to_vec())
            .map_err(|e| {
                DsperseError::Pipeline(format!("tile ({},{}) reshape failed: {e}", ty, tx))
            })?;
        let y_start = ty * out_h;
        let x_start = tx * out_w;
        output
            .slice_mut(s![
                ..,
                ..,
                y_start..y_start + out_h,
                x_start..x_start + out_w
            ])
            .assign(&tile_4d);
    }

    Ok(output.into_dyn())
}

pub(crate) fn trim_to_original_dims(arr: ArrayD<f64>, tiling: &TilingInfo) -> Result<ArrayD<f64>> {
    if tiling.h == 0 || tiling.w == 0 {
        return Ok(arr);
    }
    let stride_h = tiling.stride[0].max(1) as usize;
    let stride_w = tiling.stride[1].max(1) as usize;
    let expected_h = tiling.h / stride_h;
    let expected_w = tiling.w / stride_w;
    let grid_h = tiling.out_tile[0].max(1) as usize * tiling.tiles_y;
    let grid_w = tiling.out_tile[1].max(1) as usize * tiling.tiles_x;
    if grid_h > expected_h || grid_w > expected_w {
        if arr.ndim() != 4 {
            return Err(DsperseError::Pipeline(format!(
                "trim_to_original_dims: expected 4D array, got {}D",
                arr.ndim()
            )));
        }
        Ok(arr
            .slice(s![.., .., ..expected_h, ..expected_w])
            .to_owned()
            .into_dyn())
    } else {
        Ok(arr)
    }
}

pub(crate) fn split_into_tiles_1d(
    input: &ArrayD<f64>,
    tiling: &TilingInfo,
) -> Result<Vec<ArrayD<f64>>> {
    let shape = input.shape();
    if shape.len() != 3 {
        return Err(DsperseError::Pipeline(format!(
            "split_into_tiles_1d: expected 3D input, got {}D",
            shape.len()
        )));
    }
    let (n, seq, _hidden) = (shape[0], shape[1], shape[2]);
    if n != 1 {
        return Err(DsperseError::Pipeline(format!(
            "split_into_tiles_1d: batch size {n} not supported, expected 1"
        )));
    }
    let tile_size = tiling.tile_size;
    if tile_size == 0 || tiling.tiles_y == 0 {
        return Err(DsperseError::Pipeline(format!(
            "split_into_tiles_1d: invalid tiling config tile_size={}, tiles_y={}",
            tile_size, tiling.tiles_y
        )));
    }
    let padded_seq = tiling
        .tiles_y
        .checked_mul(tile_size)
        .ok_or_else(|| DsperseError::Pipeline("split_into_tiles_1d: padded_seq overflow".into()))?;
    if seq > padded_seq {
        return Err(DsperseError::Pipeline(format!(
            "split_into_tiles_1d: input seq {seq} exceeds padded seq {padded_seq}"
        )));
    }
    let mut padded = ArrayD::<f64>::zeros(vec![n, padded_seq, shape[2]]);
    padded.slice_mut(s![.., ..seq, ..]).assign(input);

    let mut tiles = Vec::with_capacity(tiling.tiles_y);
    for ty in 0..tiling.tiles_y {
        let start = ty * tile_size;
        let tile = padded
            .slice(s![.., start..start + tile_size, ..])
            .to_owned()
            .into_dyn();
        tiles.push(tile);
    }
    Ok(tiles)
}

pub(crate) fn reconstruct_from_tiles_1d(
    tile_outputs: &[ArrayD<f64>],
    tiling: &TilingInfo,
) -> Result<ArrayD<f64>> {
    if tile_outputs.is_empty() {
        return Err(DsperseError::Pipeline(
            "reconstruct_1d: no tile outputs".into(),
        ));
    }
    if tile_outputs.len() != tiling.tiles_y {
        return Err(DsperseError::Pipeline(format!(
            "reconstruct_1d: expected {} tiles, got {}",
            tiling.tiles_y,
            tile_outputs.len()
        )));
    }
    let first = &tile_outputs[0];
    if first.ndim() != 3 {
        return Err(DsperseError::Pipeline(format!(
            "reconstruct_1d: expected 3D tiles, got {}D",
            first.ndim()
        )));
    }
    let fshape = first.shape();
    let (tile_len, hidden) = (fshape[1], fshape[2]);
    let total_seq = tile_len * tile_outputs.len();
    let mut output = ArrayD::<f64>::zeros(vec![1, total_seq, hidden]);
    for (idx, tile) in tile_outputs.iter().enumerate() {
        if tile.shape() != fshape {
            return Err(DsperseError::Pipeline(format!(
                "reconstruct_1d: tile {idx} shape {:?} != first tile shape {:?}",
                tile.shape(),
                fshape
            )));
        }
        let start = idx * tile_len;
        output
            .slice_mut(s![.., start..start + tile_len, ..])
            .assign(tile);
    }
    Ok(output)
}

pub(crate) fn trim_to_original_seq(arr: ArrayD<f64>, tiling: &TilingInfo) -> Result<ArrayD<f64>> {
    if tiling.h == 0 {
        return Ok(arr);
    }
    if arr.ndim() != 3 {
        return Err(DsperseError::Pipeline(format!(
            "trim_to_original_seq: expected 3D array, got {}D",
            arr.ndim()
        )));
    }
    let current_seq = arr.shape()[1];
    if current_seq > tiling.h {
        Ok(arr.slice(s![.., ..tiling.h, ..]).to_owned().into_dyn())
    } else {
        Ok(arr)
    }
}

pub(crate) fn prepare_fixed_segments_from_cache(
    tiling: &TilingInfo,
    tensor_cache: &TensorStore,
) -> Result<Vec<Vec<ArrayD<f64>>>> {
    let segment_size = tiling.segment_size.ok_or_else(|| {
        DsperseError::Pipeline("fixed segment tiling missing segment_size".into())
    })?;
    if segment_size == 0 {
        return Err(DsperseError::Pipeline(
            "fixed segment tiling has segment_size=0".into(),
        ));
    }
    let total_elements = tiling.total_elements.ok_or_else(|| {
        DsperseError::Pipeline("fixed segment tiling missing total_elements".into())
    })?;
    let all_names = tiling.all_input_names();
    let num_segments = total_elements.div_ceil(segment_size);
    let mut all_segments: Vec<Vec<ArrayD<f64>>> = Vec::with_capacity(all_names.len());
    for name in &all_names {
        let input_arr = tensor_cache.get(name)?.clone();
        let flat: Vec<f64> = input_arr.iter().copied().collect();
        if flat.len() < total_elements {
            return Err(DsperseError::Pipeline(format!(
                "fixed segment: input '{}' has {} elements, expected at least {}",
                name,
                flat.len(),
                total_elements
            )));
        }
        let mut segments = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let start = i * segment_size;
            let end = (start + segment_size).min(total_elements);
            let mut seg_data = vec![0.0f64; segment_size];
            seg_data[..end - start].copy_from_slice(&flat[start..end]);
            let seg = ArrayD::from_shape_vec(IxDyn(&[segment_size]), seg_data)
                .map_err(|e| DsperseError::Pipeline(format!("fixed segment reshape: {e}")))?;
            segments.push(seg);
        }
        all_segments.push(segments);
    }
    Ok(all_segments)
}

pub(crate) fn reconstruct_from_fixed_segments(
    segment_outputs: &[ArrayD<f64>],
    tiling: &TilingInfo,
) -> Result<ArrayD<f64>> {
    let total_elements = tiling.total_elements.ok_or_else(|| {
        DsperseError::Pipeline("reconstruct fixed segments: missing total_elements".into())
    })?;
    if segment_outputs.is_empty() {
        return Err(DsperseError::Pipeline(
            "reconstruct fixed segments: no outputs".into(),
        ));
    }
    let mut flat = Vec::with_capacity(total_elements);
    for seg in segment_outputs {
        flat.extend(seg.iter().copied());
    }
    flat.truncate(total_elements);
    let shape: Vec<usize> = if tiling.original_shape.is_empty() {
        vec![total_elements]
    } else {
        tiling.original_shape.iter().map(|&d| d as usize).collect()
    };
    ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .map_err(|e| DsperseError::Pipeline(format!("reconstruct fixed segments reshape: {e}")))
}

pub(crate) fn reshape_to_4d(flat: &[f64], c: usize, h: usize, w: usize) -> Result<Array4<f64>> {
    let n = 1usize;
    let total = flat.len();
    if n * c * h * w != total {
        return Err(DsperseError::Pipeline(format!(
            "cannot reshape {total} elements to 4D (n={n}, c={c}, h={h}, w={w})"
        )));
    }
    Array4::from_shape_vec((n, c, h, w), flat.to_vec())
        .map_err(|e| DsperseError::Pipeline(format!("reshape: {e}")))
}

pub(crate) fn flatten_tile_inputs(all_tiles: &[Vec<ArrayD<f64>>], tile_idx: usize) -> Vec<f64> {
    let total: usize = all_tiles.iter().map(|tiles| tiles[tile_idx].len()).sum();
    let mut flat = Vec::with_capacity(total);
    for input_tiles in all_tiles {
        flat.extend(input_tiles[tile_idx].iter().copied());
    }
    flat
}
