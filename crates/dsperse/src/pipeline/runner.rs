use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array4, ArrayD, IxDyn, s};
use rayon::prelude::*;

use jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams;

use crate::backend::jstprove::JstproveBackend;
use crate::backend::onnx::NamedOutputs;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{
    ExecutionChain, ExecutionInfo, ExecutionMethod, ExecutionNode, ExecutionResultEntry,
    RunMetadata, TileResult,
};
use crate::schema::metadata::{ModelMetadata, RunSliceMetadata};
use crate::schema::tiling::{ChannelGroupInfo, ChannelSplitInfo, TilingInfo};
use crate::utils::io::{
    arrayd_to_json, extract_input_data, gather_inputs_from_cache, json_to_arrayd, read_input_json,
    write_input_json,
};
use crate::slicer::onnx_proto::TensorProto;
use crate::utils::paths::{find_metadata_path, resolve_relative_path, slice_dir_path};

pub struct RunConfig {
    pub parallel: usize,
    pub batch: bool,
    pub weights_onnx: Option<PathBuf>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            parallel: 1,
            batch: false,
            weights_onnx: None,
        }
    }
}

pub(crate) fn load_model_metadata(slices_dir: &Path) -> Result<ModelMetadata> {
    let meta_path = find_metadata_path(slices_dir)
        .ok_or_else(|| DsperseError::Metadata("no metadata.json in slices".into()))?;
    let mut model_meta = ModelMetadata::load(&meta_path)?;

    if model_meta.slices.is_empty() {
        let dslice_files = crate::archive::converter::find_dslice_files(slices_dir);
        if dslice_files.is_empty() {
            return Err(DsperseError::Metadata(format!(
                "metadata.json has no slices and no .dslice files found in {}",
                slices_dir.display()
            )));
        }
        let mut slices = Vec::with_capacity(dslice_files.len());
        for dslice_path in &dslice_files {
            slices.push(crate::archive::converter::read_dslice_slice_metadata(
                dslice_path,
            )?);
        }
        slices.sort_by_key(|s| s.index);
        model_meta.slices = slices;
    }

    model_meta.slices.sort_by_key(|s| s.index);

    Ok(model_meta)
}

fn validate_weights_onnx(
    donor_init_map: &HashMap<String, &TensorProto>,
    model_meta: &ModelMetadata,
    slices_dir: &Path,
) -> Result<()> {
    for slice in &model_meta.slices {
        let slice_dir = slice_dir_path(slices_dir, slice.index);
        let onnx_path = resolve_relative_path(&slice_dir, &slice.path);
        if !onnx_path.exists() {
            return Err(DsperseError::Pipeline(format!(
                "slice_{} ONNX not found at {}",
                slice.index,
                onnx_path.display()
            )));
        }
        let slice_model = crate::slicer::onnx_proto::load_model(&onnx_path)?;
        let slice_graph = slice_model.graph.as_ref().ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "slice_{} ONNX at {} has no graph",
                slice.index,
                onnx_path.display()
            ))
        })?;
        for init in &slice_graph.initializer {
            if let Some(donor_init) = donor_init_map.get(&init.name) {
                if init.data_type != donor_init.data_type {
                    return Err(DsperseError::Pipeline(format!(
                        "dtype mismatch for initializer '{}' in slice_{}: slice has dtype {}, consumer has dtype {}",
                        init.name, slice.index, init.data_type, donor_init.data_type
                    )));
                }
                if init.dims != donor_init.dims {
                    return Err(DsperseError::Pipeline(format!(
                        "shape mismatch for initializer '{}': slice expects {:?}, consumer provides {:?}",
                        init.name, init.dims, donor_init.dims
                    )));
                }
            } else {
                return Err(DsperseError::Pipeline(format!(
                    "consumer weights ONNX missing initializer '{}' required by slice_{}",
                    init.name, slice.index
                )));
            }
        }
    }
    Ok(())
}

pub fn run_inference(
    slices_dir: &Path,
    input_path: &Path,
    run_dir: &Path,
    backend: &JstproveBackend,
    config: &RunConfig,
) -> Result<RunMetadata> {
    let model_meta = load_model_metadata(slices_dir)?;

    let donor_model = if let Some(ref weights_path) = config.weights_onnx {
        if !weights_path.is_file() {
            return Err(DsperseError::Other(format!(
                "consumer weights ONNX not found: {}",
                weights_path.display()
            )));
        }
        Some(crate::slicer::onnx_proto::load_model(weights_path)?)
    } else {
        None
    };
    let donor_init_map = match donor_model.as_ref() {
        Some(model) => {
            let graph = model.graph.as_ref().ok_or_else(|| {
                DsperseError::Pipeline("consumer weights ONNX missing graph".into())
            })?;
            Some(crate::slicer::onnx_proto::build_initializer_map(graph))
        }
        None => None,
    };
    if let Some(ref map) = donor_init_map {
        validate_weights_onnx(map, &model_meta, slices_dir)?;
        tracing::info!(
            weights = %config.weights_onnx.as_ref().unwrap().display(),
            "validated consumer weights ONNX"
        );
    }

    std::fs::create_dir_all(run_dir).map_err(|e| DsperseError::io(e, run_dir))?;

    let input_data = read_input_json(input_path)?;

    let chain = build_execution_chain(&model_meta, slices_dir);
    let run_meta = build_run_metadata(&model_meta, slices_dir, &chain);

    let mut tensor_cache: HashMap<String, ArrayD<f64>> = HashMap::new();

    let input_val = extract_input_data(&input_data).ok_or_else(|| {
        DsperseError::Pipeline(
            "input JSON has no recognized input key (input_data, input, data, inputs)".into(),
        )
    })?;
    let first_slice = model_meta
        .slices
        .first()
        .ok_or_else(|| DsperseError::Pipeline("model has no slices".into()))?;
    let declared_inputs = &first_slice.dependencies.filtered_inputs;
    if declared_inputs.is_empty() {
        return Err(DsperseError::Pipeline(
            "first slice has no input dependency".into(),
        ));
    }
    if input_val.is_object() {
        for name in declared_inputs {
            let v = input_val.get(name).ok_or_else(|| {
                DsperseError::Pipeline(format!("input JSON object missing key {name:?}"))
            })?;
            tensor_cache.insert(name.clone(), json_to_arrayd(v)?);
        }
    } else if declared_inputs.len() == 1 {
        tensor_cache.insert(declared_inputs[0].clone(), json_to_arrayd(input_val)?);
    } else {
        return Err(DsperseError::Pipeline(format!(
            "model declares {} inputs but input JSON is not an object",
            declared_inputs.len()
        )));
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

        let slice_meta = run_meta.slices.get(&slice_id).ok_or_else(|| {
            DsperseError::Pipeline(format!("missing run slice metadata {slice_id}"))
        })?;

        let slice_run_dir = run_dir.join(&slice_id);
        std::fs::create_dir_all(&slice_run_dir).map_err(|e| DsperseError::io(e, &slice_run_dir))?;

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
            donor_init_map.as_ref(),
        );

        let exec_info = match exec_result {
            Ok(info) => info,
            Err(e) => {
                tracing::error!(slice = %slice_id, error = %e, "execution failed");
                let method = if slice_meta.channel_split.is_some() {
                    ExecutionMethod::ChannelSplit.to_string()
                } else if slice_meta.tiling.is_some() {
                    ExecutionMethod::Tiled.to_string()
                } else if node.use_circuit {
                    ExecutionMethod::JstproveGenWitness.to_string()
                } else {
                    ExecutionMethod::OnnxOnly.to_string()
                };
                results.push(ExecutionResultEntry {
                    slice_id: slice_id.clone(),
                    witness_execution: Some(ExecutionInfo {
                        method,
                        success: false,
                        error: Some(e.to_string()),
                        witness_file: None,
                        tile_exec_infos: Vec::new(),
                    }),
                    proof_execution: None,
                    verification_execution: None,
                });
                break;
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

    let mut final_meta = run_meta;
    final_meta.execution_chain.execution_results = results;
    final_meta.run_directory = Some(run_dir.to_string_lossy().into_owned());

    let meta_out = run_dir.join("metadata.json");
    let meta_json = serde_json::to_string_pretty(&final_meta)?;
    std::fs::write(&meta_out, meta_json).map_err(|e| DsperseError::io(e, &meta_out))?;

    let last_slice = model_meta
        .slices
        .last()
        .ok_or_else(|| DsperseError::Pipeline("model has no slices".into()))?;
    let last_slice_id = format!("slice_{}", last_slice.index);
    let output_arr = if let Some(meta) = final_meta.slices.get(&last_slice_id) {
        if let Some(ref cs) = meta.channel_split {
            tensor_cache.get(&cs.output_name)
        } else if let Some(ref tiling) = meta.tiling {
            tensor_cache.get(&tiling.output_name)
        } else {
            last_slice
                .dependencies
                .output
                .first()
                .and_then(|n| tensor_cache.get(n))
        }
    } else {
        last_slice
            .dependencies
            .output
            .first()
            .and_then(|n| tensor_cache.get(n))
    };
    let output_arr = output_arr.ok_or_else(|| {
        let first_error = final_meta
            .execution_chain
            .execution_results
            .iter()
            .filter_map(|r| {
                r.witness_execution
                    .as_ref()
                    .and_then(|w| w.error.as_deref())
                    .map(|err| format!("{}: {err}", r.slice_id))
            })
            .next();
        match first_error {
            Some(err) => DsperseError::Pipeline(format!("pipeline failed at {err}")),
            None => DsperseError::Pipeline(format!(
                "no output tensor found for last slice {last_slice_id}"
            )),
        }
    })?;
    let output_path = run_dir.join("output.json");
    let output_json = arrayd_to_json(output_arr);
    write_input_json(
        &output_path,
        &serde_json::json!({ "output_data": output_json }),
    )?;

    Ok(final_meta)
}

#[allow(clippy::too_many_arguments)]
fn execute_slice(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    node: &ExecutionNode,
    meta: &RunSliceMetadata,
    tensor_cache: &mut HashMap<String, ArrayD<f64>>,
    backend: &JstproveBackend,
    config: &RunConfig,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    if let Some(ref cs) = meta.channel_split {
        return execute_channel_split(
            slices_dir,
            slice_run_dir,
            slice_id,
            cs,
            tensor_cache,
            backend,
            donor_init_map,
        );
    }

    if let Some(ref tiling) = meta.tiling {
        let slice_circuit = meta
            .jstprove_circuit_path
            .as_deref()
            .map(std::path::PathBuf::from);
        return execute_tiled(
            slices_dir,
            slice_run_dir,
            slice_id,
            tiling,
            slice_circuit.as_deref(),
            tensor_cache,
            backend,
            config,
            donor_init_map,
        );
    }

    execute_single(
        slices_dir,
        slice_run_dir,
        slice_id,
        node,
        meta,
        tensor_cache,
        backend,
        donor_init_map,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_single(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    node: &ExecutionNode,
    meta: &RunSliceMetadata,
    tensor_cache: &mut HashMap<String, ArrayD<f64>>,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let slice_idx = parse_slice_idx(slice_id)?;
    let slice_dir = slice_dir_path(slices_dir, slice_idx);

    let inputs: Vec<String> = meta
        .dependencies
        .filtered_inputs
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect();
    let multi_input = inputs.len() > 1;

    let onnx_path = resolve_relative_path(&slice_dir, &meta.path);

    let patched_onnx = if let Some(map) = donor_init_map {
        Some(crate::slicer::onnx_proto::build_patched_onnx(&onnx_path, map)?)
    } else {
        None
    };
    let effective_onnx: &Path = patched_onnx.as_ref().map_or(onnx_path.as_path(), |t| t.path());

    if node.use_circuit {
        let circuit_path = meta
            .jstprove_circuit_path
            .as_deref()
            .map(std::path::PathBuf::from)
            .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {slice_id}")))?;

        let params = backend.load_params(&circuit_path)?;
        let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

        if donor_init_map.is_some() && !is_wai {
            return Err(DsperseError::Pipeline(format!(
                "{slice_id}: consumer weights require circuits compiled with --weights-as-inputs"
            )));
        }

        if multi_input {
            return Err(DsperseError::Pipeline(format!(
                "{slice_id}: circuit path does not support multiple activation inputs"
            )));
        }

        let input_tensor = gather_inputs_from_cache(tensor_cache, &inputs[..1])?;
        let named = run_onnx_inference_named(effective_onnx, &input_tensor)?;

        let witness_bytes = if is_wai {
            generate_wai_witness(
                backend,
                &circuit_path,
                &onnx_path,
                donor_init_map,
                params.as_ref().unwrap(),
                &input_tensor,
            )?
        } else {
            let output_name = meta.dependencies.output.first().ok_or_else(|| {
                DsperseError::Pipeline(format!("no output name for {slice_id}"))
            })?;
            let (data, shape) = named.get(output_name).ok_or_else(|| {
                DsperseError::Pipeline(format!(
                    "{slice_id}: named output '{output_name}' missing from inference results"
                ))
            })?;
            let output_tensor = ArrayD::from_shape_vec(IxDyn(shape), data.clone())
                .map_err(|e| DsperseError::Pipeline(format!("output reshape: {e}")))?;
            let input_json_bytes = serde_json::to_vec(
                &serde_json::json!({ "input_data": arrayd_to_json(&input_tensor) }),
            )?;
            let output_json_bytes = serde_json::to_vec(
                &serde_json::json!({ "output_data": arrayd_to_json(&output_tensor) }),
            )?;
            backend.witness(&circuit_path, &input_json_bytes, &output_json_bytes)?
        };

        let witness_path = slice_run_dir.join("witness.bin");
        std::fs::write(&witness_path, &witness_bytes)
            .map_err(|e| DsperseError::io(e, &witness_path))?;

        store_named_outputs(tensor_cache, &meta.dependencies.output, named)?;

        Ok(ExecutionInfo {
            method: ExecutionMethod::JstproveGenWitness.to_string(),
            success: true,
            error: None,
            witness_file: Some(witness_path.to_string_lossy().into_owned()),
            tile_exec_infos: Vec::new(),
        })
    } else {
        let named = if multi_input {
            run_onnx_inference_multi_named(effective_onnx, tensor_cache, &inputs)?
        } else {
            let input_tensor = gather_inputs_from_cache(tensor_cache, &inputs)?;
            run_onnx_inference_named(effective_onnx, &input_tensor)?
        };
        store_named_outputs(tensor_cache, &meta.dependencies.output, named)?;

        Ok(ExecutionInfo {
            method: ExecutionMethod::OnnxOnly.to_string(),
            success: true,
            error: None,
            witness_file: None,
            tile_exec_infos: Vec::new(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_tiled(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    tiling: &TilingInfo,
    slice_circuit_path: Option<&Path>,
    tensor_cache: &mut HashMap<String, ArrayD<f64>>,
    backend: &JstproveBackend,
    config: &RunConfig,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let input_arr = tensor_cache
        .get(&tiling.input_name)
        .ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "tiling input '{}' not in cache for {slice_id}",
                tiling.input_name
            ))
        })?
        .clone();

    let input_4d = if input_arr.ndim() == 4 {
        let s = input_arr.shape();
        Array4::from_shape_vec(
            (s[0], s[1], s[2], s[3]),
            input_arr.iter().copied().collect(),
        )
        .map_err(|e| DsperseError::Pipeline(format!("tiling input reshape: {e}")))?
    } else {
        let input_flat: Vec<f64> = input_arr.iter().copied().collect();
        let h = tiling.tiles_y * tiling.tile_size;
        let w = tiling.tiles_x * tiling.tile_size;
        reshape_to_4d(&input_flat, tiling.c_in, h, w)?
    };

    let tiles = split_into_tiles(&input_4d, tiling)?;

    tracing::info!(
        slice = %slice_id,
        num_tiles = tiles.len(),
        tile_size = tiling.tile_size,
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
    let tile_onnx = first_tile_info.map(|ti| resolve_relative_path(slices_dir, &ti.path));

    let patched_tile_onnx = match (&tile_onnx, donor_init_map) {
        (Some(onnx_path), Some(map)) => {
            Some(crate::slicer::onnx_proto::build_patched_onnx(onnx_path, map)?)
        }
        _ => None,
    };
    let effective_tile_onnx = patched_tile_onnx.as_ref().map(|t| t.path().to_path_buf());
    let effective_tile_onnx_ref = effective_tile_onnx.as_deref().or(tile_onnx.as_deref());

    let warm_model = match (effective_tile_onnx_ref, tiles.first()) {
        (Some(onnx_path), Some(sample)) => {
            let shape = sample.clone().into_dyn().shape().to_vec();
            let model = crate::backend::onnx::WarmModel::load(onnx_path, &shape)?;
            tracing::info!(slice = %slice_id, "loaded ONNX model");
            Some(model)
        }
        _ => None,
    };

    let circuit_path = first_tile_info
        .and_then(|ti| {
            ti.jstprove_circuit_path
                .as_deref()
                .map(|p| resolve_relative_path(slices_dir, p))
        })
        .or_else(|| slice_circuit_path.map(|p| p.to_path_buf()));

    let warm_circuit = match (&circuit_path, &tile_onnx) {
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
            let wc = crate::backend::jstprove::WarmCircuit::load(cp, initializers, backend.compress())?;
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

    let collected: Vec<(TileResult, Option<ArrayD<f64>>)> = pool.install(|| {
        tiles
            .par_iter()
            .enumerate()
            .map(|(tile_idx, tile_data)| {
                let start = std::time::Instant::now();
                let tile_dir = slice_run_dir.join(format!("tile_{tile_idx}"));
                if let Err(e) = std::fs::create_dir_all(&tile_dir) {
                    return (
                        TileResult {
                            tile_idx,
                            success: false,
                            error: Some(format!("mkdir: {e}")),
                            method: None,
                            time_sec: 0.0,
                            proof_path: None,
                        },
                        None,
                    );
                }

                let tile_info = tile_infos.get(tile_idx).or(single_tile);
                let tile_dyn = tile_data.clone().into_dyn();

                if tile_info.is_none() {
                    return (
                        TileResult {
                            tile_idx,
                            success: false,
                            error: Some("no tile circuit info".into()),
                            method: None,
                            time_sec: 0.0,
                            proof_path: None,
                        },
                        None,
                    );
                }

                let tile_output = if let Some(ref wm) = warm_model {
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
                            TileResult {
                                tile_idx,
                                success: false,
                                error: Some(format!("onnx inference: {e}")),
                                method: Some("onnx".into()),
                                time_sec: start.elapsed().as_secs_f64(),
                                proof_path: None,
                            },
                            None,
                        );
                    }
                };

                if circuit_path.is_none() {
                    return (
                        TileResult {
                            tile_idx,
                            success: true,
                            error: None,
                            method: Some("onnx".into()),
                            time_sec: start.elapsed().as_secs_f64(),
                            proof_path: None,
                        },
                        Some(output_tensor),
                    );
                }

                let witness_result = if let Some(ref wc) = warm_circuit {
                    if wc.params.weights_as_inputs {
                        let flat: Vec<f64> = tile_dyn.iter().copied().collect();
                        wc.witness_f64(&flat)
                    } else {
                        let Ok(input_json_bytes) = serde_json::to_vec(
                            &serde_json::json!({ "input_data": arrayd_to_json(&tile_dyn) }),
                        ) else {
                            return (
                                TileResult {
                                    tile_idx,
                                    success: false,
                                    error: Some("json serialize input".into()),
                                    method: Some("jstprove".into()),
                                    time_sec: start.elapsed().as_secs_f64(),
                                    proof_path: None,
                                },
                                None,
                            );
                        };
                        let Ok(output_json_bytes) = serde_json::to_vec(
                            &serde_json::json!({ "output_data": arrayd_to_json(&output_tensor) }),
                        ) else {
                            return (
                                TileResult {
                                    tile_idx,
                                    success: false,
                                    error: Some("json serialize output".into()),
                                    method: Some("jstprove".into()),
                                    time_sec: start.elapsed().as_secs_f64(),
                                    proof_path: None,
                                },
                                None,
                            );
                        };
                        backend.witness(circuit_path.as_ref().unwrap(), &input_json_bytes, &output_json_bytes)
                    }
                } else {
                    let Ok(input_json_bytes) = serde_json::to_vec(
                        &serde_json::json!({ "input_data": arrayd_to_json(&tile_dyn) }),
                    ) else {
                        return (
                            TileResult {
                                tile_idx,
                                success: false,
                                error: Some("json serialize input".into()),
                                method: Some("jstprove".into()),
                                time_sec: start.elapsed().as_secs_f64(),
                                proof_path: None,
                            },
                            None,
                        );
                    };
                    let Ok(output_json_bytes) = serde_json::to_vec(
                        &serde_json::json!({ "output_data": arrayd_to_json(&output_tensor) }),
                    ) else {
                        return (
                            TileResult {
                                tile_idx,
                                success: false,
                                error: Some("json serialize output".into()),
                                method: Some("jstprove".into()),
                                time_sec: start.elapsed().as_secs_f64(),
                                proof_path: None,
                            },
                            None,
                        );
                    };
                    backend.witness(circuit_path.as_ref().unwrap(), &input_json_bytes, &output_json_bytes)
                };

                match witness_result {
                    Ok(witness_bytes) => {
                        let witness_path = tile_dir.join("witness.bin");
                        if let Err(e) = std::fs::write(&witness_path, &witness_bytes) {
                            return (
                                TileResult {
                                    tile_idx,
                                    success: false,
                                    error: Some(format!("write witness: {e}")),
                                    method: Some("jstprove".into()),
                                    time_sec: start.elapsed().as_secs_f64(),
                                    proof_path: None,
                                },
                                None,
                            );
                        }
                        (
                            TileResult {
                                tile_idx,
                                success: true,
                                error: None,
                                method: Some("jstprove".into()),
                                time_sec: start.elapsed().as_secs_f64(),
                                proof_path: None,
                            },
                            Some(output_tensor),
                        )
                    }
                    Err(e) => (
                        TileResult {
                            tile_idx,
                            success: false,
                            error: Some(e.to_string()),
                            method: Some("jstprove".into()),
                            time_sec: start.elapsed().as_secs_f64(),
                            proof_path: None,
                        },
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
    let reconstructed = reconstruct_from_tiles(&tile_outputs, tiling)?;
    tensor_cache.insert(tiling.output_name.clone(), reconstructed);

    Ok(ExecutionInfo {
        method: ExecutionMethod::Tiled.to_string(),
        success: true,
        error: None,
        witness_file: None,
        tile_exec_infos: tile_results,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_channel_split(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    cs: &ChannelSplitInfo,
    tensor_cache: &mut HashMap<String, ArrayD<f64>>,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let slice_idx = parse_slice_idx(slice_id)?;
    let slice_dir = slice_dir_path(slices_dir, slice_idx);

    let input_arr = tensor_cache
        .get(&cs.input_name)
        .ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "channel split input '{}' not in cache for {slice_id}",
                cs.input_name
            ))
        })?
        .clone();

    let (input_4d, n, h) = if input_arr.ndim() == 4 {
        let s = input_arr.shape();
        let n = s[0];
        if n != 1 {
            return Err(DsperseError::Pipeline(format!(
                "channel split: batch size {n} not supported, expected 1"
            )));
        }
        let h = s[2];
        let arr =
            Array4::from_shape_vec((n, s[1], s[2], s[3]), input_arr.iter().copied().collect())
                .map_err(|e| DsperseError::Pipeline(format!("channel split reshape: {e}")))?;
        (arr, n, h)
    } else {
        let n = 1usize;
        let input_flat: Vec<f64> = input_arr.iter().copied().collect();
        let total_elements = input_flat.len();
        let nc = n * cs.c_in;
        if nc > 0 && total_elements % nc != 0 {
            return Err(DsperseError::Pipeline(format!(
                "channel split reshape: total_elements {total_elements} not divisible by n*c_in ({nc})"
            )));
        }
        let spatial = if cs.c_in > 0 && total_elements > 0 {
            total_elements / nc
        } else {
            cs.h * cs.w
        };
        let h = cs.h.max(1);
        if spatial > 0 && h > 0 && spatial % h != 0 {
            return Err(DsperseError::Pipeline(format!(
                "channel split reshape: spatial {spatial} not divisible by h={h}"
            )));
        }
        let w = if spatial > 0 && h > 0 {
            spatial / h
        } else {
            cs.w.max(1)
        };
        let arr = Array4::from_shape_vec((n, cs.c_in, h, w), input_flat)
            .map_err(|e| DsperseError::Pipeline(format!("channel split reshape: {e}")))?;
        (arr, n, h)
    };

    let mut accumulated: Option<Array4<f64>> = None;

    tracing::info!(
        slice = %slice_id,
        num_groups = cs.groups.len(),
        "channel split execution"
    );

    let n_channels = input_4d.shape()[1];
    for group in &cs.groups {
        if group.c_end > n_channels || group.c_start > group.c_end {
            return Err(DsperseError::Pipeline(format!(
                "channel group {} bounds [{}, {}) exceed channel dimension {}",
                group.group_idx, group.c_start, group.c_end, n_channels
            )));
        }
        let group_input = input_4d
            .slice(s![.., group.c_start..group.c_end, .., ..])
            .to_owned();
        let group_input_dyn = group_input.into_dyn();

        let group_dir = slice_run_dir.join(format!("group_{}", group.group_idx));
        std::fs::create_dir_all(&group_dir).map_err(|e| DsperseError::io(e, &group_dir))?;

        let group_output =
            execute_channel_group(&slice_dir, &group_dir, group, &group_input_dyn, backend, donor_init_map)?;

        let group_4d = if group_output.ndim() == 4 {
            let s = group_output.shape();
            Array4::from_shape_vec(
                (s[0], s[1], s[2], s[3]),
                group_output.iter().copied().collect(),
            )
            .map_err(|e| DsperseError::Pipeline(format!("group output reshape: {e}")))?
        } else {
            let group_flat: Vec<f64> = group_output.iter().copied().collect();
            let (out_h, out_w) = if cs.out_h > 0 && cs.out_w > 0 {
                (cs.out_h, cs.out_w)
            } else if cs.c_out > 0 {
                let out_spatial = group_flat.len() / (n * cs.c_out);
                if h > 0 && out_spatial > 0 && out_spatial % h == 0 {
                    (h, out_spatial / h)
                } else {
                    return Err(DsperseError::Pipeline(format!(
                        "cannot determine spatial layout for channel_split output: {} elements, c_out={}, set out_h/out_w in metadata",
                        group_flat.len(),
                        cs.c_out
                    )));
                }
            } else {
                return Err(DsperseError::Pipeline("channel split c_out is 0".into()));
            };
            if n * cs.c_out * out_h * out_w != group_flat.len() {
                return Err(DsperseError::Pipeline(format!(
                    "group output reshape mismatch: expected {} elements (n={}, c_out={}, h={}, w={}), got {}",
                    n * cs.c_out * out_h * out_w,
                    n,
                    cs.c_out,
                    out_h,
                    out_w,
                    group_flat.len()
                )));
            }
            Array4::from_shape_vec((n, cs.c_out, out_h, out_w), group_flat)
                .map_err(|e| DsperseError::Pipeline(format!("group output reshape: {e}")))?
        };

        accumulated = Some(match accumulated {
            Some(acc) => {
                if acc.shape() != group_4d.shape() {
                    return Err(DsperseError::Pipeline(format!(
                        "channel group {} shape {:?} does not match accumulator shape {:?}",
                        group.group_idx,
                        group_4d.shape(),
                        acc.shape()
                    )));
                }
                acc + &group_4d
            }
            None => group_4d,
        });
    }

    if let Some(ref bias_path_str) = cs.bias_path {
        let bias_file = resolve_relative_path(&slice_dir, bias_path_str);
        if !bias_file.exists() {
            return Err(DsperseError::Pipeline(format!(
                "configured bias file not found: {} (bias_path={bias_path_str})",
                bias_file.display()
            )));
        }
        let bias_data = read_input_json(&bias_file)?;
        let bias_flat = crate::utils::io::flatten_nested_list(&bias_data);
        if bias_flat.len() != cs.c_out {
            return Err(DsperseError::Pipeline(format!(
                "bias length {} does not match c_out {}",
                bias_flat.len(),
                cs.c_out
            )));
        }
        if let Some(ref mut acc) = accumulated {
            for ((_, c, _, _), val) in acc.indexed_iter_mut() {
                *val += bias_flat[c];
            }
        }
    }

    match accumulated {
        Some(acc) => {
            tensor_cache.insert(cs.output_name.clone(), acc.into_dyn());
        }
        None => {
            return Err(DsperseError::Pipeline(format!(
                "channel_split produced no output for '{}'",
                cs.output_name
            )));
        }
    }

    Ok(ExecutionInfo {
        method: ExecutionMethod::ChannelSplit.to_string(),
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
    group_input: &ArrayD<f64>,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ArrayD<f64>> {
    let onnx_path = resolve_relative_path(slice_dir, &group.path);

    let patched_onnx = if let Some(map) = donor_init_map {
        Some(crate::slicer::onnx_proto::build_patched_onnx(&onnx_path, map)?)
    } else {
        None
    };
    let effective_onnx = patched_onnx.as_ref().map_or(onnx_path.as_path(), |t| t.path());

    if let Some(ref circuit_path_str) = group.jstprove_circuit_path {
        let circuit_path = resolve_relative_path(slice_dir, circuit_path_str);

        let params = backend.load_params(&circuit_path)?;
        let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

        if donor_init_map.is_some() && !is_wai {
            return Err(DsperseError::Pipeline(format!(
                "group_{}: consumer weights require circuits compiled with --weights-as-inputs",
                group.group_idx
            )));
        }

        let output_tensor = run_onnx_inference(effective_onnx, group_input)?;

        let witness_bytes = if is_wai {
            generate_wai_witness(
                backend,
                &circuit_path,
                &onnx_path,
                donor_init_map,
                params.as_ref().unwrap(),
                group_input,
            )?
        } else {
            let input_json_bytes = serde_json::to_vec(
                &serde_json::json!({ "input_data": arrayd_to_json(group_input) }),
            )?;
            let output_json_bytes = serde_json::to_vec(
                &serde_json::json!({ "output_data": arrayd_to_json(&output_tensor) }),
            )?;
            backend.witness(&circuit_path, &input_json_bytes, &output_json_bytes)?
        };

        let witness_path = group_dir.join("witness.bin");
        std::fs::write(&witness_path, &witness_bytes)
            .map_err(|e| DsperseError::io(e, &witness_path))?;

        Ok(output_tensor)
    } else {
        run_onnx_inference(effective_onnx, group_input)
    }
}

fn split_into_tiles(input: &Array4<f64>, tiling: &TilingInfo) -> Result<Vec<Array4<f64>>> {
    if tiling.halo[0] < 0 || tiling.halo[1] < 0 {
        return Err(DsperseError::Pipeline(format!(
            "negative halo values not supported: halo=[{}, {}]",
            tiling.halo[0], tiling.halo[1]
        )));
    }
    let (n, c, h, w) = input.dim();
    if n != 1 {
        return Err(DsperseError::Pipeline(format!(
            "split_into_tiles: batch size {n} not supported, expected 1"
        )));
    }
    let halo_h = tiling.halo[0] as usize;
    let halo_w = tiling.halo[1] as usize;
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
            let y_start = ty * tiling.tile_size;
            let x_start = tx * tiling.tile_size;
            let y_end = (y_start + tile_h).min(padded_h);
            let x_end = (x_start + tile_w).min(padded_w);

            let tile = padded
                .slice(s![.., .., y_start..y_end, x_start..x_end])
                .to_owned();
            tiles.push(tile);
        }
    }

    Ok(tiles)
}

fn reconstruct_from_tiles(
    tile_outputs: &[ArrayD<f64>],
    tiling: &TilingInfo,
) -> Result<ArrayD<f64>> {
    if tile_outputs.is_empty() {
        return Err(DsperseError::Pipeline(
            "no tile outputs to reconstruct".into(),
        ));
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

fn reshape_to_4d(flat: &[f64], c: usize, h: usize, w: usize) -> Result<Array4<f64>> {
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

fn parse_slice_idx(slice_id: &str) -> Result<usize> {
    slice_id
        .strip_prefix("slice_")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| DsperseError::Pipeline(format!("invalid slice_id format: {slice_id:?}")))
}

fn store_named_outputs(
    tensor_cache: &mut HashMap<String, ArrayD<f64>>,
    output_names: &[String],
    named_outputs: HashMap<String, (Vec<f64>, Vec<usize>)>,
) -> Result<()> {
    for name in output_names {
        if let Some((data, shape)) = named_outputs.get(name) {
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data.clone())
                .map_err(|e| DsperseError::Pipeline(format!("output reshape '{name}': {e}")))?;
            tensor_cache.insert(name.clone(), arr);
        }
    }
    Ok(())
}

fn run_onnx_inference(onnx_path: &Path, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
    let input_flat: Vec<f64> = input.iter().copied().collect();
    let input_shape = input.shape();
    let (output_data, output_shape) =
        crate::backend::onnx::run_inference(onnx_path, &input_flat, input_shape)?;

    ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)
        .map_err(|e| DsperseError::Pipeline(format!("output reshape: {e}")))
}

fn run_onnx_inference_named(
    onnx_path: &Path,
    input: &ArrayD<f64>,
) -> Result<NamedOutputs> {
    let input_flat: Vec<f64> = input.iter().copied().collect();
    let input_shape = input.shape();
    crate::backend::onnx::run_inference_named(onnx_path, &input_flat, input_shape)
}

fn run_onnx_inference_multi_named(
    onnx_path: &Path,
    tensor_cache: &HashMap<String, ArrayD<f64>>,
    input_names: &[String],
) -> Result<NamedOutputs> {
    let inputs: Vec<(&str, Vec<f64>, Vec<usize>)> = input_names
        .iter()
        .map(|name| {
            let arr = tensor_cache.get(name).ok_or_else(|| {
                DsperseError::Pipeline(format!("missing tensor '{name}' in cache"))
            })?;
            Ok((
                name.as_str(),
                arr.iter().copied().collect(),
                arr.shape().to_vec(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    crate::backend::onnx::run_inference_multi_named(onnx_path, &inputs)
}

pub(crate) fn build_execution_chain(
    model_meta: &ModelMetadata,
    slices_dir: &Path,
) -> ExecutionChain {
    let mut nodes = HashMap::new();
    let mut head = None;

    for (i, slice) in model_meta.slices.iter().enumerate() {
        let slice_id = format!("slice_{}", slice.index);
        let slice_dir = slice_dir_path(slices_dir, slice.index);

        if i == 0 {
            head = Some(slice_id.clone());
        }

        let (has_circuit, circuit_path) = if slice.compilation.jstprove.compiled {
            let path = slice.compilation.jstprove.files.compiled.as_ref().map(|p| {
                slices_dir.join(p).to_string_lossy().into_owned()
            });
            (true, path)
        } else {
            let msgpack = slice_dir.join("jstprove/circuit.msgpack");
            if msgpack.exists() {
                tracing::info!(slice = %slice_id, "detected circuit on filesystem (metadata.compiled=false)");
                (true, Some(msgpack.to_string_lossy().into_owned()))
            } else {
                (false, None)
            }
        };
        let next = model_meta
            .slices
            .get(i + 1)
            .map(|s| format!("slice_{}", s.index));

        let onnx_path = Some(slice_dir.join(&slice.path).to_string_lossy().into_owned());

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

pub(crate) fn build_run_metadata(
    model_meta: &ModelMetadata,
    slices_dir: &Path,
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
            path: slice_dir.join(&slice.path).to_string_lossy().into_owned(),
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

fn extract_initializers_from_map(
    init_map: &HashMap<String, &TensorProto>,
    params: &CircuitParams,
) -> Result<Vec<(Vec<f64>, Vec<usize>)>> {
    let mut initializers = Vec::new();
    for io in &params.inputs {
        if let Some(tensor) = init_map.get(&io.name) {
            let f32_vals = crate::slicer::onnx_proto::tensor_to_f32(tensor);
            let f64_vals: Vec<f64> = f32_vals.iter().map(|&v| f64::from(v)).collect();
            let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
            initializers.push((f64_vals, shape));
        }
    }
    Ok(initializers)
}

fn extract_onnx_initializers(
    onnx_path: &Path,
    params: &CircuitParams,
) -> Result<Vec<(Vec<f64>, Vec<usize>)>> {
    let model = crate::slicer::onnx_proto::load_model(onnx_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Pipeline("ONNX model missing graph".into()))?;
    let init_map = crate::slicer::onnx_proto::build_initializer_map(graph);
    extract_initializers_from_map(&init_map, params)
}

fn generate_wai_witness(
    backend: &JstproveBackend,
    circuit_path: &Path,
    slice_onnx_path: &Path,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
    params: &CircuitParams,
    activations: &ArrayD<f64>,
) -> Result<Vec<u8>> {
    let initializers = if let Some(map) = donor_init_map {
        extract_initializers_from_map(map, params)?
    } else {
        extract_onnx_initializers(slice_onnx_path, params)?
    };
    let flat_activations: Vec<f64> = activations.iter().copied().collect();
    backend.witness_f64(circuit_path, &flat_activations, &initializers)
}
