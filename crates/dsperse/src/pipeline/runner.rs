use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array4, ArrayD, IxDyn, s};
use rayon::prelude::*;

use jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams;

use super::strategy::ExecutionStrategy;
use super::tensor_store::TensorStore;
use crate::backend::jstprove::JstproveBackend;
use crate::backend::onnx::NamedOutputs;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{
    ExecutionChain, ExecutionInfo, ExecutionMethod, ExecutionNode, ExecutionResultEntry,
    RunMetadata, TileResult,
};
use crate::schema::metadata::{BackendKind, ModelMetadata, RunSliceMetadata};
use crate::schema::tiling::{ChannelGroupInfo, ChannelSplitInfo, TilingInfo};
use crate::slicer::onnx_proto::TensorProto;
use crate::utils::io::{
    arrayd_to_value, build_msgpack_map, extract_input_data, map_get_ref, read_msgpack,
    value_to_arrayd, write_msgpack,
};
use crate::utils::paths::{find_metadata_path, resolve_relative_path, slice_dir_path};
use rmpv::Value;

pub struct RunConfig {
    pub parallel: usize,
    pub batch: bool,
    pub weights_onnx: Option<PathBuf>,
    pub combined: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            parallel: 1,
            batch: false,
            weights_onnx: None,
            combined: true,
        }
    }
}

pub fn load_model_metadata(slices_dir: &Path) -> Result<ModelMetadata> {
    let meta_path = find_metadata_path(slices_dir).ok_or_else(|| {
        DsperseError::Metadata(format!(
            "no {} in slices",
            crate::utils::paths::METADATA_FILE
        ))
    })?;
    let mut model_meta = ModelMetadata::load(&meta_path)?;

    if model_meta.slices.is_empty() {
        return Err(DsperseError::Metadata(format!(
            "{} has no slices in {}",
            crate::utils::paths::METADATA_FILE,
            slices_dir.display()
        )));
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
        let onnx_path = slice.resolve_onnx(slices_dir)?;
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
        let context = format!("slice_{}", slice.index);
        crate::slicer::onnx_proto::validate_initializer_compatibility(
            &slice_graph.initializer,
            donor_init_map,
            &context,
        )?;
    }
    Ok(())
}

fn load_donor_model(
    weights_onnx: Option<&PathBuf>,
) -> Result<Option<crate::slicer::onnx_proto::ModelProto>> {
    let weights_path = match weights_onnx {
        Some(p) => p,
        None => return Ok(None),
    };
    if !weights_path.is_file() {
        return Err(DsperseError::Other(format!(
            "consumer weights ONNX not found: {}",
            weights_path.display()
        )));
    }
    Ok(Some(crate::slicer::onnx_proto::load_model(weights_path)?))
}

fn donor_init_map(
    model: Option<&crate::slicer::onnx_proto::ModelProto>,
) -> Result<Option<HashMap<String, &TensorProto>>> {
    match model {
        Some(m) => {
            let graph = m.graph.as_ref().ok_or_else(|| {
                DsperseError::Pipeline("consumer weights ONNX missing graph".into())
            })?;
            Ok(Some(crate::slicer::onnx_proto::build_initializer_map(
                graph,
            )))
        }
        None => Ok(None),
    }
}

pub fn run_inference(
    slices_dir: &Path,
    input_path: &Path,
    run_dir: &Path,
    backend: &JstproveBackend,
    config: &RunConfig,
) -> Result<RunMetadata> {
    let model_meta = load_model_metadata(slices_dir)?;

    if config.combined
        && model_meta.original_model_path.is_some()
        && model_meta.traced_shapes.is_some()
    {
        return run_combined_inference(
            slices_dir,
            input_path,
            run_dir,
            backend,
            config,
            &model_meta,
        );
    } else if config.combined {
        tracing::warn!(
            "combined mode requested but metadata missing original_model_path or traced_shapes, using per-slice execution"
        );
    }

    if model_meta.original_model_path.is_some() {
        crate::slicer::materializer::ensure_all_slices_materialized(slices_dir, &model_meta)?;
    }

    let donor_model = load_donor_model(config.weights_onnx.as_ref())?;
    let donor_map = donor_init_map(donor_model.as_ref())?;
    if let Some(ref map) = donor_map {
        validate_weights_onnx(map, &model_meta, slices_dir)?;
        tracing::info!(
            weights = %config.weights_onnx.as_ref().unwrap().display(),
            "validated consumer weights ONNX"
        );
    }

    std::fs::create_dir_all(run_dir).map_err(|e| DsperseError::io(e, run_dir))?;

    let input_data = read_msgpack(input_path)?;

    let chain = build_execution_chain(&model_meta, slices_dir)?;
    let run_meta = build_run_metadata(&model_meta, slices_dir, &chain)?;

    let mut tensor_cache = TensorStore::new();

    let input_val = extract_input_data(&input_data).ok_or_else(|| {
        DsperseError::Pipeline(
            "input has no recognized input key (input_data, input, data, inputs)".into(),
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
    if input_val.is_map() {
        for name in declared_inputs {
            let v = map_get_ref(input_val, name)
                .ok_or_else(|| DsperseError::Pipeline(format!("input map missing key {name:?}")))?;
            tensor_cache.put(name.clone(), value_to_arrayd(v)?);
        }
    } else if declared_inputs.len() == 1 {
        tensor_cache.put(declared_inputs[0].clone(), value_to_arrayd(input_val)?);
    } else {
        return Err(DsperseError::Pipeline(format!(
            "model declares {} inputs but input is not a map",
            declared_inputs.len()
        )));
    }

    let input_copy = run_dir.join(crate::utils::paths::INPUT_FILE);
    write_msgpack(&input_copy, &input_data)?;

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
            donor_map.as_ref(),
        );

        let exec_info = match exec_result {
            Ok(info) => info,
            Err(e) => {
                tracing::error!(slice = %slice_id, error = %e, "execution failed");
                let method = ExecutionStrategy::from_metadata(slice_meta, node.use_circuit)
                    .map(|s| s.execution_method())
                    .unwrap_or(ExecutionMethod::OnnxOnly);
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

    let meta_out = run_dir.join(crate::utils::paths::METADATA_FILE);
    crate::utils::metadata::save_run_metadata(&meta_out, &final_meta)?;

    let last_slice = model_meta
        .slices
        .last()
        .ok_or_else(|| DsperseError::Pipeline("model has no slices".into()))?;
    let last_slice_id = format!("slice_{}", last_slice.index);
    if let Some(failed) = final_meta
        .execution_chain
        .execution_results
        .iter()
        .find(|r| r.witness_execution.as_ref().is_some_and(|w| !w.success))
    {
        let err_msg = failed
            .witness_execution
            .as_ref()
            .and_then(|w| w.error.as_deref())
            .unwrap_or("unknown");
        return Err(DsperseError::Pipeline(format!(
            "pipeline failed at {}: {err_msg}",
            failed.slice_id
        )));
    }

    let slice_run_meta = final_meta.slices.get(&last_slice_id);
    let last_strategy = match slice_run_meta {
        Some(m) => {
            let use_circuit = final_meta
                .execution_chain
                .nodes
                .get(&last_slice_id)
                .is_some_and(|n| n.use_circuit);
            Some(ExecutionStrategy::from_metadata(m, use_circuit)?)
        }
        None => None,
    };
    let output_arrs: Vec<&ArrayD<f64>> = {
        let strategy_output = last_strategy
            .as_ref()
            .and_then(|s| s.output_name())
            .and_then(|name| tensor_cache.try_get(name));
        if let Some(arr) = strategy_output {
            vec![arr]
        } else if !model_meta.output_names.is_empty() {
            let found: Vec<_> = model_meta
                .output_names
                .iter()
                .filter_map(|n| tensor_cache.try_get(n))
                .collect();
            if found.is_empty() {
                tracing::warn!(
                    expected = ?model_meta.output_names,
                    available = ?tensor_cache.keys().collect::<Vec<_>>(),
                    "none of the declared output_names found in tensor cache"
                );
            }
            found
        } else {
            last_slice
                .dependencies
                .output
                .iter()
                .find_map(|n| tensor_cache.try_get(n))
                .into_iter()
                .collect()
        }
    };
    if output_arrs.is_empty() {
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
        return Err(match first_error {
            Some(err) => DsperseError::Pipeline(format!("pipeline failed at {err}")),
            None => DsperseError::Pipeline(format!(
                "no output tensor found for last slice {last_slice_id}"
            )),
        });
    }
    let output_path = run_dir.join(crate::utils::paths::OUTPUT_FILE);
    let output_val = Value::Array(output_arrs.iter().map(|arr| arrayd_to_value(arr)).collect());
    write_msgpack(
        &output_path,
        &build_msgpack_map(vec![("output_data", output_val)]),
    )?;

    Ok(final_meta)
}

fn run_combined_inference(
    slices_dir: &Path,
    input_path: &Path,
    run_dir: &Path,
    backend: &JstproveBackend,
    config: &RunConfig,
    model_meta: &ModelMetadata,
) -> Result<RunMetadata> {
    let combined_path =
        crate::slicer::combiner::ensure_combined_materialized(slices_dir, model_meta)?;

    let donor_model = load_donor_model(config.weights_onnx.as_ref())?;
    let donor_map = donor_init_map(donor_model.as_ref())?;
    if let Some(ref map) = donor_map {
        let combined_model = crate::slicer::onnx_proto::load_model(&combined_path)?;
        let combined_graph = combined_model
            .graph
            .as_ref()
            .ok_or_else(|| DsperseError::Pipeline("combined ONNX missing graph".into()))?;
        crate::slicer::onnx_proto::validate_initializer_compatibility(
            &combined_graph.initializer,
            map,
            "combined",
        )?;
        tracing::info!(
            weights = %config.weights_onnx.as_ref().unwrap().display(),
            "validated consumer weights against combined ONNX"
        );
    }

    std::fs::create_dir_all(run_dir).map_err(|e| DsperseError::io(e, run_dir))?;

    let input_data = read_msgpack(input_path)?;
    let input_val = extract_input_data(&input_data).ok_or_else(|| {
        DsperseError::Pipeline(
            "input has no recognized input key (input_data, input, data, inputs)".into(),
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

    let input_copy = run_dir.join(crate::utils::paths::INPUT_FILE);
    write_msgpack(&input_copy, &input_data)?;

    let effective_combined = if let Some(ref map) = donor_map {
        Some(crate::slicer::onnx_proto::build_patched_onnx(
            &combined_path,
            map,
        )?)
    } else {
        None
    };
    let effective_path = effective_combined
        .as_ref()
        .map_or(combined_path.as_path(), |t| t.path());

    let named_outputs = if input_val.is_map() {
        let mut cache = TensorStore::new();
        for name in declared_inputs {
            let v = map_get_ref(input_val, name)
                .ok_or_else(|| DsperseError::Pipeline(format!("input map missing key {name:?}")))?;
            cache.put(name.clone(), value_to_arrayd(v)?);
        }
        let inputs: Vec<String> = declared_inputs.clone();
        run_onnx_inference_multi_named(effective_path, &cache, &inputs)?
    } else if declared_inputs.len() == 1 {
        let input_arr = value_to_arrayd(input_val)?;
        run_onnx_inference_named(effective_path, &input_arr)?
    } else {
        return Err(DsperseError::Pipeline(format!(
            "model declares {} inputs but input is not a map",
            declared_inputs.len()
        )));
    };

    tracing::info!(
        outputs = named_outputs.len(),
        "combined model inference complete"
    );

    let mut tensor_cache = TensorStore::new();
    for (name, (data, shape)) in &named_outputs {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data.clone())
            .map_err(|e| DsperseError::Pipeline(format!("output reshape '{name}': {e}")))?;
        tensor_cache.put(name.clone(), arr);
    }

    for name in declared_inputs {
        if !tensor_cache.contains(name) {
            if input_val.is_map() {
                let v = map_get_ref(input_val, name).ok_or_else(|| {
                    DsperseError::Pipeline(format!(
                        "combined fallback: input map missing key {name:?}"
                    ))
                })?;
                tensor_cache.put(name.clone(), value_to_arrayd(v)?);
            } else if declared_inputs.len() == 1 {
                tensor_cache.put(name.clone(), value_to_arrayd(input_val)?);
            }
        }
    }

    crate::slicer::materializer::ensure_all_slices_materialized(slices_dir, model_meta)?;
    let chain = build_execution_chain(model_meta, slices_dir)?;
    let run_meta = build_run_metadata(model_meta, slices_dir, &chain)?;

    let mut results: Vec<ExecutionResultEntry> = Vec::new();

    for slice in &model_meta.slices {
        let slice_id = format!("slice_{}", slice.index);
        let node = chain
            .nodes
            .get(&slice_id)
            .ok_or_else(|| DsperseError::Pipeline(format!("missing node {slice_id}")))?;

        let slice_meta = run_meta.slices.get(&slice_id).ok_or_else(|| {
            DsperseError::Pipeline(format!("missing run slice metadata {slice_id}"))
        })?;

        let slice_run_dir = run_dir.join(&slice_id);
        std::fs::create_dir_all(&slice_run_dir).map_err(|e| DsperseError::io(e, &slice_run_dir))?;

        if !node.use_circuit {
            results.push(ExecutionResultEntry {
                slice_id: slice_id.clone(),
                witness_execution: Some(ExecutionInfo {
                    method: ExecutionMethod::OnnxOnly,
                    success: true,
                    error: None,
                    witness_file: None,
                    tile_exec_infos: Vec::new(),
                }),
                proof_execution: None,
                verification_execution: None,
            });
            continue;
        }

        let strategy = ExecutionStrategy::from_metadata(slice_meta, node.use_circuit)?;

        if let ExecutionStrategy::ChannelSplit(_) = &strategy {
            return Err(DsperseError::Pipeline(format!(
                "{slice_id}: combined mode does not support channel-split circuit slices; use --combined false"
            )));
        }

        if let ExecutionStrategy::Tiled(tiling) = &strategy {
            let exec_info = execute_combined_tiled(
                slices_dir,
                &slice_run_dir,
                &slice_id,
                tiling,
                slice_meta.jstprove_circuit_path.as_deref(),
                &mut tensor_cache,
                backend,
                config,
                donor_map.as_ref(),
            )?;

            let success = exec_info.success;
            results.push(ExecutionResultEntry {
                slice_id: slice_id.clone(),
                witness_execution: Some(exec_info),
                proof_execution: None,
                verification_execution: None,
            });

            if !success {
                break;
            }
            continue;
        }

        let circuit_path = slice_meta
            .jstprove_circuit_path
            .as_deref()
            .map(|p| resolve_relative_path(slices_dir, p))
            .transpose()?
            .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {slice_id}")))?;

        let params = backend.load_params(&circuit_path)?;
        let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

        if donor_map.is_some() && !is_wai {
            return Err(DsperseError::Pipeline(format!(
                "{slice_id}: consumer weights require circuits compiled with --weights-as-inputs"
            )));
        }

        let activation_inputs: Vec<String> = slice
            .dependencies
            .filtered_inputs
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();

        let witness_result = if activation_inputs.is_empty() {
            Err(DsperseError::Pipeline(format!(
                "{slice_id}: no activation inputs declared for circuit slice"
            )))
        } else if activation_inputs.len() == 1 {
            let input_name = &activation_inputs[0];
            let input_arr = tensor_cache.get(input_name).map_err(|_| {
                DsperseError::Pipeline(format!(
                    "{slice_id}: activation input '{input_name}' not found in combined model outputs"
                ))
            })?;

            if is_wai {
                let onnx_path = slice.resolve_onnx(slices_dir)?;
                generate_wai_witness(
                    backend,
                    &circuit_path,
                    &onnx_path,
                    donor_map.as_ref(),
                    params.as_ref().unwrap(),
                    input_arr,
                )
            } else {
                let flat: Vec<f64> = input_arr.iter().copied().collect();
                backend.witness_f64(&circuit_path, &flat, &[])
            }
        } else {
            Err(DsperseError::Pipeline(format!(
                "{slice_id}: combined mode does not support multi-input circuit slices; use --combined false for per-slice execution"
            )))
        };

        match witness_result {
            Ok(witness_bytes) => {
                let witness_path = slice_run_dir.join(crate::utils::paths::WITNESS_FILE);
                std::fs::write(&witness_path, &witness_bytes)
                    .map_err(|e| DsperseError::io(e, &witness_path))?;

                tracing::info!(slice = %slice_id, "witness generated from combined outputs");

                results.push(ExecutionResultEntry {
                    slice_id: slice_id.clone(),
                    witness_execution: Some(ExecutionInfo {
                        method: ExecutionMethod::JstproveGenWitness,
                        success: true,
                        error: None,
                        witness_file: Some(witness_path.to_string_lossy().into_owned()),
                        tile_exec_infos: Vec::new(),
                    }),
                    proof_execution: None,
                    verification_execution: None,
                });
            }
            Err(e) => {
                tracing::error!(slice = %slice_id, error = %e, "witness generation failed");
                results.push(ExecutionResultEntry {
                    slice_id: slice_id.clone(),
                    witness_execution: Some(ExecutionInfo {
                        method: ExecutionMethod::JstproveGenWitness,
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
        }
    }

    let mut final_meta = run_meta;
    final_meta.execution_chain.execution_results = results;
    final_meta.run_directory = Some(run_dir.to_string_lossy().into_owned());

    let witness_failure = final_meta
        .execution_chain
        .execution_results
        .iter()
        .filter_map(|r| {
            r.witness_execution
                .as_ref()
                .filter(|w| !w.success)
                .and_then(|w| w.error.as_ref())
                .map(|err| format!("{}: {err}", r.slice_id))
        })
        .next();
    if let Some(err) = witness_failure {
        let meta_out = run_dir.join(crate::utils::paths::METADATA_FILE);
        let _ = crate::utils::metadata::save_run_metadata(&meta_out, &final_meta);
        return Err(DsperseError::Pipeline(format!(
            "combined pipeline failed at {err}"
        )));
    }

    let meta_out = run_dir.join(crate::utils::paths::METADATA_FILE);
    crate::utils::metadata::save_run_metadata(&meta_out, &final_meta)?;

    let last_slice = model_meta
        .slices
        .last()
        .ok_or_else(|| DsperseError::Pipeline("model has no slices".into()))?;
    let output_arrs: Vec<&ArrayD<f64>> = if !model_meta.output_names.is_empty() {
        model_meta
            .output_names
            .iter()
            .filter_map(|n| tensor_cache.try_get(n))
            .collect()
    } else {
        last_slice
            .dependencies
            .output
            .iter()
            .find_map(|n| tensor_cache.try_get(n))
            .into_iter()
            .collect()
    };

    if output_arrs.is_empty() {
        let expected: Vec<&str> = if !model_meta.output_names.is_empty() {
            model_meta.output_names.iter().map(String::as_str).collect()
        } else {
            last_slice
                .dependencies
                .output
                .iter()
                .map(String::as_str)
                .collect()
        };
        let available: Vec<&String> = tensor_cache.keys().collect();
        return Err(DsperseError::Pipeline(format!(
            "no output tensor found in combined model outputs; expected {expected:?}, available {available:?}"
        )));
    }

    let output_path = run_dir.join(crate::utils::paths::OUTPUT_FILE);
    let output_val = Value::Array(output_arrs.iter().map(|arr| arrayd_to_value(arr)).collect());
    write_msgpack(
        &output_path,
        &build_msgpack_map(vec![("output_data", output_val)]),
    )?;

    tracing::info!(
        run_dir = %run_dir.display(),
        slices = model_meta.slices.len(),
        "combined inference complete"
    );

    Ok(final_meta)
}

#[allow(clippy::too_many_arguments)]
fn execute_slice(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    node: &ExecutionNode,
    meta: &RunSliceMetadata,
    tensor_cache: &mut TensorStore,
    backend: &JstproveBackend,
    config: &RunConfig,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let strategy = ExecutionStrategy::from_metadata(meta, node.use_circuit)?;
    match strategy {
        ExecutionStrategy::ChannelSplit(cs) => {
            let target_shape = meta
                .dependencies
                .output
                .iter()
                .position(|name| name == &cs.output_name)
                .and_then(|idx| meta.output_shape.get(idx))
                .map(|v| v.as_slice());
            if target_shape.is_none() {
                tracing::debug!(
                    slice = %slice_id,
                    output_name = %cs.output_name,
                    "target_shape lookup failed; output will not be reshaped"
                );
            }
            execute_channel_split(
                slices_dir,
                slice_run_dir,
                slice_id,
                cs,
                target_shape,
                tensor_cache,
                backend,
                donor_init_map,
            )
        }
        ExecutionStrategy::Tiled(tiling) => {
            let slice_circuit = meta
                .jstprove_circuit_path
                .as_deref()
                .map(|p| resolve_relative_path(slices_dir, p))
                .transpose()?;
            execute_tiled(
                slices_dir,
                slice_run_dir,
                slice_id,
                tiling,
                slice_circuit.as_deref(),
                tensor_cache,
                backend,
                config,
                donor_init_map,
            )
        }
        ExecutionStrategy::Single { .. } => execute_single(
            slices_dir,
            slice_run_dir,
            slice_id,
            node,
            meta,
            tensor_cache,
            backend,
            donor_init_map,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_single(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    node: &ExecutionNode,
    meta: &RunSliceMetadata,
    tensor_cache: &mut TensorStore,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let inputs: Vec<String> = meta
        .dependencies
        .filtered_inputs
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect();
    let multi_input = inputs.len() > 1;

    if inputs.is_empty() {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: no activation inputs declared"
        )));
    }

    let onnx_path = PathBuf::from(&meta.path);

    let patched_onnx = if let Some(map) = donor_init_map {
        Some(crate::slicer::onnx_proto::build_patched_onnx(
            &onnx_path, map,
        )?)
    } else {
        None
    };
    let effective_onnx: &Path = patched_onnx
        .as_ref()
        .map_or(onnx_path.as_path(), |t| t.path());

    if node.use_circuit {
        let circuit_path = meta
            .jstprove_circuit_path
            .as_deref()
            .map(|p| resolve_relative_path(slices_dir, p))
            .transpose()?
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

        let input_tensor = tensor_cache.gather(&inputs[..1])?;
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
            let flat: Vec<f64> = input_tensor.iter().copied().collect();
            backend.witness_f64(&circuit_path, &flat, &[])?
        };

        let witness_path = slice_run_dir.join(crate::utils::paths::WITNESS_FILE);
        std::fs::write(&witness_path, &witness_bytes)
            .map_err(|e| DsperseError::io(e, &witness_path))?;

        store_named_outputs(tensor_cache, &meta.dependencies.output, named)?;

        Ok(ExecutionInfo {
            method: ExecutionMethod::JstproveGenWitness,
            success: true,
            error: None,
            witness_file: Some(witness_path.to_string_lossy().into_owned()),
            tile_exec_infos: Vec::new(),
        })
    } else {
        let named = if multi_input {
            run_onnx_inference_multi_named(effective_onnx, tensor_cache, &inputs)?
        } else {
            let input_tensor = tensor_cache.gather(&inputs)?;
            run_onnx_inference_named(effective_onnx, &input_tensor)?
        };
        store_named_outputs(tensor_cache, &meta.dependencies.output, named)?;

        Ok(ExecutionInfo {
            method: ExecutionMethod::OnnxOnly,
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
    tensor_cache: &mut TensorStore,
    backend: &JstproveBackend,
    config: &RunConfig,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let all_names = tiling.all_input_names();
    let multi_input = all_names.len() > 1;
    let is_1d = tiling.ndim == 3;

    let all_tiles_dyn = prepare_tiles_from_cache(tiling, tensor_cache, is_1d)?;

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
    let tile_onnx = first_tile_info
        .map(|ti| resolve_relative_path(slices_dir, &ti.path))
        .transpose()?;

    let patched_tile_onnx = match (&tile_onnx, donor_init_map) {
        (Some(onnx_path), Some(map)) => Some(crate::slicer::onnx_proto::build_patched_onnx(
            onnx_path, map,
        )?),
        _ => None,
    };
    let effective_tile_onnx = patched_tile_onnx.as_ref().map(|t| t.path().to_path_buf());
    let effective_tile_onnx_ref = effective_tile_onnx.as_deref().or(tile_onnx.as_deref());

    let warm_model = if multi_input || is_1d {
        None
    } else {
        match (effective_tile_onnx_ref, all_tiles_dyn[0].first()) {
            (Some(onnx_path), Some(sample)) => {
                let shape = sample.shape().to_vec();
                let model = crate::backend::onnx::WarmModel::load(onnx_path, &shape)?;
                tracing::info!(slice = %slice_id, "loaded ONNX model");
                Some(model)
            }
            _ => None,
        }
    };

    let circuit_path = match first_tile_info.and_then(|ti| ti.jstprove_circuit_path.as_deref()) {
        Some(p) => Some(resolve_relative_path(slices_dir, p)?),
        None => None,
    }
    .or_else(|| slice_circuit_path.map(|p| p.to_path_buf()));

    if multi_input && circuit_path.is_some() {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: tiled circuit execution does not support multiple activation inputs"
        )));
    }

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

                let tile_output = if multi_input || is_1d {
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

                let witness_result = if let Some(ref wc) = warm_circuit {
                    let flat: Vec<f64> = tile_dyn.iter().copied().collect();
                    wc.witness_f64(&flat)
                } else {
                    let flat: Vec<f64> = tile_dyn.iter().copied().collect();
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
    let reconstructed = if is_1d {
        let r = reconstruct_from_tiles_1d(&tile_outputs, tiling)?;
        trim_to_original_seq(r, tiling)?
    } else {
        let r = reconstruct_from_tiles(&tile_outputs, tiling)?;
        trim_to_original_dims(r, tiling)?
    };
    tensor_cache.put(tiling.output_name.clone(), reconstructed);

    Ok(ExecutionInfo {
        method: ExecutionMethod::Tiled,
        success: true,
        error: None,
        witness_file: None,
        tile_exec_infos: tile_results,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_combined_tiled(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    tiling: &TilingInfo,
    slice_circuit_path: Option<&str>,
    tensor_cache: &mut TensorStore,
    backend: &JstproveBackend,
    config: &RunConfig,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let all_names = tiling.all_input_names();

    let is_1d = tiling.ndim == 3;
    let all_tiles_dyn = prepare_tiles_from_cache(tiling, tensor_cache, is_1d)?;

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

    let circuit_path = match first_tile_info
        .and_then(|ti| ti.jstprove_circuit_path.as_deref())
        .or(slice_circuit_path)
    {
        Some(p) => Some(resolve_relative_path(slices_dir, p)?),
        None => None,
    };

    let circuit_path = match circuit_path {
        Some(p) => p,
        None => {
            return Ok(ExecutionInfo {
                method: ExecutionMethod::Tiled,
                success: true,
                error: None,
                witness_file: None,
                tile_exec_infos: (0..num_tiles)
                    .map(|i| TileResult::success(i, Some(ExecutionMethod::OnnxOnly), 0.0))
                    .collect(),
            });
        }
    };

    let tile_onnx = first_tile_info
        .map(|ti| resolve_relative_path(slices_dir, &ti.path))
        .transpose()?;

    let patched_tile_onnx = match (&tile_onnx, donor_init_map) {
        (Some(onnx_path), Some(map)) => Some(crate::slicer::onnx_proto::build_patched_onnx(
            onnx_path, map,
        )?),
        _ => None,
    };
    let effective_tile_onnx = patched_tile_onnx.as_ref().map(|t| t.path().to_path_buf());
    let effective_tile_onnx_ref = effective_tile_onnx.as_deref().or(tile_onnx.as_deref());

    let params = backend.load_params(&circuit_path)?;
    let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

    if donor_init_map.is_some() && !is_wai {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: consumer weights require circuits compiled with --weights-as-inputs"
        )));
    }

    if all_names.len() > 1 {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: tiled circuit execution does not support multiple activation inputs"
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

                let tile_dyn = all_tiles_dyn[0][tile_idx].clone();
                let flat: Vec<f64> = tile_dyn.iter().copied().collect();

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

    Ok(ExecutionInfo {
        method: ExecutionMethod::Tiled,
        success: true,
        error: None,
        witness_file: None,
        tile_exec_infos: collected,
    })
}

fn reshape_channel_split_output(
    arr: ArrayD<f64>,
    target_shape: Option<&[i64]>,
) -> Result<ArrayD<f64>> {
    let Some(raw) = target_shape else {
        return Ok(arr);
    };
    let target: Vec<usize> = raw
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                DsperseError::Pipeline(format!("negative dimension {d} in output_shape"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if arr.shape() == target.as_slice() {
        return Ok(arr);
    }
    let actual_elems: usize = arr.shape().iter().product();
    let target_elems: usize = target.iter().product();
    if actual_elems != target_elems {
        return Ok(arr);
    }
    let actual_shape: Vec<usize> = arr.shape().to_vec();
    arr.into_shape_with_order(ndarray::IxDyn(&target))
        .map_err(|e| {
            DsperseError::Pipeline(format!(
                "channel_split output reshape from {actual_shape:?} to {target:?}: {e}",
            ))
        })
}

#[allow(clippy::too_many_arguments)]
fn execute_channel_split(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    cs: &ChannelSplitInfo,
    target_shape: Option<&[i64]>,
    tensor_cache: &mut TensorStore,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ExecutionInfo> {
    let input_arr = tensor_cache.get(&cs.input_name)?.clone();

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
        if nc > 0 && !total_elements.is_multiple_of(nc) {
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

        let group_output = execute_channel_group(
            slices_dir,
            &group_dir,
            group,
            &group_input_dyn,
            backend,
            donor_init_map,
        )?;

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
                if h > 0 && out_spatial > 0 && out_spatial.is_multiple_of(h) {
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
        let bias_file = resolve_relative_path(slices_dir, bias_path_str)?;
        if !bias_file.exists() {
            return Err(DsperseError::Pipeline(format!(
                "configured bias file not found: {} (bias_path={bias_path_str})",
                bias_file.display()
            )));
        }
        let bias_data = read_msgpack(&bias_file)?;
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
            let output = reshape_channel_split_output(acc.into_dyn(), target_shape)?;
            tensor_cache.put(cs.output_name.clone(), output);
        }
        None => {
            return Err(DsperseError::Pipeline(format!(
                "channel_split produced no output for '{}'",
                cs.output_name
            )));
        }
    }

    Ok(ExecutionInfo {
        method: ExecutionMethod::ChannelSplit,
        success: true,
        error: None,
        witness_file: None,
        tile_exec_infos: Vec::new(),
    })
}

fn execute_channel_group(
    slices_dir: &Path,
    group_dir: &Path,
    group: &ChannelGroupInfo,
    group_input: &ArrayD<f64>,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ArrayD<f64>> {
    let onnx_path = resolve_relative_path(slices_dir, &group.path)?;

    let patched_onnx = if let Some(map) = donor_init_map {
        Some(crate::slicer::onnx_proto::build_patched_onnx(
            &onnx_path, map,
        )?)
    } else {
        None
    };
    let effective_onnx = patched_onnx
        .as_ref()
        .map_or(onnx_path.as_path(), |t| t.path());

    if let Some(ref circuit_path_str) = group.jstprove_circuit_path {
        let circuit_path = resolve_relative_path(slices_dir, circuit_path_str)?;

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
            let flat: Vec<f64> = group_input.iter().copied().collect();
            backend.witness_f64(&circuit_path, &flat, &[])?
        };

        let witness_path = group_dir.join(crate::utils::paths::WITNESS_FILE);
        std::fs::write(&witness_path, &witness_bytes)
            .map_err(|e| DsperseError::io(e, &witness_path))?;

        Ok(output_tensor)
    } else {
        run_onnx_inference(effective_onnx, group_input)
    }
}

fn prepare_tiles_from_cache(
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

fn trim_to_original_dims(arr: ArrayD<f64>, tiling: &TilingInfo) -> Result<ArrayD<f64>> {
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

fn split_into_tiles_1d(input: &ArrayD<f64>, tiling: &TilingInfo) -> Result<Vec<ArrayD<f64>>> {
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

fn reconstruct_from_tiles_1d(
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

fn trim_to_original_seq(arr: ArrayD<f64>, tiling: &TilingInfo) -> Result<ArrayD<f64>> {
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

fn store_named_outputs(
    tensor_cache: &mut TensorStore,
    output_names: &[String],
    named_outputs: HashMap<String, (Vec<f64>, Vec<usize>)>,
) -> Result<()> {
    for name in output_names {
        if let Some((data, shape)) = named_outputs.get(name) {
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data.clone())
                .map_err(|e| DsperseError::Pipeline(format!("output reshape '{name}': {e}")))?;
            tensor_cache.put(name.clone(), arr);
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

fn run_onnx_inference_named(onnx_path: &Path, input: &ArrayD<f64>) -> Result<NamedOutputs> {
    let input_flat: Vec<f64> = input.iter().copied().collect();
    let input_shape = input.shape();
    crate::backend::onnx::run_inference_named(onnx_path, &input_flat, input_shape)
}

fn run_onnx_inference_multi_named(
    onnx_path: &Path,
    tensor_cache: &TensorStore,
    input_names: &[String],
) -> Result<NamedOutputs> {
    let inputs: Vec<(&str, Vec<f64>, Vec<usize>)> = input_names
        .iter()
        .map(|name| {
            let arr = tensor_cache.get(name)?;
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
) -> Result<ExecutionChain> {
    let mut nodes = HashMap::new();
    let mut head = None;

    for (i, slice) in model_meta.slices.iter().enumerate() {
        let slice_id = format!("slice_{}", slice.index);
        let slice_dir = slice_dir_path(slices_dir, slice.index);

        if i == 0 {
            head = Some(slice_id.clone());
        }

        let (has_circuit, circuit_path) = if slice.compilation.jstprove.compiled {
            let path = slice.compilation.jstprove.files.compiled.clone();
            (true, path)
        } else {
            let bundle = slice_dir.join("jstprove/circuit.bundle");
            if bundle.is_dir() {
                tracing::info!(slice = %slice_id, "detected circuit on filesystem (metadata.compiled=false)");
                let rel = format!("slice_{}/jstprove/circuit.bundle", slice.index);
                (true, Some(rel))
            } else {
                (false, None)
            }
        };
        let next = model_meta
            .slices
            .get(i + 1)
            .map(|s| format!("slice_{}", s.index));

        let onnx_path = Some(
            slice
                .resolve_onnx(slices_dir)?
                .to_string_lossy()
                .into_owned(),
        );

        let backend = if has_circuit {
            BackendKind::Jstprove
        } else {
            BackendKind::Onnx
        };

        nodes.insert(
            slice_id.clone(),
            ExecutionNode {
                slice_id: slice_id.clone(),
                primary: Some(backend.to_string()),
                fallbacks: if has_circuit {
                    vec!["onnx".into()]
                } else {
                    Vec::new()
                },
                use_circuit: has_circuit,
                next,
                circuit_path,
                onnx_path,
                backend,
            },
        );
    }

    Ok(ExecutionChain {
        head,
        nodes,
        fallback_map: HashMap::new(),
        execution_results: Vec::new(),
        jstprove_proved_slices: 0,
        jstprove_verified_slices: 0,
    })
}

pub(crate) fn build_run_metadata(
    model_meta: &ModelMetadata,
    slices_dir: &Path,
    chain: &ExecutionChain,
) -> Result<RunMetadata> {
    let mut slices = HashMap::new();

    for slice in &model_meta.slices {
        let slice_id = format!("slice_{}", slice.index);
        let node = chain.nodes.get(&slice_id);
        let has_circuit = node.is_some_and(|n| n.use_circuit);

        let run_slice = RunSliceMetadata {
            path: slice
                .resolve_onnx(slices_dir)?
                .to_string_lossy()
                .into_owned(),
            input_shape: slice.shape.tensor_shape.input.clone(),
            output_shape: slice.shape.tensor_shape.output.clone(),
            dependencies: slice.dependencies.clone(),
            tiling: slice.tiling.clone(),
            channel_split: slice.channel_split.clone(),
            backend: if has_circuit {
                BackendKind::Jstprove
            } else {
                BackendKind::Onnx
            },
            jstprove_circuit_path: node.and_then(|n| n.circuit_path.clone()),
            jstprove_settings_path: None,
        };

        slices.insert(slice_id, run_slice);
    }

    Ok(RunMetadata {
        slices,
        execution_chain: chain.clone(),
        packaging_type: None,
        source_path: Some(slices_dir.to_string_lossy().into_owned()),
        run_directory: None,
        model_path: None,
    })
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

pub fn extract_onnx_initializers(
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    fn make_tiling(
        tile_size: usize,
        tiles_y: usize,
        tiles_x: usize,
        halo: [i64; 4],
        out_tile: [i64; 2],
        c_out: usize,
    ) -> TilingInfo {
        TilingInfo {
            slice_idx: 0,
            tile_size,
            num_tiles: tiles_y * tiles_x,
            tiles_y,
            tiles_x,
            halo,
            out_tile,
            stride: [1, 1],
            c_in: 1,
            c_out,
            input_name: "input".into(),
            output_name: "output".into(),
            input_names: vec![],
            ndim: 4,
            h: tiles_y * tile_size,
            w: tiles_x * tile_size,
            tile: None,
            tiles: None,
        }
    }

    #[test]
    fn reshape_to_4d_valid() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let arr = reshape_to_4d(&data, 2, 3, 4).unwrap();
        assert_eq!(arr.dim(), (1, 2, 3, 4));
    }

    #[test]
    fn reshape_to_4d_single_element() {
        let data = vec![42.0];
        let arr = reshape_to_4d(&data, 1, 1, 1).unwrap();
        assert_eq!(arr.dim(), (1, 1, 1, 1));
        assert_eq!(arr[[0, 0, 0, 0]], 42.0);
    }

    #[test]
    fn reshape_to_4d_mismatch() {
        let data = vec![1.0; 10];
        assert!(reshape_to_4d(&data, 2, 3, 4).is_err());
    }

    #[test]
    fn reshape_to_4d_empty() {
        let data: Vec<f64> = vec![];
        assert!(reshape_to_4d(&data, 1, 1, 1).is_err());
    }

    #[test]
    fn split_into_tiles_2x2_no_halo() {
        let input =
            Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|i| i as f64).collect()).unwrap();
        let tiling = make_tiling(2, 2, 2, [0, 0, 0, 0], [2, 2], 1);
        let tiles = split_into_tiles(&input, &tiling).unwrap();
        assert_eq!(tiles.len(), 4);
        for tile in &tiles {
            assert_eq!(tile.dim(), (1, 1, 2, 2));
        }
    }

    #[test]
    fn split_into_tiles_with_halo() {
        let input =
            Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|i| i as f64).collect()).unwrap();
        let tiling = make_tiling(2, 2, 2, [1, 1, 1, 1], [2, 2], 1);
        let tiles = split_into_tiles(&input, &tiling).unwrap();
        assert_eq!(tiles.len(), 4);
        for tile in &tiles {
            assert_eq!(tile.dim(), (1, 1, 4, 4));
        }
    }

    #[test]
    fn split_into_tiles_negative_halo_rejected() {
        let input = Array4::zeros((1, 1, 4, 4));
        let tiling = make_tiling(2, 2, 2, [-1, 0, 0, 0], [2, 2], 1);
        assert!(split_into_tiles(&input, &tiling).is_err());
    }

    #[test]
    fn split_into_tiles_batch_gt1_rejected() {
        let input = Array4::zeros((2, 1, 4, 4));
        let tiling = make_tiling(2, 1, 1, [0, 0, 0, 0], [2, 2], 1);
        assert!(split_into_tiles(&input, &tiling).is_err());
    }

    #[test]
    fn reconstruct_from_tiles_2x2() {
        let c_out = 1;
        let out_h = 2usize;
        let out_w = 2usize;
        let tiling = make_tiling(4, 2, 2, [0, 0, 0, 0], [out_h as i64, out_w as i64], c_out);

        let tiles: Vec<ArrayD<f64>> = (0..4)
            .map(|i| {
                ArrayD::from_shape_vec(
                    IxDyn(&[1, c_out, out_h, out_w]),
                    vec![i as f64; c_out * out_h * out_w],
                )
                .unwrap()
            })
            .collect();

        let output = reconstruct_from_tiles(&tiles, &tiling).unwrap();
        assert_eq!(output.shape(), &[1, c_out, 4, 4]);
    }

    #[test]
    fn reconstruct_from_tiles_empty() {
        let tiling = make_tiling(2, 1, 1, [0, 0, 0, 0], [2, 2], 1);
        assert!(reconstruct_from_tiles(&[], &tiling).is_err());
    }

    #[test]
    fn reconstruct_from_tiles_wrong_element_count() {
        let tiling = make_tiling(2, 1, 1, [0, 0, 0, 0], [2, 2], 1);
        let bad_tile = vec![ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0; 3]).unwrap()];
        assert!(reconstruct_from_tiles(&bad_tile, &tiling).is_err());
    }

    #[test]
    fn reconstruct_from_tiles_wrong_tile_count() {
        let c_out = 1;
        let out_h = 2i64;
        let out_w = 2i64;
        let tiling = make_tiling(4, 2, 2, [0, 0, 0, 0], [out_h, out_w], c_out);
        let make_tile = || {
            ArrayD::from_shape_vec(
                IxDyn(&[1, c_out, out_h as usize, out_w as usize]),
                vec![0.0f64; c_out * out_h as usize * out_w as usize],
            )
            .unwrap()
        };
        let too_few: Vec<ArrayD<f64>> = (0..3).map(|_| make_tile()).collect();
        assert!(reconstruct_from_tiles(&too_few, &tiling).is_err());
        let too_many: Vec<ArrayD<f64>> = (0..5).map(|_| make_tile()).collect();
        assert!(reconstruct_from_tiles(&too_many, &tiling).is_err());
    }

    #[test]
    fn split_reconstruct_roundtrip() {
        let c = 2;
        let h = 8;
        let w = 8;
        let data: Vec<f64> = (0..(c * h * w)).map(|i| i as f64).collect();
        let input = Array4::from_shape_vec((1, c, h, w), data).unwrap();

        let tile_size = 4;
        let tiling = make_tiling(tile_size, 2, 2, [0, 0, 0, 0], [4, 4], c);

        let tiles = split_into_tiles(&input, &tiling).unwrap();
        assert_eq!(tiles.len(), 4);

        let tile_outputs: Vec<ArrayD<f64>> = tiles.into_iter().map(|t| t.into_dyn()).collect();
        let reconstructed = reconstruct_from_tiles(&tile_outputs, &tiling).unwrap();
        assert_eq!(reconstructed.shape(), &[1, c, h, w]);

        let input_dyn = input.into_dyn();
        assert_eq!(input_dyn, reconstructed);
    }

    #[test]
    fn store_named_outputs_basic() {
        let mut cache = TensorStore::new();
        let names = vec!["out_a".to_string(), "out_b".to_string()];
        let mut named = HashMap::new();
        named.insert("out_a".to_string(), (vec![1.0, 2.0], vec![2]));
        named.insert("out_b".to_string(), (vec![3.0], vec![1]));

        store_named_outputs(&mut cache, &names, named).unwrap();
        assert_eq!(cache.get("out_a").unwrap().shape(), &[2]);
        assert_eq!(cache.get("out_b").unwrap().shape(), &[1]);
    }

    #[test]
    fn store_named_outputs_missing_name_ignored() {
        let mut cache = TensorStore::new();
        let names = vec!["missing".to_string()];
        let named = HashMap::new();
        store_named_outputs(&mut cache, &names, named).unwrap();
        assert!(!cache.contains("missing"));
    }

    #[test]
    fn run_config_default() {
        let config = RunConfig::default();
        assert_eq!(config.parallel, 1);
        assert!(!config.batch);
        assert!(config.weights_onnx.is_none());
        assert!(config.combined);
    }
}
