use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};

use jstprove_circuits::api::CircuitParamsType as CircuitParams;

use super::strategy::ExecutionStrategy;
use super::tensor_store::TensorStore;
use crate::backend::jstprove::JstproveBackend;
use crate::backend::onnx::NamedOutputs;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{
    ExecutionChain, ExecutionInfo, ExecutionMethod, ExecutionNode, ExecutionResultEntry,
    RunMetadata,
};
use crate::schema::metadata::{BackendKind, ModelMetadata, RunSliceMetadata};
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
    pub activations_dir: Option<PathBuf>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            parallel: 1,
            batch: false,
            weights_onnx: None,
            combined: true,
            activations_dir: None,
        }
    }
}

fn resolve_circuit_path_required(
    slices_dir: &Path,
    circuit_path: Option<&str>,
    label: &str,
) -> Result<PathBuf> {
    circuit_path
        .map(|p| resolve_relative_path(slices_dir, p))
        .transpose()?
        .ok_or_else(|| DsperseError::Pipeline(format!("no circuit path for {label}")))
}

pub(crate) fn resolve_circuit_path_optional(
    slices_dir: &Path,
    circuit_path: Option<&str>,
) -> Result<Option<PathBuf>> {
    circuit_path
        .map(|p| resolve_relative_path(slices_dir, p))
        .transpose()
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
            ExecutionStrategy::from_metadata(m, use_circuit).ok()
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

        if let ExecutionStrategy::DimSplit(_) = &strategy {
            return Err(DsperseError::Pipeline(format!(
                "{slice_id}: combined mode does not support dim-split circuit slices; use --combined false"
            )));
        }

        if let ExecutionStrategy::Tiled(tiling) = &strategy {
            let result = super::tiled::execute_combined_tiled(
                slices_dir,
                &slice_run_dir,
                &slice_id,
                tiling,
                slice_meta.jstprove_circuit_path.as_deref(),
                &tensor_cache,
                backend,
                config,
                donor_map.as_ref(),
            )?;
            for (name, tensor) in result.outputs {
                tensor_cache.put(name, tensor);
            }

            let success = result.info.success;
            results.push(ExecutionResultEntry {
                slice_id: slice_id.clone(),
                witness_execution: Some(result.info),
                proof_execution: None,
                verification_execution: None,
            });

            if !success {
                break;
            }
            continue;
        }

        let circuit_path = resolve_circuit_path_required(
            slices_dir,
            slice_meta.jstprove_circuit_path.as_deref(),
            &slice_id,
        )?;

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
        } else {
            let mut flat_activations: Vec<f64> = Vec::new();
            for input_name in &activation_inputs {
                let input_arr = tensor_cache.get(input_name).map_err(|_| {
                    DsperseError::Pipeline(format!(
                        "{slice_id}: activation input '{input_name}' not found in combined model outputs"
                    ))
                })?;
                flat_activations.extend(input_arr.iter());
            }

            if is_wai {
                let onnx_path = slice.resolve_onnx(slices_dir)?;
                let initializers = if let Some(donor) = donor_map.as_ref() {
                    let slice_model = crate::slicer::onnx_proto::load_model(&onnx_path)?;
                    let slice_graph = slice_model.graph.as_ref().ok_or_else(|| {
                        DsperseError::Pipeline(format!("{slice_id}: ONNX missing graph"))
                    })?;
                    let mut merged = crate::slicer::onnx_proto::build_initializer_map(slice_graph);
                    for (k, v) in donor.iter() {
                        merged.insert(k.clone(), *v);
                    }
                    extract_initializers_from_map(&merged, params.as_ref().unwrap())?
                } else {
                    extract_onnx_initializers(&onnx_path, params.as_ref().unwrap())?
                };
                backend.witness_f64(&circuit_path, &flat_activations, &initializers)
            } else {
                backend.witness_f64(&circuit_path, &flat_activations, &[])
            }
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

    if let Some(activations_dir) = config.activations_dir.as_ref() {
        crate::pipeline::activations::write_slice_activations(
            activations_dir,
            model_meta,
            &tensor_cache,
        )?;
    }

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
            let result = super::channel_split::execute_channel_split(
                slices_dir,
                slice_run_dir,
                slice_id,
                cs,
                target_shape,
                tensor_cache,
                backend,
                donor_init_map,
            )?;
            for (name, tensor) in result.outputs {
                tensor_cache.put(name, tensor);
            }
            Ok(result.info)
        }
        ExecutionStrategy::Tiled(tiling) => {
            let slice_circuit =
                resolve_circuit_path_optional(slices_dir, meta.jstprove_circuit_path.as_deref())?;
            let result = super::tiled::execute_tiled(
                slices_dir,
                slice_run_dir,
                slice_id,
                tiling,
                slice_circuit.as_deref(),
                tensor_cache,
                backend,
                config,
                donor_init_map,
            )?;
            for (name, tensor) in result.outputs {
                tensor_cache.put(name, tensor);
            }
            Ok(result.info)
        }
        ExecutionStrategy::DimSplit(ds) => {
            let target_shape = meta
                .dependencies
                .output
                .iter()
                .position(|name| name == &ds.output_name)
                .and_then(|idx| meta.output_shape.get(idx))
                .map(|v| v.as_slice());
            let result = super::dim_split::execute_dim_split(
                slices_dir,
                slice_run_dir,
                slice_id,
                ds,
                target_shape,
                tensor_cache,
                backend,
                donor_init_map,
            )?;
            for (name, tensor) in result.outputs {
                tensor_cache.put(name, tensor);
            }
            Ok(result.info)
        }
        ExecutionStrategy::Single { .. } => {
            let result = execute_single(
                slices_dir,
                slice_run_dir,
                slice_id,
                node,
                meta,
                tensor_cache,
                backend,
                donor_init_map,
            )?;
            for (name, tensor) in result.outputs {
                tensor_cache.put(name, tensor);
            }
            Ok(result.info)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_single(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    node: &ExecutionNode,
    meta: &RunSliceMetadata,
    tensor_cache: &TensorStore,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<crate::schema::execution::StrategyOutput> {
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
        let circuit_path = resolve_circuit_path_required(
            slices_dir,
            meta.jstprove_circuit_path.as_deref(),
            slice_id,
        )?;

        let params = backend.load_params(&circuit_path)?;
        let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

        if donor_init_map.is_some() && !is_wai {
            return Err(DsperseError::Pipeline(format!(
                "{slice_id}: consumer weights require circuits compiled with --weights-as-inputs"
            )));
        }

        let named = if multi_input {
            run_onnx_inference_multi_named(effective_onnx, tensor_cache, &inputs)?
        } else {
            let input_tensor = tensor_cache.gather(&inputs[..1])?;
            run_onnx_inference_named(effective_onnx, &input_tensor)?
        };

        let outputs = collect_named_outputs(&meta.dependencies.output, named)?;

        let flat_activations = flatten_cached_inputs(tensor_cache, &inputs)?;
        let witness_bytes = if is_wai {
            generate_wai_witness(
                backend,
                &circuit_path,
                &onnx_path,
                donor_init_map,
                params.as_ref().unwrap(),
                &flat_activations,
            )?
        } else {
            backend.witness_f64(&circuit_path, &flat_activations, &[])?
        };

        let witness_path = slice_run_dir.join(crate::utils::paths::WITNESS_FILE);
        std::fs::write(&witness_path, &witness_bytes)
            .map_err(|e| DsperseError::io(e, &witness_path))?;

        Ok(crate::schema::execution::StrategyOutput {
            info: ExecutionInfo {
                method: ExecutionMethod::JstproveGenWitness,
                success: true,
                error: None,
                witness_file: Some(witness_path.to_string_lossy().into_owned()),
                tile_exec_infos: Vec::new(),
            },
            outputs,
        })
    } else {
        let named = if multi_input {
            run_onnx_inference_multi_named(effective_onnx, tensor_cache, &inputs)?
        } else {
            let input_tensor = tensor_cache.gather(&inputs)?;
            run_onnx_inference_named(effective_onnx, &input_tensor)?
        };
        let outputs = collect_named_outputs(&meta.dependencies.output, named)?;

        Ok(crate::schema::execution::StrategyOutput {
            info: ExecutionInfo {
                method: ExecutionMethod::OnnxOnly,
                success: true,
                error: None,
                witness_file: None,
                tile_exec_infos: Vec::new(),
            },
            outputs,
        })
    }
}

#[cfg(test)]
fn store_named_outputs(
    tensor_cache: &mut TensorStore,
    output_names: &[String],
    named_outputs: HashMap<String, (Vec<f64>, Vec<usize>)>,
) -> Result<()> {
    for (name, tensor) in collect_named_outputs(output_names, named_outputs)? {
        tensor_cache.put(name, tensor);
    }
    Ok(())
}

fn collect_named_outputs(
    output_names: &[String],
    mut named_outputs: HashMap<String, (Vec<f64>, Vec<usize>)>,
) -> Result<Vec<(String, ArrayD<f64>)>> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for name in output_names {
        if !seen.insert(name) {
            return Err(DsperseError::Pipeline(format!(
                "duplicate declared output '{name}'"
            )));
        }
        let (data, shape) = named_outputs
            .remove(name)
            .ok_or_else(|| DsperseError::Pipeline(format!("missing declared output '{name}'")))?;
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| DsperseError::Pipeline(format!("output reshape '{name}': {e}")))?;
        result.push((name.clone(), arr));
    }
    Ok(result)
}

pub(crate) fn run_onnx_inference(onnx_path: &Path, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
    let input_flat: Vec<f64> = input.iter().copied().collect();
    let input_shape = input.shape();
    let (output_data, output_shape) =
        crate::backend::onnx::run_inference(onnx_path, &input_flat, input_shape)?;

    ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)
        .map_err(|e| DsperseError::Pipeline(format!("output reshape: {e}")))
}

pub(crate) fn run_onnx_inference_named(
    onnx_path: &Path,
    input: &ArrayD<f64>,
) -> Result<NamedOutputs> {
    let input_flat: Vec<f64> = input.iter().copied().collect();
    let input_shape = input.shape();
    crate::backend::onnx::run_inference_named(onnx_path, &input_flat, input_shape)
}

pub(crate) fn run_onnx_inference_multi_named(
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

        let bundle = slice_dir.join("jstprove/circuit.bundle");
        let (has_circuit, circuit_path) = if bundle.is_dir() {
            let rel = format!("slice_{}/jstprove/circuit.bundle", slice.index);
            (true, Some(rel))
        } else {
            (false, None)
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
            dim_split: slice.dim_split.clone(),
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

/// Names the slicer assigns to tiled/template runtime activation inputs. These
/// are never weights, so they must be excluded from the element-count fallback
/// in [`extract_initializers_from_map`].
fn is_activation_placeholder(name: &str) -> bool {
    name == "tile_in"
        || name.starts_with("tile_in_")
        || name == "dim_tmpl_in"
        || (name.starts_with("group_") && name.ends_with("_in"))
}

pub fn split_inline_wai_inputs(
    params: &CircuitParams,
    flat: &[f64],
) -> Option<(Vec<f64>, Vec<(Vec<f64>, Vec<usize>)>)> {
    if !params.weights_as_inputs {
        return None;
    }
    let total: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    if total == 0 || flat.len() != total {
        return None;
    }
    let mut activations: Vec<f64> = Vec::new();
    let mut initializers: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
    let mut cursor = 0usize;
    for io in &params.inputs {
        let n: usize = io.shape.iter().product();
        let seg = &flat[cursor..cursor + n];
        cursor += n;
        if is_activation_placeholder(&io.name) {
            activations.extend_from_slice(seg);
        } else {
            initializers.push((seg.to_vec(), io.shape.clone()));
        }
    }
    Some((activations, initializers))
}

pub(crate) fn extract_initializers_from_map(
    init_map: &HashMap<String, &TensorProto>,
    params: &CircuitParams,
) -> Result<Vec<(Vec<f64>, Vec<usize>)>> {
    // Resolve each declared input to its backing initializer tensor (if any).
    // First by exact name. DimSplit / ChannelSplit template circuits rename the
    // weight to a placeholder ("W") that never appears in the slice ONNX, whose
    // initializer keeps the original name (e.g. "onnx::MatMul_4162"); for those
    // we fall back to an element-count match against the as-yet-unconsumed
    // initializers, but only when that match is unique. An ambiguous or absent
    // match is left unresolved (treated as an activation) so a wrong weight can
    // never be silently bound into the witness.
    let mut consumed: HashSet<&str> = HashSet::new();
    let mut resolved: Vec<Option<&TensorProto>> = Vec::with_capacity(params.inputs.len());
    for io in &params.inputs {
        if let Some(tensor) = init_map.get(&io.name) {
            consumed.insert(io.name.as_str());
            resolved.push(Some(*tensor));
        } else {
            resolved.push(None);
        }
    }
    for (slot, io) in resolved.iter_mut().zip(&params.inputs) {
        if slot.is_some() {
            continue;
        }
        // Never resolve a tiled activation placeholder to an initializer. Names
        // like "tile_in", "tile_in_N", "dim_tmpl_in" and "group_N_in" are the
        // runtime activation inputs the slicer emits, not weights. Without this
        // guard, a model whose per-slice ONNX carries an unrelated initializer
        // of the same element count (e.g. a sibling layer's bias, or a full-model
        // ONNX) would bind that weight into the activation slot, dropping the
        // activation count to zero and rejecting the dispatched payload.
        if is_activation_placeholder(&io.name) {
            continue;
        }
        let target_elems: usize = io.shape.iter().product();
        if target_elems == 0 {
            continue;
        }
        let mut matches = init_map.iter().filter(|(name, t)| {
            !consumed.contains(name.as_str())
                && t.dims.iter().map(|&d| d as usize).product::<usize>() == target_elems
        });
        if let (Some((name, tensor)), None) = (matches.next(), matches.next()) {
            consumed.insert(name.as_str());
            *slot = Some(*tensor);
        }
    }

    let mut initializers = Vec::new();
    for (slot, io) in resolved.iter().zip(&params.inputs) {
        if let Some(tensor) = slot {
            let f32_vals = crate::slicer::onnx_proto::tensor_to_f32(tensor);
            let mut f64_vals: Vec<f64> = f32_vals.iter().map(|&v| f64::from(v)).collect();
            let target_shape = &io.shape;
            let tensor_shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
            let target_elems: usize = target_shape.iter().product();
            if f64_vals.len() < target_elems && !target_shape.is_empty() && !tensor_shape.is_empty()
            {
                let is_bias = tensor_shape.len() == 1;
                let pad_val: f64 = if is_bias { -10.0 } else { 0.0 };
                let last = target_shape.len() - 1;
                let target_last = target_shape[last];
                let donor_last = tensor_shape[last];
                if donor_last < target_last {
                    let rows = f64_vals.len() / donor_last.max(1);
                    let mut padded = Vec::with_capacity(target_elems);
                    for row in 0..rows {
                        let start = row * donor_last;
                        let end = start + donor_last;
                        padded.extend_from_slice(&f64_vals[start..end.min(f64_vals.len())]);
                        padded.resize(padded.len() + (target_last - donor_last), pad_val);
                    }
                    f64_vals = padded;
                }
            }
            let shape: Vec<usize> = if f64_vals.len() == target_elems {
                target_shape.clone()
            } else {
                tensor_shape
            };
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

pub(crate) fn flatten_cached_inputs(cache: &TensorStore, names: &[String]) -> Result<Vec<f64>> {
    let arrays: Vec<&ArrayD<f64>> = names.iter().map(|n| cache.get(n)).collect::<Result<_>>()?;
    let total: usize = arrays.iter().map(|a| a.len()).sum();
    let mut flat = Vec::with_capacity(total);
    for arr in arrays {
        flat.extend(arr.iter());
    }
    Ok(flat)
}

pub(crate) fn generate_wai_witness(
    backend: &JstproveBackend,
    circuit_path: &Path,
    slice_onnx_path: &Path,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
    params: &CircuitParams,
    flat_activations: &[f64],
) -> Result<Vec<u8>> {
    let initializers = if let Some(donor) = donor_init_map {
        let slice_model = crate::slicer::onnx_proto::load_model(slice_onnx_path)?;
        let slice_graph = slice_model
            .graph
            .as_ref()
            .ok_or_else(|| DsperseError::Pipeline("slice ONNX missing graph".into()))?;
        let mut merged = crate::slicer::onnx_proto::build_initializer_map(slice_graph);
        for (k, v) in donor.iter() {
            merged.insert(k.clone(), *v);
        }
        extract_initializers_from_map(&merged, params)?
    } else {
        extract_onnx_initializers(slice_onnx_path, params)?
    };
    backend.witness_f64(circuit_path, flat_activations, &initializers)
}

#[cfg(test)]
mod tests {
    use super::super::tiled::{reconstruct_from_tiles, reshape_to_4d, split_into_tiles};
    use super::*;
    use crate::schema::tiling::TilingInfo;
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
            segment_size: None,
            total_elements: None,
            original_shape: vec![],
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
    fn extract_initializers_resolves_template_renamed_weight() {
        // DimSplit / ChannelSplit template circuits rename the weight input to
        // "W" while the slice ONNX keeps the original initializer name
        // (e.g. "onnx::MatMul_4162"). The by-name lookup misses it, so without
        // the element-count fallback the weight is miscounted as an activation
        // and witness generation fails with an activation-length mismatch.
        use crate::slicer::onnx_proto::TensorProto;
        let params: CircuitParams = serde_json::from_value(serde_json::json!({
            "scale_base": 2, "scale_exponent": 18, "rescale_config": {},
            "inputs": [
                {"name": "dim_tmpl_in", "elem_type": 1, "shape": [1, 4]},
                {"name": "W", "elem_type": 1, "shape": [4, 4]}
            ],
            "outputs": [{"name": "dim_tmpl_out", "elem_type": 1, "shape": [1, 4]}],
            "weights_as_inputs": true
        }))
        .unwrap();
        let weight = TensorProto {
            name: "onnx::MatMul_4162".to_string(),
            data_type: TensorProto::FLOAT,
            dims: vec![4, 4],
            float_data: (0..16).map(|i| i as f32).collect(),
            ..Default::default()
        };
        let mut map: HashMap<String, &TensorProto> = HashMap::new();
        map.insert(weight.name.clone(), &weight);

        let inits = extract_initializers_from_map(&map, &params).unwrap();
        // Only W resolves; the activation (dim_tmpl_in, 4 elems) has no
        // matching initializer and stays an activation.
        assert_eq!(
            inits.len(),
            1,
            "renamed weight W must resolve by element count"
        );
        assert_eq!(inits[0].0.len(), 16);
    }

    #[test]
    fn extract_initializers_ambiguous_size_left_unresolved() {
        // If more than one unconsumed initializer matches the input's element
        // count the match is ambiguous and must be skipped so a wrong weight is
        // never silently bound into the witness.
        use crate::slicer::onnx_proto::TensorProto;
        let params: CircuitParams = serde_json::from_value(serde_json::json!({
            "scale_base": 2, "scale_exponent": 18, "rescale_config": {},
            "inputs": [{"name": "W", "elem_type": 1, "shape": [2, 2]}],
            "outputs": [{"name": "o", "elem_type": 1, "shape": [1]}],
            "weights_as_inputs": true
        }))
        .unwrap();
        let a = TensorProto {
            name: "a".to_string(),
            data_type: TensorProto::FLOAT,
            dims: vec![2, 2],
            float_data: vec![1.0; 4],
            ..Default::default()
        };
        let b = TensorProto {
            name: "b".to_string(),
            data_type: TensorProto::FLOAT,
            dims: vec![4],
            float_data: vec![2.0; 4],
            ..Default::default()
        };
        let mut map: HashMap<String, &TensorProto> = HashMap::new();
        map.insert("a".to_string(), &a);
        map.insert("b".to_string(), &b);

        let inits = extract_initializers_from_map(&map, &params).unwrap();
        assert!(
            inits.is_empty(),
            "ambiguous element-count match must not bind a guessed weight"
        );
    }

    #[test]
    fn extract_initializers_never_binds_activation_placeholder() {
        // A tiled activation placeholder (tile_in) must never be element-count
        // matched to an initializer, even when exactly one same-size initializer
        // remains after the named weight is consumed (e.g. a sibling layer's bias
        // present in a full-model ONNX). Regression for the slice_399/442
        // expected_activation=0 over-match.
        use crate::slicer::onnx_proto::TensorProto;
        let params: CircuitParams = serde_json::from_value(serde_json::json!({
            "scale_base": 2, "scale_exponent": 18, "rescale_config": {},
            "inputs": [
                {"name": "tile_in", "elem_type": 1, "shape": [2048]},
                {"name": "layers.0.linear1.bias", "elem_type": 1, "shape": [2048]}
            ],
            "outputs": [{"name": "tile_out", "elem_type": 1, "shape": [2048]}],
            "weights_as_inputs": true
        }))
        .unwrap();
        let bias0 = TensorProto {
            name: "layers.0.linear1.bias".to_string(),
            data_type: TensorProto::FLOAT,
            dims: vec![2048],
            float_data: vec![1.0; 2048],
            ..Default::default()
        };
        let bias1 = TensorProto {
            name: "layers.1.linear1.bias".to_string(),
            data_type: TensorProto::FLOAT,
            dims: vec![2048],
            float_data: vec![2.0; 2048],
            ..Default::default()
        };
        let mut map: HashMap<String, &TensorProto> = HashMap::new();
        map.insert(bias0.name.clone(), &bias0);
        map.insert(bias1.name.clone(), &bias1);

        let inits = extract_initializers_from_map(&map, &params).unwrap();
        // Only the named bias resolves (1). tile_in stays an activation; it must
        // not be bound to the spare same-size bias1, which would force a 0 count.
        assert_eq!(
            inits.len(),
            1,
            "activation placeholder must not be bound to a spare same-size initializer"
        );
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
    fn store_named_outputs_missing_name_errors() {
        let mut cache = TensorStore::new();
        let names = vec!["missing".to_string()];
        let named = HashMap::new();
        let result = store_named_outputs(&mut cache, &names, named);
        assert!(result.is_err());
    }

    #[test]
    fn store_named_outputs_partial_write_errors() {
        let mut cache = TensorStore::new();
        cache.put(
            "pre_existing".into(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![99.0]).unwrap(),
        );
        let names = vec!["present".to_string(), "missing".to_string()];
        let mut named = HashMap::new();
        named.insert("present".to_string(), (vec![1.0, 2.0], vec![2]));
        let result = store_named_outputs(&mut cache, &names, named);
        assert!(result.is_err());
        assert!(cache.contains("pre_existing"));
        assert!(!cache.contains("present"));
    }

    #[test]
    fn run_config_default() {
        let config = RunConfig::default();
        assert_eq!(config.parallel, 1);
        assert!(!config.batch);
        assert!(config.weights_onnx.is_none());
        assert!(config.combined);
    }

    #[test]
    fn multi_input_activation_concatenation_ordering() {
        use ndarray::IxDyn;
        let mut cache = TensorStore::new();
        cache.put(
            "act_a".into(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        cache.put(
            "act_b".into(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![7.0, 8.0]).unwrap(),
        );
        cache.put(
            "act_c".into(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![9.0]).unwrap(),
        );

        let inputs = vec![
            "act_a".to_string(),
            "act_b".to_string(),
            "act_c".to_string(),
        ];
        let mut flat: Vec<f64> = Vec::new();
        for name in &inputs {
            let arr = cache.get(name).unwrap();
            flat.extend(arr.iter());
        }

        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn multi_input_activation_missing_tensor_error() {
        let mut cache = TensorStore::new();
        cache.put(
            "act_a".into(),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        );

        let inputs = vec!["act_a".to_string(), "act_missing".to_string()];
        let mut flat: Vec<f64> = Vec::new();
        let mut err = None;
        for name in &inputs {
            match cache.get(name) {
                Ok(arr) => flat.extend(arr.iter()),
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }

        assert!(err.is_some());
        assert!(err.unwrap().to_string().contains("act_missing"));
    }
}
