use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};

use super::incremental::SliceWork;
use super::runner::{build_execution_chain, build_run_metadata, load_model_metadata};
use super::strategy::ExecutionStrategy;
use super::tensor_store::TensorStore;
use crate::backend::onnx::NamedOutputs;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{ExecutionChain, RunMetadata};
use crate::schema::metadata::ModelMetadata;

pub struct CombinedRun {
    tensor_cache: TensorStore,
    model_meta: ModelMetadata,
    run_meta: RunMetadata,
    execution_chain: ExecutionChain,
    slices_dir: PathBuf,
    pending_slices: HashSet<String>,
    failed_slices: HashSet<String>,
}

impl CombinedRun {
    pub fn new(slices_dir: &Path, input: ArrayD<f64>) -> Result<Self> {
        let model_meta = load_model_metadata(slices_dir)?;

        let combined_path =
            crate::slicer::combiner::ensure_combined_materialized(slices_dir, &model_meta)?;

        crate::slicer::materializer::ensure_all_slices_materialized(slices_dir, &model_meta)?;

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

        let named_outputs = run_combined_onnx(&combined_path, &input, declared_inputs)?;

        let mut tensor_cache = TensorStore::new();
        for (name, (data, shape)) in &named_outputs {
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data.clone())
                .map_err(|e| DsperseError::Pipeline(format!("output reshape '{name}': {e}")))?;
            tensor_cache.put(name.clone(), arr);
        }
        for name in declared_inputs {
            if !tensor_cache.contains(name) {
                tensor_cache.put(name.clone(), input.clone());
            }
        }

        let chain = build_execution_chain(&model_meta, slices_dir)?;
        let run_meta = build_run_metadata(&model_meta, slices_dir, &chain)?;

        let mut pending_slices = HashSet::new();
        for slice in &model_meta.slices {
            let slice_id = format!("slice_{}", slice.index);
            let node = chain.nodes.get(&slice_id).ok_or_else(|| {
                DsperseError::Pipeline(format!("execution chain missing node for {slice_id}"))
            })?;
            if node.use_circuit {
                pending_slices.insert(slice_id);
            }
        }

        tracing::info!(
            total_slices = model_meta.slices.len(),
            circuit_slices = pending_slices.len(),
            cached_tensors = tensor_cache.len(),
            "combined inference complete, all circuit work queued"
        );

        Ok(Self {
            tensor_cache,
            model_meta,
            run_meta,
            execution_chain: chain,
            slices_dir: slices_dir.to_path_buf(),
            pending_slices,
            failed_slices: HashSet::new(),
        })
    }

    pub fn all_circuit_work(&self) -> Result<Vec<SliceWork>> {
        let mut work_items = Vec::with_capacity(self.pending_slices.len());

        for slice in &self.model_meta.slices {
            let slice_id = format!("slice_{}", slice.index);
            if !self.pending_slices.contains(&slice_id) {
                continue;
            }

            let node = self.execution_chain.nodes.get(&slice_id).ok_or_else(|| {
                DsperseError::Pipeline(format!("execution chain missing node for {slice_id}"))
            })?;

            let meta = self.run_meta.slices.get(&slice_id).ok_or_else(|| {
                DsperseError::Pipeline(format!("run metadata missing slice {slice_id}"))
            })?;

            let strategy = ExecutionStrategy::from_metadata(meta, node.use_circuit)?;
            let (input, named_inputs) = match strategy {
                ExecutionStrategy::ChannelSplit(cs) => {
                    let t = self.tensor_cache.get(&cs.input_name)?.clone();
                    (t, Vec::new())
                }
                ExecutionStrategy::DimSplit(ds) => {
                    let t = self.tensor_cache.get(&ds.input_name)?.clone();
                    (t, Vec::new())
                }
                ExecutionStrategy::Tiled(tiling) => {
                    let t = self.tensor_cache.get(&tiling.input_name)?.clone();
                    (t, Vec::new())
                }
                ExecutionStrategy::Single { .. } => {
                    let filtered = &meta.dependencies.filtered_inputs;
                    let mut named = Vec::with_capacity(filtered.len());
                    let mut flat_elems: Vec<f64> = Vec::new();
                    for name in filtered {
                        let arr = self.tensor_cache.get(name)?;
                        named.push((name.clone(), arr.clone()));
                        flat_elems.extend(arr.iter());
                    }
                    let concatenated = ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[flat_elems.len()]),
                        flat_elems,
                    )
                    .map_err(|e| DsperseError::Pipeline(format!("flatten inputs: {e}")))?;
                    (concatenated, named)
                }
            };

            work_items.push(SliceWork {
                slice_id,
                input,
                named_inputs,
                backend: node.backend,
                use_circuit: node.use_circuit,
                tiling: meta.tiling.clone(),
                channel_split: meta.channel_split.clone(),
                circuit_path: node.circuit_path.clone(),
                onnx_path: node.onnx_path.clone(),
                slice_meta: meta.clone(),
            });
        }

        Ok(work_items)
    }

    pub fn mark_slice_done(&mut self, slice_id: &str) -> bool {
        self.pending_slices.remove(slice_id)
    }

    pub fn mark_slice_failed(&mut self, slice_id: &str) -> bool {
        let was_pending = self.pending_slices.remove(slice_id);
        if was_pending {
            self.failed_slices.insert(slice_id.to_string());
        }
        was_pending
    }

    pub fn is_slice_failed(&self, slice_id: &str) -> bool {
        self.failed_slices.contains(slice_id)
    }

    pub fn failed_count(&self) -> usize {
        self.failed_slices.len()
    }

    pub fn is_complete(&self) -> bool {
        self.pending_slices.is_empty()
    }

    pub fn model_meta(&self) -> &ModelMetadata {
        &self.model_meta
    }

    pub fn final_output(&self) -> Option<&ArrayD<f64>> {
        let last_slice = self.model_meta.slices.last()?;
        let slice_id = format!("slice_{}", last_slice.index);
        let meta = self.run_meta.slices.get(&slice_id)?;

        let strategy = ExecutionStrategy::from_metadata(meta, false).ok()?;
        match strategy.output_name() {
            Some(name) => self.tensor_cache.try_get(name),
            None => {
                let output_name = meta.dependencies.output.first()?;
                self.tensor_cache.try_get(output_name)
            }
        }
    }

    pub fn expected_slice_outputs(&self, slice_id: &str) -> Option<Vec<f64>> {
        let meta = self.run_meta.slices.get(slice_id)?;
        let output_names = &meta.dependencies.output;
        self.outputs_for_names(output_names)
    }

    pub fn outputs_for_names(&self, names: &[String]) -> Option<Vec<f64>> {
        let mut flat = Vec::new();
        for name in names {
            let tensor = self.tensor_cache.try_get(name)?;
            flat.extend(tensor.iter());
        }
        if flat.is_empty() { None } else { Some(flat) }
    }

    pub fn slice_tile_counts(&self) -> (usize, usize, HashMap<String, usize>) {
        let total_slices = self.model_meta.slices.len();
        let mut map = HashMap::with_capacity(total_slices);
        let mut total_tiles = 0usize;
        for s in &self.model_meta.slices {
            let tiles = s.tiling.as_ref().map(|t| t.num_tiles).unwrap_or(1);
            map.insert(format!("slice_{}", s.index), tiles);
            total_tiles += tiles;
        }
        (total_slices, total_tiles, map)
    }

    pub fn slices_dir(&self) -> &Path {
        &self.slices_dir
    }

    pub fn pending_count(&self) -> usize {
        self.pending_slices.len()
    }
}

fn run_combined_onnx(
    combined_path: &Path,
    input: &ArrayD<f64>,
    declared_inputs: &[String],
) -> Result<NamedOutputs> {
    if declared_inputs.len() == 1 {
        let input_flat: Vec<f64> = input.iter().copied().collect();
        let input_shape = input.shape();
        crate::backend::onnx::run_inference_named(combined_path, &input_flat, input_shape)
    } else {
        Err(DsperseError::Pipeline(format!(
            "combined mode requires single input, got {}",
            declared_inputs.len()
        )))
    }
}
