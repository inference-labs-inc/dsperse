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

        // Seed the tensor_cache with any initializer-backed tensor
        // the slice metadata references.  The slicer's constant-
        // folding passes can turn intermediate tensors (e.g. a
        // Transpose over a constant) into initializers in the
        // transformed graph, while leaving downstream slice
        // metadata pointing at the original tensor name.  ORT
        // does not emit those names among its named outputs (they
        // are not declared as graph outputs of combined.onnx and
        // have no producing node), so without this seed the
        // subsequent `tensor_cache.get` in `all_circuit_work` fails
        // with `tensor '<name>' not found in store` and the whole
        // run aborts before a single DSlice gets dispatched.
        seed_tensor_cache_from_initializers(&combined_path, &model_meta, &mut tensor_cache)?;

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

    pub fn circuit_work_ids(&self) -> Vec<String> {
        self.model_meta
            .slices
            .iter()
            .map(|slice| format!("slice_{}", slice.index))
            .filter(|slice_id| self.pending_slices.contains(slice_id))
            .collect()
    }

    pub fn circuit_work_for(&self, slice_id: &str) -> Result<SliceWork> {
        let node = self.execution_chain.nodes.get(slice_id).ok_or_else(|| {
            DsperseError::Pipeline(format!("execution chain missing node for {slice_id}"))
        })?;

        let meta = self.run_meta.slices.get(slice_id).ok_or_else(|| {
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
                let names = tiling.all_input_names();
                if names.len() > 1 {
                    let mut named: Vec<(String, ArrayD<f64>)> = Vec::with_capacity(names.len());
                    let mut flat: Vec<f64> = Vec::new();
                    for name in &names {
                        let arr = self.tensor_cache.get(name)?;
                        named.push(((*name).to_string(), arr.clone()));
                        flat.extend(arr.iter());
                    }
                    let concatenated =
                        ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).map_err(|e| {
                            DsperseError::Pipeline(format!("tiled multi-input concat: {e}"))
                        })?;
                    (concatenated, named)
                } else {
                    let t = self.tensor_cache.get(&tiling.input_name)?.clone();
                    (t, Vec::new())
                }
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

        Ok(SliceWork {
            slice_id: slice_id.to_string(),
            input,
            named_inputs,
            backend: node.backend,
            use_circuit: node.use_circuit,
            tiling: matches!(strategy, ExecutionStrategy::Tiled(_))
                .then(|| meta.tiling.clone())
                .flatten(),
            channel_split: matches!(strategy, ExecutionStrategy::ChannelSplit(_))
                .then(|| meta.channel_split.clone())
                .flatten(),
            dim_split: matches!(strategy, ExecutionStrategy::DimSplit(_))
                .then(|| meta.dim_split.clone())
                .flatten(),
            circuit_path: node
                .circuit_path
                .as_deref()
                .map(|p| self.absolute_work_path(p))
                .transpose()?,
            onnx_path: node
                .onnx_path
                .as_deref()
                .map(|p| self.absolute_work_path(p))
                .transpose()?,
            slice_meta: meta.clone(),
        })
    }

    fn absolute_work_path(&self, path: &str) -> Result<String> {
        if Path::new(path).is_absolute() {
            return Ok(path.to_string());
        }
        crate::utils::paths::resolve_relative_path(&self.slices_dir, path)
            .map(|p| p.to_string_lossy().into_owned())
    }

    pub fn all_circuit_work(&self) -> Result<Vec<SliceWork>> {
        self.circuit_work_ids()
            .iter()
            .map(|slice_id| self.circuit_work_for(slice_id))
            .collect()
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

    pub fn output_arrays_for_names(&self, names: &[String]) -> Option<Vec<ArrayD<f64>>> {
        let mut arrays = Vec::with_capacity(names.len());
        for name in names {
            arrays.push(self.tensor_cache.try_get(name)?.clone());
        }
        if arrays.is_empty() {
            None
        } else {
            Some(arrays)
        }
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

/// Populate `tensor_cache` with any combined-graph initializer
/// whose name appears in slice metadata as a `filtered_input` or a
/// declared `output`.  Without this, a slice that depends on a
/// constant-folded tensor (one the slicer turned from a node
/// output into an initializer) would fail at the
/// `tensor_cache.get(name)` call in `all_circuit_work` even though
/// the value is right there in the combined ONNX.
fn seed_tensor_cache_from_initializers(
    combined_path: &Path,
    model_meta: &ModelMetadata,
    tensor_cache: &mut TensorStore,
) -> Result<()> {
    let needed: HashSet<&str> = model_meta
        .slices
        .iter()
        .flat_map(|s| {
            s.dependencies
                .filtered_inputs
                .iter()
                .chain(s.dependencies.output.iter())
        })
        .map(String::as_str)
        .collect();
    if needed.is_empty() {
        return Ok(());
    }

    let model = crate::slicer::onnx_proto::load_model(combined_path)?;
    let graph = match &model.graph {
        Some(g) => g,
        None => return Ok(()),
    };

    let mut seeded = 0usize;
    for init in &graph.initializer {
        if !needed.contains(init.name.as_str()) {
            continue;
        }
        if tensor_cache.contains(&init.name) {
            continue;
        }
        // Negative dims would silently wrap to huge positive
        // values via `as usize`; reject up front so a malformed
        // initialiser surfaces an error here instead of
        // allocating a multi-petabyte array below.
        let shape: Vec<usize> = match init
            .dims
            .iter()
            .map(|&d| usize::try_from(d))
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!(
                    name = %init.name,
                    dims = ?init.dims,
                    error = %e,
                    "skipping initializer-backed slice tensor: invalid (negative) dimension"
                );
                continue;
            }
        };
        // Use checked_mul so an arithmetic overflow surfaces as a
        // skip (and the slice executor downstream produces a
        // clearer error if it actually needed the value), instead
        // of wrapping silently and mis-comparing against
        // `data.len()`.
        let expected: Option<usize> = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        let Some(expected) = expected else {
            tracing::debug!(
                name = %init.name,
                dims = ?init.dims,
                "skipping initializer-backed slice tensor: shape product overflowed usize"
            );
            continue;
        };
        // Decode straight to f64 so DOUBLE / INT64 initialisers
        // keep their full precision -- the previous f32-then-widen
        // chain truncated DOUBLE mantissas and silently lost
        // precision on INT64 magnitudes outside f32's exact range.
        let data: Vec<f64> = crate::slicer::onnx_proto::tensor_to_f64(init);
        if data.len() != expected {
            // Skip rather than fail: an initialiser whose declared
            // shape doesn't match its element count can still be
            // useful elsewhere (some quantised tensors store packed
            // bytes), but we cannot reshape it into ArrayD<f64>
            // here without guessing.  Leave it to the slice ONNX
            // executor to surface a clearer error if it actually
            // needs the value.
            tracing::debug!(
                name = %init.name,
                declared_shape = ?shape,
                declared_elements = expected,
                actual_elements = data.len(),
                "skipping initializer-backed slice tensor: declared shape != element count"
            );
            continue;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
            DsperseError::Pipeline(format!(
                "seed initializer-backed tensor '{}' from combined.onnx: {e}",
                init.name
            ))
        })?;
        tensor_cache.put(init.name.clone(), arr);
        seeded += 1;
    }
    if seeded > 0 {
        tracing::info!(
            seeded,
            "seeded tensor_cache with constant-folded slice-input initializers"
        );
    }
    Ok(())
}
