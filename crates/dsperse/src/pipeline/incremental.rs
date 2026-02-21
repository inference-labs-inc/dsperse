use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::ArrayD;

use crate::error::{DsperseError, Result};
use crate::schema::execution::{
    ExecutionChain, ExecutionInfo, ExecutionResultEntry, RunMetadata,
};
use crate::schema::metadata::{ModelMetadata, RunSliceMetadata};
use crate::schema::tiling::{ChannelSplitInfo, TilingInfo};
use crate::utils::paths::find_metadata_path;

use super::runner::{build_execution_chain, build_run_metadata};

pub struct SliceWork {
    pub slice_id: String,
    pub input: ArrayD<f64>,
    pub backend: String,
    pub use_circuit: bool,
    pub tiling: Option<TilingInfo>,
    pub channel_split: Option<ChannelSplitInfo>,
    pub circuit_path: Option<String>,
    pub onnx_path: Option<String>,
    pub settings_path: Option<String>,
    pub slice_meta: RunSliceMetadata,
}

pub struct SliceExecutionResult {
    pub slice_id: String,
    pub output: ArrayD<f64>,
    pub execution_info: ExecutionInfo,
}

pub struct IncrementalRun {
    tensor_cache: HashMap<String, ArrayD<f64>>,
    execution_chain: ExecutionChain,
    model_meta: ModelMetadata,
    run_meta: RunMetadata,
    slices_dir: PathBuf,
    current_slice: Option<String>,
    results: Vec<ExecutionResultEntry>,
}

impl IncrementalRun {
    pub fn new(slices_dir: &Path, input: ArrayD<f64>) -> Result<Self> {
        let meta_path = find_metadata_path(slices_dir)
            .ok_or_else(|| DsperseError::Metadata("no metadata.json in slices".into()))?;
        let model_meta = ModelMetadata::load(&meta_path)?;

        let chain = build_execution_chain(&model_meta, slices_dir);
        let run_meta = build_run_metadata(&model_meta, slices_dir, Path::new(""), &chain);

        let mut tensor_cache = HashMap::new();
        if let Some(first_slice) = model_meta.slices.first() {
            if let Some(name) = first_slice.dependencies.input.first() {
                tensor_cache.insert(name.clone(), input);
            }
        }

        let current_slice = chain.head.clone();

        Ok(Self {
            tensor_cache,
            execution_chain: chain,
            model_meta,
            run_meta,
            slices_dir: slices_dir.to_path_buf(),
            current_slice,
            results: Vec::new(),
        })
    }

    pub fn next_slice(&self) -> Option<SliceWork> {
        let slice_id = self.current_slice.as_ref()?;
        let node = self.execution_chain.nodes.get(slice_id)?;
        let meta = self.run_meta.slices.get(slice_id)?;

        let input = if let Some(ref cs) = meta.channel_split {
            self.tensor_cache.get(&cs.input_name)?.clone()
        } else if let Some(ref tiling) = meta.tiling {
            self.tensor_cache.get(&tiling.input_name)?.clone()
        } else {
            self.gather_inputs(&meta.dependencies.filtered_inputs).ok()?
        };

        Some(SliceWork {
            slice_id: slice_id.clone(),
            input,
            backend: node.backend.clone(),
            use_circuit: node.use_circuit,
            tiling: meta.tiling.clone(),
            channel_split: meta.channel_split.clone(),
            circuit_path: node.circuit_path.clone(),
            onnx_path: node.onnx_path.clone(),
            settings_path: meta.settings_path.clone(),
            slice_meta: meta.clone(),
        })
    }

    pub fn apply_result(&mut self, result: SliceExecutionResult) -> Result<()> {
        let slice_id = &result.slice_id;

        let meta = self
            .run_meta
            .slices
            .get(slice_id)
            .ok_or_else(|| DsperseError::Pipeline(format!("unknown slice {slice_id}")))?;

        if let Some(ref cs) = meta.channel_split {
            self.tensor_cache
                .insert(cs.output_name.clone(), result.output);
        } else if let Some(ref tiling) = meta.tiling {
            self.tensor_cache
                .insert(tiling.output_name.clone(), result.output);
        } else {
            for name in &meta.dependencies.output {
                self.tensor_cache
                    .insert(name.clone(), result.output.clone());
            }
        }

        self.results.push(ExecutionResultEntry {
            slice_id: slice_id.clone(),
            witness_execution: Some(result.execution_info),
            proof_execution: None,
            verification_execution: None,
        });

        let next = self
            .execution_chain
            .nodes
            .get(slice_id)
            .and_then(|n| n.next.clone());
        self.current_slice = next;

        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        self.current_slice.is_none()
    }

    pub fn final_output(&self) -> Option<&ArrayD<f64>> {
        let last_slice = self.model_meta.slices.last()?;
        let output_name = last_slice.dependencies.output.first()?;
        self.tensor_cache.get(output_name)
    }

    pub fn into_run_metadata(self) -> RunMetadata {
        let mut meta = self.run_meta;
        meta.execution_chain.execution_results = self.results;
        meta.source_path = Some(self.slices_dir.to_string_lossy().into_owned());
        meta
    }

    pub fn slices_dir(&self) -> &Path {
        &self.slices_dir
    }

    pub fn model_meta(&self) -> &ModelMetadata {
        &self.model_meta
    }

    pub fn run_meta(&self) -> &RunMetadata {
        &self.run_meta
    }

    pub fn tensor_cache(&self) -> &HashMap<String, ArrayD<f64>> {
        &self.tensor_cache
    }

    fn gather_inputs(&self, inputs: &[String]) -> Result<ArrayD<f64>> {
        let mut collected = Vec::new();
        let mut missing = Vec::new();
        for name in inputs {
            if let Some(val) = self.tensor_cache.get(name) {
                collected.push(val.clone());
            } else {
                missing.push(name.clone());
            }
        }
        if collected.is_empty() {
            return Err(DsperseError::Pipeline(format!(
                "no cached tensor found for inputs: {inputs:?}"
            )));
        }
        if !missing.is_empty() {
            return Err(DsperseError::Pipeline(format!(
                "missing tensors in cache: {missing:?} (found {} of {})",
                collected.len(),
                inputs.len()
            )));
        }
        if collected.len() == 1 {
            return Ok(collected.into_iter().next().unwrap());
        }
        ndarray::concatenate(
            ndarray::Axis(0),
            &collected.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| DsperseError::Pipeline(format!("concat inputs: {e}")))
    }
}
