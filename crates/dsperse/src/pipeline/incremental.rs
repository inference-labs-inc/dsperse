use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::ArrayD;

use crate::error::{DsperseError, Result};
use crate::schema::execution::{
    ExecutionChain, ExecutionInfo, ExecutionResultEntry, RunMetadata,
};
use crate::schema::metadata::{ModelMetadata, RunSliceMetadata};
use crate::schema::tiling::{ChannelSplitInfo, TilingInfo};
use crate::utils::io::gather_inputs_from_cache;
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
        let input_name = model_meta
            .slices
            .first()
            .and_then(|s| s.dependencies.input.first().cloned())
            .ok_or_else(|| {
                DsperseError::Pipeline(
                    "model has no slices or first slice has no input dependency".into(),
                )
            })?;
        tensor_cache.insert(input_name, input);

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

    pub fn next_slice(&self) -> Result<Option<SliceWork>> {
        let slice_id = match self.current_slice.as_ref() {
            Some(id) => id,
            None => return Ok(None),
        };
        let node = self.execution_chain.nodes.get(slice_id).ok_or_else(|| {
            DsperseError::Pipeline(format!("execution chain missing node for {slice_id}"))
        })?;
        let meta = self.run_meta.slices.get(slice_id).ok_or_else(|| {
            DsperseError::Pipeline(format!("run metadata missing slice {slice_id}"))
        })?;

        let input = if let Some(ref cs) = meta.channel_split {
            self.tensor_cache.get(&cs.input_name).ok_or_else(|| {
                DsperseError::Pipeline(format!(
                    "channel split input '{}' not in cache for {slice_id}",
                    cs.input_name
                ))
            })?.clone()
        } else if let Some(ref tiling) = meta.tiling {
            self.tensor_cache.get(&tiling.input_name).ok_or_else(|| {
                DsperseError::Pipeline(format!(
                    "tiling input '{}' not in cache for {slice_id}",
                    tiling.input_name
                ))
            })?.clone()
        } else {
            gather_inputs_from_cache(&self.tensor_cache, &meta.dependencies.filtered_inputs)?
        };

        Ok(Some(SliceWork {
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
        }))
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
}
