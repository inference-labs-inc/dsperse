use crate::error::{DsperseError, Result};
use crate::schema::execution::ExecutionMethod;
use crate::schema::metadata::RunSliceMetadata;
use crate::schema::tiling::{ChannelSplitInfo, DimSplitInfo, SplitStrategy, TilingInfo};

pub enum ExecutionStrategy<'a> {
    ChannelSplit(&'a ChannelSplitInfo),
    DimSplit(&'a DimSplitInfo),
    Tiled(&'a TilingInfo),
    Single { use_circuit: bool },
}

impl<'a> ExecutionStrategy<'a> {
    pub fn from_metadata(meta: &'a RunSliceMetadata, use_circuit: bool) -> Result<Self> {
        let has_cs = meta.channel_split.is_some();
        let has_ds = meta.dim_split.is_some();
        let has_tiling = meta.tiling.is_some();
        let count = has_cs as u8 + has_ds as u8 + has_tiling as u8;
        if count > 1 {
            return Err(DsperseError::Metadata(format!(
                "slice has multiple split metadata (channel_split={has_cs}, \
                 dim_split={has_ds}, tiling={has_tiling}; path={:?})",
                meta.path
            )));
        }
        let multi_input = meta.dependencies.filtered_inputs.len() > 1;
        match meta.split_strategy() {
            Some(SplitStrategy::ChannelSplit(cs)) => {
                if multi_input {
                    tracing::debug!(
                        path = ?meta.path,
                        inputs = meta.dependencies.filtered_inputs.len(),
                        "channel_split slice consumes multiple activation inputs, falling back to single execution"
                    );
                    Ok(Self::Single { use_circuit })
                } else {
                    Ok(Self::ChannelSplit(cs))
                }
            }
            Some(SplitStrategy::DimSplit(ds)) => {
                if multi_input {
                    tracing::debug!(
                        path = ?meta.path,
                        split_kind = ?ds.split_kind,
                        inputs = meta.dependencies.filtered_inputs.len(),
                        "dim_split slice consumes multiple activation inputs, falling back to single execution"
                    );
                    Ok(Self::Single { use_circuit })
                } else if ds.template_path.is_none() {
                    // Template creation may have been rejected (axis-
                    // separability, unsupported split kind) or the template
                    // was not included in the bundle. Fall back to the
                    // non-template Single execution path (which may still
                    // use circuit-based witness generation if use_circuit is
                    // set) so already-published bundles with template-less
                    // dim_split metadata remain runnable.
                    tracing::debug!(
                        path = ?meta.path,
                        split_kind = ?ds.split_kind,
                        "dim_split template_path missing, falling back to single execution"
                    );
                    Ok(Self::Single { use_circuit })
                } else {
                    Ok(Self::DimSplit(ds))
                }
            }
            Some(SplitStrategy::Tiled(t)) => Ok(Self::Tiled(t)),
            None => Ok(Self::Single { use_circuit }),
        }
    }

    pub fn execution_method(&self) -> ExecutionMethod {
        match self {
            Self::ChannelSplit(_) => ExecutionMethod::ChannelSplit,
            Self::DimSplit(_) => ExecutionMethod::DimSplit,
            Self::Tiled(_) => ExecutionMethod::Tiled,
            Self::Single { use_circuit: true } => ExecutionMethod::JstproveGenWitness,
            Self::Single { use_circuit: false } => ExecutionMethod::OnnxOnly,
        }
    }

    pub fn output_name(&self) -> Option<&str> {
        match self {
            Self::ChannelSplit(cs) => Some(&cs.output_name),
            Self::DimSplit(ds) => Some(&ds.output_name),
            Self::Tiled(tiling) => Some(&tiling.output_name),
            Self::Single { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::metadata::{Dependencies, RunSliceMetadata};
    use crate::schema::tiling::{DimSplitInfo, DimSplitKind};

    fn dim_split_meta(inputs: Vec<String>) -> RunSliceMetadata {
        RunSliceMetadata {
            dependencies: Dependencies {
                input: inputs.clone(),
                output: vec!["out".to_string()],
                filtered_inputs: inputs,
            },
            dim_split: Some(DimSplitInfo {
                split_kind: DimSplitKind::BatchDim,
                input_name: "a".to_string(),
                output_name: "out".to_string(),
                template_path: Some("t.onnx".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn multi_input_dim_split_falls_back_to_single() {
        let meta = dim_split_meta(vec!["a".to_string(), "b".to_string()]);
        let strategy = ExecutionStrategy::from_metadata(&meta, true).unwrap();
        assert!(matches!(strategy, ExecutionStrategy::Single { .. }));
    }

    #[test]
    fn single_input_dim_split_keeps_dim_split_strategy() {
        let meta = dim_split_meta(vec!["a".to_string()]);
        let strategy = ExecutionStrategy::from_metadata(&meta, true).unwrap();
        assert!(matches!(strategy, ExecutionStrategy::DimSplit(_)));
    }
}
