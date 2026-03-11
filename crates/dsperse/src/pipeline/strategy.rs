use crate::schema::execution::ExecutionMethod;
use crate::schema::metadata::RunSliceMetadata;
use crate::schema::tiling::{ChannelSplitInfo, TilingInfo};

pub enum ExecutionStrategy<'a> {
    ChannelSplit(&'a ChannelSplitInfo),
    Tiled(&'a TilingInfo),
    Single { use_circuit: bool },
}

impl<'a> ExecutionStrategy<'a> {
    pub fn from_metadata(meta: &'a RunSliceMetadata, use_circuit: bool) -> Self {
        if let Some(ref cs) = meta.channel_split {
            Self::ChannelSplit(cs)
        } else if let Some(ref tiling) = meta.tiling {
            Self::Tiled(tiling)
        } else {
            Self::Single { use_circuit }
        }
    }

    pub fn execution_method(&self) -> ExecutionMethod {
        match self {
            Self::ChannelSplit(_) => ExecutionMethod::ChannelSplit,
            Self::Tiled(_) => ExecutionMethod::Tiled,
            Self::Single { use_circuit: true } => ExecutionMethod::JstproveGenWitness,
            Self::Single { use_circuit: false } => ExecutionMethod::OnnxOnly,
        }
    }

    pub fn output_name(&self) -> Option<&str> {
        match self {
            Self::ChannelSplit(cs) => Some(&cs.output_name),
            Self::Tiled(tiling) => Some(&tiling.output_name),
            Self::Single { .. } => None,
        }
    }
}
