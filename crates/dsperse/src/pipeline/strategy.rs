use crate::error::{DsperseError, Result};
use crate::schema::execution::ExecutionMethod;
use crate::schema::metadata::RunSliceMetadata;
use crate::schema::tiling::{ChannelSplitInfo, DimSplitInfo, TilingInfo};

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
                "slice has multiple split metadata (channel_split={has_cs}, dim_split={has_ds}, tiling={has_tiling})"
            )));
        }
        if let Some(ref cs) = meta.channel_split {
            Ok(Self::ChannelSplit(cs))
        } else if let Some(ref ds) = meta.dim_split
            && ds.template_path.is_some()
        {
            Ok(Self::DimSplit(ds))
        } else if let Some(ref tiling) = meta.tiling {
            Ok(Self::Tiled(tiling))
        } else {
            Ok(Self::Single { use_circuit })
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
