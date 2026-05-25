//! Per-slice activation snapshot writer.
//!
//! After a combined inference run the `TensorStore` holds every slice's
//! output tensor keyed by ONNX tensor name. This module serializes each
//! slice's output tensor to disk under `<dir>/slice_<index>.bin` as a
//! self-describing fp16 + zstd payload so downstream consumers can render
//! activation maps for every slice including non-circuit ones.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use half::f16;
use ndarray::ArrayD;
use zstd::stream::write::Encoder;

use crate::error::{DsperseError, Result};
use crate::pipeline::tensor_store::TensorStore;
use crate::schema::metadata::ModelMetadata;

/// File magic for `slice_<N>.bin` activation snapshots: "DAST" (Dsperse
/// Activation Snapshot Tensor). Little-endian u32 = 0x54534144.
pub const ACTIVATION_MAGIC: [u8; 4] = *b"DAST";
pub const ACTIVATION_VERSION: u8 = 1;
const DTYPE_FP16: u8 = 1;
const ZSTD_LEVEL: i32 = 3;
const MAX_RANK: usize = 8;

/// Writes one `slice_<index>.bin` per slice into `dir`. Slices whose
/// output tensor is not present in the cache are skipped (logged at debug).
pub fn write_slice_activations(
    dir: &Path,
    model_meta: &ModelMetadata,
    cache: &TensorStore,
) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| DsperseError::io(e, dir))?;

    let mut written = 0usize;
    let mut skipped = 0usize;

    for slice in &model_meta.slices {
        let candidate_names: Vec<&String> = slice.dependencies.output.iter().collect();
        let tensor = candidate_names
            .iter()
            .find_map(|name| cache.try_get(name.as_str()));

        let Some(tensor) = tensor else {
            tracing::debug!(
                slice_index = slice.index,
                "slice activation snapshot skipped (no cached output tensor)"
            );
            skipped += 1;
            continue;
        };

        let target = dir.join(format!("slice_{}.bin", slice.index));
        write_one(&target, tensor)?;
        written += 1;
    }

    tracing::info!(
        dir = %dir.display(),
        written,
        skipped,
        total = model_meta.slices.len(),
        "wrote per-slice activation snapshots"
    );

    Ok(())
}

fn write_one(target: &Path, tensor: &ArrayD<f64>) -> Result<()> {
    let rank = tensor.ndim();
    if rank > MAX_RANK {
        return Err(DsperseError::Pipeline(format!(
            "activation tensor rank {rank} exceeds max {MAX_RANK}"
        )));
    }
    let total: usize = tensor.shape().iter().product();
    let total_u32 = u32::try_from(total).map_err(|_| {
        DsperseError::Pipeline(format!(
            "activation tensor too large for u32 element count: {total}"
        ))
    })?;
    let rank_u8 = u8::try_from(rank).map_err(|_| {
        DsperseError::Pipeline(format!("activation tensor rank {rank} exceeds u8::MAX"))
    })?;

    let file = File::create(target).map_err(|e| DsperseError::io(e, target))?;
    let mut enc = Encoder::new(file, ZSTD_LEVEL).map_err(|e| DsperseError::io(e, target))?;

    // Header (12 + rank*4 bytes), all little-endian.
    enc.write_all(&ACTIVATION_MAGIC)
        .map_err(|e| DsperseError::io(e, target))?;
    enc.write_all(&[ACTIVATION_VERSION, DTYPE_FP16, rank_u8, 0])
        .map_err(|e| DsperseError::io(e, target))?;
    enc.write_all(&total_u32.to_le_bytes())
        .map_err(|e| DsperseError::io(e, target))?;
    for dim in tensor.shape() {
        let d = u32::try_from(*dim).map_err(|_| {
            DsperseError::Pipeline(format!(
                "activation tensor dimension {dim} exceeds u32::MAX in {}",
                target.display()
            ))
        })?;
        enc.write_all(&d.to_le_bytes())
            .map_err(|e| DsperseError::io(e, target))?;
    }

    // Body: contiguous fp16 values in row-major order.
    let contiguous = tensor.as_standard_layout();
    let slice = contiguous
        .as_slice()
        .ok_or_else(|| DsperseError::Pipeline("activation tensor not contiguous".into()))?;
    let mut chunk = [0u8; 8192];
    let mut buf_i = 0usize;
    for &v in slice {
        let h = f16::from_f64(v).to_le_bytes();
        chunk[buf_i] = h[0];
        chunk[buf_i + 1] = h[1];
        buf_i += 2;
        if buf_i == chunk.len() {
            enc.write_all(&chunk)
                .map_err(|e| DsperseError::io(e, target))?;
            buf_i = 0;
        }
    }
    if buf_i > 0 {
        enc.write_all(&chunk[..buf_i])
            .map_err(|e| DsperseError::io(e, target))?;
    }

    enc.finish().map_err(|e| DsperseError::io(e, target))?;
    Ok(())
}
