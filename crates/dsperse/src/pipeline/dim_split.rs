use std::collections::HashMap;
use std::path::Path;

use super::runner::{run_onnx_inference, run_onnx_inference_multi_named};
use super::tensor_store::TensorStore;
use crate::backend::jstprove::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::ExecutionInfo;
use crate::schema::tiling::DimSplitKind;
use crate::slicer::onnx_proto::TensorProto;

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dim_split(
    slices_dir: &Path,
    _slice_run_dir: &Path,
    slice_id: &str,
    ds: &crate::schema::tiling::DimSplitInfo,
    target_shape: Option<&[i64]>,
    tensor_cache: &TensorStore,
    _backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<crate::schema::execution::StrategyOutput> {
    let tmpl_rel = ds.template_path.as_ref().ok_or_else(|| {
        DsperseError::Pipeline(format!("{slice_id}: dim_split has no template_path"))
    })?;
    let tmpl_path = slices_dir.join(tmpl_rel);
    if !tmpl_path.exists() {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: dim-split template not found: {}",
            tmpl_path.display()
        )));
    }

    let use_matmul_split = matches!(ds.split_kind, DimSplitKind::MatMulOutputDim);

    let final_result = if use_matmul_split {
        execute_matmul_dim_split(
            slices_dir,
            slice_id,
            ds,
            target_shape,
            tensor_cache,
            &tmpl_path,
            donor_init_map,
        )?
    } else {
        execute_generic_dim_split(slice_id, ds, target_shape, tensor_cache, &tmpl_path)?
    };

    Ok(crate::schema::execution::StrategyOutput {
        info: ExecutionInfo {
            method: crate::schema::execution::ExecutionMethod::DimSplit,
            success: true,
            error: None,
            witness_file: None,
            tile_exec_infos: Vec::new(),
        },
        outputs: vec![(ds.output_name.clone(), final_result)],
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_matmul_dim_split(
    slices_dir: &Path,
    slice_id: &str,
    ds: &crate::schema::tiling::DimSplitInfo,
    target_shape: Option<&[i64]>,
    tensor_cache: &TensorStore,
    tmpl_path: &Path,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ndarray::ArrayD<f64>> {
    let input_tensor = tensor_cache.get(&ds.input_name)?.clone();
    let input_shape = input_tensor.shape().to_vec();
    let k_dim = *input_shape.last().unwrap_or(&0);
    if ds.k_dim != 0 && k_dim != ds.k_dim {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: runtime k_dim {} from input {:?} does not match metadata k_dim {}",
            k_dim, ds.input_name, ds.k_dim
        )));
    }
    if k_dim == 0 {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: dim-split input {:?} has zero-width last dim; expected k_dim > 0",
            ds.input_name
        )));
    }
    let k_chunks = ds.k_chunks.max(1);
    let k_chunk_size = k_dim.div_ceil(k_chunks);

    let total_rows: usize = input_shape
        .iter()
        .take(input_shape.len().saturating_sub(1))
        .product();
    let flat_input = input_tensor
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(ndarray::IxDyn(&[total_rows, k_dim]))
        .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: flatten input: {e}")))?;

    let slice_onnx_path = slices_dir
        .join(format!("slice_{}", ds.slice_idx))
        .join("payload")
        .join(format!("slice_{}.onnx", ds.slice_idx));

    let orig_model = crate::slicer::onnx_proto::load_model(&slice_onnx_path)?;
    let orig_graph = orig_model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Pipeline(format!("{slice_id}: slice ONNX has no graph")))?;
    let weight_name = ds.weight_name.as_ref().ok_or_else(|| {
        DsperseError::Pipeline(format!(
            "{slice_id}: dim_split missing weight_name in metadata"
        ))
    })?;
    let matmul_node = orig_graph
        .node
        .iter()
        .find(|n| {
            matches!(n.op_type.as_str(), "MatMul" | "Gemm")
                && n.input.iter().any(|i| i == weight_name)
                && n.input.iter().any(|i| i == &ds.input_name)
                && n.output.iter().any(|o| o == &ds.output_name)
        })
        .ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "{slice_id}: no MatMul/Gemm node matches weight={weight_name:?} input={:?} output={:?}",
                ds.input_name, ds.output_name
            ))
        })?;
    let trans_b = matmul_node.op_type == "Gemm"
        && crate::slicer::onnx_proto::get_attribute_int(matmul_node, "transB").unwrap_or(0) == 1;
    let full_weight: Vec<f32> = if let Some(map) = donor_init_map
        && let Some(t) = map.get(weight_name.as_str())
    {
        crate::slicer::onnx_proto::tensor_to_f32(t)
    } else {
        let init = orig_graph
            .initializer
            .iter()
            .find(|i| i.name == *weight_name)
            .ok_or_else(|| {
                DsperseError::Pipeline(format!(
                    "{slice_id}: weight {weight_name:?} not found in slice ONNX initializers"
                ))
            })?;
        crate::slicer::onnx_proto::tensor_to_f32(init)
    };
    let expected_weight_len = ds.k_dim.saturating_mul(ds.n_dim);
    if expected_weight_len > 0 && full_weight.len() != expected_weight_len {
        return Err(DsperseError::Pipeline(format!(
            "{slice_id}: weight {weight_name:?} length {} does not match expected k_dim*n_dim = {}*{} = {}",
            full_weight.len(),
            ds.k_dim,
            ds.n_dim,
            expected_weight_len
        )));
    }

    let n_dim = ds.n_dim;
    let tmpl_model = crate::slicer::onnx_proto::load_model(tmpl_path)?;

    let tmp_dir = tempfile::tempdir()
        .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: tmpdir: {e}")))?;

    let mut patched_paths: Vec<std::path::PathBuf> = Vec::with_capacity(k_chunks);
    for kc in 0..k_chunks {
        let weight_chunk = dim_split_weight_chunk(&full_weight, ds, kc, trans_b);

        let mut patched = tmpl_model.clone();
        let graph = patched.graph.as_mut().ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "{slice_id}: dim-split template at {} has no graph",
                tmpl_path.display()
            ))
        })?;
        let w_init = graph
            .initializer
            .iter_mut()
            .find(|i| i.name == "W")
            .ok_or_else(|| {
                DsperseError::Pipeline(format!(
                    "{slice_id}: dim-split template at {} missing 'W' initializer",
                    tmpl_path.display()
                ))
            })?;
        w_init.float_data = weight_chunk;
        w_init.raw_data.clear();

        let patched_path = tmp_dir.path().join(format!("chunk_{kc}.onnx"));
        crate::slicer::onnx_proto::save_model(&patched, &patched_path)?;
        patched_paths.push(patched_path);
    }

    let mut row_outputs: Vec<ndarray::ArrayD<f64>> = Vec::with_capacity(total_rows);

    for r in 0..total_rows {
        let full_row: Vec<f64> = flat_input
            .slice(ndarray::s![r, ..])
            .iter()
            .copied()
            .collect();

        let mut row_accum = vec![0.0f64; n_dim];

        for (kc, patched_path) in patched_paths.iter().enumerate() {
            let input_chunk = dim_split_row_chunk(&full_row, kc, k_dim, k_chunk_size);

            let input_arr =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, k_chunk_size]), input_chunk)
                    .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: input chunk: {e}")))?;

            let out = run_onnx_inference(patched_path, &input_arr)?;
            if out.len() != n_dim {
                return Err(DsperseError::Pipeline(format!(
                    "{slice_id}: dim-split k-chunk {kc} produced {} outputs, expected n_dim={n_dim}",
                    out.len()
                )));
            }
            for (acc, v) in row_accum.iter_mut().zip(out.iter().copied()) {
                *acc += v;
            }
        }

        let row_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, n_dim]), row_accum)
            .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: row output: {e}")))?;
        row_outputs.push(row_arr);
    }

    let stacked: ndarray::ArrayD<f64> = if row_outputs.is_empty() {
        ndarray::ArrayD::zeros(ndarray::IxDyn(&[0, n_dim]))
    } else {
        ndarray::concatenate(
            ndarray::Axis(0),
            &row_outputs.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: row concat: {e}")))?
    };

    let output_shape_vec = resolve_output_shape(slice_id, &input_shape, n_dim, target_shape)?;

    let final_result = stacked
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(ndarray::IxDyn(&output_shape_vec))
        .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: dim-split reshape: {e}")))?;

    tracing::info!(
        slice = %slice_id,
        rows = total_rows,
        k_chunks = k_chunks,
        "executed dim-split (sequence + K tiled)"
    );

    Ok(final_result)
}

fn execute_generic_dim_split(
    slice_id: &str,
    ds: &crate::schema::tiling::DimSplitInfo,
    target_shape: Option<&[i64]>,
    tensor_cache: &TensorStore,
    tmpl_path: &Path,
) -> Result<ndarray::ArrayD<f64>> {
    use ndarray::Axis;

    let concat_axis = ds.concat_axis;
    let split_dim = ds.split_dim;
    let epg = ds.elements_per_group;

    let tmpl_model = crate::slicer::onnx_proto::load_model(tmpl_path)?;
    let tmpl_graph = tmpl_model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Pipeline(format!("{slice_id}: template has no graph")))?;
    let tmpl_init_names: std::collections::HashSet<&str> = tmpl_graph
        .initializer
        .iter()
        .map(|i| i.name.as_str())
        .collect();
    let input_names: Vec<String> = tmpl_graph
        .input
        .iter()
        .filter(|vi| !tmpl_init_names.contains(vi.name.as_str()))
        .map(|vi| vi.name.clone())
        .collect();

    let tmp_dir = tempfile::tempdir()
        .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: tmpdir: {e}")))?;
    let tmpl_on_disk = tmp_dir.path().join("dim_tmpl.onnx");
    crate::slicer::onnx_proto::save_model(&tmpl_model, &tmpl_on_disk)?;

    let mut group_outputs: Vec<ndarray::ArrayD<f64>> = Vec::new();

    for g in 0..ds.num_groups {
        let dim_start = g * epg;
        if dim_start >= ds.dim_size {
            break;
        }
        let dim_end = ((g + 1) * epg).min(ds.dim_size);
        let actual_size = dim_end - dim_start;

        // dim_size is required to be an exact multiple of epg by the
        // detector (`smallest_divisor_at_least`), so every group is
        // exactly `epg` wide and we can feed the sliced view straight
        // in -- no zero-padding, no output trimming, no risk of
        // contaminating reductions on non-split axes.
        debug_assert_eq!(
            actual_size, epg,
            "dim-split detector must enforce dim_size % epg == 0"
        );
        let mut group_cache = TensorStore::new();
        for vi_name in &input_names {
            let arr = tensor_cache.try_get(vi_name).ok_or_else(|| {
                DsperseError::Pipeline(format!(
                    "{slice_id}: template input {vi_name:?} not found in tensor cache"
                ))
            })?;
            let shape = arr.shape();
            if split_dim < shape.len() && shape[split_dim] == ds.dim_size {
                let sliced = arr
                    .slice_axis(Axis(split_dim), ndarray::Slice::from(dim_start..dim_end))
                    .to_owned();
                group_cache.put(vi_name.clone(), sliced);
            } else {
                group_cache.put(vi_name.clone(), arr.clone());
            }
        }

        let mut named = run_onnx_inference_multi_named(&tmpl_on_disk, &group_cache, &input_names)?;
        let (data, shape) = named.remove(&ds.output_name).ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "{slice_id}: dim-split group {g} missing output {:?} (available: {:?})",
                ds.output_name,
                named.keys().collect::<Vec<_>>()
            ))
        })?;
        let group_output = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
            .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: group {g} reshape: {e}")))?;

        // Output is naturally `epg` wide along concat_axis.
        let trimmed = group_output;

        group_outputs.push(trimmed);
    }

    let result = ndarray::concatenate(
        Axis(concat_axis),
        &group_outputs.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: dim-split concat: {e}")))?;

    let output_shape_vec = if let Some(target) = target_shape {
        target
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| {
                    DsperseError::Pipeline(format!(
                        "{slice_id}: invalid target dim {d} in dim-split reshape"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        result.shape().to_vec()
    };

    let final_result = result
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(ndarray::IxDyn(&output_shape_vec))
        .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: dim-split reshape: {e}")))?;

    tracing::info!(
        slice = %slice_id,
        groups = ds.num_groups,
        split_kind = ?ds.split_kind,
        "executed dim-split (generic)"
    );

    Ok(final_result)
}

pub fn dim_split_unit_count(total_rows: usize, ds: &crate::schema::tiling::DimSplitInfo) -> usize {
    total_rows.saturating_mul(ds.k_chunks.max(1))
}

fn dim_split_row_chunk(row: &[f64], kc: usize, k_dim: usize, k_chunk_size: usize) -> Vec<f64> {
    let k_start = kc * k_chunk_size;
    let k_end = (k_start + k_chunk_size).min(k_dim);
    let actual_k = k_end.saturating_sub(k_start);
    let mut chunk = vec![0.0f64; k_chunk_size];
    if actual_k > 0 {
        chunk[..actual_k].copy_from_slice(&row[k_start..k_end]);
    }
    chunk
}

pub fn dim_split_weight_chunk(
    full_weight: &[f32],
    ds: &crate::schema::tiling::DimSplitInfo,
    kc: usize,
    trans_b: bool,
) -> Vec<f32> {
    let k_dim = ds.k_dim;
    let n_dim = ds.n_dim;
    let k_chunks = ds.k_chunks.max(1);
    let k_chunk_size = k_dim.div_ceil(k_chunks);
    let k_start = kc * k_chunk_size;
    let k_end = (k_start + k_chunk_size).min(k_dim);
    let actual_k = k_end.saturating_sub(k_start);
    if trans_b {
        let mut w = Vec::with_capacity(n_dim * k_chunk_size);
        for row_idx in 0..n_dim {
            let row_start = row_idx * k_dim + k_start;
            let avail = actual_k.min(full_weight.len().saturating_sub(row_start));
            w.extend_from_slice(&full_weight[row_start..row_start + avail]);
            if avail < k_chunk_size {
                w.resize(w.len() + k_chunk_size - avail, 0.0);
            }
        }
        w
    } else {
        let mut w = Vec::with_capacity(k_chunk_size * n_dim);
        for ki in k_start..k_start + actual_k {
            let start = ki * n_dim;
            let end = start + n_dim;
            if end <= full_weight.len() {
                w.extend_from_slice(&full_weight[start..end]);
            } else {
                w.resize(w.len() + n_dim, 0.0);
            }
        }
        if actual_k < k_chunk_size {
            w.resize(k_chunk_size * n_dim, 0.0);
        }
        w
    }
}

pub fn split_for_dim_split_dispatch(
    input: &ndarray::ArrayD<f64>,
    ds: &crate::schema::tiling::DimSplitInfo,
) -> Result<Vec<Vec<f64>>> {
    let input_shape = input.shape().to_vec();
    let k_dim = *input_shape.last().unwrap_or(&0);
    if k_dim == 0 {
        return Err(DsperseError::Pipeline(format!(
            "dim-split slice {}: input {:?} has zero-width last dim",
            ds.slice_idx, ds.input_name
        )));
    }
    if ds.k_dim != 0 && k_dim != ds.k_dim {
        return Err(DsperseError::Pipeline(format!(
            "dim-split slice {}: runtime k_dim {k_dim} does not match metadata k_dim {}",
            ds.slice_idx, ds.k_dim
        )));
    }
    let k_chunks = ds.k_chunks.max(1);
    let k_chunk_size = k_dim.div_ceil(k_chunks);
    let total_rows: usize = input_shape
        .iter()
        .take(input_shape.len().saturating_sub(1))
        .product();
    let flat = input
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(ndarray::IxDyn(&[total_rows, k_dim]))
        .map_err(|e| {
            DsperseError::Pipeline(format!(
                "dim-split slice {}: flatten input: {e}",
                ds.slice_idx
            ))
        })?;
    let mut units = Vec::with_capacity(dim_split_unit_count(total_rows, ds));
    for r in 0..total_rows {
        let row: Vec<f64> = flat.slice(ndarray::s![r, ..]).iter().copied().collect();
        for kc in 0..k_chunks {
            units.push(dim_split_row_chunk(&row, kc, k_dim, k_chunk_size));
        }
    }
    Ok(units)
}

pub fn dim_split_weight_and_transb(
    slice_onnx_path: &Path,
    ds: &crate::schema::tiling::DimSplitInfo,
) -> Result<(Vec<f32>, bool)> {
    let weight_name = ds.weight_name.as_ref().ok_or_else(|| {
        DsperseError::Pipeline(format!(
            "dim-split slice {}: missing weight_name in metadata",
            ds.slice_idx
        ))
    })?;
    let model = crate::slicer::onnx_proto::load_model(slice_onnx_path)?;
    let graph = model.graph.as_ref().ok_or_else(|| {
        DsperseError::Pipeline(format!(
            "dim-split slice {}: ONNX has no graph",
            ds.slice_idx
        ))
    })?;
    let matmul_node = graph
        .node
        .iter()
        .find(|n| {
            matches!(n.op_type.as_str(), "MatMul" | "Gemm")
                && n.input.iter().any(|i| i == weight_name)
                && n.input.iter().any(|i| i == &ds.input_name)
                && n.output.iter().any(|o| o == &ds.output_name)
        })
        .ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "dim-split slice {}: no MatMul/Gemm node matches weight={weight_name:?}",
                ds.slice_idx
            ))
        })?;
    let trans_b = matmul_node.op_type == "Gemm"
        && crate::slicer::onnx_proto::get_attribute_int(matmul_node, "transB").unwrap_or(0) == 1;
    let full_weight = graph
        .initializer
        .iter()
        .find(|i| i.name == *weight_name)
        .map(crate::slicer::onnx_proto::tensor_to_f32)
        .ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "dim-split slice {}: weight {weight_name:?} not in ONNX initializers",
                ds.slice_idx
            ))
        })?;
    let expected_weight_len = ds.k_dim.saturating_mul(ds.n_dim);
    if full_weight.len() != expected_weight_len {
        return Err(DsperseError::Pipeline(format!(
            "dim-split slice {}: weight {weight_name:?} length {} does not match k_dim*n_dim = {}*{} = {}",
            ds.slice_idx,
            full_weight.len(),
            ds.k_dim,
            ds.n_dim,
            expected_weight_len
        )));
    }
    Ok((full_weight, trans_b))
}

pub fn dim_split_bound_inputs(
    input: &ndarray::ArrayD<f64>,
    slice_onnx_path: &Path,
    ds: &crate::schema::tiling::DimSplitInfo,
) -> Result<Vec<Vec<f64>>> {
    let activations = split_for_dim_split_dispatch(input, ds)?;
    let (full_weight, trans_b) = dim_split_weight_and_transb(slice_onnx_path, ds)?;
    let k_chunks = ds.k_chunks.max(1);
    let weight_chunks: Vec<Vec<f64>> = (0..k_chunks)
        .map(|kc| {
            dim_split_weight_chunk(&full_weight, ds, kc, trans_b)
                .into_iter()
                .map(f64::from)
                .collect()
        })
        .collect();
    let mut bound = Vec::with_capacity(activations.len());
    for (unit_idx, mut payload) in activations.into_iter().enumerate() {
        payload.extend_from_slice(&weight_chunks[unit_idx % k_chunks]);
        bound.push(payload);
    }
    Ok(bound)
}

fn resolve_output_shape(
    slice_id: &str,
    input_shape: &[usize],
    n_dim: usize,
    target_shape: Option<&[i64]>,
) -> Result<Vec<usize>> {
    if let Some(target) = target_shape {
        target
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| {
                    DsperseError::Pipeline(format!(
                        "{slice_id}: invalid target dimension {d} in dim-split reshape"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()
    } else {
        let mut s = input_shape.to_vec();
        if let Some(last) = s.last_mut() {
            *last = n_dim;
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::tiling::DimSplitInfo;
    use ndarray::{ArrayD, IxDyn};

    fn slice_148_meta() -> DimSplitInfo {
        DimSplitInfo {
            slice_idx: 148,
            k_dim: 384,
            n_dim: 1536,
            k_chunks: 2,
            ..Default::default()
        }
    }

    #[test]
    fn dispatch_split_matches_local_chunk_geometry() {
        let ds = slice_148_meta();
        let input = ArrayD::from_shape_fn(IxDyn(&[4, 145, 384]), |d| (d[2] as f64) + 1.0);
        let units = split_for_dim_split_dispatch(&input, &ds).unwrap();
        assert_eq!(units.len(), 4 * 145 * 2);
        assert_eq!(dim_split_unit_count(4 * 145, &ds), units.len());
        assert!(units.iter().all(|u| u.len() == 192));
        assert_eq!(units[0][0], 1.0);
        assert_eq!(units[0][191], 192.0);
        assert_eq!(units[1][0], 193.0);
        assert_eq!(units[1][191], 384.0);
    }

    #[test]
    fn weight_chunk_partitions_full_weight_along_k() {
        let ds = slice_148_meta();
        let full: Vec<f32> = (0..ds.k_dim)
            .flat_map(|k| std::iter::repeat_n(k as f32, ds.n_dim))
            .collect();
        let c0 = dim_split_weight_chunk(&full, &ds, 0, false);
        let c1 = dim_split_weight_chunk(&full, &ds, 1, false);
        assert_eq!(c0.len(), 192 * ds.n_dim);
        assert_eq!(c1.len(), 192 * ds.n_dim);
        assert_eq!(c0[0], 0.0);
        assert_eq!(*c0.last().unwrap(), 191.0);
        assert_eq!(c1[0], 192.0);
        assert_eq!(*c1.last().unwrap(), 383.0);
    }

    #[test]
    fn weight_chunk_transposed_gemm_band_ordering() {
        let ds = DimSplitInfo {
            slice_idx: 7,
            k_dim: 4,
            n_dim: 3,
            k_chunks: 2,
            ..Default::default()
        };
        let full: Vec<f32> = (0..ds.n_dim)
            .flat_map(|r| (0..ds.k_dim).map(move |c| (r * 10 + c) as f32))
            .collect();
        let c0 = dim_split_weight_chunk(&full, &ds, 0, true);
        let c1 = dim_split_weight_chunk(&full, &ds, 1, true);
        assert_eq!(c0, vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
        assert_eq!(c1, vec![2.0, 3.0, 12.0, 13.0, 22.0, 23.0]);
    }

    #[test]
    fn split_rejects_k_dim_mismatch() {
        let ds = slice_148_meta();
        let bad = ArrayD::zeros(IxDyn(&[4, 145, 256]));
        assert!(split_for_dim_split_dispatch(&bad, &ds).is_err());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupPayloadPart {
    Whole(usize),
    Split(usize),
}

pub fn plan_group_payload(
    manifest_shapes: &[Vec<usize>],
    ds: &crate::schema::tiling::DimSplitInfo,
    contract: &[(String, Vec<usize>)],
) -> Result<Vec<GroupPayloadPart>> {
    if ds.num_groups == 0
        || ds.elements_per_group == 0
        || ds.num_groups * ds.elements_per_group != ds.dim_size
    {
        return Err(DsperseError::Pipeline(format!(
            "dim-split slice {}: groups {}x{} do not cover dim_size {}",
            ds.slice_idx, ds.num_groups, ds.elements_per_group, ds.dim_size
        )));
    }
    let reduced = |shape: &[usize]| -> Option<Vec<usize>> {
        if ds.split_dim < shape.len() && shape[ds.split_dim] == ds.dim_size {
            let mut r = shape.to_vec();
            r[ds.split_dim] = ds.elements_per_group;
            Some(r)
        } else {
            None
        }
    };
    let mut used = vec![false; manifest_shapes.len()];
    let mut parts = Vec::new();
    let mut trailing_started = false;
    for (name, entry_shape) in contract {
        let mut matched = None;
        for (i, shape) in manifest_shapes.iter().enumerate() {
            if used[i] {
                continue;
            }
            if shape == entry_shape {
                matched = Some(GroupPayloadPart::Whole(i));
                break;
            }
            if reduced(shape).as_deref() == Some(entry_shape.as_slice()) {
                matched = Some(GroupPayloadPart::Split(i));
                break;
            }
        }
        match matched {
            Some(part) => {
                if trailing_started {
                    return Err(DsperseError::Pipeline(format!(
                        "dim-split slice {}: activation entry {name} {entry_shape:?} appears after initializer entries",
                        ds.slice_idx
                    )));
                }
                let idx = match part {
                    GroupPayloadPart::Whole(i) | GroupPayloadPart::Split(i) => i,
                };
                used[idx] = true;
                parts.push(part);
            }
            None => {
                trailing_started = true;
            }
        }
    }
    if let Some(unused) = used.iter().position(|u| !u) {
        return Err(DsperseError::Pipeline(format!(
            "dim-split slice {}: manifest tensor {unused} unused by circuit contract",
            ds.slice_idx
        )));
    }
    Ok(parts)
}

pub fn dim_split_group_payloads_planned(
    tensors: &[&ndarray::ArrayD<f64>],
    plan: &[GroupPayloadPart],
    ds: &crate::schema::tiling::DimSplitInfo,
) -> Result<Vec<Vec<f64>>> {
    let axis = ndarray::Axis(ds.split_dim);
    let mut payloads = Vec::with_capacity(ds.num_groups);
    for g in 0..ds.num_groups {
        let start = g * ds.elements_per_group;
        let range = ndarray::Slice::from(start..start + ds.elements_per_group);
        let mut payload: Vec<f64> = Vec::new();
        for part in plan {
            match part {
                GroupPayloadPart::Whole(i) => {
                    payload.extend(tensors[*i].as_standard_layout().iter());
                }
                GroupPayloadPart::Split(i) => {
                    payload.extend(
                        tensors[*i]
                            .slice_axis(axis, range)
                            .as_standard_layout()
                            .iter(),
                    );
                }
            }
        }
        payloads.push(payload);
    }
    Ok(payloads)
}

#[cfg(test)]
mod group_payload_tests {
    use super::*;
    use crate::schema::tiling::{DimSplitInfo, DimSplitKind};
    use ndarray::{ArrayD, IxDyn};

    fn ds(split_dim: usize, dim_size: usize, num_groups: usize, epg: usize) -> DimSplitInfo {
        DimSplitInfo {
            split_kind: DimSplitKind::BatchDim,
            split_dim,
            dim_size,
            num_groups,
            elements_per_group: epg,
            ..Default::default()
        }
    }

    #[test]
    fn contraction_with_coincidental_axis_size_stays_whole() {
        let ds_info = ds(2, 4, 2, 2);
        let manifest = vec![vec![2, 3, 4, 4], vec![2, 3, 4, 5]];
        let contract = vec![
            ("in0".to_string(), vec![2, 3, 4, 5]),
            ("in1".to_string(), vec![2, 3, 2, 4]),
        ];
        let plan = plan_group_payload(&manifest, &ds_info, &contract).unwrap();
        assert_eq!(
            plan,
            vec![GroupPayloadPart::Whole(1), GroupPayloadPart::Split(0)],
            "secondary sharing the axis size must stay whole when the contract says so"
        );
    }

    #[test]
    fn broadcast_secondary_splits_when_contract_says_so() {
        let ds_info = ds(2, 4, 2, 2);
        let manifest = vec![vec![2, 3, 4, 4], vec![2, 1, 4, 4]];
        let contract = vec![
            ("in0".to_string(), vec![2, 1, 2, 4]),
            ("in1".to_string(), vec![2, 3, 2, 4]),
        ];
        let plan = plan_group_payload(&manifest, &ds_info, &contract).unwrap();
        assert_eq!(
            plan,
            vec![GroupPayloadPart::Split(1), GroupPayloadPart::Split(0)]
        );
    }

    #[test]
    fn trailing_initializer_entries_are_tolerated_but_gaps_are_not() {
        let ds_info = ds(1, 4, 2, 2);
        let manifest = vec![vec![1, 4, 1], vec![1, 4, 1]];
        let ok = vec![
            ("a".to_string(), vec![1, 2, 1]),
            ("b".to_string(), vec![1, 2, 1]),
            ("freq".to_string(), vec![128]),
        ];
        assert!(plan_group_payload(&manifest, &ds_info, &ok).is_ok());
        let gap = vec![
            ("a".to_string(), vec![1, 2, 1]),
            ("freq".to_string(), vec![128]),
            ("b".to_string(), vec![1, 2, 1]),
        ];
        assert!(plan_group_payload(&manifest, &ds_info, &gap).is_err());
        let unused = vec![("a".to_string(), vec![1, 2, 1])];
        assert!(plan_group_payload(&manifest, &ds_info, &unused).is_err());
    }
}
