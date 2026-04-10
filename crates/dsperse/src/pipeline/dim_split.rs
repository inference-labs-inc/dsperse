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
        let k_start = kc * k_chunk_size;
        let k_end = (k_start + k_chunk_size).min(k_dim);
        let actual_k = k_end.saturating_sub(k_start);

        let weight_chunk: Vec<f32> = if trans_b {
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
        };

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
            let k_start = kc * k_chunk_size;
            let k_end = (k_start + k_chunk_size).min(k_dim);
            let actual_k = k_end.saturating_sub(k_start);

            let mut input_chunk = vec![0.0f64; k_chunk_size];
            if actual_k > 0 {
                input_chunk[..actual_k].copy_from_slice(&full_row[k_start..k_end]);
            }

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
                if actual_size < epg {
                    let mut padded_shape = sliced.shape().to_vec();
                    padded_shape[split_dim] = epg;
                    let mut padded = ndarray::ArrayD::zeros(padded_shape);
                    for (mut dst, src) in padded
                        .axis_iter_mut(Axis(split_dim))
                        .zip(sliced.axis_iter(Axis(split_dim)))
                    {
                        dst.assign(&src);
                    }
                    group_cache.put(vi_name.clone(), padded);
                } else {
                    group_cache.put(vi_name.clone(), sliced);
                }
            } else {
                group_cache.put(vi_name.clone(), arr.clone());
            }
        }

        let named = run_onnx_inference_multi_named(&tmpl_on_disk, &group_cache, &input_names)?;
        if named.len() != 1 {
            return Err(DsperseError::Pipeline(format!(
                "{slice_id}: dim-split group {g} produced {} outputs, expected 1",
                named.len()
            )));
        }
        let (data, shape) = named.into_values().next().ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "{slice_id}: dim-split group {g} produced no output"
            ))
        })?;
        let group_output = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
            .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: group {g} reshape: {e}")))?;

        let trimmed = if actual_size < epg && concat_axis < group_output.ndim() {
            group_output
                .slice_axis(Axis(concat_axis), ndarray::Slice::from(0..actual_size))
                .to_owned()
        } else {
            group_output
        };

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
