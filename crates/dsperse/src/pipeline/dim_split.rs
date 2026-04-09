use std::collections::HashMap;
use std::path::Path;

use super::runner::{run_onnx_inference, run_onnx_inference_multi_named};
use super::tensor_store::TensorStore;
use crate::backend::jstprove::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::ExecutionInfo;
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
    use ndarray::Axis;

    let concat_axis = ds.concat_axis;
    let epg = ds.elements_per_group;

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

    let use_matmul_split = matches!(
        ds.split_kind,
        crate::schema::tiling::DimSplitKind::MatMulOutputDim
    );

    let slice_onnx_path = slices_dir
        .join(format!("slice_{}", ds.slice_idx))
        .join("payload")
        .join(format!("slice_{}.onnx", ds.slice_idx));

    let fallback_model = if use_matmul_split && donor_init_map.is_none() {
        Some(crate::slicer::onnx_proto::load_model(&slice_onnx_path)?)
    } else {
        None
    };

    let original_weights = if use_matmul_split {
        let wn = ds.weight_name.as_ref().ok_or_else(|| {
            DsperseError::Pipeline(format!(
                "{slice_id}: MatMulOutputDim split requires weight_name in metadata"
            ))
        })?;
        let tensor = if let Some(map) = donor_init_map
            && let Some(t) = map.get(wn.as_str())
        {
            (*t).clone()
        } else {
            let graph = fallback_model
                .as_ref()
                .and_then(|m| m.graph.as_ref())
                .ok_or_else(|| {
                    DsperseError::Pipeline(format!("{slice_id}: slice ONNX has no graph"))
                })?;
            graph
                .initializer
                .iter()
                .find(|i| i.name == *wn)
                .ok_or_else(|| {
                    DsperseError::Pipeline(format!(
                        "{slice_id}: weight {wn:?} not found in slice ONNX"
                    ))
                })?
                .clone()
        };
        Some(crate::slicer::onnx_proto::tensor_to_f32(&tensor))
    } else {
        None
    };

    let mut tmpl_model = crate::slicer::onnx_proto::load_model(&tmpl_path)?;

    let tmpl_init_names: std::collections::HashSet<String> = tmpl_model
        .graph
        .as_ref()
        .map(|g| g.initializer.iter().map(|i| i.name.clone()).collect())
        .unwrap_or_default();
    let tmpl_non_init_inputs: Vec<String> = tmpl_model
        .graph
        .as_ref()
        .map(|g| {
            g.input
                .iter()
                .filter(|vi| !tmpl_init_names.contains(&vi.name))
                .map(|vi| vi.name.clone())
                .collect()
        })
        .unwrap_or_default();

    let mut group_outputs: Vec<ndarray::ArrayD<f64>> = Vec::new();

    for g in 0..ds.num_groups {
        let dim_start = g * epg;
        if dim_start >= ds.dim_size {
            break;
        }
        let dim_end = ((g + 1) * epg).min(ds.dim_size);
        let actual_size = dim_end - dim_start;

        let group_input = if use_matmul_split {
            tensor_cache.get(&ds.input_name)?.clone()
        } else {
            let split_dim = ds.split_dim;
            let full = tensor_cache.get(&ds.input_name)?;
            if split_dim >= full.ndim() {
                return Err(DsperseError::Pipeline(format!(
                    "{slice_id}: split_dim {split_dim} out of range for tensor with {} dimensions",
                    full.ndim()
                )));
            }
            let sliced = full
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
                padded
            } else {
                sliced
            }
        };

        if use_matmul_split
            && let Some(ref weights) = original_weights
            && let Some(graph) = tmpl_model.graph.as_mut()
        {
            let matmul_node = graph
                .node
                .iter()
                .find(|n| matches!(n.op_type.as_str(), "MatMul" | "Gemm"));
            let trans_b = matmul_node.is_some_and(|n| {
                n.op_type == "Gemm"
                    && crate::slicer::onnx_proto::get_attribute_int(n, "transB").unwrap_or(0) == 1
            });

            if let Some(w_init) = graph.initializer.iter_mut().find(|i| i.name == "W") {
                if w_init.dims.len() < 2 {
                    return Err(DsperseError::Pipeline(format!(
                        "{slice_id}: weight initializer 'W' has {} dims, expected at least 2",
                        w_init.dims.len()
                    )));
                }
                let rows = usize::try_from(w_init.dims[0]).map_err(|_| {
                    DsperseError::Pipeline(format!(
                        "{slice_id}: weight dim[0]={} is negative",
                        w_init.dims[0]
                    ))
                })?;
                let cols = usize::try_from(w_init.dims[1]).map_err(|_| {
                    DsperseError::Pipeline(format!(
                        "{slice_id}: weight dim[1]={} is negative",
                        w_init.dims[1]
                    ))
                })?;
                let k = if trans_b { cols } else { rows };
                let orig_cols = ds.dim_size;
                let mut chunk = Vec::with_capacity(k * epg);
                if trans_b {
                    for r in dim_start..dim_end.min(ds.dim_size) {
                        let start = r * k;
                        let end = start + k;
                        let slice = weights.get(start..end).ok_or_else(|| {
                            DsperseError::Pipeline(format!(
                                "{slice_id}: weight slice [{start}..{end}] out of bounds \
                                 (weights len={})",
                                weights.len()
                            ))
                        })?;
                        chunk.extend_from_slice(slice);
                    }
                    chunk.resize(epg * k, 0.0);
                } else {
                    for r in 0..k {
                        let start = r * orig_cols + dim_start;
                        let end = start + actual_size;
                        let row = weights.get(start..end).ok_or_else(|| {
                            DsperseError::Pipeline(format!(
                                "{slice_id}: weight slice [{start}..{end}] out of bounds \
                                 (weights len={})",
                                weights.len()
                            ))
                        })?;
                        chunk.extend_from_slice(row);
                        let row_target = (r + 1) * epg;
                        chunk.resize(row_target, 0.0);
                    }
                }
                w_init.float_data = chunk;
                w_init.raw_data.clear();
            }

            if let Some(bias_init) = graph.initializer.iter_mut().find(|i| i.name == "C") {
                let bias_name = matmul_node.and_then(|n| n.input.get(2).cloned());
                let bias_data: Option<Vec<f32>> = bias_name.as_ref().and_then(|bn| {
                    if let Some(map) = donor_init_map
                        && let Some(t) = map.get(bn.as_str())
                    {
                        return Some(crate::slicer::onnx_proto::tensor_to_f32(t));
                    }
                    fallback_model
                        .as_ref()
                        .and_then(|m| m.graph.as_ref())
                        .and_then(|g| {
                            g.initializer
                                .iter()
                                .find(|i| i.name == *bn)
                                .map(crate::slicer::onnx_proto::tensor_to_f32)
                        })
                });
                if let Some(bd) = bias_data {
                    if bd.len() == ds.dim_size {
                        let mut sliced = bd[dim_start..dim_end].to_vec();
                        sliced.resize(epg, 0.0);
                        bias_init.float_data = sliced;
                        bias_init.raw_data.clear();
                    } else {
                        tracing::warn!(
                            slice = %slice_id,
                            bias_len = bd.len(),
                            dim_size = ds.dim_size,
                            group = g,
                            "bias length mismatch; skipping bias patching for this group"
                        );
                    }
                }
            }
        }

        let tmp_dir = tempfile::tempdir()
            .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: tmpdir: {e}")))?;
        let patched_path = tmp_dir.path().join("dim_chunk.onnx");
        crate::slicer::onnx_proto::save_model(&tmpl_model, &patched_path)?;

        let group_output = if use_matmul_split {
            run_onnx_inference(&patched_path, &group_input)?
        } else {
            let tmpl_graph = tmpl_model.graph.as_ref().ok_or_else(|| {
                DsperseError::Pipeline(format!("{slice_id}: template has no graph"))
            })?;
            let mut group_cache = TensorStore::new();
            for vi in &tmpl_graph.input {
                if tmpl_init_names.contains(&vi.name) {
                    continue;
                }
                if vi.name == "dim_tmpl_in" {
                    group_cache.put(vi.name.clone(), group_input.clone());
                } else {
                    let arr = tensor_cache.try_get(&vi.name).ok_or_else(|| {
                        DsperseError::Pipeline(format!(
                            "{slice_id}: template input {:?} not found in tensor cache",
                            vi.name
                        ))
                    })?;
                    let shape = arr.shape();
                    if ds.split_dim < shape.len() && shape[ds.split_dim] == ds.dim_size {
                        let sliced = arr
                            .slice_axis(
                                Axis(ds.split_dim),
                                ndarray::Slice::from(dim_start..dim_end),
                            )
                            .to_owned();
                        group_cache.put(vi.name.clone(), sliced);
                    } else {
                        group_cache.put(vi.name.clone(), arr.clone());
                    }
                }
            }
            let named =
                run_onnx_inference_multi_named(&patched_path, &group_cache, &tmpl_non_init_inputs)?;
            if named.len() != 1 {
                return Err(DsperseError::Pipeline(format!(
                    "{slice_id}: dim-split group produced {} outputs, expected exactly 1",
                    named.len()
                )));
            }
            let (data, shape) = named.into_values().next().ok_or_else(|| {
                DsperseError::Pipeline(format!("{slice_id}: dim-split group produced no output"))
            })?;
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data).map_err(|e| {
                DsperseError::Pipeline(format!("{slice_id}: dim-split array reshape: {e}"))
            })?
        };

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
    .map_err(|e| DsperseError::Pipeline(format!("{slice_id}: dim-split concat failed: {e}")))?;

    let final_result = if let Some(target) = target_shape {
        let target_usize: Vec<usize> = target
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| {
                    DsperseError::Pipeline(format!(
                        "{slice_id}: invalid target dimension {d} in dim-split reshape"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        result
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(ndarray::IxDyn(&target_usize))
            .map_err(|e| {
                DsperseError::Pipeline(format!(
                    "{slice_id}: dim-split reshape to target failed: {e}"
                ))
            })?
    } else {
        result
    };

    tracing::info!(
        slice = %slice_id,
        groups = ds.num_groups,
        "executed dim-split"
    );

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
