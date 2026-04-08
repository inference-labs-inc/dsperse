use std::collections::HashMap;
use std::path::Path;

use ndarray::{Array4, ArrayD, s};

use super::runner::{generate_wai_witness, resolve_circuit_path_optional, run_onnx_inference};
use super::tensor_store::TensorStore;
use crate::backend::jstprove::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::schema::execution::{ExecutionInfo, ExecutionMethod};
use crate::schema::tiling::{ChannelGroupInfo, ChannelSplitInfo};
use crate::slicer::onnx_proto::TensorProto;
use crate::utils::io::read_msgpack;
use crate::utils::paths::resolve_relative_path;

pub(crate) fn reshape_channel_split_output(
    arr: ArrayD<f64>,
    target_shape: Option<&[i64]>,
) -> Result<ArrayD<f64>> {
    let Some(raw) = target_shape else {
        return Ok(arr);
    };
    let target: Vec<usize> = raw
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                DsperseError::Pipeline(format!("negative dimension {d} in output_shape"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if arr.shape() == target.as_slice() {
        return Ok(arr);
    }
    let actual_shape: Vec<usize> = arr.shape().to_vec();
    let actual_elems: usize = actual_shape.iter().product();
    let target_elems: usize = target.iter().product();
    if actual_elems != target_elems {
        return Err(DsperseError::Pipeline(format!(
            "channel_split output element count mismatch: \
             actual {actual_elems} (shape {actual_shape:?}) vs target {target_elems} (shape {target:?})"
        )));
    }
    arr.into_shape_with_order(ndarray::IxDyn(&target))
        .map_err(|e| {
            DsperseError::Pipeline(format!(
                "channel_split output reshape from {actual_shape:?} to {target:?}: {e}",
            ))
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_channel_split(
    slices_dir: &Path,
    slice_run_dir: &Path,
    slice_id: &str,
    cs: &ChannelSplitInfo,
    target_shape: Option<&[i64]>,
    tensor_cache: &TensorStore,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<crate::schema::execution::StrategyOutput> {
    let input_arr = tensor_cache.get(&cs.input_name)?.clone();

    let (input_4d, n, h) = if input_arr.ndim() == 4 {
        let s = input_arr.shape();
        let n = s[0];
        if n != 1 {
            return Err(DsperseError::Pipeline(format!(
                "channel split: batch size {n} not supported, expected 1"
            )));
        }
        let h = s[2];
        let arr =
            Array4::from_shape_vec((n, s[1], s[2], s[3]), input_arr.iter().copied().collect())
                .map_err(|e| DsperseError::Pipeline(format!("channel split reshape: {e}")))?;
        (arr, n, h)
    } else {
        let n = 1usize;
        let input_flat: Vec<f64> = input_arr.iter().copied().collect();
        let total_elements = input_flat.len();
        let nc = n * cs.c_in;
        if nc > 0 && !total_elements.is_multiple_of(nc) {
            return Err(DsperseError::Pipeline(format!(
                "channel split reshape: total_elements {total_elements} not divisible by n*c_in ({nc})"
            )));
        }
        let spatial = if cs.c_in > 0 && total_elements > 0 {
            total_elements / nc
        } else {
            cs.h * cs.w
        };
        let h = cs.h.max(1);
        if spatial > 0 && h > 0 && spatial % h != 0 {
            return Err(DsperseError::Pipeline(format!(
                "channel split reshape: spatial {spatial} not divisible by h={h}"
            )));
        }
        let w = if spatial > 0 && h > 0 {
            spatial / h
        } else {
            cs.w.max(1)
        };
        let arr = Array4::from_shape_vec((n, cs.c_in, h, w), input_flat)
            .map_err(|e| DsperseError::Pipeline(format!("channel split reshape: {e}")))?;
        (arr, n, h)
    };

    let mut accumulated: Option<Array4<f64>> = None;

    tracing::info!(
        slice = %slice_id,
        num_groups = cs.groups.len(),
        "channel split execution"
    );

    let n_channels = input_4d.shape()[1];
    for group in &cs.groups {
        if group.c_end > n_channels || group.c_start > group.c_end {
            return Err(DsperseError::Pipeline(format!(
                "channel group {} bounds [{}, {}) exceed channel dimension {}",
                group.group_idx, group.c_start, group.c_end, n_channels
            )));
        }
        let group_input = input_4d
            .slice(s![.., group.c_start..group.c_end, .., ..])
            .to_owned();
        let group_input_dyn = group_input.into_dyn();

        let group_dir = slice_run_dir.join(format!("group_{}", group.group_idx));
        std::fs::create_dir_all(&group_dir).map_err(|e| DsperseError::io(e, &group_dir))?;

        let group_output = execute_channel_group(
            slices_dir,
            &group_dir,
            group,
            &group_input_dyn,
            backend,
            donor_init_map,
        )?;

        let group_4d = if group_output.ndim() == 4 {
            let s = group_output.shape();
            Array4::from_shape_vec(
                (s[0], s[1], s[2], s[3]),
                group_output.iter().copied().collect(),
            )
            .map_err(|e| DsperseError::Pipeline(format!("group output reshape: {e}")))?
        } else {
            let group_flat: Vec<f64> = group_output.iter().copied().collect();
            let (out_h, out_w) = if cs.out_h > 0 && cs.out_w > 0 {
                (cs.out_h, cs.out_w)
            } else if cs.c_out > 0 {
                let out_spatial = group_flat.len() / (n * cs.c_out);
                if h > 0 && out_spatial > 0 && out_spatial.is_multiple_of(h) {
                    (h, out_spatial / h)
                } else {
                    return Err(DsperseError::Pipeline(format!(
                        "cannot determine spatial layout for channel_split output: {} elements, c_out={}, set out_h/out_w in metadata",
                        group_flat.len(),
                        cs.c_out
                    )));
                }
            } else {
                return Err(DsperseError::Pipeline("channel split c_out is 0".into()));
            };
            if n * cs.c_out * out_h * out_w != group_flat.len() {
                return Err(DsperseError::Pipeline(format!(
                    "group output reshape mismatch: expected {} elements (n={}, c_out={}, h={}, w={}), got {}",
                    n * cs.c_out * out_h * out_w,
                    n,
                    cs.c_out,
                    out_h,
                    out_w,
                    group_flat.len()
                )));
            }
            Array4::from_shape_vec((n, cs.c_out, out_h, out_w), group_flat)
                .map_err(|e| DsperseError::Pipeline(format!("group output reshape: {e}")))?
        };

        accumulated = Some(match accumulated {
            Some(acc) => {
                if acc.shape() != group_4d.shape() {
                    return Err(DsperseError::Pipeline(format!(
                        "channel group {} shape {:?} does not match accumulator shape {:?}",
                        group.group_idx,
                        group_4d.shape(),
                        acc.shape()
                    )));
                }
                acc + &group_4d
            }
            None => group_4d,
        });
    }

    if let Some(ref bias_path_str) = cs.bias_path {
        let bias_file = resolve_relative_path(slices_dir, bias_path_str)?;
        if !bias_file.exists() {
            return Err(DsperseError::Pipeline(format!(
                "configured bias file not found: {} (bias_path={bias_path_str})",
                bias_file.display()
            )));
        }
        let bias_data = read_msgpack(&bias_file)?;
        let bias_flat = crate::utils::io::flatten_nested_list(&bias_data);
        if bias_flat.len() != cs.c_out {
            return Err(DsperseError::Pipeline(format!(
                "bias length {} does not match c_out {}",
                bias_flat.len(),
                cs.c_out
            )));
        }
        if let Some(ref mut acc) = accumulated {
            for ((_, c, _, _), val) in acc.indexed_iter_mut() {
                *val += bias_flat[c];
            }
        }
    }

    let output = match accumulated {
        Some(acc) => reshape_channel_split_output(acc.into_dyn(), target_shape)?,
        None => {
            return Err(DsperseError::Pipeline(format!(
                "channel_split produced no output for '{}'",
                cs.output_name
            )));
        }
    };

    Ok(crate::schema::execution::StrategyOutput {
        info: ExecutionInfo {
            method: ExecutionMethod::ChannelSplit,
            success: true,
            error: None,
            witness_file: None,
            tile_exec_infos: Vec::new(),
        },
        outputs: vec![(cs.output_name.clone(), output)],
    })
}

fn execute_channel_group(
    slices_dir: &Path,
    group_dir: &Path,
    group: &ChannelGroupInfo,
    group_input: &ArrayD<f64>,
    backend: &JstproveBackend,
    donor_init_map: Option<&HashMap<String, &TensorProto>>,
) -> Result<ArrayD<f64>> {
    let onnx_path = resolve_relative_path(slices_dir, &group.path)?;

    let patched_onnx = if let Some(map) = donor_init_map {
        Some(crate::slicer::onnx_proto::build_patched_onnx(
            &onnx_path, map,
        )?)
    } else {
        None
    };
    let effective_onnx = patched_onnx
        .as_ref()
        .map_or(onnx_path.as_path(), |t| t.path());

    if let Some(circuit_path) =
        resolve_circuit_path_optional(slices_dir, group.jstprove_circuit_path.as_deref())?
    {
        let params = backend.load_params(&circuit_path)?;
        let is_wai = params.as_ref().is_some_and(|p| p.weights_as_inputs);

        if donor_init_map.is_some() && !is_wai {
            return Err(DsperseError::Pipeline(format!(
                "group_{}: consumer weights require circuits compiled with --weights-as-inputs",
                group.group_idx
            )));
        }

        let output_tensor = run_onnx_inference(effective_onnx, group_input)?;

        let flat: Vec<f64> = group_input.iter().copied().collect();
        let witness_bytes = if is_wai {
            generate_wai_witness(
                backend,
                &circuit_path,
                &onnx_path,
                donor_init_map,
                params.as_ref().unwrap(),
                &flat,
            )?
        } else {
            backend.witness_f64(&circuit_path, &flat, &[])?
        };

        let witness_path = group_dir.join(crate::utils::paths::WITNESS_FILE);
        std::fs::write(&witness_path, &witness_bytes)
            .map_err(|e| DsperseError::io(e, &witness_path))?;

        Ok(output_tensor)
    } else {
        run_onnx_inference(effective_onnx, group_input)
    }
}
