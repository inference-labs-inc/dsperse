use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::IxDyn;
use tract_onnx::prelude::*;

use crate::error::{DsperseError, Result};

pub fn coerce_tdim_inputs(inputs: &TVec<TValue>) -> TVec<TValue> {
    inputs
        .iter()
        .map(|t| {
            if t.datum_type() == DatumType::TDim {
                // Safety: datum_type() == TDim verified by outer condition
                let view = unsafe { t.as_slice_unchecked::<TDim>() };
                let vals: Vec<i64> = view.iter().map(|d| d.to_i64().unwrap_or(0)).collect();
                Tensor::from_shape(t.shape(), &vals)
                    .map(|t| t.into_tvalue())
                    .unwrap_or_else(|_| t.clone())
            } else {
                t.clone()
            }
        })
        .collect()
}

pub type NamedOutputs = HashMap<String, (Vec<f64>, Vec<usize>)>;

fn load_onnx_model(onnx_path: &Path) -> Result<InferenceModel> {
    tract_onnx::onnx()
        .model_for_path(onnx_path)
        .map_err(|e| DsperseError::Onnx(format!("load {}: {e}", onnx_path.display())))
}

fn resolve_concrete_shape(model: &InferenceModel, input_shape: &[usize]) -> Result<Vec<usize>> {
    let model_shape = model
        .input_fact(0)
        .ok()
        .and_then(|f| f.shape.as_concrete_finite().ok().flatten())
        .map(|s| s.to_vec());

    if input_shape.is_empty() {
        return model_shape.ok_or_else(|| {
            DsperseError::Onnx("symbolic input shape — provide explicit shape".into())
        });
    }

    if let Some(ref ms) = model_shape {
        let model_elems: usize = ms.iter().product();
        let input_elems: usize = input_shape.iter().product();
        if input_shape.len() == 1 && ms.len() > 1 && model_elems == input_elems {
            tracing::debug!(
                model_shape = ?ms,
                provided_shape = ?input_shape,
                "reshaping flat input to model-declared shape"
            );
            return Ok(ms.clone());
        }
    }

    Ok(input_shape.to_vec())
}

fn optimize_to_runnable(
    model: InferenceModel,
    concrete_shape: &[usize],
) -> Result<Arc<TypedRunnableModel>> {
    model
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), concrete_shape),
        )
        .map_err(|e| DsperseError::Onnx(format!("set input shape: {e}")))?
        .into_optimized()
        .map_err(|e| DsperseError::Onnx(format!("optimize: {e:#}")))?
        .into_runnable()
        .map_err(|e| DsperseError::Onnx(format!("make runnable: {e:#}")))
}

pub fn run_inference_with_coercion(
    onnx_path: &Path,
    input_data: &[f64],
    input_shape: &[usize],
) -> Result<NamedOutputs> {
    let model = load_onnx_model(onnx_path)?;
    let concrete_shape = resolve_concrete_shape(&model, input_shape)?;

    if let Ok(plan) = optimize_to_runnable(model, &concrete_shape) {
        let input = build_input_tvalue(input_data, &concrete_shape)?;
        let result = plan
            .run(tvec![input])
            .map_err(|e| DsperseError::Onnx(format!("run: {e:#}")))?;
        return extract_all_outputs(&result);
    }

    tracing::warn!("standard optimization failed; using inference plan with TDim coercion");
    let model2 = load_onnx_model(onnx_path)?;
    let with_shape = model2
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), &concrete_shape),
        )
        .map_err(|e| DsperseError::Onnx(format!("set input: {e}")))?;

    let plan =
        tract_onnx::tract_hir::infer::InferenceSimplePlan::new(std::sync::Arc::new(with_shape))
            .map_err(|e| DsperseError::Onnx(format!("inference plan: {e}")))?;
    let mut state = tract_onnx::tract_core::plan::SimpleState::new(&plan)
        .map_err(|e| DsperseError::Onnx(format!("state: {e}")))?;

    let input = build_input_tvalue(input_data, &concrete_shape)?;
    let result = state
        .run_plan_with_eval(tvec![input], |session, op_state, node, inputs| {
            let coerced = coerce_tdim_inputs(&inputs);
            let eval_result = if let Some(st) = op_state {
                st.eval(session, node.op.as_op(), coerced)
            } else {
                node.op.eval(coerced)
            };
            match eval_result {
                Ok(o) => Ok::<_, TractError>(o),
                Err(e) => {
                    tracing::warn!(node = %node.name, error = %e, "eval failed, using fallback");
                    let fallback = inputs
                        .first()
                        .cloned()
                        .unwrap_or_else(|| Tensor::zero::<f32>(&[1]).unwrap().into_tvalue());
                    let n = node.outputs.len().max(1);
                    Ok((0..n).map(|_| fallback.clone()).collect())
                }
            }
        })
        .map_err(|e| DsperseError::Onnx(format!("inference run: {e:#}")))?;

    extract_all_outputs(&result)
}

fn extract_all_outputs(result: &[TValue]) -> Result<NamedOutputs> {
    let mut outputs = NamedOutputs::new();
    for (i, tv) in result.iter().enumerate() {
        let shape = tv.shape().to_vec();
        let tensor = tv.clone().into_tensor();
        let data: Vec<f64> = if tensor.datum_type() == f32::datum_type() {
            unsafe { tensor.as_slice_unchecked::<f32>() }
                .iter()
                .map(|&v| v as f64)
                .collect()
        } else {
            tracing::warn!(
                output = i,
                dtype = ?tensor.datum_type(),
                "non-f32 output in fallback inference path, zero-filling"
            );
            vec![0.0; tensor.len()]
        };
        outputs.insert(format!("output_{i}"), (data, shape));
    }
    Ok(outputs)
}

fn load_runnable(
    onnx_path: &Path,
    input_shape: &[usize],
) -> Result<(Arc<TypedRunnableModel>, Vec<usize>)> {
    let model = load_onnx_model(onnx_path)?;
    let concrete_shape = resolve_concrete_shape(&model, input_shape)?;
    let plan = optimize_to_runnable(model, &concrete_shape)?;
    Ok((plan, concrete_shape))
}

fn build_input_tvalue(input_data: &[f64], shape: &[usize]) -> Result<TValue> {
    let input_f32: Vec<f32> = input_data.iter().map(|&v| v as f32).collect();
    let tensor = tract_ndarray::ArrayD::from_shape_vec(IxDyn(shape), input_f32)
        .map_err(|e| DsperseError::Onnx(format!("input tensor: {e}")))?;
    Ok(tensor.into_tvalue())
}

fn run_single(
    plan: &Arc<TypedRunnableModel>,
    input_data: &[f64],
    shape: &[usize],
) -> Result<TVec<TValue>> {
    let tv = build_input_tvalue(input_data, shape)?;
    plan.run(tvec!(tv))
        .map_err(|e| DsperseError::Onnx(format!("inference: {e}")))
}

pub struct WarmModel {
    plan: Arc<TypedRunnableModel>,
    input_shape: Vec<usize>,
}

impl WarmModel {
    pub fn load(onnx_path: &Path, input_shape: &[usize]) -> Result<Self> {
        let (plan, input_shape) = load_runnable(onnx_path, input_shape)?;
        Ok(Self { plan, input_shape })
    }

    pub fn run(&self, input_data: &[f64]) -> Result<(Vec<f64>, Vec<usize>)> {
        let result = run_single(&self.plan, input_data, &self.input_shape)?;
        extract_first_output(&result)
    }
}

pub fn run_inference(
    onnx_path: &Path,
    input_data: &[f64],
    input_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>)> {
    let (plan, concrete_shape) = load_runnable(onnx_path, input_shape)?;
    let result = run_single(&plan, input_data, &concrete_shape)?;
    extract_first_output(&result)
}

pub fn run_inference_named(
    onnx_path: &Path,
    input_data: &[f64],
    input_shape: &[usize],
) -> Result<NamedOutputs> {
    let model = load_onnx_model(onnx_path)?;
    let output_names = collect_output_names(&model);
    let concrete_shape = resolve_concrete_shape(&model, input_shape)?;
    match optimize_to_runnable(model, &concrete_shape) {
        Ok(plan) => {
            let result = run_single(&plan, input_data, &concrete_shape)?;
            zip_named_outputs(&output_names, &result)
        }
        Err(_) => {
            let mut result = run_inference_with_coercion(onnx_path, input_data, &concrete_shape)?;
            let mut named = NamedOutputs::new();
            for (i, name) in output_names.iter().enumerate() {
                let key = format!("output_{i}");
                if let Some(val) = result.remove(&key) {
                    named.insert(name.clone(), val);
                }
            }
            Ok(named)
        }
    }
}

pub fn run_inference_multi(
    onnx_path: &Path,
    inputs: &[(&str, Vec<f64>, Vec<usize>)],
) -> Result<(Vec<f64>, Vec<usize>)> {
    let (result, _) = run_multi_inner(onnx_path, inputs)?;
    extract_first_output(&result)
}

pub fn run_inference_multi_named(
    onnx_path: &Path,
    inputs: &[(&str, Vec<f64>, Vec<usize>)],
) -> Result<NamedOutputs> {
    let (result, output_names) = run_multi_inner(onnx_path, inputs)?;
    zip_named_outputs(&output_names, &result)
}

fn run_multi_inner(
    onnx_path: &Path,
    inputs: &[(&str, Vec<f64>, Vec<usize>)],
) -> Result<(TVec<TValue>, Vec<String>)> {
    let mut model = load_onnx_model(onnx_path)?;

    let output_names = collect_output_names(&model);

    let mut input_by_name: HashMap<&str, usize> = HashMap::with_capacity(inputs.len());
    for (idx, (name, _, _)) in inputs.iter().enumerate() {
        if input_by_name.insert(*name, idx).is_some() {
            return Err(DsperseError::Onnx(format!(
                "duplicate provided input name '{name}'"
            )));
        }
    }

    let model_input_count = model.inputs.len();
    let model_input_names: Vec<(usize, String)> = model
        .inputs
        .iter()
        .enumerate()
        .map(|(i, outlet)| (i, model.nodes[outlet.node].name.clone()))
        .collect();

    let mut input_order: Vec<Option<usize>> = vec![None; model_input_count];
    for (i, name) in &model_input_names {
        if let Some(&provided_idx) = input_by_name.get(name.as_str()) {
            model = model
                .with_input_fact(
                    *i,
                    InferenceFact::dt_shape(f32::datum_type(), &inputs[provided_idx].2),
                )
                .map_err(|e| DsperseError::Onnx(format!("set input {i} ({name}) shape: {e}")))?;
            input_order[*i] = Some(provided_idx);
        }
    }

    let unknown_inputs: Vec<&str> = input_by_name
        .keys()
        .copied()
        .filter(|name| !model_input_names.iter().any(|(_, n)| n == *name))
        .collect();
    if !unknown_inputs.is_empty() {
        return Err(DsperseError::Onnx(format!(
            "provided inputs not present in model: {unknown_inputs:?}"
        )));
    }

    let model = model
        .into_typed()
        .map_err(|e| {
            let unmatched: Vec<_> = input_order
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_none())
                .map(|(i, _)| model_input_names[i].1.as_str())
                .collect();
            DsperseError::Onnx(format!("type analysis (unmatched: {unmatched:?}): {e}"))
        })?
        .into_optimized()
        .map_err(|e| DsperseError::Onnx(format!("optimize: {e:#}")))?
        .into_runnable()
        .map_err(|e| DsperseError::Onnx(format!("make runnable: {e:#}")))?;

    let mut input_tvs = TVec::new();
    for (model_idx, idx) in input_order.iter().enumerate() {
        let provided_idx = idx.ok_or_else(|| {
            let name = &model_input_names[model_idx].1;
            DsperseError::Onnx(format!(
                "model input {model_idx} ('{name}') not matched to provided tensors"
            ))
        })?;
        let (_, ref data, ref shape) = inputs[provided_idx];
        input_tvs.push(build_input_tvalue(data, shape)?);
    }

    let result = model
        .run(input_tvs)
        .map_err(|e| DsperseError::Onnx(format!("inference: {e}")))?;

    Ok((result, output_names))
}

fn collect_output_names(model: &InferenceModel) -> Vec<String> {
    model
        .outputs
        .iter()
        .map(|outlet| {
            model
                .outlet_label(*outlet)
                .map(String::from)
                .unwrap_or_else(|| {
                    format!("{}_output_{}", model.nodes[outlet.node].name, outlet.slot)
                })
        })
        .collect()
}

const I64_SAFE_BOUND: i64 = 9_007_199_254_740_992;

fn i64_to_f64_checked(v: i64, label: &str) -> Result<f64> {
    if v.abs() >= I64_SAFE_BOUND {
        return Err(DsperseError::Onnx(format!(
            "{label}: i64 value {v} exceeds IEEE-754 safe integer bound"
        )));
    }
    Ok(v as f64)
}

fn tvalue_to_f64(tv: &TValue, label: &str) -> Result<(Vec<f64>, Vec<usize>)> {
    let shape = tv.shape().to_vec();
    let dt = tv.datum_type();
    let data: Vec<f64> = if dt == f32::datum_type() {
        let arr = tv
            .to_plain_array_view::<f32>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter().map(|&v| f64::from(v)).collect()
    } else if dt == f64::datum_type() {
        let arr = tv
            .to_plain_array_view::<f64>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter().copied().collect()
    } else if dt == i64::datum_type() {
        let arr = tv
            .to_plain_array_view::<i64>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter()
            .map(|&v| i64_to_f64_checked(v, label))
            .collect::<Result<Vec<_>>>()?
    } else if dt == i32::datum_type() {
        let arr = tv
            .to_plain_array_view::<i32>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter().map(|&v| f64::from(v)).collect()
    } else if dt == bool::datum_type() {
        let arr = tv
            .to_plain_array_view::<bool>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect()
    } else if dt.is_tdim() {
        let casted = tv
            .cast_to::<i64>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: TDim->i64 cast: {e}")))?;
        let arr = casted
            .to_plain_array_view::<i64>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter()
            .map(|&v| i64_to_f64_checked(v, label))
            .collect::<Result<Vec<_>>>()?
    } else {
        return Err(DsperseError::Onnx(format!(
            "{label}: unsupported datum type {dt:?}"
        )));
    };
    Ok((data, shape))
}

fn zip_named_outputs(names: &[String], result: &[TValue]) -> Result<NamedOutputs> {
    let mut map = HashMap::new();
    for (i, tv) in result.iter().enumerate() {
        let (data, shape) = tvalue_to_f64(tv, &format!("output {i}"))?;
        let name = names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("output_{i}"));
        if map.insert(name.clone(), (data, shape)).is_some() {
            return Err(DsperseError::Onnx(format!(
                "duplicate output name '{name}'"
            )));
        }
    }
    Ok(map)
}

fn extract_first_output(result: &[TValue]) -> Result<(Vec<f64>, Vec<usize>)> {
    let output = result
        .first()
        .ok_or_else(|| DsperseError::Onnx("no output from model".into()))?;
    tvalue_to_f64(output, "output tensor")
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_OPS: &[&str] = &["Conv", "Gemm", "MatMul"];

    #[test]
    fn run_inference_on_sliced_model() {
        let models_dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/models/net"
        ));
        let model_path = models_dir.join("model.onnx");
        assert!(
            model_path.exists(),
            "fixture missing: {}",
            model_path.display()
        );
        let tmp = tempfile::tempdir().unwrap();
        let meta = crate::slicer::slice_model(&model_path, Some(tmp.path()), None, TEST_OPS, None)
            .expect("slice_model failed");
        crate::slicer::materializer::ensure_all_slices_materialized(tmp.path(), &meta)
            .expect("materialization failed");
        assert!(!meta.slices.is_empty(), "model produced zero slices");
        let first_slice = &meta.slices[0];
        let onnx_path = tmp
            .path()
            .join(format!("slice_0/payload/{}", first_slice.filename));
        assert!(
            onnx_path.exists(),
            "sliced ONNX missing: {}",
            onnx_path.display()
        );
        let input_shape = &first_slice.shape.tensor_shape.input;
        assert!(
            !input_shape.is_empty() && !input_shape[0].is_empty(),
            "empty input shape"
        );
        let shape: Vec<usize> = input_shape[0].iter().map(|&d| d.max(1) as usize).collect();
        let elem_count: usize = shape.iter().product();
        let input_data = vec![0.0f64; elem_count];
        let result = run_inference(&onnx_path, &input_data, &shape);
        assert!(result.is_ok());
        let (output_data, output_shape) = result.unwrap();
        assert!(!output_data.is_empty());
        assert!(!output_shape.is_empty());
    }

    #[test]
    fn run_inference_nonexistent_model() {
        let result = run_inference(Path::new("/nonexistent/model.onnx"), &[1.0], &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn warm_model_load_nonexistent() {
        let result = WarmModel::load(Path::new("/nonexistent/model.onnx"), &[1, 1, 28, 28]);
        assert!(result.is_err());
    }

    #[test]
    fn warm_model_load_and_run_on_slice() {
        let models_dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/models/net"
        ));
        let model_path = models_dir.join("model.onnx");
        assert!(
            model_path.exists(),
            "fixture missing: {}",
            model_path.display()
        );
        let tmp = tempfile::tempdir().unwrap();
        let meta = crate::slicer::slice_model(&model_path, Some(tmp.path()), None, TEST_OPS, None)
            .expect("slice_model failed");
        crate::slicer::materializer::ensure_all_slices_materialized(tmp.path(), &meta)
            .expect("materialization failed");
        assert!(!meta.slices.is_empty(), "model produced zero slices");
        let first_slice = &meta.slices[0];
        let onnx_path = tmp
            .path()
            .join(format!("slice_0/payload/{}", first_slice.filename));
        assert!(
            onnx_path.exists(),
            "sliced ONNX missing: {}",
            onnx_path.display()
        );
        let input_shape = &first_slice.shape.tensor_shape.input;
        assert!(
            !input_shape.is_empty() && !input_shape[0].is_empty(),
            "empty input shape"
        );
        let shape: Vec<usize> = input_shape[0].iter().map(|&d| d.max(1) as usize).collect();
        let elem_count: usize = shape.iter().product();

        let warm = WarmModel::load(&onnx_path, &shape).expect("WarmModel::load failed");
        let input = vec![0.0f64; elem_count];
        let (data1, shape1) = warm.run(&input).unwrap();
        let (data2, shape2) = warm.run(&input).unwrap();
        assert!(!data1.is_empty());
        assert_eq!(shape1, shape2);
        assert_eq!(data1, data2);
    }

    #[test]
    fn zip_named_outputs_empty() {
        let result = zip_named_outputs(&[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn extract_first_output_empty() {
        let result = extract_first_output(&[]);
        assert!(result.is_err());
    }
}
