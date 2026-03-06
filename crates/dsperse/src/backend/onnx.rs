use std::collections::HashMap;
use std::path::Path;

use ndarray::IxDyn;
use tract_onnx::prelude::*;

use crate::error::{DsperseError, Result};

pub type NamedOutputs = HashMap<String, (Vec<f64>, Vec<usize>)>;

fn load_onnx_model(onnx_path: &Path) -> Result<InferenceModel> {
    tract_onnx::onnx()
        .model_for_path(onnx_path)
        .map_err(|e| DsperseError::Onnx(format!("load {}: {e}", onnx_path.display())))
}

fn resolve_concrete_shape(model: &InferenceModel, input_shape: &[usize]) -> Result<Vec<usize>> {
    if input_shape.is_empty() {
        let input_fact = model
            .input_fact(0)
            .map_err(|e| DsperseError::Onnx(format!("input fact: {e}")))?;
        input_fact
            .shape
            .as_concrete_finite()
            .map_err(|e| DsperseError::Onnx(format!("shape analysis: {e}")))?
            .ok_or_else(|| {
                DsperseError::Onnx("symbolic input shape — provide explicit shape".into())
            })
            .map(|s| s.to_vec())
    } else {
        Ok(input_shape.to_vec())
    }
}

fn optimize_to_runnable(
    model: InferenceModel,
    concrete_shape: &[usize],
) -> Result<TypedRunnableModel<TypedModel>> {
    model
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), concrete_shape),
        )
        .map_err(|e| DsperseError::Onnx(format!("set input shape: {e}")))?
        .into_optimized()
        .map_err(|e| DsperseError::Onnx(format!("optimize: {e}")))?
        .into_runnable()
        .map_err(|e| DsperseError::Onnx(format!("make runnable: {e}")))
}

fn load_runnable(
    onnx_path: &Path,
    input_shape: &[usize],
) -> Result<(TypedRunnableModel<TypedModel>, Vec<usize>)> {
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
    plan: &TypedRunnableModel<TypedModel>,
    input_data: &[f64],
    shape: &[usize],
) -> Result<TVec<TValue>> {
    let tv = build_input_tvalue(input_data, shape)?;
    plan.run(tvec!(tv))
        .map_err(|e| DsperseError::Onnx(format!("inference: {e}")))
}

pub struct WarmModel {
    plan: TypedRunnableModel<TypedModel>,
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
    let plan = optimize_to_runnable(model, &concrete_shape)?;
    let result = run_single(&plan, input_data, &concrete_shape)?;
    zip_named_outputs(&output_names, &result)
}

pub fn run_inference_multi(
    onnx_path: &Path,
    inputs: &[(&str, Vec<f64>, Vec<usize>)],
) -> Result<(Vec<f64>, Vec<usize>)> {
    let named = run_inference_multi_named(onnx_path, inputs)?;
    named
        .into_values()
        .next()
        .ok_or_else(|| DsperseError::Onnx("no outputs".into()))
}

pub fn run_inference_multi_named(
    onnx_path: &Path,
    inputs: &[(&str, Vec<f64>, Vec<usize>)],
) -> Result<NamedOutputs> {
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
        .map_err(|e| DsperseError::Onnx(format!("optimize: {e}")))?
        .into_runnable()
        .map_err(|e| DsperseError::Onnx(format!("make runnable: {e}")))?;

    let mut input_tvs = TVec::new();
    for idx in &input_order {
        let provided_idx = idx.ok_or_else(|| {
            DsperseError::Onnx("model input not matched to provided tensors".into())
        })?;
        let (_, ref data, ref shape) = inputs[provided_idx];
        input_tvs.push(build_input_tvalue(data, shape)?);
    }

    let result = model
        .run(input_tvs)
        .map_err(|e| DsperseError::Onnx(format!("inference: {e}")))?;

    zip_named_outputs(&output_names, &result)
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
            .to_array_view::<f32>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter().map(|&v| f64::from(v)).collect()
    } else if dt == f64::datum_type() {
        let arr = tv
            .to_array_view::<f64>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter().copied().collect()
    } else if dt == i64::datum_type() {
        let arr = tv
            .to_array_view::<i64>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter()
            .map(|&v| i64_to_f64_checked(v, label))
            .collect::<Result<Vec<_>>>()?
    } else if dt == i32::datum_type() {
        let arr = tv
            .to_array_view::<i32>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: {e}")))?;
        arr.iter().map(|&v| f64::from(v)).collect()
    } else if dt.is_tdim() {
        let casted = tv
            .cast_to::<i64>()
            .map_err(|e| DsperseError::Onnx(format!("{label}: TDim->i64 cast: {e}")))?;
        let arr = casted
            .to_array_view::<i64>()
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
        let meta = crate::slicer::slice_model(&model_path, Some(tmp.path()), None, TEST_OPS)
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
        let meta = crate::slicer::slice_model(&model_path, Some(tmp.path()), None, TEST_OPS)
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
