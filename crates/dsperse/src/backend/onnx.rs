use std::path::Path;

use ndarray::IxDyn;
use tract_onnx::prelude::*;

use crate::error::{DsperseError, Result};

pub fn run_inference(
    onnx_path: &Path,
    input_data: &[f64],
    input_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>)> {
    let model = tract_onnx::onnx()
        .model_for_path(onnx_path)
        .map_err(|e| DsperseError::Onnx(format!("load {}: {e}", onnx_path.display())))?;

    let concrete_shape: Vec<usize> = if input_shape.is_empty() {
        let input_fact = model
            .input_fact(0)
            .map_err(|e| DsperseError::Onnx(format!("input fact: {e}")))?;
        input_fact
            .shape
            .as_concrete_finite()
            .map_err(|e| DsperseError::Onnx(format!("shape analysis: {e}")))?
            .ok_or_else(|| {
                DsperseError::Onnx("symbolic input shape — provide explicit shape".into())
            })?
            .to_vec()
    } else {
        input_shape.to_vec()
    };

    let model = model
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), &concrete_shape),
        )
        .map_err(|e| DsperseError::Onnx(format!("set input shape: {e}")))?
        .into_optimized()
        .map_err(|e| DsperseError::Onnx(format!("optimize: {e}")))?
        .into_runnable()
        .map_err(|e| DsperseError::Onnx(format!("make runnable: {e}")))?;

    let input_f32: Vec<f32> = input_data.iter().map(|&v| v as f32).collect();
    let input_tensor = tract_ndarray::ArrayD::from_shape_vec(IxDyn(&concrete_shape), input_f32)
        .map_err(|e| DsperseError::Onnx(format!("input tensor: {e}")))?;

    let result = model
        .run(tvec!(input_tensor.into_tvalue()))
        .map_err(|e| DsperseError::Onnx(format!("inference: {e}")))?;

    extract_first_output(&result)
}

pub fn run_inference_multi(
    onnx_path: &Path,
    inputs: &[(&str, Vec<f64>, Vec<usize>)],
) -> Result<(Vec<f64>, Vec<usize>)> {
    let mut model = tract_onnx::onnx()
        .model_for_path(onnx_path)
        .map_err(|e| DsperseError::Onnx(format!("load {}: {e}", onnx_path.display())))?;

    let input_by_name: std::collections::HashMap<&str, usize> =
        inputs.iter().enumerate().map(|(idx, entry)| (entry.0, idx)).collect();

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

    let model = model
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
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let tensor = tract_ndarray::ArrayD::from_shape_vec(IxDyn(shape), f32_data)
            .map_err(|e| DsperseError::Onnx(format!("input tensor: {e}")))?;
        input_tvs.push(tensor.into_tvalue());
    }

    let result = model
        .run(input_tvs)
        .map_err(|e| DsperseError::Onnx(format!("inference: {e}")))?;

    extract_first_output(&result)
}

fn extract_first_output(result: &[TValue]) -> Result<(Vec<f64>, Vec<usize>)> {
    let output = result
        .first()
        .ok_or_else(|| DsperseError::Onnx("no output from model".into()))?;

    let output_tensor = output
        .to_array_view::<f32>()
        .map_err(|e| DsperseError::Onnx(format!("output tensor: {e}")))?;

    let output_shape = output_tensor.shape().to_vec();
    let output_data: Vec<f64> = output_tensor.iter().map(|&v| f64::from(v)).collect();

    Ok((output_data, output_shape))
}
