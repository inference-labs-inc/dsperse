use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::onnx_proto::ModelProto;
use crate::error::{DsperseError, Result};

pub(crate) fn fold_and_trace_via_tract(
    onnx_path: &Path,
    model: &mut ModelProto,
) -> Result<HashMap<String, Vec<i64>>> {
    use tract_onnx::prelude::*;
    use tract_onnx::tract_hir::infer::InferenceSimplePlan;

    let tract_model = std::sync::Arc::new(
        tract_onnx::onnx()
            .model_for_path(onnx_path)
            .map_err(|e| DsperseError::Slicer(format!("tract load: {e}")))?,
    );

    let plan = InferenceSimplePlan::new(tract_model.clone())
        .map_err(|e| DsperseError::Slicer(format!("plan creation: {e}")))?;

    let mut state = tract_onnx::tract_core::plan::SimpleState::new(&plan)
        .map_err(|e| DsperseError::Slicer(format!("state creation: {e}")))?;

    let mut input_tvs: TVec<TValue> = tvec![];
    for outlet in tract_model
        .input_outlets()
        .map_err(|e| DsperseError::Slicer(format!("input outlets: {e}")))?
    {
        let fact = tract_model
            .outlet_fact(*outlet)
            .map_err(|e| DsperseError::Slicer(format!("input fact: {e}")))?;
        if let Ok(tf) = fact.to_typed_fact() {
            let shape: Vec<usize> = tf
                .shape
                .iter()
                .map(|d| d.to_i64().unwrap_or(1).max(1) as usize)
                .collect();
            let tensor = Tensor::zero_dt(tf.datum_type, &shape)
                .map_err(|e| DsperseError::Slicer(format!("zero tensor: {e}")))?;
            input_tvs.push(tensor.into_tvalue());
        }
    }

    let shapes_cell = std::cell::RefCell::new(HashMap::<usize, Vec<Vec<i64>>>::new());
    let failed_nodes = std::cell::RefCell::new(HashSet::<usize>::new());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.run_plan_with_eval(input_tvs, |session, op_state, node, inputs| {
            let tainted = node
                .inputs
                .iter()
                .any(|inp| failed_nodes.borrow().contains(&inp.node));
            let outputs = if tainted {
                failed_nodes.borrow_mut().insert(node.id);
                let fallback = inputs
                    .first()
                    .cloned()
                    .unwrap_or_else(|| Tensor::zero::<f32>(&[1]).unwrap().into_tvalue());
                let n = node.outputs.len().max(1);
                (0..n).map(|_| fallback.clone()).collect()
            } else {
                let coerced = crate::backend::onnx::coerce_tdim_inputs(&inputs);
                let eval_result = if let Some(st) = op_state {
                    st.eval(session, node.op.as_op(), coerced)
                } else {
                    node.op.eval(coerced)
                };
                match eval_result {
                    Ok(o) => o,
                    Err(e) => {
                        tracing::warn!(
                            node = %node.name,
                            op = %node.op.name(),
                            error = %e,
                            "op eval failed, using input[0] shape as fallback"
                        );
                        failed_nodes.borrow_mut().insert(node.id);
                        let fallback = inputs
                            .first()
                            .cloned()
                            .unwrap_or_else(|| Tensor::zero::<f32>(&[1]).unwrap().into_tvalue());
                        let n = node.outputs.len().max(1);
                        (0..n).map(|_| fallback.clone()).collect()
                    }
                }
            };
            let node_shapes: Vec<Vec<i64>> = outputs
                .iter()
                .map(|t| t.shape().iter().map(|&d| d as i64).collect())
                .collect();
            shapes_cell.borrow_mut().insert(node.id, node_shapes);
            Ok::<_, TractError>(outputs)
        })
    }));

    match &result {
        Ok(Ok(_)) => tracing::info!("tract inference run succeeded"),
        Ok(Err(e)) => tracing::warn!(error = %e, "tract inference run failed"),
        Err(_) => tracing::warn!("tract inference run panicked"),
    }

    let run_shapes = shapes_cell.into_inner();
    let failed = failed_nodes.into_inner();

    tracing::info!(
        traced_nodes = run_shapes.len(),
        "constant folding and shape capture complete"
    );

    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    for (node_id, node_shapes) in &run_shapes {
        if failed.contains(node_id) {
            continue;
        }
        for (slot, shape) in node_shapes.iter().enumerate() {
            let outlet = OutletId::new(*node_id, slot);
            if let Some(label) = tract_model.outlet_label(outlet)
                && !label.is_empty()
            {
                shapes.insert(label.to_string(), shape.clone());
            }
            let node = tract_model.node(*node_id);
            let name = if slot == 0 && !node.name.is_empty() {
                node.name.clone()
            } else {
                format!("{}:{}", node.name, slot)
            };
            shapes.entry(name).or_insert_with(|| shape.clone());
        }
    }

    if let Some(graph) = &model.graph {
        let mut extra: Vec<(String, Vec<i64>)> = Vec::new();
        for n in &graph.node {
            for (slot, out) in n.output.iter().enumerate() {
                if out.is_empty() || shapes.contains_key(out) {
                    continue;
                }
                let key = if slot == 0 {
                    n.name.clone()
                } else {
                    format!("{}:{}", n.name, slot)
                };
                if let Some(shape) = shapes.get(&key) {
                    extra.push((out.clone(), shape.clone()));
                }
            }
        }
        for (name, shape) in extra {
            shapes.insert(name, shape);
        }
        for init in &graph.initializer {
            if !init.dims.is_empty() {
                shapes
                    .entry(init.name.clone())
                    .or_insert_with(|| init.dims.clone());
            }
        }
    }

    tracing::info!(tensors = shapes.len(), "shape trace complete");
    Ok(shapes)
}
