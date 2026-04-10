use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::onnx_proto::ModelProto;
use crate::error::{DsperseError, Result};

pub(crate) struct TraceResult {
    pub shapes: HashMap<String, Vec<i64>>,
    pub types: HashMap<String, i32>,
}

pub(crate) fn fold_and_trace_via_tract(
    onnx_path: &Path,
    model: &ModelProto,
) -> Result<TraceResult> {
    use tract_onnx::prelude::*;
    use tract_onnx::tract_hir::infer::InferenceSimplePlan;

    let loop_bodies = collect_loop_bodies(model);

    let tract_path = tag_all_outputs(onnx_path, model)?;
    let tract_model = std::sync::Arc::new(
        tract_onnx::onnx()
            .model_for_path(&tract_path)
            .map_err(|e| DsperseError::Slicer(format!("tract load: {e}")))?,
    );
    let _ = std::fs::remove_file(&tract_path);

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
        let tensor = if let Ok(tf) = fact.to_typed_fact() {
            let shape: Vec<usize> = tf
                .shape
                .iter()
                .map(|d| d.to_i64().unwrap_or(1).max(1) as usize)
                .collect();
            Tensor::zero_dt(tf.datum_type, &shape)
                .map_err(|e| DsperseError::Slicer(format!("zero tensor: {e}")))?
        } else {
            Tensor::zero::<f32>(&[1]).expect("scalar f32 allocation")
        };
        input_tvs.push(tensor.into_tvalue());
    }

    let shapes_cell = std::cell::RefCell::new(HashMap::<usize, Vec<Vec<i64>>>::new());
    let dtypes_cell = std::cell::RefCell::new(HashMap::<usize, Vec<u8>>::new());
    let failed_nodes = std::cell::RefCell::new(HashSet::<usize>::new());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.run_plan_with_eval(input_tvs, |session, op_state, node, inputs| {
            let tainted = node
                .inputs
                .iter()
                .any(|inp| failed_nodes.borrow().contains(&inp.node));
            let outputs = if tainted {
                failed_nodes.borrow_mut().insert(node.id);
                let fallback = inputs.first().cloned().unwrap_or_else(|| {
                    Tensor::zero::<f32>(&[1])
                        .expect("scalar f32 allocation")
                        .into_tvalue()
                });
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
                        if let Some(synth) =
                            synthesize_loop_outputs(&node.name, &inputs, &loop_bodies)
                        {
                            tracing::info!(
                                node = %node.name,
                                outputs = synth.len(),
                                "synthesized Loop output tensors from body subgraph shapes"
                            );
                            synth
                        } else {
                            tracing::warn!(
                                node = %node.name,
                                op = %node.op.name(),
                                error = %e,
                                "op eval failed, using input[0] shape as fallback"
                            );
                            failed_nodes.borrow_mut().insert(node.id);
                            let fallback = inputs.first().cloned().unwrap_or_else(|| {
                                Tensor::zero::<f32>(&[1])
                                    .expect("scalar f32 allocation")
                                    .into_tvalue()
                            });
                            let n = node.outputs.len().max(1);
                            (0..n).map(|_| fallback.clone()).collect()
                        }
                    }
                }
            };
            let node_shapes: Vec<Vec<i64>> = outputs
                .iter()
                .map(|t| t.shape().iter().map(|&d| d as i64).collect())
                .collect();
            let node_dtypes: Vec<u8> = outputs
                .iter()
                .map(|t| datum_type_to_onnx(t.datum_type()))
                .collect();
            shapes_cell.borrow_mut().insert(node.id, node_shapes);
            dtypes_cell.borrow_mut().insert(node.id, node_dtypes);
            Ok::<_, TractError>(outputs)
        })
    }));

    match &result {
        Ok(Ok(_)) => tracing::info!("tract inference run succeeded"),
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "tract inference run produced errors; partial shapes may be available")
        }
        Err(_) => {
            return Err(DsperseError::Slicer(
                "tract inference panicked; no shape data recovered".into(),
            ));
        }
    }

    let run_shapes = shapes_cell.into_inner();
    let run_dtypes = dtypes_cell.into_inner();
    let failed = failed_nodes.into_inner();

    tracing::info!(
        traced_nodes = run_shapes.len(),
        "constant folding and shape capture complete"
    );

    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    let mut types: HashMap<String, i32> = HashMap::new();
    for (node_id, node_shapes) in &run_shapes {
        if failed.contains(node_id) {
            continue;
        }
        let node_dtypes = run_dtypes.get(node_id);
        for (slot, shape) in node_shapes.iter().enumerate() {
            let dt = node_dtypes.and_then(|d| d.get(slot)).copied().unwrap_or(1) as i32; // 1 = FLOAT
            let outlet = OutletId::new(*node_id, slot);
            if let Some(label) = tract_model.outlet_label(outlet)
                && !label.is_empty()
            {
                shapes.insert(label.to_string(), shape.clone());
                types.insert(label.to_string(), dt);
            }
            let node = tract_model.node(*node_id);
            if !node.name.is_empty() {
                if slot == 0 {
                    shapes
                        .entry(node.name.clone())
                        .or_insert_with(|| shape.clone());
                    types.entry(node.name.clone()).or_insert(dt);
                }
                let qualified = format!("{}:{}", node.name, slot);
                shapes
                    .entry(qualified.clone())
                    .or_insert_with(|| shape.clone());
                types.entry(qualified).or_insert(dt);
            }
        }
    }

    if let Some(graph) = &model.graph {
        let mut extra: Vec<(String, Vec<i64>, Option<i32>)> = Vec::new();
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
                    let dt = types.get(&key).copied();
                    extra.push((out.clone(), shape.clone(), dt));
                }
            }
        }
        for (name, shape, dt) in extra {
            shapes.insert(name.clone(), shape);
            if let Some(dt) = dt {
                types.insert(name, dt);
            }
        }
        for init in &graph.initializer {
            if !init.dims.is_empty() {
                shapes
                    .entry(init.name.clone())
                    .or_insert_with(|| init.dims.clone());
            }
            if init.data_type != 0 {
                types.entry(init.name.clone()).or_insert(init.data_type);
            }
        }
        for inp in &graph.input {
            if let Some(shape) = super::onnx_shapes::shape_from_value_info(inp) {
                shapes.entry(inp.name.clone()).or_insert(shape);
            }
            if let Some(dt) = super::onnx_shapes::elem_type_from_value_info(inp) {
                types.entry(inp.name.clone()).or_insert(dt);
            }
        }
        resolve_absorbed_nodes(graph, &mut shapes);
    }

    tracing::info!(tensors = shapes.len(), "shape trace complete");
    Ok(TraceResult { shapes, types })
}

/// Save a copy of the ONNX model with every node output declared as a graph
/// output.  This forces tract to preserve outlet labels for all intermediate
/// tensors, preventing them from being lost during op fusion.
fn tag_all_outputs(onnx_path: &Path, model: &ModelProto) -> Result<std::path::PathBuf> {
    let mut tagged = model.clone();
    if let Some(ref mut graph) = tagged.graph {
        let existing: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();
        for node in &graph.node {
            for out in &node.output {
                if !out.is_empty() && !existing.contains(out) {
                    graph.output.push(super::onnx_proto::ValueInfoProto {
                        name: out.clone(),
                        ..Default::default()
                    });
                }
            }
        }
    }
    let dir = onnx_path.parent().unwrap_or_else(|| Path::new("."));
    let tagged_path = dir.join("_tract_tagged.onnx");
    super::onnx_proto::save_model(&tagged, &tagged_path)?;
    Ok(tagged_path)
}

fn onnx_elem_type_to_datum(onnx_type: i32) -> Option<tract_onnx::prelude::DatumType> {
    use tract_onnx::prelude::DatumType;
    match onnx_type {
        1 => Some(DatumType::F32),
        2 => Some(DatumType::U8),
        3 => Some(DatumType::I8),
        5 => Some(DatumType::I16),
        6 => Some(DatumType::I32),
        7 => Some(DatumType::I64),
        9 => Some(DatumType::Bool),
        10 => Some(DatumType::F16),
        11 => Some(DatumType::F64),
        12 => Some(DatumType::U32),
        13 => Some(DatumType::U64),
        _ => None,
    }
}

fn datum_type_to_onnx(dt: tract_onnx::prelude::DatumType) -> u8 {
    use tract_onnx::prelude::DatumType;
    match dt {
        DatumType::F32 => 1,
        DatumType::U8 => 2,
        DatumType::I8 => 3,
        DatumType::U16 => 4,
        DatumType::I16 => 5,
        DatumType::I32 => 6,
        DatumType::I64 => 7,
        DatumType::Bool => 9,
        DatumType::F16 => 10,
        DatumType::F64 => 11,
        DatumType::U32 => 12,
        DatumType::U64 => 13,
        _ => 1,
    }
}

struct LoopBody {
    num_loop_carried: usize,
    num_scan: usize,
    scan_body_output_shapes: Vec<Option<Vec<i64>>>,
    scan_body_output_dtypes: Vec<Option<i32>>,
}

/// Collect Loop node body metadata from the ONNX graph.  For scan outputs
/// whose shapes can be statically determined from the body subgraph, store
/// the body-side shape (without the leading trip-count dimension).
fn collect_loop_bodies(model: &ModelProto) -> HashMap<String, LoopBody> {
    let graph = match model.graph.as_ref() {
        Some(g) => g,
        None => return HashMap::new(),
    };

    let mut known: HashMap<String, Vec<i64>> = HashMap::new();
    for init in &graph.initializer {
        if !init.dims.is_empty() {
            known.insert(init.name.clone(), init.dims.clone());
        }
    }
    for vi in graph
        .input
        .iter()
        .chain(graph.value_info.iter())
        .chain(graph.output.iter())
    {
        if let Some(shape) = super::onnx_shapes::shape_from_value_info(vi) {
            known.insert(vi.name.clone(), shape);
        }
    }

    let mut result = HashMap::new();
    for node in &graph.node {
        if node.op_type != "Loop" {
            continue;
        }
        let body = match node
            .attribute
            .iter()
            .find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
        {
            Some(b) => b,
            None => continue,
        };

        let num_loop_carried = node.input.len().saturating_sub(2);
        let num_body_out = body.output.len().saturating_sub(1);
        let num_scan = num_body_out.saturating_sub(num_loop_carried);

        let mut scan_shapes = Vec::with_capacity(num_scan);
        let mut scan_dtypes = Vec::with_capacity(num_scan);
        for j in 0..num_scan {
            let body_out_idx = 1 + num_loop_carried + j;
            let body_vi = body.output.get(body_out_idx);
            let shape =
                body_vi.and_then(|vi| resolve_body_tensor_shape(&vi.name, body, graph, &known));
            let dtype = body_vi.and_then(super::onnx_shapes::elem_type_from_value_info);
            scan_shapes.push(shape);
            scan_dtypes.push(dtype);
        }

        result.insert(
            node.name.clone(),
            LoopBody {
                num_loop_carried,
                num_scan,
                scan_body_output_shapes: scan_shapes,
                scan_body_output_dtypes: scan_dtypes,
            },
        );
    }
    result
}

/// During tract evaluation, when a Loop node fails, produce correctly-shaped
/// zero tensors so downstream nodes receive valid inputs and are not tainted.
///
/// Loop-carried output shapes come directly from the actual input tensors
/// (inputs\[2..\]).  Scan output shapes come from the pre-analyzed body
/// subgraph with a leading dimension of 1 (single iteration assumption).
fn synthesize_loop_outputs(
    node_name: &str,
    inputs: &[tract_onnx::prelude::TValue],
    loop_bodies: &HashMap<String, LoopBody>,
) -> Option<tract_onnx::prelude::TVec<tract_onnx::prelude::TValue>> {
    use tract_onnx::prelude::*;

    let body = loop_bodies.get(node_name)?;
    let mut tvs: TVec<TValue> = tvec![];

    for i in 0..body.num_loop_carried {
        let init_tensor = inputs.get(i + 2)?;
        let shape: Vec<usize> = init_tensor.shape().to_vec();
        let tensor = Tensor::zero_dt(init_tensor.datum_type(), &shape).ok()?;
        tvs.push(tensor.into_tvalue());
    }

    for j in 0..body.num_scan {
        let body_shape = body.scan_body_output_shapes.get(j)?;
        let shape: Vec<usize> = match body_shape {
            Some(bs) => {
                let mut s = vec![1usize];
                s.extend(bs.iter().map(|&d| d.max(1) as usize));
                s
            }
            None => {
                tracing::warn!(
                    node = node_name,
                    scan_idx = j,
                    "scan output shape unknown, using [1,1] placeholder"
                );
                vec![1, 1]
            }
        };
        let dt = body
            .scan_body_output_dtypes
            .get(j)
            .and_then(|d| *d)
            .and_then(onnx_elem_type_to_datum)
            .unwrap_or(DatumType::F32);
        let tensor = Tensor::zero_dt(dt, &shape).ok()?;
        tvs.push(tensor.into_tvalue());
    }

    Some(tvs)
}

/// Resolve shapes for ONNX graph nodes that tract absorbed or renamed,
/// making them invisible in the tract shape output.  Iterates until no
/// more progress, using only rules already defined in the slicer module
/// (shape-preserving ops, binary broadcast).
fn resolve_absorbed_nodes(
    graph: &super::onnx_proto::GraphProto,
    shapes: &mut HashMap<String, Vec<i64>>,
) {
    let max_passes = 10;
    for _ in 0..max_passes {
        let mut progress = false;
        for node in &graph.node {
            for out in &node.output {
                if out.is_empty() || shapes.contains_key(out) {
                    continue;
                }
                let op = node.op_type.as_str();
                let shape = if super::is_shape_preserving(op) || op == "Identity" {
                    node.input.first().and_then(|inp| shapes.get(inp).cloned())
                } else if super::is_binary_arithmetic(op) {
                    let resolved: Vec<&Vec<i64>> =
                        node.input.iter().filter_map(|i| shapes.get(i)).collect();
                    let non_empty = node.input.iter().filter(|i| !i.is_empty()).count();
                    if resolved.len() == non_empty {
                        super::onnx_slicer::broadcast_shapes(&resolved)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(s) = shape {
                    shapes.insert(out.clone(), s);
                    progress = true;
                }
            }
        }
        if !progress {
            break;
        }
    }
}

fn resolve_body_tensor_shape(
    name: &str,
    body: &super::onnx_proto::GraphProto,
    outer_graph: &super::onnx_proto::GraphProto,
    known_shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    resolve_body_tensor_shape_inner(name, body, outer_graph, known_shapes, 0)
}

fn resolve_body_tensor_shape_inner(
    name: &str,
    body: &super::onnx_proto::GraphProto,
    outer_graph: &super::onnx_proto::GraphProto,
    known_shapes: &HashMap<String, Vec<i64>>,
    depth: usize,
) -> Option<Vec<i64>> {
    if depth > 32 {
        return None;
    }

    for vi in body.output.iter().chain(body.value_info.iter()) {
        if vi.name == name
            && let Some(shape) = super::onnx_shapes::shape_from_value_info(vi)
        {
            return Some(shape);
        }
    }

    for init in body
        .initializer
        .iter()
        .chain(outer_graph.initializer.iter())
    {
        if init.name == name && !init.dims.is_empty() {
            return Some(init.dims.to_vec());
        }
    }

    if let Some(shape) = known_shapes.get(name) {
        return Some(shape.clone());
    }

    let producer = body
        .node
        .iter()
        .find(|n| n.output.contains(&name.to_string()))?;
    let op = producer.op_type.as_str();

    if super::is_shape_preserving(op) || op == "Identity" {
        let inp = producer.input.first()?;
        return resolve_body_tensor_shape_inner(inp, body, outer_graph, known_shapes, depth + 1);
    }

    if super::is_binary_arithmetic(op) {
        let resolved: Vec<Vec<i64>> = producer
            .input
            .iter()
            .filter_map(|inp| {
                resolve_body_tensor_shape_inner(inp, body, outer_graph, known_shapes, depth + 1)
            })
            .collect();
        let refs: Vec<&Vec<i64>> = resolved.iter().collect();
        return super::onnx_slicer::broadcast_shapes(&refs);
    }

    if op == "Concat" {
        let axis = super::onnx_proto::get_attribute_int(producer, "axis")?;
        let input_shapes: Vec<Vec<i64>> = producer
            .input
            .iter()
            .filter_map(|inp| {
                resolve_body_tensor_shape_inner(inp, body, outer_graph, known_shapes, depth + 1)
            })
            .collect();
        if input_shapes.len() != producer.input.len() || input_shapes.is_empty() {
            return None;
        }
        let rank = input_shapes[0].len();
        let axis_idx = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };
        let mut result = input_shapes[0].clone();
        for shape in &input_shapes[1..] {
            if let Some(d) = result.get_mut(axis_idx) {
                *d += shape.get(axis_idx).copied().unwrap_or(0);
            }
        }
        return Some(result);
    }

    if op == "Transpose" {
        let inp = producer.input.first()?;
        let in_shape =
            resolve_body_tensor_shape_inner(inp, body, outer_graph, known_shapes, depth + 1)?;
        let perm = &producer.attribute.iter().find(|a| a.name == "perm")?.ints;
        let result: Vec<i64> = perm
            .iter()
            .filter_map(|&p| in_shape.get(p as usize).copied())
            .collect();
        if result.len() == in_shape.len() {
            return Some(result);
        }
    }

    None
}
