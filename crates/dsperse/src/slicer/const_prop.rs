#[allow(dead_code)]
use std::collections::{HashMap, HashSet};

use super::onnx_proto::{self, ModelProto, NodeProto, TensorProto};

#[derive(Clone, Debug)]
enum ConstVal {
    F32(Vec<f32>, Vec<i64>, i32),
    I64(Vec<i64>, Vec<i64>, i32),
    ShapeOnly(Vec<i64>),
}

impl ConstVal {
    fn from_tensor(t: &TensorProto) -> Option<Self> {
        let dims = t.dims.clone();
        let dt = t.data_type;
        if let Some(v) = Self::extract_f32(t) {
            return Some(Self::F32(v, dims, dt));
        }
        if let Some(v) = Self::extract_i64(t) {
            return Some(Self::I64(v, dims, dt));
        }
        if !dims.is_empty() {
            if dims.contains(&0) {
                return if dt == TensorProto::FLOAT || dt == TensorProto::DOUBLE {
                    Some(Self::F32(vec![], dims, dt))
                } else {
                    Some(Self::I64(vec![], dims, dt))
                };
            }
            return Some(Self::ShapeOnly(dims));
        }
        None
    }

    fn into_tensor(self, name: &str) -> Option<TensorProto> {
        match self {
            Self::F32(data, dims, dt) => Some(TensorProto {
                name: name.to_string(),
                data_type: dt,
                dims,
                float_data: data,
                ..Default::default()
            }),
            Self::I64(data, dims, dt) => {
                let mut tensor = TensorProto {
                    name: name.to_string(),
                    data_type: dt,
                    dims,
                    ..Default::default()
                };
                if dt == TensorProto::INT32 {
                    tensor.int32_data = data.iter().map(|&v| v as i32).collect();
                } else {
                    tensor.int64_data = data;
                }
                Some(tensor)
            }
            Self::ShapeOnly(_) => None,
        }
    }

    fn dims(&self) -> &[i64] {
        match self {
            Self::F32(_, d, _) | Self::I64(_, d, _) | Self::ShapeOnly(d) => d,
        }
    }

    fn as_f32(&self) -> Option<Vec<f32>> {
        match self {
            Self::F32(v, _, _) => Some(v.clone()),
            Self::I64(v, _, _) => Some(v.iter().map(|&x| x as f32).collect()),
            Self::ShapeOnly(_) => None,
        }
    }

    fn as_i64(&self) -> Option<Vec<i64>> {
        match self {
            Self::I64(v, _, _) => Some(v.clone()),
            _ => None,
        }
    }

    fn extract_f32(t: &TensorProto) -> Option<Vec<f32>> {
        if !t.float_data.is_empty() {
            return Some(t.float_data.clone());
        }
        if !t.raw_data.is_empty() && t.data_type == TensorProto::FLOAT {
            return Some(
                t.raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
        }
        None
    }

    fn extract_i64(t: &TensorProto) -> Option<Vec<i64>> {
        if !t.int64_data.is_empty() {
            return Some(t.int64_data.clone());
        }
        if !t.raw_data.is_empty() && t.data_type == TensorProto::INT64 {
            return Some(
                t.raw_data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            );
        }
        if !t.int32_data.is_empty() {
            return Some(t.int32_data.iter().map(|&v| i64::from(v)).collect());
        }
        if !t.raw_data.is_empty() && t.data_type == TensorProto::INT32 {
            return Some(
                t.raw_data
                    .chunks_exact(4)
                    .map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
            );
        }
        None
    }
}

pub fn propagate_constants(model: &mut ModelProto) -> HashSet<String> {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return HashSet::new(),
    };

    let mut known: HashMap<String, ConstVal> = HashMap::new();
    for init in &graph.initializer {
        if let Some(val) = ConstVal::from_tensor(init) {
            known.insert(init.name.clone(), val);
        }
    }
    for vi in graph.input.iter().chain(graph.value_info.iter()) {
        if known.contains_key(&vi.name) {
            continue;
        }
        if let Some(shape) = super::onnx_proto::shape_from_value_info(vi)
            && !shape.is_empty()
            && shape.iter().all(|&d| d > 0)
        {
            known.insert(vi.name.clone(), ConstVal::ShapeOnly(shape));
        }
    }

    let mut evaluated: HashSet<usize> = HashSet::new();
    let mut progress = true;
    while progress {
        progress = false;
        for (idx, node) in graph.node.iter().enumerate() {
            if evaluated.contains(&idx) {
                continue;
            }
            if can_evaluate(node, &known) {
                let inputs: Vec<Option<&ConstVal>> = node
                    .input
                    .iter()
                    .map(|inp| if inp.is_empty() { None } else { known.get(inp) })
                    .collect();
                if let Some(outputs) = evaluate(node, &inputs) {
                    for (name, val) in outputs {
                        known.insert(name, val);
                    }
                    evaluated.insert(idx);
                    progress = true;
                    continue;
                }
            }
            if let Some(out) = node.output.first()
                && !out.is_empty()
                && !known.contains_key(out)
                && node.input.iter().all(|inp| {
                    inp.is_empty()
                        || matches!(
                            known.get(inp),
                            Some(ConstVal::I64(_, _, _) | ConstVal::F32(_, _, _))
                        )
                })
                && let Some(shape) = infer_output_shape(node, &known, &graph.initializer)
            {
                known.insert(out.clone(), ConstVal::ShapeOnly(shape));
                progress = true;
            }
        }
    }

    if evaluated.is_empty() {
        return HashSet::new();
    }

    let existing: HashSet<String> = graph.initializer.iter().map(|i| i.name.clone()).collect();
    let mut new_inits: Vec<TensorProto> = Vec::new();
    let mut propagated_names: HashSet<String> = HashSet::new();
    for idx in &evaluated {
        for out in &graph.node[*idx].output {
            if !out.is_empty()
                && !existing.contains(out)
                && let Some(val) = known.get(out)
                && let Some(tensor) = val.clone().into_tensor(out)
            {
                propagated_names.insert(out.clone());
                new_inits.push(tensor);
            }
        }
    }

    let mut indices: Vec<usize> = evaluated.into_iter().collect();
    indices.sort_unstable_by(|a, b| b.cmp(a));
    for idx in indices {
        graph.node.remove(idx);
    }

    let count = new_inits.len();
    graph.initializer.extend(new_inits);

    if count > 0 {
        tracing::info!(count, "propagated constant subgraphs into initializers");
    }
    propagated_names
}

pub fn fill_shapes_from_graph(
    graph: &super::onnx_proto::GraphProto,
    shapes: &mut HashMap<String, Vec<i64>>,
) {
    let mut known: HashMap<String, ConstVal> = HashMap::new();
    for (name, shape) in shapes.iter() {
        known.insert(name.clone(), ConstVal::ShapeOnly(shape.clone()));
    }
    for init in &graph.initializer {
        if let Some(val) = ConstVal::from_tensor(init) {
            known.insert(init.name.clone(), val);
        } else if !init.dims.is_empty() {
            known
                .entry(init.name.clone())
                .or_insert_with(|| ConstVal::ShapeOnly(init.dims.clone()));
        }
    }

    let before = shapes.len();
    let mut progress = true;
    while progress {
        progress = false;
        for node in &graph.node {
            let all_resolved = node.output.iter().all(|o| {
                o.is_empty()
                    || matches!(
                        known.get(o),
                        Some(ConstVal::I64(_, _, _) | ConstVal::F32(_, _, _))
                    )
            });
            if all_resolved {
                continue;
            }
            if can_evaluate(node, &known) {
                let inputs: Vec<Option<&ConstVal>> = node
                    .input
                    .iter()
                    .map(|inp| if inp.is_empty() { None } else { known.get(inp) })
                    .collect();
                if let Some(outputs) = evaluate(node, &inputs) {
                    for (name, val) in outputs {
                        if let Some(shape) = get_shape(
                            &name,
                            &std::iter::once((name.clone(), val.clone())).collect(),
                        ) && shape.iter().all(|&d| d > 0)
                        {
                            shapes.entry(name.clone()).or_insert(shape);
                        }
                        known.insert(name, val);
                        progress = true;
                    }
                    continue;
                }
            }
            for out in &node.output {
                if out.is_empty() || known.contains_key(out) {
                    continue;
                }
                if let Some(shape) = infer_output_shape(node, &known, &graph.initializer)
                    && shape.iter().all(|&d| d >= 0)
                {
                    known.insert(out.clone(), ConstVal::ShapeOnly(shape.clone()));
                    shapes.insert(out.clone(), shape);
                    progress = true;
                }
            }
        }
    }
    let added = shapes.len() - before;
    if added > 0 {
        tracing::info!(
            added,
            "filled additional shapes from post-propagation graph"
        );
    }
}

fn get_shape(name: &str, known: &HashMap<String, ConstVal>) -> Option<Vec<i64>> {
    match known.get(name)? {
        ConstVal::F32(_, dims, _) | ConstVal::I64(_, dims, _) => Some(dims.clone()),
        ConstVal::ShapeOnly(s) => Some(s.clone()),
    }
}

fn infer_output_shape(
    node: &NodeProto,
    known: &HashMap<String, ConstVal>,
    initializers: &[TensorProto],
) -> Option<Vec<i64>> {
    let input_shape =
        |idx: usize| -> Option<Vec<i64>> { node.input.get(idx).and_then(|n| get_shape(n, known)) };

    match node.op_type.as_str() {
        "Conv" => {
            let x = input_shape(0)?;
            if x.len() != 4 {
                return None;
            }
            let w_name = node.input.get(1)?;
            let w_dims = initializers
                .iter()
                .find(|i| &i.name == w_name)
                .map(|i| &i.dims)
                .or_else(|| {
                    known.get(w_name).map(|v| match v {
                        ConstVal::F32(_, d, _)
                        | ConstVal::I64(_, d, _)
                        | ConstVal::ShapeOnly(d) => d,
                    })
                })?;
            if w_dims.len() != 4 {
                return None;
            }
            let strides = onnx_proto::get_attribute_ints(node, "strides").unwrap_or_default();
            let pads = onnx_proto::get_attribute_ints(node, "pads").unwrap_or_default();
            let dilations = onnx_proto::get_attribute_ints(node, "dilations").unwrap_or_default();
            let sh = strides.first().copied().unwrap_or(1);
            let sw = strides.get(1).copied().unwrap_or(1);
            let ph = if pads.len() >= 4 {
                pads[0] + pads[2]
            } else {
                0
            };
            let pw = if pads.len() >= 4 {
                pads[1] + pads[3]
            } else {
                0
            };
            let dh = dilations.first().copied().unwrap_or(1);
            let dw = dilations.get(1).copied().unwrap_or(1);
            let kh = (w_dims[2] - 1) * dh + 1;
            let kw = (w_dims[3] - 1) * dw + 1;
            Some(vec![
                x[0],
                w_dims[0],
                (x[2] + ph - kh) / sh + 1,
                (x[3] + pw - kw) / sw + 1,
            ])
        }
        "Transpose" => {
            let x = input_shape(0)?;
            let perm = onnx_proto::get_attribute_ints(node, "perm")?;
            if perm.len() != x.len() {
                return None;
            }
            Some(perm.iter().map(|&p| x[p as usize]).collect())
        }
        "MatMul" => {
            let a = input_shape(0)?;
            let b = input_shape(1)?;
            if a.len() < 2 || b.len() < 2 {
                return None;
            }
            let mut out = a[..a.len() - 1].to_vec();
            out.push(*b.last().unwrap());
            Some(out)
        }
        "Gemm" => {
            let a = input_shape(0)?;
            let b = input_shape(1)?;
            if a.len() != 2 || b.len() != 2 {
                return None;
            }
            let trans_a = onnx_proto::get_attribute_int(node, "transA").unwrap_or(0);
            let trans_b = onnx_proto::get_attribute_int(node, "transB").unwrap_or(0);
            let m = if trans_a != 0 { a[1] } else { a[0] };
            let n = if trans_b != 0 { b[0] } else { b[1] };
            Some(vec![m, n])
        }
        "Concat" => {
            let axis = onnx_proto::get_attribute_int(node, "axis")?;
            let shapes: Vec<Vec<i64>> = node
                .input
                .iter()
                .filter_map(|n| get_shape(n, known))
                .collect();
            if shapes.is_empty() || shapes.len() != node.input.len() {
                return None;
            }
            let rank = shapes[0].len();
            if rank == 0 || shapes.iter().any(|s| s.len() != rank) {
                return None;
            }
            let axis = if axis < 0 { rank as i64 + axis } else { axis } as usize;
            if axis >= rank {
                return None;
            }
            let mut out = shapes[0].clone();
            for s in &shapes[1..] {
                out[axis] += s[axis];
            }
            Some(out)
        }
        "Reshape" => {
            let x = input_shape(0)?;
            let target_name = node.input.get(1)?;
            let target = match known.get(target_name)? {
                ConstVal::I64(v, _, _) => v.clone(),
                _ => return None,
            };
            let vol: i64 = x.iter().product();
            let mut out = target;
            let neg_idx = out.iter().position(|&v| v == -1);
            let known_vol: i64 = out.iter().filter(|&&v| v != -1).product();
            if let Some(idx) = neg_idx {
                out[idx] = if known_vol != 0 { vol / known_vol } else { 0 };
            }
            for (i, d) in out.iter_mut().enumerate() {
                if *d == 0 && i < x.len() {
                    *d = x[i];
                }
            }
            Some(out)
        }
        "Resize" => {
            let x = input_shape(0)?;
            if let Some(name) = node.input.get(3)
                && let Some(ConstVal::I64(sizes, _, _)) = known.get(name)
            {
                return Some(sizes.clone());
            }
            if let Some(name) = node.input.get(2)
                && let Some(ConstVal::F32(scales, _, _)) = known.get(name)
                && scales.len() == x.len()
            {
                return Some(
                    x.iter()
                        .zip(scales.iter())
                        .map(|(&d, &s)| (d as f32 * s).floor() as i64)
                        .collect(),
                );
            }
            None
        }
        "Slice" => {
            let x = input_shape(0)?;
            let get_i64 = |idx: usize| -> Option<Vec<i64>> {
                let name = node.input.get(idx)?;
                match known.get(name)? {
                    ConstVal::I64(v, _, _) => Some(v.clone()),
                    ConstVal::F32(v, _, _) => Some(v.iter().map(|&f| f as i64).collect()),
                    ConstVal::ShapeOnly(_) => None,
                }
            };
            let starts = get_i64(1)?;
            let ends = get_i64(2)?;
            let axes = get_i64(3);
            let steps = get_i64(4);
            let rank = x.len() as i64;
            let mut out = x.clone();
            for i in 0..starts.len() {
                let axis = axes
                    .as_ref()
                    .and_then(|a| a.get(i).copied())
                    .unwrap_or(i as i64);
                let axis = if axis < 0 { rank + axis } else { axis } as usize;
                if axis >= out.len() {
                    continue;
                }
                let dim = out[axis];
                let step = steps.as_ref().and_then(|s| s.get(i).copied()).unwrap_or(1);
                let mut s = starts[i];
                let mut e = ends[i];
                if s < 0 {
                    s += dim;
                }
                if e < 0 {
                    e += dim;
                }
                s = s.clamp(0, dim);
                e = e.clamp(0, dim);
                let len = if step > 0 {
                    (e - s + step - 1) / step
                } else if step < 0 {
                    (s - e + (-step) - 1) / (-step)
                } else {
                    return None;
                };
                out[axis] = len.max(0);
            }
            Some(out)
        }
        "Tile" => {
            let x = input_shape(0)?;
            let repeats_name = node.input.get(1)?;
            let repeats = match known.get(repeats_name)? {
                ConstVal::I64(v, _, _) => v.clone(),
                _ => return None,
            };
            if repeats.len() != x.len() {
                return None;
            }
            Some(x.iter().zip(repeats.iter()).map(|(&d, &r)| d * r).collect())
        }
        "ConstantOfShape" => {
            let shape_name = node.input.first()?;
            let dims = match known.get(shape_name)? {
                ConstVal::I64(v, _, _) => v.clone(),
                _ => return None,
            };
            Some(dims)
        }
        "LayerNormalization" | "BatchNormalization" | "ScatterND" | "ScatterElements"
        | "GatherElements" => input_shape(0),
        "ReduceMean" | "ReduceMax" | "ReduceMin" | "ReduceSum" | "ReduceProd" => {
            let x = input_shape(0)?;
            let axes = onnx_proto::get_attribute_ints(node, "axes").or_else(|| {
                node.input.get(1).and_then(|n| match known.get(n)? {
                    ConstVal::I64(v, _, _) => Some(v.clone()),
                    _ => None,
                })
            })?;
            let keepdims = onnx_proto::get_attribute_int(node, "keepdims").unwrap_or(1) != 0;
            let rank = x.len() as i64;
            let norm: HashSet<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (rank + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            if keepdims {
                Some(
                    x.iter()
                        .enumerate()
                        .map(|(i, &d)| if norm.contains(&i) { 1 } else { d })
                        .collect(),
                )
            } else {
                Some(
                    x.iter()
                        .enumerate()
                        .filter(|(i, _)| !norm.contains(i))
                        .map(|(_, &d)| d)
                        .collect(),
                )
            }
        }
        "TopK" => {
            let x = input_shape(0)?;
            let k_name = node.input.get(1)?;
            let k = match known.get(k_name)? {
                ConstVal::I64(v, _, _) => *v.first()?,
                _ => return None,
            };
            let axis = onnx_proto::get_attribute_int(node, "axis").unwrap_or(-1);
            let axis = if axis < 0 {
                x.len() as i64 + axis
            } else {
                axis
            } as usize;
            let mut out = x;
            if axis < out.len() {
                out[axis] = k;
            }
            Some(out)
        }
        "Expand" => {
            let x = input_shape(0)?;
            let target_name = node.input.get(1)?;
            let target_val = known.get(target_name)?;
            let target = match target_val {
                ConstVal::I64(v, _, _) => v.clone(),
                ConstVal::ShapeOnly(s) => s.clone(),
                _ => return None,
            };
            let rank = x.len().max(target.len());
            let mut out = vec![1i64; rank];
            for (i, d) in out.iter_mut().enumerate() {
                let xi = x
                    .get(i + x.len().saturating_sub(rank))
                    .copied()
                    .unwrap_or(1);
                let ti = target
                    .get(i + target.len().saturating_sub(rank))
                    .copied()
                    .unwrap_or(1);
                *d = xi.max(ti);
            }
            Some(out)
        }
        "Range" => {
            let to_f64 = |idx: usize| -> Option<f64> {
                let name = node.input.get(idx)?;
                match known.get(name)? {
                    ConstVal::I64(v, _, _) => v.first().map(|&x| x as f64),
                    ConstVal::F32(v, _, _) => v.first().map(|&x| x as f64),
                    _ => None,
                }
            };
            let start = to_f64(0)?;
            let limit = to_f64(1)?;
            let delta = to_f64(2)?;
            if delta == 0.0 {
                return None;
            }
            let len = ((limit - start) / delta).ceil().max(0.0) as i64;
            Some(vec![len])
        }
        "Where" => {
            let cond = input_shape(0)?;
            let x = input_shape(1)?;
            let y = input_shape(2)?;
            let rank = cond.len().max(x.len()).max(y.len());
            let mut out = vec![1i64; rank];
            for (i, d) in out.iter_mut().enumerate() {
                let ci = cond
                    .get(i + cond.len().saturating_sub(rank))
                    .copied()
                    .unwrap_or(1);
                let xi = x
                    .get(i + x.len().saturating_sub(rank))
                    .copied()
                    .unwrap_or(1);
                let yi = y
                    .get(i + y.len().saturating_sub(rank))
                    .copied()
                    .unwrap_or(1);
                *d = ci.max(xi).max(yi);
            }
            Some(out)
        }
        "Flatten" => {
            let x = input_shape(0)?;
            let axis = onnx_proto::get_attribute_int(node, "axis").unwrap_or(1) as usize;
            let d0: i64 = x[..axis].iter().product();
            let d1: i64 = x[axis..].iter().product();
            Some(vec![d0, d1])
        }
        "Unsqueeze" => {
            let x = input_shape(0)?;
            let axes_name = node.input.get(1)?;
            let axes = match known.get(axes_name)? {
                ConstVal::I64(v, _, _) => v.clone(),
                _ => return None,
            };
            let new_rank = x.len() + axes.len();
            let mut out = vec![0i64; new_rank];
            let normalized: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (new_rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            for &a in &normalized {
                if a < new_rank {
                    out[a] = 1;
                }
            }
            let mut xi = 0;
            for d in &mut out {
                if *d == 0 && xi < x.len() {
                    *d = x[xi];
                    xi += 1;
                }
            }
            Some(out)
        }
        "Squeeze" => {
            let x = input_shape(0)?;
            let axes = if let Some(axes_name) = node.input.get(1) {
                match known.get(axes_name)? {
                    ConstVal::I64(v, _, _) => v.clone(),
                    _ => return None,
                }
            } else {
                x.iter()
                    .enumerate()
                    .filter(|&(_, &d)| d == 1)
                    .map(|(i, _)| i as i64)
                    .collect()
            };
            let rank = x.len() as i64;
            let normalized: HashSet<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (rank + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            Some(
                x.iter()
                    .enumerate()
                    .filter(|(i, _)| !normalized.contains(i))
                    .map(|(_, &d)| d)
                    .collect(),
            )
        }
        op if super::is_shape_preserving(op) || super::is_elementwise(op) => input_shape(0),
        op if super::is_binary_arithmetic(op) => {
            let a = input_shape(0)?;
            let b = input_shape(1)?;
            let rank = a.len().max(b.len());
            let mut out = vec![1i64; rank];
            for (i, d) in out.iter_mut().enumerate().rev() {
                let ai = if i >= rank - a.len() {
                    a[i - (rank - a.len())]
                } else {
                    1
                };
                let bi = if i >= rank - b.len() {
                    b[i - (rank - b.len())]
                } else {
                    1
                };
                *d = ai.max(bi);
            }
            Some(out)
        }
        _ => None,
    }
}

fn can_evaluate_strict(node: &NodeProto, known: &HashMap<String, ConstVal>) -> bool {
    if node.output.is_empty() || node.output.iter().all(|o| o.is_empty()) {
        return false;
    }
    if node
        .output
        .iter()
        .filter(|o| !o.is_empty())
        .all(|o| matches!(known.get(o), Some(v) if !matches!(v, ConstVal::ShapeOnly(_))))
    {
        return false;
    }
    node.input.iter().all(|inp| {
        inp.is_empty()
            || matches!(
                known.get(inp),
                Some(ConstVal::I64(_, _, _) | ConstVal::F32(_, _, _))
            )
    })
}

fn can_evaluate(node: &NodeProto, known: &HashMap<String, ConstVal>) -> bool {
    if node.output.is_empty() || node.output.iter().all(|o| o.is_empty()) {
        return false;
    }
    if node
        .output
        .iter()
        .filter(|o| !o.is_empty())
        .all(|o| matches!(known.get(o), Some(v) if !matches!(v, ConstVal::ShapeOnly(_))))
    {
        return false;
    }
    node.input
        .iter()
        .all(|inp| inp.is_empty() || known.contains_key(inp))
}

fn evaluate(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<Vec<(String, ConstVal)>> {
    let out = out_name(node, 0)?;
    let result = match node.op_type.as_str() {
        "Shape" => eval_shape(node, inputs),
        "Gather" => eval_gather(node, inputs),
        "Slice" => eval_slice(inputs),
        "Cast" => eval_cast(node, inputs),
        "Sqrt" => map_f32(inputs, |v| v.sqrt()),
        "Neg" => map_f32(inputs, |v| -v),
        "Exp" => map_f32(inputs, |v| v.exp()),
        "Floor" => map_f32(inputs, |v| v.floor()),
        "Ceil" => map_f32(inputs, |v| v.ceil()),
        "Add" => binary_f32(inputs, |a, b| a + b),
        "Sub" => binary_f32(inputs, |a, b| a - b),
        "Mul" => binary_f32(inputs, |a, b| a * b),
        "Div" => binary_f32(inputs, |a, b| a / b),
        "Pow" => binary_f32(inputs, |a, b| a.powf(b)),
        "Unsqueeze" => eval_unsqueeze(node, inputs),
        "Squeeze" => eval_squeeze(node, inputs),
        "Concat" => eval_concat(node, inputs),
        "Reshape" => eval_reshape(node, inputs),
        "ConstantOfShape" => eval_constant_of_shape(node, inputs),
        "Split" => return eval_split(node, inputs),
        _ => None,
    }?;
    Some(vec![(out.to_string(), result)])
}

fn out_name(node: &NodeProto, idx: usize) -> Option<&str> {
    node.output
        .get(idx)
        .map(String::as_str)
        .filter(|s| !s.is_empty())
}

fn attr_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

fn attr_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

fn inp<'a>(inputs: &'a [Option<&'a ConstVal>], idx: usize) -> Option<&'a ConstVal> {
    inputs.get(idx)?.as_ref().copied()
}

fn normalize_idx(val: i64, len: i64) -> usize {
    (if val < 0 {
        (len + val).max(0)
    } else {
        val.min(len)
    }) as usize
}

fn broadcast_dims(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let rank = a.len().max(b.len());
    let mut result = vec![1i64; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        if da == db {
            result[i] = da;
        } else if da == 1 {
            result[i] = db;
        } else if db == 1 {
            result[i] = da;
        } else {
            return None;
        }
    }
    Some(result)
}

fn eval_shape(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let rank = t.dims().len() as i64;
    let s = normalize_idx(attr_int(node, "start").unwrap_or(0), rank);
    let e = normalize_idx(attr_int(node, "end").unwrap_or(rank), rank);
    let vals: Vec<i64> = if s <= e {
        t.dims()[s..e].to_vec()
    } else {
        vec![]
    };
    Some(ConstVal::I64(
        vals.clone(),
        vec![vals.len() as i64],
        TensorProto::INT64,
    ))
}

fn eval_gather(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let data_val = inp(inputs, 0)?;
    let axis = attr_int(node, "axis").unwrap_or(0);
    if axis != 0 || data_val.dims().len() != 1 {
        return None;
    }
    let indices = inp(inputs, 1)?.as_i64()?;
    let dims = inp(inputs, 1)?.dims().to_vec();
    match data_val {
        ConstVal::I64(data, _, dt) => {
            let len = data.len() as i64;
            let result: Option<Vec<i64>> = indices
                .iter()
                .map(|&i| {
                    let idx = if i < 0 { len + i } else { i };
                    if idx < 0 || idx >= len {
                        None
                    } else {
                        data.get(idx as usize).copied()
                    }
                })
                .collect();
            Some(ConstVal::I64(result?, dims, *dt))
        }
        ConstVal::F32(data, _, dt) => {
            let len = data.len() as i64;
            let result: Option<Vec<f32>> = indices
                .iter()
                .map(|&i| {
                    let idx = if i < 0 { len + i } else { i };
                    if idx < 0 || idx >= len {
                        None
                    } else {
                        data.get(idx as usize).copied()
                    }
                })
                .collect();
            Some(ConstVal::F32(result?, dims, *dt))
        }
        ConstVal::ShapeOnly(_) => None,
    }
}

fn eval_slice(inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let data = inp(inputs, 0)?;
    if data.dims().len() != 1 {
        return None;
    }
    let starts = inp(inputs, 1)?.as_i64()?;
    let ends = inp(inputs, 2)?.as_i64()?;

    if let Some(axes) = inputs.get(3).and_then(|o| o.as_ref()) {
        let ax = axes.as_i64()?;
        if ax.len() != 1 || ax[0] != 0 {
            return None;
        }
    }
    if let Some(steps) = inputs.get(4).and_then(|o| o.as_ref()) {
        let st = steps.as_i64()?;
        if st.iter().any(|&s| s != 1) {
            return None;
        }
    }

    match data {
        ConstVal::I64(vals, _, dt) => {
            let len = vals.len() as i64;
            let s = normalize_idx(*starts.first()?, len);
            let e = normalize_idx(*ends.first()?, len);
            let result: Vec<i64> = if s <= e { vals[s..e].to_vec() } else { vec![] };
            Some(ConstVal::I64(
                result.clone(),
                vec![result.len() as i64],
                *dt,
            ))
        }
        ConstVal::F32(vals, _, dt) => {
            let len = vals.len() as i64;
            let s = normalize_idx(*starts.first()?, len);
            let e = normalize_idx(*ends.first()?, len);
            let result: Vec<f32> = if s <= e { vals[s..e].to_vec() } else { vec![] };
            Some(ConstVal::F32(
                result.clone(),
                vec![result.len() as i64],
                *dt,
            ))
        }
        ConstVal::ShapeOnly(_) => None,
    }
}

fn eval_cast(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let to = attr_int(node, "to")? as i32;
    let dims = t.dims().to_vec();
    if to == TensorProto::FLOAT {
        match t {
            ConstVal::F32(v, _, dt) => Some(ConstVal::F32(v.clone(), dims, *dt)),
            ConstVal::I64(v, _, _) => Some(ConstVal::F32(
                v.iter().map(|&i| i as f32).collect(),
                dims,
                TensorProto::FLOAT,
            )),
            ConstVal::ShapeOnly(_) => None,
        }
    } else if to == TensorProto::INT64 {
        match t {
            ConstVal::I64(v, _, _) => Some(ConstVal::I64(v.clone(), dims, TensorProto::INT64)),
            ConstVal::F32(v, _, _) => Some(ConstVal::I64(
                v.iter().map(|&f| f as i64).collect(),
                dims,
                TensorProto::INT64,
            )),
            ConstVal::ShapeOnly(_) => None,
        }
    } else {
        None
    }
}

fn map_f32(inputs: &[Option<&ConstVal>], f: impl Fn(f32) -> f32) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let vals = t.as_f32()?;
    let result: Vec<f32> = vals.iter().map(|&v| f(v)).collect();
    Some(ConstVal::F32(result, t.dims().to_vec(), TensorProto::FLOAT))
}

fn binary_f32(inputs: &[Option<&ConstVal>], f: impl Fn(f32, f32) -> f32) -> Option<ConstVal> {
    let a = inp(inputs, 0)?;
    let b = inp(inputs, 1)?;
    let av = a.as_f32()?;
    let bv = b.as_f32()?;
    let result: Vec<f32> = if av.len() == 1 {
        bv.iter().map(|&bv| f(av[0], bv)).collect()
    } else if bv.len() == 1 {
        av.iter().map(|&av| f(av, bv[0])).collect()
    } else if a.dims() == b.dims() {
        av.iter().zip(&bv).map(|(&a, &b)| f(a, b)).collect()
    } else {
        return None;
    };
    let dims = broadcast_dims(a.dims(), b.dims())?;
    Some(ConstVal::F32(result, dims, TensorProto::FLOAT))
}

fn eval_unsqueeze(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let axes = if let Some(ax) = inputs.get(1).and_then(|o| o.as_ref()) {
        ax.as_i64()?
    } else {
        attr_ints(node, "axes")?
    };
    let out_rank = t.dims().len() + axes.len();
    let rank_i64 = out_rank as i64;
    for &ax in &axes {
        if ax < -rank_i64 || ax >= rank_i64 {
            return None;
        }
    }
    let normalized: Vec<usize> = axes.iter().map(|&ax| normalize_idx(ax, rank_i64)).collect();
    let unique: HashSet<usize> = normalized.iter().copied().collect();
    if unique.len() != normalized.len() {
        return None;
    }
    let mut dims: Vec<i64> = Vec::with_capacity(out_rank);
    let mut src = 0;
    for i in 0..out_rank {
        if normalized.contains(&i) {
            dims.push(1);
        } else {
            dims.push(t.dims().get(src).copied().unwrap_or(1));
            src += 1;
        }
    }
    match t {
        ConstVal::F32(v, _, dt) => Some(ConstVal::F32(v.clone(), dims, *dt)),
        ConstVal::ShapeOnly(_) => None,
        ConstVal::I64(v, _, dt) => Some(ConstVal::I64(v.clone(), dims, *dt)),
    }
}

fn eval_squeeze(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let data = t.as_i64()?;
    let old_dims = t.dims();
    let axes = if let Some(ax) = inputs.get(1).and_then(|o| o.as_ref()) {
        ax.as_i64()?
    } else {
        attr_ints(node, "axes")?
    };
    let rank = old_dims.len() as i64;
    let remove: std::collections::HashSet<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (rank + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    let new_dims: Vec<i64> = old_dims
        .iter()
        .enumerate()
        .filter(|(i, _)| !remove.contains(i))
        .map(|(_, &d)| d)
        .collect();
    Some(ConstVal::I64(data, new_dims, TensorProto::INT64))
}

fn eval_split(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<Vec<(String, ConstVal)>> {
    let t = inp(inputs, 0)?;
    let data = t.as_i64()?;
    let old_dims = t.dims();
    let axis = attr_int(node, "axis").unwrap_or(0);
    let rank = old_dims.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };
    if axis >= old_dims.len() {
        return None;
    }
    let split_sizes = if let Some(sp) = inputs.get(1).and_then(|o| o.as_ref()) {
        sp.as_i64()?
    } else if let Some(sp) = attr_ints(node, "split") {
        sp
    } else {
        let num = node.output.len() as i64;
        let dim = old_dims[axis];
        let chunk = dim / num;
        vec![chunk; num as usize]
    };
    if old_dims.len() != 1 || axis != 0 {
        return None;
    }
    let mut results = Vec::new();
    let mut offset = 0usize;
    for (i, &size) in split_sizes.iter().enumerate() {
        let size = size as usize;
        let out_name = node.output.get(i)?;
        if out_name.is_empty() {
            offset += size;
            continue;
        }
        let chunk: Vec<i64> = data[offset..offset + size].to_vec();
        results.push((
            out_name.clone(),
            ConstVal::I64(chunk, vec![size as i64], TensorProto::INT64),
        ));
        offset += size;
    }
    Some(results)
}

fn eval_concat(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let axis = attr_int(node, "axis").unwrap_or(0);
    let first = inp(inputs, 0)?;
    if first.dims().len() != 1 || axis != 0 {
        return None;
    }
    let has_i64 = inputs
        .iter()
        .any(|i| matches!(i, Some(ConstVal::I64(_, _, _))));
    if has_i64 {
        let mut result: Vec<i64> = Vec::new();
        for i in inputs {
            let v = i.as_ref()?;
            let vals = v
                .as_i64()
                .or_else(|| v.as_f32().map(|f| f.iter().map(|&x| x as i64).collect()))?;
            result.extend(vals);
        }
        Some(ConstVal::I64(
            result.clone(),
            vec![result.len() as i64],
            TensorProto::INT64,
        ))
    } else {
        let mut result: Vec<f32> = Vec::new();
        for i in inputs {
            result.extend(i.as_ref()?.as_f32()?);
        }
        Some(ConstVal::F32(
            result.clone(),
            vec![result.len() as i64],
            TensorProto::FLOAT,
        ))
    }
}

fn eval_reshape(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let data = inp(inputs, 0)?;
    let raw_shape = inp(inputs, 1)?.as_i64()?;
    let old_dims = data.dims();
    let total_elems: i64 = old_dims.iter().product();
    let allowzero = attr_int(node, "allowzero").unwrap_or(0) != 0;

    if raw_shape.iter().filter(|&&d| d == -1).count() > 1 {
        return None;
    }

    let mut shape: Vec<i64> = raw_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 && !allowzero {
                old_dims.get(i).copied().unwrap_or(1)
            } else {
                d
            }
        })
        .collect();

    if let Some(neg_pos) = shape.iter().position(|&d| d == -1) {
        let known: i64 = shape.iter().filter(|&&d| d > 0).product();
        if known <= 0 || total_elems % known != 0 {
            return None;
        }
        shape[neg_pos] = total_elems / known;
    }

    let result_elems: i64 = shape.iter().product();
    if result_elems != total_elems {
        return None;
    }

    match data {
        ConstVal::F32(v, _, dt) => Some(ConstVal::F32(v.clone(), shape, *dt)),
        ConstVal::I64(v, _, dt) => Some(ConstVal::I64(v.clone(), shape, *dt)),
        ConstVal::ShapeOnly(_) => None,
    }
}

fn eval_constant_of_shape(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let shape = inp(inputs, 0)?.as_i64()?;
    if shape.iter().any(|&d| d < 0) {
        return None;
    }
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let value_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .and_then(|a| a.t.as_ref());

    let Some(value_tensor) = value_attr else {
        return Some(ConstVal::F32(vec![0.0; total], shape, TensorProto::FLOAT));
    };

    if let Some(fv) = ConstVal::extract_f32(value_tensor) {
        let fill = fv.first().copied().unwrap_or(0.0);
        return Some(ConstVal::F32(vec![fill; total], shape, TensorProto::FLOAT));
    }
    if let Some(iv) = ConstVal::extract_i64(value_tensor) {
        let fill = iv.first().copied().unwrap_or(0);
        return Some(ConstVal::I64(vec![fill; total], shape, TensorProto::INT64));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_const_node(name: &str, out: &str, val: f32) -> NodeProto {
        use super::super::onnx_proto;
        let mut node = onnx_proto::make_node("Constant", vec![], vec![out.to_string()], vec![]);
        node.name = name.to_string();
        node.attribute.push(onnx_proto::AttributeProto {
            name: "value".to_string(),
            t: Some(TensorProto {
                data_type: TensorProto::FLOAT,
                dims: vec![1],
                float_data: vec![val],
                ..Default::default()
            }),
            ..Default::default()
        });
        node
    }

    #[test]
    fn propagates_div_of_constants() {
        use super::super::onnx_proto;
        let mut model = ModelProto {
            graph: Some(onnx_proto::GraphProto {
                node: vec![
                    make_const_node("c1", "a", 1.0),
                    make_const_node("c2", "b", 8.0),
                    onnx_proto::make_node(
                        "Div",
                        vec!["a".into(), "b".into()],
                        vec!["result".into()],
                        vec![],
                    ),
                ],
                output: vec![onnx_proto::make_tensor_value_info(
                    "result",
                    TensorProto::FLOAT,
                    &[1],
                )],
                ..Default::default()
            }),
            ..Default::default()
        };

        onnx_proto::fold_constant_nodes(&mut model);
        let propagated = propagate_constants(&mut model);

        assert!(!propagated.is_empty());
        let graph = model.graph.as_ref().unwrap();
        assert!(graph.node.is_empty());
        let init = graph
            .initializer
            .iter()
            .find(|i| i.name == "result")
            .expect("result initializer");
        assert!((init.float_data[0] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn propagates_shape_slice_cast_sqrt_chain() {
        use super::super::onnx_proto;
        let shape_data = TensorProto {
            name: "input".to_string(),
            data_type: TensorProto::FLOAT,
            dims: vec![1, 6, 300, 64],
            ..Default::default()
        };
        let starts = TensorProto {
            name: "starts".to_string(),
            data_type: TensorProto::INT64,
            dims: vec![1],
            int64_data: vec![3],
            ..Default::default()
        };
        let ends = TensorProto {
            name: "ends".to_string(),
            data_type: TensorProto::INT64,
            dims: vec![1],
            int64_data: vec![4],
            ..Default::default()
        };

        let mut cast_node =
            onnx_proto::make_node("Cast", vec!["sliced".into()], vec!["casted".into()], vec![]);
        cast_node.attribute.push(onnx_proto::AttributeProto {
            name: "to".to_string(),
            i: TensorProto::FLOAT as i64,
            ..Default::default()
        });

        let mut model = ModelProto {
            graph: Some(onnx_proto::GraphProto {
                node: vec![
                    onnx_proto::make_node(
                        "Shape",
                        vec!["input".into()],
                        vec!["shape_out".into()],
                        vec![],
                    ),
                    onnx_proto::make_node(
                        "Slice",
                        vec!["shape_out".into(), "starts".into(), "ends".into()],
                        vec!["sliced".into()],
                        vec![],
                    ),
                    cast_node,
                    onnx_proto::make_node(
                        "Sqrt",
                        vec!["casted".into()],
                        vec!["sqrt_out".into()],
                        vec![],
                    ),
                ],
                initializer: vec![shape_data, starts, ends],
                output: vec![onnx_proto::make_tensor_value_info(
                    "sqrt_out",
                    TensorProto::FLOAT,
                    &[1],
                )],
                ..Default::default()
            }),
            ..Default::default()
        };

        let propagated = propagate_constants(&mut model);
        assert_eq!(propagated.len(), 4);

        let graph = model.graph.as_ref().unwrap();
        assert!(graph.node.is_empty());
        let init = graph
            .initializer
            .iter()
            .find(|i| i.name == "sqrt_out")
            .expect("sqrt_out initializer");
        assert!((init.float_data[0] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn unsqueeze_multiple_axes() {
        let val = ConstVal::I64(vec![1, 2, 3], vec![3], TensorProto::INT64);
        let inputs = vec![Some(&val)];
        let mut node = super::super::onnx_proto::make_node(
            "Unsqueeze",
            vec!["x".into()],
            vec!["y".into()],
            vec![],
        );
        node.attribute
            .push(super::super::onnx_proto::AttributeProto {
                name: "axes".to_string(),
                ints: vec![0, 2],
                ..Default::default()
            });
        let result = eval_unsqueeze(&node, &inputs).unwrap();
        assert_eq!(result.dims(), &[1, 3, 1]);
    }

    #[test]
    fn reshape_infers_neg_one() {
        let data = ConstVal::F32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            TensorProto::FLOAT,
        );
        let shape = ConstVal::I64(vec![3, -1], vec![2], TensorProto::INT64);
        let inputs: Vec<Option<&ConstVal>> = vec![Some(&data), Some(&shape)];
        let node = super::super::onnx_proto::make_node(
            "Reshape",
            vec!["d".into(), "s".into()],
            vec!["out".into()],
            vec![],
        );
        let result = eval_reshape(&node, &inputs).unwrap();
        assert_eq!(result.dims(), &[3, 2]);
    }

    #[test]
    fn reshape_rejects_invalid() {
        let data = ConstVal::F32(vec![1.0, 2.0, 3.0, 4.0], vec![4], TensorProto::FLOAT);
        let shape = ConstVal::I64(vec![3], vec![1], TensorProto::INT64);
        let inputs: Vec<Option<&ConstVal>> = vec![Some(&data), Some(&shape)];
        let node = super::super::onnx_proto::make_node(
            "Reshape",
            vec!["d".into(), "s".into()],
            vec!["out".into()],
            vec![],
        );
        assert!(eval_reshape(&node, &inputs).is_none());
    }

    #[test]
    fn scalar_tensor_binary() {
        let scalar = ConstVal::F32(vec![2.0], vec![1], TensorProto::FLOAT);
        let tensor = ConstVal::F32(vec![3.0, 6.0, 9.0], vec![3], TensorProto::FLOAT);
        let inputs: Vec<Option<&ConstVal>> = vec![Some(&scalar), Some(&tensor)];
        let result = binary_f32(&inputs, |a, b| a * b).unwrap();
        assert_eq!(result.as_f32().unwrap(), vec![6.0, 12.0, 18.0]);
    }

    #[test]
    fn unsqueeze_rejects_duplicate_axes() {
        let val = ConstVal::I64(vec![1, 2, 3], vec![3], TensorProto::INT64);
        let inputs = vec![Some(&val)];
        let mut node = super::super::onnx_proto::make_node(
            "Unsqueeze",
            vec!["x".into()],
            vec!["y".into()],
            vec![],
        );
        node.attribute
            .push(super::super::onnx_proto::AttributeProto {
                name: "axes".to_string(),
                ints: vec![0, 0],
                ..Default::default()
            });
        assert!(eval_unsqueeze(&node, &inputs).is_none());
    }

    #[test]
    fn slice_bails_on_non_default_steps() {
        let data = ConstVal::I64(vec![10, 20, 30, 40], vec![4], TensorProto::INT64);
        let starts = ConstVal::I64(vec![0], vec![1], TensorProto::INT64);
        let ends = ConstVal::I64(vec![4], vec![1], TensorProto::INT64);
        let steps = ConstVal::I64(vec![2], vec![1], TensorProto::INT64);
        let inputs: Vec<Option<&ConstVal>> =
            vec![Some(&data), Some(&starts), Some(&ends), None, Some(&steps)];
        assert!(eval_slice(&inputs).is_none());
    }

    #[test]
    fn slice_bails_on_non_zero_axis() {
        let data = ConstVal::I64(vec![10, 20, 30], vec![3], TensorProto::INT64);
        let starts = ConstVal::I64(vec![0], vec![1], TensorProto::INT64);
        let ends = ConstVal::I64(vec![2], vec![1], TensorProto::INT64);
        let axes = ConstVal::I64(vec![1], vec![1], TensorProto::INT64);
        let inputs: Vec<Option<&ConstVal>> =
            vec![Some(&data), Some(&starts), Some(&ends), Some(&axes)];
        assert!(eval_slice(&inputs).is_none());
    }

    #[test]
    fn shape_only_output_does_not_block_propagation() {
        use super::super::onnx_proto;

        let input_tensor = TensorProto {
            name: "input".to_string(),
            data_type: TensorProto::FLOAT,
            dims: vec![1, 6, 300, 64],
            ..Default::default()
        };

        let mut model = ModelProto {
            graph: Some(onnx_proto::GraphProto {
                node: vec![onnx_proto::make_node(
                    "Shape",
                    vec!["input".into()],
                    vec!["shape_out".into()],
                    vec![],
                )],
                initializer: vec![input_tensor],
                value_info: vec![onnx_proto::make_tensor_value_info(
                    "shape_out",
                    TensorProto::INT64,
                    &[4],
                )],
                output: vec![onnx_proto::make_tensor_value_info(
                    "shape_out",
                    TensorProto::INT64,
                    &[4],
                )],
                ..Default::default()
            }),
            ..Default::default()
        };

        let propagated = propagate_constants(&mut model);
        assert_eq!(propagated.len(), 1);

        let graph = model.graph.as_ref().unwrap();
        assert!(graph.node.is_empty());
        let init = graph
            .initializer
            .iter()
            .find(|i| i.name == "shape_out")
            .expect("shape_out should be an initializer");
        assert_eq!(init.int64_data, vec![1, 6, 300, 64]);
    }
}
