use std::collections::{HashMap, HashSet};

use super::onnx_proto::{
    GraphProto, ModelProto, NodeProto, TensorProto, tensor_to_f32, tensor_to_i64,
};

pub fn fold_constant_nodes(model: &mut ModelProto) -> HashSet<String> {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return HashSet::new(),
    };

    let mut folded_tensors: Vec<TensorProto> = Vec::new();
    let mut folded_names: HashSet<String> = HashSet::new();

    for node in &graph.node {
        if node.op_type != "Constant" {
            continue;
        }
        let out_name = match node.output.first() {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let tensor = match node.attribute.iter().find(|a| a.name == "value") {
            Some(a) => match a.t.as_ref() {
                Some(t) => t,
                None => continue,
            },
            None => continue,
        };
        let mut t = tensor.clone();
        t.name = out_name.clone();
        folded_tensors.push(t);
        folded_names.insert(out_name.clone());
    }

    if folded_names.is_empty() {
        return folded_names;
    }

    graph
        .node
        .retain(|n| n.op_type != "Constant" || !n.output.iter().any(|o| folded_names.contains(o)));

    let count = folded_tensors.len();
    graph.initializer.extend(folded_tensors);

    tracing::info!(count, "folded Constant ops into initializers");

    let propagated_names = propagate_constants(graph);
    if !propagated_names.is_empty() {
        tracing::info!(
            propagated = propagated_names.len(),
            "propagated constants after Constant-node folding"
        );
    }
    folded_names.extend(propagated_names);

    // Graph simplification runs before Conv+BN fusion so that any
    // Identity chain sitting between a Conv and a BatchNormalization
    // collapses first, exposing a contiguous Conv -> BN pattern to
    // the fusion pass.
    let identity_count = remove_identity_nodes(graph);
    if identity_count > 0 {
        tracing::info!(identity_count, "removed Identity nodes");
    }

    let dead_count = eliminate_dead_nodes(graph);
    if dead_count > 0 {
        tracing::info!(dead_count, "eliminated dead nodes");
    }

    let fused = fuse_conv_batchnorm(graph);
    if fused > 0 {
        tracing::info!(fused, "fused Conv+BatchNormalization pairs");
    }

    folded_names
}

pub fn remove_identity_nodes(graph: &mut GraphProto) -> usize {
    let identity_map: HashMap<String, String> = graph
        .node
        .iter()
        .filter(|n| n.op_type == "Identity" && n.input.len() == 1 && n.output.len() == 1)
        .filter(|n| !n.input[0].is_empty() && !n.output[0].is_empty())
        .map(|n| (n.output[0].clone(), n.input[0].clone()))
        .collect();

    if identity_map.is_empty() {
        return 0;
    }

    fn resolve(name: &str, map: &HashMap<String, String>) -> String {
        let mut current = name;
        let mut visited = HashSet::new();
        while let Some(target) = map.get(current) {
            if !visited.insert(current) {
                break;
            }
            current = target;
        }
        current.to_string()
    }

    let output_names: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    for node in &mut graph.node {
        if node.op_type == "Identity" && identity_map.contains_key(&node.output[0]) {
            continue;
        }
        for inp in &mut node.input {
            if identity_map.contains_key(inp.as_str()) {
                *inp = resolve(inp, &identity_map);
            }
        }
    }

    for out in &mut graph.output {
        if identity_map.contains_key(out.name.as_str()) {
            out.name = resolve(&out.name, &identity_map);
        }
    }

    let count = identity_map.len();
    graph.node.retain(|n| {
        !(n.op_type == "Identity"
            && n.output.len() == 1
            && identity_map.contains_key(&n.output[0])
            && !output_names.contains(&n.output[0]))
    });
    count
}

pub fn eliminate_dead_nodes(graph: &mut GraphProto) -> usize {
    let output_names: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    let mut consumed: HashSet<String> = output_names;
    let mut changed = true;
    while changed {
        changed = false;
        for node in &graph.node {
            let produces_consumed = node.output.iter().any(|o| consumed.contains(o));
            if produces_consumed {
                for inp in &node.input {
                    if !inp.is_empty() && consumed.insert(inp.clone()) {
                        changed = true;
                    }
                }
            }
        }
    }

    let before = graph.node.len();
    graph
        .node
        .retain(|n| n.output.iter().any(|o| consumed.contains(o)));
    let removed = before - graph.node.len();

    if removed > 0 {
        graph.initializer.retain(|i| consumed.contains(&i.name));
        graph.value_info.retain(|vi| consumed.contains(&vi.name));
    }

    removed
}

pub fn propagate_constants_with_shapes(
    graph: &mut GraphProto,
    traced_shapes: &HashMap<String, Vec<i64>>,
) -> usize {
    for node in &graph.node {
        if node.op_type == "Shape"
            && let Some(inp_name) = node.input.first()
            && let Some(full_shape) = traced_shapes.get(inp_name)
            && let Some(out_name) = node.output.first()
            && !out_name.is_empty()
            && !graph.initializer.iter().any(|i| i.name == *out_name)
        {
            let ndim = full_shape.len() as i64;
            let start_attr = node
                .attribute
                .iter()
                .find(|a| a.name == "start")
                .map(|a| a.i)
                .unwrap_or(0);
            let end_attr = node
                .attribute
                .iter()
                .find(|a| a.name == "end")
                .map(|a| a.i)
                .unwrap_or(ndim);
            let start = if start_attr < 0 {
                (ndim + start_attr).max(0) as usize
            } else {
                (start_attr as usize).min(full_shape.len())
            };
            let end = if end_attr < 0 {
                (ndim + end_attr).max(0) as usize
            } else {
                (end_attr as usize).min(full_shape.len())
            };
            let sliced: Vec<i64> = if start < end {
                full_shape[start..end].to_vec()
            } else {
                vec![]
            };
            graph.initializer.push(TensorProto {
                name: out_name.clone(),
                data_type: TensorProto::INT64,
                dims: vec![sliced.len() as i64],
                int64_data: sliced,
                ..Default::default()
            });
        }
    }
    let init_names: HashSet<String> = graph.initializer.iter().map(|i| i.name.clone()).collect();
    graph
        .node
        .retain(|n| n.op_type != "Shape" || !n.output.iter().any(|o| init_names.contains(o)));
    let folded = propagate_constants(graph);
    folded.len()
}

pub(crate) fn propagate_constants(graph: &mut GraphProto) -> HashSet<String> {
    let mut constants: HashMap<String, TensorProto> = graph
        .initializer
        .iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();

    let mut folded_node_indices: HashSet<usize> = HashSet::new();

    loop {
        let mut progress = false;
        for (idx, node) in graph.node.iter().enumerate() {
            if folded_node_indices.contains(&idx) {
                continue;
            }
            let inputs: Vec<&str> = node
                .input
                .iter()
                .filter(|s| !s.is_empty())
                .map(String::as_str)
                .collect();
            if inputs.is_empty() {
                continue;
            }
            if !inputs.iter().all(|name| constants.contains_key(*name)) {
                continue;
            }
            let input_tensors: Vec<&TensorProto> = inputs.iter().map(|n| &constants[*n]).collect();
            if let Some(outputs) = eval_const_node(node, &input_tensors) {
                for (out_name, tensor) in outputs {
                    constants.insert(out_name, tensor);
                }
                folded_node_indices.insert(idx);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }

    if folded_node_indices.is_empty() {
        return HashSet::new();
    }

    let mut new_init_names: HashSet<String> = HashSet::new();
    for idx in &folded_node_indices {
        for out in &graph.node[*idx].output {
            if !out.is_empty() && constants.contains_key(out) {
                new_init_names.insert(out.clone());
            }
        }
    }

    let mut consumed_by_remaining: HashSet<String> = graph
        .node
        .iter()
        .enumerate()
        .filter(|(i, _)| !folded_node_indices.contains(i))
        .flat_map(|(_, n)| n.input.iter().cloned())
        .collect();
    for node in &graph.node {
        if super::is_control_flow(&node.op_type) {
            let outer_refs = super::collect_subgraph_outer_refs(node, graph);
            consumed_by_remaining.extend(outer_refs);
        }
    }
    let output_names: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    for name in &new_init_names {
        if (consumed_by_remaining.contains(name) || output_names.contains(name))
            && let Some(t) = constants.get(name)
            && !graph.initializer.iter().any(|i| i.name == *name)
        {
            graph.initializer.push(t.clone());
        }
    }

    let removed_outputs: HashSet<String> = folded_node_indices
        .iter()
        .flat_map(|idx| graph.node[*idx].output.iter().cloned())
        .collect();
    graph
        .input
        .retain(|vi| !removed_outputs.contains(&vi.name) || output_names.contains(&vi.name));

    let count = folded_node_indices.len();
    let mut kept = Vec::with_capacity(graph.node.len() - count);
    for (idx, node) in graph.node.drain(..).enumerate() {
        if !folded_node_indices.contains(&idx) {
            kept.push(node);
        }
    }
    graph.node = kept;

    tracing::info!(count, "propagated constant subgraphs into initializers");
    new_init_names
}

fn eval_const_node(
    node: &NodeProto,
    inputs: &[&TensorProto],
) -> Option<Vec<(String, TensorProto)>> {
    let out_name = node.output.first()?.clone();
    if out_name.is_empty() {
        return None;
    }
    match node.op_type.as_str() {
        "Identity" => {
            let mut t = inputs[0].clone();
            t.name = out_name.clone();
            Some(vec![(out_name, t)])
        }
        "Cast" => eval_cast(node, inputs[0], &out_name),
        "Sqrt" => eval_unary_f32(inputs[0], &out_name, f32::sqrt),
        "Neg" => eval_unary_f32(inputs[0], &out_name, |x| -x),
        "Abs" => eval_unary_f32(inputs[0], &out_name, f32::abs),
        "Exp" => eval_unary_f32(inputs[0], &out_name, f32::exp),
        "Log" => eval_unary_f32(inputs[0], &out_name, f32::ln),
        "Ceil" => eval_unary_f32(inputs[0], &out_name, f32::ceil),
        "Floor" => eval_unary_f32(inputs[0], &out_name, f32::floor),
        "Reciprocal" => eval_unary_f32(inputs[0], &out_name, |x| 1.0 / x),
        "Relu" => eval_unary_f32(inputs[0], &out_name, |x| x.max(0.0)),
        "Sigmoid" => eval_unary_f32(inputs[0], &out_name, |x| 1.0 / (1.0 + (-x).exp())),
        "Tanh" => eval_unary_f32(inputs[0], &out_name, f32::tanh),
        "Add" => eval_binary_f32(inputs, &out_name, |a, b| a + b),
        "Sub" => eval_binary_f32(inputs, &out_name, |a, b| a - b),
        "Mul" => eval_binary_f32(inputs, &out_name, |a, b| a * b),
        "Div" => eval_binary_f32(inputs, &out_name, |a, b| a / b),
        "Pow" => eval_binary_f32(inputs, &out_name, f32::powf),
        "Reshape" => eval_reshape(node, inputs, &out_name),
        "Squeeze" => eval_squeeze(node, inputs, &out_name),
        "Unsqueeze" => eval_unsqueeze(node, inputs, &out_name),
        "Shape" => eval_shape(node, inputs[0], &out_name),
        "Gather" if inputs.len() >= 2 => eval_gather(node, inputs, &out_name),
        "Slice" if inputs.len() >= 3 => eval_slice(inputs, &out_name),
        "Concat" => eval_concat(node, inputs, &out_name),
        "ConstantOfShape" => eval_constant_of_shape(node, inputs[0], &out_name),
        "Where" if inputs.len() == 3 => eval_where(inputs, &out_name),
        "Range" if inputs.len() == 3 => eval_range(inputs, &out_name),
        "Equal" => eval_cmp(inputs, &out_name, |a, b| a == b, |a, b| a == b),
        "Less" => eval_cmp(inputs, &out_name, |a, b| a < b, |a, b| a < b),
        "Greater" => eval_cmp(inputs, &out_name, |a, b| a > b, |a, b| a > b),
        "Not" => eval_not(inputs[0], &out_name),
        "And" => eval_logical(inputs, &out_name, |a, b| a & b),
        "Or" => eval_logical(inputs, &out_name, |a, b| a | b),
        "Transpose" => eval_transpose(node, inputs[0], &out_name),
        "ReduceMean" => eval_reduce(node, inputs, &out_name, ReduceOp::Mean),
        "ReduceSum" => eval_reduce(node, inputs, &out_name, ReduceOp::Sum),
        "ReduceMax" => eval_reduce(node, inputs, &out_name, ReduceOp::Max),
        "ReduceMin" => eval_reduce(node, inputs, &out_name, ReduceOp::Min),
        "Resize" => eval_resize(node, inputs, &out_name),
        "Expand" if inputs.len() == 2 => eval_expand(inputs, &out_name),
        "Tile" if inputs.len() == 2 => eval_tile(inputs, &out_name),
        "ScatterND" if inputs.len() == 3 => eval_scatter_nd(inputs, &out_name),
        "Split" => eval_split(node, inputs, &node.output),
        _ => None,
    }
}

fn eval_expand(inputs: &[&TensorProto], out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let data = inputs[0];
    let shape = tensor_to_i64(inputs[1]);
    if shape.is_empty() {
        return None;
    }
    let out_dims = broadcast_shape(&data.dims, &shape)?;
    let total = broadcast_total(&out_dims)?;
    if data.data_type == TensorProto::INT64 {
        let v = tensor_to_i64(data);
        if v.is_empty() {
            return None;
        }
        let mut result = Vec::with_capacity(total);
        for i in 0..total {
            let di = broadcast_index(i, &out_dims, &data.dims);
            result.push(v[di]);
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: out_dims,
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let v = tensor_to_f32(data);
    if v.is_empty() {
        return None;
    }
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let di = broadcast_index(i, &out_dims, &data.dims);
        result.push(v[di]);
    }
    let t = make_f32_tensor(out_name, &out_dims, &result, data.data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_tile(inputs: &[&TensorProto], out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let data = inputs[0];
    let repeats = tensor_to_i64(inputs[1]);
    if repeats.is_empty() || repeats.len() != data.dims.len() {
        return None;
    }
    let rank = data.dims.len();
    let out_dims: Vec<i64> = data
        .dims
        .iter()
        .zip(&repeats)
        .map(|(&d, &r)| d * r)
        .collect();
    let total = broadcast_total(&out_dims)?;

    let in_strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * data.dims[i + 1] as usize;
        }
        s
    };
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_dims[i + 1] as usize;
        }
        s
    };

    if data.data_type == TensorProto::INT64 {
        let v = tensor_to_i64(data);
        if v.is_empty() {
            return None;
        }
        let mut result = vec![0i64; total];
        for (o, out_slot) in result.iter_mut().enumerate().take(total) {
            let mut src = 0usize;
            let mut rem = o;
            for i in 0..rank {
                let coord = rem / out_strides[i];
                rem %= out_strides[i];
                src += (coord % data.dims[i] as usize) * in_strides[i];
            }
            *out_slot = v[src];
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: out_dims,
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let v = tensor_to_f32(data);
    if v.is_empty() {
        return None;
    }
    let mut result = vec![0f32; total];
    for (o, out_slot) in result.iter_mut().enumerate().take(total) {
        let mut src = 0usize;
        let mut rem = o;
        for i in 0..rank {
            let coord = rem / out_strides[i];
            rem %= out_strides[i];
            src += (coord % data.dims[i] as usize) * in_strides[i];
        }
        *out_slot = v[src];
    }
    let t = make_f32_tensor(out_name, &out_dims, &result, data.data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_constant_of_shape(
    node: &NodeProto,
    shape_t: &TensorProto,
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let dims = tensor_to_i64(shape_t);
    if dims.is_empty() {
        return None;
    }
    let total = broadcast_total(&dims)?;
    let (dtype, f_val, i_val) = match node.attribute.iter().find(|a| a.name == "value") {
        Some(a) => match a.t.as_ref() {
            Some(t) => {
                let fv = tensor_to_f32(t).first().copied().unwrap_or(0.0);
                let iv = tensor_to_i64(t).first().copied().unwrap_or(fv as i64);
                (t.data_type, fv, iv)
            }
            None => (TensorProto::FLOAT, 0.0, 0),
        },
        None => (TensorProto::FLOAT, 0.0, 0),
    };
    let t = match dtype {
        TensorProto::INT64 => TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: dims.clone(),
            int64_data: vec![i_val; total],
            ..Default::default()
        },
        TensorProto::INT32 => TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT32,
            dims: dims.clone(),
            int32_data: vec![i_val as i32; total],
            ..Default::default()
        },
        TensorProto::BOOL => TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::BOOL,
            dims: dims.clone(),
            int32_data: vec![(i_val != 0) as i32; total],
            ..Default::default()
        },
        _ => make_f32_tensor(out_name, &dims, &vec![f_val; total], dtype),
    };
    Some(vec![(out_name.to_string(), t)])
}

fn eval_where(inputs: &[&TensorProto], out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let cond = tensor_to_i64(inputs[0]);
    let data_type =
        if inputs[1].data_type == TensorProto::INT64 && inputs[2].data_type == TensorProto::INT64 {
            TensorProto::INT64
        } else {
            TensorProto::FLOAT
        };
    if data_type == TensorProto::INT64 {
        let x = tensor_to_i64(inputs[1]);
        let y = tensor_to_i64(inputs[2]);
        if x.is_empty() || y.is_empty() || cond.is_empty() {
            return None;
        }
        let xy_dims = broadcast_shape(&inputs[1].dims, &inputs[2].dims)?;
        let out_dims = broadcast_shape(&xy_dims, &inputs[0].dims)?;
        let total = broadcast_total(&out_dims)?;
        let mut result = Vec::with_capacity(total);
        for i in 0..total {
            let ci = broadcast_index(i, &out_dims, &inputs[0].dims);
            let xi = broadcast_index(i, &out_dims, &inputs[1].dims);
            let yi = broadcast_index(i, &out_dims, &inputs[2].dims);
            result.push(if cond[ci] != 0 { x[xi] } else { y[yi] });
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: out_dims,
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let x = tensor_to_f32(inputs[1]);
    let y = tensor_to_f32(inputs[2]);
    if x.is_empty() || y.is_empty() || cond.is_empty() {
        return None;
    }
    let xy_dims = broadcast_shape(&inputs[1].dims, &inputs[2].dims)?;
    let out_dims = broadcast_shape(&xy_dims, &inputs[0].dims)?;
    let total = broadcast_total(&out_dims)?;
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let ci = broadcast_index(i, &out_dims, &inputs[0].dims);
        let xi = broadcast_index(i, &out_dims, &inputs[1].dims);
        let yi = broadcast_index(i, &out_dims, &inputs[2].dims);
        result.push(if cond[ci] != 0 { x[xi] } else { y[yi] });
    }
    let t = make_f32_tensor(out_name, &out_dims, &result, inputs[1].data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_range(inputs: &[&TensorProto], out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let is_int = inputs[0].data_type == TensorProto::INT64
        && inputs[1].data_type == TensorProto::INT64
        && inputs[2].data_type == TensorProto::INT64;
    if is_int {
        let start = tensor_to_i64(inputs[0]).first().copied()?;
        let limit = tensor_to_i64(inputs[1]).first().copied()?;
        let delta = tensor_to_i64(inputs[2]).first().copied()?;
        if delta == 0 {
            return None;
        }
        let producing = (delta > 0 && start < limit) || (delta < 0 && start > limit);
        let count = if producing {
            let span = (limit - start) as i128;
            let d = delta as i128;
            let c = (span + d - d.signum()) / d;
            usize::try_from(c).ok()?
        } else {
            0
        };
        if count > MAX_BROADCAST_ELEMENTS {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        let mut v = start;
        for _ in 0..count {
            out.push(v);
            v = v.checked_add(delta)?;
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: vec![out.len() as i64],
            int64_data: out,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let start = tensor_to_f32(inputs[0]).first().copied()?;
    let limit = tensor_to_f32(inputs[1]).first().copied()?;
    let delta = tensor_to_f32(inputs[2]).first().copied()?;
    if delta == 0.0 || !delta.is_finite() || !start.is_finite() || !limit.is_finite() {
        return None;
    }
    let count = ((limit - start) / delta).ceil();
    if count <= 0.0 {
        let dims = vec![0i64];
        let t = make_f32_tensor(out_name, &dims, &[], inputs[0].data_type);
        return Some(vec![(out_name.to_string(), t)]);
    }
    if count as usize > MAX_BROADCAST_ELEMENTS {
        return None;
    }
    let count = count as usize;
    let mut out = Vec::with_capacity(count);
    let mut v = start;
    for _ in 0..count {
        if (delta > 0.0 && v >= limit) || (delta < 0.0 && v <= limit) {
            break;
        }
        out.push(v);
        v += delta;
    }
    let dims = vec![out.len() as i64];
    let t = make_f32_tensor(out_name, &dims, &out, inputs[0].data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_cmp(
    inputs: &[&TensorProto],
    out_name: &str,
    f_f32: fn(f32, f32) -> bool,
    f_i64: fn(i64, i64) -> bool,
) -> Option<Vec<(String, TensorProto)>> {
    if inputs.len() < 2 {
        return None;
    }
    let out_dims = broadcast_shape(&inputs[0].dims, &inputs[1].dims)?;
    let total = broadcast_total(&out_dims)?;

    let both_int =
        inputs[0].data_type == TensorProto::INT64 && inputs[1].data_type == TensorProto::INT64;
    let mut result = Vec::with_capacity(total);
    if both_int {
        let a = tensor_to_i64(inputs[0]);
        let b = tensor_to_i64(inputs[1]);
        if a.is_empty() || b.is_empty() {
            return None;
        }
        for i in 0..total {
            let ai = broadcast_index(i, &out_dims, &inputs[0].dims);
            let bi = broadcast_index(i, &out_dims, &inputs[1].dims);
            result.push(f_i64(a[ai], b[bi]) as i32);
        }
    } else {
        let a = tensor_to_f32(inputs[0]);
        let b = tensor_to_f32(inputs[1]);
        if a.is_empty() || b.is_empty() {
            return None;
        }
        for i in 0..total {
            let ai = broadcast_index(i, &out_dims, &inputs[0].dims);
            let bi = broadcast_index(i, &out_dims, &inputs[1].dims);
            result.push(f_f32(a[ai], b[bi]) as i32);
        }
    }
    let t = TensorProto {
        name: out_name.to_string(),
        data_type: TensorProto::BOOL,
        dims: out_dims,
        int32_data: result,
        ..Default::default()
    };
    Some(vec![(out_name.to_string(), t)])
}

fn eval_not(input: &TensorProto, out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let vals = tensor_to_i64(input);
    if vals.is_empty() {
        return None;
    }
    let t = TensorProto {
        name: out_name.to_string(),
        data_type: TensorProto::BOOL,
        dims: input.dims.clone(),
        int32_data: vals.iter().map(|&v| (v == 0) as i32).collect(),
        ..Default::default()
    };
    Some(vec![(out_name.to_string(), t)])
}

fn eval_logical(
    inputs: &[&TensorProto],
    out_name: &str,
    f: fn(i32, i32) -> i32,
) -> Option<Vec<(String, TensorProto)>> {
    if inputs.len() < 2 {
        return None;
    }
    let a = tensor_to_i64(inputs[0]);
    let b = tensor_to_i64(inputs[1]);
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let out_dims = broadcast_shape(&inputs[0].dims, &inputs[1].dims)?;
    let total = broadcast_total(&out_dims)?;
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let ai = broadcast_index(i, &out_dims, &inputs[0].dims);
        let bi = broadcast_index(i, &out_dims, &inputs[1].dims);
        result.push(f((a[ai] != 0) as i32, (b[bi] != 0) as i32));
    }
    let t = TensorProto {
        name: out_name.to_string(),
        data_type: TensorProto::BOOL,
        dims: out_dims,
        int32_data: result,
        ..Default::default()
    };
    Some(vec![(out_name.to_string(), t)])
}

fn eval_transpose(
    node: &NodeProto,
    input: &TensorProto,
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let rank = input.dims.len();
    if rank == 0 {
        return None;
    }
    let perm: Vec<usize> = match node.attribute.iter().find(|a| a.name == "perm") {
        Some(attr) => {
            if attr.ints.len() != rank {
                return None;
            }
            let mut out = Vec::with_capacity(rank);
            let mut seen = vec![false; rank];
            for &raw in &attr.ints {
                if raw < 0 || (raw as usize) >= rank {
                    return None;
                }
                let p = raw as usize;
                if seen[p] {
                    return None;
                }
                seen[p] = true;
                out.push(p);
            }
            out
        }
        None => (0..rank).rev().collect(),
    };
    let out_dims: Vec<i64> = perm.iter().map(|&p| input.dims[p]).collect();
    let total = broadcast_total(&out_dims)?;

    let src_strides = {
        let mut s = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * input.dims[i + 1];
        }
        s
    };
    let out_strides = {
        let mut s = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_dims[i + 1];
        }
        s
    };

    let permute_index = |out_linear: usize| -> usize {
        let mut src = 0i64;
        let mut rem = out_linear as i64;
        for i in 0..rank {
            let coord = rem / out_strides[i];
            rem %= out_strides[i];
            src += coord * src_strides[perm[i]];
        }
        src as usize
    };

    if input.data_type == TensorProto::INT64 {
        let vals = tensor_to_i64(input);
        if vals.is_empty() {
            return None;
        }
        let mut result = Vec::with_capacity(total);
        for i in 0..total {
            result.push(vals[permute_index(i)]);
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: out_dims,
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let vals = tensor_to_f32(input);
    if vals.is_empty() {
        return None;
    }
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        result.push(vals[permute_index(i)]);
    }
    let t = make_f32_tensor(out_name, &out_dims, &result, input.data_type);
    Some(vec![(out_name.to_string(), t)])
}

#[allow(clippy::too_many_lines)]
fn eval_resize(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let named: Vec<(&str, Option<&TensorProto>)> = {
        let mut it = inputs.iter().copied();
        node.input
            .iter()
            .map(|name| {
                let entry = if name.is_empty() { None } else { it.next() };
                (name.as_str(), entry)
            })
            .collect()
    };
    let x = named.first().and_then(|(_, t)| *t)?;
    if x.dims.len() < 2 {
        return None;
    }
    let rank = x.dims.len();
    let vals = tensor_to_f32(x);
    if vals.is_empty() {
        return None;
    }

    let mode = node
        .attribute
        .iter()
        .find(|a| a.name == "mode")
        .map(|a| std::str::from_utf8(&a.s).unwrap_or("").to_string())
        .unwrap_or_else(|| "nearest".to_string());
    let ctm = node
        .attribute
        .iter()
        .find(|a| a.name == "coordinate_transformation_mode")
        .map(|a| std::str::from_utf8(&a.s).unwrap_or("").to_string())
        .unwrap_or_else(|| "half_pixel".to_string());
    let cubic_a = node
        .attribute
        .iter()
        .find(|a| a.name == "cubic_coeff_a")
        .map(|a| a.f)
        .unwrap_or(-0.75);
    let exclude_outside = node
        .attribute
        .iter()
        .find(|a| a.name == "exclude_outside")
        .map(|a| a.i != 0)
        .unwrap_or(false);
    let extrapolation = node
        .attribute
        .iter()
        .find(|a| a.name == "extrapolation_value")
        .map(|a| a.f)
        .unwrap_or(0.0);

    let sizes_opt = named.get(3).and_then(|(_, t)| *t).and_then(|t| {
        if t.dims.is_empty() || t.dims.iter().all(|&d| d == 0) {
            None
        } else {
            let v = tensor_to_i64(t);
            if v.len() == rank { Some(v) } else { None }
        }
    });
    let scales_opt = named.get(2).and_then(|(_, t)| *t).and_then(|t| {
        if t.dims.is_empty() || t.dims.iter().all(|&d| d == 0) {
            None
        } else {
            let v = tensor_to_f32(t);
            if v.len() == rank { Some(v) } else { None }
        }
    });

    let out_dims: Vec<i64> = if let Some(sizes) = sizes_opt {
        sizes
    } else if let Some(scales) = scales_opt {
        x.dims
            .iter()
            .zip(&scales)
            .map(|(&d, &s)| (d as f32 * s) as i64)
            .collect()
    } else {
        return None;
    };
    let total_out = broadcast_total(&out_dims)?;

    let scales_eff: Vec<f32> = x
        .dims
        .iter()
        .zip(&out_dims)
        .map(|(&s, &o)| o as f32 / s as f32)
        .collect();

    let src_stride: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * x.dims[i + 1] as usize;
        }
        s
    };

    let coord = |out_i: i64, d: usize| -> f32 {
        let out_d = out_dims[d] as f32;
        let in_d = x.dims[d] as f32;
        let s = scales_eff[d];
        match ctm.as_str() {
            "half_pixel" => (out_i as f32 + 0.5) / s - 0.5,
            "pytorch_half_pixel" => {
                if out_d > 1.0 {
                    (out_i as f32 + 0.5) / s - 0.5
                } else {
                    0.0
                }
            }
            "align_corners" => {
                if out_d > 1.0 {
                    out_i as f32 * (in_d - 1.0) / (out_d - 1.0)
                } else {
                    0.0
                }
            }
            "asymmetric" => out_i as f32 / s,
            _ => (out_i as f32 + 0.5) / s - 0.5,
        }
    };

    let mode_kind = match mode.as_str() {
        "cubic" => ResizeMode::Cubic,
        "linear" => ResizeMode::Linear,
        "nearest" => ResizeMode::Nearest,
        _ => return None,
    };

    let mut result = vec![0f32; total_out];

    let dst_stride: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_dims[i + 1] as usize;
        }
        s
    };

    let resize_axes: Vec<usize> = (0..rank).filter(|&d| x.dims[d] != out_dims[d]).collect();
    if resize_axes.is_empty() {
        let t = make_f32_tensor(out_name, &out_dims, &vals, x.data_type);
        return Some(vec![(out_name.to_string(), t)]);
    }
    if !(resize_axes.len() == 2
        && resize_axes[0] + 1 == resize_axes[1]
        && resize_axes[1] + 1 == rank)
    {
        return None;
    }
    let h_axis = resize_axes[0];
    let w_axis = resize_axes[1];

    let outer_total: usize = x.dims[..h_axis].iter().map(|&d| d as usize).product();
    let in_h = x.dims[h_axis] as usize;
    let in_w = x.dims[w_axis] as usize;
    let out_h = out_dims[h_axis] as usize;
    let out_w = out_dims[w_axis] as usize;

    for outer in 0..outer_total {
        let in_plane = outer * in_h * in_w;
        let out_plane = outer * out_h * out_w;
        for oy in 0..out_h {
            let sy = coord(oy as i64, h_axis);
            for ox in 0..out_w {
                let sx = coord(ox as i64, w_axis);
                let v = match mode_kind {
                    ResizeMode::Nearest => {
                        let yi = nearest_idx(sy, in_h);
                        let xi = nearest_idx(sx, in_w);
                        vals[in_plane + yi * in_w + xi]
                    }
                    ResizeMode::Linear => sample_linear_2d(
                        &vals[in_plane..in_plane + in_h * in_w],
                        in_h,
                        in_w,
                        sy,
                        sx,
                        exclude_outside,
                        extrapolation,
                    ),
                    ResizeMode::Cubic => sample_cubic_2d(
                        &vals[in_plane..in_plane + in_h * in_w],
                        in_h,
                        in_w,
                        sy,
                        sx,
                        cubic_a,
                        exclude_outside,
                        extrapolation,
                    ),
                };
                result[out_plane + oy * out_w + ox] = v;
            }
        }
    }
    let _ = (src_stride, dst_stride);
    let t = make_f32_tensor(out_name, &out_dims, &result, x.data_type);
    Some(vec![(out_name.to_string(), t)])
}

#[derive(Clone, Copy)]
enum ResizeMode {
    Nearest,
    Linear,
    Cubic,
}

fn nearest_idx(s: f32, dim: usize) -> usize {
    if s < 0.0 {
        0
    } else {
        let i = s.round() as isize;
        if i >= dim as isize {
            dim - 1
        } else {
            i as usize
        }
    }
}

fn sample_linear_2d(
    plane: &[f32],
    h: usize,
    w: usize,
    sy: f32,
    sx: f32,
    exclude_outside: bool,
    extrap: f32,
) -> f32 {
    let (y0_in, y0) = clamp_axis(sy.floor() as isize, h);
    let (y1_in, y1) = clamp_axis(sy.floor() as isize + 1, h);
    let (x0_in, x0) = clamp_axis(sx.floor() as isize, w);
    let (x1_in, x1) = clamp_axis(sx.floor() as isize + 1, w);
    if exclude_outside && (!y0_in && !y1_in || !x0_in && !x1_in) {
        return extrap;
    }
    let dy = sy - sy.floor();
    let dx = sx - sx.floor();
    let v00 = plane[y0 * w + x0];
    let v01 = plane[y0 * w + x1];
    let v10 = plane[y1 * w + x0];
    let v11 = plane[y1 * w + x1];
    let a = v00 * (1.0 - dx) + v01 * dx;
    let b = v10 * (1.0 - dx) + v11 * dx;
    a * (1.0 - dy) + b * dy
}

#[allow(clippy::too_many_arguments)]
fn sample_cubic_2d(
    plane: &[f32],
    h: usize,
    w: usize,
    sy: f32,
    sx: f32,
    a_coef: f32,
    exclude_outside: bool,
    extrap: f32,
) -> f32 {
    let fx = sx.floor();
    let fy = sy.floor();
    let dx = sx - fx;
    let dy = sy - fy;
    let wx = cubic_weights(dx, a_coef);
    let wy = cubic_weights(dy, a_coef);
    let mut wx_eff = wx;
    let mut wy_eff = wy;
    if exclude_outside {
        for (i, w_ref) in wx_eff.iter_mut().enumerate() {
            let xi = fx as isize - 1 + i as isize;
            if xi < 0 || xi >= w as isize {
                *w_ref = 0.0;
            }
        }
        for (i, w_ref) in wy_eff.iter_mut().enumerate() {
            let yi = fy as isize - 1 + i as isize;
            if yi < 0 || yi >= h as isize {
                *w_ref = 0.0;
            }
        }
        let sx_sum: f32 = wx_eff.iter().sum();
        let sy_sum: f32 = wy_eff.iter().sum();
        if sx_sum == 0.0 || sy_sum == 0.0 {
            return extrap;
        }
        for w_ref in &mut wx_eff {
            *w_ref /= sx_sum;
        }
        for w_ref in &mut wy_eff {
            *w_ref /= sy_sum;
        }
    }
    let mut out = 0f32;
    for (iy, &wyv) in wy_eff.iter().enumerate() {
        if wyv == 0.0 {
            continue;
        }
        let yi = (fy as isize - 1 + iy as isize).clamp(0, h as isize - 1) as usize;
        let mut row_sum = 0f32;
        for (ix, &wxv) in wx_eff.iter().enumerate() {
            if wxv == 0.0 {
                continue;
            }
            let xi = (fx as isize - 1 + ix as isize).clamp(0, w as isize - 1) as usize;
            row_sum += plane[yi * w + xi] * wxv;
        }
        out += row_sum * wyv;
    }
    out
}

fn cubic_weights(t: f32, a: f32) -> [f32; 4] {
    let t1 = 1.0 + t;
    let t2 = t;
    let t3 = 1.0 - t;
    let t4 = 2.0 - t;
    [
        cubic_kernel(t1, a),
        cubic_kernel(t2, a),
        cubic_kernel(t3, a),
        cubic_kernel(t4, a),
    ]
}

fn cubic_kernel(x: f32, a: f32) -> f32 {
    let ax = x.abs();
    if ax <= 1.0 {
        (a + 2.0) * ax.powi(3) - (a + 3.0) * ax.powi(2) + 1.0
    } else if ax < 2.0 {
        a * ax.powi(3) - 5.0 * a * ax.powi(2) + 8.0 * a * ax - 4.0 * a
    } else {
        0.0
    }
}

fn clamp_axis(i: isize, dim: usize) -> (bool, usize) {
    if i < 0 {
        (false, 0)
    } else if i >= dim as isize {
        (false, dim - 1)
    } else {
        (true, i as usize)
    }
}

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

fn eval_reduce(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
    op: ReduceOp,
) -> Option<Vec<(String, TensorProto)>> {
    let input = inputs[0];
    let rank = input.dims.len();
    if rank == 0 {
        return None;
    }
    // Reduce* for non-floating-point tensors would lose precision
    // through the tensor_to_f32 path below; refuse to fold them so
    // the compiler can emit a proper integer reduction.
    if !matches!(
        input.data_type,
        TensorProto::FLOAT | TensorProto::DOUBLE | TensorProto::FLOAT16
    ) {
        return None;
    }
    let keepdims = node
        .attribute
        .iter()
        .find(|a| a.name == "keepdims")
        .map(|a| a.i != 0)
        .unwrap_or(true);
    let axes: Vec<i64> = if inputs.len() >= 2 {
        tensor_to_i64(inputs[1])
    } else {
        node.attribute
            .iter()
            .find(|a| a.name == "axes")
            .map(|a| a.ints.clone())
            .unwrap_or_else(|| (0..rank as i64).collect())
    };
    let norm_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    for &ax in &norm_axes {
        if ax >= rank {
            return None;
        }
    }
    let mut out_dims_full = input.dims.clone();
    for &ax in &norm_axes {
        out_dims_full[ax] = 1;
    }
    let out_dims: Vec<i64> = if keepdims {
        out_dims_full.clone()
    } else {
        out_dims_full
            .iter()
            .enumerate()
            .filter(|(i, _)| !norm_axes.contains(i))
            .map(|(_, &d)| d)
            .collect()
    };
    let total_out = broadcast_total(&out_dims_full)?;
    let total_in = broadcast_total(&input.dims)?;
    let vals = tensor_to_f32(input);
    if vals.is_empty() {
        return None;
    }

    let reduced_count: i64 = norm_axes.iter().map(|&a| input.dims[a]).product();

    let mut accum = vec![
        match op {
            ReduceOp::Sum | ReduceOp::Mean => 0.0f32,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        };
        total_out
    ];

    for (in_idx, &v) in vals.iter().enumerate().take(total_in) {
        let mut rem = in_idx as i64;
        let mut out_idx = 0i64;
        let mut out_stride = 1i64;
        for i in (0..rank).rev() {
            let dim_i = input.dims[i];
            let coord = rem % dim_i;
            rem /= dim_i;
            let coord_out = if norm_axes.contains(&i) { 0 } else { coord };
            out_idx += coord_out * out_stride;
            out_stride *= out_dims_full[i];
        }
        let o = out_idx as usize;
        accum[o] = match op {
            ReduceOp::Sum | ReduceOp::Mean => accum[o] + v,
            ReduceOp::Max => accum[o].max(v),
            ReduceOp::Min => accum[o].min(v),
        };
    }
    if matches!(op, ReduceOp::Mean) && reduced_count > 0 {
        for a in &mut accum {
            *a /= reduced_count as f32;
        }
    }
    let t = make_f32_tensor(out_name, &out_dims, &accum, input.data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_cast(
    node: &NodeProto,
    input: &TensorProto,
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let target_type = node
        .attribute
        .iter()
        .find(|a| a.name == "to")
        .map(|a| a.i as i32)?;
    match target_type {
        TensorProto::INT64 => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT64,
                dims: input.dims.clone(),
                int64_data: vals.iter().map(|&v| v as i64).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::INT32 => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT32,
                dims: input.dims.clone(),
                int32_data: vals.iter().map(|&v| v as i32).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::FLOAT => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::FLOAT,
                dims: input.dims.clone(),
                float_data: vals,
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::DOUBLE => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::DOUBLE,
                dims: input.dims.clone(),
                double_data: vals.iter().map(|&v| v as f64).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        TensorProto::BOOL => {
            let vals = tensor_to_f32(input);
            if vals.is_empty() {
                return None;
            }
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::BOOL,
                dims: input.dims.clone(),
                int32_data: vals.iter().map(|&v| (v != 0.0) as i32).collect(),
                ..Default::default()
            };
            Some(vec![(out_name.to_string(), t)])
        }
        _ => None,
    }
}

fn eval_unary_f32(
    input: &TensorProto,
    out_name: &str,
    f: fn(f32) -> f32,
) -> Option<Vec<(String, TensorProto)>> {
    let vals: Vec<f32> = tensor_to_f32(input).into_iter().map(f).collect();
    if vals.is_empty() {
        return None;
    }
    let out_type = input.data_type;
    let t = make_f32_tensor(out_name, &input.dims, &vals, out_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_binary_f32(
    inputs: &[&TensorProto],
    out_name: &str,
    f: fn(f32, f32) -> f32,
) -> Option<Vec<(String, TensorProto)>> {
    if inputs.len() < 2 {
        return None;
    }
    let both_int64 =
        inputs[0].data_type == TensorProto::INT64 && inputs[1].data_type == TensorProto::INT64;
    if both_int64 {
        let a = tensor_to_i64(inputs[0]);
        let b = tensor_to_i64(inputs[1]);
        if a.is_empty() || b.is_empty() {
            return None;
        }
        let (result, dims) =
            broadcast_binary_i64(&a, &inputs[0].dims, &b, &inputs[1].dims, |x, y| {
                f(x as f32, y as f32) as i64
            })?;
        let t = TensorProto {
            name: out_name.to_string(),
            dims,
            data_type: TensorProto::INT64,
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let a = tensor_to_f32(inputs[0]);
    let b = tensor_to_f32(inputs[1]);
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (result, dims) = broadcast_binary(&a, &inputs[0].dims, &b, &inputs[1].dims, f)?;
    let t = make_f32_tensor(out_name, &dims, &result, TensorProto::FLOAT);
    Some(vec![(out_name.to_string(), t)])
}

fn broadcast_shape(a_dims: &[i64], b_dims: &[i64]) -> Option<Vec<i64>> {
    let rank = a_dims.len().max(b_dims.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i < rank - a_dims.len() {
            1
        } else {
            a_dims[i - (rank - a_dims.len())]
        };
        let db = if i < rank - b_dims.len() {
            1
        } else {
            b_dims[i - (rank - b_dims.len())]
        };
        if da == db {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else if db == 1 {
            out.push(da);
        } else {
            return None;
        }
    }
    Some(out)
}

fn broadcast_index(out_idx: usize, out_dims: &[i64], src_dims: &[i64]) -> usize {
    let rank = out_dims.len();
    let src_rank = src_dims.len();
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..src_rank).rev() {
        let out_i = rank - src_rank + i;
        let coord = (out_idx / out_dims[out_i + 1..].iter().product::<i64>().max(1) as usize)
            % out_dims[out_i] as usize;
        let src_coord = if src_dims[i] == 1 { 0 } else { coord };
        idx += src_coord * stride;
        stride *= src_dims[i] as usize;
    }
    idx
}

const MAX_BROADCAST_ELEMENTS: usize = 100_000_000;

fn broadcast_total(out_dims: &[i64]) -> Option<usize> {
    let mut total: usize = 1;
    for &d in out_dims {
        let d = usize::try_from(d).ok()?;
        total = total.checked_mul(d)?;
        if total > MAX_BROADCAST_ELEMENTS {
            return None;
        }
    }
    Some(total)
}

fn broadcast_binary(
    a: &[f32],
    a_dims: &[i64],
    b: &[f32],
    b_dims: &[i64],
    f: fn(f32, f32) -> f32,
) -> Option<(Vec<f32>, Vec<i64>)> {
    let out_dims = broadcast_shape(a_dims, b_dims)?;
    let total = broadcast_total(&out_dims)?;
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let ai = broadcast_index(i, &out_dims, a_dims);
        let bi = broadcast_index(i, &out_dims, b_dims);
        result.push(f(a[ai], b[bi]));
    }
    Some((result, out_dims))
}

fn broadcast_binary_i64(
    a: &[i64],
    a_dims: &[i64],
    b: &[i64],
    b_dims: &[i64],
    f: impl Fn(i64, i64) -> i64,
) -> Option<(Vec<i64>, Vec<i64>)> {
    let out_dims = broadcast_shape(a_dims, b_dims)?;
    let total = broadcast_total(&out_dims)?;
    let mut result = Vec::with_capacity(total);
    for i in 0..total {
        let ai = broadcast_index(i, &out_dims, a_dims);
        let bi = broadcast_index(i, &out_dims, b_dims);
        result.push(f(a[ai], b[bi]));
    }
    Some((result, out_dims))
}

fn eval_reshape(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    if inputs.len() < 2 {
        return None;
    }
    let vals = tensor_to_f32(inputs[0]);
    let shape = tensor_to_i64(inputs[1]);
    if vals.is_empty() || shape.is_empty() {
        return None;
    }
    let allowzero = node
        .attribute
        .iter()
        .find(|a| a.name == "allowzero")
        .map(|a| a.i != 0)
        .unwrap_or(false);
    let mut new_dims: Vec<i64> = shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 {
                if allowzero {
                    0
                } else {
                    *inputs[0].dims.get(i).unwrap_or(&1)
                }
            } else {
                d
            }
        })
        .collect();
    if let Some(neg_idx) = new_dims.iter().position(|&d| d == -1) {
        let known: i64 = new_dims
            .iter()
            .enumerate()
            .filter(|&(i, &d)| i != neg_idx && d > 0)
            .map(|(_, &d)| d)
            .product();
        let total: i64 = vals.len() as i64;
        if known > 0 {
            new_dims[neg_idx] = total / known;
        }
    }
    let t = make_f32_tensor(out_name, &new_dims, &vals, inputs[0].data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_squeeze(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let input = inputs[0];
    let ndim = input.dims.len() as i64;
    let raw_axes: Vec<i64> = if inputs.len() >= 2 {
        tensor_to_i64(inputs[1])
    } else {
        node.attribute
            .iter()
            .find(|a| a.name == "axes")
            .map(|a| a.ints.clone())
            .unwrap_or_default()
    };
    let axes: Vec<usize> = raw_axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    if axes.is_empty() {
        let new_dims: Vec<i64> = input.dims.iter().copied().filter(|&d| d != 1).collect();
        let vals = tensor_to_f32(input);
        if vals.is_empty() {
            return None;
        }
        let t = make_f32_tensor(out_name, &new_dims, &vals, input.data_type);
        return Some(vec![(out_name.to_string(), t)]);
    }
    for &ax in &axes {
        if ax >= input.dims.len() || input.dims[ax] != 1 {
            return None;
        }
    }
    let new_dims: Vec<i64> = input
        .dims
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, &d)| d)
        .collect();
    let vals = tensor_to_f32(input);
    if vals.is_empty() {
        return None;
    }
    let t = make_f32_tensor(out_name, &new_dims, &vals, input.data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_unsqueeze(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let axes: Vec<i64> = if inputs.len() >= 2 {
        tensor_to_i64(inputs[1])
    } else {
        node.attribute
            .iter()
            .find(|a| a.name == "axes")
            .map(|a| a.ints.clone())
            .unwrap_or_default()
    };
    let ndim = inputs[0].dims.len() + axes.len();
    let mut new_dims = inputs[0].dims.clone();
    let mut sorted_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    sorted_axes.sort();
    for &ax in &sorted_axes {
        if ax <= new_dims.len() {
            new_dims.insert(ax, 1);
        }
    }
    let vals = tensor_to_f32(inputs[0]);
    if vals.is_empty() {
        return None;
    }
    let t = make_f32_tensor(out_name, &new_dims, &vals, inputs[0].data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_shape(
    node: &NodeProto,
    input: &TensorProto,
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let dims = &input.dims;
    if dims.is_empty() {
        return None;
    }
    let ndim = dims.len() as i64;
    let start_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "start")
        .map(|a| a.i)
        .unwrap_or(0);
    let end_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "end")
        .map(|a| a.i)
        .unwrap_or(ndim);
    let start = if start_attr < 0 {
        (ndim + start_attr).max(0) as usize
    } else {
        (start_attr as usize).min(dims.len())
    };
    let end = if end_attr < 0 {
        (ndim + end_attr).max(0) as usize
    } else {
        (end_attr as usize).min(dims.len())
    };
    let sliced: Vec<i64> = if start < end {
        dims[start..end].to_vec()
    } else {
        vec![]
    };
    let t = TensorProto {
        name: out_name.to_string(),
        data_type: TensorProto::INT64,
        dims: vec![sliced.len() as i64],
        int64_data: sliced,
        ..Default::default()
    };
    Some(vec![(out_name.to_string(), t)])
}

fn eval_gather(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    let axis = node
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
        .unwrap_or(0);
    let data = inputs[0];
    let indices = tensor_to_i64(inputs[1]);
    if indices.is_empty() || data.dims.is_empty() {
        return None;
    }
    if data.dims.len() == 1 && axis == 0 {
        let data_vals = tensor_to_f32(data);
        if data_vals.is_empty() {
            let data_i64 = tensor_to_i64(data);
            if data_i64.is_empty() {
                return None;
            }
            let result: Vec<i64> = indices
                .iter()
                .map(|&i| {
                    let idx = if i < 0 {
                        (data.dims[0] + i) as usize
                    } else {
                        i as usize
                    };
                    data_i64.get(idx).copied().unwrap_or(0)
                })
                .collect();
            let out_dims = if inputs[1].dims.is_empty() {
                vec![]
            } else {
                inputs[1].dims.clone()
            };
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT64,
                dims: out_dims,
                int64_data: result,
                ..Default::default()
            };
            return Some(vec![(out_name.to_string(), t)]);
        }
        let result: Vec<f32> = indices
            .iter()
            .map(|&i| {
                let idx = if i < 0 {
                    (data.dims[0] + i) as usize
                } else {
                    i as usize
                };
                data_vals.get(idx).copied().unwrap_or(0.0)
            })
            .collect();
        let out_dims = if inputs[1].dims.is_empty() {
            vec![]
        } else {
            inputs[1].dims.clone()
        };
        let t = make_f32_tensor(out_name, &out_dims, &result, data.data_type);
        return Some(vec![(out_name.to_string(), t)]);
    }
    None
}

fn eval_slice(inputs: &[&TensorProto], out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let data = inputs[0];
    let starts = tensor_to_i64(inputs[1]);
    let ends = tensor_to_i64(inputs[2]);
    if starts.is_empty() || ends.is_empty() {
        return None;
    }
    let axes: Vec<i64> = if inputs.len() > 3 {
        tensor_to_i64(inputs[3])
    } else {
        (0..starts.len() as i64).collect()
    };
    let steps: Vec<i64> = if inputs.len() > 4 {
        tensor_to_i64(inputs[4])
    } else {
        vec![1; starts.len()]
    };
    if starts.len() != ends.len() || axes.len() != starts.len() || steps.len() != starts.len() {
        return None;
    }
    let rank = data.dims.len();
    if rank == 0 {
        return None;
    }

    let mut per_axis_range: Vec<(i64, i64, i64)> = (0..rank as i64)
        .map(|d| (0, data.dims[d as usize], 1))
        .collect();
    for (i, &raw_axis) in axes.iter().enumerate() {
        let a = if raw_axis < 0 {
            rank as i64 + raw_axis
        } else {
            raw_axis
        };
        if a < 0 || a >= rank as i64 {
            return None;
        }
        let dim = data.dims[a as usize];
        let step = steps[i];
        if step == 0 {
            return None;
        }
        if dim == 0 {
            // Zero-length axis: any slice yields an empty output on that
            // axis.  Record (0, 0, step) and skip clamping to avoid the
            // clamp(..., 0, dim - 1) == clamp(..., 0, -1) inverted range.
            per_axis_range[a as usize] = (0, 0, step);
            continue;
        }
        let raw_start = starts[i];
        let raw_end = ends[i];
        let clamp = |v: i64, lo: i64, hi: i64| -> i64 { v.clamp(lo, hi) };
        let (s, e) = if step > 0 {
            // ONNX forward slice: start in [0, dim], end in [0, dim],
            // both treated as exclusive upper bound.
            let s = clamp(
                if raw_start < 0 {
                    dim + raw_start
                } else {
                    raw_start
                },
                0,
                dim,
            );
            let e = clamp(if raw_end < 0 { dim + raw_end } else { raw_end }, 0, dim);
            (s, e)
        } else {
            // ONNX reverse slice: start in [0, dim-1] (inclusive first
            // read), end in [-1, dim-1] (exclusive lower bound; -1
            // means "walk past index 0", i.e. include element 0).
            let s = clamp(
                if raw_start < 0 {
                    dim + raw_start
                } else {
                    raw_start
                },
                0,
                dim - 1,
            );
            let resolved_end = if raw_end == i64::MIN {
                -1
            } else if raw_end < 0 {
                dim + raw_end
            } else {
                raw_end
            };
            let e = clamp(resolved_end, -1, dim - 1);
            (s, e)
        };
        per_axis_range[a as usize] = (s, e, step);
    }

    let out_dims: Vec<i64> = per_axis_range
        .iter()
        .map(|(s, e, st)| {
            if *st > 0 {
                ((e - s + st - 1) / st).max(0)
            } else {
                ((s - e + (-st) - 1) / (-st)).max(0)
            }
        })
        .collect();
    let total = broadcast_total(&out_dims)?;
    if total == 0 {
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: data.data_type,
            dims: out_dims,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }

    let in_strides: Vec<i64> = {
        let mut s = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * data.dims[i + 1];
        }
        s
    };
    let out_strides: Vec<i64> = {
        let mut s = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_dims[i + 1];
        }
        s
    };

    let src_index = |o: i64| -> i64 {
        let mut rem = o;
        let mut src = 0i64;
        for d in 0..rank {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let (s_axis, _, st) = per_axis_range[d];
            src += (s_axis + coord * st) * in_strides[d];
        }
        src
    };

    if data.data_type == TensorProto::INT64 {
        let vals = tensor_to_i64(data);
        if vals.is_empty() {
            return None;
        }
        let mut result = Vec::with_capacity(total);
        for o in 0..total {
            result.push(*vals.get(src_index(o as i64) as usize)?);
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: out_dims,
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let vals = tensor_to_f32(data);
    if vals.is_empty() {
        return None;
    }
    let mut result = Vec::with_capacity(total);
    for o in 0..total {
        result.push(*vals.get(src_index(o as i64) as usize)?);
    }
    let t = make_f32_tensor(out_name, &out_dims, &result, data.data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_scatter_nd(inputs: &[&TensorProto], out_name: &str) -> Option<Vec<(String, TensorProto)>> {
    let data = inputs[0];
    let indices = inputs[1];
    let updates = inputs[2];
    let rank = data.dims.len();
    if rank == 0 || indices.dims.is_empty() {
        return None;
    }
    let q = *indices.dims.last()? as usize;
    if q == 0 || q > rank {
        return None;
    }
    let total = broadcast_total(&data.dims)?;
    let in_strides: Vec<i64> = {
        let mut s = vec![1i64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * data.dims[i + 1];
        }
        s
    };
    let trail_size: usize = data.dims[q..].iter().map(|&d| d as usize).product();
    let scatter_count: usize = indices.dims[..indices.dims.len() - 1]
        .iter()
        .map(|&d| d as usize)
        .product();
    let idx_vals = tensor_to_i64(indices);
    if idx_vals.len() != scatter_count * q {
        return None;
    }

    if data.data_type == TensorProto::INT64 {
        let mut buf = tensor_to_i64(data);
        if buf.len() != total {
            return None;
        }
        let upd_vals = tensor_to_i64(updates);
        if upd_vals.len() != scatter_count * trail_size {
            return None;
        }
        for s in 0..scatter_count {
            let mut base = 0i64;
            for d in 0..q {
                let mut idx = idx_vals[s * q + d];
                if idx < 0 {
                    idx += data.dims[d];
                }
                if idx < 0 || idx >= data.dims[d] {
                    return None;
                }
                base += idx * in_strides[d];
            }
            for k in 0..trail_size {
                buf[base as usize + k] = upd_vals[s * trail_size + k];
            }
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: data.dims.clone(),
            int64_data: buf,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }

    let mut buf = tensor_to_f32(data);
    if buf.len() != total {
        return None;
    }
    let upd_vals = tensor_to_f32(updates);
    if upd_vals.len() != scatter_count * trail_size {
        return None;
    }
    for s in 0..scatter_count {
        let mut base = 0i64;
        for d in 0..q {
            let mut idx = idx_vals[s * q + d];
            if idx < 0 {
                idx += data.dims[d];
            }
            if idx < 0 || idx >= data.dims[d] {
                return None;
            }
            base += idx * in_strides[d];
        }
        for k in 0..trail_size {
            buf[base as usize + k] = upd_vals[s * trail_size + k];
        }
    }
    let t = make_f32_tensor(out_name, &data.dims, &buf, data.data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn eval_split(
    node: &NodeProto,
    inputs: &[&TensorProto],
    output_names: &[String],
) -> Option<Vec<(String, TensorProto)>> {
    let data = inputs.first()?;
    let rank = data.dims.len();
    if rank == 0 {
        return None;
    }
    let raw_axis = node
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
        .unwrap_or(0);
    let axis = if raw_axis < 0 {
        rank as i64 + raw_axis
    } else {
        raw_axis
    } as usize;
    if axis >= rank {
        return None;
    }
    let split_sizes: Vec<i64> = if inputs.len() >= 2 {
        tensor_to_i64(inputs[1])
    } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "split") {
        attr.ints.clone()
    } else {
        let n = output_names.iter().filter(|s| !s.is_empty()).count() as i64;
        if n == 0 {
            return None;
        }
        let dim = data.dims[axis];
        if dim % n != 0 {
            return None;
        }
        vec![dim / n; n as usize]
    };
    if split_sizes.iter().sum::<i64>() != data.dims[axis] {
        return None;
    }
    let outputs: Vec<&str> = output_names
        .iter()
        .filter(|s| !s.is_empty())
        .map(String::as_str)
        .collect();
    if outputs.len() != split_sizes.len() {
        return None;
    }

    let prefix: usize = data.dims[..axis].iter().map(|&d| d as usize).product();
    let suffix: usize = data.dims[axis + 1..].iter().map(|&d| d as usize).product();
    let axis_in: usize = data.dims[axis] as usize;

    let mut result = Vec::with_capacity(outputs.len());
    let is_int64 = data.data_type == TensorProto::INT64;
    let mut offset = 0usize;
    for (i, &sz) in split_sizes.iter().enumerate() {
        let sz_us = usize::try_from(sz).ok()?;
        if sz_us == 0 {
            return None;
        }
        let mut out_dims = data.dims.clone();
        out_dims[axis] = sz;
        let total = prefix * sz_us * suffix;
        if is_int64 {
            let vals = tensor_to_i64(data);
            if vals.is_empty() {
                return None;
            }
            let mut chunk = Vec::with_capacity(total);
            for p in 0..prefix {
                for ai in 0..sz_us {
                    let src_axis = offset + ai;
                    let src_base = (p * axis_in + src_axis) * suffix;
                    chunk.extend_from_slice(&vals[src_base..src_base + suffix]);
                }
            }
            let t = TensorProto {
                name: outputs[i].to_string(),
                data_type: TensorProto::INT64,
                dims: out_dims,
                int64_data: chunk,
                ..Default::default()
            };
            result.push((outputs[i].to_string(), t));
        } else {
            let vals = tensor_to_f32(data);
            if vals.is_empty() {
                return None;
            }
            let mut chunk = Vec::with_capacity(total);
            for p in 0..prefix {
                for ai in 0..sz_us {
                    let src_axis = offset + ai;
                    let src_base = (p * axis_in + src_axis) * suffix;
                    chunk.extend_from_slice(&vals[src_base..src_base + suffix]);
                }
            }
            let t = make_f32_tensor(outputs[i], &out_dims, &chunk, data.data_type);
            result.push((outputs[i].to_string(), t));
        }
        offset += sz_us;
    }
    Some(result)
}

fn eval_concat(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    if inputs.is_empty() {
        return None;
    }
    let raw_axis = node
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
        .unwrap_or(0);
    let rank = inputs[0].dims.len();
    if !inputs.iter().all(|t| t.dims.len() == rank) {
        return None;
    }
    if rank == 0 {
        return None;
    }
    let axis = if raw_axis < 0 {
        (rank as i64 + raw_axis) as usize
    } else {
        raw_axis as usize
    };
    if axis >= rank {
        return None;
    }
    for d in 0..rank {
        if d == axis {
            continue;
        }
        let expected = inputs[0].dims[d];
        if !inputs.iter().all(|t| t.dims[d] == expected) {
            return None;
        }
    }
    let mut out_dims = inputs[0].dims.clone();
    out_dims[axis] = inputs.iter().map(|t| t.dims[axis]).sum();

    let prefix_size: usize = out_dims[..axis].iter().map(|&d| d as usize).product();
    let out_axis: usize = out_dims[axis] as usize;
    let suffix_size: usize = out_dims[axis + 1..].iter().map(|&d| d as usize).product();
    let out_total = prefix_size
        .checked_mul(out_axis)?
        .checked_mul(suffix_size)?;
    if out_total > MAX_BROADCAST_ELEMENTS {
        return None;
    }

    // ONNX Concat requires homogeneous input element types, so the first
    // input's declared type is authoritative.
    let is_int64 = inputs[0].data_type == TensorProto::INT64;

    if is_int64 {
        let mut result: Vec<i64> = vec![0; out_total];
        let mut axis_offset: usize = 0;
        for t in inputs {
            let t_vals = tensor_to_i64(t);
            let t_axis = t.dims[axis] as usize;
            if t_axis > 0 && t_vals.is_empty() {
                return None;
            }
            for p in 0..prefix_size {
                for ai in 0..t_axis {
                    for s in 0..suffix_size {
                        let src = (p * t_axis + ai) * suffix_size + s;
                        let dst = (p * out_axis + axis_offset + ai) * suffix_size + s;
                        result[dst] = t_vals[src];
                    }
                }
            }
            axis_offset += t_axis;
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: out_dims,
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }

    let mut result: Vec<f32> = vec![0.0; out_total];
    let mut axis_offset: usize = 0;
    for t in inputs {
        let t_vals = tensor_to_f32(t);
        let t_axis = t.dims[axis] as usize;
        if t_axis > 0 && t_vals.is_empty() {
            return None;
        }
        for p in 0..prefix_size {
            for ai in 0..t_axis {
                for s in 0..suffix_size {
                    let src = (p * t_axis + ai) * suffix_size + s;
                    let dst = (p * out_axis + axis_offset + ai) * suffix_size + s;
                    result[dst] = t_vals[src];
                }
            }
        }
        axis_offset += t_axis;
    }
    let t = make_f32_tensor(out_name, &out_dims, &result, inputs[0].data_type);
    Some(vec![(out_name.to_string(), t)])
}

fn make_f32_tensor(name: &str, dims: &[i64], vals: &[f32], target_type: i32) -> TensorProto {
    match target_type {
        TensorProto::INT64 => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::INT64,
            dims: dims.to_vec(),
            int64_data: vals.iter().map(|&v| v as i64).collect(),
            ..Default::default()
        },
        TensorProto::INT32 => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::INT32,
            dims: dims.to_vec(),
            int32_data: vals.iter().map(|&v| v as i32).collect(),
            ..Default::default()
        },
        TensorProto::DOUBLE => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::DOUBLE,
            dims: dims.to_vec(),
            double_data: vals.iter().map(|&v| v as f64).collect(),
            ..Default::default()
        },
        TensorProto::BOOL => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::BOOL,
            dims: dims.to_vec(),
            int32_data: vals.iter().map(|&v| (v != 0.0) as i32).collect(),
            ..Default::default()
        },
        _ => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::FLOAT,
            dims: dims.to_vec(),
            float_data: vals.to_vec(),
            ..Default::default()
        },
    }
}

struct ConvBnFusion {
    conv_idx: usize,
    bn_idx: usize,
    bn_output: String,
    w_name: String,
    bias_name: String,
    has_bias: bool,
    orig_bias: Vec<f32>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
    eps: f32,
    // Initialiser names that become dead after the fusion: the BN's
    // four parameter inputs (gamma / beta / running mean / running
    // variance) and, if the Conv had no bias before fusion, the
    // auto-named "<w>_fused_bias" we create.  Collected here so the
    // caller can purge them in a single post-pass sweep without
    // re-walking every BN node.
    stale_bn_param_names: Vec<String>,
}

pub fn fuse_conv_batchnorm(graph: &mut GraphProto) -> usize {
    let fusions = {
        let init_map: HashMap<&str, &TensorProto> = graph
            .initializer
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        let node_output_map: HashMap<&str, usize> = graph
            .node
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.output.iter().map(move |o| (o.as_str(), i)))
            .collect();

        let mut fusions: Vec<ConvBnFusion> = Vec::new();

        for (bn_idx, bn_node) in graph.node.iter().enumerate() {
            if bn_node.op_type != "BatchNormalization" || bn_node.input.len() < 5 {
                continue;
            }
            let bn_input = &bn_node.input[0];
            let conv_idx = match node_output_map.get(bn_input.as_str()) {
                Some(&idx) => idx,
                None => continue,
            };
            let conv_node = &graph.node[conv_idx];
            if conv_node.op_type != "Conv" || conv_node.output.is_empty() {
                continue;
            }
            let consumers: usize = graph
                .node
                .iter()
                .filter(|n| n.input.contains(&conv_node.output[0]))
                .count();
            if consumers != 1 {
                continue;
            }

            let gamma = match init_map.get(bn_node.input[1].as_str()) {
                Some(t) => tensor_to_f32(t),
                None => continue,
            };
            let beta = match init_map.get(bn_node.input[2].as_str()) {
                Some(t) => tensor_to_f32(t),
                None => continue,
            };
            let mean = match init_map.get(bn_node.input[3].as_str()) {
                Some(t) => tensor_to_f32(t),
                None => continue,
            };
            let var = match init_map.get(bn_node.input[4].as_str()) {
                Some(t) => tensor_to_f32(t),
                None => continue,
            };

            if gamma.is_empty()
                || gamma.len() != beta.len()
                || gamma.len() != mean.len()
                || gamma.len() != var.len()
            {
                continue;
            }

            let bn_output = match bn_node.output.first() {
                Some(o) if !o.is_empty() => o.clone(),
                _ => continue,
            };

            let eps = bn_node
                .attribute
                .iter()
                .find(|a| a.name == "epsilon")
                .map(|a| a.f)
                .unwrap_or(1e-5);

            let w_name = conv_node.input[1].clone();
            let has_bias = conv_node.input.len() > 2;
            let bias_name = if has_bias {
                conv_node.input[2].clone()
            } else {
                format!("{}_fused_bias", w_name)
            };
            let orig_bias = if has_bias {
                init_map
                    .get(conv_node.input[2].as_str())
                    .map(|t| tensor_to_f32(t))
                    .unwrap_or_default()
            } else {
                vec![]
            };

            let stale_bn_param_names = vec![
                bn_node.input[1].clone(),
                bn_node.input[2].clone(),
                bn_node.input[3].clone(),
                bn_node.input[4].clone(),
            ];

            fusions.push(ConvBnFusion {
                conv_idx,
                bn_idx,
                bn_output,
                w_name,
                bias_name,
                has_bias,
                orig_bias,
                gamma,
                beta,
                mean,
                var,
                eps,
                stale_bn_param_names,
            });
        }

        fusions
    };

    if fusions.is_empty() {
        return 0;
    }

    let mut removed_bn: HashSet<usize> = HashSet::new();
    let mut stale_init_names: HashSet<String> = HashSet::new();

    for f in &fusions {
        let channels = f.gamma.len();
        let scale: Vec<f32> = (0..channels)
            .map(|c| f.gamma[c] / (f.var[c] + f.eps).sqrt())
            .collect();

        let w_ok = if let Some(w_init) = graph.initializer.iter_mut().find(|i| i.name == f.w_name) {
            let mut w_data = tensor_to_f32(w_init);
            // tensor_to_f32 returns empty for unsupported dtypes
            // (e.g. f16 / bf16 weights we don't yet convert); skip
            // the fusion for this Conv rather than silently clearing
            // the initializer into a zero-length FLOAT tensor that
            // would fail every downstream shape check.
            if w_data.is_empty() {
                false
            } else if !w_init.dims.is_empty() && w_init.dims[0] as usize == channels {
                let per_filter = w_data.len() / channels;
                for c in 0..channels {
                    for j in 0..per_filter {
                        w_data[c * per_filter + j] *= scale[c];
                    }
                }
                w_init.float_data = w_data;
                w_init.raw_data.clear();
                // The initialiser may have arrived as half / bfloat
                // encoded in raw_data; float_data is FLOAT by
                // definition, so stamp the tensor metadata to match
                // the new representation.
                w_init.data_type = TensorProto::FLOAT;
                true
            } else {
                false
            }
        } else {
            false
        };
        if !w_ok {
            continue;
        }

        let fused_bias: Vec<f32> = (0..channels)
            .map(|c| {
                let ob = f.orig_bias.get(c).copied().unwrap_or(0.0);
                (ob - f.mean[c]) * scale[c] + f.beta[c]
            })
            .collect();

        if let Some(b_init) = graph.initializer.iter_mut().find(|i| i.name == f.bias_name) {
            b_init.float_data = fused_bias;
            b_init.raw_data.clear();
            b_init.dims = vec![channels as i64];
            b_init.data_type = TensorProto::FLOAT;
        } else {
            graph.initializer.push(TensorProto {
                name: f.bias_name.clone(),
                data_type: TensorProto::FLOAT,
                dims: vec![channels as i64],
                float_data: fused_bias,
                ..Default::default()
            });
        }

        let conv_node = &mut graph.node[f.conv_idx];
        if !f.has_bias {
            conv_node.input.push(f.bias_name.clone());
        }
        conv_node.output[0] = f.bn_output.clone();

        removed_bn.insert(f.bn_idx);
        stale_init_names.extend(f.stale_bn_param_names.iter().cloned());
    }

    if !removed_bn.is_empty() {
        let mut idx = 0;
        graph.node.retain(|_| {
            let keep = !removed_bn.contains(&idx);
            idx += 1;
            keep
        });
    }

    if !stale_init_names.is_empty() {
        // Only drop BN parameter initialisers that no surviving node
        // still references.  Rare in practice but cheap to verify
        // and prevents accidentally deleting an initialiser shared
        // between a fused Conv+BN and an unrelated node elsewhere.
        let still_used: HashSet<&str> = graph
            .node
            .iter()
            .flat_map(|n| n.input.iter().map(String::as_str))
            .collect();
        graph.initializer.retain(|init| {
            !stale_init_names.contains(&init.name) || still_used.contains(init.name.as_str())
        });
    }

    removed_bn.len()
}
