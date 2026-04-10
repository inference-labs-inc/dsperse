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

    folded_names
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
        _ => None,
    }
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
    if data.dims.len() == 1 && axes == [0] && steps.iter().all(|&s| s == 1) {
        let dim = data.dims[0];
        let start = if starts[0] < 0 {
            (dim + starts[0]).max(0) as usize
        } else {
            (starts[0] as usize).min(dim as usize)
        };
        let end = if ends[0] < 0 {
            (dim + ends[0]).max(0) as usize
        } else {
            (ends[0] as usize).min(dim as usize)
        };
        if start >= end {
            return None;
        }
        if data.data_type == TensorProto::INT64 {
            let vals = tensor_to_i64(data);
            let sliced: Vec<i64> = vals.get(start..end)?.to_vec();
            let t = TensorProto {
                name: out_name.to_string(),
                data_type: TensorProto::INT64,
                dims: vec![(end - start) as i64],
                int64_data: sliced,
                ..Default::default()
            };
            return Some(vec![(out_name.to_string(), t)]);
        }
        let vals = tensor_to_f32(data);
        let sliced: Vec<f32> = vals.get(start..end)?.to_vec();
        let t = make_f32_tensor(out_name, &[(end - start) as i64], &sliced, data.data_type);
        return Some(vec![(out_name.to_string(), t)]);
    }
    None
}

fn eval_concat(
    node: &NodeProto,
    inputs: &[&TensorProto],
    out_name: &str,
) -> Option<Vec<(String, TensorProto)>> {
    // axis attribute is read but only 1-D concat is evaluated below;
    // multi-dimensional concat returns None and falls through to tract.
    let _axis = node
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
        .unwrap_or(0);
    if inputs.is_empty() {
        return None;
    }
    let all_1d = inputs.iter().all(|t| t.dims.len() <= 1);
    if !all_1d {
        return None;
    }
    if inputs[0].data_type == TensorProto::INT64
        || inputs.iter().all(|t| !tensor_to_i64(t).is_empty())
    {
        let mut result = Vec::new();
        for t in inputs {
            result.extend(tensor_to_i64(t));
        }
        let t = TensorProto {
            name: out_name.to_string(),
            data_type: TensorProto::INT64,
            dims: vec![result.len() as i64],
            int64_data: result,
            ..Default::default()
        };
        return Some(vec![(out_name.to_string(), t)]);
    }
    let mut result = Vec::new();
    for t in inputs {
        let vals = tensor_to_f32(t);
        if vals.is_empty() {
            return None;
        }
        result.extend(vals);
    }
    let t = make_f32_tensor(
        out_name,
        &[result.len() as i64],
        &result,
        inputs[0].data_type,
    );
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
        _ => TensorProto {
            name: name.to_string(),
            data_type: TensorProto::FLOAT,
            dims: dims.to_vec(),
            float_data: vals.to_vec(),
            ..Default::default()
        },
    }
}
