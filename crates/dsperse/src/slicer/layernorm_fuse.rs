use std::collections::{HashMap, HashSet};

use super::onnx_proto::{
    AttributeProto, ModelProto, NodeProto, TensorProto, tensor_to_f32, tensor_to_i64,
};

pub fn fuse_inline_layernorms(
    model: &mut ModelProto,
    traced_shapes: &mut HashMap<String, Vec<i64>>,
) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };

    let initializers: HashMap<String, TensorProto> = graph
        .initializer
        .iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();

    let producers: HashMap<String, usize> = graph
        .node
        .iter()
        .enumerate()
        .flat_map(|(i, n)| {
            n.output
                .iter()
                .filter(|o| !o.is_empty())
                .map(move |o| (o.clone(), i))
        })
        .collect();

    let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, n) in graph.node.iter().enumerate() {
        for inp in &n.input {
            if !inp.is_empty() {
                consumers.entry(inp.clone()).or_default().push(i);
            }
        }
    }

    let mut drop: HashSet<usize> = HashSet::new();
    let mut insertions: Vec<(usize, Vec<NodeProto>, Vec<TensorProto>)> = Vec::new();
    let mut fused_id = 0usize;

    for (mean_idx, mean_node) in graph.node.iter().enumerate() {
        if drop.contains(&mean_idx) || mean_node.op_type != "ReduceMean" {
            continue;
        }
        let Some(m) = try_match_layernorm(
            mean_idx,
            mean_node,
            &graph.node,
            &producers,
            &consumers,
            &initializers,
            traced_shapes,
            &drop,
        ) else {
            continue;
        };
        let (nodes, inits, shapes) = emit_replacement(&m, fused_id, &initializers);
        for (name, shape) in shapes {
            traced_shapes.insert(name, shape);
        }
        fused_id += 1;
        drop.extend(m.nodes_to_drop.iter().copied());
        insertions.push((mean_idx, nodes, inits));
    }

    let fused = insertions.len();
    if fused == 0 {
        return 0;
    }

    for (_, _, inits) in &insertions {
        for t in inits {
            graph.initializer.push(t.clone());
        }
    }

    let insertion_map: HashMap<usize, Vec<NodeProto>> = insertions
        .into_iter()
        .map(|(idx, nodes, _)| (idx, nodes))
        .collect();

    let mut new_nodes: Vec<NodeProto> = Vec::with_capacity(graph.node.len());
    for (i, n) in graph.node.drain(..).enumerate() {
        if let Some(inserts) = insertion_map.get(&i) {
            new_nodes.extend(inserts.iter().cloned());
            continue;
        }
        if drop.contains(&i) {
            continue;
        }
        new_nodes.push(n);
    }
    graph.node = new_nodes;
    fused
}

struct MatchedPattern {
    x_name: String,
    axes: Vec<usize>,
    rank: usize,
    x_shape: Vec<i64>,
    eps: f32,
    scale_init: Option<String>,
    bias_init: Option<String>,
    output_name: String,
    nodes_to_drop: Vec<usize>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn try_match_layernorm(
    mean_idx: usize,
    mean_node: &NodeProto,
    nodes: &[NodeProto],
    producers: &HashMap<String, usize>,
    consumers: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, TensorProto>,
    traced_shapes: &HashMap<String, Vec<i64>>,
    drop: &HashSet<usize>,
) -> Option<MatchedPattern> {
    let raw_axes = reduce_axes(mean_node, initializers)?;
    if get_keepdims(mean_node).unwrap_or(1) != 1 {
        return None;
    }
    let x_name = mean_node.input.first()?.clone();
    let mean_out = mean_node.output.first()?.clone();

    let sub_idx = find_unique_consumer(consumers, &mean_out, "Sub", nodes, drop)?;
    let sub_node = &nodes[sub_idx];
    if sub_node.input.len() < 2
        || sub_node.input.first()? != &x_name
        || sub_node.input.get(1)? != &mean_out
    {
        return None;
    }
    let centered = sub_node.output.first()?.clone();

    let sq_idx = find_square_consumer(consumers, &centered, nodes, initializers, drop)?;
    let sq_node = &nodes[sq_idx];
    let sq_out = sq_node.output.first()?.clone();

    let mean2_idx = find_unique_consumer(consumers, &sq_out, "ReduceMean", nodes, drop)?;
    let mean2_node = &nodes[mean2_idx];
    let raw_axes2 = reduce_axes(mean2_node, initializers)?;
    if raw_axes2 != raw_axes {
        return None;
    }
    if get_keepdims(mean2_node).unwrap_or(1) != 1 {
        return None;
    }
    let var_out = mean2_node.output.first()?.clone();

    let add_idx = find_unique_consumer(consumers, &var_out, "Add", nodes, drop)?;
    let add_node = &nodes[add_idx];
    let eps = extract_binary_const_scalar(add_node, &var_out, initializers)?;
    let var_eps = add_node.output.first()?.clone();

    let sqrt_idx = find_unique_consumer(consumers, &var_eps, "Sqrt", nodes, drop)?;
    let sqrt_node = &nodes[sqrt_idx];
    let std_out = sqrt_node.output.first()?.clone();

    let div_idx = find_unique_consumer(consumers, &std_out, "Div", nodes, drop)?;
    let div_node = &nodes[div_idx];
    if div_node.input.len() < 2
        || div_node.input.first()? != &centered
        || div_node.input.get(1)? != &std_out
    {
        return None;
    }
    let norm_out = div_node.output.first()?.clone();

    let mut nodes_to_drop = vec![
        mean_idx, sub_idx, sq_idx, mean2_idx, add_idx, sqrt_idx, div_idx,
    ];
    let mut output_name = norm_out.clone();
    let mut scale_init: Option<String> = None;
    let mut bias_init: Option<String> = None;

    if let Some(mul_idx) = find_unique_consumer(consumers, &norm_out, "Mul", nodes, drop) {
        let mul_node = &nodes[mul_idx];
        if let Some(scale) = other_input_if_init(mul_node, &norm_out, initializers) {
            scale_init = Some(scale);
            output_name = mul_node.output.first()?.clone();
            nodes_to_drop.push(mul_idx);

            if let Some(add2_idx) =
                find_unique_consumer(consumers, &output_name, "Add", nodes, drop)
            {
                let add2_node = &nodes[add2_idx];
                if let Some(bias) = other_input_if_init(add2_node, &output_name, initializers) {
                    bias_init = Some(bias);
                    output_name = add2_node.output.first()?.clone();
                    nodes_to_drop.push(add2_idx);
                }
            }
        }
    }

    // Soundness check: every intermediate tensor we are about to drop
    // (mean_out, centered, sq_out, var_out, var_eps, std_out, plus the
    // pre-affine norm_out when scale/bias are present) must have all
    // its live consumers inside nodes_to_drop.  Otherwise some
    // downstream node still reads the intermediate and fusing would
    // disconnect it.
    let drop_set: HashSet<usize> = nodes_to_drop.iter().copied().collect();
    let mut intermediates: Vec<&str> = vec![
        mean_out.as_str(),
        centered.as_str(),
        sq_out.as_str(),
        var_out.as_str(),
        var_eps.as_str(),
        std_out.as_str(),
    ];
    if scale_init.is_some() {
        intermediates.push(norm_out.as_str());
    }
    for tname in intermediates {
        if let Some(list) = consumers.get(tname) {
            for &idx in list {
                if drop.contains(&idx) || drop_set.contains(&idx) {
                    continue;
                }
                return None;
            }
        }
    }

    let x_shape = resolve_shape(&x_name, traced_shapes, initializers, nodes, producers)?;
    let rank = x_shape.len();
    if rank == 0 {
        return None;
    }
    let axes: Vec<usize> = raw_axes.iter().map(|&a| normalize_axis(a, rank)).collect();
    for &a in &axes {
        if a >= rank {
            return None;
        }
        // Reject dynamic / unresolved dims along the reduction axes: the
        // fused LayerNormalization circuit needs a concrete lane_size
        // and consumers of m.x_shape[a] later cast the dim to usize,
        // which silently wraps negative sentinels into huge values.
        if x_shape[a] <= 0 {
            return None;
        }
    }

    Some(MatchedPattern {
        x_name,
        axes,
        rank,
        x_shape,
        eps,
        scale_init,
        bias_init,
        output_name,
        nodes_to_drop,
    })
}

fn resolve_shape(
    name: &str,
    traced_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, TensorProto>,
    _nodes: &[NodeProto],
    _producers: &HashMap<String, usize>,
) -> Option<Vec<i64>> {
    if let Some(s) = traced_shapes.get(name)
        && !s.is_empty()
    {
        return Some(s.clone());
    }
    if let Some(t) = initializers.get(name) {
        return Some(t.dims.clone());
    }
    None
}

fn reduce_axes(node: &NodeProto, initializers: &HashMap<String, TensorProto>) -> Option<Vec<i64>> {
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes")
        && !attr.ints.is_empty()
    {
        return Some(attr.ints.clone());
    }
    if let Some(name) = node.input.get(1)
        && let Some(t) = initializers.get(name)
    {
        let v = tensor_to_i64(t);
        if !v.is_empty() {
            return Some(v);
        }
    }
    None
}

fn get_keepdims(node: &NodeProto) -> Option<i64> {
    node.attribute
        .iter()
        .find(|a| a.name == "keepdims")
        .map(|a| a.i)
}

fn find_unique_consumer(
    consumers: &HashMap<String, Vec<usize>>,
    tensor: &str,
    op_type: &str,
    nodes: &[NodeProto],
    drop: &HashSet<usize>,
) -> Option<usize> {
    let list = consumers.get(tensor)?;
    let live: Vec<usize> = list.iter().copied().filter(|i| !drop.contains(i)).collect();
    if live.len() != 1 {
        return None;
    }
    let idx = live[0];
    (nodes[idx].op_type == op_type).then_some(idx)
}

fn find_square_consumer(
    consumers: &HashMap<String, Vec<usize>>,
    tensor: &str,
    nodes: &[NodeProto],
    initializers: &HashMap<String, TensorProto>,
    drop: &HashSet<usize>,
) -> Option<usize> {
    // The centered tensor in the inline-LN pattern has TWO legitimate
    // consumers: Pow / Mul (for the variance branch) AND Div (for the
    // normalization branch).  Both belong to the fusion -- don't reject
    // them as orphan consumers.  Final orphan-leak check happens after
    // the whole pattern matches in try_match_layernorm.
    let list = consumers.get(tensor)?;
    for &idx in list.iter().filter(|i| !drop.contains(i)) {
        let n = &nodes[idx];
        match n.op_type.as_str() {
            "Pow" => {
                if n.input.len() >= 2
                    && n.input.first().map(String::as_str) == Some(tensor)
                    && pow_exponent_is_two(n.input.get(1)?, initializers)
                {
                    return Some(idx);
                }
            }
            "Mul" => {
                if n.input.len() == 2 && n.input.iter().all(|i| i == tensor) {
                    return Some(idx);
                }
            }
            _ => {}
        }
    }
    None
}

fn pow_exponent_is_two(name: &str, initializers: &HashMap<String, TensorProto>) -> bool {
    let Some(t) = initializers.get(name) else {
        return false;
    };
    let f = tensor_to_f32(t);
    if let Some(&v) = f.first()
        && (v - 2.0).abs() < f32::EPSILON
    {
        return true;
    }
    let i = tensor_to_i64(t);
    matches!(i.first(), Some(&2))
}

fn extract_binary_const_scalar(
    node: &NodeProto,
    non_const_input: &str,
    initializers: &HashMap<String, TensorProto>,
) -> Option<f32> {
    if node.input.len() != 2 {
        return None;
    }
    let (a, b) = (node.input.first()?, node.input.get(1)?);
    let other_name = if a.as_str() == non_const_input {
        b
    } else if b.as_str() == non_const_input {
        a
    } else {
        return None;
    };
    let t = initializers.get(other_name)?;
    tensor_to_f32(t).first().copied()
}

fn other_input_if_init(
    node: &NodeProto,
    non_const_input: &str,
    initializers: &HashMap<String, TensorProto>,
) -> Option<String> {
    if node.input.len() != 2 {
        return None;
    }
    let a = node.input.first()?.clone();
    let b = node.input.get(1)?.clone();
    let other = if a == non_const_input {
        b
    } else if b == non_const_input {
        a
    } else {
        return None;
    };
    initializers.get(&other).map(|_| other)
}

type ReplacementShapes = Vec<(String, Vec<i64>)>;
type Replacement = (Vec<NodeProto>, Vec<TensorProto>, ReplacementShapes);

fn emit_replacement(
    m: &MatchedPattern,
    fused_id: usize,
    initializers: &HashMap<String, TensorProto>,
) -> Replacement {
    let rank = m.rank;
    let axes_set: HashSet<usize> = m.axes.iter().copied().collect();
    let mut forward_perm: Vec<i64> = (0..rank)
        .filter(|d| !axes_set.contains(d))
        .map(|d| d as i64)
        .collect();
    for &a in &m.axes {
        forward_perm.push(a as i64);
    }
    let mut inverse_perm: Vec<i64> = vec![0; rank];
    for (new_pos, &old_pos) in forward_perm.iter().enumerate() {
        inverse_perm[old_pos as usize] = new_pos as i64;
    }

    let lane_size: usize = m.axes.iter().map(|&a| m.x_shape[a] as usize).product();

    let prefix = format!("/__dsperse/fused_ln_{fused_id}");
    let xt_name = format!("{prefix}/xt");
    let yt_name = format!("{prefix}/yt");

    let (scale_name, scale_init_opt) = materialize_1d_initializer(
        &format!("{prefix}/scale"),
        m.scale_init.as_deref(),
        initializers,
        lane_size,
        1.0,
    );
    let (bias_name, bias_init_opt) = materialize_1d_initializer(
        &format!("{prefix}/bias"),
        m.bias_init.as_deref(),
        initializers,
        lane_size,
        0.0,
    );

    let mut nodes = Vec::new();

    nodes.push(NodeProto {
        name: format!("{prefix}/Transpose_in"),
        op_type: "Transpose".to_string(),
        input: vec![m.x_name.clone()],
        output: vec![xt_name.clone()],
        attribute: vec![int_list_attr("perm", &forward_perm)],
        ..Default::default()
    });

    let ln_axis = (rank - m.axes.len()) as i64;
    nodes.push(NodeProto {
        name: format!("{prefix}/LayerNormalization"),
        op_type: "LayerNormalization".to_string(),
        input: vec![xt_name, scale_name, bias_name],
        output: vec![yt_name.clone()],
        attribute: vec![int_attr("axis", ln_axis), float_attr("epsilon", m.eps)],
        ..Default::default()
    });

    nodes.push(NodeProto {
        name: format!("{prefix}/Transpose_out"),
        op_type: "Transpose".to_string(),
        input: vec![yt_name],
        output: vec![m.output_name.clone()],
        attribute: vec![int_list_attr("perm", &inverse_perm)],
        ..Default::default()
    });

    let mut inits = Vec::new();
    inits.extend(scale_init_opt);
    inits.extend(bias_init_opt);

    let xt_shape: Vec<i64> = forward_perm
        .iter()
        .map(|&p| m.x_shape[p as usize])
        .collect();
    let yt_shape = xt_shape.clone();
    let shapes = vec![
        (format!("{prefix}/xt"), xt_shape),
        (format!("{prefix}/yt"), yt_shape),
    ];

    (nodes, inits, shapes)
}

fn materialize_1d_initializer(
    new_name: &str,
    source: Option<&str>,
    initializers: &HashMap<String, TensorProto>,
    lane_size: usize,
    default_fill: f32,
) -> (String, Option<TensorProto>) {
    let Some(src) = source else {
        return (
            new_name.to_string(),
            Some(const_vector(new_name, lane_size, default_fill)),
        );
    };
    let Some(t) = initializers.get(src) else {
        return (
            new_name.to_string(),
            Some(const_vector(new_name, lane_size, default_fill)),
        );
    };
    let elems = tensor_to_f32(t);
    if elems.len() == lane_size && t.dims.len() == 1 {
        return (src.to_string(), None);
    }
    let vals: Vec<f32> = if elems.len() == lane_size {
        elems
    } else if elems.len() == 1 {
        vec![elems[0]; lane_size]
    } else {
        return (
            new_name.to_string(),
            Some(const_vector(new_name, lane_size, default_fill)),
        );
    };
    (new_name.to_string(), Some(make_f32_vector(new_name, &vals)))
}

fn const_vector(name: &str, len: usize, fill: f32) -> TensorProto {
    make_f32_vector(name, &vec![fill; len])
}

fn make_f32_vector(name: &str, vals: &[f32]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: TensorProto::FLOAT,
        dims: vec![vals.len() as i64],
        float_data: vals.to_vec(),
        ..Default::default()
    }
}

fn normalize_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    }
}

fn int_attr(name: &str, v: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: 2,
        i: v,
        ..Default::default()
    }
}

fn float_attr(name: &str, v: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: 1,
        f: v,
        ..Default::default()
    }
}

fn int_list_attr(name: &str, vals: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: 7,
        ints: vals.to_vec(),
        ..Default::default()
    }
}
