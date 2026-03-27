use std::collections::{HashMap, HashSet};

use super::onnx_proto::{ModelProto, NodeProto, TensorProto};

#[derive(Clone, Debug)]
enum ConstVal {
    F32(Vec<f32>, Vec<i64>),
    I64(Vec<i64>, Vec<i64>),
}

impl ConstVal {
    fn from_tensor(t: &TensorProto) -> Option<Self> {
        let dims = t.dims.clone();
        if let Some(v) = Self::extract_f32(t) {
            return Some(Self::F32(v, dims));
        }
        if let Some(v) = Self::extract_i64(t) {
            return Some(Self::I64(v, dims));
        }
        if !dims.is_empty() {
            return Some(Self::F32(vec![], dims));
        }
        None
    }

    fn into_tensor(self, name: &str) -> TensorProto {
        match self {
            Self::F32(data, dims) => TensorProto {
                name: name.to_string(),
                data_type: TensorProto::FLOAT,
                dims,
                float_data: data,
                ..Default::default()
            },
            Self::I64(data, dims) => TensorProto {
                name: name.to_string(),
                data_type: TensorProto::INT64,
                dims,
                int64_data: data,
                ..Default::default()
            },
        }
    }

    fn dims(&self) -> &[i64] {
        match self {
            Self::F32(_, d) | Self::I64(_, d) => d,
        }
    }

    fn as_f32(&self) -> Option<Vec<f32>> {
        match self {
            Self::F32(v, _) => Some(v.clone()),
            Self::I64(v, _) => Some(v.iter().map(|&i| i as f32).collect()),
        }
    }

    fn as_i64(&self) -> Option<Vec<i64>> {
        match self {
            Self::I64(v, _) => Some(v.clone()),
            Self::F32(v, _) => Some(v.iter().map(|&f| f as i64).collect()),
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

pub fn propagate_constants(model: &mut ModelProto) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };

    let mut known: HashMap<String, ConstVal> = HashMap::new();
    for init in &graph.initializer {
        if let Some(val) = ConstVal::from_tensor(init) {
            known.insert(init.name.clone(), val);
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
            if !can_evaluate(node, &known) {
                continue;
            }
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
            }
        }
    }

    if evaluated.is_empty() {
        return 0;
    }

    let existing: HashSet<String> = graph.initializer.iter().map(|i| i.name.clone()).collect();
    let mut new_inits: Vec<TensorProto> = Vec::new();
    for idx in &evaluated {
        for out in &graph.node[*idx].output {
            if !out.is_empty()
                && !existing.contains(out)
                && let Some(val) = known.get(out)
            {
                new_inits.push(val.clone().into_tensor(out));
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
    count
}

fn can_evaluate(node: &NodeProto, known: &HashMap<String, ConstVal>) -> bool {
    if node.output.is_empty() || node.output.iter().all(|o| o.is_empty()) {
        return false;
    }
    if node.output.iter().all(|o| known.contains_key(o)) {
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
        "Gather" => eval_gather(inputs),
        "Slice" => eval_slice(inputs),
        "Cast" => eval_cast(node, inputs),
        "Sqrt" => map_f32(inputs, |v| v.sqrt()),
        "Neg" => map_f32(inputs, |v| -v),
        "Exp" => map_f32(inputs, |v| v.exp()),
        "Add" => binary_f32(inputs, |a, b| a + b),
        "Sub" => binary_f32(inputs, |a, b| a - b),
        "Mul" => binary_f32(inputs, |a, b| a * b),
        "Div" => binary_f32(inputs, |a, b| if b != 0.0 { a / b } else { 0.0 }),
        "Unsqueeze" => eval_unsqueeze(node, inputs),
        "Concat" => eval_concat(node, inputs),
        "Reshape" => eval_reshape(inputs),
        "ConstantOfShape" => eval_constant_of_shape(node, inputs),
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

fn eval_shape(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let rank = t.dims().len() as i64;
    let s = normalize_idx(attr_int(node, "start").unwrap_or(0), rank);
    let e = normalize_idx(attr_int(node, "end").unwrap_or(rank), rank);
    let vals: Vec<i64> = t.dims()[s..e].to_vec();
    Some(ConstVal::I64(vals.clone(), vec![vals.len() as i64]))
}

fn eval_gather(inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let data = inp(inputs, 0)?.as_i64()?;
    let indices = inp(inputs, 1)?.as_i64()?;
    let result: Vec<i64> = indices
        .iter()
        .map(|&i| {
            let idx = normalize_idx(i, data.len() as i64);
            data.get(idx).copied().unwrap_or(0)
        })
        .collect();
    let dims = inp(inputs, 1)?.dims().to_vec();
    Some(ConstVal::I64(result, dims))
}

fn eval_slice(inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let data = inp(inputs, 0)?;
    if data.dims().len() != 1 {
        return None;
    }
    let vals = data.as_i64()?;
    let starts = inp(inputs, 1)?.as_i64()?;
    let ends = inp(inputs, 2)?.as_i64()?;
    let len = vals.len() as i64;
    let s = normalize_idx(*starts.first()?, len);
    let e = normalize_idx(*ends.first()?, len);
    let result: Vec<i64> = vals[s..e].to_vec();
    Some(ConstVal::I64(result.clone(), vec![result.len() as i64]))
}

fn eval_cast(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let to = attr_int(node, "to")? as i32;
    let dims = t.dims().to_vec();
    if to == TensorProto::FLOAT {
        Some(ConstVal::F32(t.as_f32()?, dims))
    } else if to == TensorProto::INT64 {
        Some(ConstVal::I64(t.as_i64()?, dims))
    } else {
        None
    }
}

fn map_f32(inputs: &[Option<&ConstVal>], f: impl Fn(f32) -> f32) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let vals = t.as_f32()?;
    let result: Vec<f32> = vals.iter().map(|&v| f(v)).collect();
    Some(ConstVal::F32(result, t.dims().to_vec()))
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
    } else if av.len() == bv.len() {
        av.iter().zip(&bv).map(|(&a, &b)| f(a, b)).collect()
    } else {
        return None;
    };
    let dims = if a.dims().len() >= b.dims().len() {
        a.dims()
    } else {
        b.dims()
    };
    Some(ConstVal::F32(result, dims.to_vec()))
}

fn eval_unsqueeze(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let t = inp(inputs, 0)?;
    let axes = if let Some(ax) = inputs.get(1).and_then(|o| o.as_ref()) {
        ax.as_i64()?
    } else {
        attr_ints(node, "axes")?
    };
    let mut dims = t.dims().to_vec();
    let mut sorted = axes;
    sorted.sort();
    for &ax in &sorted {
        let pos = normalize_idx(ax, dims.len() as i64 + 1);
        dims.insert(pos.min(dims.len()), 1);
    }
    match t {
        ConstVal::F32(v, _) => Some(ConstVal::F32(v.clone(), dims)),
        ConstVal::I64(v, _) => Some(ConstVal::I64(v.clone(), dims)),
    }
}

fn eval_concat(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let axis = attr_int(node, "axis").unwrap_or(0);
    let first = inp(inputs, 0)?;
    if first.dims().len() != 1 || axis != 0 {
        return None;
    }
    let mut result: Vec<i64> = Vec::new();
    for i in inputs {
        result.extend(i.as_ref()?.as_i64()?);
    }
    Some(ConstVal::I64(result.clone(), vec![result.len() as i64]))
}

fn eval_reshape(inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let data = inp(inputs, 0)?;
    let shape = inp(inputs, 1)?.as_i64()?;
    match data {
        ConstVal::F32(v, _) => Some(ConstVal::F32(v.clone(), shape)),
        ConstVal::I64(v, _) => Some(ConstVal::I64(v.clone(), shape)),
    }
}

fn eval_constant_of_shape(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let shape = inp(inputs, 0)?.as_i64()?;
    let total: usize = shape.iter().map(|&d| d.max(0) as usize).product();
    let value_tensor = node
        .attribute
        .iter()
        .find(|a| a.name == "value")?
        .t
        .as_ref()?;

    if let Some(fv) = ConstVal::extract_f32(value_tensor) {
        let fill = fv.first().copied().unwrap_or(0.0);
        return Some(ConstVal::F32(vec![fill; total], shape));
    }
    if let Some(iv) = ConstVal::extract_i64(value_tensor) {
        let fill = iv.first().copied().unwrap_or(0);
        return Some(ConstVal::I64(vec![fill; total], shape));
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
        let count = propagate_constants(&mut model);

        assert!(count > 0);
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

        let count = propagate_constants(&mut model);
        assert_eq!(count, 4);

        let graph = model.graph.as_ref().unwrap();
        assert!(graph.node.is_empty());
        let init = graph
            .initializer
            .iter()
            .find(|i| i.name == "sqrt_out")
            .expect("sqrt_out initializer");
        assert!((init.float_data[0] - 8.0).abs() < 1e-6);
    }
}
