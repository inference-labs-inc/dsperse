use std::collections::{HashMap, HashSet};

use super::onnx_proto::{ModelProto, NodeProto, TensorProto};

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
            _ => None,
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
                && let Some(tensor) = val.clone().into_tensor(out)
            {
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
        "Gather" => eval_gather(node, inputs),
        "Slice" => eval_slice(inputs),
        "Cast" => eval_cast(node, inputs),
        "Sqrt" => map_f32(inputs, |v| v.sqrt()),
        "Neg" => map_f32(inputs, |v| -v),
        "Exp" => map_f32(inputs, |v| v.exp()),
        "Add" => binary_f32(inputs, |a, b| a + b),
        "Sub" => binary_f32(inputs, |a, b| a - b),
        "Mul" => binary_f32(inputs, |a, b| a * b),
        "Div" => binary_f32(inputs, |a, b| a / b),
        "Unsqueeze" => eval_unsqueeze(node, inputs),
        "Concat" => eval_concat(node, inputs),
        "Reshape" => eval_reshape(node, inputs),
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
        ConstVal::I64(data, _, _) => {
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
            Some(ConstVal::I64(result?, dims, TensorProto::INT64))
        }
        ConstVal::F32(data, _, _) => {
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
            Some(ConstVal::F32(result?, dims, TensorProto::FLOAT))
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
        ConstVal::I64(vals, _, _) => {
            let len = vals.len() as i64;
            let s = normalize_idx(*starts.first()?, len);
            let e = normalize_idx(*ends.first()?, len);
            let result: Vec<i64> = if s <= e { vals[s..e].to_vec() } else { vec![] };
            Some(ConstVal::I64(
                result.clone(),
                vec![result.len() as i64],
                TensorProto::INT64,
            ))
        }
        ConstVal::F32(vals, _, _) => {
            let len = vals.len() as i64;
            let s = normalize_idx(*starts.first()?, len);
            let e = normalize_idx(*ends.first()?, len);
            let result: Vec<f32> = if s <= e { vals[s..e].to_vec() } else { vec![] };
            Some(ConstVal::F32(
                result.clone(),
                vec![result.len() as i64],
                TensorProto::FLOAT,
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
            ConstVal::I64(v, _, dt) => Some(ConstVal::I64(v.clone(), dims, *dt)),
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

fn eval_concat(node: &NodeProto, inputs: &[Option<&ConstVal>]) -> Option<ConstVal> {
    let axis = attr_int(node, "axis").unwrap_or(0);
    let first = inp(inputs, 0)?;
    if first.dims().len() != 1 || axis != 0 {
        return None;
    }
    match first {
        ConstVal::I64(..) => {
            let mut result: Vec<i64> = Vec::new();
            for i in inputs {
                result.extend(i.as_ref()?.as_i64()?);
            }
            Some(ConstVal::I64(
                result.clone(),
                vec![result.len() as i64],
                TensorProto::INT64,
            ))
        }
        ConstVal::F32(..) => {
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
        ConstVal::ShapeOnly(_) => None,
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
    let value_tensor = node
        .attribute
        .iter()
        .find(|a| a.name == "value")?
        .t
        .as_ref()?;

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
}
