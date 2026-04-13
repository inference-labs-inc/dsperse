use std::collections::HashMap;

use super::onnx_proto::{AttributeProto, ModelProto, NodeProto, TensorProto};

pub fn rewrite_self_div_to_one(
    model: &mut ModelProto,
    traced_shapes: &mut HashMap<String, Vec<i64>>,
) -> usize {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return 0,
    };

    let mut rewrites = 0usize;
    let mut new_inits: Vec<TensorProto> = Vec::new();

    for (idx, node) in graph.node.iter_mut().enumerate() {
        if node.op_type != "Div" || node.input.len() != 2 {
            continue;
        }
        if node.input[0].is_empty() || node.input[0] != node.input[1] {
            continue;
        }
        let Some(out_name) = node.output.first().cloned() else {
            continue;
        };
        if out_name.is_empty() {
            continue;
        }
        let Some(out_shape) = traced_shapes.get(&node.input[0]).cloned() else {
            continue;
        };
        if out_shape.iter().any(|&d| d < 0) {
            continue;
        }

        let shape_init_name = format!("/__dsperse/self_div_one_{idx}/shape");
        let shape_init = TensorProto {
            name: shape_init_name.clone(),
            data_type: TensorProto::INT64,
            dims: vec![out_shape.len() as i64],
            int64_data: out_shape.clone(),
            ..Default::default()
        };
        new_inits.push(shape_init);

        let one_value_init = TensorProto {
            name: format!("/__dsperse/self_div_one_{idx}/value"),
            data_type: TensorProto::FLOAT,
            dims: vec![1],
            float_data: vec![1.0],
            ..Default::default()
        };
        let value_attr = AttributeProto {
            name: "value".to_string(),
            r#type: 4,
            t: Some(one_value_init),
            ..Default::default()
        };

        node.op_type = "ConstantOfShape".to_string();
        node.input = vec![shape_init_name];
        node.attribute = vec![value_attr];

        traced_shapes.insert(out_name.clone(), out_shape);
        rewrites += 1;
    }

    if rewrites == 0 {
        return 0;
    }
    graph.initializer.extend(new_inits);
    tracing::info!(
        rewrites,
        "rewrote degenerate Div(X, X) to ConstantOfShape(shape(X), 1.0)"
    );
    rewrites
}
