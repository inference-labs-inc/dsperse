use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::onnx_proto::{self, GraphProto, ModelProto, TensorProto};
use crate::error::{DsperseError, Result};
use crate::schema::metadata::Dependencies;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnalysis {
    pub index: usize,
    pub slice_name: String,
    pub node_type: String,
    pub parameter_details: HashMap<String, ParameterDetail>,
    pub dependencies: NodeDependencies,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDetail {
    pub shape: Vec<i64>,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDependencies {
    pub input: Vec<String>,
    pub output: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub original_model: Option<String>,
    pub model_type: String,
    pub node_count: usize,
    pub initializer_count: usize,
    pub input_shape: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub output_names: Vec<String>,
    pub opset_version: Option<i64>,
    pub nodes: HashMap<String, NodeAnalysis>,
    pub initializer_names: HashSet<String>,
}

pub fn analyze(model: &ModelProto, onnx_path: Option<&Path>) -> Result<AnalysisResult> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Onnx("model has no graph".into()))?;
    let initializer_map: HashMap<&str, &TensorProto> = graph
        .initializer
        .iter()
        .map(|i| (i.name.as_str(), i))
        .collect();

    let input_shapes = get_model_input_shapes(graph, &initializer_map);
    let output_shapes = get_model_output_shapes(graph);
    let output_names = get_model_output_names(graph);

    let mut nodes = HashMap::new();
    for (i, node) in graph.node.iter().enumerate() {
        let node_key = if node.name.is_empty() {
            format!("{}_{}", node.op_type, i)
        } else {
            node.name.clone()
        };

        let parameter_details = get_parameter_details(node, &initializer_map);

        let mut inputs: Vec<String> = node
            .input
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        if super::is_control_flow(&node.op_type) {
            let outer_refs = super::collect_subgraph_outer_refs(node, graph);
            for r in outer_refs {
                if !inputs.contains(&r) {
                    inputs.push(r);
                }
            }
        }

        nodes.insert(
            node_key,
            NodeAnalysis {
                index: i,
                slice_name: format!("{}_{}", node.op_type, i),
                node_type: node.op_type.clone(),
                parameter_details,
                dependencies: NodeDependencies {
                    input: inputs,
                    output: node.output.clone(),
                },
            },
        );
    }

    let opset_version = model
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .map(|o| o.version);

    if let Some(v) = opset_version
        && v < 18
    {
        tracing::warn!(opset = v, "opset < 18 detected; continuing anyway");
    }

    let initializer_names: HashSet<String> =
        graph.initializer.iter().map(|i| i.name.clone()).collect();

    Ok(AnalysisResult {
        original_model: onnx_path.map(|p| p.to_string_lossy().to_string()),
        model_type: "ONNX".to_string(),
        node_count: graph.node.len(),
        initializer_count: graph.initializer.len(),
        input_shape: input_shapes,
        output_shapes,
        output_names,
        opset_version,
        nodes,
        initializer_names,
    })
}

fn get_model_input_shapes(
    graph: &GraphProto,
    initializer_map: &HashMap<&str, &TensorProto>,
) -> Vec<Vec<i64>> {
    graph
        .input
        .iter()
        .filter(|inp| !initializer_map.contains_key(inp.name.as_str()))
        .map(onnx_proto::vi_shape)
        .collect()
}

fn get_model_output_shapes(graph: &GraphProto) -> Vec<Vec<i64>> {
    graph.output.iter().map(onnx_proto::vi_shape).collect()
}

fn get_model_output_names(graph: &GraphProto) -> Vec<String> {
    graph.output.iter().map(|o| o.name.clone()).collect()
}

fn get_parameter_details(
    node: &onnx_proto::NodeProto,
    initializer_map: &HashMap<&str, &TensorProto>,
) -> HashMap<String, ParameterDetail> {
    let mut details = HashMap::new();
    if !matches!(node.op_type.as_str(), "Conv" | "Gemm" | "MatMul") {
        return details;
    }
    for inp_name in &node.input {
        if let Some(init) = initializer_map.get(inp_name.as_str()) {
            let size: usize = init.dims.iter().map(|&d| d as usize).product();
            if size > 0 {
                details.insert(
                    inp_name.clone(),
                    ParameterDetail {
                        shape: init.dims.clone(),
                        size,
                    },
                );
            }
        }
    }
    details
}

pub fn get_segment_dependencies(
    analysis: &AnalysisResult,
    start_idx: usize,
    end_idx: usize,
) -> Dependencies {
    let mut inputs = Vec::new();
    let mut output_map: HashMap<String, bool> = HashMap::new();

    let mut sorted_nodes: Vec<&NodeAnalysis> = analysis
        .nodes
        .values()
        .filter(|n| n.index >= start_idx && n.index < end_idx)
        .collect();
    sorted_nodes.sort_by_key(|n| n.index);

    for node in &sorted_nodes {
        for out in &node.dependencies.output {
            output_map.insert(out.clone(), true);
        }
        for inp in &node.dependencies.input {
            if !output_map.contains_key(inp) && !inputs.contains(inp) {
                inputs.push(inp.clone());
            }
        }
    }

    let mut outputs: Vec<String> = output_map
        .keys()
        .filter(|output| !inputs.contains(output))
        .cloned()
        .collect();
    outputs.sort();

    let filtered = inputs
        .iter()
        .filter(|name| !analysis.initializer_names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    let filtered_inputs = if filtered.is_empty() && !inputs.is_empty() {
        vec![inputs[0].clone()]
    } else {
        filtered
    };

    Dependencies {
        input: inputs,
        output: outputs,
        filtered_inputs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(
        op: &str,
        idx: usize,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
    ) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            op_type: op.into(),
            name: format!("{}_{}", op, idx),
            input: inputs.into_iter().map(String::from).collect(),
            output: outputs.into_iter().map(String::from).collect(),
            attribute: vec![],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        }
    }

    fn make_model_with_nodes(nodes: Vec<onnx_proto::NodeProto>) -> ModelProto {
        let input = onnx_proto::make_tensor_value_info("x", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let output = onnx_proto::make_tensor_value_info("y", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let graph = onnx_proto::make_graph("test", nodes, vec![input], vec![output], vec![]);
        onnx_proto::make_model(graph, 13)
    }

    fn make_model_with_initializers(
        nodes: Vec<onnx_proto::NodeProto>,
        initializers: Vec<TensorProto>,
    ) -> ModelProto {
        let input = onnx_proto::make_tensor_value_info("x", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let output = onnx_proto::make_tensor_value_info("y", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let graph = onnx_proto::make_graph("test", nodes, vec![input], vec![output], initializers);
        onnx_proto::make_model(graph, 13)
    }

    #[test]
    fn analyze_empty_model() {
        let model = make_model_with_nodes(vec![]);
        let result = analyze(&model, None).unwrap();
        assert_eq!(result.node_count, 0);
        assert!(result.nodes.is_empty());
        assert_eq!(result.model_type, "ONNX");
    }

    #[test]
    fn analyze_single_relu() {
        let model = make_model_with_nodes(vec![make_node("Relu", 0, vec!["x"], vec!["y"])]);
        let result = analyze(&model, None).unwrap();
        assert_eq!(result.node_count, 1);
        let node = result.nodes.values().next().unwrap();
        assert_eq!(node.node_type, "Relu");
        assert!(node.parameter_details.is_empty());
    }

    #[test]
    fn analyze_conv_with_initializer() {
        let weight_data: Vec<f32> = vec![1.0; 27];
        let weight_tensor = onnx_proto::make_tensor(
            "conv_weight",
            TensorProto::FLOAT,
            &[1, 3, 3, 3],
            weight_data,
        );
        let conv = make_node("Conv", 0, vec!["x", "conv_weight"], vec!["y"]);
        let model = make_model_with_initializers(vec![conv], vec![weight_tensor]);
        let result = analyze(&model, None).unwrap();
        assert_eq!(result.initializer_count, 1);
        let node = result.nodes.values().next().unwrap();
        assert!(!node.parameter_details.is_empty());
        let detail = node.parameter_details.get("conv_weight").unwrap();
        assert_eq!(detail.shape, vec![1, 3, 3, 3]);
        assert_eq!(detail.size, 27);
    }

    #[test]
    fn analyze_non_param_op_has_no_details() {
        let weight_data: Vec<f32> = vec![1.0; 27];
        let weight_tensor =
            onnx_proto::make_tensor("add_weight", TensorProto::FLOAT, &[1, 3, 3, 3], weight_data);
        let add = make_node("Add", 0, vec!["x", "add_weight"], vec!["y"]);
        let model = make_model_with_initializers(vec![add], vec![weight_tensor]);
        let result = analyze(&model, None).unwrap();
        let node = result.nodes.values().next().unwrap();
        assert!(node.parameter_details.is_empty());
    }

    #[test]
    fn analyze_model_no_graph() {
        let model = ModelProto {
            graph: None,
            ..Default::default()
        };
        assert!(analyze(&model, None).is_err());
    }

    #[test]
    fn analyze_dependencies_tracked() {
        let conv = make_node("Conv", 0, vec!["x", "w"], vec!["conv_out"]);
        let relu = make_node("Relu", 1, vec!["conv_out"], vec!["y"]);
        let model = make_model_with_nodes(vec![conv, relu]);
        let result = analyze(&model, None).unwrap();
        assert_eq!(result.node_count, 2);

        let relu_node = result
            .nodes
            .values()
            .find(|n| n.node_type == "Relu")
            .unwrap();
        assert_eq!(relu_node.dependencies.input, vec!["conv_out"]);
        assert_eq!(relu_node.dependencies.output, vec!["y"]);
    }

    #[test]
    fn analyze_unnamed_nodes_get_generated_keys() {
        let mut node = make_node("Relu", 0, vec!["x"], vec!["y"]);
        node.name = String::new();
        let model = make_model_with_nodes(vec![node]);
        let result = analyze(&model, None).unwrap();
        assert!(result.nodes.contains_key("Relu_0"));
    }

    #[test]
    fn get_segment_dependencies_basic() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "conv".into(),
            NodeAnalysis {
                index: 0,
                slice_name: "Conv_0".into(),
                node_type: "Conv".into(),
                parameter_details: HashMap::new(),
                dependencies: NodeDependencies {
                    input: vec!["x".into(), "w".into()],
                    output: vec!["conv_out".into()],
                },
            },
        );
        nodes.insert(
            "relu".into(),
            NodeAnalysis {
                index: 1,
                slice_name: "Relu_1".into(),
                node_type: "Relu".into(),
                parameter_details: HashMap::new(),
                dependencies: NodeDependencies {
                    input: vec!["conv_out".into()],
                    output: vec!["relu_out".into()],
                },
            },
        );
        let analysis = AnalysisResult {
            original_model: None,
            model_type: "ONNX".into(),
            node_count: 2,
            initializer_count: 1,
            input_shape: vec![],
            output_shapes: vec![],
            output_names: vec![],
            opset_version: Some(13),
            nodes,
            initializer_names: HashSet::from(["w".into()]),
        };
        let deps = get_segment_dependencies(&analysis, 0, 2);
        assert!(deps.output.contains(&"relu_out".to_string()));
        assert!(!deps.filtered_inputs.contains(&"w".to_string()));
    }

    fn make_attribute_graph(
        name: &str,
        graph: onnx_proto::GraphProto,
    ) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            r#type: onnx_proto::onnx::attribute_proto::AttributeType::Graph as i32,
            g: Some(graph),
            ..Default::default()
        }
    }

    #[test]
    fn analyze_loop_captures_outer_scope_refs() {
        let relu = make_node("Relu", 0, vec!["x"], vec!["relu_out"]);

        let body_node = onnx_proto::NodeProto {
            op_type: "Add".into(),
            name: "body_add".into(),
            input: vec!["body_in".into(), "relu_out".into()],
            output: vec!["body_out".into()],
            attribute: vec![],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };
        let body_input =
            onnx_proto::make_tensor_value_info("body_in", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let body_cond_in = onnx_proto::make_tensor_value_info("cond_in", TensorProto::BOOL, &[]);
        let body_cond_out = onnx_proto::make_tensor_value_info("cond_out", TensorProto::BOOL, &[]);
        let body_output =
            onnx_proto::make_tensor_value_info("body_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let body_graph = onnx_proto::make_graph(
            "loop_body",
            vec![body_node],
            vec![body_cond_in.clone(), body_input],
            vec![body_cond_out, body_output],
            vec![],
        );

        let loop_node = onnx_proto::NodeProto {
            op_type: "Loop".into(),
            name: "Loop_1".into(),
            input: vec!["trip_count".into(), "cond".into(), "init_val".into()],
            output: vec!["loop_out".into()],
            attribute: vec![make_attribute_graph("body", body_graph)],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };

        let input = onnx_proto::make_tensor_value_info("x", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let output =
            onnx_proto::make_tensor_value_info("loop_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let trip_vi = onnx_proto::make_tensor_value_info("trip_count", TensorProto::INT64, &[]);
        let cond_vi = onnx_proto::make_tensor_value_info("cond", TensorProto::BOOL, &[]);
        let init_vi =
            onnx_proto::make_tensor_value_info("init_val", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let graph = onnx_proto::make_graph(
            "test",
            vec![relu, loop_node],
            vec![input, trip_vi, cond_vi, init_vi],
            vec![output],
            vec![],
        );
        let model = onnx_proto::make_model(graph, 13);

        let result = analyze(&model, None).unwrap();
        let loop_analysis = result
            .nodes
            .values()
            .find(|n| n.node_type == "Loop")
            .unwrap();

        let loop_inputs = &loop_analysis.dependencies.input;
        assert!(
            loop_inputs.contains(&"relu_out".to_string()),
            "Loop node must include outer-scope ref 'relu_out' in its dependencies, got: {:?}",
            loop_inputs
        );
        for local in &["body_in", "body_out", "cond_in", "cond_out"] {
            assert!(
                !loop_inputs.contains(&local.to_string()),
                "body-local name '{}' must not leak into Loop dependencies, got: {:?}",
                local,
                loop_inputs
            );
        }
    }

    #[test]
    fn analyze_if_captures_outer_scope_refs() {
        let relu = make_node("Relu", 0, vec!["x"], vec!["relu_out"]);

        let then_node = onnx_proto::NodeProto {
            op_type: "Identity".into(),
            name: "then_id".into(),
            input: vec!["relu_out".into()],
            output: vec!["then_out".into()],
            attribute: vec![],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };
        let then_output =
            onnx_proto::make_tensor_value_info("then_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let then_graph = onnx_proto::make_graph(
            "then_branch",
            vec![then_node],
            vec![],
            vec![then_output],
            vec![],
        );

        let else_node = onnx_proto::NodeProto {
            op_type: "Neg".into(),
            name: "else_neg".into(),
            input: vec!["relu_out".into()],
            output: vec!["else_out".into()],
            attribute: vec![],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };
        let else_output =
            onnx_proto::make_tensor_value_info("else_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let else_graph = onnx_proto::make_graph(
            "else_branch",
            vec![else_node],
            vec![],
            vec![else_output],
            vec![],
        );

        let if_node = onnx_proto::NodeProto {
            op_type: "If".into(),
            name: "If_1".into(),
            input: vec!["cond".into()],
            output: vec!["if_out".into()],
            attribute: vec![
                make_attribute_graph("then_branch", then_graph),
                make_attribute_graph("else_branch", else_graph),
            ],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };

        let input = onnx_proto::make_tensor_value_info("x", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let cond_vi = onnx_proto::make_tensor_value_info("cond", TensorProto::BOOL, &[]);
        let output =
            onnx_proto::make_tensor_value_info("if_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let graph = onnx_proto::make_graph(
            "test",
            vec![relu, if_node],
            vec![input, cond_vi],
            vec![output],
            vec![],
        );
        let model = onnx_proto::make_model(graph, 13);

        let result = analyze(&model, None).unwrap();
        let if_analysis = result.nodes.values().find(|n| n.node_type == "If").unwrap();

        let if_inputs = &if_analysis.dependencies.input;
        assert!(
            if_inputs.contains(&"relu_out".to_string()),
            "If node must include outer-scope ref 'relu_out' from both branches, got: {:?}",
            if_inputs
        );
        for local in &["then_out", "else_out"] {
            assert!(
                !if_inputs.contains(&local.to_string()),
                "branch-local name '{}' must not leak into If dependencies, got: {:?}",
                local,
                if_inputs
            );
        }
    }

    #[test]
    fn segment_deps_include_subgraph_outer_refs() {
        let relu = make_node("Relu", 0, vec!["x"], vec!["relu_out"]);

        let body_node = onnx_proto::NodeProto {
            op_type: "Add".into(),
            name: "body_add".into(),
            input: vec!["body_in".into(), "relu_out".into()],
            output: vec!["body_out".into()],
            attribute: vec![],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };
        let body_input =
            onnx_proto::make_tensor_value_info("body_in", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let body_cond_in = onnx_proto::make_tensor_value_info("cond_in", TensorProto::BOOL, &[]);
        let body_cond_out = onnx_proto::make_tensor_value_info("cond_out", TensorProto::BOOL, &[]);
        let body_output =
            onnx_proto::make_tensor_value_info("body_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let body_graph = onnx_proto::make_graph(
            "loop_body",
            vec![body_node],
            vec![body_cond_in, body_input],
            vec![body_cond_out, body_output],
            vec![],
        );

        let loop_node = onnx_proto::NodeProto {
            op_type: "Loop".into(),
            name: "Loop_1".into(),
            input: vec!["trip_count".into(), "cond".into(), "init_val".into()],
            output: vec!["loop_out".into()],
            attribute: vec![make_attribute_graph("body", body_graph)],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };

        let input = onnx_proto::make_tensor_value_info("x", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let output =
            onnx_proto::make_tensor_value_info("loop_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let trip_vi = onnx_proto::make_tensor_value_info("trip_count", TensorProto::INT64, &[]);
        let cond_vi = onnx_proto::make_tensor_value_info("cond", TensorProto::BOOL, &[]);
        let init_vi =
            onnx_proto::make_tensor_value_info("init_val", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let graph = onnx_proto::make_graph(
            "test",
            vec![relu, loop_node],
            vec![input, trip_vi, cond_vi, init_vi],
            vec![output],
            vec![],
        );
        let model = onnx_proto::make_model(graph, 13);
        let result = analyze(&model, None).unwrap();

        let deps = get_segment_dependencies(&result, 1, 2);
        assert!(
            deps.input.contains(&"relu_out".to_string()),
            "segment containing only Loop must list 'relu_out' as input dep, got: {:?}",
            deps.input
        );
        for local in &["body_in", "body_out", "cond_in", "cond_out"] {
            assert!(
                !deps.input.contains(&local.to_string()),
                "body-local name '{}' must not appear in segment inputs, got: {:?}",
                local,
                deps.input
            );
        }
    }

    #[test]
    fn analyze_nested_subgraph_captures_outer_scope_refs() {
        let relu = make_node("Relu", 0, vec!["x"], vec!["relu_out"]);

        let inner_add = onnx_proto::NodeProto {
            op_type: "Add".into(),
            name: "inner_add".into(),
            input: vec!["inner_in".into(), "relu_out".into()],
            output: vec!["inner_out".into()],
            attribute: vec![],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };
        let inner_input =
            onnx_proto::make_tensor_value_info("inner_in", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let inner_output =
            onnx_proto::make_tensor_value_info("inner_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let inner_graph = onnx_proto::make_graph(
            "inner_then",
            vec![inner_add],
            vec![inner_input],
            vec![inner_output],
            vec![],
        );

        let if_node_in_body = onnx_proto::NodeProto {
            op_type: "If".into(),
            name: "nested_if".into(),
            input: vec!["body_cond".into()],
            output: vec!["body_out".into()],
            attribute: vec![
                make_attribute_graph("then_branch", inner_graph.clone()),
                make_attribute_graph("else_branch", inner_graph),
            ],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };
        let body_cond_in = onnx_proto::make_tensor_value_info("cond_in", TensorProto::BOOL, &[]);
        let body_cond = onnx_proto::make_tensor_value_info("body_cond", TensorProto::BOOL, &[]);
        let body_cond_out = onnx_proto::make_tensor_value_info("cond_out", TensorProto::BOOL, &[]);
        let body_output =
            onnx_proto::make_tensor_value_info("body_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let body_graph = onnx_proto::make_graph(
            "loop_body",
            vec![if_node_in_body],
            vec![body_cond_in, body_cond],
            vec![body_cond_out, body_output],
            vec![],
        );

        let loop_node = onnx_proto::NodeProto {
            op_type: "Loop".into(),
            name: "Loop_1".into(),
            input: vec!["trip_count".into(), "cond".into(), "init_val".into()],
            output: vec!["loop_out".into()],
            attribute: vec![make_attribute_graph("body", body_graph)],
            domain: String::new(),
            doc_string: String::new(),
            overload: String::new(),
            metadata_props: vec![],
            device_configurations: vec![],
        };

        let input = onnx_proto::make_tensor_value_info("x", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let output =
            onnx_proto::make_tensor_value_info("loop_out", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let trip_vi = onnx_proto::make_tensor_value_info("trip_count", TensorProto::INT64, &[]);
        let cond_vi = onnx_proto::make_tensor_value_info("cond", TensorProto::BOOL, &[]);
        let init_vi =
            onnx_proto::make_tensor_value_info("init_val", TensorProto::FLOAT, &[1, 3, 8, 8]);
        let graph = onnx_proto::make_graph(
            "test",
            vec![relu, loop_node],
            vec![input, trip_vi, cond_vi, init_vi],
            vec![output],
            vec![],
        );
        let model = onnx_proto::make_model(graph, 13);

        let result = analyze(&model, None).unwrap();
        let loop_analysis = result
            .nodes
            .values()
            .find(|n| n.node_type == "Loop")
            .unwrap();

        let nested_inputs = &loop_analysis.dependencies.input;
        assert!(
            nested_inputs.contains(&"relu_out".to_string()),
            "Loop with nested If subgraph referencing outer-scope 'relu_out' must capture it, got: {:?}",
            nested_inputs
        );
        for local in &["body_cond", "inner_in", "inner_out", "body_out"] {
            assert!(
                !nested_inputs.contains(&local.to_string()),
                "nested-body-local name '{}' must not leak into Loop dependencies, got: {:?}",
                local,
                nested_inputs
            );
        }
    }
}
