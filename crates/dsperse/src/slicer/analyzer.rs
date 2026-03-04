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

        nodes.insert(
            node_key,
            NodeAnalysis {
                index: i,
                slice_name: format!("{}_{}", node.op_type, i),
                node_type: node.op_type.clone(),
                parameter_details,
                dependencies: NodeDependencies {
                    input: node.input.clone(),
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

    if let Some(v) = opset_version {
        if v < 18 {
            tracing::warn!(opset = v, "opset < 18 detected; continuing anyway");
        }
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
    graph
        .output
        .iter()
        .map(onnx_proto::vi_shape)
        .collect()
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

    fn make_node(op: &str, idx: usize, inputs: Vec<&str>, outputs: Vec<&str>) -> onnx_proto::NodeProto {
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
        let weight_tensor = onnx_proto::make_tensor(
            "add_weight",
            TensorProto::FLOAT,
            &[1, 3, 3, 3],
            weight_data,
        );
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

        let relu_node = result.nodes.values().find(|n| n.node_type == "Relu").unwrap();
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
}
