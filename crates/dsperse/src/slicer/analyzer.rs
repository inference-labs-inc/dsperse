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
