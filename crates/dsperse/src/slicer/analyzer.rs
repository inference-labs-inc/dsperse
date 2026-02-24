use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::onnx_proto::{self, GraphProto, ModelProto, TensorProto};
use crate::error::{DsperseError, Result};
use crate::schema::metadata::{
    Compilation, Dependencies, ModelMetadata, SliceMetadata, SliceShapeWrapper, TensorShape,
};
use crate::schema::tiling::{ChannelSplitInfo, TilingInfo};

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
        .map(|inp| onnx_proto::vi_shape(inp))
        .collect()
}

fn get_model_output_shapes(graph: &GraphProto) -> Vec<Vec<i64>> {
    graph
        .output
        .iter()
        .map(|out| onnx_proto::vi_shape(out))
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

pub fn generate_slices_metadata(
    analysis: &AnalysisResult,
    slice_points: &[usize],
    slices_paths: &HashMap<usize, String>,
    output_dir: &Path,
    tiled_info: &HashMap<usize, TiledResult>,
) -> Result<ModelMetadata> {
    let mut slices = Vec::new();

    for i in 1..slice_points.len() {
        let segment_idx = i - 1;
        let start_idx = slice_points[i - 1];
        let end_idx = slice_points[i];
        if start_idx == end_idx {
            continue;
        }

        let slice_path = slices_paths.get(&segment_idx).map(|s| s.as_str());
        let shape = get_segment_shape(slice_path);
        let dependencies = get_segment_dependencies(analysis, start_idx, end_idx);

        let filename = format!("slice_{segment_idx}.onnx");
        let slice_dir = output_dir.join(format!("slice_{segment_idx}"));
        let payload_dir = slice_dir.join("payload");
        std::fs::create_dir_all(&payload_dir).map_err(|e| DsperseError::io(e, &payload_dir))?;
        let onnx_path = payload_dir.join(&filename);

        let mut tiling: Option<TilingInfo> = None;
        let mut channel_split: Option<ChannelSplitInfo> = None;
        if let Some(result) = tiled_info.get(&segment_idx) {
            tiling = result.tiling.clone();
            channel_split = result.channel_split.clone();
        }

        let relative_path = format!("slice_{segment_idx}/payload/{filename}");

        slices.push(SliceMetadata {
            index: segment_idx,
            filename: filename.clone(),
            path: onnx_path.to_string_lossy().to_string(),
            relative_path,
            shape: SliceShapeWrapper {
                tensor_shape: shape,
            },
            dependencies,
            tiling,
            channel_split,
            compilation: Compilation::default(),
            slice_metadata: Some(format!("slice_{segment_idx}/metadata.json")),
            slice_metadata_relative_path: Some(format!("slice_{segment_idx}/metadata.json")),
        });
    }

    let metadata = ModelMetadata {
        original_model: analysis.original_model.clone().unwrap_or_default(),
        model_type: analysis.model_type.clone(),
        input_shape: analysis.input_shape.clone(),
        output_shapes: analysis.output_shapes.clone(),
        slice_points: slice_points[..slice_points.len().saturating_sub(1)].to_vec(),
        slices,
    };

    metadata.save(&output_dir.join("metadata.json"))?;
    write_per_slice_metadata(&metadata, output_dir)?;

    Ok(metadata)
}

fn write_per_slice_metadata(metadata: &ModelMetadata, output_dir: &Path) -> Result<()> {
    for slice_meta in &metadata.slices {
        let slice_dir = output_dir.join(format!("slice_{}", slice_meta.index));
        let start = metadata
            .slice_points
            .get(slice_meta.index)
            .copied()
            .unwrap_or(slice_meta.index);
        let end = metadata
            .slice_points
            .get(slice_meta.index + 1)
            .copied()
            .unwrap_or(start);
        let per_slice = ModelMetadata {
            original_model: metadata.original_model.clone(),
            model_type: metadata.model_type.clone(),
            input_shape: metadata.input_shape.clone(),
            output_shapes: metadata.output_shapes.clone(),
            slice_points: vec![start, end],
            slices: vec![slice_meta.clone()],
        };
        per_slice.save(&slice_dir.join("metadata.json"))?;
    }
    Ok(())
}

fn get_segment_shape(slice_path: Option<&str>) -> TensorShape {
    let Some(path_str) = slice_path else {
        return TensorShape::default();
    };
    let path = Path::new(path_str);
    let Ok(model) = onnx_proto::load_model(path) else {
        return TensorShape::default();
    };
    let graph = match model.graph.as_ref() {
        Some(g) => g,
        None => return TensorShape::default(),
    };

    let inputs: Vec<Vec<i64>> = graph
        .input
        .iter()
        .map(|inp| onnx_proto::vi_shape(inp))
        .collect();

    let outputs: Vec<Vec<i64>> = graph
        .output
        .iter()
        .map(|out| onnx_proto::vi_shape(out))
        .collect();

    TensorShape {
        input: inputs,
        output: outputs,
    }
}

fn get_segment_dependencies(
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

#[derive(Debug, Clone, Default)]
pub struct TiledResult {
    pub tiling: Option<TilingInfo>,
    pub channel_split: Option<ChannelSplitInfo>,
}
