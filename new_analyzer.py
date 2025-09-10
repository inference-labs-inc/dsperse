#!/usr/bin/env python3
"""
Simplified ONNX Model Analyzer - No GraphSurgeon or Complex Nodes
"""

import os
import json
import onnx
from typing import Dict, Any, List

class SimpleOnnxAnalyzer:
    """
    A simplified class for analyzing ONNX models without GraphSurgeon dependencies.
    """

    def __init__(self, onnx_model=None, model_path=None):
        """
        Initialize the SimpleOnnxAnalyzer with either an ONNX model or a path to an ONNX model.

        Args:
            onnx_model: An ONNX model object
            onnx_path: Path to an ONNX model file
        """
        if onnx_model is not None:
            self.onnx_model = onnx_model
        elif model_path is not None:
            if os.path.isabs(model_path):
                self.onnx_path = model_path
            else:
                self.onnx_path = model_path
            self.onnx_model = onnx.load(self.onnx_path)
        else:
            raise ValueError("onnx_model path is not found")

        self.model_metadata = None

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the ONNX model and generate basic metadata.

        Returns:
            Dict[str, Any]: Basic metadata about the ONNX model
        """
        # Create output directory for analysis results
        output_dir = os.path.join(os.path.dirname(self.onnx_path), "simple_analysis")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Extract basic model metadata
        graph = self.onnx_model.graph

        # Create maps for initializers and value info
        initializer_map = {init.name: init for init in graph.initializer}

        # Build basic value_info map
        value_info_map = {vi.name: vi for vi in graph.value_info}
        value_info_map.update({vi.name: vi for vi in graph.input})
        value_info_map.update({vi.name: vi for vi in graph.output})

        model_input_shape = self._get_model_input_shapes(graph, initializer_map)
        model_output_shape = self._get_model_output_shapes(graph)

        # Store basic node metadata
        node_metadata = {}

        # Process each node for basic metadata
        for i, node in enumerate(graph.node):
            node_info = self.analyze_node(node, i, initializer_map)
            node_key = node.name if node.name else f"{node.op_type}_{i}"
            node_metadata[node_key] = node_info

        # Create simplified model metadata
        model_metadata = {
            "original_model": self.onnx_path,
            "model_type": "ONNX",
            "node_count": len(graph.node),
            "initializer_count": len(graph.initializer),
            "input_shape": model_input_shape,
            "output_shapes": model_output_shape,
            "nodes": node_metadata,
            "total_parameters": sum(node_info.get("parameters", 0) for node_info in node_metadata.values())
        }

        # Save model metadata
        self._save_metadata_file(model_metadata, output_dir, "simple_model_metadata.json")
        self.model_metadata = model_metadata

        print(f"✅ Analysis complete: {len(node_metadata)} nodes analyzed")
        return model_metadata

    def analyze_node(self, node, index, initializer_map):
        """
        Analyze a single node from the ONNX graph and gather basic metadata.

        Args:
            node: ONNX node to analyze
            index: Index of the node in the graph
            initializer_map: Map of initializer names to initializers

        Returns:
            dict: Basic metadata for the node
        """
        node_inputs = list(node.input)
        node_outputs = list(node.output)

        # Gather parameter information
        parameters, parameter_details = self._get_parameter_info(node, node_inputs, initializer_map)

        # Determine in_features and out_features
        in_features, out_features = self._get_feature_info(node, parameter_details)

        # Determine node type
        node_type = node.op_type

        # Return basic node metadata
        return {
            "index": index,
            "segment_name": f"{node_type}_{index}",
            "parameters": parameters,
            "node_type": node_type,
            "in_features": in_features,
            "out_features": out_features,
            "parameter_details": parameter_details,
            "dependencies": {
                "input": node_inputs,
                "output": node_outputs
            }
        }

    def _get_model_input_shapes(self, graph, initializer_map):
        """
        Extract input shapes from the model graph.

        Args:
            graph: ONNX model graph
            initializer_map: Map of initializer names to initializers

        Returns:
            list: List of input shapes
        """
        model_input_shapes = []
        for input_info in graph.input:
            if input_info.name not in initializer_map:  # Skip initializers (weights)
                shape = []
                if input_info.type.tensor_type.shape.dim:
                    for dim in input_info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value if dim.dim_value != 0 else None)
                model_input_shapes.append(shape)
        return model_input_shapes

    def _get_model_output_shapes(self, graph):
        """
        Extract output shapes from the model graph.

        Args:
            graph: ONNX model graph

        Returns:
            list: List of output shapes
        """
        model_output_shapes = []
        for output_info in graph.output:
            shape = []
            if output_info.type.tensor_type.shape.dim:
                for dim in output_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value if dim.dim_value != 0 else None)
            model_output_shapes.append(shape)
        return model_output_shapes

    def _get_parameter_info(self, node, node_inputs, initializer_map):
        """
        Determine parameter information for a node.

        Args:
            node: ONNX node
            node_inputs: List of node inputs
            initializer_map: Map of initializer names to initializers

        Returns:
            tuple: (parameters, parameter_details)
        """
        # Calculate parameters if possible
        parameters = 0
        parameter_details = {}

        # For Conv and Gemm nodes, we can extract parameter information from initializers
        if node.op_type in ["Conv", "Gemm"]:
            for inp in node_inputs:
                if inp in initializer_map:
                    init = initializer_map[inp]
                    # Calculate size (number of elements)
                    size = 1
                    for dim in init.dims:
                        size *= dim

                    # Add to total parameters
                    parameters += size

                    # Store parameter details
                    parameter_details[inp] = {
                        "shape": list(init.dims),
                        "size": size
                    }

        return parameters, parameter_details

    def _get_feature_info(self, node, parameter_details):
        """
        Determine in_features and out_features for a node.

        Args:
            node: ONNX node
            parameter_details: Dictionary of parameter details

        Returns:
            tuple: (in_features, out_features)
        """
        in_features = None
        out_features = None

        if node.op_type == "Conv":
            # For Conv, in_features is input channels, out_features is output channels
            if len(parameter_details) >= 1:
                # Typically weight shape for Conv is [out_channels, in_channels, kernel_h, kernel_w]
                weight_name = next(iter(parameter_details))
                weight_shape = parameter_details[weight_name]["shape"]
                if len(weight_shape) >= 2:
                    out_features = weight_shape[0]
                    in_features = weight_shape[1]

        elif node.op_type in ["Gemm", "MatMul"]:
            # For Gemm/MatMul, in_features is input dim, out_features is output dim
            if len(parameter_details) >= 1:
                # Typically weight shape for Gemm is [out_features, in_features]
                weight_name = next(iter(parameter_details))
                weight_shape = parameter_details[weight_name]["shape"]
                if len(weight_shape) >= 2:
                    out_features = weight_shape[0]
                    in_features = weight_shape[1]

        return in_features, out_features

    @staticmethod
    def _save_metadata_file(metadata, output_dir, filename):
        """Save metadata to a JSON file."""
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {filepath}")

    def get_nodes_with_parameters(self):
        """
        Get list of node indices that have parameters (Conv, Gemm, etc.)

        Returns:
            List[int]: List of node indices with parameters
        """
        if not self.model_metadata:
            raise ValueError("Please run analyze() first")

        slice_points = []
        for node_name, node_info in self.model_metadata["nodes"].items():
            if node_info.get("parameter_details") and node_info["parameter_details"]:
                slice_points.append(node_info["index"])

        return sorted(slice_points)

    def determine_slice_points(self):
        """
        Determine the slice points for the model based on nodes with parameter_details
        and complex nodes in the model_metadata.

        Returns:
            List[int]: List of indices representing nodes with parameter details and complex nodes
        """
        if not self.model_metadata or "nodes" not in self.model_metadata:
            raise ValueError("Invalid model metadata. Please run analyze() first.")

        # Find nodes with parameter_details in model_metadata
        slice_points = []
        for node_name, node_info in self.model_metadata["nodes"].items():
            if node_info.get("parameter_details") and node_info["parameter_details"]:
                slice_points.append(node_info["index"])

        # Add complex nodes as slice points using the complex node info from analyzer
        if "complex_nodes" in self.model_metadata:
            complex_nodes = self.model_metadata["complex_nodes"]

            # Add complex node indices to slice points
            for node_type, nodes in complex_nodes.items():
                for i, node_info in enumerate(nodes):
                    node_name = node_info['node_name']
                    # For generated names like "Conv_0", extract the index
                    if '_' in node_name and node_name.split('_')[-1].isdigit():
                        try:
                            generated_idx = int(node_name.split('_')[-1])
                            # Find nodes with the same operation type at the corresponding position
                            op_type = '_'.join(node_name.split('_')[:-1])
                            matching_nodes = [(name, info) for name, info in self.model_metadata["nodes"].items()
                                            if info.get("node_type") == op_type]

                            if generated_idx < len(matching_nodes):
                                _, meta_node_info = matching_nodes[generated_idx]
                                slice_points.append(meta_node_info["index"])
                                print(f"Added complex node {node_type} '{node_name}' at index {meta_node_info['index']} as slice point")
                        except (ValueError, IndexError):
                            # Fallback: try to find any node with this operation type
                            op_type = node_name.split('_')[0] if '_' in node_name else node_name
                            for meta_node_name, meta_node_info in self.model_metadata["nodes"].items():
                                if meta_node_info.get("node_type") == op_type:
                                    slice_points.append(meta_node_info["index"])
                                    print(f"Added complex node {node_type} '{node_name}' at index {meta_node_info['index']} as slice point (fallback)")
                                    break

        # Sort slice points by index and remove duplicates
        slice_points = sorted(list(set(slice_points)))

        return slice_points

    def get_model_summary(self):
        """
        Get a summary of the analyzed model.

        Returns:
            Dict[str, Any]: Model summary
        """
        if not self.model_metadata:
            return None

        nodes_with_params = self.get_nodes_with_parameters()
        slice_points = self.determine_slice_points()

        summary = {
            "model_path": self.model_metadata["original_model"],
            "total_nodes": self.model_metadata["node_count"],
            "total_parameters": self.model_metadata["total_parameters"],
            "nodes_with_parameters": len(nodes_with_params),
            "slice_points": len(slice_points),
            "parameter_nodes_indices": nodes_with_params,
            "all_slice_points": slice_points,
            "input_shapes": self.model_metadata["input_shape"],
            "output_shapes": self.model_metadata["output_shapes"],
            "node_types": list(set(node["node_type"] for node in self.model_metadata["nodes"].values()))
        }

        return summary

    def generate_slices_metadata(self, slice_points, output_dir=None):
        """
        Generate metadata for sliced ONNX models.

        Args:
            slice_points: List of indices representing nodes with parameter details
            output_dir: Directory where the metadata will be saved

        Returns:
            dict: Complete metadata for the sliced models
        """
        # Get model-level metadata
        model_overview = self._get_model_metadata(slice_points)

        # Process each segment
        segments = []

        for i in range(len(slice_points)):
            segment_idx = i - 1
            if segment_idx < 0:
                continue

            start_idx = slice_points[i - 1] if i > 0 else 0
            end_idx = slice_points[i]

            # Skip if start and end are the same
            if start_idx == end_idx:
                continue

            # Get segment metadata
            segment_metadata = self._get_segment_metadata(
                segment_idx,
                start_idx,
                end_idx
            )

            if segment_metadata:
                segments.append(segment_metadata)

        # Add segments to metadata
        model_overview["segments"] = segments

        # Save metadata if output_dir is provided
        if output_dir:
            self._save_metadata_file(model_overview, output_dir, "slices_metadata.json")

        return model_overview

    def _get_model_metadata(self, slice_points):
        """
        Get model-level metadata.

        Args:
            slice_points: List of indices representing nodes with parameter details

        Returns:
            dict: Model-level metadata
        """
        # Get model input and output shapes
        model_input_shapes = self.model_metadata["input_shape"]
        model_output_shapes = self.model_metadata["output_shapes"]

        # Calculate total parameters
        total_parameters = self.model_metadata["total_parameters"]

        # Format the original_model path to be consistent with the expected format
        original_model_path = self.model_metadata["original_model"]
        model_type = self.model_metadata["model_type"]

        # Create model metadata
        metadata = {
            "original_model": original_model_path,
            "model_type": model_type,
            "total_parameters": total_parameters,
            "input_shape": model_input_shapes,
            "output_shapes": model_output_shapes,
            "slice_points": slice_points[:-1]
        }

        return metadata

    def _get_segment_metadata(self, segment_idx, start_idx, end_idx):
        """
        Get metadata for a specific segment.

        Args:
            segment_idx: Index of the segment
            start_idx: Start index of the segment
            end_idx: End index of the segment

        Returns:
            dict: Segment metadata
        """
        # Collect nodes for this segment
        segment_nodes = []
        for idx in range(start_idx, end_idx):
            for node_name, node_info in self.model_metadata["nodes"].items():
                if node_info["index"] == idx:
                    segment_nodes.append((node_name, node_info))

        # Skip if no nodes in this segment
        if not segment_nodes:
            return None

        # Calculate segment parameters
        segment_parameters = sum(node_info.get("parameters", 0) for _, node_info in segment_nodes)

        # Create layers information
        layers = []
        for node_name, node_info in segment_nodes:
            layer_metadata = self._get_layer_metadata(node_name, node_info)
            layers.append(layer_metadata)

        segment_dependencies = self._get_segment_dependencies(start_idx, end_idx)

        segment_shape = self._get_segment_shape(end_idx, start_idx)

        # Create segment info
        segment_info = {
            "index": segment_idx,
            "filename": f"segment_{segment_idx}.onnx",
            "path": os.path.join(os.path.dirname(self.onnx_path), "simple_slices", f"segment_{segment_idx}.onnx"),
            "parameters": segment_parameters,
            "shape": segment_shape,
            "dependencies": segment_dependencies,
            "layers": layers,
        }

        return segment_info

    def _get_segment_dependencies(self, start_idx, end_idx):
        """
        Create segment dependencies.

        Args:
            start_idx: Start index of the segment
            end_idx: End index of the segment

        Returns:
            dict: Segment dependencies
        """
        # Create segment dependencies
        segment_dependencies = {
            "input": [],
            "output": []
        }

        # Create an output_map dictionary to store all tensor names we have encountered
        output_map = {}

        # Go through each node in segment and populate output_map
        for idx in range(start_idx, end_idx):
            for node_name, node_info in self.model_metadata['nodes'].items():
                if node_info['index'] == idx:
                    # Add outputs to map
                    for output in node_info['dependencies']['output']:
                        output_map[output] = True

                    # Check inputs and add any missing to dependencies
                    for input_name in node_info['dependencies']['input']:
                        if input_name not in output_map:
                            if input_name not in segment_dependencies['input']:
                                segment_dependencies['input'].append(input_name)

        # Whatever outputs we have in the map that aren't already in input dependencies
        # need to be added to segment output dependencies
        for output in output_map:
            if output not in segment_dependencies['input']:
                segment_dependencies['output'].append(output)

        return segment_dependencies

    def _get_segment_shape(self, end_idx, start_idx):
        """
        Get segment shape information.

        Args:
            end_idx: End index of the segment
            start_idx: Start index of the segment

        Returns:
            dict: Segment shape information
        """
        segment_shape = {
            "input": [],
            "output": []
        }

        # Get first and last nodes of segment
        first_node = None
        last_node = None
        next_node = None

        for node_name, node_info in self.model_metadata['nodes'].items():
            if node_info['index'] == start_idx:
                first_node = node_info
            if node_info['index'] == end_idx - 1:
                last_node = node_info
            if node_info['index'] == end_idx:
                next_node = node_info

        # Get segment shapes from first and last nodes if available
        if start_idx == 0:
            segment_shape["input"] = self.model_metadata["input_shape"][0]
        elif first_node and "parameter_details" in first_node:
            for param_name, param_info in first_node["parameter_details"].items():
                if "shape" in param_info:
                    segment_shape["input"] = param_info["shape"]
                    break

        # For the output shape:
        if last_node:
            # For the last segment, use model output shape
            if end_idx == len(self.model_metadata['nodes']):
                segment_shape["output"] = self.model_metadata["output_shapes"][0]
            # Otherwise, use the weight shape of the next node
            elif next_node:
                # If the next node has dependencies, use the shape of the first input
                if "dependencies" in next_node and "input" in next_node["dependencies"] and next_node["dependencies"]["input"]:
                    # Try to find the shape from the next node's parameter details
                    if "parameter_details" in next_node:
                        # First, try to find a weight parameter with a 4D shape (for Conv layers)
                        for param_name, param_info in next_node["parameter_details"].items():
                            if "shape" in param_info and len(param_info["shape"]) == 4:
                                # This is likely a Conv weight tensor
                                segment_shape["output"] = param_info["shape"]
                                break

                        # If we didn't find a 4D shape, try to find a 2D shape (for Gemm/Linear layers)
                        if not segment_shape["output"]:
                            for param_name, param_info in next_node["parameter_details"].items():
                                if "shape" in param_info and len(param_info["shape"]) == 2:
                                    # This is likely a Gemm/Linear weight tensor
                                    segment_shape["output"] = param_info["shape"]
                                    break

                        # If we still didn't find a shape, try any parameter with a shape
                        if not segment_shape["output"]:
                            for param_name, param_info in next_node["parameter_details"].items():
                                if "shape" in param_info and len(param_info["shape"]) > 1:
                                    segment_shape["output"] = param_info["shape"]
                                    break

            # If we couldn't determine the output shape from the next node, use the last node's output features if available
            if not segment_shape["output"] and "out_features" in last_node:
                segment_shape["output"] = ["batch_size", last_node["out_features"]]

        return segment_shape

    def _get_layer_metadata(self, node_name, node_info):
        """
        Get metadata for a specific layer.

        Args:
            node_name: Name of the node
            node_info: Dictionary of node information

        Returns:
            dict: Layer metadata
        """
        # Determine layer type
        layer_type = node_info["node_type"]

        # Determine activation function
        activation = self._get_activation_info(node_info["node_type"])

        # Add parameter details if available
        node_details = {}
        if "parameter_details" in node_info and node_info["parameter_details"]:
            for param_name, param_info in node_info["parameter_details"].items():
                node_details[param_name] = param_info

        # Add in/out features if available
        if "in_features" in node_info and node_info["in_features"] is not None:
            node_details["in_features"] = node_info["in_features"]
            node_details["in_channels"] = node_info["in_features"]

        if "out_features" in node_info and node_info["out_features"] is not None:
            node_details["out_features"] = node_info["out_features"]
            node_details["out_channels"] = node_info["out_features"]

        # For Conv layers, add kernel_size, stride, padding if available
        if layer_type == "Conv" and "parameter_details" in node_info:
            for param_name, param_info in node_info["parameter_details"].items():
                if "shape" in param_info and len(param_info["shape"]) == 4:  # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                    node_details["kernel_size"] = [param_info["shape"][2], param_info["shape"][3]]
                    # Default stride and padding (could be extracted from attributes if needed)
                    node_details["stride"] = [1, 1]
                    node_details["padding"] = [0, 0]
                    break

        # Create layer info
        layer_info = {
            "name": node_name,
            "type": layer_type,
            "activation": activation,
            "parameter_details": node_details,
        }

        return layer_info

    def _get_activation_info(self, node_type):
        """
        Determine activation function for a node.

        Args:
            node_type: Type of the node

        Returns:
            str: Activation function name
        """
        activation = node_type
        if node_type == "Relu":
            activation = "ReLU"
        elif node_type == "Sigmoid":
            activation = "Sigmoid"
        elif node_type == "Tanh":
            activation = "Tanh"
        elif node_type == "Softmax":
            activation = "Softmax"

        return activation
