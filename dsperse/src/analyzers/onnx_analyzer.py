import logging
import os
from pathlib import Path
from typing import Dict, Any

import onnx

from dsperse.src.slice.utils.onnx_utils import OnnxUtils
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class OnnxAnalyzer:
    """
    A class for analyzing ONNX models and generating metadata.
    """

    def __init__(self, model_path:str):
        """
        Initialize the OnnxAnalyzer with either an ONNX model or a path to an ONNX model.
        """
        self.onnx_path = os.path.abspath(model_path)
        self.onnx_model = onnx.load(self.onnx_path)

        self.model_metadata = None

    def analyze(self, save_path:str = None) -> Dict[str, Any]:
        """
        Analyze the ONNX model and generate comprehensive metadata.

        Returns:
            Dict[str, Any]: Comprehensive metadata about the ONNX model
        """
        # Extract model metadata
        graph = self.onnx_model.graph

        # Create maps for initializers and value info
        initializer_map = {init.name: init for init in graph.initializer}

        # Build a comprehensive value_info map from the original full model
        full_model_value_info_map = {vi.name: vi for vi in graph.value_info}
        full_model_value_info_map.update({vi.name: vi for vi in graph.input})
        full_model_value_info_map.update({vi.name: vi for vi in graph.output})

        model_input_shape = self._get_model_input_shapes(graph, initializer_map)
        model_output_shape = self._get_model_output_shapes(graph)

        # Store node metadata
        node_metadata = {}

        # Process each node to collect metadata
        for i, node in enumerate(graph.node):
            # Analyze the node and store metadata
            node_info = self.analyze_node(node, i, initializer_map)
            # Use index-based key to handle empty node names
            node_key = node.name if node.name else f"{node.op_type}_{i}"
            node_metadata[node_key] = node_info

        # Determine opset information
        opset_imports_list = []
        default_opset_version = None
        try:
            for opset in self.onnx_model.opset_import:
                domain = opset.domain
                version = int(opset.version)
                opset_imports_list.append({"domain": domain if domain else "ai.onnx", "version": version})
                # Prefer default domain ""; fallback to explicit "ai.onnx"
                if default_opset_version is None and (domain == "" or domain == "ai.onnx"):
                    default_opset_version = version
        except Exception:
            default_opset_version = None

        # Warn (but continue) if default opset is below 18
        if default_opset_version is not None and default_opset_version < 18:
            msg = (
                f"ONNX opset {default_opset_version} detected for model {self.onnx_path}. "
                "Opset < 18 is not officially supported; continuing anyway."
            )
            logger.warning(msg)
            print(f"WARNING: {msg}")

        # Create model metadata
        model_metadata = {
            "original_model": self.onnx_path,
            "model_type": "ONNX",
            "node_count": len(graph.node),
            "initializer_count": len(graph.initializer),
            "input_shape": model_input_shape,
            "output_shapes": model_output_shape,
            "opset_version": default_opset_version,
            "opset_imports": opset_imports_list,
            "nodes": node_metadata
        }

        # Save model metadata
        if save_path is not None:
            # output_dir = os.path.join(os.path.dirname(self.onnx_path), "onnx_analysis")
            if Path(save_path).is_dir():
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, "model_metadata.json")

            Utils.save_metadata_file(model_metadata, save_path)
        self.model_metadata = model_metadata

        return model_metadata

    def analyze_node(self, node, index, initializer_map):
        """
        Analyze a single node from the ONNX graph and gather metadata.

        Args:
            node: ONNX node to analyze
            index: Index of the node in the graph
            initializer_map: Map of initializer names to initializers

        Returns:
            dict: Metadata for the node
        """
        node_inputs = list(node.input)
        node_outputs = list(node.output)

        # gather parameter information
        parameters, parameter_details = self._get_parameter_info(node, node_inputs, initializer_map)

        # Determine in_features and out_features
        in_features, out_features = self._get_feature_info(node, parameter_details)

        # Determine activation function
        node_type = node.op_type

        # Return node metadata
        return {
            "index": index,
            "slice_name": f"{node_type}_{index}",
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
            tuple: ( parameters, parameter_details)
        """
        # Calculate parameters if possible
        parameters = 0
        parameter_details = {}

        # For Conv, Gemm, and MatMul nodes, we can extract parameter information from initializers
        if node.op_type in ["Conv", "Gemm", "MatMul"]:
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

        elif node.op_type == "Gemm" or node.op_type == "MatMul":
            # For Gemm/MatMul, in_features is input dim, out_features is output dim
            if len(parameter_details) >= 1:
                # Typically weight shape for Gemm is [out_features, in_features]
                weight_name = next(iter(parameter_details))
                weight_shape = parameter_details[weight_name]["shape"]
                if len(weight_shape) >= 2:
                    out_features = weight_shape[0]
                    in_features = weight_shape[1]

        return in_features, out_features

    def _get_activation_info(self, node):
        """
        Determine activation function for a node.

        Args:
            node: ONNX node

        Returns:
            str: Activation function name
        """
        activation = node.op_type
        if node.op_type == "Relu":
            activation = "ReLU"
        elif node.op_type == "Sigmoid":
            activation = "Sigmoid"
        elif node.op_type == "Tanh":
            activation = "Tanh"
        elif node.op_type == "Softmax":
            activation = "Softmax"

        return activation

    def _create_layer_info(self, node_name, node_info):
        """
        Create layer information from node info.

        Args:
            node_name: Name of the node
            node_info: Dictionary of node information

        Returns:
            dict: Layer information
        """
        layer_info = {
            "name": node_name,
            "type": node_info["type"],
            "activation": node_info["activation"]
        }

        # Add shape information if available
        if node_info["input_shape"]:
            layer_info["input_shape"] = node_info["input_shape"]
        if node_info["output_shape"]:
            layer_info["output_shape"] = node_info["output_shape"]

        # Add parameter details if available
        if "parameter_details" in node_info and node_info["parameter_details"]:
            layer_info["parameters"] = {}
            for param_name, param_info in node_info["parameter_details"].items():
                layer_info["parameters"][param_name] = param_info

        # Add in/out features if available
        if node_info.get("in_features") is not None:
            layer_info["in_features"] = node_info["in_features"]
            layer_info["in_channels"] = node_info["in_features"]  # For compatibility with conv layers

        if node_info.get("out_features") is not None:
            layer_info["out_features"] = node_info["out_features"]
            layer_info["out_channels"] = node_info["out_features"]  # For compatibility with conv layers

        # For Conv layers, add kernel_size, stride, padding if available
        if node_info["type"] == "conv" and "parameter_details" in node_info:
            for param_name, param_info in node_info["parameter_details"].items():
                if len(param_info["shape"]) == 4:  # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                    layer_info["kernel_size"] = [param_info["shape"][2], param_info["shape"][3]]
                    # Default stride and padding (could be extracted from attributes if needed)
                    layer_info["stride"] = [1, 1]
                    layer_info["padding"] = [0, 0]
                    break

        return layer_info

    def generate_slices_metadata(self, model_metadata, slice_points, slices_paths, output_dir=None):
        """
        Generate metadata for sliced ONNX models.

        Args:
            model_metadata: The model analysis metadata containing node information
            slice_points: List of indices representing nodes with parameter details
            output_dir: Directory where the metadata will be saved
            slices_paths: Paths to sliced onnx files

        Returns:
            dict: Complete metadata for the sliced models
        """
        # Get model-level metadata
        model_overview = self._get_model_metadata(model_metadata, slice_points)

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

            slice_path = slices_paths[segment_idx] if slices_paths else None

            # Get segment metadata
            segment_metadata = self._get_segment_metadata(
                model_metadata, 
                segment_idx, 
                start_idx, 
                end_idx,
                slice_path,
                output_dir
            )
            # extract shape
            if segment_metadata:
                segments.append(segment_metadata)

        # Add segments to metadata
        model_overview["slices"] = segments

        # Save metadata if output_dir is provided
        Utils.save_metadata_file(model_overview, output_path=output_dir)
        OnnxUtils.write_slice_dirs_metadata(output_dir)

        return model_overview

    def _get_model_metadata(self, model_metadata, slice_points):
        """
        Get model-level metadata.

        Args:
            model_metadata: The model analysis metadata containing node information
            slice_points: List of indices representing nodes with parameter details

        Returns:
            dict: Model-level metadata
        """
        graph = self.onnx_model.graph

        # Create maps for initializers
        initializer_map = {init.name: init for init in graph.initializer}

        # Get model input and output shapes
        model_input_shapes = model_metadata["input_shape"]
        model_output_shapes = model_metadata["output_shapes"]

        # Calculate total parameters
        total_parameters = sum(node_info.get("parameters", 0) for node_info in model_metadata["nodes"].values())

        # Format the original_model path to be consistent with the expected format
        original_model_path = model_metadata["original_model"]
        model_type = model_metadata["model_type"]

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

    def _get_segment_metadata(self, model_metadata, segment_idx, start_idx, end_idx, slice_path, output_dir=None):
        """
        Get metadata for a specific segment.

        Args:
            model_metadata: The model analysis metadata containing node information
            segment_idx: Index of the segment
            start_idx: Start index of the segment
            end_idx: End index of the segment
            slice_path: Path to the sliced ONNX model

        Returns:
            dict: Segment metadata
        """
        # Collect nodes for this segment
        segment_nodes = []
        for idx in range(start_idx, end_idx):
            for node_name, node_info in model_metadata["nodes"].items():
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

        segment_dependencies = self._get_segment_dependencies(model_metadata, start_idx, end_idx)

        segment_shape = self._get_segment_shape(end_idx, model_metadata, start_idx, slice_path)

        output_dir = os.path.join(output_dir, f"slice_{segment_idx}") if output_dir else os.path.join(os.path.dirname(self.onnx_path), "slices", f"slice_{segment_idx}")
        # Ensure dslice-style payload directory exists
        payload_dir = os.path.join(output_dir, "payload")
        os.makedirs(payload_dir, exist_ok=True)
        segment_filename = f"slice_{segment_idx}.onnx"
        segment_path = os.path.abspath(os.path.join(payload_dir, segment_filename))

        # Create segment info
        segment_info = {
            "index": segment_idx,
            "filename": segment_filename,
            "path": segment_path,
            "parameters": segment_parameters,
            "shape": segment_shape,
            "dependencies": segment_dependencies,
            "layers": layers,
        }

        return segment_info

    def _get_segment_dependencies(self, model_metadata, start_idx, end_idx):
        # Create segment dependencies
        segment_dependencies = {
            "input": [],
            "output": [],
            "filtered_inputs": []
        }

        # Create an output_map dictionary to store all tensor names we have encountered
        output_map = {}

        # Go through each node in segment and populate output_map
        for idx in range(start_idx, end_idx):
            for node_name, node_info in model_metadata['nodes'].items():
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
                
        # Filter input names to exclude weights and biases
        filtered_inputs = []
        for input_name in segment_dependencies['input']:
            # Only include actual inputs that are not weights or biases
            # Typically, weights and biases have names containing "weight" or "bias"
            if not any(pattern in input_name.lower() for pattern in ["weight", "bias"]):
                # Include model inputs and intermediate tensors
                if input_name in [inp.name for inp in self.onnx_model.graph.input] or input_name.startswith('/'):
                    filtered_inputs.append(input_name)
        
        # If there are no inputs after filtering, include the first non-weight/bias input
        if not filtered_inputs:
            for input_name in segment_dependencies['input']:
                if not any(pattern in input_name.lower() for pattern in ["weight", "bias"]):
                    filtered_inputs.append(input_name)
                    break
            
            # If still no inputs, use the first input as a fallback
            if not filtered_inputs and segment_dependencies['input']:
                filtered_inputs.append(segment_dependencies['input'][0])
        
        segment_dependencies['filtered_inputs'] = filtered_inputs

        return segment_dependencies

    @staticmethod
    def _get_segment_shape(end_idx, model_metadata, start_idx, slice_path):
        segment_shape = {
            "weight_shape": OnnxAnalyzer._get_weight_shape(end_idx, model_metadata, start_idx),
            "tensor_shape": OnnxAnalyzer._get_tensor_shape(slice_path),
        }

        return segment_shape


    @staticmethod
    def _get_tensor_shape(slice_path):
        tensor_shape = {
            "input": [],
            "output": []
        }

        if slice_path:
            onnx_model = onnx.load(slice_path)
            graph = onnx_model.graph
            for init in graph.initializer:
                tensor_shape["input"].append(list(init.dims))
            for inp in graph.input:
                # Convert Dimension objects to simple values (string or number)
                dimensions = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField('dim_param'):
                        dimensions.append(dim.dim_param)  # Use the string parameter directly
                    else:
                        dimensions.append(dim.dim_value)  # Use the numeric value directly
                tensor_shape["input"].append(dimensions)
            for out in graph.output:
                # Convert Dimension objects to simple values (string or number)
                dimensions = []
                for dim in out.type.tensor_type.shape.dim:
                    if dim.HasField('dim_param'):
                        dimensions.append(dim.dim_param)  # Use the string parameter directly
                    else:
                        dimensions.append(dim.dim_value)  # Use the numeric value directly
                tensor_shape["output"].append(dimensions)

        return tensor_shape


    @staticmethod
    def _get_weight_shape(end_idx, model_metadata, start_idx):
        weight_shape = {
            "input": [],
            "output": []
        }
        # Get first and last nodes of segment
        first_node = None
        last_node = None
        next_node = None
        for node_name, node_info in model_metadata['nodes'].items():
            if node_info['index'] == start_idx:
                first_node = node_info
            if node_info['index'] == end_idx - 1:
                last_node = node_info
            if node_info['index'] == end_idx:
                next_node = node_info

        # Get segment shapes from first and last nodes if available
        if start_idx == 0:
            weight_shape["input"] = model_metadata["input_shape"][0]
        elif first_node and "parameter_details" in first_node:
            for param_name, param_info in first_node["parameter_details"].items():
                if "shape" in param_info:
                    weight_shape["input"] = param_info["shape"]
                    break

        # For the output shape:
        if last_node:
            # For the last segment, use model output shape
            if end_idx == len(model_metadata['nodes']):
                weight_shape["output"] = model_metadata["output_shapes"][0]
            # Otherwise, use the weight shape of the next node
            elif next_node:
                # If the next node has dependencies, use the shape of the first input
                if "dependencies" in next_node and "input" in next_node["dependencies"] and next_node["dependencies"][
                    "input"]:
                    # Try to find the shape from the next node's parameter details
                    if "parameter_details" in next_node:
                        # First, try to find a weight parameter with a 4D shape (for Conv layers)
                        for param_name, param_info in next_node["parameter_details"].items():
                            if "shape" in param_info and len(param_info["shape"]) == 4:
                                # This is likely a Conv weight tensor
                                weight_shape["output"] = param_info["shape"]
                                break

                        # If we didn't find a 4D shape, try to find a 2D shape (for Gemm/Linear layers)
                        if not weight_shape["output"]:
                            for param_name, param_info in next_node["parameter_details"].items():
                                if "shape" in param_info and len(param_info["shape"]) == 2:
                                    # This is likely a Gemm/Linear weight tensor
                                    weight_shape["output"] = param_info["shape"]
                                    break

                        # If we still didn't find a shape, try any parameter with a shape
                        if not weight_shape["output"]:
                            for param_name, param_info in next_node["parameter_details"].items():
                                if "shape" in param_info and len(param_info["shape"]) > 1:
                                    weight_shape["output"] = param_info["shape"]
                                    break

                # If we couldn't determine the output shape from the next node, use the last node's output features if available
                if not weight_shape["output"] and "out_features" in last_node:
                    weight_shape["output"] = ["batch_size", last_node["out_features"]]

        return weight_shape

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
        activation = self._get_activation_info(onnx.NodeProto(op_type=node_info["node_type"]))

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
        if layer_type == "conv" and "parameter_details" in node_info:
            for param_name, param_info in node_info["parameter_details"].items():
                if "shape" in param_info and len(
                        param_info["shape"]) == 4:  # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                    node_details["kernel_size"] = [param_info["shape"][2], param_info["shape"][3]]
                    # Default stride and padding (could be extracted from attributes if needed) TODO: Extract this
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