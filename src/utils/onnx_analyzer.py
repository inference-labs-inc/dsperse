import os
import json
import onnx
from onnx import shape_inference

class OnnxAnalyzer:
    """
    A class for analyzing ONNX models and generating metadata.
    """

    def __init__(self, onnx_model=None, onnx_path=None):
        """
        Initialize the OnnxAnalyzer with either an ONNX model or a path to an ONNX model.

        Args:
            onnx_model: An ONNX model object
            onnx_path: Path to an ONNX model file
        """
        if onnx_model is not None:
            self.onnx_model = onnx_model
            self.onnx_path = None
        elif onnx_path is not None:
            self.onnx_path = onnx_path
            self.onnx_model = onnx.load(onnx_path)
        else:
            raise ValueError("Either onnx_model or onnx_path must be provided")

    def generate_metadata(self, segments_info, output_dir=None):
        """
        Generate metadata for the ONNX model based on segment information.

        Args:
            segments_info: List of dictionaries containing segment information
            output_dir: Directory where the metadata will be saved

        Returns:
            dict: Complete metadata for the model
        """
        graph = self.onnx_model.graph

        # Create maps for initializers and value info
        initializer_map = {init.name: init for init in graph.initializer}

        # Get model input and output shapes
        model_input_shapes = self._get_model_input_shapes(graph, initializer_map)
        model_output_shapes = self._get_model_output_shapes(graph)

        # Calculate total parameters
        total_parameters = sum(segment.get("parameters", 0) for segment in segments_info)

        # Generate slice points
        slice_points = []
        if len(segments_info) > 1:
            for i in range(len(segments_info) - 1):
                slice_points.append(i)

        # Create complete metadata
        # Format the original_model path to be consistent with the expected format
        original_model_path = self.onnx_path
        if original_model_path and os.path.isabs(original_model_path):
            # Convert absolute path to relative path format like "models/doom/model.onnx"
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(original_model_path)))
            original_model_path = os.path.relpath(original_model_path, base_dir)

        metadata = {
            "original_model": original_model_path,
            "model_type": "ONNX",
            "total_parameters": total_parameters,
            "slicing_strategy": "single_layer",
            "segments": segments_info,
            "slice_points": slice_points,
            "input_shapes": model_input_shapes,
            "output_shapes": model_output_shapes,
            "input_data_info": {
                "input_file": os.path.join(os.path.dirname(self.onnx_path), "input.json") if self.onnx_path else None,
                "input_shape": model_input_shapes[0] if model_input_shapes else []
            }
        }

        # Save metadata if output_dir is provided
        if output_dir and self.onnx_path:
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        return metadata

    def analyze_node(self, node, index, initializer_map, full_model_value_info_map):
        """
        Analyze a single node from the ONNX graph and gather metadata.

        Args:
            node: ONNX node to analyze
            index: Index of the node in the graph
            initializer_map: Map of initializer names to initializers
            full_model_value_info_map: Map of value info names to value infos

        Returns:
            dict: Metadata for the node
        """
        node_inputs = list(node.input)
        node_outputs = list(node.output)

        # Get input and output shapes
        input_shapes = self._get_tensor_shapes(node_inputs, full_model_value_info_map)
        output_shapes = self._get_tensor_shapes(node_outputs, full_model_value_info_map)

        # Determine layer type and gather parameter information
        layer_type, parameters, parameter_details = self._get_layer_info(node, node_inputs, initializer_map)

        # Determine in_features and out_features
        in_features, out_features = self._get_feature_info(node, parameter_details)

        # Determine activation function
        activation = self._get_activation_info(node)

        # Return node metadata
        return {
            "index": index,
            "type": layer_type,
            "segment_name": f"{layer_type}_{index}",
            "filename": f"segment_{index}.onnx",
            "path": None,  # This will be set by the slicer
            "layer_count": 1,
            "parameters": parameters,
            "activation": activation,
            "in_features": in_features,
            "out_features": out_features,
            "input_shape": input_shapes,
            "output_shape": output_shapes,
            "parameter_details": parameter_details,
            "dependencies": {
                "input": list(node.input),
                "output": list(node.output)
            }
        }

    def create_segments_from_metadata(self, node_metadata):
        """
        Create segments array from node metadata.

        Args:
            node_metadata: Dictionary of node metadata

        Returns:
            tuple: (segments, total_parameters)
        """
        segments = []
        total_parameters = 0

        for node_name, node_info in node_metadata.items():
            # Create layer information
            layer_info = self._create_layer_info(node_name, node_info)

            # Create segment information
            segment = {
                "index": node_info["index"],
                "type": node_info["type"],
                "segment_name": node_info["segment_name"],
                "filename": node_info["filename"],
                "path": node_info["path"],
                "layer_count": 1,
                "parameters": node_info.get("parameters", 0),
                "layers": [layer_info],
                "activation": node_info["activation"],
                "dependencies": node_info["dependencies"]
            }

            # Add in/out features to segment level if available
            if node_info.get("in_features") is not None:
                segment["in_features"] = node_info["in_features"]

            if node_info.get("out_features") is not None:
                segment["out_features"] = node_info["out_features"]

            segments.append(segment)

            # Add to total parameters
            total_parameters += node_info.get("parameters", 0)

        return segments, total_parameters

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

    def _get_tensor_shapes(self, tensor_names, value_info_map):
        """
        Get shapes for a list of tensors.

        Args:
            tensor_names: List of tensor names
            value_info_map: Map of value info names to value infos

        Returns:
            list: List of shapes for each tensor
        """
        shapes = []
        for name in tensor_names:
            if name in value_info_map:
                shape = []
                if value_info_map[name].type.tensor_type.shape.dim:
                    for dim in value_info_map[name].type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value if dim.dim_value != 0 else None)
                shapes.append(shape)
            else:
                shapes.append([None])
        return shapes

    def _get_layer_info(self, node, node_inputs, initializer_map):
        """
        Determine layer type and parameter information for a node.

        Args:
            node: ONNX node
            node_inputs: List of node inputs
            initializer_map: Map of initializer names to initializers

        Returns:
            tuple: (layer_type, parameters, parameter_details)
        """
        # Determine layer type based on op_type
        layer_type = "misc"
        if node.op_type == "Conv":
            layer_type = "conv"
        elif node.op_type == "Gemm":
            layer_type = "fc"
        elif node.op_type == "MatMul":
            layer_type = "fc"
        elif node.op_type == "BatchNormalization":
            layer_type = "norm"

        # Calculate parameters if possible
        parameters = 0
        parameter_details = {}

        # For Conv and Gemm nodes, we can extract parameter information from initializers
        if node.op_type == "Conv" or node.op_type == "Gemm":
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

        return layer_type, parameters, parameter_details

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
