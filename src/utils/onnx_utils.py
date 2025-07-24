import os
import json
import logging
from pathlib import Path

import onnxruntime as ort
from onnxruntime.tools import optimize_onnx_model, symbolic_shape_infer
import onnxruntime_extensions as ortx
import numpy as np
import onnx
from onnx import shape_inference
import onnx_graphsurgeon as gs

# Set onnxruntime logger severity to suppress warnings
ort.set_default_logger_severity(3)  # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal

# Configure logger
logger = logging.getLogger('kubz.utils.onnx_utils')


class OnnxUtils:
    """
    Utility functions for working with ONNX models.
    """

    @staticmethod
    def save_metadata_file(metadata, output_path, filename="metadata.json"):
        """
        Save metadata to a JSON file.

        Args:
            metadata: Dictionary containing metadata
            output_path: Directory where the metadata will be saved
            filename: Name of the metadata file (default: "metadata.json")
        """
        output = Path(output_path)

        # Check if the provided path is a directory
        if output.is_dir():
            # Combine the directory with the default or given filename
            file_path = output / filename
        else:
            # Use the path as-is, assuming it includes the filename
            file_path = output

        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write metadata to the file
        with file_path.open('w') as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def create_node_model(node, inputs, outputs, initializers):
        """
        Create a minimal ONNX model containing a single node.

        Args:
            node: ONNX node to include in the model
            inputs: List of input tensor value infos
            outputs: List of output tensor value infos
            initializers: List of initializers needed by the node

        Returns:
            onnx.ModelProto: ONNX model containing the node
        """
        # Create a graph with the node
        graph = onnx.helper.make_graph(
            [node],  # single node
            f"{node.name}_graph",
            inputs,
            outputs,
            initializers
        )

        # Create a model from the graph
        model = onnx.helper.make_model(graph)

        return model

    @staticmethod
    def infer_shapes(model):
        """
        Infer shapes for an ONNX model.

        Args:
            model: ONNX model

        Returns:
            onnx.ModelProto: ONNX model with inferred shapes
        """
        try:
            logger.debug("Performing shape inference on ONNX model")
            inferred_model = shape_inference.infer_shapes(model)
            logger.debug("Shape inference completed successfully")
            return inferred_model
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}")
            return model

    @staticmethod
    def optimize_model(abs_path, model):
        try:
            logger.debug(f"Optimizing ONNX model with ONNX Runtime Extensions: {abs_path}")
            ortx.optimize_model(model, abs_path)
            model = onnx.load(abs_path)
            logger.debug("ONNX Runtime Extensions optimization completed successfully")
        except Exception as opt_error:
            logger.debug(f"ONNX Runtime Extensions optimization failed: {opt_error}, falling back to standard optimization")
            try:
                logger.debug("Attempting standard ONNX optimization")
                optimized_model = optimize_onnx_model.optimize_model(abs_path, output_path=abs_path)
                if optimized_model is not None:
                    model = optimized_model
                    logger.debug("Standard ONNX optimization completed successfully")
            except Exception as fallback_error:
                logger.info(f"Fallback optimization failed: {fallback_error}, continuing with original model")
        return model

    @staticmethod
    def add_shape_inference(model, model_metadata, path):
        try:
            logger.debug("Performing symbolic shape inference")
            model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model, verbose=0)
            logger.debug("Symbolic shape inference completed successfully")
        except Exception as shape_error:
            logger.info(f"Symbolic shape inference failed: {shape_error}, falling back to metadata-driven shape application")
            # Fall back to metadata-driven shape application
            if model_metadata:
                try:
                    # Extract segment index from path
                    segment_idx = int(path.split('segment_')[1].split('.')[0])
                    logger.debug(f"Applying metadata shapes for segment {segment_idx}")
                    model = OnnxUtils.apply_metadata_shapes(model, model_metadata)
                    logger.info("Metadata-driven shape application successful")
                except Exception as metadata_error:
                    logger.info(f"Metadata-driven shape application failed: {metadata_error}, continuing without shape inference")
            else:
                logger.info("No metadata available for fallback shape application")
        return model

    @staticmethod
    def get_tensor_value_info(name, dtype, shape):
        """
        Create a tensor value info.

        Args:
            name: Name of the tensor
            dtype: Data type of the tensor (e.g., onnx.TensorProto.FLOAT)
            shape: Shape of the tensor

        Returns:
            onnx.ValueInfoProto: Tensor value info
        """
        return onnx.helper.make_tensor_value_info(name, dtype, shape)

    @staticmethod
    def extract_shape_from_value_info(value_info):
        """
        Extract shape from a value info.

        Args:
            value_info: ONNX value info

        Returns:
            list: Shape of the tensor
        """
        shape = []
        if value_info.type.tensor_type.shape.dim:
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value if dim.dim_value != 0 else None)
        return shape

    @staticmethod
    def filter_inputs(segment_inputs, graph):
        # Filter input names from segment details
        segment_filtered_inputs = []
        for input_info in segment_inputs:
            # Only include actual inputs that are not weights or biases
            # Typically, weights and biases have names containing "weight" or "bias"
            if (not any(pattern in input_info.name.lower() for pattern in ["weight", "bias"]) and
                    input_info.name in [inp.name for inp in graph.input]):
                segment_filtered_inputs.append(input_info.name)
            # Also include intermediate tensors from previous layers
            elif input_info.name.startswith('/'):  # Intermediate tensors often start with '/'
                segment_filtered_inputs.append(input_info.name)
        # If there are no inputs after filtering, include the first non-weight/bias input
        if not segment_filtered_inputs:
            for input_info in segment_inputs:
                if not any(pattern in input_info.name.lower() for pattern in ["weight", "bias"]):
                    segment_filtered_inputs.append(input_info.name)
                    break

            # If still no inputs, use the first input as a fallback
            if not segment_filtered_inputs and segment_inputs:
                segment_filtered_inputs.append(segment_inputs[0].name)
        return segment_filtered_inputs

    @staticmethod
    def _get_original_model_shapes(model_metadata: dict):
        """
        Extract shape information from model metadata.

        Args:
            model_metadata: Dictionary containing model metadata with shape information

        Returns:
            dict: Dictionary mapping tensor names to their shapes
        """
        shapes = {}

        # Extract shapes from input_shape
        input_shape = model_metadata.get("input_shape", [])
        if input_shape and len(input_shape) > 0:
            shapes["input"] = input_shape[0]

        # Extract shapes from output_shapes
        output_shapes = model_metadata.get("output_shapes", [])
        if output_shapes and len(output_shapes) > 0:
            shapes["output"] = output_shapes[0]

        # Extract shapes from nodes if available
        nodes = model_metadata.get("nodes", {})
        for node_name, node_info in nodes.items():
            if "parameter_details" in node_info:
                for param_name, param_info in node_info["parameter_details"].items():
                    if "shape" in param_info:
                        shapes[param_name] = param_info["shape"]

        return shapes

    @staticmethod
    def apply_metadata_shapes(model, model_metadata: dict):
        """
        Apply shape information from metadata to the model.

        Args:
            model: ONNX model to process
            model_metadata: Dictionary containing model metadata with shape information

        Returns:
            ONNX model with shapes applied from metadata, or original model if GraphSurgeon unavailable
        """
        try:
            logger.debug("Starting metadata shape application process")
            graph = gs.import_onnx(model)

            # Get original model shapes for reference
            original_shapes = OnnxUtils._get_original_model_shapes(model_metadata)
            logger.debug(f"Original model shapes: {original_shapes}")

            # Apply proper shapes to all tensors based on original model
            shapes_applied = 0
            missing_shapes = []

            # First pass: apply shapes from original model
            for tensor_name, tensor in graph.tensors().items():
                if isinstance(tensor, gs.Variable):
                    # Ensure dtype is set
                    if not hasattr(tensor, 'dtype') or tensor.dtype is None:
                        tensor.dtype = np.float32

                    # Apply shape from original model if available
                    if tensor_name in original_shapes:
                        tensor.shape = original_shapes[tensor_name]
                        logger.debug(f"Applied shape {original_shapes[tensor_name]} to {tensor_name}")
                        shapes_applied += 1
                    elif tensor_name == "input":
                        # Apply model input shape to main input
                        model_input_shape = model_metadata.get("input_shape", [])
                        if model_input_shape and len(model_input_shape) > 0:
                            first_input_shape = model_input_shape[0]
                            if first_input_shape and not str(first_input_shape[0]).startswith('batch'):
                                first_input_shape = ["batch_size"] + first_input_shape[1:]
                            tensor.shape = first_input_shape
                            logger.debug(f"Applied input shape {first_input_shape} to {tensor_name}")
                            shapes_applied += 1
                    else:
                        # Keep track of tensors with missing shapes
                        missing_shapes.append(tensor_name)

            # Second pass: try to infer missing shapes using onnx-graphsurgeon
            if missing_shapes:
                logger.debug(f"Attempting to infer shapes for {len(missing_shapes)} tensors")
                try:
                    # Try to infer shapes using graph operations
                    for node in graph.nodes:
                        for i, output in enumerate(node.outputs):
                            if output.name in missing_shapes and output.shape is None:
                                # Try to infer shape from node attributes and inputs
                                if node.op == "Reshape" and len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
                                    # For Reshape nodes, shape is in the second input
                                    output.shape = node.inputs[1].values.tolist()
                                    logger.debug(f"Inferred shape {output.shape} for {output.name} from Reshape node")
                                    shapes_applied += 1
                                    missing_shapes.remove(output.name)
                                elif node.op == "Transpose" and node.inputs[0].shape is not None:
                                    # For Transpose nodes, permute the input shape
                                    perm = node.attrs.get("perm", list(range(len(node.inputs[0].shape) - 1, -1, -1)))
                                    output.shape = [node.inputs[0].shape[p] for p in perm]
                                    logger.debug(f"Inferred shape {output.shape} for {output.name} from Transpose node")
                                    shapes_applied += 1
                                    missing_shapes.remove(output.name)
                except Exception as e:
                    logger.debug(f"Error during shape inference: {e}")

            # Third pass: look for missing shapes in segments section of model_metadata
            if missing_shapes and "segments" in model_metadata:
                logger.debug(f"Looking for shapes in segments section for {len(missing_shapes)} tensors")
                for segment in model_metadata["segments"]:
                    # Check shape information in segment
                    if "shape" in segment:
                        for tensor_name in missing_shapes[:]:  # Use a copy to safely modify during iteration
                            if tensor_name in segment["shape"]:
                                # Find the tensor in the graph
                                for graph_tensor_name, tensor in graph.tensors().items():
                                    if graph_tensor_name == tensor_name and isinstance(tensor, gs.Variable):
                                        tensor.shape = segment["shape"][tensor_name]
                                        logger.debug(f"Applied shape {segment['shape'][tensor_name]} to {tensor_name} from segments")
                                        shapes_applied += 1
                                        missing_shapes.remove(tensor_name)
                                        break

                    # Check layers information in segment
                    if "layers" in segment:
                        for layer in segment["layers"]:
                            if "parameter_details" in layer:
                                for tensor_name in missing_shapes[:]:
                                    # Check if tensor name matches any layer name or is in parameter_details
                                    if tensor_name == layer["name"] or tensor_name in layer["parameter_details"]:
                                        shape_info = None
                                        if tensor_name in layer["parameter_details"]:
                                            shape_info = layer["parameter_details"][tensor_name].get("shape")
                                        elif "in_features" in layer["parameter_details"] and "out_features" in layer["parameter_details"]:
                                            # Construct shape based on in/out features
                                            in_features = layer["parameter_details"]["in_features"]
                                            out_features = layer["parameter_details"]["out_features"]
                                            if layer["type"] == "Conv":
                                                # For Conv layers, shape depends on kernel size
                                                kernel_size = 3  # Default kernel size
                                                shape_info = [out_features, in_features, kernel_size, kernel_size]
                                            elif layer["type"] == "Gemm":
                                                # For Gemm layers, shape is [out_features, in_features]
                                                shape_info = [out_features, in_features]

                                        if shape_info:
                                            # Find the tensor in the graph
                                            for graph_tensor_name, tensor in graph.tensors().items():
                                                if graph_tensor_name == tensor_name and isinstance(tensor, gs.Variable):
                                                    tensor.shape = shape_info
                                                    logger.debug(f"Applied shape {shape_info} to {tensor_name} from layer info")
                                                    shapes_applied += 1
                                                    missing_shapes.remove(tensor_name)
                                                    break

            if shapes_applied > 0:
                logger.info(f"Successfully applied {shapes_applied} shapes from metadata")
                return gs.export_onnx(graph)
            else:
                logger.debug("No shapes applied from metadata")
                return model

        except Exception as e:
            logger.warning(f"Failed to apply metadata shapes: {e}")
            return model
