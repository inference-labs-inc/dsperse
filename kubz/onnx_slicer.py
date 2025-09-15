import os.path
import json
import onnx
from kubz.utils.onnx_analyzer import OnnxAnalyzer
from pathlib import Path
import onnxruntime_extensions as ortx
import onnxruntime as ort
from onnxruntime.tools import optimize_onnx_model, symbolic_shape_infer
from typing import List, Dict


class OnnxSlicer:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        # load onnx model
        self.onnx_model = onnx.load(onnx_path)
        self.model_metadata = None
        self.slice_points = None

    def determine_slice_points(self, model_metadata) -> List[int]:
        """
        Determine the slice points for the model based on nodes with parameter_details in the model_metadata.

        Args:
            model_metadata: The model analysis metadata containing node information.

        Returns:
            List[int]: List of indices representing nodes with parameter details
        """
        if not model_metadata or "nodes" not in model_metadata:
            raise ValueError("Invalid model metadata. Please run 'analyze()' first.")

        # Find nodes with parameter_details in model_metadata
        slice_points = []
        for node_name, node_info in model_metadata["nodes"].items():
            if node_info.get("parameter_details") and node_info["parameter_details"]:
                slice_points.append(node_info["index"])

        # Sort slice points by index
        slice_points.sort()

        self.slice_points = slice_points
        return slice_points

    def _slice_setup(self, model_metadata):
        """
        Set up the necessary data structures for slicing.

        Args:
            model_metadata: The model analysis metadata containing node information

        Returns:
            tuple: (graph, node_map, node_type_index_map, initializer_map, value_info_map,
                    index_to_node_name, index_to_segment_name, output_dir)
        """
        # Create output directory
        output_dir = os.path.join(os.path.dirname(self.onnx_path), "onnx_slices")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Get the graph from the ONNX model
        graph = self.onnx_model.graph

        # Create maps for node lookup
        node_map = {node.name: node for node in graph.node}

        # Also create a map with just the op_type and index to handle name mismatches
        node_type_index_map = {}
        for i, node in enumerate(graph.node):
            key = f"{node.op_type}_{i}"
            node_type_index_map[key] = node

        initializer_map = {init.name: init for init in graph.initializer}
        value_info_map = {vi.name: vi for vi in graph.value_info}
        value_info_map.update({vi.name: vi for vi in graph.input})
        value_info_map.update({vi.name: vi for vi in graph.output})

        # Create a map of node indices to node names
        index_to_node_name = {}
        index_to_segment_name = {}
        for node_name, node_info in model_metadata["nodes"].items():
            index_to_node_name[node_info["index"]] = node_name
            index_to_segment_name[node_info["index"]] = node_info["segment_name"]

        return (
            graph,
            node_map,
            node_type_index_map,
            initializer_map,
            value_info_map,
            index_to_node_name,
            index_to_segment_name,
            output_dir,
        )

    @staticmethod
    def _get_nodes(
        start_idx,
        end_idx,
        index_to_node_name,
        index_to_segment_name,
        node_map,
        node_type_index_map,
        segment_idx,
    ):
        """
        Collect nodes for a specific slice.

        Args:
            start_idx: Start index of the slice
            end_idx: End index of the slice
            index_to_node_name: Map of node indices to node names
            index_to_segment_name: Map of node indices to segment names
            node_map: Map of node names to nodes
            node_type_index_map: Map of node type and index to nodes
            segment_idx: Index of the current segment

        Returns:
            list: List of nodes for this slice
        """
        segment_nodes = []
        for idx in range(start_idx, end_idx):
            if idx in index_to_node_name:
                node_name = index_to_node_name[idx]
                if node_name in node_map:
                    segment_nodes.append(node_map[node_name])
                else:
                    # Try to find the node using segment name (op_type_index)
                    segment_name = index_to_segment_name.get(idx)
                    if segment_name in node_type_index_map:
                        segment_nodes.append(node_type_index_map[segment_name])
                    else:
                        print(
                            f"Warning: Node {node_name} (index {idx}) not found in the ONNX model"
                        )

        # Skip if no nodes in this slice
        if not segment_nodes:
            print(
                f"Warning: No nodes found for segment {segment_idx} (indices {start_idx}-{end_idx - 1})"
            )

        return segment_nodes

    @staticmethod
    def _get_segment_details(segment_nodes, graph, value_info_map, initializer_map):
        """
        Determine inputs, outputs, and initializers for a segment.

        Args:
            segment_nodes: List of nodes in the segment
            graph: ONNX graph
            value_info_map: Map of value info names to value infos
            initializer_map: Map of initializer names to initializers

        Returns:
            tuple: (segment_inputs, segment_outputs, segment_initializers)
        """
        segment_inputs = []
        segment_outputs = []
        segment_initializers = []

        # Build a complete map of all value infos including intermediate outputs
        all_value_infos = {}

        # Add model inputs
        for input_info in graph.input:
            all_value_infos[input_info.name] = input_info

        # Add model outputs
        for output_info in graph.output:
            all_value_infos[output_info.name] = output_info

        # Add any intermediate value infos
        for value_info in graph.value_info:
            all_value_infos[value_info.name] = value_info

        # Get all outputs from nodes in this segment
        segment_node_outputs = set()
        for node in segment_nodes:
            for output in node.output:
                segment_node_outputs.add(output)

        # Get all inputs from nodes in this segment
        segment_node_inputs = set()
        for node in segment_nodes:
            for inp in node.input:
                segment_node_inputs.add(inp)

        # Inputs are those that are used by nodes in this segment but not produced by any node in this segment
        for inp in segment_node_inputs:
            if inp not in segment_node_outputs:
                # Check if it's a model input, intermediate value, or an initializer
                if inp in all_value_infos:
                    segment_inputs.append(all_value_infos[inp])
                elif inp in initializer_map:
                    init = initializer_map[inp]
                    segment_initializers.append(init)
                    # Create a value info for this initializer
                    t = onnx.helper.make_tensor_value_info(
                        inp, init.data_type, list(init.dims)
                    )
                    segment_inputs.append(t)
                else:
                    # For unknown intermediate tensors, we need to infer reasonable shapes
                    # Look at the node that would consume this input to guess the shape
                    inferred_shape = OnnxSlicer._infer_input_shape(inp, segment_nodes)
                    t = onnx.helper.make_tensor_value_info(
                        inp, onnx.TensorProto.FLOAT, inferred_shape
                    )
                    segment_inputs.append(t)

        # Outputs are those that are produced by nodes in this segment but not consumed by any node in this segment
        # or are model outputs
        for out in segment_node_outputs:
            # Check if this output is used as an input by any node in this segment
            is_output = True
            for node in segment_nodes:
                if out in node.input:
                    is_output = False
                    break

            # If it's not used as an input or it's a model output, add it as a segment output
            if is_output or out in [o.name for o in graph.output]:
                if out in all_value_infos:
                    segment_outputs.append(all_value_infos[out])
                else:
                    # For unknown outputs, infer shape from the producing node
                    inferred_shape = OnnxSlicer._infer_output_shape(out, segment_nodes)
                    t = onnx.helper.make_tensor_value_info(
                        out, onnx.TensorProto.FLOAT, inferred_shape
                    )
                    segment_outputs.append(t)

        return segment_inputs, segment_outputs, segment_initializers

    @staticmethod
    def _infer_input_shape(input_name, segment_nodes):
        """
        Infer a reasonable shape for an input tensor based on the nodes that consume it.
        """
        for node in segment_nodes:
            if input_name in node.input:
                if node.op_type == "Conv":
                    # Conv expects 4D input: [batch, channels, height, width]
                    return ["batch_size", None, None, None]
                elif node.op_type == "Gemm":
                    # Gemm expects 2D input: [batch, features]
                    return ["batch_size", None]
                elif node.op_type in ["Relu", "BatchNormalization"]:
                    # These preserve input shape, so use a flexible 4D shape
                    return ["batch_size", None, None, None]

        # Default fallback for unknown cases
        return ["batch_size", None]

    @staticmethod
    def _infer_output_shape(output_name, segment_nodes):
        """
        Infer a reasonable shape for an output tensor based on the node that produces it.
        """
        for node in segment_nodes:
            if output_name in node.output:
                if node.op_type == "Conv":
                    # Conv output is 4D: [batch, out_channels, height, width]
                    return ["batch_size", None, None, None]
                elif node.op_type == "Gemm":
                    # Gemm output is 2D: [batch, out_features]
                    return ["batch_size", None]
                elif node.op_type in ["Relu", "BatchNormalization"]:
                    # These preserve input shape
                    return ["batch_size", None, None, None]
                elif node.op_type == "Reshape":
                    # Reshape output depends on the target shape
                    return ["batch_size", None]

        # Default fallback
        return ["batch_size", None]

    def slice(self, slice_points: List[int], model_metadata):
        """
        Slice the ONNX model based on the provided slice points.

        Args:
            slice_points: List of indices representing nodes with parameter details
            model_metadata: The model analysis metadata containing node information

        Returns:
            List[str]: Paths to the sliced model files
        """
        # Error handling
        if not slice_points:
            raise ValueError("No slice points provided.")

        if not model_metadata or "nodes" not in model_metadata:
            raise ValueError("Invalid model metadata. Please run 'analyze()' first.")

        # Set up slicing environment
        (
            graph,
            node_map,
            node_type_index_map,
            initializer_map,
            value_info_map,
            index_to_node_name,
            index_to_segment_name,
            output_dir,
        ) = self._slice_setup(model_metadata)

        # Add the end of the model as a final slice point
        max_index = max(
            node_info["index"] for node_info in model_metadata["nodes"].values()
        )
        # Always add max_index + 1 to ensure we create a segment for the last node
        if max_index + 1 not in slice_points:
            slice_points.append(max_index + 1)

        # Sort slice points to ensure they're in order
        slice_points.sort()

        # Store paths to sliced models
        slice_paths = []

        # Process each segment
        for i in range(len(slice_points)):
            segment_idx = i - 1
            start_idx = slice_points[i - 1] if i > 0 else 0
            end_idx = slice_points[i]

            # Skip if start and end are the same
            if start_idx == end_idx:
                continue

            # Get nodes for this segment
            segment_nodes = self._get_nodes(
                start_idx,
                end_idx,
                index_to_node_name,
                index_to_segment_name,
                node_map,
                node_type_index_map,
                segment_idx,
            )

            # Skip if no nodes in this segment
            if not segment_nodes:
                continue

            # Get segment details
            segment_inputs, segment_outputs, segment_initializers = (
                self._get_segment_details(
                    segment_nodes, graph, value_info_map, initializer_map
                )
            )

            # Create the segment model
            # Create a graph with the nodes
            segment_graph = onnx.helper.make_graph(
                segment_nodes,
                f"segment_{segment_idx}_graph",
                segment_inputs,
                segment_outputs,
                segment_initializers,
            )

            # Create a model from the graph
            segment_model = onnx.helper.make_model(segment_graph)

            # Save the segment model
            save_path = os.path.join(output_dir, f"segment_{segment_idx}.onnx")
            onnx.save(segment_model, save_path)
            slice_paths.append(save_path)

        return slice_paths

    @staticmethod
    def slice_post_process(slices_paths):
        for path in slices_paths:
            print(f"Processing {path}")
            # Convert to absolute path if it's relative
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), path)
            print(f"Path: {path}")

            try:
                # Load the model
                model = onnx.load(path)
                print(f"Model loaded successfully")

                # Check if model is valid before optimization
                onnx.checker.check_model(model)
                print(f"Model validation passed")

                # Create output path for optimized model
                path_obj = Path(path)
                optimized_path = str(path_obj)

                try:
                    # Use onnxruntime-extensions for optimization
                    # This provides more advanced optimization than the basic onnxruntime optimizer
                    ortx.optimize_model(model, optimized_path)
                    print(f"Model optimization with onnxruntime-extensions successful")

                    # Load the optimized model
                    model = onnx.load(optimized_path)
                except Exception as opt_error:
                    print(
                        f"Optimization with onnxruntime-extensions failed: {opt_error}, trying fallback optimization"
                    )

                    # Fallback to original optimization method
                    try:
                        optimized_model = optimize_onnx_model.optimize_model(
                            path_obj, output_path=path_obj
                        )
                        if optimized_model is not None:
                            model = optimized_model
                            print(f"Fallback model optimization successful")
                        else:
                            print(
                                f"Fallback model optimization returned None, using original model"
                            )
                    except Exception as fallback_error:
                        print(
                            f"Fallback optimization failed: {fallback_error}, continuing with original model"
                        )

                # Try shape inference
                try:
                    model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
                        model
                    )
                    print(f"Shape inference successful")
                except Exception as shape_error:
                    print(
                        f"Shape inference failed: {shape_error}, continuing without shape inference"
                    )

                # Save the processed model
                onnx.save(model, path)
                print(f"Model saved successfully to {path}")

                # Additional verification step - try to create a session with the optimized model
                try:
                    # Create session options with additional optimizations
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    session_options.optimized_model_filepath = str(
                        path_obj
                    )  # + ".optimized"

                    # Register custom ops from onnxruntime-extensions
                    session_options.register_custom_ops_library(ortx.get_library_path())

                    # Create a session to verify and further optimize the model
                    _ = ort.InferenceSession(path, session_options)
                    print(f"Additional optimization and verification successful")
                except Exception as verify_error:
                    print(
                        f"Additional optimization verification failed: {verify_error}, but model should still be usable"
                    )

            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Continue with next slice instead of failing completely
                continue

    def slice_model(self, model_metadata=None):
        """
        Run the complete workflow: determine slice points and slice.

        Args:
            model_metadata: The model analysis metadata. If None, it will be determined.

        Returns:
            Dict[str, Any]: Metadata about the sliced model
        """
        # Step 1: Set model metadata if provided
        if not model_metadata:
            # Check if model metadata exists in onnx_analysis directory
            metadata_path = os.path.join(
                os.path.dirname(self.onnx_path), "onnx_analysis", "model_metadata.json"
            )
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.model_metadata = json.load(f)
            else:
                raise ValueError(
                    "Model metadata not found. Please run 'analyze()' first."
                )

        # Step 2: Determine slice points
        slice_points = self.determine_slice_points(model_metadata)

        # Step 3: Slice the model
        slices_paths = self.slice(slice_points, model_metadata)
        self.slice_post_process(slices_paths)

        # Step 4: generate slices metadata
        onnx_analyzer.generate_slices_metadata(
            model_metadata,
            slice_points,
            os.path.join(os.path.dirname(self.onnx_path), "onnx_slices"),
        )

        return slices_paths


if __name__ == "__main__":

    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet",
        4: "models/yolov3",
    }

    model_dir = os.path.join(base_paths[model_choice], "model.onnx")
    onnx_analyzer = OnnxAnalyzer(model_path=model_dir)
    onnx_slicer = OnnxSlicer(model_dir)

    # Run the complete workflow: analyze, determine slice points, and slice
    model_analysis = (
        onnx_analyzer.analyze()
    )  # Produces the onnx_analysis/model_metadata.json file
    onnx_slicer.slice_model(
        model_analysis
    )  # this uses the model_metadata.json file to first get slice points, then it slices it.
