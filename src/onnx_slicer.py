import os.path
import json
import onnx
from src.utils.onnx_analyzer import OnnxAnalyzer
from src.utils.onnx_utils import OnnxUtils
from typing import List, Dict, Any, Optional


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

    # def slice_model_with_points(self, slice_points: Optional[List[Dict[str, Any]]] = None,
    #                             strategy: str = "logical_layer"):
    #     """
    #     Slice the model based on the provided slice points or determine them using the specified strategy.
    #     This is step 3 of the workflow.
    #
    #     Args:
    #         slice_points: List of slice points to use. If None, they will be determined using the strategy.
    #         strategy: The strategy to use for determining slice points if not provided.
    #     """
    #     if not slice_points:
    #         if not self.slice_points:
    #             self.slice_points = self.determine_slice_points()
    #         slice_points = self.slice_points
    #
    #     # Create output directory
    #     output_dir = os.path.join(os.path.dirname(self.onnx_path), "onnx_slices")
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir, exist_ok=True)
    #
    #     # Store node metadata for generating the final metadata
    #     node_metadata = {}
    #
    #     # Create an analyzer for the model
    #     analyzer = OnnxAnalyzer(onnx_model=self.onnx_model, onnx_path=self.onnx_path)
    #
    #     # Process each slice point
    #     for slice_point in slice_points:
    #         i = slice_point["index"]
    #         nodes = slice_point["nodes"]
    #         actual_inputs = slice_point["inputs"]
    #         actual_outputs = slice_point["outputs"]
    #         node_initializers = slice_point["initializers"]
    #
    #         # Create and save the multi-node model
    #         model = OnnxUtils.create_multi_node_model(nodes, actual_inputs, actual_outputs, node_initializers)
    #         save_path = os.path.join(output_dir, f"segment_{i}.onnx")
    #         OnnxUtils.save_model(model, save_path)
    #
    #         # Get the node metadata from the analysis step
    #         primary_node = nodes[0]  # Use the first node as the primary node for metadata
    #         if self.model_metadata and primary_node.name in self.model_metadata["nodes"]:
    #             node_info = self.model_metadata["nodes"][primary_node.name].copy()
    #         else:
    #             # If we don't have metadata from analysis, analyze the node now
    #             graph = self.onnx_model.graph
    #             initializer_map = {init.name: init for init in graph.initializer}
    #             full_model_value_info_map = {vi.name: vi for vi in graph.value_info}
    #             full_model_value_info_map.update({vi.name: vi for vi in graph.input})
    #             full_model_value_info_map.update({vi.name: vi for vi in graph.output})
    #             node_info = analyzer.analyze_node(primary_node, i, initializer_map, full_model_value_info_map)
    #
    #         # Update node info with activation from the last node if it's an activation function
    #         if len(nodes) > 1 and nodes[-1].op_type in ['Relu', 'Sigmoid', 'Tanh']:
    #             node_info["activation"] = nodes[-1].op_type
    #
    #         node_info["path"] = save_path  # Update the path in the metadata
    #         node_metadata[primary_node.name] = node_info
    #
    #     # Create segments from node metadata
    #     segments, total_parameters = analyzer.create_segments_from_metadata(node_metadata)
    #
    #     # Generate metadata for the sliced model
    #     metadata = analyzer.generate_metadata(segments, output_dir)
    #     if metadata["original_model"] is None:
    #         metadata["original_model"] = self.onnx_path
    #
    #     # Add slicing strategy to metadata
    #     metadata["slicing_strategy"] = strategy
    #
    #     # Save metadata
    #     OnnxUtils.save_metadata_file(metadata, output_dir)
    #
    #     return metadata

    def slice(self, slice_points: List[int], model_metadata):
        """
        Slice the ONNX model based on the provided slice points.

        Args:
            slice_points: List of indices representing nodes with parameter details
            model_metadata: The model analysis metadata containing node information

        Returns:
            List[str]: Paths to the sliced model files
        """
        if not slice_points:
            raise ValueError("No slice points provided.")

        if not model_metadata or "nodes" not in model_metadata:
            raise ValueError("Invalid model metadata. Please run 'analyze()' first.")

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

        # Add the end of the model as a final slice point
        max_index = max(node_info["index"] for node_info in model_metadata["nodes"].values())
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

            # Collect nodes for this slice
            slice_nodes = []
            for idx in range(start_idx, end_idx):
                if idx in index_to_node_name:
                    node_name = index_to_node_name[idx]
                    if node_name in node_map:
                        slice_nodes.append(node_map[node_name])
                    else:
                        # Try to find the node using segment name (op_type_index)
                        segment_name = index_to_segment_name.get(idx)
                        if segment_name in node_type_index_map:
                            slice_nodes.append(node_type_index_map[segment_name])
                        else:
                            print(f"Warning: Node {node_name} (index {idx}) not found in the ONNX model")

            # Skip if no nodes in this slice
            if not slice_nodes:
                print(
                    f"Warning: No nodes found for segment {i - 1 if i > 0 else 0} (indices {start_idx}-{end_idx - 1})")
                continue

            # Determine inputs, outputs, and initializers for this slice
            slice_inputs = []
            slice_outputs = []
            slice_initializers = []

            # Get all outputs from nodes in this slice
            slice_node_outputs = set()
            for node in slice_nodes:
                for output in node.output:
                    slice_node_outputs.add(output)

            # Get all inputs from nodes in this slice
            slice_node_inputs = set()
            for node in slice_nodes:
                for inp in node.input:
                    slice_node_inputs.add(inp)

            # Inputs are those that are used by nodes in this slice but not produced by any node in this slice
            for inp in slice_node_inputs:
                if inp not in slice_node_outputs:
                    # Check if it's a model input or an initializer
                    if inp in value_info_map:
                        slice_inputs.append(value_info_map[inp])
                    elif inp in initializer_map:
                        init = initializer_map[inp]
                        slice_initializers.append(init)
                        # Create a value info for this initializer
                        t = onnx.helper.make_tensor_value_info(
                            inp,
                            init.data_type,
                            list(init.dims)
                        )
                        slice_inputs.append(t)
                    else:
                        # Create a dummy value info
                        t = onnx.helper.make_tensor_value_info(
                            inp,
                            onnx.TensorProto.FLOAT,
                            [None]
                        )
                        slice_inputs.append(t)

            # Outputs are those that are produced by nodes in this slice but not consumed by any node in this slice
            # or are model outputs
            for out in slice_node_outputs:
                # Check if this output is used as an input by any node in this slice
                is_output = True
                for node in slice_nodes:
                    if out in node.input:
                        is_output = False
                        break

                # If it's not used as an input or it's a model output, add it as a slice output
                if is_output or out in [o.name for o in graph.output]:
                    if out in value_info_map:
                        slice_outputs.append(value_info_map[out])
                    else:
                        # Create a dummy value info
                        t = onnx.helper.make_tensor_value_info(
                            out,
                            onnx.TensorProto.FLOAT,
                            [None]
                        )
                        slice_outputs.append(t)

            # Create the slice model
            # Create a graph with the nodes
            slice_graph = onnx.helper.make_graph(
                slice_nodes,
                f"segment_{segment_idx}_graph",
                slice_inputs,
                slice_outputs,
                slice_initializers
            )

            # Create a model from the graph
            slice_model = onnx.helper.make_model(slice_graph)

            # Save the slice model
            save_path = os.path.join(output_dir, f"segment_{segment_idx}.onnx")
            onnx.save(slice_model, save_path)
            slice_paths.append(save_path)

            print(f"Created slice segment_{segment_idx}.onnx with {len(slice_nodes)} nodes")

        return slice_paths

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
            metadata_path = os.path.join(os.path.dirname(self.onnx_path), "onnx_analysis", "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            else:
                raise ValueError("Model metadata not found. Please run 'analyze()' first.")

        # Step 2: Determine slice points
        slice_points = self.determine_slice_points(model_metadata)

        # Step 3: Slice the model
        return self.slice(slice_points, model_metadata)

if __name__ == "__main__":

    model_choice = 4 # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet",
        4: "models/yolov3"
    }

    model_dir = os.path.join(base_paths[model_choice], "model.onnx")
    onnx_analyzer = OnnxAnalyzer(model_path=model_dir)
    onnx_slicer = OnnxSlicer(model_dir)

    # Run the complete workflow: analyze, determine slice points, and slice
    model_analysis = onnx_analyzer.analyze()  # Produces the onnx_analysis/model_metadata.json file
    onnx_slicer.slice_model(model_analysis)  # this uses the model_metadata.json file to first get slice points, then it slices it.
