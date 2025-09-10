#!/usr/bin/env python3
"""
Simplified ONNX Model Slicer - No GraphSurgeon or Complex Nodes
"""

import os.path
import json
import onnx
from onnx import helper, numpy_helper
from typing import List, Dict, Tuple, Set
import onnxruntime as ort
from onnxruntime.tools import optimize_onnx_model, symbolic_shape_infer


class SimpleOnnxSlicer:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        # load onnx model
        self.onnx_model = onnx.load(onnx_path)
        self.model_metadata = None
        self.slice_points = None

    def determine_slice_points(self, model_metadata) -> List[int]:
        """
        Determine the slice points for the model based on nodes with parameter_details.

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

        # Sort slice points by index and remove duplicates
        slice_points = sorted(list(set(slice_points)))

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
        output_dir = os.path.join(os.path.dirname(self.onnx_path), "simple_slices")
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

        return (graph, node_map, node_type_index_map, initializer_map, value_info_map,
                index_to_node_name, index_to_segment_name, output_dir)

    @staticmethod
    def _get_nodes(start_idx, end_idx, index_to_node_name, index_to_segment_name, node_map, node_type_index_map,
                   segment_idx):
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
                        print(f"Warning: Node {node_name} (index {idx}) not found in the ONNX model")

        # Skip if no nodes in this slice
        if not segment_nodes:
            print(f"Warning: No nodes found for segment {segment_idx} (indices {start_idx}-{end_idx - 1})")

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
                        inp,
                        init.data_type,
                        list(init.dims)
                    )
                    segment_inputs.append(t)
                else:
                    # For unknown intermediate tensors, create basic tensor info
                    t = onnx.helper.make_tensor_value_info(
                        inp,
                        onnx.TensorProto.FLOAT,
                        ["batch_size", None]  # Default shape
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
                    # For unknown outputs, create basic tensor info
                    t = onnx.helper.make_tensor_value_info(
                        out,
                        onnx.TensorProto.FLOAT,
                        ["batch_size", None]  # Default shape
                    )
                    segment_outputs.append(t)

        return segment_inputs, segment_outputs, segment_initializers

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

        # Apply basic shape inference to the original model
        print("🔧 Applying shape inference to original model...")
        try:
            self.onnx_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(self.onnx_model)
            print("✅ Shape inference applied successfully to original model")
        except Exception as e:
            print(f"⚠️  Shape inference failed on original model: {e}, continuing with original model")

        # Set up slicing environment
        (graph, node_map, node_type_index_map, initializer_map, value_info_map,
         index_to_node_name, index_to_segment_name, output_dir) = self._slice_setup(model_metadata)

        # Add the end of the model as a final slice point
        max_index = max(node_info["index"] for node_info in model_metadata["nodes"].values())
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
            segment_nodes = self._get_nodes(start_idx, end_idx, index_to_node_name,
                                            index_to_segment_name, node_map, node_type_index_map, segment_idx)

            # Skip if no nodes in this segment
            if not segment_nodes:
                continue

            # Get segment details
            segment_inputs, segment_outputs, segment_initializers = self._get_segment_details(
                segment_nodes, graph, value_info_map, initializer_map)

            # Create the segment model
            segment_graph = onnx.helper.make_graph(
                segment_nodes,
                f"segment_{segment_idx}_graph",
                segment_inputs,
                segment_outputs,
                segment_initializers
            )

            # Create a model from the graph
            segment_model = onnx.helper.make_model(segment_graph)

            # Fix Pad and Unsqueeze attributes before shape inference
            for node in segment_model.graph.node:
                if node.op_type == "Pad" and len(node.input) < 2:
                    # Convert pads attribute to input
                    for attr in node.attribute:
                        if attr.name == "pads":
                            pads_tensor = onnx.helper.make_tensor(
                                name=f"{node.name or node.output[0]}_pads",
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(attr.ints)],
                                vals=list(attr.ints)
                            )
                            segment_model.graph.initializer.append(pads_tensor)
                            node.input.append(pads_tensor.name)
                            node.attribute.remove(attr)
                            break

                    # Handle constant_value if present
                    for attr in list(node.attribute):
                        if attr.name == "value":
                            const_tensor = onnx.helper.make_tensor(
                                name=f"{node.name or node.output[0]}_value",
                                data_type=onnx.TensorProto.FLOAT,
                                dims=[],
                                vals=[attr.f]
                            )
                            segment_model.graph.initializer.append(const_tensor)
                            if len(node.input) < 3:
                                node.input.append(const_tensor.name)
                            node.attribute.remove(attr)
                            break

                elif node.op_type == "Unsqueeze" and len(node.input) < 2:
                    # Convert axes attribute to input
                    for attr in node.attribute:
                        if attr.name == "axes":
                            axes_tensor = onnx.helper.make_tensor(
                                name=f"{node.name or node.output[0]}_axes",
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(attr.ints)],
                                vals=list(attr.ints)
                            )
                            segment_model.graph.initializer.append(axes_tensor)
                            node.input.append(axes_tensor.name)
                            node.attribute.remove(attr)
                            break

            # Apply shape inference to each segment
            print(f"🔧 Applying shape inference to segment {segment_idx}...")
            try:
                segment_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(segment_model)
                print(f"✅ Shape inference applied successfully to segment {segment_idx}")
            except Exception as e:
                print(f"⚠️  Shape inference failed on segment {segment_idx}: {e}")

            # Save the segment model
            save_path = os.path.join(output_dir, f"segment_{segment_idx}.onnx")
            onnx.save(segment_model, save_path)
            slice_paths.append(save_path)

            print(f"✅ Created segment {segment_idx}: {os.path.basename(save_path)}")

        return slice_paths

    @staticmethod
    def validate_segment(segment_path: str) -> bool:
        """
        Validate that a sliced segment is a valid ONNX model.

        Args:
            segment_path: Path to the segment file

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Load the model
            model = onnx.load(segment_path)

            # Check if model is valid
            onnx.checker.check_model(model)

            # Try to create a session (basic validation)
            session = ort.InferenceSession(segment_path)
            print(f"✅ Segment validation passed: {os.path.basename(segment_path)}")
            return True

        except Exception as e:
            print(f"❌ Segment validation failed: {os.path.basename(segment_path)} - {e}")
            return False

    def slice_model(self, model_metadata=None):
        """
        Run the complete slicing workflow.

        Args:
            model_metadata: The model analysis metadata. If None, it will be determined.

        Returns:
            Dict[str, Any]: Metadata about the sliced model
        """
        # Set model metadata if provided
        if model_metadata:
            self.model_metadata = model_metadata
        else:
            # Try to load from analysis directory
            metadata_path = os.path.join(os.path.dirname(self.onnx_path), "simple_analysis", "simple_model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            else:
                raise ValueError("Model metadata not found. Please run analyzer first.")

        # Determine slice points
        slice_points = self.determine_slice_points(self.model_metadata)

        print(f"\n=== Simple Slicing Strategy ===")
        print(f"Total nodes: {self.model_metadata['node_count']}")
        print(f"Nodes with parameters: {len(slice_points)}")
        print(f"Slice points: {slice_points}")

        # Slice the model
        slices_paths = self.slice(slice_points, self.model_metadata)

        # Validate all segments
        print(f"\n=== Validation Results ===")
        valid_count = 0
        for path in slices_paths:
            if self.validate_segment(path):
                valid_count += 1

        print(f"Valid segments: {valid_count}/{len(slices_paths)}")

        return {
            'slice_paths': slices_paths,
            'slice_points': slice_points,
            'total_segments': len(slices_paths),
            'valid_segments': valid_count,
            'model_metadata': self.model_metadata
        }
