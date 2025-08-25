#!/usr/bin/env python3
"""
Nested FFT Slicer
Further slices FFT-decomposed models into smaller chunks based on FFT sub-circuits.
Each Conv → FFT decomposition (DFT → Mul → DFT) gets split into separate segments.
Follows the exact same format and logic as slicer.py and onnx_slicer.py.
Creates flattened segments with names like segment_0_1, segment_0_2, etc. in a single parent folder.
"""

import os.path
import onnx
import logging
from src.analyzers.onnx_analyzer import OnnxAnalyzer
from typing import List, Dict
from src.utils.utils import Utils
from onnx.utils import extract_model
import shutil

# Configure logger
logger = logging.getLogger(__name__)


class NestedFFTSlicer:
    def __init__(self, onnx_path, save_path=None):
        self.onnx_path = onnx_path
        self.onnx_model = onnx.load(onnx_path)
        self.model_metadata = None
        self.slice_points = None

        self.onnx_analyzer = OnnxAnalyzer(onnx_path)
        self.analysis = self.onnx_analyzer.analyze(save_path=save_path)
        self.graph = self.onnx_model.graph

    def determine_slice_points(self, model_metadata) -> List[int]:
        """
        Determine the slice points for the model based on DFT and Mul nodes from FFT decomposition.
        Each FFT sub-circuit (DFT → Mul → DFT) becomes a separate slice.

        Args:
            model_metadata: The model analysis metadata containing node information

        Returns:
            List[int]: List of indices representing start of each FFT sub-circuit
        """
        slice_points = []
        
        for i, node in enumerate(self.graph.node):
            if node.op_type == 'DFT':
                attrs = {attr.name: attr for attr in node.attribute}
                inverse = attrs.get('inverse').i if 'inverse' in attrs else 0
                
                if inverse == 0:
                    # Forward FFT - start of a new FFT sub-circuit
                    if node.name.endswith('_fft') or node.name.endswith('_kernel_fft'):
                        slice_points.append(i)
                elif inverse == 1 and node.name.endswith('_ifft'):
                    # IFFT - end of current FFT sub-circuit, start of next
                    slice_points.append(i + 1)  # Start of next segment after IFFT
        
        # Add start and end points
        if slice_points and slice_points[0] != 0:
            slice_points.insert(0, 0)
        if slice_points and slice_points[-1] != len(self.graph.node):
            slice_points.append(len(self.graph.node))
        
        # Sort slice points by index
        slice_points.sort()

        self.slice_points = slice_points
        return slice_points

    def _slice_setup(self, model_metadata, output_path=None):
        """
        Set up the necessary data structures for slicing.

        Args:
            model_metadata: The model analysis metadata containing node information

        Returns:
            tuple: (graph, node_map, node_type_index_map, initializer_map, value_info_map,
                    index_to_node_name, index_to_segment_name, output_dir)
        """
        # Create output directory
        output_path = os.path.join(os.path.dirname(self.onnx_path), "slices") if output_path is None else output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

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
                index_to_node_name, index_to_segment_name, output_path)

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
                        logger.warning(f"Node {node_name} (index {idx}) not found in the ONNX model")

        # Skip if no nodes in this slice
        if not segment_nodes:
            logger.warning(f"No nodes found for segment {segment_idx} (indices {start_idx}-{end_idx - 1})")

        return segment_nodes

    @staticmethod
    def _get_segment_details(segment_nodes, graph, initializer_map):
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
                    # For unknown intermediate tensors, we need to infer reasonable shapes
                    # Look at the node that would consume this input to guess the shape
                    inferred_shape = NestedFFTSlicer._infer_input_shape(inp, segment_nodes)
                    t = onnx.helper.make_tensor_value_info(
                        inp,
                        onnx.TensorProto.FLOAT,
                        inferred_shape
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
                    inferred_shape = NestedFFTSlicer._infer_output_shape(out, segment_nodes)
                    t = onnx.helper.make_tensor_value_info(
                        out,
                        onnx.TensorProto.FLOAT,
                        inferred_shape
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
                elif node.op_type in ["Relu", "BatchNormalization", "DFT", "Mul"]:
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
                elif node.op_type in ["Relu", "BatchNormalization", "DFT", "Mul"]:
                    # These preserve input shape
                    return ["batch_size", None, None, None]
                elif node.op_type == "Reshape":
                    # Reshape output depends on the target shape
                    return ["batch_size", None]

        # Default fallback
        return ["batch_size", None]

    def slice(self, slice_points: List[int], model_metadata, output_path=None):
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
        (graph, node_map, node_type_index_map, initializer_map, value_info_map,
         index_to_node_name, index_to_segment_name, output_path) = self._slice_setup(model_metadata, output_path)

        # Store paths to sliced models
        slice_paths = []

        # Process each segment
        for i in range(len(slice_points) - 1):
            segment_idx = i
            start_idx = slice_points[i]
            end_idx = slice_points[i + 1]

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
                segment_nodes, graph, initializer_map)

            # Save the segment model directly in the output directory with flattened naming
            file_path = os.path.join(output_path, f"segment_{segment_idx}.onnx")

            input_names = Utils.filter_inputs(segment_inputs, graph)
            output_names = [output_info.name for output_info in segment_outputs]

            # Use extract_model to create the segment
            try:
                logger.info(f"Extracting nested segment {segment_idx}: {input_names} -> {output_names}")
                # Extract the model directly to final path
                extract_model(
                    input_path=self.onnx_path,
                    output_path=file_path,
                    input_names=input_names,
                    output_names=output_names
                )

                slice_paths.append(file_path)

            except Exception as e:
                try:
                    logger.info(f"Error extracting nested segment, trying to create it instead {segment_idx}: {e}")
                    segment_graph = onnx.helper.make_graph(
                        segment_nodes,
                        f"nested_segment_{segment_idx}_graph",
                        segment_inputs,
                        segment_outputs,
                        segment_initializers
                    )

                    # Create a model from the graph
                    segment_model = onnx.helper.make_model(segment_graph)

                    onnx.save(segment_model, file_path)
                    slice_paths.append(file_path)

                except Exception as e:
                    logger.error(f"Error creating nested segment {segment_idx}: {e}")
                    continue

        return self.slice_post_process(slice_paths, self.analysis)

    @staticmethod
    def slice_post_process(slices_paths, model_metadata):
        abs_paths = []
        for path in slices_paths:
            abs_path = os.path.abspath(path)
            abs_paths.append(abs_path)
            try:
                model = onnx.load(path)
                # if OnnxUtils.has_fused_operations(model):
                #     model = OnnxUtils.unfuse_operations(model)

                onnx.checker.check_model(model)
                # model = OnnxUtils.optimize_model(abs_path, model)
                # model = OnnxUtils.add_shape_inference(model, model_metadata, path)
                onnx.save(model, path)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue

        return abs_paths

    def slice_model(self, output_path=None):
        """
        Run the complete workflow: determine slice points and slice.

        Args:
            output_path: The path to save the slices to.

        Returns:
            Dict[str, Any]: Metadata about the sliced model
        """

        # Step 1: Determine slice points
        slice_points = self.determine_slice_points(self.analysis)

        # Step 2: Slice the model
        slices_paths = self.slice(slice_points, self.analysis, output_path)

        # Step 3: generate slices metadata
        self.onnx_analyzer.generate_slices_metadata(self.analysis, slice_points, slices_paths, output_path)

        return slices_paths


def main():
    """Main function to run nested slicing on all FFT-decomposed segments."""
    # Use the FFT_cov_dft directory which has properly decomposed models
    input_dir = "./src/models/resnet/FFT_cov_dft/slices"
    output_base = "./src/models/resnet/flattened_nested_slices"
    final_output_dir = "./src/models/resnet/flattened_segments"
    
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    print("🔪 Nested FFT Slicer - Creating flattened granular slices from FFT-decomposed models")
    print("=" * 70)
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Intermediate output: {output_base}")
    print(f"📁 Final flattened output: {final_output_dir}")

    # Find all segment directories
    segment_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith('segment_')])
    
    for segment_dir in segment_dirs:
        segment_num = segment_dir.split('_')[1]
        input_path = os.path.join(input_dir, segment_dir, f"{segment_dir}.onnx")
        
        if not os.path.exists(input_path):
            print(f"   ⚠️  Skipping {segment_dir}: {input_path} not found")
            continue
            
        # Create output directory for this segment's nested slices
        nested_output_dir = os.path.join(output_base, f"segment_{segment_num}_nested")
        os.makedirs(nested_output_dir, exist_ok=True)
        
        print(f"\n📁 Processing {segment_dir} -> segment_{segment_num}_nested/")
        
        try:
            nested_slicer = NestedFFTSlicer(input_path, save_path=nested_output_dir)
            slice_paths = nested_slicer.slice_model(output_path=nested_output_dir)
            
            print(f"   ✅ Created {len(slice_paths)} nested segments")
            for path in slice_paths:
                print(f"      📄 {os.path.basename(path)}")
                
        except Exception as e:
            print(f"   ❌ Error processing {segment_dir}: {e}")
            continue

    # Now flatten all segments into the final directory with proper naming
    print(f"\n🔄 Flattening segments into final directory...")
    
    for segment_dir in segment_dirs:
        segment_num = segment_dir.split('_')[1]
        nested_dir = os.path.join(output_base, f"segment_{segment_num}_nested")
        
        if not os.path.exists(nested_dir):
            continue
            
        # Find all segment files in the nested directory
        segment_files = [f for f in os.listdir(nested_dir) if f.endswith('.onnx') and f.startswith('segment_')]
        segment_files.sort()
        
        for segment_file in segment_files:
            nested_segment_num = segment_file.split('_')[1].split('.')[0]
            source_path = os.path.join(nested_dir, segment_file)
            target_filename = f"segment_{segment_num}_{nested_segment_num}.onnx"
            target_path = os.path.join(final_output_dir, target_filename)
            
            try:
                shutil.copy2(source_path, target_path)
                print(f"   📋 {segment_file} -> {target_filename}")
            except Exception as e:
                print(f"   ❌ Error copying {segment_file}: {e}")

    print(f"\n🎉 Nested slicing and flattening complete!")
    print(f"📁 Check intermediate output in: {output_base}")
    print(f"📁 Check final flattened output in: {final_output_dir}")


if __name__ == "__main__":
    main()
