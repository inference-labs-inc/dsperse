import os.path
import json
import onnx
from utils.onnx_analyzer import OnnxAnalyzer
from pathlib import Path
import onnxruntime_extensions as ortx
import onnxruntime as ort
from onnxruntime.tools import optimize_onnx_model, symbolic_shape_infer
from typing import List, Dict
import numpy as np
import functools

# Add GraphSurgeon import
try:
    import onnx_graphsurgeon as gs
    GRAPHSURGEON_AVAILABLE = True
except ImportError:
    print("Warning: GraphSurgeon not available. Install with: pip install onnx-graphsurgeon")
    GRAPHSURGEON_AVAILABLE = False

import polygraphy.backend.onnx.loader as pg_loader


def metadata_driven_onnx_processing(func):
    """
    Decorator that provides metadata-driven ONNX model processing using GraphSurgeon and Polygraphy.
    This handles:
    1. Unfusing FusedConv/FusedGemm operations 
    2. Shape propagation using model metadata
    3. Polygraphy validation
    4. Batch size normalization
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the slice paths from the function
        slice_paths = func(*args, **kwargs)
        
        # Get model metadata for shape propagation
        if args and hasattr(args[0], 'model_metadata'):
            model_metadata = args[0].model_metadata
        else:
            # Try to load metadata from expected location
            slicer_instance = args[0] if args else None
            if slicer_instance and hasattr(slicer_instance, 'onnx_path'):
                metadata_path = os.path.join(os.path.dirname(slicer_instance.onnx_path), "onnx_analysis", "model_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        model_metadata = json.load(f)
                else:
                    print("Warning: Model metadata not found for shape propagation")
                    model_metadata = None
            else:
                model_metadata = None
        
        # Process each slice with metadata-driven approach
        if slice_paths and model_metadata:
            _process_slices_with_metadata(slice_paths, model_metadata)
        elif slice_paths:
            # Fallback to basic processing without metadata
            _process_slices_basic(slice_paths)
            
        return slice_paths
    return wrapper


def _process_slices_with_metadata(slice_paths: List[str], model_metadata: Dict):
    """Process ONNX slices using metadata-driven shape propagation."""
    print("Processing slices with metadata-driven approach...")
    
    for path in slice_paths:
        segment_idx = _extract_segment_index(path)
        print(f"\nProcessing segment {segment_idx}: {path}")
        
        try:
            # Load the model
            model = onnx.load(path)
            print("✓ Model loaded successfully")

            # Step 1: Unfuse operations using GraphSurgeon
            model = _unfuse_operations_gs(model)
            print("✓ Operations unfused")

            # Step 2: Apply metadata-driven shape propagation
            model = _apply_metadata_shapes(model, model_metadata, segment_idx)
            print("✓ Metadata-driven shapes applied")

            # Step 3: Ensure batch_size consistency
            model = _normalize_batch_dimensions(model)
            print("✓ Batch dimensions normalized")

            # Step 4: Final shape inference and validation
            model = _finalize_shapes(model)
            print("✓ Final shape inference completed")

            # Step 5: Polygraphy validation
            _validate_with_polygraphy(model, path)
            print("✓ Polygraphy validation completed")

            # Step 6: Save processed model
            onnx.save(model, path)
            print(f"✓ Model saved to {path}")

            # Step 7: Print diagnostic information
            _print_shape_diagnostics(model, segment_idx)

        except Exception as e:
            print(f"✗ Error processing {path}: {e}")
            continue


def _process_slices_basic(slice_paths: List[str]):
    """Fallback processing without metadata."""
    print("Processing slices with basic approach (no metadata)...")
    
    for path in slice_paths:
        try:
            model = onnx.load(path)
            model = _unfuse_operations_gs(model)
            model = _normalize_batch_dimensions(model)
            model = _finalize_shapes(model)
            _validate_with_polygraphy(model, path)
            onnx.save(model, path)
            print(f"✓ Basic processing completed for {path}")
        except Exception as e:
            print(f"✗ Error in basic processing {path}: {e}")


def _extract_segment_index(path: str) -> int:
    """Extract segment index from file path."""
    import re
    match = re.search(r'segment_(\d+)', path)
    return int(match.group(1)) if match else -1


def _unfuse_operations_gs(model):
    """Unfuse FusedConv and FusedGemm operations using GraphSurgeon."""
    if not GRAPHSURGEON_AVAILABLE:
        print("Warning: GraphSurgeon not available, skipping unfusing")
        return model
        
    graph = gs.import_onnx(model)
    
    for node in graph.nodes:
        if node.op in ["FusedConv", "FusedGemm"]:
            print(f"  Unfusing {node.op} node: {node.name}")
            # Convert to standard operations
            if node.op == "FusedConv":
                node.op = "Conv"
            elif node.op == "FusedGemm":
                node.op = "Gemm"
            # Remove fusion-related attributes
            node.attrs = {k: v for k, v in node.attrs.items() if "fusion" not in k.lower()}
    
    return gs.export_onnx(graph)


def _apply_metadata_shapes(model, model_metadata: Dict, segment_idx: int):
    """Apply shape information from metadata to the model."""
    if not GRAPHSURGEON_AVAILABLE:
        print("Warning: GraphSurgeon not available, skipping metadata shape application")
        return model
    
    # Get nodes in this segment from metadata
    segment_nodes = _get_segment_nodes_from_metadata(model_metadata, segment_idx)
    
    if not segment_nodes:
        print(f"  No metadata found for segment {segment_idx}")
        return model
    
    graph = gs.import_onnx(model)
    
    # Get original model shapes for reference
    original_shapes = _get_original_model_shapes(model_metadata)
    
    # Apply proper shapes to all tensors based on original model
    for tensor_name, tensor in graph.tensors().items():
        if isinstance(tensor, gs.Variable):
            # Ensure dtype is set
            if not hasattr(tensor, 'dtype') or tensor.dtype is None:
                tensor.dtype = np.float32
            
            # Apply shape from original model if available
            if tensor_name in original_shapes:
                tensor.shape = original_shapes[tensor_name]
                print(f"  Applied shape {original_shapes[tensor_name]} to {tensor_name}")
            elif tensor_name == "input":
                # Apply model input shape to main input
                model_input_shape = model_metadata.get("input_shape", [])
                if model_input_shape and len(model_input_shape) > 0:
                    first_input_shape = model_input_shape[0]
                    if first_input_shape and not str(first_input_shape[0]).startswith('batch'):
                        first_input_shape = ["batch_size"] + first_input_shape[1:]
                    tensor.shape = first_input_shape
                    print(f"  Applied input shape {first_input_shape} to {tensor_name}")
    
    return gs.export_onnx(graph)


def _get_original_model_shapes(model_metadata: Dict) -> Dict[str, List]:
    """Extract shape information from original model metadata."""
    original_shapes = {}
    
    # Try to load and process the original model to get intermediate shapes
    try:
        original_model_path = model_metadata.get("original_model")
        if original_model_path:
            original_model = onnx.load(original_model_path)
            from onnx import shape_inference
            original_model = shape_inference.infer_shapes(original_model)
            
            # Extract all intermediate tensor shapes
            for value_info in original_model.graph.value_info:
                shape = []
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
                original_shapes[value_info.name] = shape
            
            # Also get output shapes
            for output_info in original_model.graph.output:
                shape = []
                for dim in output_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
                original_shapes[output_info.name] = shape
                
            print(f"  Loaded {len(original_shapes)} tensor shapes from original model")
            
    except Exception as e:
        print(f"  Warning: Could not load original model shapes: {e}")
    
    return original_shapes


def _get_segment_nodes_from_metadata(model_metadata: Dict, segment_idx: int) -> Dict:
    """Extract nodes belonging to a specific segment from metadata."""
    segment_nodes = {}
    all_nodes = model_metadata.get("nodes", {})
    
    # We need to determine which nodes belong to this segment
    # This requires knowing the slice points
    slice_points = []
    for node_name, node_info in all_nodes.items():
        if node_info.get("parameter_details"):
            slice_points.append(node_info["index"])
    
    slice_points.sort()
    # Add end point
    max_index = max(node_info["index"] for node_info in all_nodes.values())
    if max_index + 1 not in slice_points:
        slice_points.append(max_index + 1)
    
    # Determine start and end indices for this segment
    if segment_idx < 0 or segment_idx >= len(slice_points):
        return segment_nodes
    
    start_idx = slice_points[segment_idx] if segment_idx > 0 else 0
    end_idx = slice_points[segment_idx + 1] if segment_idx + 1 < len(slice_points) else max_index + 1
    
    # Collect nodes in this range
    for node_name, node_info in all_nodes.items():
        node_index = node_info["index"]
        if start_idx <= node_index < end_idx:
            segment_nodes[node_name] = node_info
    
    print(f"  Found {len(segment_nodes)} nodes in segment {segment_idx} metadata")
    return segment_nodes


def _apply_feature_shapes(graph, node_name: str, in_features: int, out_features: int, node_type: str):
    """Apply feature-based shapes to tensors in the graph."""
    # Find tensors associated with this node and apply appropriate shapes
    for tensor_name, tensor in graph.tensors().items():
        if isinstance(tensor, gs.Variable) and hasattr(tensor, 'shape'):
            # Apply shapes based on node type and position
            if node_type == "Conv":
                # For Conv layers, maintain 4D shapes
                if not tensor.shape or len(tensor.shape) != 4:
                    if "weight" in tensor_name.lower() or any(node_name in tensor_name for node_name in [node_name]):
                        tensor.shape = ["batch_size", in_features, None, None]
                        # Ensure dtype is set
                        if not hasattr(tensor, 'dtype') or tensor.dtype is None:
                            tensor.dtype = np.float32
                        print(f"    Applied Conv input shape to {tensor_name}")
            elif node_type in ["Gemm", "MatMul"]:
                # For Gemm/MatMul, use 2D shapes
                if not tensor.shape or len(tensor.shape) != 2:
                    if any(node_name in tensor_name for node_name in [node_name]):
                        tensor.shape = ["batch_size", in_features]
                        # Ensure dtype is set
                        if not hasattr(tensor, 'dtype') or tensor.dtype is None:
                            tensor.dtype = np.float32
                        print(f"    Applied Gemm input shape to {tensor_name}")


def _normalize_batch_dimensions(model):
    """Ensure all tensors have batch_size as the first dimension."""
    if not GRAPHSURGEON_AVAILABLE:
        return model
        
    graph = gs.import_onnx(model)
    
    for tensor_name, tensor in graph.tensors().items():
        if isinstance(tensor, gs.Variable) and hasattr(tensor, 'shape') and tensor.shape:
            if len(tensor.shape) > 0:
                # Ensure first dimension is batch_size
                if not str(tensor.shape[0]).startswith('batch'):
                    new_shape = ["batch_size"] + list(tensor.shape[1:])
                    tensor.shape = new_shape
                    # Ensure dtype is preserved/set
                    if not hasattr(tensor, 'dtype') or tensor.dtype is None:
                        tensor.dtype = np.float32
                    print(f"  Normalized batch dimension for {tensor_name}: {new_shape}")
    
    return gs.export_onnx(graph)


def _finalize_shapes(model):
    """Final shape inference using ONNX built-in tools."""
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
        print("  ✓ ONNX shape inference successful")
    except Exception as e:
        print(f"  ⚠ ONNX shape inference failed: {e}")
    
    return model


def _validate_with_polygraphy(model, path: str):
    """Validate model using Polygraphy."""
    try:
        import subprocess
        result = subprocess.run(['polygraphy', 'inspect', 'model', path], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("  ✓ Polygraphy validation passed")
        else:
            print(f"  ⚠ Polygraphy warning: {result.stderr[:100]}")
    except Exception as e:
        print(f"  ⚠ Polygraphy validation failed: {e}")


def _print_shape_diagnostics(model, segment_idx: int):
    """Print diagnostic information about shapes in the model."""
    print(f"  Shape diagnostics for segment {segment_idx}:")
    
    # Print input shapes
    for input_info in model.graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(f"    Input {input_info.name}: {shape}")
    
    # Print output shapes
    for output_info in model.graph.output:
        shape = []
        for dim in output_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(f"    Output {output_info.name}: {shape}")
    
    # Print operations
    ops = [node.op_type for node in model.graph.node]
    print(f"    Operations: {', '.join(set(ops))}")


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
                    # For unknown intermediate tensors, we need to infer reasonable shapes
                    # Look at the node that would consume this input to guess the shape
                    inferred_shape = OnnxSlicer._infer_input_shape(inp, segment_nodes)
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
                    inferred_shape = OnnxSlicer._infer_output_shape(out, segment_nodes)
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

    @metadata_driven_onnx_processing
    def slice(self, slice_points: List[int], model_metadata):
        """
        Slice the ONNX model based on the provided slice points.

        Args:
            slice_points: List of indices representing nodes with parameter details
            model_metadata: The model analysis metadata containing node information

        Returns:
            List[str]: Paths to the sliced model files
        """
        # Store metadata for use by decorator
        self.model_metadata = model_metadata
        
        # Error handling
        if not slice_points:
            raise ValueError("No slice points provided.")

        if not model_metadata or "nodes" not in model_metadata:
            raise ValueError("Invalid model metadata. Please run 'analyze()' first.")

        # Set up slicing environment
        (graph, node_map, node_type_index_map, initializer_map, value_info_map,
         index_to_node_name, index_to_segment_name, output_dir) = self._slice_setup(model_metadata)

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
            # Create a graph with the nodes
            segment_graph = onnx.helper.make_graph(
                segment_nodes,
                f"segment_{segment_idx}_graph",
                segment_inputs,
                segment_outputs,
                segment_initializers
            )

            # Create a model from the graph with opset version 18 for EZKL compatibility
            segment_model = onnx.helper.make_model(segment_graph, opset_imports=[onnx.helper.make_opsetid("", 18)])

            # Save the segment model
            save_path = os.path.join(output_dir, f"segment_{segment_idx}.onnx")
            onnx.save(segment_model, save_path)
            slice_paths.append(save_path)

        return slice_paths

    @staticmethod
    def unfuse_nodes_with_graphsurgeon(model):
        import onnx_graphsurgeon as gs
        graph = gs.import_onnx(model)
        new_nodes = []
        for node in graph.nodes:
            if node.op in ["FusedConv", "FusedGemm"]:
                print(f"Unfusing {node.op} node: {node.name}")
                # For demonstration, just convert op_type to Conv or Gemm and remove fusion attributes
                if node.op == "FusedConv":
                    node.op = "Conv"
                elif node.op == "FusedGemm":
                    node.op = "Gemm"
                # Remove all attributes related to fusion
                node.attrs = {k: v for k, v in node.attrs.items() if "fusion" not in k.lower()}
            new_nodes.append(node)
        graph.nodes = new_nodes
        return gs.export_onnx(graph)

    @staticmethod
    def polygraphy_check(path):
        try:
            # Use polygraphy CLI to check the model
            import subprocess
            result = subprocess.run(['polygraphy', 'inspect', 'model', path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[Polygraphy] Model check passed")
            else:
                print(f"[Polygraphy] Warning: {result.stderr}")
        except Exception as e:
            print(f"[Polygraphy] Warning: {e}")

    @staticmethod
    def ensure_proper_shapes(model):
        """Ensure all tensors have proper shape information using GraphSurgeon and Polygraphy."""
        import onnx_graphsurgeon as gs
        
        # Convert to GraphSurgeon format for better shape handling
        graph = gs.import_onnx(model)
        
        # Use ONNX's built-in shape inference first
        try:
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model)
            print("ONNX shape inference successful")
        except Exception as e:
            print(f"ONNX shape inference failed: {e}")
        
        # Convert back to GraphSurgeon to manipulate shapes
        graph = gs.import_onnx(model)
        
        # Ensure all Variable tensors have batch_size as the first dimension
        for tensor_name, tensor in graph.tensors().items():
            # Only modify Variable tensors, not Constants
            if isinstance(tensor, gs.Variable) and hasattr(tensor, 'shape') and tensor.shape:
                # Ensure first dimension is batch_size
                if len(tensor.shape) > 0 and not str(tensor.shape[0]).startswith('batch'):
                    # Create a new shape with batch_size as first dimension
                    new_shape = ["batch_size"] + list(tensor.shape[1:])
                    tensor.shape = new_shape
        
        # Convert back to ONNX
        model = gs.export_onnx(graph)
        
        return model

    @staticmethod
    def slice_post_process(slices_paths):
        for path in slices_paths:
            print(f"Processing {path}")
            print(f"Path: {path}")

            try:
                # Load the model
                model = onnx.load(path)
                print(f"Model loaded successfully")

                # Unfuse FusedConv and FusedGemm nodes using GraphSurgeon
                model = OnnxSlicer.unfuse_nodes_with_graphsurgeon(model)
                print(f"GraphSurgeon unfusing complete")

                # Ensure proper shapes including batch size
                model = OnnxSlicer.ensure_proper_shapes(model)
                print(f"Shape handling complete")

                # Check if model is valid after processing
                onnx.checker.check_model(model)
                print(f"Model validation passed")

                # Print all ops in the segment for verification
                print("Ops in this segment:")
                for node in model.graph.node:
                    print(f"  {node.op_type}")

                # Print shape information for inputs and outputs
                print("Input shapes:")
                for input_info in model.graph.input:
                    shape = []
                    for dim in input_info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value)
                    print(f"  {input_info.name}: {shape}")

                print("Output shapes:")
                for output_info in model.graph.output:
                    shape = []
                    for dim in output_info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value)
                    print(f"  {output_info.name}: {shape}")

                # Polygraphy check for unsupported ops
                OnnxSlicer.polygraphy_check(path)

                # Save the processed model (TensorProto export)
                onnx.save(model, path)
                print(f"Model saved successfully to {path}")

                # Additional verification step - try to create a session with the model
                try:
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    _ = ort.InferenceSession(path, session_options)
                    print(f"Model verification successful")
                except Exception as verify_error:
                    print(f"Model verification failed: {verify_error}, but model should still be usable")

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

    def slice_model(self, model_metadata=None):
        """
        Run the complete workflow: determine slice points and slice with metadata-driven processing.

        Args:
            model_metadata: The model analysis metadata. If None, it will be loaded from expected location.

        Returns:
            List[str]: Paths to the sliced model files
        """
        # Step 1: Set model metadata if provided
        if not model_metadata:
            # Check if model metadata exists in onnx_analysis directory
            metadata_path = os.path.join(os.path.dirname(self.onnx_path), "onnx_analysis", "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    model_metadata = json.load(f)
            else:
                raise ValueError("Model metadata not found. Please run 'analyze()' first.")

        # Step 2: Determine slice points
        slice_points = self.determine_slice_points(model_metadata)
        print(f"Determined slice points: {slice_points}")

        # Step 3: Slice the model (processing handled by metadata_driven_onnx_processing decorator)
        slice_paths = self.slice(slice_points, model_metadata)
        
        print(f"✓ Slicing completed. Generated {len(slice_paths)} segments.")
        
        return slice_paths

if __name__ == "__main__":

    model_choice = 1 # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet",
        4: "models/yolov3"
    }

    full_paths = {
        1: "src/models/doom",
        2: "src/models/net",
        3: "src/models/resnet",
        4: "src/models/yolov3"
    }

    model_dir = os.path.join(base_paths[model_choice], "model.onnx")
    full_model_dir = os.path.join(full_paths[model_choice], "model.onnx")
    onnx_analyzer = OnnxAnalyzer(model_path=model_dir)
    onnx_slicer = OnnxSlicer(full_model_dir)

    # Run the complete workflow: analyze, determine slice points, and slice
    model_analysis = onnx_analyzer.analyze()  # Produces the onnx_analysis/model_metadata.json file
    onnx_slicer.slice_model(model_analysis)  # this uses the model_metadata.json file to first get slice points, then it slices it.
