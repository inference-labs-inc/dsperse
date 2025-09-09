import os.path
import onnx
import logging
from src.analyzers.onnx_analyzer import OnnxAnalyzer
from typing import List, Dict, Tuple
from src.utils.utils import Utils
from onnx.utils import extract_model
from onnx import shape_inference

# Configure logger
logger = logging.getLogger(__name__)


class OnnxSlicer:
    # Control-input specification: op_type -> {input_index: (symbolic_shape, label)}
    CONTROL_INPUT_SPECS = {
        "Unsqueeze": {1: (["M"], "axes")},
        "Reshape": {1: (["M"], "shape")},
        "Slice": {1: (["M"], "starts"), 2: (["M"], "ends"), 3: (["M"], "axes"), 4: (["M"], "steps")},
        "TopK": {1: ([], "k")},
        "Gather": {1: (["M"], "indices")},
        "ReduceMean": {1: (["M"], "axes")},
        "ReduceSum": {1: (["M"], "axes")},
        "Pad": {1: (["M"], "pads")},
        "Expand": {1: (["M"], "shape")},
    }
    def __init__(self, onnx_path, save_path=None):
        self.onnx_path = onnx_path
        self.onnx_model = onnx.load(onnx_path)
        self.model_metadata = None
        self.slice_points = None

        self.onnx_analyzer = OnnxAnalyzer(self.onnx_path)
        self.analysis = self.onnx_analyzer.analyze(save_path=save_path)

    def determine_slice_points(self, model_metadata) -> List[int]:
        """
        Determine the slice points for the model based on nodes with parameter_details in the model_metadata.

        Args:
            model_metadata: The model analysis metadata containing node information.

        Returns:
            List[int]: List of indices representing nodes with parameter details
        """
        # Find nodes with parameter_details in model_metadata
        slice_points = []
        for node_name, node_info in model_metadata["nodes"].items():
            if node_info.get("parameter_details") and node_info["parameter_details"]:
                slice_points.append(node_info["index"])

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
                    # For unknown intermediate tensors, we need to infer reasonable shapes and correct dtypes
                    # Detect if this input is a control tensor (e.g., axes/shape/indices) based on consumer op and input index
                    consumer = None
                    consumer_input_idx = None
                    for n in segment_nodes:
                        for idx_i, name_i in enumerate(n.input):
                            if name_i == inp:
                                consumer = n
                                consumer_input_idx = idx_i
                                break
                        if consumer is not None:
                            break
                    # Map control inputs requiring INT64 dtype
                    int64_controls = OnnxSlicer.CONTROL_INPUT_SPECS
                    if consumer is not None and consumer.op_type in int64_controls and consumer_input_idx in int64_controls[consumer.op_type]:
                        shape_sym, _ = int64_controls[consumer.op_type][consumer_input_idx]
                        t = onnx.helper.make_tensor_value_info(
                            inp,
                            onnx.TensorProto.INT64,
                            shape_sym
                        )
                        segment_inputs.append(t)
                    else:
                        # Data tensor: infer shape and use FLOAT as a safe default
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
        Use only symbolic strings (dim_param) and never None.
        """
        for node in segment_nodes:
            if input_name in node.input:
                if node.op_type == "Conv":
                    # Conv expects 4D input: [N, C, H, W]
                    return ["N", "C", "H", "W"]
                elif node.op_type == "Gemm":
                    # Gemm expects 2D input: [N, F]
                    return ["N", "F"]
                elif node.op_type in ["Relu", "BatchNormalization"]:
                    # These preserve input shape, assume 4D
                    return ["N", "C", "H", "W"]

        # Default fallback for unknown cases
        return ["N", "D"]

    @staticmethod
    def _infer_output_shape(output_name, segment_nodes):
        """
        Infer a reasonable shape for an output tensor based on the node that produces it.
        Use only symbolic strings (dim_param) and never None.
        """
        for node in segment_nodes:
            if output_name in node.output:
                if node.op_type == "Conv":
                    # Conv output is 4D: [N, C, H, W]
                    return ["N", "C", "H", "W"]
                elif node.op_type == "Gemm":
                    # Gemm output is 2D: [N, F]
                    return ["N", "F"]
                elif node.op_type in ["Relu", "BatchNormalization"]:
                    # These preserve input shape
                    return ["N", "C", "H", "W"]
                elif node.op_type == "Reshape":
                    # Reshape output depends on the target shape; keep generic 2D
                    return ["N", "F"]

        # Default fallback
        return ["N", "D"]

    def _index_graph(self, graph):
        name_to_producer = {}
        for node in graph.node:
            for out in node.output:
                name_to_producer[out] = node
        init_map = {init.name: init for init in graph.initializer}
        value_info_map = {vi.name: vi for vi in list(graph.input) + list(graph.value_info) + list(graph.output)}
        return name_to_producer, init_map, value_info_map

    def _collect_control_tensors(self, graph, segment_nodes) -> List[onnx.TensorProto]:
        name_to_producer, init_map, _value_info_map = self._index_graph(graph)
        to_inject: Dict[str, onnx.TensorProto] = {}
        for n in segment_nodes:
            spec = self.CONTROL_INPUT_SPECS.get(n.op_type)
            if not spec:
                continue
            for idx_i in spec.keys():
                if idx_i >= len(n.input):
                    continue
                inp_name = n.input[idx_i]
                if not inp_name or inp_name in to_inject:
                    continue
                if inp_name in init_map:
                    t = onnx.TensorProto(); t.CopyFrom(init_map[inp_name])
                    to_inject[inp_name] = t
                    continue
                prod = name_to_producer.get(inp_name)
                if prod is not None and prod.op_type == "Constant":
                    for attr in prod.attribute:
                        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                            const_t = onnx.TensorProto(); const_t.CopyFrom(attr.t)
                            if not const_t.name:
                                const_t.name = prod.output[0] if prod.output and prod.output[0] else inp_name
                            to_inject[inp_name] = const_t
                            break
        return list(to_inject.values())

    def prepare_extract_model(self, graph, segment_nodes, segment_inputs) -> Tuple[List[str], List[onnx.TensorProto]]:
        """
        Prepare input names for onnx.utils.extract_model using true boundary inputs:
        - Exclude Constant-produced tensors from external inputs (capture their TensorProtos).
        - Exclude initializers (weights/biases/stats).
        - Keep boundary names even if untyped; the extractor retry will add temporary ValueInfos.
        - Fallback to the first model input only if nothing remains.
        """
        name_to_producer, init_map, _value_info_map = self._index_graph(graph)
        initializer_names = set(init_map.keys())

        prepared_inputs: List[str] = []
        seen: set = set()
        const_tensors: List[onnx.TensorProto] = []

        for vi in segment_inputs:
            name = getattr(vi, 'name', None)
            if not name or name in seen:
                continue
            seen.add(name)

            # Skip parameters (weights/biases/stats)
            if name in initializer_names:
                continue

            producer = name_to_producer.get(name)
            if producer is not None and producer.op_type == "Constant":
                # Capture Constant tensor for later injection; do not expose as external input
                for attr in producer.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        t = onnx.TensorProto(); t.CopyFrom(attr.t)
                        if not t.name:
                            t.name = producer.output[0] if (producer.output and producer.output[0]) else name
                        const_tensors.append(t)
                        break
                continue

            # Keep this boundary name even if untyped; we will type it in the temporary pre-extract model
            prepared_inputs.append(name)

        if not prepared_inputs and graph.input:
            prepared_inputs = [graph.input[0].name]

        # Deduplicate while preserving order
        deduped: List[str] = []
        seen2 = set()
        for n in prepared_inputs:
            if n not in seen2:
                seen2.add(n)
                deduped.append(n)

        return deduped, const_tensors

    def _prepare_typed_model_for_extraction(self, base_model_path: str, save_dir: str, segment_idx: int,
                                            segment_nodes, input_names: List[str], output_names: List[str]) -> str:
        """
        Ensure the model used for extraction has type/shape info for this segment's boundary tensors
        (inputs/outputs) to avoid extractor errors about missing 'shape'. Writes a temporary
        model file in save_dir when needed and returns its path; otherwise returns base_model_path.
        """
        model = onnx.load(base_model_path)
        graph = model.graph
        typed_names = {vi.name for vi in list(graph.input) + list(graph.value_info) + list(graph.output)}

        added_any = False
        # Ensure output value_infos
        for out_name in (output_names or []):
            if out_name and out_name not in typed_names:
                shape = self._infer_output_shape(out_name, segment_nodes)
                vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, shape)
                graph.value_info.extend([vi])
                typed_names.add(out_name)
                added_any = True
        # Ensure input value_infos (only for names that are not already typed and not true graph inputs)
        graph_input_names = {vi.name for vi in graph.input}
        for in_name in (input_names or []):
            if in_name and in_name not in typed_names and in_name not in graph_input_names:
                shape = self._infer_input_shape(in_name, segment_nodes)
                vi = onnx.helper.make_tensor_value_info(in_name, onnx.TensorProto.FLOAT, shape)
                graph.value_info.extend([vi])
                typed_names.add(in_name)
                added_any = True

        if not added_any:
            return base_model_path
        tmp_path = os.path.join(save_dir, f"__pre_extract_{segment_idx}.onnx")
        onnx.save(model, tmp_path)
        return tmp_path


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
                segment_nodes, graph, initializer_map)

            # Save the segment model
            save_path = os.path.join(output_path, f"segment_{segment_idx}")
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"segment_{segment_idx}.onnx")

            filtered_inputs = Utils.filter_inputs(segment_inputs, graph)
            unfiltered_inputs = Utils.get_unfiltered_inputs(segment_inputs)
            output_names = [output_info.name for output_info in segment_outputs]

            # Prepare inputs: keep constants internal and collect their tensors
            prepared_inputs, const_inits_boundary = self.prepare_extract_model(graph, segment_nodes, segment_inputs)
            control_inits = self._collect_control_tensors(graph, segment_nodes)
            # Deduplicate by tensor name
            seen_names = set()
            const_inits = []
            for t in list(const_inits_boundary) + list(control_inits):
                if getattr(t, 'name', None) and t.name not in seen_names:
                    seen_names.add(t.name)
                    const_inits.append(t)

            # Use extract_model to create the segment
            try:
                # Build a human-friendly display of inputs similar to old output
                display_inputs = []
                # Include initializers consumed by the segment for readability
                init_names_set = {init.name for init in graph.initializer}
                boundary_inputs = [getattr(vi, 'name', None) for vi in segment_inputs if getattr(vi, 'name', None)]
                for n in boundary_inputs:
                    if n in init_names_set or n in boundary_inputs:
                        display_inputs.append(n)
                # Fall back to unfiltered list for display if empty
                if not display_inputs:
                    display_inputs = unfiltered_inputs
                logger.info(f"Extracting segment {segment_idx}: {display_inputs} -> {output_names}")
                print(f"Extracting segment {segment_idx}: {display_inputs} -> {output_names}")
                # Extract the model directly to final path with strictly typed inputs
                try:
                    extract_model(
                        input_path=self.onnx_path,
                        output_path=file_path,
                        input_names=(prepared_inputs or filtered_inputs),
                        output_names=output_names
                    )
                except Exception as ex_first:
                    # Retry using a temporary model with typed boundary tensors
                    prepped_model_path = self._prepare_typed_model_for_extraction(
                        base_model_path=self.onnx_path,
                        save_dir=save_path,
                        segment_idx=segment_idx,
                        segment_nodes=segment_nodes,
                        input_names=(prepared_inputs or filtered_inputs),
                        output_names=output_names,
                    )
                    extract_model(
                        input_path=prepped_model_path,
                        output_path=file_path,
                        input_names=(prepared_inputs or filtered_inputs),
                        output_names=output_names
                    )
                # Post-fix: normalize control attributes and inject constants (collision-safe)
                try:
                    sub = onnx.load(file_path)
                    init_names = {i.name for i in sub.graph.initializer}
                    produced_names = set()
                    for n in sub.graph.node:
                        for out in n.output:
                            if out:
                                produced_names.add(out)
                    # Convert Unsqueeze 'axes' attribute to second input if missing
                    new_inits = []
                    for n in sub.graph.node:
                        if n.op_type == "Unsqueeze" and len(n.input) < 2:
                            axes_vals = None
                            for a in list(n.attribute):
                                if a.name == "axes" and a.ints:
                                    axes_vals = list(a.ints)
                                    n.attribute.remove(a)
                                    break
                            if axes_vals is not None:
                                axes_name = f"{n.name or n.output[0]}__axes"
                                if axes_name not in init_names and axes_name not in produced_names:
                                    t = onnx.helper.make_tensor(
                                        name=axes_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(axes_vals)],
                                        vals=axes_vals,
                                    )
                                    new_inits.append(t)
                                    n.input.extend([axes_name])
                    if new_inits:
                        sub.graph.initializer.extend(new_inits)
                        # Remove any same-named graph inputs
                        for t in new_inits:
                            for idx, gi in enumerate(list(sub.graph.input)):
                                if gi.name == t.name:
                                    del sub.graph.input[idx]
                                    break
                    # Inject captured const/control tensors
                    def remove_input_by_name(g, n):
                        for idx, gi in enumerate(list(g.input)):
                            if gi.name == n:
                                del g.input[idx]
                                return
                    for t in const_inits:
                        if not getattr(t, 'name', None):
                            continue
                        # Skip if already present or would violate SSA by colliding with produced value names
                        if t.name in init_names or t.name in produced_names:
                            continue
                        sub.graph.initializer.extend([t])
                        remove_input_by_name(sub.graph, t.name)
                    onnx.checker.check_model(sub)
                    onnx.save(sub, file_path)
                except Exception as post_e:
                    logger.info(f"Post-injection check skipped/failed: {post_e}")
                slice_paths.append(file_path)

            except Exception as e:
                try:
                    logger.info(f"Error extracting segment, trying to create it instead {segment_idx}: {e}")
                    print(f"Error extracting segment, trying to create it instead {segment_idx}: {e}")
                    # Before creating graph, inject control/constants as initializers and remove them from inputs
                    # Compute names produced by nodes in this segment to avoid SSA collisions
                    produced_by_segment = set()
                    for n in segment_nodes:
                        for out in n.output:
                            if out:
                                produced_by_segment.add(out)
                    existing_init_names = {init.name for init in segment_initializers}
                    # Keep only safe constants that do not collide with produced outputs or existing initializers
                    safe_const_inits = [t for t in const_inits if getattr(t, 'name', None) and t.name not in produced_by_segment and t.name not in existing_init_names]
                    init_name_set = {t.name for t in safe_const_inits}
                    # Extend initializers with safe const/control
                    segment_initializers = list(segment_initializers) + safe_const_inits
                    # Filter out any inputs that collide with injected initializers
                    segment_inputs = [vi for vi in segment_inputs if getattr(vi, 'name', None) not in init_name_set]

                    segment_graph = onnx.helper.make_graph(
                        segment_nodes,
                        f"segment_{segment_idx}_graph",
                        segment_inputs,
                        segment_outputs,
                        segment_initializers
                    )

                    # Create a model from the graph
                    segment_model = onnx.helper.make_model(segment_graph)

                    # Normalize Unsqueeze in manual path: convert 'axes' attribute to second INT64 input if missing
                    try:
                        g = segment_model.graph
                        init_names_manual = {i.name for i in g.initializer}
                        produced_names_manual = set()
                        for node_m in g.node:
                            for out_m in node_m.output:
                                if out_m:
                                    produced_names_manual.add(out_m)
                        new_axes_inits = []
                        for node_m in g.node:
                            if node_m.op_type == "Unsqueeze" and len(node_m.input) < 2:
                                axes_vals = None
                                for a in list(node_m.attribute):
                                    if a.name == "axes" and a.ints:
                                        axes_vals = list(a.ints)
                                        node_m.attribute.remove(a)
                                        break
                                if axes_vals is not None:
                                    base_name = f"{node_m.name or (node_m.output[0] if node_m.output else 'unsq')}__axes"
                                    axes_name = base_name
                                    # ensure uniqueness to avoid SSA collisions
                                    suffix = 1
                                    while axes_name in init_names_manual or axes_name in produced_names_manual:
                                        axes_name = f"{base_name}_{suffix}"
                                        suffix += 1
                                    t_axes = onnx.helper.make_tensor(
                                        name=axes_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(axes_vals)],
                                        vals=axes_vals,
                                    )
                                    new_axes_inits.append(t_axes)
                                    node_m.input.extend([axes_name])
                        if new_axes_inits:
                            g.initializer.extend(new_axes_inits)
                            # Remove any same-named graph inputs
                            for t_ax in new_axes_inits:
                                for idx_gi, gi in enumerate(list(g.input)):
                                    if gi.name == t_ax.name:
                                        del g.input[idx_gi]
                                        break
                    except Exception:
                        pass

                    # Validate and save
                    try:
                        onnx.checker.check_model(segment_model)
                    except Exception:
                        pass
                    onnx.save(segment_model, file_path)
                    slice_paths.append(file_path)

                except Exception as e:
                    logger.error(f"Error creating segment {segment_idx}: {e}")
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

                # Infer concrete shapes so saved slices carry numeric dims where possible
                try:
                    model = shape_inference.infer_shapes(model)
                except Exception as inf_err:
                    logger.info(f"Shape inference skipped for {path}: {inf_err}")

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

if __name__ == "__main__":

    model_choice = 5 # Change this to test different models

    base_paths = {
        1: "../../models/doom",
        2: "../../models/net",
        3: "../../models/resnet",
        4: "../../models/yolov3",
        5: "../../models/age",
    }
    abs_path = os.path.abspath(base_paths[model_choice])
    model_dir = os.path.join(abs_path, "model.onnx")
    output_dir = os.path.join(abs_path, "slices")
    onnx_slicer = OnnxSlicer(model_dir, save_path=base_paths[model_choice])
    onnx_slicer.slice_model(output_path=output_dir)
