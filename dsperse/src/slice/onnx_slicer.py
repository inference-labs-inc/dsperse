import os
import os.path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import onnx
from onnx import shape_inference
import logging
from dsperse.src.analyzers.onnx_analyzer import OnnxAnalyzer
from dsperse.src.backends.jstprove import JSTPROVE_SUPPORTED_OPS
from typing import List, Dict, Tuple, Any
from dsperse.src.utils.utils import Utils
from dsperse.src.slice.autotiler import apply_tiling_to_slices, ELEMENTWISE_OPS
from dsperse.src.slice.tensor_graph import TensorGraph
from dsperse.src.utils.utils import save_onnx_model
from onnx.utils import extract_model
from onnxruntime.tools import symbolic_shape_infer

logger = logging.getLogger(__name__)


def _extract_single_slice(spec: Tuple[str, int, List[str], List[str], str]) -> Tuple[int, str] | None:
    onnx_path, segment_idx, input_names, output_names, file_path = spec
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        extract_model(
            input_path=onnx_path,
            output_path=file_path,
            input_names=input_names,
            output_names=output_names
        )
        return (segment_idx, file_path)
    except Exception as e:
        err_msg = str(e) if str(e) else f"{type(e).__name__}"
        logger.warning(f"extract_model failed for slice {segment_idx}: {err_msg} (inputs={input_names}, outputs={output_names})")
        return None


class OnnxSlicer:
    def __init__(self, onnx_path, save_path=None):
        self.onnx_path = onnx_path
        self.onnx_model = onnx.load(onnx_path)
        self.model_metadata = None
        self.slice_points = None
        self.traced_shapes = {}
        self.traced_dtypes = {}

        print("Running shape trace with dummy input...")
        self._trace_shapes()
        self._apply_traced_shapes(self.onnx_model, self.traced_shapes, self.traced_dtypes)

        traced_model_path = self.onnx_path.replace('.onnx', '_traced.onnx')
        onnx.save(self.onnx_model, traced_model_path)
        self.onnx_path = traced_model_path

        self.onnx_analyzer = OnnxAnalyzer(self.onnx_path)
        self.analysis = self.onnx_analyzer.analyze(save_path=save_path)

    def _trace_shapes(self):
        """Run model with dummy input to capture all intermediate tensor shapes."""
        import onnxruntime as ort
        import numpy as np
        import tempfile

        all_tensor_names = []
        for node in self.onnx_model.graph.node:
            for out in node.output:
                all_tensor_names.append(out)

        trace_model = onnx.ModelProto()
        trace_model.CopyFrom(self.onnx_model)
        existing_outputs = {o.name for o in trace_model.graph.output}
        for tensor_name in all_tensor_names:
            if tensor_name not in existing_outputs:
                new_output = trace_model.graph.output.add()
                new_output.name = tensor_name

        fd, trace_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            onnx.save(trace_model, trace_path)

            session = ort.InferenceSession(trace_path, providers=['CPUExecutionProvider'])

            inputs = {}
            for inp in session.get_inputs():
                shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
                inp_type = inp.type.lower()
                if 'float16' in inp_type:
                    dtype = np.float16
                elif 'double' in inp_type or 'float64' in inp_type:
                    dtype = np.float64
                elif 'int64' in inp_type:
                    dtype = np.int64
                elif 'int32' in inp_type:
                    dtype = np.int32
                elif 'bool' in inp_type:
                    dtype = np.bool_
                else:
                    dtype = np.float32
                if dtype == np.bool_:
                    inputs[inp.name] = np.random.randint(0, 2, shape).astype(dtype)
                else:
                    inputs[inp.name] = np.random.randn(*shape).astype(dtype)
                self.traced_shapes[inp.name] = shape

            outputs = session.run(None, inputs)
            output_names = [o.name for o in session.get_outputs()]
            for name, arr in zip(output_names, outputs):
                self.traced_shapes[name] = list(arr.shape)
                self.traced_dtypes[name] = arr.dtype
        finally:
            os.remove(trace_path)
        print(f"Shape trace complete: captured {len(self.traced_shapes)} tensor shapes")

    @staticmethod
    def _concretize_symbolic_dims(model: onnx.ModelProto, value: int = 1) -> onnx.ModelProto:
        def fix_vi(vi):
            ttype = vi.type.tensor_type
            if not ttype.HasField("shape"):
                return
            for dim in ttype.shape.dim:
                if dim.dim_param:
                    dim.dim_param = ""
                    dim.dim_value = value
                elif not dim.HasField("dim_value"):
                    dim.dim_value = value
        for vi in list(model.graph.input):
            fix_vi(vi)
        for vo in list(model.graph.output):
            fix_vi(vo)
        for vv in list(model.graph.value_info):
            fix_vi(vv)
        return model

    @staticmethod
    def optimize_jstprove_slices(slice_points: List[int], model_metadata: Dict) -> List[int]:
        updated_points = set(slice_points)
        nodes_dict = model_metadata.get("nodes", {})

        sorted_nodes = sorted(nodes_dict.values(), key=lambda x: x.get("index", 0))

        def is_supported(node):
            return node.get("node_type") in JSTPROVE_SUPPORTED_OPS

        for i in range(len(sorted_nodes) - 1):
            curr_node = sorted_nodes[i]
            next_node = sorted_nodes[i+1]

            curr_supported = is_supported(curr_node)
            next_supported = is_supported(next_node)

            if curr_supported != next_supported:
                idx = next_node.get("index")
                if idx is not None:
                    updated_points.add(idx)

        max_idx = max((n.get("index", 0) for n in nodes_dict.values()), default=0)
        return [p for p in updated_points if p is not None and p <= max_idx]

    @staticmethod
    def optimize_for_tiling(slice_points: List[int], model_metadata: Dict) -> List[int]:
        updated_points = set(slice_points)
        nodes_dict = model_metadata.get("nodes", {})

        sorted_nodes = sorted(nodes_dict.values(), key=lambda x: x.get("index", 0))

        def is_tileable(node):
            node_type = node.get("node_type")
            return node_type == "Conv" or node_type in ELEMENTWISE_OPS

        skip_next = False
        for i in range(len(sorted_nodes) - 1):
            if skip_next:
                skip_next = False
                continue

            curr_node = sorted_nodes[i]
            next_node = sorted_nodes[i+1]

            curr_tileable = is_tileable(curr_node)
            next_tileable = is_tileable(next_node)

            if not curr_tileable and next_node.get("node_type") == "Relu":
                skip_next = True
                continue

            if curr_tileable != next_tileable:
                idx = next_node.get("index")
                if idx is not None:
                    updated_points.add(idx)

        max_idx = max((n.get("index", 0) for n in nodes_dict.values()), default=0)
        return [p for p in updated_points if p is not None and p <= max_idx]

    @staticmethod
    def isolate_conv(slice_points, model_metadata):
        """
        Ensure each Conv layer gets its own isolated slice by adding slice points
        immediately after each Conv node.
        
        Args:
            slice_points: Current set of slice points
            model_metadata: The model analysis metadata containing node information
            
        Returns:
            set: Updated set of slice points with Conv isolation
        """
        updated_points = set(slice_points)
        nodes_dict = model_metadata.get("nodes", {})

        max_idx = max((n.get("index", 0) for n in nodes_dict.values()), default=0)

        for _node_name, node_info in nodes_dict.items():
            if node_info.get("node_type") == "Conv":
                idx = node_info.get("index")
                if idx is None:
                    continue
                updated_points.add(idx)
                if idx + 1 <= max_idx:
                    updated_points.add(idx + 1)

        return updated_points

    def determine_slice_points(self, model_metadata, tile_size=None, isolate_convs=True) -> List[int]:
        """
        Determine the slice points for the model based on nodes with parameter_details in the model_metadata.

        Args:
            model_metadata: The model analysis metadata containing node information.
            tile_size: If set, optimize slicing for tiling.
            isolate_convs: If True, each Conv gets its own isolated slice.

        Returns:
            List[int]: List of indices representing nodes with parameter details
        """
        slice_points = set()
        max_idx = max((n.get("index", 0) for n in model_metadata["nodes"].values()), default=0)
        for _node_name, node_info in model_metadata["nodes"].items():
            if node_info.get("parameter_details") and node_info["parameter_details"]:
                idx = node_info["index"]
                slice_points.add(idx)
                if isolate_convs and node_info.get("node_type") == "Conv":
                    if idx + 1 <= max_idx:
                        slice_points.add(idx + 1)

        print(f"Original slice points: {sorted(slice_points)}")
        if isolate_convs:
            slice_points = self.isolate_conv(slice_points, model_metadata)
        slice_points = self.optimize_jstprove_slices(list(slice_points), model_metadata)

        if tile_size:
            slice_points = self.optimize_for_tiling(slice_points, model_metadata)

        slice_points = sorted(set(slice_points))

        print(f"Optimized slice points: {slice_points}")

        self.slice_points = slice_points
        return slice_points

    def _slice_setup(self, model_metadata, output_path=None):
        output_path = os.path.join(os.path.dirname(self.onnx_path), "slices") if output_path is None else output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        graph = self.onnx_model.graph

        node_map = {node.name: node for node in graph.node}

        node_type_index_map = {}
        for i, node in enumerate(graph.node):
            key = f"{node.op_type}_{i}"
            node_type_index_map[key] = node

        initializer_map = {init.name: init for init in graph.initializer}
        value_info_map = {vi.name: vi for vi in graph.value_info}
        value_info_map.update({vi.name: vi for vi in graph.input})
        value_info_map.update({vi.name: vi for vi in graph.output})

        index_to_node_name = {}
        index_to_segment_name = {}
        for node_name, node_info in model_metadata["nodes"].items():
            index_to_node_name[node_info["index"]] = node_name
            index_to_segment_name[node_info["index"]] = node_info["slice_name"]

        return (graph, node_map, node_type_index_map, initializer_map, value_info_map,
                index_to_node_name, index_to_segment_name, output_path)

    @staticmethod
    def _get_nodes(start_idx, end_idx, index_to_node_name, index_to_segment_name, node_map, node_type_index_map,
                   segment_idx):
        segment_nodes = []
        for idx in range(start_idx, end_idx):
            if idx in index_to_node_name:
                node_name = index_to_node_name[idx]
                if node_name in node_map:
                    segment_nodes.append(node_map[node_name])
                else:
                    segment_name = index_to_segment_name.get(idx)
                    if segment_name in node_type_index_map:
                        segment_nodes.append(node_type_index_map[segment_name])
                    else:
                        logger.warning(f"Node {node_name} (index {idx}) not found in the ONNX model")

        if not segment_nodes:
            logger.warning(f"No nodes found for segment {segment_idx} (indices {start_idx}-{end_idx - 1})")

        return segment_nodes

    @staticmethod
    def _get_segment_details(segment_nodes, graph, initializer_map, future_inputs=None):
        future_inputs = future_inputs or set()
        segment_inputs = []
        segment_outputs = []
        segment_initializers = []

        all_value_infos = {}

        for input_info in graph.input:
            all_value_infos[input_info.name] = input_info

        for output_info in graph.output:
            all_value_infos[output_info.name] = output_info

        for value_info in graph.value_info:
            all_value_infos[value_info.name] = value_info

        constant_producers = {}
        for node in graph.node:
            if node.op_type == "Constant":
                for output in node.output:
                    for attr in node.attribute:
                        if attr.name == "value":
                            constant_producers[output] = attr.t

        segment_node_outputs = set()
        for node in segment_nodes:
            for output in node.output:
                segment_node_outputs.add(output)

        segment_node_inputs = set()
        for node in segment_nodes:
            for inp in node.input:
                if inp:
                    segment_node_inputs.add(inp)

        for inp in segment_node_inputs:
            if inp not in segment_node_outputs:
                if inp in initializer_map:
                    init = initializer_map[inp]
                    segment_initializers.append(init)
                elif inp in constant_producers:
                    tensor = constant_producers[inp]
                    init = onnx.TensorProto()
                    init.CopyFrom(tensor)
                    init.name = inp
                    segment_initializers.append(init)
                elif inp in all_value_infos:
                    segment_inputs.append(all_value_infos[inp])
                else:
                    inferred_shape = OnnxSlicer._infer_input_shape(inp, segment_nodes)
                    t = onnx.helper.make_tensor_value_info(
                        inp,
                        onnx.TensorProto.FLOAT,
                        inferred_shape
                    )
                    segment_inputs.append(t)

        model_output_names = {o.name for o in graph.output}
        for out in segment_node_outputs:
            consumed_internally = any(out in node.input for node in segment_nodes)
            needed_externally = out in future_inputs or out in model_output_names

            if not consumed_internally or needed_externally:
                if out in all_value_infos:
                    segment_outputs.append(all_value_infos[out])
                else:
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
        for node in segment_nodes:
            if input_name in node.input:
                if node.op_type == "Conv":
                    return ["batch_size", None, None, None]
                elif node.op_type == "Gemm":
                    return ["batch_size", None]
                elif node.op_type in ["Relu", "Tanh", "Sigmoid", "LeakyRelu", "BatchNormalization", "LayerNormalization"]:
                    return ["batch_size", None, None, None]
                elif node.op_type in ["Add", "Mul", "Sub", "Div"]:
                    return ["batch_size", None, None, None]
                elif node.op_type == "GlobalAveragePool":
                    return ["batch_size", None, None, None]
                elif node.op_type == "AveragePool":
                    return ["batch_size", None, None, None]

        return ["batch_size", None]

    @staticmethod
    def _infer_output_shape(output_name, segment_nodes):
        for node in segment_nodes:
            if output_name in node.output:
                if node.op_type == "Conv":
                    return ["batch_size", None, None, None]
                elif node.op_type == "Gemm":
                    return ["batch_size", None]
                elif node.op_type in ["Relu", "Tanh", "Sigmoid", "LeakyRelu", "BatchNormalization", "LayerNormalization"]:
                    return ["batch_size", None, None, None]
                elif node.op_type in ["Add", "Mul", "Sub", "Div"]:
                    return ["batch_size", None, None, None]
                elif node.op_type == "GlobalAveragePool":
                    return ["batch_size", None, 1, 1]
                elif node.op_type == "AveragePool":
                    return ["batch_size", None, None, None]
                elif node.op_type == "Flatten":
                    return ["batch_size", None]
                elif node.op_type == "Reshape":
                    return ["batch_size", None]

        return ["batch_size", None]

    def slice(self, slice_points: List[int], model_metadata, output_path=None, parallel: bool = False):
        """
        Slice the ONNX model based on the provided slice points.

        Args:
            slice_points: List of indices representing nodes with parameter details
            model_metadata: The model analysis metadata containing node information
            output_path: The path to save the slices to
            parallel: If True, parallelize extraction and post-processing

        Returns:
            Tuple[List[str], Dict]: Paths to sliced models and tiling metadata
        """
        if not slice_points:
            raise ValueError("No slice points provided.")

        if not model_metadata or "nodes" not in model_metadata:
            raise ValueError("Invalid model metadata. Please run 'analyze()' first.")

        logger.info("Applying shape inference to original model...")
        try:
            self.onnx_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(self.onnx_model)
            logger.info("Shape inference applied successfully to original model")
        except Exception as e:
            logger.warning(f"Shape inference failed on original model: {e}, continuing with original model")

        tensor_graph = TensorGraph(self.onnx_path)
        logger.info(f"Built tensor graph: {tensor_graph}")

        (graph, node_map, node_type_index_map, initializer_map, _value_info_map,
         index_to_node_name, index_to_segment_name, output_path) = self._slice_setup(model_metadata, output_path)

        max_index = max(node_info["index"] for node_info in model_metadata["nodes"].values())
        if max_index + 1 not in slice_points:
            slice_points.append(max_index + 1)

        slice_points.sort()

        segment_inputs_map = {}
        for i in range(len(slice_points)):
            seg_idx = i
            start_idx = slice_points[i - 1] if i > 0 else 0
            end_idx = slice_points[i]
            if start_idx == end_idx:
                continue
            seg_nodes = self._get_nodes(start_idx, end_idx, index_to_node_name,
                                        index_to_segment_name, node_map, node_type_index_map, seg_idx)
            seg_outputs = set()
            for node in seg_nodes:
                seg_outputs.update(node.output)
            seg_inputs = set()
            for node in seg_nodes:
                for inp in node.input:
                    if inp not in seg_outputs and inp not in initializer_map:
                        seg_inputs.add(inp)
            segment_inputs_map[seg_idx] = seg_inputs

        slice_specs = []
        fallback_data = {}

        for i in range(len(slice_points)):
            segment_idx = i
            start_idx = slice_points[i - 1] if i > 0 else 0
            end_idx = slice_points[i]

            if start_idx == end_idx:
                continue

            segment_nodes = self._get_nodes(start_idx, end_idx, index_to_node_name,
                                            index_to_segment_name, node_map, node_type_index_map, segment_idx)

            if not segment_nodes:
                continue

            future_inputs = set()
            for future_idx in segment_inputs_map:
                if future_idx > segment_idx:
                    future_inputs.update(segment_inputs_map[future_idx])

            segment_inputs, segment_outputs, segment_initializers = self._get_segment_details(
                segment_nodes, graph, initializer_map, future_inputs)

            save_path = os.path.join(output_path, f"slice_{segment_idx}")
            payload_dir = os.path.join(save_path, "payload")
            file_path = os.path.join(payload_dir, f"slice_{segment_idx}.onnx")

            input_names = Utils.filter_inputs(segment_inputs, graph)
            output_names = [output_info.name for output_info in segment_outputs]

            spec = (self.onnx_path, segment_idx, input_names, output_names, file_path)
            slice_specs.append(spec)

            fallback_data[segment_idx] = {
                'segment_nodes': segment_nodes,
                'segment_inputs': segment_inputs,
                'segment_outputs': segment_outputs,
                'segment_initializers': segment_initializers,
                'file_path': file_path,
            }

            logger.info(f"Prepared slice {segment_idx}: {input_names} -> {output_names}")

        extracted = {}
        failed_indices = []

        if parallel and len(slice_specs) > 1:
            max_workers = min(len(slice_specs), multiprocessing.cpu_count())
            print(f"Extracting {len(slice_specs)} slices in parallel (workers={max_workers})...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_extract_single_slice, spec): spec[1] for spec in slice_specs}
                for future in as_completed(futures):
                    segment_idx = futures[future]
                    result = future.result()
                    if result:
                        idx, path = result
                        extracted[idx] = path
                        print(f"  Extracted slice {idx}")
                    else:
                        failed_indices.append(segment_idx)
        else:
            print(f"Extracting {len(slice_specs)} slices sequentially...")
            for spec in slice_specs:
                result = _extract_single_slice(spec)
                if result:
                    idx, path = result
                    extracted[idx] = path
                    print(f"  Extracted slice {idx}")
                else:
                    failed_indices.append(spec[1])

        for segment_idx in failed_indices:
            data = fallback_data[segment_idx]
            file_path = data['file_path']
            try:
                print(f"  Fallback: building slice {segment_idx} manually...")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                segment_graph = onnx.helper.make_graph(
                    data['segment_nodes'],
                    f"segment_{segment_idx}_graph",
                    data['segment_inputs'],
                    data['segment_outputs'],
                    data['segment_initializers']
                )
                segment_model = onnx.helper.make_model(segment_graph)
                segment_model = self._concretize_symbolic_dims(segment_model, value=1)
                save_onnx_model(segment_model, file_path)
                extracted[segment_idx] = file_path
            except Exception as e:
                logger.error(f"Fallback failed for segment {segment_idx}: {e}")

        slice_paths = [extracted[idx] for idx in sorted(extracted.keys())]

        abs_paths = self.slice_post_process(slice_paths, parallel=parallel, traced_shapes=self.traced_shapes, traced_dtypes=self.traced_dtypes)

        abs_paths_dict = {idx: abs_paths[i] for i, idx in enumerate(sorted(extracted.keys())) if i < len(abs_paths)}

        tiled_info = {}
        return abs_paths_dict, tiled_info, tensor_graph

    @staticmethod
    def _process_single_slice(path: str, traced_shapes: dict = None, traced_dtypes: dict = None) -> str | None:
        abs_path = os.path.abspath(path)
        try:
            model = onnx.load(path)

            if traced_shapes:
                OnnxSlicer._apply_traced_shapes(model, traced_shapes, traced_dtypes)
            else:
                model = OnnxSlicer._concretize_symbolic_dims(model, value=1)

            onnx.checker.check_model(model)
            save_onnx_model(model, path)
            return abs_path
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None

    @staticmethod
    def _apply_traced_shapes(model: onnx.ModelProto, traced_shapes: dict, traced_dtypes: dict = None):
        """Apply traced runtime shapes to model inputs/outputs/intermediates."""
        import numpy as np
        from onnx import helper, TensorProto

        dtype_map = {
            np.dtype('float16'): TensorProto.FLOAT16,
            np.dtype('float32'): TensorProto.FLOAT,
            np.dtype('float64'): TensorProto.DOUBLE,
            np.dtype('int32'): TensorProto.INT32,
            np.dtype('int64'): TensorProto.INT64,
            np.dtype('bool'): TensorProto.BOOL,
        }

        def get_onnx_dtype(name):
            if traced_dtypes and name in traced_dtypes:
                return dtype_map.get(traced_dtypes[name], TensorProto.FLOAT)
            return TensorProto.FLOAT

        def set_shape(value_info, shape):
            value_info.type.tensor_type.shape.ClearField('dim')
            for dim_val in shape:
                dim = value_info.type.tensor_type.shape.dim.add()
                dim.dim_value = dim_val

        existing_names = set()
        for inp in model.graph.input:
            existing_names.add(inp.name)
            if inp.name in traced_shapes:
                set_shape(inp, traced_shapes[inp.name])

        for out in model.graph.output:
            existing_names.add(out.name)
            if out.name in traced_shapes:
                set_shape(out, traced_shapes[out.name])

        for vi in model.graph.value_info:
            existing_names.add(vi.name)
            if vi.name in traced_shapes:
                set_shape(vi, traced_shapes[vi.name])

        for name, shape in traced_shapes.items():
            if name not in existing_names:
                onnx_dtype = get_onnx_dtype(name)
                vi = helper.make_tensor_value_info(name, onnx_dtype, shape)
                model.graph.value_info.append(vi)

    @staticmethod
    def slice_post_process(slices_paths, parallel: bool = False, traced_shapes: dict = None, traced_dtypes: dict = None):
        """Post-process sliced models with traced shape and dtype application."""
        if parallel and len(slices_paths) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            max_workers = min(len(slices_paths), multiprocessing.cpu_count())
            results = [None] * len(slices_paths)
            path_to_idx = {p: i for i, p in enumerate(slices_paths)}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(OnnxSlicer._process_single_slice, p, traced_shapes, traced_dtypes): p for p in slices_paths}
                for future in as_completed(futures):
                    original_path = futures[future]
                    result = future.result()
                    if result:
                        results[path_to_idx[original_path]] = result
            return [r for r in results if r is not None]

        abs_paths = []
        for path in slices_paths:
            result = OnnxSlicer._process_single_slice(path, traced_shapes, traced_dtypes)
            if result:
                abs_paths.append(result)
        return abs_paths

    def slice_model(self, output_path=None, max_conv_size: int = None, tile_size: int = None, parallel: bool = False):
        """
        Run the complete workflow: determine slice points, slice, and optionally tile.

        Two-phase approach:
        1. Slice model (isolating Convs into separate slices)
        2. Apply tiling transform (inject split/concat into bridge slices)

        Args:
            output_path: The path to save the slices to.
            max_conv_size: Maximum elements per tile. Tile size is calculated dynamically
                           per-Conv based on channel count: tile_size = sqrt(max_conv_size / channels).
                           Recommended over tile_size for better ZK circuit sizing.
            tile_size: Fixed tile size for all Convs (legacy). Ignored if max_conv_size is set.
            parallel: If True, parallelize operations.

        Returns:
            Dict[str, Any]: Metadata about the sliced model
        """
        should_tile = max_conv_size is not None or tile_size is not None
        slice_points = self.determine_slice_points(self.analysis, tile_size if not max_conv_size else 1)
        slices_paths, tiled_info, tensor_graph = self.slice(slice_points, self.analysis, output_path, parallel=parallel)

        self.onnx_analyzer.generate_slices_metadata(self.analysis, slice_points, slices_paths, output_path, tiled_info)

        if should_tile:
            if max_conv_size:
                logger.info(f"Applying tiling transform with max_conv_size={max_conv_size}")
                apply_tiling_to_slices(output_path, max_conv_size=max_conv_size, parallel=parallel, tensor_graph=tensor_graph)
            else:
                logger.info(f"Applying tiling transform with tile_size={tile_size}")
                apply_tiling_to_slices(output_path, tile_size=tile_size, parallel=parallel, tensor_graph=tensor_graph)

        return slices_paths


if __name__ == "__main__":
    model_choice = 1

    base_paths = {
        1: "../../models/doom",
        2: "../../models/net",
        3: "../../models/resnet",
        4: "../../models/age",
        5: "../../models/version",
        6: "../../models/bert",
        7: "../../models/roberta",
        8: "../../models/yolov8"
    }

    abs_path = os.path.abspath(base_paths[model_choice])
    model_dir = os.path.join(abs_path, "model.onnx")
    output_dir = os.path.join(abs_path, "slices")
    onnx_slicer = OnnxSlicer(model_dir, save_path=base_paths[model_choice])
    onnx_slicer.slice_model(output_path=output_dir)#, tile_size=16)
