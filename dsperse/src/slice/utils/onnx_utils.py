import json
import os
import shutil
import tempfile
import numpy as np
import onnxruntime as ort
from onnx.utils import extract_model
from onnx import helper, TensorProto
from pathlib import Path
from typing import Tuple, List, Dict

import onnx
import logging

from dsperse.src.analyzers.schema import SliceMetadata, ModelMetadata, SliceShape, TensorShape, WeightShape, TilingInfo, ChannelSplitInfo
from dsperse.src.backends.jstprove import JSTPROVE_SUPPORTED_OPS
from dsperse.src.slice.autotiler import ELEMENTWISE_OPS
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

_MAX_ORT_IR_VERSION = 10
_MAX_ORT_OPSET_VERSION = 22


class OnnxUtils:
    def __init__(self):
        pass

    @staticmethod
    def clamp_for_ort(model: onnx.ModelProto) -> None:
        """Clamp IR and opset versions to the maximums supported by ORT."""
        if model.ir_version > _MAX_ORT_IR_VERSION:
            model.ir_version = _MAX_ORT_IR_VERSION
        for opset in model.opset_import:
            if opset.domain in ("", "ai.onnx") and opset.version > _MAX_ORT_OPSET_VERSION:
                opset.version = _MAX_ORT_OPSET_VERSION

    # =========================================================================
    # Section 1: Graph Setup & Map Building
    # =========================================================================

    @staticmethod
    def build_graph_maps(graph: onnx.GraphProto) -> Tuple[Dict, Dict, Dict, Dict]:
        """Build lookup maps for ONNX graph components."""
        node_map = {node.name: node for node in graph.node}

        node_type_index_map = {}
        for i, node in enumerate(graph.node):
            key = f"{node.op_type}_{i}"
            node_type_index_map[key] = node

        initializer_map = {init.name: init for init in graph.initializer}
        value_info_map = {vi.name: vi for vi in graph.value_info}
        value_info_map.update({vi.name: vi for vi in graph.input})
        value_info_map.update({vi.name: vi for vi in graph.output})

        return node_map, node_type_index_map, initializer_map, value_info_map

    @staticmethod
    def build_metadata_maps(model_metadata: Dict) -> Tuple[Dict, Dict]:
        """Build maps from analysis metadata to node/segment names."""
        index_to_node_name = {}
        index_to_segment_name = {}
        for node_name, node_info in model_metadata.get("nodes", {}).items():
            index_to_node_name[node_info["index"]] = node_name
            index_to_segment_name[node_info["index"]] = node_info["slice_name"]
        return index_to_node_name, index_to_segment_name

    @staticmethod
    def initialize_tensor_graph(onnx_path: str):
        """Initialize TensorGraph from ONNX path."""
        from dsperse.src.slice.tensor_graph import TensorGraph
        tensor_graph = TensorGraph(onnx_path)
        logger.info(f"Built tensor graph: {tensor_graph}")
        return tensor_graph

    @staticmethod
    def apply_symbolic_shape_inference(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply symbolic shape inference to the model."""
        from onnxruntime.tools import symbolic_shape_infer
        logger.info("Applying shape inference to model...")
        try:
            onnx_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx_model)
            logger.info("Shape inference applied successfully")
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}, continuing with original model")
        return onnx_model

    # =========================================================================
    # Section 2: Shape Tracing & Runtime Inference
    # =========================================================================

    @staticmethod
    def trace_shapes(onnx_model: onnx.ModelProto) -> Tuple[Dict[str, List[int]], Dict[str, np.dtype]]:
        """Run model with dummy input to capture all intermediate tensor shapes."""
        all_tensor_names = []
        for node in onnx_model.graph.node:
            for out in node.output:
                all_tensor_names.append(out)

        trace_model = onnx.ModelProto()
        trace_model.CopyFrom(onnx_model)
        existing_outputs = {o.name for o in trace_model.graph.output}
        for tensor_name in all_tensor_names:
            if tensor_name not in existing_outputs:
                new_output = trace_model.graph.output.add()
                new_output.name = tensor_name

        fd, trace_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        
        traced_shapes = {}
        traced_dtypes = {}
        
        try:
            OnnxUtils.clamp_for_ort(trace_model)
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
                traced_shapes[inp.name] = shape

            outputs = session.run(None, inputs)
            output_names = [o.name for o in session.get_outputs()]
            for name, arr in zip(output_names, outputs, strict=True):
                traced_shapes[name] = list(arr.shape)
                traced_dtypes[name] = arr.dtype
        finally:
            if os.path.exists(trace_path):
                os.remove(trace_path)
        
        print(f"Shape trace complete: captured {len(traced_shapes)} tensor shapes")
        return traced_shapes, traced_dtypes

    @staticmethod
    def apply_traced_shapes(model: onnx.ModelProto, traced_shapes: dict, traced_dtypes: dict = None):
        """Apply traced runtime shapes to model inputs/outputs/intermediates."""
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
    def concretize_symbolic_dims(model: onnx.ModelProto, value: int = 1) -> onnx.ModelProto:
        """Replace symbolic dimensions with concrete values."""
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

    # =========================================================================
    # Section 3: Slice Point Optimization
    # =========================================================================

    @staticmethod
    def isolate_conv(slice_points, model_metadata):
        """
        Ensure each Conv layer gets its own isolated slice by adding slice points
        immediately after each Conv node.
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
        return sorted(updated_points)

    @staticmethod
    def optimize_jstprove_slices(slice_points: List[int], model_metadata: Dict) -> List[int]:
        """Optimize slice points for JSTprove compatibility."""
        updated_points = set(slice_points)
        nodes_dict = model_metadata.get("nodes", {})
        sorted_nodes = sorted(nodes_dict.values(), key=lambda x: x.get("index", 0))

        def is_supported(node):
            return node.get("node_type") in JSTPROVE_SUPPORTED_OPS

        for i in range(len(sorted_nodes) - 1):
            curr_node = sorted_nodes[i]
            next_node = sorted_nodes[i+1]
            if is_supported(curr_node) != is_supported(next_node):
                idx = next_node.get("index")
                if idx is not None:
                    updated_points.add(idx)

        max_idx = max((n.get("index", 0) for n in nodes_dict.values()), default=0)
        return sorted(p for p in updated_points if p is not None and p <= max_idx)

    @staticmethod
    def optimize_for_tiling(slice_points: List[int], model_metadata: Dict) -> List[int]:
        """Optimize slice points for tiling."""
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

            if not is_tileable(curr_node) and next_node.get("node_type") == "Relu":
                skip_next = True
                continue

            if is_tileable(curr_node) != is_tileable(next_node):
                idx = next_node.get("index")
                if idx is not None:
                    updated_points.add(idx)

        max_idx = max((n.get("index", 0) for n in nodes_dict.values()), default=0)
        return sorted(p for p in updated_points if p is not None and p <= max_idx)

    @staticmethod
    def filter_constant_only_slices(slice_points: List[int], model_metadata: Dict) -> List[int]:
        """Remove slice points that would create constant-only slices (no inputs)."""
        if not slice_points:
            return slice_points

        nodes_dict = model_metadata.get("nodes", {})
        nodes_by_idx = {n.get("index"): n for n in nodes_dict.values()}
        sorted_points = sorted(slice_points)
        points_to_remove = set()

        for i, end_idx in enumerate(sorted_points):
            start_idx = sorted_points[i - 1] if i > 0 else 0
            all_constant = True
            for idx in range(start_idx, end_idx):
                node = nodes_by_idx.get(idx)
                if node and node.get("node_type") != "Constant":
                    all_constant = False
                    break
            if all_constant and start_idx != end_idx:
                points_to_remove.add(end_idx)

        filtered = [p for p in sorted_points if p not in points_to_remove]
        if points_to_remove:
            logger.info(f"Merged {len(points_to_remove)} constant-only slices")
        return filtered

    @staticmethod
    def complete_slice_points(slice_points: List[int], model_metadata: Dict) -> List[int]:
        """Ensure slice points include the end of the model and are sorted."""
        updated_points = list(set(slice_points))
        max_index = max(node_info["index"] for node_info in model_metadata["nodes"].values())
        if max_index + 1 not in updated_points:
            updated_points.append(max_index + 1)
        updated_points.sort()
        return updated_points

    # =========================================================================
    # Section 4: Extraction Specification & Execution
    # =========================================================================

    @staticmethod
    def analyze_future_dependencies(slice_points: List[int], index_to_node_name: Dict,
                                    index_to_segment_name: Dict, node_map: Dict,
                                    node_type_index_map: Dict, initializer_map: Dict) -> Dict[int, set]:
        """Pre-calculate segment inputs to determine future dependencies."""
        segment_inputs_map = {}
        seg_idx = 0
        for i in range(len(slice_points)):
            start_idx = slice_points[i - 1] if i > 0 else 0
            end_idx = slice_points[i]
            if start_idx == end_idx:
                continue
            seg_nodes = OnnxUtils.get_nodes_by_index_range(start_idx, end_idx, index_to_node_name,
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
            seg_idx += 1
        return segment_inputs_map

    @staticmethod
    def prepare_extraction_specs(onnx_path: str, slice_points: List[int], output_path: str,
                                 index_to_node_name: Dict, index_to_segment_name: Dict,
                                 node_map: Dict, node_type_index_map: Dict,
                                 initializer_map: Dict, graph: onnx.GraphProto,
                                 segment_inputs_map: Dict[int, set]) -> Tuple[List, Dict]:
        """Prepare slice specifications and fallback data."""
        slice_specs = []
        fallback_data = {}

        segment_idx = 0
        for i in range(len(slice_points)):
            start_idx = slice_points[i - 1] if i > 0 else 0
            end_idx = slice_points[i]

            if start_idx == end_idx:
                continue

            segment_nodes = OnnxUtils.get_nodes_by_index_range(start_idx, end_idx, index_to_node_name,
                                                               index_to_segment_name, node_map, node_type_index_map, segment_idx)

            if not segment_nodes:
                continue

            future_inputs = set()
            for future_idx in segment_inputs_map:
                if future_idx > segment_idx:
                    future_inputs.update(segment_inputs_map[future_idx])

            segment_inputs, segment_outputs, segment_initializers = OnnxUtils.get_segment_details(
                segment_nodes, graph, initializer_map, future_inputs)

            save_path = os.path.join(output_path, f"slice_{segment_idx}")
            payload_dir = os.path.join(save_path, "payload")
            file_path = os.path.join(payload_dir, f"slice_{segment_idx}.onnx")

            input_names = Utils.filter_inputs(segment_inputs, graph)
            output_names = [output_info.name for output_info in segment_outputs]

            spec = (onnx_path, segment_idx, input_names, output_names, file_path)
            slice_specs.append(spec)

            fallback_data[segment_idx] = {
                'segment_nodes': segment_nodes,
                'segment_inputs': segment_inputs,
                'segment_outputs': segment_outputs,
                'segment_initializers': segment_initializers,
                'file_path': file_path,
            }

            logger.info(f"Prepared slice {segment_idx}: {input_names} -> {output_names}")
            segment_idx += 1

        return slice_specs, fallback_data

    @staticmethod
    def run_extraction_pipeline(slice_specs: List, fallback_data: Dict) -> Dict[int, str]:
        """Run slice extraction with manual fallback."""
        extracted = {}

        print(f"Extracting {len(slice_specs)} slices...")
        for spec in slice_specs:
            # spec: (onnx_path, segment_idx, input_names, output_names, file_path)
            segment_idx = spec[1]
            file_path = spec[4]

            result = OnnxUtils.extract_single_slice(spec)
            if result:
                idx, path = result
                extracted[idx] = path
                print(f"  Extracted slice {idx}")
            else:
                # Fallback: building manually
                data = fallback_data[segment_idx]
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
                    segment_model = OnnxUtils.concretize_symbolic_dims(segment_model, value=1)
                    Utils.save_onnx_model(segment_model, file_path)
                    extracted[segment_idx] = file_path
                except Exception as e:
                    logger.error(f"Fallback failed for segment {segment_idx}: {e}")

        return extracted

    @staticmethod
    def extract_single_slice(spec: Tuple[str, int, List[str], List[str], str]) -> Tuple[int, str] | None:
        """Extract a single slice using ONNX's extract_model."""
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
            logger.warning(
                f"extract_model failed for slice {segment_idx}: {err_msg} (inputs={input_names}, outputs={output_names})")
            return None

    @staticmethod
    def finalize_slice(path: str, traced_shapes: dict = None, traced_dtypes: dict = None) -> str | None:
        """
        Finalize a sliced model by applying shapes, validating the graph, and saving.
        """
        abs_path = os.path.abspath(path)
        try:
            model = onnx.load(path)

            if traced_shapes:
                OnnxUtils.apply_traced_shapes(model, traced_shapes, traced_dtypes)
            else:
                model = OnnxUtils.concretize_symbolic_dims(model, value=1)

            onnx.checker.check_model(model)
            Utils.save_onnx_model(model, path)
            return abs_path
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None

    # =========================================================================
    # Section 5: Graph Analysis Utilities
    # =========================================================================

    @staticmethod
    def get_nodes_by_index_range(start_idx, end_idx, index_to_node_name, index_to_segment_name, node_map, node_type_index_map,
                                 segment_idx):
        """Retrieve ONNX nodes for a given index range."""
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
    def _infer_shape_from_op(op_type: str, is_output: bool = False) -> List[int | str]:
        """Common logic for fallback shape inference."""
        if op_type == "Conv":
            return ["batch_size", None, None, None]
        if op_type == "Gemm":
            return ["batch_size", None]
        if op_type in ["Relu", "Tanh", "Sigmoid", "LeakyRelu", "BatchNormalization", "LayerNormalization"]:
            return ["batch_size", None, None, None]
        if op_type in ["Add", "Mul", "Sub", "Div"]:
            return ["batch_size", None, None, None]
        if op_type == "GlobalAveragePool":
            return ["batch_size", None, 1, 1] if is_output else ["batch_size", None, None, None]
        if op_type == "AveragePool":
            return ["batch_size", None, None, None]
        if is_output and op_type in ["Flatten", "Reshape"]:
            return ["batch_size", None]
        return ["batch_size", None]

    @staticmethod
    def get_segment_details(segment_nodes, graph, initializer_map, future_inputs=None):
        """Extract inputs, outputs, and initializers for a model segment."""
        future_inputs = future_inputs or set()
        segment_inputs = []
        segment_outputs = []
        segment_initializers = []

        all_value_infos = {vi.name: vi for vi in list(graph.input) + list(graph.output) + list(graph.value_info)}

        constant_producers = {}
        for node in graph.node:
            if node.op_type == "Constant":
                for output in node.output:
                    for attr in node.attribute:
                        if attr.name == "value":
                            constant_producers[output] = attr.t

        segment_node_outputs = set()
        for node in segment_nodes:
            segment_node_outputs.update(node.output)

        segment_node_inputs = set()
        for node in segment_nodes:
            for inp in node.input:
                if inp:
                    segment_node_inputs.add(inp)

        for inp in segment_node_inputs:
            if inp not in segment_node_outputs:
                if inp in initializer_map:
                    segment_initializers.append(initializer_map[inp])
                elif inp in constant_producers:
                    tensor = constant_producers[inp]
                    init = onnx.TensorProto()
                    init.CopyFrom(tensor)
                    init.name = inp
                    segment_initializers.append(init)
                elif inp in all_value_infos:
                    segment_inputs.append(all_value_infos[inp])
                else:
                    inferred_shape = ["batch_size", None]
                    for node in segment_nodes:
                        if inp in node.input:
                            inferred_shape = OnnxUtils._infer_shape_from_op(node.op_type, is_output=False)
                            break
                    inferred_dtype = OnnxUtils.infer_tensor_dtype(inp, segment_nodes, graph, is_input=True)
                    t = helper.make_tensor_value_info(inp, inferred_dtype, inferred_shape)
                    segment_inputs.append(t)

        model_output_names = {o.name for o in graph.output}
        for out in segment_node_outputs:
            consumed_internally = any(out in node.input for node in segment_nodes)
            needed_externally = out in future_inputs or out in model_output_names

            if not consumed_internally or needed_externally:
                if out in all_value_infos:
                    segment_outputs.append(all_value_infos[out])
                else:
                    inferred_shape = ["batch_size", None]
                    for node in segment_nodes:
                        if out in node.output:
                            inferred_shape = OnnxUtils._infer_shape_from_op(node.op_type, is_output=True)
                            break
                    inferred_dtype = OnnxUtils.infer_tensor_dtype(out, segment_nodes, graph, is_input=False)
                    t = helper.make_tensor_value_info(out, inferred_dtype, inferred_shape)
                    segment_outputs.append(t)

        return segment_inputs, segment_outputs, segment_initializers

    @staticmethod
    def infer_tensor_dtype(tensor_name, segment_nodes, graph=None, is_input=True):
        """Infer dtype for a tensor based on producing/consuming nodes."""
        INT64_OUTPUT_OPS = {"Shape", "Size", "NonZero"}
        INT64_INPUT_POSITIONS = {
            "Gather": {1}, "GatherElements": {1}, "GatherND": {1},
            "Scatter": {1}, "ScatterElements": {1}, "ScatterND": {1},
            "Reshape": {1}, "Slice": {1, 2, 3, 4}, "Squeeze": {1},
            "Unsqueeze": {1}, "Expand": {1}, "Tile": {1}, "TopK": {1},
            "Pad": {1, 2}, "ConstantOfShape": {0}, "Range": {0, 1, 2},
        }

        all_nodes = list(segment_nodes)
        if graph:
            all_nodes = list(graph.node)

        if is_input:
            for node in segment_nodes:
                if tensor_name in node.input:
                    input_idx = list(node.input).index(tensor_name)
                    if node.op_type in INT64_INPUT_POSITIONS:
                        if input_idx in INT64_INPUT_POSITIONS[node.op_type]:
                            return TensorProto.INT64
        else:
            for node in all_nodes:
                if tensor_name in node.output:
                    if node.op_type in INT64_OUTPUT_OPS:
                        return TensorProto.INT64
                    if node.op_type == "Cast":
                        for attr in node.attribute:
                            if attr.name == "to" and attr.i == TensorProto.INT64:
                                return TensorProto.INT64

        return TensorProto.FLOAT


    # =========================================================================
    # Section 6: Metadata Utilities
    # =========================================================================

    @staticmethod
    def write_slice_dirs_metadata(slices_root: str):
        """
        Ensure each per-slice directory (slice_#) contains a dslice-style metadata.json
        alongside payload/model.onnx so the folder can be zipped to become a valid .dslice.
        Also updates the global slices metadata slice 'path' to payload/model.onnx if needed.
        """
        root = Path(slices_root).resolve()
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            alt = root / "slices" / "metadata.json"
            if alt.exists():
                metadata_path = alt
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found near {root}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        original_model = meta.get("original_model")
        dsperse_ver = Utils.get_dsperse_version()
        opset_version = OnnxUtils._get_opset_version(original_model)

        slices = meta.get("slices", []) or []
        for idx, seg in enumerate(slices):
            # 1. Normalize directory structure
            slice_dir, payload_dir = OnnxUtils._normalize_slice_dirs(idx, seg, root)
            if not slice_dir:
                continue

            # 2. Normalize ONNX payload name
            expected_filename = f"slice_{idx}.onnx"
            onnx_path = OnnxUtils._normalize_onnx_payload(idx, seg, payload_dir, expected_filename)
            if not onnx_path:
                continue

            # 3. Build and save per-slice metadata
            updated_slice = OnnxUtils._finalize_slice_metadata(
                idx, seg, root, slice_dir, onnx_path, expected_filename,
                dsperse_ver, opset_version, meta
            )

            # 4. Update global metadata entry
            seg.update({
                "path": updated_slice.path,
                "relative_path": str(Path(updated_slice.path).resolve().relative_to(root)),
                "slice_metadata": updated_slice.slice_metadata,
                "slice_metadata_relative_path": updated_slice.slice_metadata_relative_path,
                "dsperse_version": updated_slice.dsperse_version,
                "opset_version": updated_slice.opset_version
            })

        Utils.save_metadata_file(meta, metadata_path.parent, metadata_path.name)

    @staticmethod
    def _normalize_slice_dirs(idx: int, seg: dict, root: Path) -> Tuple[Path | None, Path | None]:
        """Normalize slice_# directory name and ensure payload dir exists."""
        seg_path_val = seg.get("path")
        if not seg_path_val:
            return None, None
        seg_path = Path(seg_path_val)

        if seg_path.suffix == ".onnx":
            payload_dir = seg_path.parent
            slice_dir = payload_dir.parent
        else:
            slice_dir = seg_path if seg_path.is_dir() else seg_path.parent
            payload_dir = slice_dir / "payload"

        expected_dir_name = f"slice_{idx}"
        if slice_dir.name != expected_dir_name:
            try:
                target_dir = slice_dir.parent / expected_dir_name
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if not target_dir.exists():
                    shutil.move(str(slice_dir), str(target_dir))
                slice_dir = target_dir
                payload_dir = slice_dir / "payload"
            except Exception as e:
                logger.warning(f"Failed to rename slice directory for idx {idx}: {e}")

        payload_dir.mkdir(parents=True, exist_ok=True)
        return slice_dir, payload_dir

    @staticmethod
    def _normalize_onnx_payload(idx: int, seg: dict, payload_dir: Path, expected_filename: str) -> Path | None:
        """Rename ONNX file to expected slice_{idx}.onnx name."""
        desired_path = payload_dir / expected_filename
        current_file = None

        if (payload_dir / expected_filename).exists():
            current_file = payload_dir / expected_filename
        elif (payload_dir / "model.onnx").exists():
            current_file = payload_dir / "model.onnx"
        elif seg.get("path") and Path(seg["path"]).is_file():
            current_file = Path(seg["path"])

        if current_file and current_file != desired_path:
            try:
                shutil.move(str(current_file), str(desired_path))
                return desired_path
            except Exception as e:
                logger.warning(f"Failed to move ONNX for idx {idx}: {e}")
                return current_file
        return desired_path if current_file else None

    @staticmethod
    def _finalize_slice_metadata(idx, seg, root, slice_dir, onnx_path, filename, dsperse_ver, opset_version, global_meta):
        """Build and save authoritative SliceMetadata and ModelMetadata for a slice."""
        orig_slice = SliceMetadata.from_dict({**seg, "index": seg.get("index", idx)})
        input_shapes = orig_slice.shape.tensor_shape.input or seg.get("input_shape") or seg.get("input_shapes") or []
        output_shapes = orig_slice.shape.tensor_shape.output or seg.get("output_shape") or seg.get("output_shapes") or []

        tiling = orig_slice.tiling
        if tiling:
            tiling_dict = Utils.relativize_tiling_info(tiling.to_dict(), root_dir=slice_dir, base_dir=root)
            tiling = TilingInfo.from_dict(tiling_dict)

        channel_split = orig_slice.channel_split
        if channel_split:
            channel_split_dict = Utils.relativize_tiling_info(channel_split.to_dict(), root_dir=slice_dir, base_dir=root)
            channel_split = ChannelSplitInfo.from_dict(channel_split_dict)

        shape = SliceShape(
            tensor_shape=TensorShape(input=input_shapes, output=output_shapes),
            weight_shape=orig_slice.shape.weight_shape if orig_slice.shape.weight_shape else WeightShape(),
        )

        slice_metadata_path = slice_dir / "metadata.json"
        updated_slice = SliceMetadata(
            index=idx,
            filename=filename,
            path=str(onnx_path),
            relative_path=str(onnx_path.resolve().relative_to(root).relative_to("slice_" + str(idx))),
            parameters=orig_slice.parameters,
            shape=shape,
            dependencies=orig_slice.dependencies,
            layers=orig_slice.layers,
            tiling=tiling,
            channel_split=channel_split,
            compilation=orig_slice.compilation,
            dsperse_version=dsperse_ver,
            opset_version=opset_version,
            slice_metadata=str(slice_metadata_path.resolve()),
            slice_metadata_relative_path=str(slice_metadata_path.resolve().relative_to(root)),
        )

        model_meta = ModelMetadata(
            original_model=global_meta.get("original_model") or "",
            model_type=global_meta.get("model_type", "ONNX"),
            total_parameters=updated_slice.parameters,
            input_shape=global_meta.get("input_shape", input_shapes),
            output_shapes=global_meta.get("output_shapes", output_shapes),
            slice_points=[idx],
            slices=[updated_slice],
        )
        model_meta.save(slice_metadata_path)
        return updated_slice

    @staticmethod
    def _get_opset_version(model_path: str) -> int | None:
        """Best-effort retrieval of ONNX opset version."""
        if not model_path or not os.path.exists(model_path):
            return None
        try:
            mdl = onnx.load(model_path)
            for ops in mdl.opset_import:
                if ops.domain in ("", "ai.onnx"):
                    return int(ops.version)
            if mdl.opset_import:
                return int(mdl.opset_import[0].version)
        except Exception:
            pass
        return None

