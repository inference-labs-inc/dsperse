import os
import os.path
from pathlib import Path
import onnx
from onnx import shape_inference
import logging
from dsperse.src.analyzers.onnx_analyzer import OnnxAnalyzer
from dsperse.src.backends.jstprove import JSTPROVE_SUPPORTED_OPS
from typing import List, Dict
from dsperse.src.utils.utils import Utils
from dsperse.src.slice.autotiler import autotile_slice
from onnx.utils import extract_model
from onnxruntime.tools import symbolic_shape_infer

logger = logging.getLogger(__name__)


class OnnxSlicer:
    def __init__(self, onnx_path, save_path=None):
        self.onnx_path = onnx_path
        self.onnx_model = onnx.load(onnx_path)
        self.model_metadata = None
        self.slice_points = None

        print("Applying shape inference to original model for better slicing...")
        try:
            self.onnx_model = shape_inference.infer_shapes(self.onnx_model)
            print("Shape inference applied successfully to original model")
        except Exception as e:
            print(f"Shape inference failed on original model: {e}, continuing with original model")

        self.onnx_analyzer = OnnxAnalyzer(self.onnx_path)
        self.analysis = self.onnx_analyzer.analyze(save_path=save_path)

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
    def maximize_jstprove_slices(slice_points: List[int], model_metadata: Dict) -> List[int]:
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
                updated_points.add(next_node.get("index"))

        max_idx = max((n.get("index", 0) for n in nodes_dict.values()), default=0)
        return [p for p in updated_points if p <= max_idx]

    def determine_slice_points(self, model_metadata) -> List[int]:
        slice_points = []
        for node_name, node_info in model_metadata["nodes"].items():
            if node_info.get("parameter_details") and node_info["parameter_details"]:
                slice_points.append(node_info["index"])

        print(f"Original slice points: {slice_points}")
        slice_points = self.maximize_jstprove_slices(slice_points, model_metadata)
        slice_points.sort()

        print(f"jstprove optimized slice points: {slice_points}")

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

        segment_node_outputs = set()
        for node in segment_nodes:
            for output in node.output:
                segment_node_outputs.add(output)

        segment_node_inputs = set()
        for node in segment_nodes:
            for inp in node.input:
                segment_node_inputs.add(inp)

        for inp in segment_node_inputs:
            if inp not in segment_node_outputs:
                if inp in all_value_infos:
                    segment_inputs.append(all_value_infos[inp])
                elif inp in initializer_map:
                    init = initializer_map[inp]
                    segment_initializers.append(init)
                    t = onnx.helper.make_tensor_value_info(
                        inp,
                        init.data_type,
                        list(init.dims)
                    )
                    segment_inputs.append(t)
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

    def slice(self, slice_points: List[int], model_metadata, output_path=None, tile_size: int = None, parallel: bool = False):
        """
        Slice the ONNX model based on the provided slice points.

        Args:
            slice_points: List of indices representing nodes with parameter details
            model_metadata: The model analysis metadata containing node information
            output_path: The path to save the slices to
            tile_size: If set, tile Conv slices with spatial dims > tile_size
            parallel: If True, parallelize post-processing

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

        (graph, node_map, node_type_index_map, initializer_map, value_info_map,
         index_to_node_name, index_to_segment_name, output_path) = self._slice_setup(model_metadata, output_path)

        max_index = max(node_info["index"] for node_info in model_metadata["nodes"].values())
        if max_index + 1 not in slice_points:
            slice_points.append(max_index + 1)

        slice_points.sort()

        segment_inputs_map = {}
        for i in range(len(slice_points)):
            seg_idx = i - 1
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

        slice_paths = []
        tiled_info = {}

        for i in range(len(slice_points)):
            segment_idx = i - 1
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
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            payload_dir = os.path.join(save_path, "payload")
            os.makedirs(payload_dir, exist_ok=True)
            file_path = os.path.join(payload_dir, f"slice_{segment_idx}.onnx")

            input_names = Utils.filter_inputs(segment_inputs, graph)
            output_names = [output_info.name for output_info in segment_outputs]

            try:
                logger.info(f"Extracting slice {segment_idx}: {input_names} -> {output_names}")
                print(f"Extracting slice {segment_idx}: {input_names} -> {output_names}")
                extract_model(
                    input_path=self.onnx_path,
                    output_path=file_path,
                    input_names=input_names,
                    output_names=output_names
                )

                try:
                    extracted_model = onnx.load(file_path)
                    extracted_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(extracted_model)
                    extracted_model = self._concretize_symbolic_dims(extracted_model, value=1)
                    onnx.save(extracted_model, file_path)
                    logger.info(f"Shape inference applied successfully to extracted slice {segment_idx}")
                except Exception as e:
                    logger.warning(f"Shape inference failed on extracted slice {segment_idx}: {e}")
                    print(f"Shape inference failed on extracted slice {segment_idx}: {e}")

                slice_paths.append(file_path)

            except Exception as e:
                try:
                    logger.info(f"Error extracting slice, trying to create it instead {segment_idx}: {e}")
                    print(f"Error extracting slice, trying to create it instead {segment_idx}: {e}")
                    segment_graph = onnx.helper.make_graph(
                        segment_nodes,
                        f"segment_{segment_idx}_graph",
                        segment_inputs,
                        segment_outputs,
                        segment_initializers
                    )

                    segment_model = onnx.helper.make_model(segment_graph)

                    try:
                        segment_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(segment_model)
                        logger.info(f"Shape inference applied successfully to segment {segment_idx}")
                    except Exception as e:
                        logger.warning(f"Shape inference failed on segment {segment_idx}: {e}")
                        print(f"Shape inference failed on segment {segment_idx}: {e}")

                    segment_model = self._concretize_symbolic_dims(segment_model, value=1)
                    onnx.save(segment_model, file_path)
                    slice_paths.append(file_path)

                except Exception as e:
                    logger.error(f"Error creating segment {segment_idx}: {e}")
                    continue

            if tile_size is not None and os.path.exists(file_path):
                try:
                    tiles_dir = os.path.join(payload_dir, "tiles")
                    info = autotile_slice(segment_idx, Path(file_path), tile_size, Path(tiles_dir), nested=True, parallel=parallel)
                    if info:
                        tiled_info[segment_idx] = info
                        logger.info(f"Tiled slice {segment_idx} successfully")
                        print(f"  -> Tiled successfully: {info['num_tiles']} tiles")
                except Exception as e:
                    logger.error(f"Error tiling slice {segment_idx}: {e}")
                    print(f"  -> Tiling failed: {e}")

        abs_paths = self.slice_post_process(slice_paths, parallel=parallel)
        return abs_paths, tiled_info

    @staticmethod
    def _process_single_slice(path: str) -> str | None:
        abs_path = os.path.abspath(path)
        try:
            model = onnx.load(path)
            logger.info(f"Applying shape inference to {path}")
            try:
                model_with_shapes = shape_inference.infer_shapes(model)
                model = model_with_shapes
                logger.info(f"Shape inference successful for {path}")
                print(f"Shape inference successful for {path}")
            except Exception as shape_error:
                logger.warning(f"Shape inference failed for {path}: {shape_error}")
                print(f"Shape inference failed for {path}: {shape_error}")
            model = OnnxSlicer._concretize_symbolic_dims(model, value=1)
            onnx.checker.check_model(model)
            onnx.save(model, path)
            logger.info(f"Successfully processed and saved {path}")
            return abs_path
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None

    @staticmethod
    def slice_post_process(slices_paths, parallel: bool = False):
        """
        Post-process sliced models with shape inference and validation.
        """
        if parallel and len(slices_paths) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            max_workers = min(len(slices_paths), multiprocessing.cpu_count())
            abs_paths = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(OnnxSlicer._process_single_slice, p): p for p in slices_paths}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        abs_paths.append(result)
            return sorted(abs_paths)

        abs_paths = []
        for path in slices_paths:
            result = OnnxSlicer._process_single_slice(path)
            if result:
                abs_paths.append(result)
        return abs_paths

    def slice_model(self, output_path=None, tile_size: int = None, parallel: bool = False):
        """
        Run the complete workflow: determine slice points, slice, and optionally tile.

        Args:
            output_path: The path to save the slices to.
            tile_size: If set, tile Conv slices with spatial dims > tile_size.
            parallel: If True, parallelize operations.

        Returns:
            Dict[str, Any]: Metadata about the sliced model
        """
        slice_points = self.determine_slice_points(self.analysis)
        slices_paths, tiled_info = self.slice(slice_points, self.analysis, output_path, tile_size=tile_size, parallel=parallel)

        self.onnx_analyzer.generate_slices_metadata(self.analysis, slice_points, slices_paths, output_path, tiled_info)
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
    onnx_slicer.slice_model(output_path=output_dir, tile_size=16)
