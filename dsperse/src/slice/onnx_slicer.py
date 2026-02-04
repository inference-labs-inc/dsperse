import logging
import os
import os.path
from typing import List, Dict, Tuple

import onnx

from dsperse.src.analyzers.onnx_analyzer import OnnxAnalyzer
from dsperse.src.slice.autotiler import Autotiler
from dsperse.src.slice.utils.onnx_utils import OnnxUtils

logger = logging.getLogger(__name__)


class OnnxSlicer:
    def __init__(self, onnx_path, save_path=None):
        self.onnx_path = onnx_path
        self.save_path = save_path
        self.onnx_model = onnx.load(onnx_path)
        
        self.traced_shapes = None
        self.traced_dtypes = None
        self.onnx_analyzer = None
        self.analysis = None
        self.slice_points = None

    def _ensure_analysis(self):
        """Lazy initialization of shape tracing and model analysis."""
        if self.analysis is not None:
            return

        logger.info("Initializing analysis and tracing shapes...")
        self.traced_shapes, self.traced_dtypes = OnnxUtils.trace_shapes(self.onnx_model)
        OnnxUtils.apply_traced_shapes(self.onnx_model, self.traced_shapes, self.traced_dtypes)

        self.onnx_analyzer = OnnxAnalyzer(self.onnx_model, onnx_path=self.onnx_path)
        self.analysis = self.onnx_analyzer.analyze(save_path=self.save_path)

    def determine_slice_points(self, model_metadata=None, tile_size:int = None, isolate_convolutions=True) -> List[int]:
        """
        Determine the slice points for the model based on nodes with parameter_details in the model_metadata.

        Args:
            model_metadata: The model analysis metadata containing node information.
            tile_size: If True, optimize slicing for tiling.
            isolate_convolutions: If True, each Conv gets its own isolated slice.

        Returns:
            List[int]: List of indices representing nodes with parameter details
        """
        if model_metadata is None:
            self._ensure_analysis()
            model_metadata = self.analysis

        slice_points = set()
        max_idx = max((n.get("index", 0) for n in model_metadata["nodes"].values()), default=0)
        for _node_name, node_info in model_metadata["nodes"].items():
            if node_info.get("parameter_details") and node_info["parameter_details"]:
                idx = node_info["index"]
                slice_points.add(idx)
                if isolate_convolutions and node_info.get("node_type") == "Conv":
                    if idx + 1 <= max_idx:
                        slice_points.add(idx + 1)

        print(f"Original slice points: {sorted(slice_points)}")
        if isolate_convolutions:
            slice_points = OnnxUtils.isolate_conv(slice_points, model_metadata)
        slice_points = OnnxUtils.optimize_jstprove_slices(list(slice_points), model_metadata)

        if tile_size:
            slice_points = OnnxUtils.optimize_for_tiling(slice_points, model_metadata)

        slice_points = OnnxUtils.filter_constant_only_slices(slice_points, model_metadata)
        slice_points = sorted(set(slice_points))

        slice_points = OnnxUtils.complete_slice_points(list(slice_points), model_metadata)
        print(f"Optimized slice points: {slice_points}")

        self.slice_points = slice_points
        return slice_points

    def _slice_setup(self, model_metadata, output_path=None):
        output_path = os.path.join(os.path.dirname(self.onnx_path), "slices") if output_path is None else output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        graph = self.onnx_model.graph

        node_map, node_type_index_map, initializer_map, value_info_map = OnnxUtils.build_graph_maps(graph)
        index_to_node_name, index_to_segment_name = OnnxUtils.build_metadata_maps(model_metadata)

        return (graph, node_map, node_type_index_map, initializer_map, value_info_map,
                index_to_node_name, index_to_segment_name, output_path)

    def slice(self, slice_points: List[int], model_metadata, output_path=None, tile_size: int = None):
        """
        Slice the ONNX model based on the provided slice points.

        All operations work from the in-memory model graph. Slices are built
        directly from graph nodes/initializers and only written to disk once
        after finalization (shape application + validation).

        Args:
            slice_points: List of indices representing nodes with parameter details
            model_metadata: The model analysis metadata containing node information
            output_path: The path to save the slices to
            tile_size: If provided, apply tiling to the slices

        Returns:
            Tuple[Dict[int, str], Dict, Any]: Paths to sliced models, tiling metadata, and tensor graph
        """
        if not slice_points:
            raise ValueError("No slice points provided.")

        if not model_metadata or "nodes" not in model_metadata:
            raise ValueError("Invalid model metadata. Please run 'analyze()' first.")

        self.onnx_model = OnnxUtils.apply_symbolic_shape_inference(self.onnx_model)
        tensor_graph = OnnxUtils.initialize_tensor_graph(self.onnx_model)

        (graph, node_map, node_type_index_map, initializer_map, _value_info_map,
         index_to_node_name, index_to_segment_name, output_path) = self._slice_setup(model_metadata, output_path)

        segment_inputs_map = OnnxUtils.analyze_future_dependencies(
            slice_points, index_to_node_name, index_to_segment_name,
            node_map, node_type_index_map, initializer_map
        )

        segment_data = OnnxUtils.prepare_segments(
            slice_points, output_path, index_to_node_name, index_to_segment_name,
            node_map, node_type_index_map, initializer_map, graph, segment_inputs_map
        )

        abs_paths_dict = OnnxUtils.build_and_save_slices(
            segment_data, self.traced_shapes, self.traced_dtypes
        )

        tiled_info = {}
        if tile_size:
            logger.info(f"Applying tiling transform with tile_size={tile_size}")
            tiled_info = Autotiler.apply_tiling(abs_paths_dict, tile_size)

        return abs_paths_dict, tiled_info, tensor_graph


    def slice_model(self, output_path=None, tile_size: int = None):
        """
        Run the complete workflow: determine slice points, slice, and optionally tile.

        Args:
            output_path: The path to save the slices to.
            tile_size: Maximum elements per tile. Tile size is calculated dynamically
                           per-Conv based on channel count: tile_size = sqrt(tile_size / channels).

        Returns:
            Dict[str, Any]: Metadata about the sliced model
        """
        self._ensure_analysis()
        slice_points = self.determine_slice_points(self.analysis, tile_size=tile_size)
        slices_paths, tiled_info, _tensor_graph = self.slice(slice_points, self.analysis, output_path, tile_size=tile_size)
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
    onnx_slicer.slice_model(output_path=output_dir, tile_size=1000)
