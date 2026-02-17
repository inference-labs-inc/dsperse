import logging
import os
from pathlib import Path
from typing import Dict, Any

import onnx

from dsperse.src.analyzers.schema import (
    SliceMetadata, TensorShape, Dependencies, ModelMetadata
)
from dsperse.src.slice.utils.onnx_utils import OnnxUtils
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class OnnxAnalyzer:
    """
    A class for analyzing ONNX models and generating metadata.
    """

    def __init__(self, model: onnx.ModelProto | str, onnx_path: str | None = None):
        """
        Initialize the OnnxAnalyzer with either an ONNX model or a path to an ONNX model.
        """
        if isinstance(model, str):
            self.onnx_path = os.path.abspath(model)
            self.onnx_model = onnx.load(self.onnx_path)
        else:
            self.onnx_path = os.path.abspath(onnx_path) if onnx_path else None
            self.onnx_model = model

        self.model_metadata = None

    def analyze(self, save_path:str = None) -> Dict[str, Any]:
        """
        Analyze the ONNX model and generate comprehensive metadata.

        Returns:
            Dict[str, Any]: Comprehensive metadata about the ONNX model
        """
        # Extract model metadata
        graph = self.onnx_model.graph

        # Create maps for initializers and value info
        initializer_map = {init.name: init for init in graph.initializer}

        model_input_shape = self._get_model_input_shapes(graph, initializer_map)
        model_output_shape = self._get_model_output_shapes(graph)

        # Store node metadata
        node_metadata = {}

        # Process each node to collect metadata
        for i, node in enumerate(graph.node):
            # Analyze the node and store metadata
            node_info = self.analyze_node(node, i, initializer_map)
            # Use index-based key to handle empty node names
            node_key = node.name if node.name else f"{node.op_type}_{i}"
            node_metadata[node_key] = node_info

        # Determine opset information
        opset_imports_list = []
        default_opset_version = None
        try:
            for opset in self.onnx_model.opset_import:
                domain = opset.domain
                version = int(opset.version)
                opset_imports_list.append({"domain": domain if domain else "ai.onnx", "version": version})
                # Prefer default domain ""; fallback to explicit "ai.onnx"
                if default_opset_version is None and (domain == "" or domain == "ai.onnx"):
                    default_opset_version = version
        except Exception:
            default_opset_version = None

        # Warn (but continue) if default opset is below 18
        if default_opset_version is not None and default_opset_version < 18:
            model_label = self.onnx_path or "<in-memory model>"
            msg = (
                f"ONNX opset {default_opset_version} detected for model {model_label}. "
                "Opset < 18 is not officially supported; continuing anyway."
            )
            logger.warning(msg)

        # Create model metadata
        model_metadata = {
            "original_model": self.onnx_path,
            "model_type": "ONNX",
            "node_count": len(graph.node),
            "initializer_count": len(graph.initializer),
            "input_shape": model_input_shape,
            "output_shapes": model_output_shape,
            "opset_version": default_opset_version,
            "opset_imports": opset_imports_list,
            "nodes": node_metadata
        }

        # Save model metadata
        if save_path is not None:
            # output_dir = os.path.join(os.path.dirname(self.onnx_path), "onnx_analysis")
            if Path(save_path).is_dir():
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, "model_metadata.json")

            Utils.save_metadata_file(model_metadata, save_path)
        self.model_metadata = model_metadata

        return model_metadata

    def analyze_node(self, node, index, initializer_map):
        """
        Analyze a single node from the ONNX graph and gather metadata.

        Args:
            node: ONNX node to analyze
            index: Index of the node in the graph
            initializer_map: Map of initializer names to initializers

        Returns:
            dict: Metadata for the node
        """
        node_inputs = list(node.input)
        node_outputs = list(node.output)

        _, parameter_details = self._get_parameter_info(node, node_inputs, initializer_map)

        node_type = node.op_type

        return {
            "index": index,
            "slice_name": f"{node_type}_{index}",
            "node_type": node_type,
            "parameter_details": parameter_details,
            "dependencies": {
                "input": node_inputs,
                "output": node_outputs
            }
        }

    @staticmethod
    def _get_model_input_shapes(graph, initializer_map):
        """
        Extract input shapes from the model graph.

        Args:
            graph: ONNX model graph
            initializer_map: Map of initializer names to initializers

        Returns:
            list: List of input shapes
        """
        model_input_shapes = []
        for input_info in graph.input:
            if input_info.name not in initializer_map:  # Skip initializers (weights)
                shape = []
                if input_info.type.tensor_type.shape.dim:
                    for dim in input_info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value if dim.dim_value != 0 else None)
                model_input_shapes.append(shape)
        return model_input_shapes

    @staticmethod
    def _get_model_output_shapes(graph):
        """
        Extract output shapes from the model graph.

        Args:
            graph: ONNX model graph

        Returns:
            list: List of output shapes
        """
        model_output_shapes = []
        for output_info in graph.output:
            shape = []
            if output_info.type.tensor_type.shape.dim:
                for dim in output_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value if dim.dim_value != 0 else None)
            model_output_shapes.append(shape)
        return model_output_shapes

    @staticmethod
    def _get_parameter_info(node, node_inputs, initializer_map):
        """
        Determine parameter information for a node.

        Args:
            node: ONNX node
            node_inputs: List of node inputs
            initializer_map: Map of initializer names to initializers

        Returns:
            tuple: ( parameters, parameter_details)
        """
        # Calculate parameters if possible
        parameters = 0
        parameter_details = {}

        # For Conv, Gemm, and MatMul nodes, we can extract parameter information from initializers
        if node.op_type in ["Conv", "Gemm", "MatMul"]:
            for inp in node_inputs:
                if inp in initializer_map:
                    init = initializer_map[inp]
                    # Calculate size (number of elements)
                    size = 1
                    for dim in init.dims:
                        size *= dim

                    # Add to total parameters
                    parameters += size

                    # Store parameter details
                    parameter_details[inp] = {
                        "shape": list(init.dims),
                        "size": size
                    }

        return parameters, parameter_details

    def generate_slices_metadata(self, model_metadata, slice_points, slices_paths, output_dir=None, tiled_info=None):
        """
        Generate metadata for sliced ONNX models.

        Args:
            model_metadata: The model analysis metadata containing node information
            slice_points: List of indices representing nodes with parameter details
            output_dir: Directory where the metadata will be saved
            slices_paths: Paths to sliced onnx files
            tiled_info: Dict of {slice_idx: tiling_info} from autotiler

        Returns:
            dict: Complete metadata for the sliced models
        """
        if output_dir is None and not self.onnx_path:
            raise ValueError("output_dir is required when analyzer is initialized without onnx_path")
        model_overview = self._get_model_metadata(model_metadata, slice_points)
        tiled_info = tiled_info or {}

        segments = []

        for i in range(len(slice_points)):
            segment_idx = i - 1
            if segment_idx < 0:
                continue

            start_idx = slice_points[i - 1] if i > 0 else 0
            end_idx = slice_points[i]

            if start_idx == end_idx:
                continue

            slice_path = slices_paths.get(segment_idx) if slices_paths else None

            segment_metadata = self._get_segment_metadata(
                model_metadata,
                segment_idx,
                start_idx,
                end_idx,
                slice_path,
                output_dir
            )
            if segment_metadata:
                if segment_idx in tiled_info:
                    info = tiled_info[segment_idx]
                    if "tiling" in info:
                        segment_metadata["tiling"] = Utils.relativize_tiling_info(info["tiling"], output_dir)
                    if "channel_split" in info:
                        segment_metadata["channel_split"] = Utils.relativize_tiling_info(info["channel_split"], output_dir)
                segments.append(segment_metadata)

        # Add segments to metadata
        model_overview["slices"] = segments

        # Save metadata if output_dir is provided
        Utils.save_metadata_file(model_overview, output_path=output_dir)
        OnnxUtils.write_slice_dirs_metadata(output_dir)

        return model_overview

    def _get_model_metadata(self, model_metadata, slice_points):
        """
        Get model-level metadata.

        Args:
            model_metadata: The model analysis metadata containing node information
            slice_points: List of indices representing nodes with parameter details

        Returns:
            dict: Model-level metadata
        """
        return {
            "original_model": model_metadata["original_model"],
            "model_type": model_metadata["model_type"],
            "input_shape": model_metadata["input_shape"],
            "output_shapes": model_metadata["output_shapes"],
            "slice_points": slice_points[:-1]
        }

    def _get_segment_metadata(self, model_metadata, segment_idx, start_idx, end_idx, slice_path, output_dir=None):
        """
        Get metadata for a specific segment.

        Args:
            model_metadata: The model analysis metadata containing node information
            segment_idx: Index of the segment
            start_idx: Start index of the segment
            end_idx: End index of the segment
            slice_path: Path to the sliced ONNX model

        Returns:
            dict: Segment metadata
        """
        # Collect nodes for this segment
        segment_nodes = []
        for idx in range(start_idx, end_idx):
            for node_name, node_info in model_metadata["nodes"].items():
                if node_info["index"] == idx:
                    segment_nodes.append((node_name, node_info))

        if not segment_nodes:
            return None

        dependencies = self._get_segment_dependencies(model_metadata, start_idx, end_idx)
        shape = self._get_segment_shape(slice_path)

        output_dir = os.path.join(output_dir, f"slice_{segment_idx}") if output_dir else os.path.join(os.path.dirname(self.onnx_path), "slices", f"slice_{segment_idx}")
        payload_dir = os.path.join(output_dir, "payload")
        os.makedirs(payload_dir, exist_ok=True)
        segment_filename = f"slice_{segment_idx}.onnx"
        segment_path = os.path.abspath(os.path.join(payload_dir, segment_filename))

        metadata = SliceMetadata(
            index=segment_idx,
            filename=segment_filename,
            path=segment_path,
            shape=shape,
            dependencies=dependencies,
        )

        return metadata.to_dict()

    def _get_segment_dependencies(self, model_metadata, start_idx, end_idx) -> Dependencies:
        inputs = []
        outputs = []
        output_map = {}

        for idx in range(start_idx, end_idx):
            for node_name, node_info in model_metadata['nodes'].items():
                if node_info['index'] == idx:
                    for output in node_info['dependencies']['output']:
                        output_map[output] = True

                    for input_name in node_info['dependencies']['input']:
                        if input_name not in output_map:
                            if input_name not in inputs:
                                inputs.append(input_name)

        for output in output_map:
            if output not in inputs:
                outputs.append(output)

        initializer_patterns = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
        initializer_names = {init.name for init in self.onnx_model.graph.initializer}
        model_input_names = {inp.name for inp in self.onnx_model.graph.input if inp.name not in initializer_names}

        filtered = []
        for input_name in inputs:
            name_lower = input_name.lower()
            if any(pattern in name_lower for pattern in initializer_patterns):
                continue
            if input_name in model_input_names or input_name not in initializer_names:
                filtered.append(input_name)

        if not filtered and inputs:
            filtered.append(inputs[0])

        return Dependencies(input=inputs, output=outputs, filtered_inputs=filtered)

    @staticmethod
    def _get_segment_shape(slice_path) -> TensorShape:
        return OnnxAnalyzer._get_tensor_shape(slice_path)

    @staticmethod
    def _get_tensor_shape(slice_path) -> TensorShape:
        inputs = []
        outputs = []

        if slice_path:
            onnx_model = onnx.load(slice_path)
            graph = onnx_model.graph
            for init in graph.initializer:
                inputs.append(list(init.dims))
            for inp in graph.input:
                dimensions = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField('dim_param'):
                        dimensions.append(dim.dim_param)
                    else:
                        dimensions.append(dim.dim_value)
                inputs.append(dimensions)
            for out in graph.output:
                dimensions = []
                for dim in out.type.tensor_type.shape.dim:
                    if dim.HasField('dim_param'):
                        dimensions.append(dim.dim_param)
                    else:
                        dimensions.append(dim.dim_value)
                outputs.append(dimensions)

        return TensorShape(input=inputs, output=outputs)


