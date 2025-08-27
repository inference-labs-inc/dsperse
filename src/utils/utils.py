import json
import logging
from pathlib import Path
import torch

# Configure logger
logger = logging.getLogger(__name__)


class Utils:
    """
    Utility functions for working with ONNX models.
    """

    @staticmethod
    def save_metadata_file(metadata, output_path, filename="metadata.json"):
        """
        Save metadata to a JSON file.

        Args:
            metadata: Dictionary containing metadata
            output_path: Directory where the metadata will be saved
            filename: Name of the metadata file (default: "metadata.json")
        """
        output = Path(output_path)

        # Check if the provided path is a directory
        if output.is_dir():
            # Combine the directory with the default or given filename
            file_path = output / filename
        else:
            # Use the path as-is, assuming it includes the filename
            file_path = output

        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write metadata to the file
        with file_path.open('w') as f:
            json.dump(metadata, f, indent=4)


    @staticmethod
    def filter_inputs(segment_inputs, graph):
        """
        Return the correct set of external input tensor names for ONNX extract_model.
        Rules:
        - Exclude initializers (weights/biases/constants) — extractor includes them automatically.
        - Prefer only names that have type/shape in the original model (present in graph.input or graph.value_info).
        - Preserve order and avoid duplicates.
        - Provide safe fallbacks so we never return an empty list.
        """
        # Names of model-level inputs and typed value_infos
        graph_input_names = [inp.name for inp in graph.input]
        graph_input_name_set = set(graph_input_names)
        typed_names = set(graph_input_names)
        typed_names.update({vi.name for vi in graph.value_info})
        # Names of initializers (weights/biases/statistics/constants)
        initializer_names = {init.name for init in graph.initializer}

        # Phase 1: collect candidate external inputs from provided segment_inputs
        # Exclude initializers
        candidates_in_order = []
        seen = set()
        for vi in segment_inputs:
            name = getattr(vi, 'name', None)
            if not name or name in seen:
                continue
            seen.add(name)
            if name in initializer_names:
                # Skip parameters; extractor will wire them automatically
                continue
            candidates_in_order.append(name)

        # Phase 2: keep only "typed" candidates (present in model inputs or value_info)
        typed_candidates = [n for n in candidates_in_order if n in typed_names]

        # Use typed candidates if available
        if typed_candidates:
            return typed_candidates

        # Phase 3: fallbacks
        # 3a) If the model has at least one input, use it as a conservative default
        if graph_input_names:
            return [graph_input_names[0]]

        # 3b) Use first non-initializer candidate if any
        if candidates_in_order:
            return [candidates_in_order[0]]

        # 3c) Last resort: if segment_inputs exist, return its first name
        if segment_inputs:
            first_name = getattr(segment_inputs[0], 'name', None)
            if first_name:
                return [first_name]

        # Should not happen, but return empty list if absolutely nothing is available
        return []

    @staticmethod
    def get_unfiltered_inputs(segment_inputs):
        """
        Return raw input tensor names from segment inputs for extract_model usage.
        This intentionally avoids filtering out weights/biases/etc.
        """
        segment_unfiltered_inputs = [getattr(inp, 'name', None) for inp in segment_inputs if getattr(inp, 'name', None)]
        if not segment_unfiltered_inputs and segment_inputs:
            segment_unfiltered_inputs = [segment_inputs[0].name]
        return segment_unfiltered_inputs

    @staticmethod
    def _get_original_model_shapes(model_metadata: dict):
        """
        Extract shape information from model metadata.

        Args:
            model_metadata: Dictionary containing model metadata with shape information

        Returns:
            dict: Dictionary mapping tensor names to their shapes
        """
        shapes = {}

        # Extract shapes from input_shape
        input_shape = model_metadata.get("input_shape", [])
        if input_shape and len(input_shape) > 0:
            shapes["input"] = input_shape[0]

        # Extract shapes from output_shapes
        output_shapes = model_metadata.get("output_shapes", [])
        if output_shapes and len(output_shapes) > 0:
            shapes["output"] = output_shapes[0]

        # Extract shapes from nodes if available
        nodes = model_metadata.get("nodes", {})
        for node_name, node_info in nodes.items():
            if "parameter_details" in node_info:
                for param_name, param_info in node_info["parameter_details"].items():
                    if "shape" in param_info:
                        shapes[param_name] = param_info["shape"]

        return shapes

    @staticmethod
    def write_input(tensor: torch.Tensor, file_path):
        """Write tensor to input.json format."""
        data = {"input_data": tensor.tolist()}
        with open(file_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def read_input(file_path) -> torch.Tensor:
        """Read tensor from input.json format."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return torch.tensor(data["input_data"])
