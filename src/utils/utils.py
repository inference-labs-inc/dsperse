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
        # Filter input names from segment details
        segment_filtered_inputs = []
        for input_info in segment_inputs:
            # Only include actual inputs that are not weights or biases
            # Typically, weights and biases have names containing "weight" or "bias"
            if (not any(pattern in input_info.name.lower() for pattern in ["weight", "bias"]) and
                    input_info.name in [inp.name for inp in graph.input]):
                segment_filtered_inputs.append(input_info.name)
            # Also include intermediate tensors from previous layers
            elif input_info.name.startswith('/'):  # Intermediate tensors often start with '/'
                segment_filtered_inputs.append(input_info.name)
        # If there are no inputs after filtering, include the first non-weight/bias input
        if not segment_filtered_inputs:
            for input_info in segment_inputs:
                if not any(pattern in input_info.name.lower() for pattern in ["weight", "bias"]):
                    segment_filtered_inputs.append(input_info.name)
                    break

            # If still no inputs, use the first input as a fallback
            if not segment_filtered_inputs and segment_inputs:
                segment_filtered_inputs.append(segment_inputs[0].name)
        return segment_filtered_inputs

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
