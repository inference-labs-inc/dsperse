import json
import logging
import os
from pathlib import Path
from typing import Optional
import time

import torch
import torch.nn.functional as F

from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.torch_utils import ModelUtils

logger = logging.getLogger(__name__)

class RunnerUtils:
    def __init__(self):
        pass

    # ----- Runtime helpers to keep Runner.run concise -----
    @staticmethod
    def normalize_for_runtime(run_metadata: dict, slices_path: Path) -> tuple[Path, str | None, Path]:
        packaging_type = (run_metadata or {}).get("packaging_type", "dirs")
        source_path = (run_metadata or {}).get("source_path") or str(slices_path)
        model_path = Path((run_metadata or {}).get("model_path", Path(slices_path).parent)).resolve()

        # Correct accidental model_path pointing to the slices folder
        #TODO: find a better way to do this
        if model_path.name == "slices":
            model_path = model_path.parent

        # Handle packaged inputs
        if packaging_type == "dsperse":
            try:
                converted = Converter.convert(source_path, output_type="dirs", cleanup=False)
                return Path(converted), "dsperse", model_path
            except Exception:
                return (model_path / "slices").resolve(), None, model_path

        if packaging_type == "dslice":
            try:
                sp = Path(source_path)
                if sp.is_file():
                    # Extract single .dslice file to a slice_* directory; use its parent as slices root
                    slice_dir = Path(Converter.convert(str(sp), output_type="dirs", cleanup=False))
                    slices_root = slice_dir.parent
                else:
                    # Directory containing .dslice files; expand into slice_* directories in place
                    expanded_dir = Path(Converter.convert(str(sp), output_type="dirs", cleanup=False))
                    slices_root = expanded_dir
                return slices_root, "dslice", model_path
            except Exception:
                return (model_path / "slices").resolve(), None, model_path

        # Default: dirs layout
        root = (model_path / "slices").resolve()
        if not root.exists():
            root = Path(slices_path)
        return root, None, model_path

    @staticmethod
    def make_run_dir(run_metadata: dict, output_path: str | None, model_path: Path) -> Path:
        base_run_dir = Path((run_metadata or {}).get("run_directory") or (model_path / "run"))
        if output_path:
            return Path(output_path)
        # If run_metadata already specified a timestamped run dir, reuse it
        if base_run_dir.name.startswith("run_"):
            return base_run_dir
        return base_run_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def prepare_slice_io(run_dir: Path, slice_id: str) -> tuple[Path, Path]:
        slice_run_dir = run_dir / slice_id
        slice_run_dir.mkdir(parents=True, exist_ok=True)
        in_file = slice_run_dir / "input.json"
        out_file = slice_run_dir / "output.json"
        return in_file, out_file

    @staticmethod
    def execute_slice(runner, node: dict, slice_info: dict, in_file: Path, out_file: Path, slice_dir: Path):
        # Respect optional global forced backend selection
        forced = getattr(runner, 'force_backend', None)
        if forced == 'onnx':
            return runner.run_onnx_slice(slice_info, in_file, out_file, slice_dir)

        if node.get("use_circuit"):
            preferred = (slice_info.get("backend") or node.get("backend") or "jstprove").lower()
            if forced in ("jstprove", "ezkl"):
                preferred = forced
            # Try JSTprove path first when selected/forced and available
            if preferred == "jstprove" and getattr(runner, "jstprove_runner", None):
                ok, tensor, j_info = runner._run_jstprove_slice(slice_info, in_file, out_file, slice_dir)
                if ok:
                    return ok, tensor, j_info
                # If forced JSTprove, do not switch to EZKL; go directly to ONNX
                if forced == 'jstprove':
                    ok, tensor, o_info = runner.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
                    o_info["method"] = "forced_jstprove_fallback_onnx"
                    o_info["attempted_jstprove"] = True
                    return ok, tensor, o_info
                # Otherwise fall back to EZKL then ONNX
                ok, tensor, e_info = runner._run_ezkl_slice(slice_info, in_file, out_file, slice_dir)
                if ok:
                    return ok, tensor, e_info
                ok, tensor, o_info = runner.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
                o_info["method"] = "jstprove_ezkl_fallback_onnx"
                o_info["attempted_jstprove"] = True
                o_info["attempted_ezkl"] = True
                if j_info.get("error") and not o_info.get("error"):
                    o_info["error"] = j_info.get("error")
                elif e_info.get("error") and not o_info.get("error"):
                    o_info["error"] = e_info.get("error")
                return ok, tensor, o_info

            # Otherwise, try EZKL first (legacy/default), then fallback to ONNX
            ok, tensor, ezkl_info = runner._run_ezkl_slice(slice_info, in_file, out_file, slice_dir)
            result_info = ezkl_info
            if not ok:
                # If forced EZKL, go directly to ONNX
                if forced == 'ezkl':
                    ok, tensor, onnx_info = runner.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
                    onnx_info["method"] = "forced_ezkl_fallback_onnx"
                    onnx_info["attempted_ezkl"] = True
                    return ok, tensor, onnx_info
                ok, tensor, onnx_info = runner.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
                onnx_info["method"] = "ezkl_fallback_onnx"
                onnx_info["attempted_ezkl"] = True
                if ezkl_info.get("error") and not onnx_info.get("error"):
                    onnx_info["error"] = ezkl_info.get("error")
                result_info = onnx_info
            return ok, tensor, result_info
        ok, tensor, onnx_info = runner.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
        onnx_info["attempted_ezkl"] = False
        return ok, tensor, onnx_info

    @staticmethod
    def _get_file_path() -> str:
        """Get the parent directory path of the current file."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def preprocess_input(input_path:str, model_directory: str = None, save_reshape: bool = False) -> torch.Tensor:
        """
        Preprocess input data from JSON.
        """

        if os.path.isfile(input_path):
            with open(input_path, 'r') as f:
                input_data = json.load(f)
        else:
            input_path = os.path.join(RunnerUtils._get_file_path(), input_path)
            print(
                f"Warning: Input file not found. Trying to use relative path: {input_path} instead."
            )
            with open(input_path, 'r') as f:
                input_data = json.load(f)

        if isinstance(input_data, dict):
            if 'input_data' in input_data:
                input_data = input_data['input_data']
            elif 'input' in input_data:
                input_data = input_data['input']

        if isinstance(input_data, list) and len(input_data) == 0:
            raise ValueError("Input data list is empty")

        # Convert to tensor
        if isinstance(input_data, list):
            if isinstance(input_data[0], list):
                # 2D input
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                # 1D input
                input_tensor = torch.tensor([input_data], dtype=torch.float32)
        else:
            raise ValueError("Expected input data to be a list or nested list")

        return input_tensor
        
        
    @staticmethod
    def process_final_output(torch_tensor):
        """Process the final output of the model."""
        # Apply softmax to get probabilities if not already applied
        if len(torch_tensor.shape) != 2:  # Ensure raw output is 2D [batch_size, num_classes]
            logger.debug(f"Warning: Raw output shape {torch_tensor.shape} is not as expected. Reshaping to [1, -1].")
            torch_tensor = torch_tensor.reshape(1, -1)

        probabilities = F.softmax(torch_tensor, dim=1)
        predicted_action = torch.argmax(probabilities, dim=1).item()

        result = {
            "logits": torch_tensor,
            "probabilities": probabilities,
            "predicted_action": predicted_action
        }

        return result

    @staticmethod
    def get_segments(slices_directory):
        metadata = ModelUtils.load_metadata(slices_directory)
        if metadata is None:
            return None

        segments = metadata.get('slices', [])
        if not segments:
            print("No segments found in metadata.json")
            return None

        return segments

    @staticmethod
    def save_to_file_shaped(input_tensor: torch.Tensor, file_path: str):
        # Convert tensor to list
        tensor_data = input_tensor.tolist()

        # Create directory if it doesn't exist
        file_dir = os.path.dirname(file_path)
        if file_dir:  # Only create directory if path has a directory component
            os.makedirs(file_dir, exist_ok=True)

        # Save tensor data as JSON
        data = {
            "input": tensor_data
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def save_to_file_flattened(input_tensor: torch.Tensor, file_path: str):
        # Flatten and convert tensor to list
        tensor_data = input_tensor.flatten().tolist()

        # Create directory if it doesn't exist
        file_dir = os.path.dirname(file_path)
        if file_dir:  # Only create directory if path has a directory component
            os.makedirs(file_dir, exist_ok=True)

        # Save flattened tensor data as JSON
        data = {
            "input_data": [tensor_data]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f)


    @staticmethod
    def _is_sliced_model(model_path: str) -> tuple[bool, Optional[str]]:
        """
        Check if the path is a sliced model (dirs, dslice, or dsperse format).

        Returns:
            Tuple of (is_sliced, slice_path) where slice_path is the actual path to the slices
        """
        path_obj = Path(model_path)

        # Check for compressed slice formats (direct file)
        if path_obj.is_file() and path_obj.suffix in ['.dsperse', '.dslice']:
            return True, str(path_obj)

        # Check for directory formats
        if path_obj.is_dir():
            # Check if directory contains a .dsperse file
            dsperse_files = [f for f in path_obj.iterdir() if f.is_file() and f.suffix == '.dsperse']
            if dsperse_files:
                return True, str(dsperse_files[0])

            # Check if directory contains a 'slices' subdirectory
            slices_subdir = path_obj / 'slices'
            if slices_subdir.is_dir():
                return True, str(slices_subdir)

            # Check using Converter's detect_type
            try:
                detected_type = Converter.detect_type(path_obj)
                if detected_type in ['dirs', 'dslice', 'dsperse']:
                    return True, str(path_obj)
            except ValueError:
                pass

        return False, None

    @staticmethod
    def filter_tensor(current_slice_metadata, tensor):
        # take the tensor object, and extract the output that is relevant to the next slice
        logits = tensor["logits"]

        # Check the shape using our new function
        output_shape = current_slice_metadata.get("output_shape")
        if output_shape is not None:
            RunnerUtils.check_expected_shape(logits, output_shape, tensor_name="logits")
        else:
            logger.debug("Output shape metadata not found for shape check.")

        return logits

    @staticmethod
    def check_expected_shape(tensor, expected_shape_data, tensor_name="tensor"):
        """
        Check if the tensor shape matches the expected shape from metadata.

        Args:
            tensor: The PyTorch tensor to check
            expected_shape_data: The shape data from metadata (usually a nested list with possible string placeholders)
            tensor_name: Name of the tensor for logging purposes

        Returns:
            bool: True if shapes match, False otherwise
        """
        # Handle the case where output_shape is a nested list
        if isinstance(expected_shape_data, list) and len(expected_shape_data) > 0:
            # Extract the inner shape list - the first element of output_shape
            shape_values = expected_shape_data[0]

            # Replace string placeholders with actual values from tensor
            expected_elements = 1
            shape_dict = {
                "batch_size": tensor.shape[0] if tensor.dim() > 0 else 1,
                "unk__0": tensor.shape[0] if tensor.dim() > 0 else 1
            }

            # Build the expected shape with placeholders replaced
            expected_shape = []
            for dim in shape_values:
                if isinstance(dim, str):
                    if dim in shape_dict:
                        expected_shape.append(shape_dict[dim])
                        expected_elements *= shape_dict[dim]
                    else:
                        logger.warning(f"Unknown dimension placeholder: {dim}")
                        # Default to using 1 for unknown dimensions
                        expected_shape.append(1)
                        expected_elements *= 1
                else:
                    expected_shape.append(dim)
                    expected_elements *= dim

            # Check total elements
            tensor_elements = torch.numel(tensor)
            if tensor_elements != expected_elements:
                logger.warning(
                    f"{tensor_name} shape {list(tensor.shape)} has {tensor_elements} elements, "
                    f"but expected shape {expected_shape} has {expected_elements} elements"
                )
                return False

            # If the tensor is flattened but should be multidimensional
            if len(tensor.shape) == 1 and len(expected_shape) > 1:
                logger.info(
                    f"{tensor_name} is flattened ({tensor.shape[0]} elements), "
                    f"but expected shape is {expected_shape}"
                )
                return True

            # Check actual dimensions if tensor is not flattened
            if len(tensor.shape) == len(expected_shape):
                for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                    if actual != expected:
                        logger.warning(
                            f"Dimension mismatch at index {i}: {tensor_name} has size {actual}, "
                            f"expected {expected}"
                        )
                        return False
                return True

        # If we can't determine expected shape, just return True
        logger.debug(f"Could not determine precise expected shape for {tensor_name}")
        return True



if __name__ == "__main__":
    print(f"Parent path: {RunnerUtils._get_file_path()}")
