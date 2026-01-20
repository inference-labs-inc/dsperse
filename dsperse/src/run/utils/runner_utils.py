import json
import logging
import os
from pathlib import Path
from typing import Optional, Union
import time

import torch
import torch.nn.functional as F

from dsperse.src.metadata.schema import RunSliceMetadata
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
    def resolve_relative_path(p: str, base_dir: Path) -> Optional[str]:
        if not p:
            return None
        p_str = str(p)
        if os.path.isabs(p_str):
            return p_str

        # Handle the case where p might start with the name of base_dir
        # (e.g. base_dir='/.../slice_0', p='slice_0/payload/...')
        sd_name = os.path.basename(os.path.abspath(str(base_dir))) if base_dir else None
        parts = p_str.split(os.sep)
        if sd_name and parts and parts[0] == sd_name:
            parts = parts[1:]
            p_str = os.path.join(*parts) if parts else ''

        if not base_dir:
            return os.path.abspath(p_str)

        path = (base_dir / p_str).resolve()
        if path.exists():
            return str(path)

        # Fallback to sibling/parent if it doesn't exist (legacy behavior in some tests/backends)
        alt_path = (base_dir.parent / p_str).resolve()
        if alt_path.exists():
            return str(alt_path)

        return str(path)

    @staticmethod
    def run_onnx_slice(slice_info: dict, input_tensor_path, output_tensor_path, slice_dir: Path = None):
        """Run ONNX inference for a slice.
        Accepts `slice_info['path']` possibly as `slice_#/payload/...` or absolute; resolves under `slice_dir` when provided.
        """
        from dsperse.src.backends.onnx_models import OnnxModels
        onnx_path = slice_info.get("path")
        if not onnx_path:
            return False, "No ONNX path in slice_info", {'success': False, 'error': 'missing_path', 'method': 'onnx_only'}

        if not os.path.isabs(str(onnx_path)):
            onnx_path = RunnerUtils.resolve_relative_path(onnx_path, slice_dir)

        if not onnx_path or not Path(onnx_path).exists():
            return False, f"ONNX file not found: {onnx_path}", {'success': False, 'error': 'file_not_found', 'method': 'onnx_only'}

        start_time = time.time()
        success, result = OnnxModels.run_inference(model_path=onnx_path, input_file=input_tensor_path, output_file=output_tensor_path)

        end_time = time.time()
        exec_info = {
            'success': success,
            'method': 'onnx_only',
            'execution_time': end_time - start_time,
            'output_tensor_path': str(output_tensor_path)
        }

        if success:
            exec_info['input_file'] = str(input_tensor_path.resolve())
            exec_info['output_file'] = str(output_tensor_path.resolve())
        else:
            exec_info['error'] = result if isinstance(result, str) else 'inference_failed'

        return success, result, exec_info

    @staticmethod
    def run_onnx_multi_input_slice(slice_info: dict, output_file: Path, slice_dir: Path, extra_tensors: dict):
        """Run ONNX inference for a multi-input slice."""
        from dsperse.src.backends.onnx_models import OnnxModels
        onnx_path = slice_info.get("path")
        if not onnx_path:
            return False, "No ONNX path in slice_info", {'error': 'missing_path'}

        onnx_path = RunnerUtils.resolve_relative_path(onnx_path, slice_dir)

        if not onnx_path or not Path(onnx_path).exists():
            return False, f"ONNX file not found: {onnx_path}", {'error': 'file_not_found'}

        start_time = time.time()
        try:
            success, result = OnnxModels.run_inference_multi(
                model_path=onnx_path,
                extra_tensors=extra_tensors,
                output_file=output_file
            )
        except Exception as e:
            success, result = False, str(e)

        exec_info = {'success': success, 'method': 'onnx_multi_input', 'execution_time': time.time() - start_time}
        if not success:
            exec_info['error'] = result if isinstance(result, str) else 'unknown'
        return success, result, exec_info

    @staticmethod
    def execute_slice(runner, node: dict, slice_info: dict, in_file: Path, out_file: Path, slice_dir: Path):
        """Execute a slice using best available backend: jstprove -> ezkl -> onnx."""
        meta = RunSliceMetadata.from_dict(slice_info)
        slice_id = slice_info.get("index", slice_dir.name if slice_dir else "unknown")
        forced = getattr(runner, 'force_backend', None)

        has_jst = bool(meta.jstprove_circuit_path) and getattr(runner, "jstprove_runner", None)
        has_ezkl = bool(meta.ezkl_circuit_path)

        available = []
        if has_jst:
            available.append("jstprove")
        if has_ezkl:
            available.append("ezkl")
        available.append("onnx")

        if forced == 'onnx':
            logger.info(f"[{slice_id}] Running with ONNX (forced)")
            return RunnerUtils.run_onnx_slice(slice_info, in_file, out_file, slice_dir)

        if forced == 'jstprove' and has_jst:
            logger.info(f"[{slice_id}] Running with JSTprove (forced)")
            j_slice = RunnerUtils._prepare_jstprove_slice(slice_info)
            return runner._run_jstprove_slice(j_slice, in_file, out_file, slice_dir)

        if forced == 'ezkl' and has_ezkl:
            logger.info(f"[{slice_id}] Running with EZKL (forced)")
            e_slice = RunnerUtils._prepare_ezkl_slice(slice_info)
            return runner._run_ezkl_slice(e_slice, in_file, out_file, slice_dir)

        if has_jst:
            logger.info(f"[{slice_id}] Running with JSTprove (available: {available})")
            j_slice = RunnerUtils._prepare_jstprove_slice(slice_info)
            ok, tensor, j_info = runner._run_jstprove_slice(j_slice, in_file, out_file, slice_dir)
            if ok:
                return ok, tensor, j_info
            logger.warning(f"[{slice_id}] JSTprove failed, trying fallback...")

            if has_ezkl:
                logger.info(f"[{slice_id}] Falling back to EZKL")
                e_slice = RunnerUtils._prepare_ezkl_slice(slice_info)
                ok, tensor, e_info = runner._run_ezkl_slice(e_slice, in_file, out_file, slice_dir)
                if ok:
                    return ok, tensor, e_info
                logger.warning(f"[{slice_id}] EZKL failed, falling back to ONNX")

            logger.info(f"[{slice_id}] Falling back to ONNX")
            ok, tensor, o_info = RunnerUtils.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
            o_info["method"] = "jstprove_fallback_onnx"
            o_info["attempted_jstprove"] = True
            o_info["attempted_ezkl"] = has_ezkl
            return ok, tensor, o_info

        if has_ezkl:
            logger.info(f"[{slice_id}] Running with EZKL (available: {available})")
            e_slice = RunnerUtils._prepare_ezkl_slice(slice_info)
            ok, tensor, e_info = runner._run_ezkl_slice(e_slice, in_file, out_file, slice_dir)
            if ok:
                return ok, tensor, e_info
            logger.warning(f"[{slice_id}] EZKL failed, falling back to ONNX")
            ok, tensor, o_info = RunnerUtils.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
            o_info["method"] = "ezkl_fallback_onnx"
            o_info["attempted_ezkl"] = True
            return ok, tensor, o_info

        logger.info(f"[{slice_id}] Running with ONNX (no ZK circuits available)")
        ok, tensor, onnx_info = RunnerUtils.run_onnx_slice(slice_info, in_file, out_file, slice_dir)
        return ok, tensor, onnx_info

    @staticmethod
    def _prepare_jstprove_slice(slice_info: dict) -> dict:
        """Prepare slice_info dict with JSTprove-specific paths."""
        meta = RunSliceMetadata.from_dict(slice_info)
        j_slice = dict(slice_info)
        j_slice["circuit_path"] = meta.jstprove_circuit_path or meta.circuit_path
        if meta.settings_path:
            j_slice["settings_path"] = meta.settings_path
        return j_slice

    @staticmethod
    def _prepare_ezkl_slice(slice_info: dict) -> dict:
        """Prepare slice_info dict with EZKL-specific paths."""
        meta = RunSliceMetadata.from_dict(slice_info)
        e_slice = dict(slice_info)
        e_slice["circuit_path"] = meta.ezkl_circuit_path or meta.circuit_path
        if meta.settings_path:
            e_slice["settings_path"] = meta.settings_path
        if meta.vk_path:
            e_slice["vk_path"] = meta.vk_path
        if meta.pk_path:
            e_slice["pk_path"] = meta.pk_path
        return e_slice

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
    def filter_tensor(current_slice_metadata: Union[dict, RunSliceMetadata], tensor, strict: bool = True):
        logits = tensor["logits"]

        if isinstance(current_slice_metadata, dict):
            current_slice_metadata = RunSliceMetadata.from_dict(current_slice_metadata)

        output_shape = current_slice_metadata.output_shape
        if output_shape:
            shape_ok = RunnerUtils.check_expected_shape(logits, output_shape, tensor_name="logits")
            if not shape_ok and strict:
                expected = output_shape[0] if output_shape else output_shape
                raise ValueError(
                    f"Shape mismatch: got {list(logits.shape)} ({logits.numel()} elements), "
                    f"expected {expected}"
                )
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
