import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from dataclasses import replace

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from dsperse.src.analyzers.schema import ExecutionInfo, RunSliceMetadata, Backend, ExecutionMethod, RunMetadata, TilingInfo, ChannelSplitInfo, TileResult, ChannelGroupInfo, ExecutionNode
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.torch_utils import ModelUtils
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class RunnerUtils:
    def __init__(self):
        pass

    @staticmethod
    def extract_output_tensor(result):
        """Extract output tensor from execution result, checking 'output', 'logits', then 'output_tensor' keys."""
        if result is None:
            return None
        if isinstance(result, dict):
            for key in ['output', 'logits', 'output_tensor']:
                if key in result and result[key] is not None:
                    return result[key]
            return None
        return result

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
    def make_run_dir(run_metadata: RunMetadata, output_path: str | None, model_path: Path) -> Path:
        base_run_dir = Path(run_metadata.run_directory or (model_path / "run"))
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
    def initialize_run_state(input_json_path, run_metadata: RunMetadata, head_id: str):
        """Prepare initial input tensor and tensor cache."""
        input_tensor = Utils.read_input(input_json_path)
        first_slice_meta = run_metadata.get_slice(head_id)
        first_slice_inputs = first_slice_meta.dependencies.filtered_inputs if first_slice_meta else []
        model_input_name = first_slice_inputs[0] if first_slice_inputs else "input"
        return input_tensor, {model_input_name: input_tensor}


    @staticmethod
    def finalize_run_results(run_metadata, input_tensor, final_tensor, slice_results, run_dir):
        """Aggregate results and save the final run summary."""
        if final_tensor is None:
            if slice_results: logger.warning("No output tensor produced by execution chain, using input tensor")
            final_tensor = input_tensor

        results = {
            "output": final_tensor.tolist(),
            "tensor_shape": list(final_tensor.shape),
            "slice_results": slice_results,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        RunnerUtils.save_run_results(run_metadata, results, run_dir / "run_results.json")
        return results

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
    def prepare_jstprove_meta(meta: RunSliceMetadata) -> RunSliceMetadata:
        """Prepare metadata with JSTprove-specific circuit path and settings."""
        return replace(meta, 
            circuit_path=meta.jstprove_circuit_path or meta.circuit_path,
            settings_path=meta.jstprove_settings_path or meta.settings_path
        )

    @staticmethod
    def prepare_ezkl_meta(meta: RunSliceMetadata) -> RunSliceMetadata:
        """Prepare metadata with EZKL-specific circuit path and keys."""
        return replace(meta, 
            circuit_path=meta.ezkl_circuit_path or meta.circuit_path,
            settings_path=meta.ezkl_settings_path or meta.settings_path,
            vk_path=meta.ezkl_vk_path or meta.vk_path,
            pk_path=meta.ezkl_pk_path or meta.pk_path
        )

    @staticmethod
    def flatten_input_for_ezkl(in_file: Path) -> Path:
        """Create flattened rank-2 input file for EZKL, return path to it."""
        with open(in_file, 'r') as f:
            data = json.load(f)
        tensor = torch.tensor(data.get("input_data", data.get("input", [])))
        flattened = tensor.flatten().tolist()
        ezkl_file = in_file.parent / "input_ezkl.json"
        with open(ezkl_file, 'w') as f:
            json.dump({"input_data": [flattened]}, f)
        return ezkl_file

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
        """Return raw output tensor. Model-specific post-processing is caller's responsibility."""
        return {"output": torch_tensor}

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
    def filter_tensor(meta: RunSliceMetadata, tensor, strict: bool = True):
        output = tensor["output"]
        output_shape = meta.output_shape
        if output_shape:
            shape_ok = RunnerUtils.check_expected_shape(output, output_shape, tensor_name="output")
            if not shape_ok and strict:
                expected = output_shape[0] if output_shape else output_shape
                raise ValueError(
                    f"Shape mismatch: got {list(output.shape)} ({output.numel()} elements), "
                    f"expected {expected}"
                )
        else:
            logger.debug("Output shape metadata not found for shape check.")

        return output

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

    @staticmethod
    def execute_tile_worker(args: dict) -> dict:
        """Execute a single tile of a tiled slice in a separate process."""
        tile_idx = args['tile_idx']
        tile_in = args['tile_in']
        tile_out = args['tile_out']
        has_jst = args['has_jst']
        has_ezkl = args['has_ezkl']
        slices_path = Path(args['slices_path'])

        try:
            if has_jst:
                from dsperse.src.backends.jstprove import JSTprove
                jst_runner = JSTprove()
                circuit_path = RunnerUtils.resolve_relative_path(args['jstprove_circuit_path'], slices_path)
                success, result = jst_runner.generate_witness(tile_in, circuit_path, tile_out)
                if success:
                    output_tensor = RunnerUtils.extract_output_tensor(result)
                    if output_tensor is not None:
                        return {
                            'tile_idx': tile_idx,
                            'success': True,
                            'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                            'method': ExecutionMethod.JSTPROVE_GEN_WITNESS
                        }
                return {'tile_idx': tile_idx, 'success': False, 'error': f'JSTprove failed: {result}'}

            elif has_ezkl:
                from dsperse.src.backends.ezkl import EZKL
                ezkl_runner = EZKL()
                circuit_path = RunnerUtils.resolve_relative_path(args['ezkl_circuit_path'], slices_path)
                vk_path = RunnerUtils.resolve_relative_path(args.get('ezkl_vk_path') or args.get('vk_path'), slices_path)
                settings_path = RunnerUtils.resolve_relative_path(args.get('ezkl_settings_path') or args.get('settings_path'), slices_path)
                success, result = ezkl_runner.generate_witness(tile_in, circuit_path, tile_out, vk_path, settings_path)
                if success:
                    output_tensor = RunnerUtils.extract_output_tensor(result)
                    if output_tensor is not None:
                        return {
                            'tile_idx': tile_idx,
                            'success': True,
                            'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                            'method': ExecutionMethod.EZKL_GEN_WITNESS
                        }
                return {'tile_idx': tile_idx, 'success': False, 'error': f'EZKL failed: {result}'}

            # Fallback to ONNX if requested or if no circuits are available
            tile_onnx_path = args.get('tile_onnx_path')
            if tile_onnx_path:
                from dsperse.src.backends.onnx_models import OnnxModels
                success, result = OnnxModels.run_inference(tile_in, tile_onnx_path, tile_out)
                if success:
                    output_tensor = RunnerUtils.extract_output_tensor(result)
                    return {
                        'tile_idx': tile_idx,
                        'success': True,
                        'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                        'method': ExecutionMethod.ONNX_ONLY
                    }
                return {'tile_idx': tile_idx, 'success': False, 'error': f'ONNX failed: {result}'}

            return {'tile_idx': tile_idx, 'success': False, 'error': 'No backend available'}

        except Exception as e:
            return {'tile_idx': tile_idx, 'success': False, 'error': str(e)}

    @staticmethod
    def save_run_results(run_metadata: RunMetadata, results: dict, output_path: str):
        """Save the final inference output and execution metadata to a JSON file."""
        model_path = run_metadata.model_path or "unknown"
        slice_results = results.get("slice_results", {})

        def _get_method(r):
            return r.method if isinstance(r, ExecutionInfo) else r.get("method", "")

        def _get_tiles(r):
            if isinstance(r, ExecutionInfo):
                return r.tiles
            return r.get("tiles") or r.get("tile_exec_infos") or []

        def _count_tile_method(tiles, method_prefix):
            return sum(1 for t in tiles if (t.method if isinstance(t, TileResult) else t.get("method", "")).startswith(method_prefix))

        ezkl_complete = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.EZKL_GEN_WITNESS)
        jstprove_complete = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.JSTPROVE_GEN_WITNESS)
        jstprove_tiled_slices = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.TILED and _count_tile_method(_get_tiles(r), Backend.JSTPROVE) > 0)
        ezkl_tiled_slices = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.TILED and _count_tile_method(_get_tiles(r), Backend.EZKL) > 0)
        jstprove_complete += jstprove_tiled_slices
        ezkl_complete += ezkl_tiled_slices
        total_slices = len(slice_results)

        execution_results = []
        for slice_id, exec_info in slice_results.items():
            if isinstance(exec_info, ExecutionInfo):
                witness_execution = exec_info.to_dict()
            else:
                witness_execution = {
                    "method": exec_info.get("method", "unknown"),
                    "success": exec_info.get("success", False),
                    "witness_file": exec_info.get("witness_file") or exec_info.get("witness_path"),
                    "tile_exec_infos": exec_info.get("tile_exec_infos", []),
                }
                if exec_info.get("error"):
                    witness_execution["error"] = exec_info["error"]

            execution_results.append({"slice_id": slice_id, "witness_execution": witness_execution})

        # Calculate security percentage
        circuit_slices = ezkl_complete + jstprove_complete
        security_percent = (circuit_slices / total_slices * 100) if total_slices > 0 else 0

        # Build output structure
        inference_output = {
            "model_path": model_path,
            "output": results["output"],
            "tensor_shape": results["tensor_shape"],
            "execution_chain": {
                "total_slices": total_slices,
                "jstprove_witness_slices": jstprove_complete,
                "ezkl_witness_slices": ezkl_complete,
                "overall_security": f"{security_percent:.1f}%",
                "execution_results": execution_results
            },
            "performance_comparison": {
                "note": "Full ONNX vs verified chain comparison would require separate pure ONNX run"
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(inference_output, f, indent=2)

    @staticmethod
    def get_input_tensor_for_tiling(slice_id: str, tiling: TilingInfo, meta: RunSliceMetadata, tensor_cache: dict) -> torch.Tensor:
        """Retrieve the correct input tensor for a tiled slice from the tensor cache."""
        input_name = tiling.input_name or (meta.dependencies.filtered_inputs[0] if meta.dependencies.filtered_inputs else "input")
        input_tensor = tensor_cache.get(input_name)
        if input_tensor is None:
            raise ValueError(f"Missing input tensor '{input_name}' for tiled slice {slice_id}")
        return input_tensor

    @staticmethod
    def split_tensor_into_tiles(slice_id: str, tiling: TilingInfo, input_tensor: torch.Tensor, tensor_cache: dict) -> float:
        """Split a large input tensor into overlapping tiles with halos for parallel processing."""
        start_time = time.time()

        slice_idx = tiling.slice_idx
        tile_size = tiling.tile_size
        halo_h, halo_w = tiling.halo
        tiles_y = tiling.tiles_y
        tiles_x = tiling.tiles_x
        num_tiles = tiling.num_tiles

        tile_with_halo_h = tile_size + 2 * halo_h
        tile_with_halo_w = tile_size + 2 * halo_w

        padded = F.pad(input_tensor, (halo_w, halo_w, halo_h, halo_h), mode='constant', value=0)

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                tile_idx = ty * tiles_x + tx
                y_start = ty * tile_size
                x_start = tx * tile_size
                y_end = y_start + tile_with_halo_h
                x_end = x_start + tile_with_halo_w

                tile = padded[:, :, y_start:y_end, x_start:x_end]
                cache_name = f"tile_{slice_idx}_{tile_idx}_in"
                tensor_cache[cache_name] = tile

        split_time = time.time() - start_time
        logger.info(f"Split {slice_id} completed in {split_time:.3f}s, produced {num_tiles} tiles")
        return split_time

    @staticmethod
    def reconstruct_tensor_from_tiles(slice_id: str, tiling: TilingInfo, tensor_cache: dict) -> float:
        """Reassemble processed tiles back into a single output tensor, removing halos."""
        concat_start = time.time()

        slice_idx = tiling.slice_idx
        tiles_y = tiling.tiles_y
        tiles_x = tiling.tiles_x
        output_name = tiling.output_name

        rows = []
        for ty in range(tiles_y):
            row_tiles = []
            for tx in range(tiles_x):
                tile_idx = ty * tiles_x + tx
                cache_name = f"tile_{slice_idx}_{tile_idx}_out"
                tile = tensor_cache.get(cache_name)
                if tile is None:
                    raise ValueError(f"Missing tile output tensor '{cache_name}' for concat")
                row_tiles.append(tile)
            row = torch.cat(row_tiles, dim=3)
            rows.append(row)

        output = torch.cat(rows, dim=2)
        tensor_cache[output_name] = output

        concat_time = time.time() - concat_start
        logger.info(f"Concat {slice_id} completed in {concat_time:.3f}s, output shape {list(output.shape)}")
        return concat_time

    @staticmethod
    def find_most_recent_run(base_dir: Path) -> Path | None:
        """Search a directory for the latest run subdirectory based on naming convention."""
        run_dirs = sorted(base_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
        return run_dirs[0] if run_dirs else None

    @staticmethod
    def check_slice_completion_status(run_dir: Path, slice_id: str) -> tuple[bool, dict | None]:
        """Determine if a specific slice has already been successfully executed and has saved output."""
        slice_dir = run_dir / slice_id
        output_file = slice_dir / "output.json"
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                return True, data
            except Exception:
                pass
        return False, None

    @staticmethod
    def execute_channel_group(runner, group: ChannelGroupInfo, group_input: torch.Tensor, run_dir: Path, output_shape: tuple, slices_path: Path, backend: str = None) -> tuple[torch.Tensor, str]:
        """Run inference on a specific group of channels for a channel-split slice."""
        forced = backend
        
        # --- Capability Check ---
        has_jst = bool(group.jstprove_circuit_path) and getattr(runner, "jstprove_runner", None)
        has_ezkl = bool(group.ezkl_circuit_path) and (group.vk_path or group.ezkl_vk_path)

        # --- Input Preparation ---
        if isinstance(group_input, torch.Tensor):
            input_arr = group_input.detach().cpu().numpy().astype(np.float32)
        else:
            input_arr = np.asarray(group_input, dtype=np.float32)

        group_dir = run_dir / f"channel_group_{group.group_idx}"
        group_dir.mkdir(parents=True, exist_ok=True)
        in_file = group_dir / "input.json"
        out_file = group_dir / "output.json"
        Utils.write_input(torch.from_numpy(input_arr), in_file)

        def _extract_and_reshape(result):
            tensor = RunnerUtils.extract_output_tensor(result)
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.dim() < 4:
                expected = np.prod(output_shape)
                if tensor.numel() == expected:
                    return tensor.reshape(output_shape)
            return tensor

        # --- Forced ONNX / No ZK fallback ---
        if forced == Backend.ONNX or (not has_jst and not has_ezkl):
            group_onnx_path = RunnerUtils.resolve_relative_path(group.path, slices_path)
            session = ort.InferenceSession(str(group_onnx_path))
            outputs = session.run(None, {session.get_inputs()[0].name: input_arr})
            return torch.from_numpy(outputs[0]), ExecutionMethod.ONNX_ONLY

        # --- Forced Backend / Best Available JSTprove ---
        if (forced == Backend.JSTPROVE and has_jst) or (has_jst and forced != Backend.EZKL):
            circuit_path = RunnerUtils.resolve_relative_path(group.jstprove_circuit_path, slices_path)
            success, result = runner.jstprove_runner.generate_witness(str(in_file), circuit_path, str(out_file))
            if success:
                tensor = _extract_and_reshape(result)
                if tensor is not None:
                    return tensor, ExecutionMethod.JSTPROVE_GEN_WITNESS
            logger.warning(f"JSTprove failed for group {group.group_idx}, falling back")

        # --- Forced Backend / Best Available EZKL ---
        if (forced == Backend.EZKL and has_ezkl) or (has_ezkl and forced != Backend.JSTPROVE):
            circuit_path = RunnerUtils.resolve_relative_path(group.ezkl_circuit_path, slices_path)
            vk_path = RunnerUtils.resolve_relative_path(group.ezkl_vk_path or group.vk_path, slices_path)
            settings_path = RunnerUtils.resolve_relative_path(group.ezkl_settings_path or group.settings_path, slices_path)
            ezkl_in = RunnerUtils.flatten_input_for_ezkl(in_file)
            success, result = runner.ezkl_runner.generate_witness(str(ezkl_in), circuit_path, str(out_file), vk_path, settings_path)
            if success:
                tensor = _extract_and_reshape(result)
                if tensor is not None:
                    return tensor, ExecutionMethod.EZKL_GEN_WITNESS
            logger.warning(f"EZKL failed for group {group.group_idx}, falling back")

        # --- Final ONNX Fallback ---
        group_onnx_path = RunnerUtils.resolve_relative_path(group.path, slices_path)
        session = ort.InferenceSession(str(group_onnx_path))
        outputs = session.run(None, {session.get_inputs()[0].name: input_arr})
        return torch.from_numpy(outputs[0]), ExecutionMethod.ONNX_ONLY




    @staticmethod
    def try_resume_slice(run_dir: Path, slice_id: str) -> tuple[bool, torch.Tensor | None, ExecutionInfo | None]:
        """Attempts to load cached results for a slice if it was previously completed."""
        slice_run_dir = run_dir / slice_id
        # Check for either slice_output.json (legacy) or output.json
        output_file = slice_run_dir / "output.json"
        if not output_file.exists():
            output_file = slice_run_dir / "slice_output.json"

        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    cached_data = json.load(f)
                cached_tensor = torch.tensor(cached_data['output'])
                exec_info = ExecutionInfo(
                    method=cached_data.get('method', 'resumed'),
                    success=True
                )
                print(f"[resume] {slice_id}: loaded cached output", flush=True)
                return True, cached_tensor, exec_info
            except Exception as e:
                logger.warning(f"Failed to load cached output for {slice_id}: {e}")
        return False, None, None

    @staticmethod
    def prepare_slice_input(info: RunSliceMetadata, tensor_cache: dict, model_input_tensor: torch.Tensor, in_file: Path) -> torch.Tensor:
        """Extracts and saves the input tensor for the current slice."""
        filtered_inputs = [n for n in info.dependencies.filtered_inputs if n]
        input_name = filtered_inputs[0] if filtered_inputs else "input"
        current_tensor = tensor_cache.get(input_name, model_input_tensor)

        # Always save aggregate input for the slice
        Utils.write_input(current_tensor, str(in_file))
        return current_tensor

    @staticmethod
    def process_inference_result(slice_id: str, info: RunSliceMetadata, ok: bool, result, exec_info, tensor_cache: dict) -> torch.Tensor | None:
        """Handles reshaping, tensor cache updates, and final tensor extraction from inference results."""
        final_tensor = None
        output_names = info.dependencies.output

        logger.info(f"[{slice_id}] ok={ok}, result type={type(result).__name__}")
        if not ok or result is None:
            return None

        if isinstance(result, dict) and 'output_tensors' in result:
            for oname, tensor in result['output_tensors'].items():
                tensor_cache[oname] = tensor
            first_output = list(result['output_tensors'].values())[0] if result['output_tensors'] else None
            if first_output is not None:
                final_tensor = first_output
            else:
                final_tensor = RunnerUtils.extract_output_tensor(result)
        else:
            out_tensor = RunnerUtils.extract_output_tensor(result)
            if isinstance(out_tensor, torch.Tensor):
                target_shape = info.get_target_shape()
                if target_shape:
                    expected_numel = torch.prod(torch.tensor(target_shape)).item()
                    if out_tensor.numel() == expected_numel:
                        out_tensor = out_tensor.reshape(target_shape)
                        logger.info(f"[{slice_id}] Reshaped output to {target_shape}")
                    else:
                        logger.warning(f"[{slice_id}] Cannot reshape: got {out_tensor.numel()} elements, expected {expected_numel}")
                for oname in output_names:
                    tensor_cache[oname] = out_tensor
                final_tensor = out_tensor
        return final_tensor

    @staticmethod
    def save_intermediate_output(out_file: Path, tensor: torch.Tensor, exec_info: ExecutionInfo):
        """Saves intermediate inference result to a JSON file."""
        method = exec_info.method if isinstance(exec_info, ExecutionInfo) else exec_info.get('method', 'unknown')
        with open(out_file, 'w') as f:
            json.dump({'output': tensor.tolist(), 'method': str(method)}, f)

    @staticmethod
    def resolve_inference_paths(meta: RunSliceMetadata, slice_dir: Path) -> dict:
        """Resolve all circuit-related paths for a slice."""
        return {
            'circuit': RunnerUtils.resolve_relative_path(meta.circuit_path, slice_dir),
            'vk': RunnerUtils.resolve_relative_path(meta.vk_path, slice_dir),
            'settings': RunnerUtils.resolve_relative_path(meta.settings_path, slice_dir)
        }

    @staticmethod
    def get_tile_execution_config(tiling: TilingInfo, meta: RunSliceMetadata, slices_path: Path, backend: str = None, has_jst_runner: bool = False) -> dict:
        """Resolve tile ONNX path and determine available backends for tiling."""
        num_tiles = tiling.num_tiles
        has_per_tile_onnx = tiling.tiles and len(tiling.tiles) == num_tiles
        
        if has_per_tile_onnx:
            tile_onnx_path = RunnerUtils.resolve_relative_path(tiling.tiles[0].path, slices_path)
        elif tiling.tile:
            tile_onnx_path = RunnerUtils.resolve_relative_path(tiling.tile.path, slices_path)
        else:
            tile_onnx_path = None

        effective_backend = backend if backend else Backend.AUTO
        has_jst = effective_backend in (Backend.JSTPROVE, Backend.AUTO) and bool(meta.jstprove_circuit_path) and has_jst_runner
        has_ezkl = effective_backend in (Backend.EZKL, Backend.AUTO) and bool(meta.ezkl_circuit_path) and (meta.ezkl_vk_path or meta.vk_path)
        
        return {
            'tile_onnx_path': tile_onnx_path,
            'has_per_tile_onnx': has_per_tile_onnx,
            'effective_backend': effective_backend,
            'has_jst': has_jst,
            'has_ezkl': has_ezkl,
            'backend_name': Backend.JSTPROVE if has_jst else (Backend.EZKL if has_ezkl else Backend.ONNX)
        }

    @staticmethod
    def prepare_tile_tasks(num_tiles: int, slice_idx: int, tiling: TilingInfo, meta: RunSliceMetadata, run_dir: Path, tensor_cache: dict, slices_path: Path, config: dict) -> list[dict]:
        """Prepare argument lists for individual tile execution workers."""
        tile_args_list = []
        slice_specific_dir = slices_path / f"slice_{slice_idx}"
        
        for tile_idx in range(num_tiles):
            cache_input_name = f"tile_{slice_idx}_{tile_idx}_in"
            tile_tensor = tensor_cache.get(cache_input_name)
            if tile_tensor is None: raise ValueError(f"Missing tile input tensor '{cache_input_name}'")

            tile_run_dir = run_dir / f"slice_{slice_idx}" / f"tile_{tile_idx}"
            tile_run_dir.mkdir(parents=True, exist_ok=True)
            tile_in, tile_out = tile_run_dir / "input.json", tile_run_dir / "output.json"
            Utils.write_input(tile_tensor, str(tile_in))

            this_tile_onnx = RunnerUtils.resolve_relative_path(tiling.tiles[tile_idx].path, slices_path) if config['has_per_tile_onnx'] else config['tile_onnx_path']
            conv_out = tiling.tiles[tile_idx].conv_out if config['has_per_tile_onnx'] else (tiling.tile.conv_out if tiling.tile else (0, 0))

            tile_args_list.append({
                'tile_idx': tile_idx, 'tile_in': str(tile_in), 'tile_out': str(tile_out),
                'tile_onnx_path': str(this_tile_onnx), 
                'jstprove_circuit_path': meta.jstprove_circuit_path,
                'ezkl_circuit_path': meta.ezkl_circuit_path, 
                'settings_path': meta.settings_path,
                'vk_path': meta.vk_path,
                'jstprove_settings_path': meta.jstprove_settings_path,
                'ezkl_settings_path': meta.ezkl_settings_path,
                'ezkl_vk_path': meta.ezkl_vk_path,
                'ezkl_pk_path': meta.ezkl_pk_path,
                'slice_specific_dir': str(slice_specific_dir),
                'slices_path': str(slices_path), 'has_jst': config['has_jst'], 'has_ezkl': config['has_ezkl'],
                'c_out': tiling.c_out, 'conv_out': conv_out,
            })
        return tile_args_list

    @staticmethod
    def process_tile_inference_result(result, tiling: TilingInfo, tensor_cache: dict, slice_idx: int, tile_idx: int):
        """Extract and optionally reshape the output tensor from a tile inference result."""
        output_tensor = RunnerUtils.extract_output_tensor(result)
        if output_tensor is None:
            return None

        # Ensure it's a tensor
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.tensor(output_tensor)

        # Tiles might need reshaping to (1, C, H, W)
        c_out = tiling.c_out
        h_out, w_out = tiling.tile.conv_out if tiling.tile else (0, 0)
        if c_out and h_out and w_out and output_tensor.numel() == (1 * c_out * h_out * w_out):
            output_tensor = output_tensor.reshape(1, c_out, h_out, w_out)

        tensor_cache[f"tile_{slice_idx}_{tile_idx}_out"] = output_tensor
        return output_tensor

    @staticmethod
    def collect_tile_outputs(results_map: dict, num_tiles: int, slice_idx: int, tiling: TilingInfo, tensor_cache: dict) -> list[TileResult]:
        """Process results from tile workers and update tensor cache."""
        tile_exec_infos = []
        for tile_idx in range(num_tiles):
            result = results_map[tile_idx]
            if result['success']:
                output = RunnerUtils.process_tile_inference_result(result, tiling, tensor_cache, slice_idx, tile_idx)
                if output is not None:
                    tile_exec_infos.append(TileResult(tile_idx=tile_idx, success=True, method=result.get('method', 'unknown')))
                else:
                    raise RuntimeError(f"Tile {tile_idx} produced no output tensor")
            else:
                raise RuntimeError(f"Tile {tile_idx} execution failed: {result.get('error', 'unknown')}")
        return tile_exec_infos

    @staticmethod
    def prepare_channel_split_config(meta: RunSliceMetadata, tensor_cache: dict) -> tuple:
        """Calculate target shapes and retrieve input tensor for channel splitting."""
        channel_split = meta.channel_split
        if not channel_split: return None, None, None
        
        input_name = channel_split.input_name or (meta.dependencies.filtered_inputs[0] if meta.dependencies.filtered_inputs else "input")
        input_tensor = tensor_cache.get(input_name)
        if input_tensor is None: return None, None, None

        target_shape = meta.get_target_shape()
        output_shape = tuple(target_shape) if target_shape else (1, channel_split.c_out, channel_split.h, channel_split.w)
        return input_tensor, output_shape, input_name

    @staticmethod
    def apply_channel_split_bias(summed: torch.Tensor, channel_split: ChannelSplitInfo, slices_path: Path) -> torch.Tensor:
        """Apply bias tensor to the summed output of channel groups if specified."""
        if channel_split.bias_path:
            bias_path = RunnerUtils.resolve_relative_path(channel_split.bias_path, slices_path)
            if Path(bias_path).exists():
                bias_tensor = torch.from_numpy(np.load(bias_path)).reshape(1, -1, 1, 1)
                summed = summed + bias_tensor
        return summed

if __name__ == "__main__":
    print(f"Parent path: {RunnerUtils._get_file_path()}")
