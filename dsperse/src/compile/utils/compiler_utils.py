import json
import logging
import os
import shutil
import time
import copy
from pathlib import Path
from typing import Optional, Dict, Any

from dsperse.src.backends.ezkl import EZKL
from dsperse.src.analyzers.schema import RunSliceMetadata, Backend
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.slice.utils.converter import Converter
logger = logging.getLogger(__name__)

class CompilerUtils:

    @staticmethod
    def is_sliced_model(model_path: str) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Check if the path is a sliced model (dirs, dslice, or dsperse format).

        Returns:
            Tuple of (is_sliced, slice_path, slice_type) where:
                - is_sliced: boolean indicating if this is a sliced model
                - slice_path: the actual path to the slices
                - slice_type: one of 'dirs', 'dslice', 'dsperse', or None
        """
        path_obj = Path(model_path)

        # Check for compressed slice formats (direct file)
        if path_obj.is_file():
            if path_obj.suffix == '.dsperse':
                return True, str(path_obj), 'dsperse'
            elif path_obj.suffix == '.dslice':
                return True, str(path_obj), 'dslice'

        # Check for directory formats
        if path_obj.is_dir():
            # Check if directory contains a .dsperse file
            dsperse_files = [f for f in path_obj.iterdir() if f.is_file() and f.suffix == '.dsperse']
            if dsperse_files:
                return True, str(dsperse_files[0]), 'dsperse'

            try:
                detected_type = Converter.detect_type(path_obj)
            except ValueError:
                detected_type = None

            if detected_type in ['dirs', 'dslice', 'dsperse']:
                return True, str(path_obj), detected_type

            # Check if directory contains a 'slices' subdirectory
            slices_subdir = path_obj / 'slices'
            if slices_subdir.is_dir():
                return True, str(slices_subdir), 'dirs'

        return False, None, None

    @staticmethod
    def parse_layers(layers_str: Optional[str]):
        if not layers_str:
            return None
        layer_indices = []
        parts = [p.strip() for p in layers_str.split(',')]
        for part in parts:
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    layer_indices.extend(range(start, end + 1))
                except ValueError:
                    logger.warning(f"Invalid layer range: {part}. Skipping.")
            else:
                try:
                    layer_indices.append(int(part))
                except ValueError:
                    logger.warning(f"Invalid layer index: {part}. Skipping.")
        return sorted(set(layer_indices)) if layer_indices else None

    @staticmethod
    def parse_backend_and_layers(layers: Optional[str]):
        """
        Parses the 'layers' argument to determine the default backend,
        whether fallback should be used, and which layer indices to compile.

        Returns:
            tuple: (default_backend, use_fallback, layer_indices)
        """
        default_backend = None
        use_fallback = True
        layer_indices = None

        if layers and layers.lower() in [Backend.JSTPROVE, Backend.EZKL]:
            default_backend = layers.lower()
            use_fallback = False
            layer_indices = None
        elif layers and (":" in layers or ";" in layers):
            # This case requires the compiler instance to call _parse_layer_backends
            # We return a special flag or handle logic in the caller
            use_fallback = True
            default_backend = None
            layer_indices = "PARSE_COMPLEX"
        else:
            layer_indices = CompilerUtils.parse_layers(layers) if layers else None

        return default_backend, use_fallback, layer_indices


    @staticmethod
    def _rel_from_payload(path: Optional[str]) -> Optional[str]:
        """
        Given an absolute or relative path, return the subpath starting from the
        'payload' directory (e.g., 'payload/ezkl/...'). If 'payload' is not present,
        return None.
        """
        if not path:
            return None
        parts = str(path).split(os.sep)
        try:
            i = parts.index('payload')
            return os.path.join(*parts[i:])
        except ValueError:
            return None

    @staticmethod
    def _with_slice_prefix(rel_path: Optional[str], slice_dirname: str) -> Optional[str]:
        """
        Prefix a payload-relative path with the slice directory name
        (e.g., 'slice_3/payload/ezkl/...'). If rel_path is None, returns None.
        """
        if not rel_path:
            return None
        return os.path.join(slice_dirname, rel_path)

    @staticmethod
    def is_ezkl_compilation_successful(compilation_data: Dict[str, Any]) -> bool:
        """
        Determine if compilation was successful based on produced file paths.
        Supports both EZKL and JSTprove backends.
        """
        def _ok(key: str) -> bool:
            p = compilation_data.get(key)
            return bool(p) and os.path.exists(p)

        # Check if this is a JSTprove compilation (has 'circuit' key, no 'vk_key'/'pk_key')
        if compilation_data.get('circuit') and not compilation_data.get('vk_key'):
            # JSTprove requires 'compiled' (circuit) and 'settings'
            return _ok('compiled') and _ok('settings')

        # EZKL requires compiled, vk_key, pk_key, settings
        return all([_ok('compiled'), _ok('vk_key'), _ok('pk_key'), _ok('settings')])

    @staticmethod
    def get_relative_paths(compilation_data: Dict[str, Any], calibration_input: Optional[str], slice_dir: Optional[str] = None) -> dict[str, str | None]:
        """
        Compute relative paths for compiled artifacts and the calibration file.
        If slice_dir is provided, paths are relative to it. Otherwise, they are
        relative starting from the 'payload' directory.
        Returns a tuple of (rel_dict, calibration_rel_path).
        """
        if slice_dir:
            def _rel(p):
                if not p: return None
                try: return os.path.relpath(p, slice_dir)
                except ValueError: return None
            calibration_rel = _rel(calibration_input) if calibration_input and os.path.exists(calibration_input) else None
        else:
            calibration_rel = CompilerUtils._rel_from_payload(calibration_input) if calibration_input and os.path.exists(calibration_input) else None

        # Detect backend by fields present in compilation_data
        is_jstprove = bool(compilation_data.get('compiled')) and not bool(compilation_data.get('vk_key'))

        if is_jstprove:
            relative_paths = CompilerUtils.get_relative_paths_jstprove(compilation_data, calibration_rel, slice_dir)
        else:
            relative_paths = CompilerUtils.get_relative_paths_ezkl(compilation_data, calibration_rel, slice_dir)

        return relative_paths

    @staticmethod
    def get_relative_paths_jstprove(compilation_data: Dict[str, Any], calibration_rel: Optional[str], slice_dir: Optional[str] = None) -> dict[str, str | None]:
        """
        Build relative files mapping for JSTprove artifacts using backend-provided keys.
        """
        def _rel(key):
            p = compilation_data.get(key)
            if not p: return None
            if slice_dir:
                try: return os.path.relpath(p, slice_dir)
                except ValueError: return None
            return CompilerUtils._rel_from_payload(p)

        return {
            'settings': _rel('settings'),
            'compiled': _rel('compiled'),
            'witness_solver': _rel('witness_solver'),
            'wandb': _rel('wandb'),
            'quantized_model': _rel('quantized_model'),
            'metadata': _rel('metadata'),
            'architecture': _rel('architecture'),
            'calibration': calibration_rel,
        }

    @staticmethod
    def get_relative_paths_ezkl(compilation_data: Dict[str, Any], calibration_rel: Optional[str], slice_dir: Optional[str] = None) -> dict[str, str | None]:
        """
        Build relative files mapping for EZKL artifacts using backend-provided keys.
        """
        def _rel(key):
            p = compilation_data.get(key)
            if not p: return None
            if slice_dir:
                try: return os.path.relpath(p, slice_dir)
                except ValueError: return None
            return CompilerUtils._rel_from_payload(p)

        return {
            'settings': _rel('settings'),
            'compiled': _rel('compiled'),
            'vk_key': _rel('vk_key'),
            'pk_key': _rel('pk_key'),
            'calibration': calibration_rel,
        }

    @staticmethod
    def apply_payload_rel_to_comp_data(compilation_data: Dict[str, Any], payload_rel: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """
        Produce a shallow copy of compilation_data with payload-relative overrides
        for keys present in payload_rel.
        """
        copy = dict(compilation_data)
        for k, v in payload_rel.items():
            if v:
                copy[k] = v
        return copy

    @staticmethod
    def get_slice_dirs(slice_path: str) -> tuple[str, str]:
        """
        From a slice ONNX path '.../payload/slice_X.onnx', return a tuple of
        (slice_dir, slice_metadata_path), where slice_dir is the parent directory
        of 'payload' (i.e., the slice folder), and slice_metadata_path is
        'slice_dir/metadata.json'.
        """
        slice_dir = os.path.dirname(os.path.dirname(slice_path))
        slice_metadata_path = os.path.join(slice_dir, 'metadata.json')
        return slice_dir, slice_metadata_path

    @staticmethod
    def build_model_level_ezkl(payload_rel: Dict[str, Optional[str]], calibration_rel: Optional[str], slice_dirname: str, compilation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the model-level 'ezkl' dictionary using slice-prefixed payload-relative paths.
        Keeps flat keys for backward-compatibility and mirrors names used elsewhere.
        Includes any '*_error' fields from compilation_data.
        """
        compiled_prefixed = CompilerUtils._with_slice_prefix(payload_rel.get('compiled'), slice_dirname)
        model_level_ezkl = {
            'settings': CompilerUtils._with_slice_prefix(payload_rel.get('settings'), slice_dirname),
            'compiled': compiled_prefixed,
            'compiled_circuit': compiled_prefixed,
            'vk_key': CompilerUtils._with_slice_prefix(payload_rel.get('vk_key'), slice_dirname),
            'pk_key': CompilerUtils._with_slice_prefix(payload_rel.get('pk_key'), slice_dirname),
            'calibration': CompilerUtils._with_slice_prefix(calibration_rel, slice_dirname),
        }
        for k, v in compilation_data.items():
            if isinstance(k, str) and k.endswith('_error'):
                model_level_ezkl[k] = v
        return model_level_ezkl

    @staticmethod
    def update_slice_metadata(idx: int, filepath: str | Path, success: bool, compilation_info: Dict[str, Any], backend_name: str = Backend.EZKL):
        """
        Update the per-slice metadata.json file with compilation results.

        Args:
            idx: Slice index
            filepath: Path to the slice's metadata.json file
            success: Boolean indicating if compilation was successful
            compilation_info: Standardized compilation block containing files, backend, etc.
            backend_name: Name of the backend used (jstprove, ezkl, or onnx)
        """
        # Load existing slice metadata or create new
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                slice_metadata = json.load(f)
        else:
            slice_metadata = {}

        # Strip slice_N/ prefix from file paths for per-slice metadata
        # (Paths should be relative to the slice directory itself)
        comp_info = copy.deepcopy(compilation_info)
        sdn = os.path.basename(os.path.dirname(str(filepath)))

        def _strip(p):
            if isinstance(p, str) and p.startswith(sdn + os.sep):
                return p[len(sdn) + 1:]
            return p

        if "files" in comp_info:
            if comp_info.get("tiled"):
                # Tiled structure: files -> tile_0 -> { file_key: path }
                for tile_key, tile_files in comp_info["files"].items():
                    if isinstance(tile_files, dict):
                        comp_info["files"][tile_key] = {k: _strip(v) for k, v in tile_files.items()}
            else:
                # Standard structure: files -> { file_key: path }
                comp_info["files"] = {k: _strip(v) for k, v in comp_info["files"].items()}

        # Find the specific slice by index and update its compilation info
        updated = False
        if 'slices' in slice_metadata and isinstance(slice_metadata['slices'], list):
            for slice_item in slice_metadata['slices']:
                # For per-slice metadata, the list usually only has one item
                if slice_item.get('index') == idx or len(slice_metadata['slices']) == 1:
                    if 'compilation' not in slice_item:
                        slice_item['compilation'] = {}
                    slice_item['compilation'][backend_name] = comp_info
                    updated = True
                    break

        if not updated:
            # Fallback: update at root level
            if 'compilation' not in slice_metadata:
                slice_metadata['compilation'] = {}
            slice_metadata['compilation'][backend_name] = comp_info

        # Save updated slice metadata
        with open(filepath, 'w') as f:
            json.dump(slice_metadata, f, indent=2)

        logger.debug(f"Updated slice metadata at {filepath} for backend {backend_name}")

    @staticmethod
    def run_onnx_inference_chain(slices_data: list, base_path: str, input_file_path: Optional[str] = None):
        """
        Phase 1: Run ONNX inference chain to generate calibration files.

        Uses a tensor cache to properly route multi-output slices by storing
        all output tensors by name and looking up inputs by dependency names.

        Args:
            slices_data: List of slice metadata
            base_path: Base path for relative file paths
            input_file_path: Path to the initial input file
        """
        import torch
        import numpy as np
        from dsperse.src.backends.onnx_models import OnnxModels

        if not input_file_path or not os.path.exists(input_file_path):
            logger.warning("No input file provided, skipping ONNX inference chain")
            return

        logger.info("Running ONNX inference chain to generate calibration files")

        initial_tensor = RunnerUtils.preprocess_input(input_file_path)
        first_slice = slices_data[0] if slices_data else None
        if not first_slice:
            logger.warning("No slices data, skipping ONNX inference chain")
            return

        deps = first_slice.get('dependencies', {})
        first_inputs = deps.get('filtered_inputs', deps.get('input', []))
        model_input_name = first_inputs[0] if first_inputs else 'input'

        tensor_cache: Dict[str, torch.Tensor] = {model_input_name: initial_tensor}

        for idx, slice_data in enumerate(slices_data):
            slice_path = slice_data.get('path')
            if slice_path and os.path.exists(slice_path):
                pass
            elif slice_data.get('relative_path'):
                slice_path = os.path.join(base_path, slice_data.get('relative_path'))
                if not os.path.exists(slice_path):
                    logger.warning(f"Slice file not found for index {idx}: {slice_path}")
                    continue
            else:
                logger.error(f"No valid path found for slice index {idx}")
                continue

            slice_output_path = os.path.join(os.path.dirname(slice_path), "ezkl")
            os.makedirs(slice_output_path, exist_ok=True)
            calibration_path = os.path.join(slice_output_path, "calibration.json")

            deps = slice_data.get('dependencies', {})
            filtered_inputs = [n for n in deps.get('filtered_inputs', deps.get('input', [])) if n]
            output_names = deps.get('output', [])

            if len(filtered_inputs) <= 1:
                input_name = filtered_inputs[0] if filtered_inputs else model_input_name
                input_tensor = tensor_cache.get(input_name)
                if input_tensor is None:
                    logger.warning(f"Slice {idx}: Input '{input_name}' not in cache, using initial tensor")
                    input_tensor = initial_tensor

                RunnerUtils.save_to_file_flattened(input_tensor, calibration_path)

                try:
                    success, result = OnnxModels.run_inference_tensor(input_tensor, slice_path)
                except Exception as e:
                    logger.error(f"ONNX inference failed for slice {idx}: {e}")
                    return
            else:
                missing = [n for n in filtered_inputs if n not in tensor_cache]
                if missing:
                    logger.warning(f"Slice {idx}: Missing inputs {missing}, cannot run multi-input inference")
                    return

                extra_tensors = {name: tensor_cache[name] for name in filtered_inputs}

                first_input = tensor_cache[filtered_inputs[0]]
                RunnerUtils.save_to_file_flattened(first_input, calibration_path)

                try:
                    success, result = OnnxModels.run_inference_multi(slice_path, extra_tensors)
                except Exception as e:
                    logger.error(f"ONNX inference failed for slice {idx}: {e}")
                    return

            if not success:
                err = result if isinstance(result, str) else 'inference_failed'
                logger.error(f"ONNX inference failed for slice {idx}: {err}")
                return

            if isinstance(result, dict) and 'output_tensors' in result:
                for oname, tensor in result['output_tensors'].items():
                    tensor_cache[oname] = tensor
                    logger.debug(f"Slice {idx}: Cached output '{oname}' shape={tensor.shape} dtype={tensor.dtype}")
            elif output_names:
                out_tensor = result.get('output') if isinstance(result, dict) else result
                if isinstance(out_tensor, torch.Tensor):
                    for oname in output_names:
                        tensor_cache[oname] = out_tensor

            logger.info(f"Slice {idx}: Saved calibration to {calibration_path}")


    @staticmethod
    def log_compilation_summary(stats: dict, compiled: int, skipped: int):
        summary = {}
        for backends in stats.values():
            for be in backends:
                summary[be] = summary.get(be, 0) + 1
        summary_str = ", ".join(f"{k}: {v}" for k, v in summary.items())
        msg = f"Compilation completed. ZK compiled: {compiled} slices ({summary_str})."
        if skipped > 0:
            msg += f" Skipped: {skipped} slices (pure ONNX)."
        logger.info(msg)


    @staticmethod
    def build_compilation_block(backend: str, version: str, success: bool, file_paths: dict, slice_dir: str,
                                tiling_info: Optional[dict]) -> dict:
        sdn = os.path.basename(slice_dir)

        def _prefix(p):
            if isinstance(p, str) and not p.startswith(sdn + os.sep):
                return os.path.join(sdn, p)
            return p

        pref_files = {k: _prefix(v) for k, v in (file_paths or {}).items()}

        block = {
            "compiled": bool(success),
            "compilation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend,
            "backend_version": version,
        }

        if tiling_info:
            num_tiles = tiling_info.get("num_tiles", 1)
            block.update({
                "tiled": True,
                "tile_size": tiling_info.get("tile_size"),
                "tile_count": num_tiles,
                "files": {"tile_0": pref_files}
            })
        else:
            block["files"] = pref_files
        return block

    @staticmethod
    def get_slice_dir(base_path: str, slice_data: dict, idx: int) -> str:
        slice_meta_rel = slice_data.get('slice_metadata_relative_path')
        if slice_meta_rel:
            return os.path.join(base_path, os.path.dirname(slice_meta_rel))
        return os.path.join(base_path, f"slice_{idx}")

    @staticmethod
    def resolve_tile_path(base_path: str, slice_dir: str, tiling_info: dict, idx: int) -> Optional[str]:
        tile_meta = tiling_info.get('tile')
        if not tile_meta:
            logger.warning(f"Slice {idx}: Tiled but no tile metadata")
            return None

        tile_path_raw = tile_meta.get('path')
        if not tile_path_raw:
            logger.warning(f"Slice {idx}: Tiled but path missing")
            return None

        if os.path.isabs(tile_path_raw):
            tile_path = tile_path_raw
        else:
            tile_path = os.path.join(base_path, tile_path_raw)
            if not os.path.exists(tile_path):
                tile_path = os.path.join(slice_dir, tile_path_raw)

        if not os.path.exists(tile_path):
            logger.warning(f"Slice {idx}: Tile not found at {tile_path}")
            return None
        return tile_path

