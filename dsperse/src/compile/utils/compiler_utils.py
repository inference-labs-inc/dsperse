import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from dsperse.src.backends.ezkl import EZKL
from dsperse.src.run.runner import Runner
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
        EZKL files are in payload subdirectories, JSTprove files are in backend directories.
        Supports both EZKL and JSTprove backends.
        """
        def _ok_ezkl(key: str) -> bool:
            p = compilation_data.get(key)
            return bool(p) and os.path.exists(p) and ('payload' in str(p).split(os.sep))

        def _ok_jstprove(key: str) -> bool:
            p = compilation_data.get(key)
            return bool(p) and os.path.exists(p)  # JSTprove doesn't use payload subdirs

        # Check if this is a JSTprove compilation (has 'circuit' key, no 'vk_key'/'pk_key')
        if compilation_data.get('circuit') and not compilation_data.get('vk_key'):
            # JSTprove requires 'compiled' (circuit) and 'settings'
            return _ok_jstprove('compiled') and _ok_jstprove('settings')

        # EZKL requires compiled, vk_key, pk_key, settings
        return all([_ok_ezkl('compiled'), _ok_ezkl('vk_key'), _ok_ezkl('pk_key'), _ok_ezkl('settings')])

    @staticmethod
    def get_relative_paths(compilation_data: Dict[str, Any], calibration_input: Optional[str]) -> dict[str, str | None]:
        """
        Compute payload-relative paths for compiled artifacts and the calibration file.
        Returns a tuple of (payload_rel_dict, calibration_rel_path).
        """
        calibration_rel = CompilerUtils._rel_from_payload(calibration_input) if calibration_input and os.path.exists(calibration_input) else None

        # Detect backend by fields present in compilation_data
        is_jstprove = bool(compilation_data.get('compiled')) and not bool(compilation_data.get('vk_key'))

        if is_jstprove:
            relative_paths = CompilerUtils.get_relative_paths_jstprove(compilation_data, calibration_rel)
        else:
            relative_paths = CompilerUtils.get_relative_paths_ezkl(compilation_data, calibration_rel)

        return relative_paths

    @staticmethod
    def get_relative_paths_jstprove(compilation_data: Dict[str, Any], calibration_rel: Optional[str]) -> dict[str, str | None]:
        """
        Build payload-relative files mapping for JSTprove artifacts using backend-provided keys.
        """
        return {
            'settings': CompilerUtils._rel_from_payload(compilation_data.get('settings')),
            'compiled': CompilerUtils._rel_from_payload(compilation_data.get('compiled')),
            'witness_solver': CompilerUtils._rel_from_payload(compilation_data.get('witness_solver')),
            'wandb': CompilerUtils._rel_from_payload(compilation_data.get('wandb')),
            'quantized_model': CompilerUtils._rel_from_payload(compilation_data.get('quantized_model')),
            'metadata': CompilerUtils._rel_from_payload(compilation_data.get('metadata')),
            'architecture': CompilerUtils._rel_from_payload(compilation_data.get('architecture')),
            'calibration': calibration_rel,
        }

    @staticmethod
    def get_relative_paths_ezkl(compilation_data: Dict[str, Any], calibration_rel: Optional[str]) -> dict[str, str | None]:
        """
        Build payload-relative files mapping for EZKL artifacts using backend-provided keys.
        """
        return {
            'settings': CompilerUtils._rel_from_payload(compilation_data.get('settings')),
            'compiled': CompilerUtils._rel_from_payload(compilation_data.get('compiled')),
            'vk_key': CompilerUtils._rel_from_payload(compilation_data.get('vk_key')),
            'pk_key': CompilerUtils._rel_from_payload(compilation_data.get('pk_key')),
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
    def update_slice_metadata(idx: int, filepath: str | Path, success: bool, file_paths: Dict[str, str | None], backend_name: str = "ezkl"):
        """
        Update the per-slice metadata.json file with compilation results.

        Args:
            idx: Slice index
            filepath: Path to the slice's metadata.json file
            success: Boolean indicating if compilation was successful
            file_paths: Dictionary containing file paths for compilation results
            backend_name: Name of the backend used (jstprove, ezkl, or onnx)
        """
        # Load existing slice metadata or create new
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                slice_metadata = json.load(f)
        else:
            slice_metadata = {}

        # Get backend version based on which backend was used
        if backend_name == "jstprove":
            from dsperse.src.backends.jstprove import JSTprove
            backend_version = JSTprove.get_version()
        elif backend_name == "ezkl":
            backend_version = EZKL.get_version()
        else:
            backend_version = None

        # Create compilation info nested under the backend name
        compilation_info = {
            "compiled": success,
            "compilation_timestamp": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend_name,
            "backend_version": backend_version,
            # Preserve keys from provided file_paths to mirror model-level structure
            "files": file_paths or {}
        }

        # Add any errors if present
        errors = {k: v for k, v in file_paths.items() if k.endswith('_error')}
        if errors:
            compilation_info["errors"] = errors

        # Find the specific slice by index and update its compilation info
        updated = False
        if 'slices' in slice_metadata and isinstance(slice_metadata['slices'], list):
            for slice_item in slice_metadata['slices']:
                if slice_item.get('index') == idx:
                    if 'compilation' not in slice_item:
                        slice_item['compilation'] = {}
                    slice_item['compilation'][backend_name] = compilation_info
                    updated = True
                    break

        if not updated:
            # Fallback: update at root level if slice not found in list
            if 'compilation' not in slice_metadata:
                slice_metadata['compilation'] = {}
            slice_metadata['compilation'][backend_name] = compilation_info
            logger.debug(f"Slice with index {idx} not found in slices list. Added compilation info at root level.")

        # Save updated slice metadata
        with open(filepath, 'w') as f:
            json.dump(slice_metadata, f, indent=2)

        logger.debug(f"Updated slice metadata at {filepath} for backend {backend_name}")


    @staticmethod
    def run_onnx_inference_chain(slices_data: list, base_path: str, input_file_path: Optional[str] = None):
        """
        Phase 1: Run ONNX inference chain to generate calibration files.

        Args:
            slices_data: List of slice metadata
            base_path: Base path for relative file paths
            input_file_path: Path to the initial input file
        """
        current_input = input_file_path
        if current_input and os.path.exists(current_input):
            logger.info("Running ONNX inference chain to generate calibration files")
            for idx, slice_data in enumerate(slices_data):
                slice_path = slice_data.get('path')
                # First try full path
                if slice_path and os.path.exists(slice_path):
                    pass  # Use the full path
                # Then try relative path
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

                # Run ONNX inference to generate calibration data
                output_tensor_path = os.path.join(slice_output_path, "calibration.json")
                logger.info(f"Running ONNX inference for slice {idx} with input file {current_input}")
                success, _tensor, exec_info = Runner.run_onnx_slice(
                    slice_info={"path": slice_path},
                    input_tensor_path=Path(current_input),
                    output_tensor_path=Path(output_tensor_path)
                )

                if not success:
                    logger.error(f"ONNX inference failed for slice {idx}: {exec_info.get('error', 'Unknown error')}")
                    return

                current_input = output_tensor_path
                logger.info(f"Generated calibration file: {output_tensor_path}")
        else:
            logger.warning("No input file provided, skipping ONNX inference chain")
