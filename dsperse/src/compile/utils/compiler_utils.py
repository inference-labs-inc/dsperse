import copy
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from dsperse.src.analyzers.schema import Backend
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


@dataclass
class CompilationTask:
    idx: int
    slice_data: dict
    base_path: str
    slice_dir: str
    backends_to_build: list[str]
    tiling_info: Optional[dict]
    channel_split_info: Optional[dict]
    compilation_slice_data: dict
    weights_as_inputs: bool = False


@dataclass
class BackendParseResult:
    default_backend: Optional[str]
    use_fallback: bool
    layer_indices: Optional[list[int]]
    needs_complex_parse: bool = False

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
            if not part:
                continue
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
    def parse_backend_and_layers(layers: Optional[str]) -> BackendParseResult:
        """
        Parses the 'layers' argument to determine the default backend,
        whether fallback should be used, and which layer indices to compile.

        Returns:
            BackendParseResult with parsed configuration
        """
        if layers and layers.lower() in [Backend.JSTPROVE, Backend.EZKL]:
            return BackendParseResult(
                default_backend=layers.lower(),
                use_fallback=False,
                layer_indices=None
            )

        if layers and (":" in layers or ";" in layers):
            return BackendParseResult(
                default_backend=None,
                use_fallback=True,
                layer_indices=None,
                needs_complex_parse=True
            )

        return BackendParseResult(
            default_backend=None,
            use_fallback=True,
            layer_indices=CompilerUtils.parse_layers(layers) if layers else None
        )

    @staticmethod
    def resolve_compilation_source(idx: int, slice_data: dict, base_path: str, slice_dir: str, tiling_info: dict | None):
        """Resolves the ONNX file to be used for compilation (e.g. representative tile)."""
        if not tiling_info:
            return slice_data

        tile_path = CompilerUtils.resolve_tile_path(base_path, slice_dir, tiling_info, idx)
        if tile_path and os.path.exists(tile_path):
            logger.info(f"Slice {idx}: Tiled. Compiling representative tile...")
            return {'path': tile_path, 'relative_path': os.path.relpath(tile_path, base_path)}

        logger.warning(f"Slice {idx}: Tiled but tile source not found.")
        return None


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


    _JSTPROVE_ARTIFACT_KEYS = ('settings', 'compiled', 'witness_solver', 'wandb', 'quantized_model', 'metadata', 'architecture')
    _EZKL_ARTIFACT_KEYS = ('settings', 'compiled', 'vk_key', 'pk_key')

    @staticmethod
    def get_relative_paths(compilation_data: Dict[str, Any], calibration_input: Optional[str], slice_dir: Optional[str] = None) -> dict[str, str | None]:
        """Compute relative paths for compiled artifacts and the calibration file."""
        def _rel(p):
            if not p:
                return None
            if slice_dir:
                try:
                    return os.path.relpath(p, slice_dir)
                except ValueError:
                    return None
            return CompilerUtils._rel_from_payload(p)

        calibration_rel = _rel(calibration_input) if calibration_input and os.path.exists(str(calibration_input)) else None

        is_jstprove = bool(compilation_data.get('compiled')) and not bool(compilation_data.get('vk_key'))
        keys = CompilerUtils._JSTPROVE_ARTIFACT_KEYS if is_jstprove else CompilerUtils._EZKL_ARTIFACT_KEYS

        result = {k: _rel(compilation_data.get(k)) for k in keys}
        result['calibration'] = calibration_rel
        return result

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

        if "group_files" in comp_info:
            # Channel split structure: group_files -> group_idx -> { file_key: path }
            for g_idx, g_files in comp_info["group_files"].items():
                if isinstance(g_files, dict):
                    comp_info["group_files"][g_idx] = {k: _strip(v) for k, v in g_files.items()}

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
    def log_compilation_summary(stats: dict, compiled: int, skipped: int):
        """Logs a summary of the compilation results."""
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
    def log_compilation_start(model_path: str, layer_indices: list | None, default_backend: str | None, use_fallback: bool):
        """Logs information about the start of the compilation process."""
        logger.info(f"Compiling: {model_path}")
        if layer_indices:
            logger.info(f"Requested layers: {layer_indices}")
        elif default_backend and not use_fallback:
            logger.info(f"Compiling all layers using {default_backend}")
        else:
            logger.info("Compiling all layers with default fallback (jstprove -> ezkl -> onnx)")


    @staticmethod
    def build_compilation_block(backend: str, version: str, success: bool, file_paths: dict, slice_dir: str,
                                tiling_info: Optional[dict], weights_as_inputs: bool = False) -> dict:
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
        if backend == Backend.JSTPROVE:
            block["weights_as_inputs"] = weights_as_inputs

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



    @staticmethod
    def initialize_compilation_metadata(dir_path: str) -> tuple[dict, str, str, list]:
        """Loads and prepares metadata for compilation."""
        logger.info(f"Loading metadata from {dir_path}...")
        metadata_path = Utils.find_metadata_path(dir_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        base_path = os.path.dirname(metadata_path)
        slices_data = metadata.get('slices', [])
        logger.info(f"Found {len(slices_data)} slices")
        return metadata, metadata_path, base_path, slices_data

    @staticmethod
    def is_slice_compiled(slice_dir: str, tiling_info: Optional[dict] = None, channel_split_info: Optional[dict] = None) -> bool:
        """Check if a slice already has compiled circuits (for resume mode)."""
        # --- Channel Split Check ---
        if channel_split_info:
            # Check if group 0 of jstprove or ezkl exists in the new structure
            jst = os.path.join(slice_dir, "payload", "jstprove", "channel_groups", "group_0", "group_0_circuit.txt")
            ezkl = os.path.join(slice_dir, "payload", "ezkl", "channel_groups", "group_0", "model.compiled")
            return os.path.exists(jst) or os.path.exists(ezkl)

        # --- Tiled Check ---
        if tiling_info:
            jst_circuit = os.path.join(slice_dir, "payload", "jstprove", "tiles", "tile_circuit.txt")
            ezkl_circuit = os.path.join(slice_dir, "payload", "ezkl", "tiles", "model.compiled")
        
        # --- Standard Check ---
        else:
            sdn = os.path.basename(slice_dir)
            jst_circuit = os.path.join(slice_dir, "payload", "jstprove", f"{sdn}_circuit.txt")
            ezkl_circuit = os.path.join(slice_dir, "payload", "ezkl", "model.compiled")
            
        return os.path.exists(jst_circuit) or os.path.exists(ezkl_circuit)

    @staticmethod
    def parse_complex_layer_backends(spec: str) -> tuple[dict[int, str], set[int]]:
        """Parse layer-specific backend specification like '0,2:jstprove;3-4:ezkl'"""
        import re
        layer_backends = {}
        default_layer_indices = set()

        parts = spec.split(';')
        all_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Split by whitespace that is followed by digits and optional range/list then colon
            # Use negative lookbehind to avoid splitting on spaces after commas (e.g., "0, 2:jstprove")
            subparts = re.split(r'(?<!,)\s+(?=\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*:)', p)
            all_parts.extend(subparts)

        for part in all_parts:
            part = part.strip()
            if not part:
                continue
            if ':' not in part:
                idxs = CompilerUtils.parse_layers(part)
                if idxs: default_layer_indices.update(idxs)
                continue
            layers_str, backend_name = part.split(':', 1)
            backend_name = backend_name.strip().lower()
            layer_indices = CompilerUtils.parse_layers(layers_str)
            if layer_indices:
                for idx in layer_indices:
                    layer_backends[idx] = backend_name
        
        return layer_backends, default_layer_indices

    @staticmethod
    def get_backends_to_build(idx: int, layer_backends: dict, default_layer_indices: set, default_backend: str | None, use_fallback: bool) -> list[str]:
        """Determines which backends to build for a specific slice index."""
        backends = []
        if idx in layer_backends:
            backends.append(layer_backends[idx])
        if idx in default_layer_indices:
            for b in [Backend.JSTPROVE, Backend.EZKL]:
                if b not in backends: backends.append(b)
        
        if not backends:
            if default_backend in {Backend.JSTPROVE, Backend.EZKL} and not use_fallback:
                backends = [default_backend]
            else:
                backends = [Backend.JSTPROVE, Backend.EZKL]
        return backends

    @staticmethod
    def prepare_compilation_tasks(slices_data: list, base_path: str, layer_indices: list[int] | None, resume: bool, layer_backends: dict, default_layer_indices: set, default_backend: str | None, use_fallback: bool, weights_as_inputs: bool = False):
        """Determines which slices need compilation and what backends to use."""
        work_items, slice_info_map = [], {}
        stats = {'compiled_count': 0, 'skipped_count': 0, 'backend_stats': {}}

        for idx, slice_data in enumerate(slices_data):
            # Skip logic (layer filter, runtime only, resume)
            should_skip, skip_reason = CompilerUtils.should_skip_slice(idx, slice_data, base_path, layer_indices, resume)
            if should_skip:
                if skip_reason != "resumed": stats['skipped_count'] += 1
                continue

            # Resolve slice specific paths and backends
            slice_dir = CompilerUtils.get_slice_dir(base_path, slice_data, idx)
            tiling_info = slice_data.get('tiling')
            channel_split_info = slice_data.get('channel_split')
            
            comp_slice_data = CompilerUtils.resolve_compilation_source(idx, slice_data, base_path, slice_dir, tiling_info)
            if not comp_slice_data: continue

            backends = CompilerUtils.get_backends_to_build(idx, layer_backends, default_layer_indices, default_backend, use_fallback)
            
            work_items.append(CompilationTask(
                idx=idx, slice_data=slice_data, base_path=base_path, slice_dir=slice_dir,
                backends_to_build=backends, tiling_info=tiling_info,
                channel_split_info=channel_split_info, compilation_slice_data=comp_slice_data,
                weights_as_inputs=weights_as_inputs,
            ))
            slice_info_map[idx] = {
                'slice_data': slice_data, 'slice_dir': slice_dir, 'tiling_info': tiling_info,
                'channel_split_info': channel_split_info, 'original_slice_entry': slice_data
            }

        return work_items, slice_info_map, stats

    @staticmethod
    def should_skip_slice(idx: int, slice_data: dict, base_path: str, layer_indices: list[int] | None, resume: bool) -> tuple[bool, str | None]:
        """Determines if a slice should be skipped based on provided criteria."""
        if layer_indices is not None and idx not in layer_indices:
            logger.info(f"Skipping slice {idx}: not in requested layer indices")
            return True, "not_requested"
        
        if slice_data.get("runtime_only"):
            logger.info(f"Skipping slice {idx}: runtime only")
            return True, "runtime_only"

        if resume:
            slice_meta_rel = slice_data.get('slice_metadata_relative_path')
            check_dir = os.path.join(base_path, os.path.dirname(slice_meta_rel)) if slice_meta_rel else os.path.join(base_path, f"slice_{idx}")
            if CompilerUtils.is_slice_compiled(check_dir, slice_data.get('tiling'), slice_data.get('channel_split')):
                logger.info(f"[resume] slice_{idx}: already compiled, skipping")
                return True, "resumed"

        return False, None

    @staticmethod
    def run_sequential_compilation(work_items: list, worker_func) -> list[dict]:
        """Executes compilation tasks sequentially."""
        logger.info(f"Compiling {len(work_items)} slices sequentially...")
        return [worker_func(item) for item in work_items]

    @staticmethod
    def handle_channel_split_results(result: dict, info: dict):
        """Updates channel split metadata if applicable."""
        channel_group_circuits = result.get('channel_group_circuits', {})
        if channel_group_circuits and info.get('channel_split_info'):
            cs_info = info['channel_split_info']
            groups_by_idx = {g['group_idx']: g for g in cs_info.get('groups', [])}
            for be, group_results in channel_group_circuits.items():
                for gr in group_results:
                    if not (gr.get('success') and gr.get('files')):
                        continue
                    g = groups_by_idx.get(gr['group_idx'])
                    if not g:
                        continue
                    files = gr['files']
                    if be == Backend.JSTPROVE:
                        g['jstprove_circuit_path'] = files.get('compiled')
                        g['jstprove_settings_path'] = files.get('settings')
                    elif be == Backend.EZKL:
                        g['ezkl_circuit_path'] = files.get('compiled')
                        g['ezkl_settings_path'] = files.get('settings')
                        g['ezkl_vk_path'] = files.get('vk_key')
                        g['ezkl_pk_path'] = files.get('pk_key')
                        # For backward compatibility
                        g.setdefault('vk_path', files.get('vk_key'))
                        g.setdefault('pk_path', files.get('pk_key'))
                    
                    g.setdefault('settings_path', files.get('settings'))
            info['original_slice_entry']['channel_split'] = cs_info

    @staticmethod
    def record_onnx_fallback(idx: int, slice_data: dict, slice_meta_path: Path):
        """Records ONNX fallback in metadata for slices where ZK compilation failed."""
        onnx_block = {
            "compiled": True,
            "compilation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": Backend.ONNX,
            "backend_version": None,
            "files": {"skipped": True, "reason": "fallback_to_onnx"}
        }
        if 'compilation' not in slice_data or not isinstance(slice_data.get('compilation'), dict):
            slice_data['compilation'] = {}
        slice_data['compilation'][Backend.ONNX] = onnx_block
        if slice_meta_path.exists():
            CompilerUtils.update_slice_metadata(idx, slice_meta_path, True, onnx_block, backend_name=Backend.ONNX)

    @staticmethod
    def get_model_input_name(first_slice: dict) -> str:
        """Determines the name of the model's first input tensor."""
        deps = first_slice.get('dependencies', {})
        first_inputs = deps.get('filtered_inputs', deps.get('input', []))
        return first_inputs[0] if first_inputs else 'input'

    @staticmethod
    def get_slice_onnx_path(slice_data: dict, base_path: str, idx: int) -> Optional[str]:
        """Resolves the absolute path to a slice's ONNX file."""
        slice_path = slice_data.get('path')
        if slice_path and os.path.exists(slice_path):
            return slice_path
        elif slice_data.get('relative_path'):
            path = os.path.join(base_path, slice_data.get('relative_path'))
            return path if os.path.exists(path) else None
        return None

    @staticmethod
    def prepare_calibration_artifact(slice_path: str) -> str:
        """Ensures the ezkl directory exists and returns the calibration file path."""
        slice_output_path = os.path.join(os.path.dirname(slice_path), "ezkl")
        os.makedirs(slice_output_path, exist_ok=True)
        return os.path.join(slice_output_path, "calibration.json")

    @staticmethod
    def cache_calibration_outputs(idx: int, slice_data: dict, result: Any, tensor_cache: dict):
        """Processes and caches output tensors from a calibration inference step."""
        deps = slice_data.get('dependencies', {})
        output_names = deps.get('output', [])
        
        if isinstance(result, dict) and 'output_tensors' in result:
            for oname, tensor in result['output_tensors'].items():
                tensor_cache[oname] = tensor
        elif output_names:
            out_tensor = result.get('output') if isinstance(result, dict) else result
            if isinstance(out_tensor, torch.Tensor):
                for oname in output_names:
                    tensor_cache[oname] = out_tensor

    @staticmethod
    def setup_compilation_dir(slice_dir: str, backend: str, tiling_info: Optional[dict]) -> str:
        """Creates and returns the output directory for compilation artifacts."""
        output_dir = os.path.join(slice_dir, "payload", backend, "tiles" if tiling_info else "")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def get_calibration_input_path(output_dir: str) -> Optional[str]:
        """Returns the path to the calibration file if it exists in the output directory."""
        path = os.path.join(output_dir, "calibration.json")
        return path if os.path.exists(path) else None

    @staticmethod
    def resolve_slice_source_path(comp_slice_data: dict, base_path: str) -> Optional[str]:
        """Resolves the source ONNX path from compilation slice data."""
        path = comp_slice_data.get('path')
        if not path or not os.path.exists(path):
            if comp_slice_data.get('relative_path'):
                path = os.path.join(base_path, comp_slice_data.get('relative_path'))
        return path if path and os.path.exists(path) else None

    @staticmethod
    def resolve_group_onnx_path(group: dict, base_path: str) -> Optional[str]:
        """Resolves the absolute path to a channel group's ONNX file."""
        path = group.get('path', '')
        if not os.path.isabs(path):
            path = os.path.join(base_path, path)
        return path if os.path.exists(path) else None

    @staticmethod
    def setup_group_compilation_dir(slice_dir: str, backend: str, group_idx: int) -> str:
        """Creates and returns the output directory for a channel group's artifacts."""
        output_dir = os.path.join(slice_dir, "payload", backend, "channel_groups", f"group_{group_idx}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def build_channel_split_block(backend: str, num_groups: int, group_results: list, slice_dir: str, weights_as_inputs: bool = False) -> dict:
        """Constructs the metadata compilation block for a channel-split slice."""
        sdn = os.path.basename(slice_dir)
        def _prefix(p):
            if isinstance(p, str) and not p.startswith(sdn + os.sep):
                return os.path.join(sdn, p)
            return p

        block = {
            "compiled": True,
            "compilation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend,
            "channel_split": True,
            "num_groups": num_groups,
            "group_files": {
                g['group_idx']: {k: _prefix(v) for k, v in g.get('files', {}).items()}
                for g in group_results if g.get('success')
            }
        }
        if backend == Backend.JSTPROVE:
            block["weights_as_inputs"] = weights_as_inputs
        return block

    @staticmethod
    def is_compilation_successful(compilation_data: Dict[str, Any]) -> bool:
        """Determine if compilation was successful based on produced file paths."""
        def _ok(key: str) -> bool:
            p = compilation_data.get(key)
            return bool(p) and os.path.exists(p)

        # Check if this is a JSTprove compilation (has 'circuit' key, no 'vk_key'/'pk_key')
        if compilation_data.get('circuit') and not compilation_data.get('vk_key'):
            return _ok('compiled') and _ok('settings')

        # EZKL requires compiled, vk_key, pk_key, settings
        return all([_ok('compiled'), _ok('vk_key'), _ok('pk_key'), _ok('settings')])

    @staticmethod
    def process_compilation_result(result: dict, info: dict, stats: dict):
        """Handles updating metadata and stats from a single compilation result."""
        idx = result['idx']
        successful_backends = result.get('successful_backends', [])

        if successful_backends:
            stats['compiled_count'] += 1
            stats['backend_stats'][idx] = successful_backends

        slice_data = info['slice_data']
        slice_dir = info['slice_dir']
        slice_meta_path = Path(slice_dir) / "metadata.json"

        # --- Update compilation blocks ---
        comp_blocks = result.get('compilation_blocks', {})
        if 'compilation' not in slice_data:
            slice_data['compilation'] = {}
        for be, block in comp_blocks.items():
            slice_data['compilation'][be] = block
            if slice_meta_path.exists():
                try:
                    CompilerUtils.update_slice_metadata(idx, slice_meta_path, block.get('compiled', False), block,
                                                        backend_name=be)
                except Exception as e:
                    logger.warning(f"Failed to update slice metadata for slice {idx} backend {be}: {e}")

        # --- Post-process Specialized Results ---
        CompilerUtils.handle_channel_split_results(result, info)

        if not successful_backends:
            CompilerUtils.record_onnx_fallback(idx, slice_data, slice_meta_path)

        for error in result.get('errors', []):
            logger.error(f"Slice {idx} error: {error}")
