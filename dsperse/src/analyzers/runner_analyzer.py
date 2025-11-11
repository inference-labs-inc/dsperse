"""
Generates execution chain metadata for EzKL circuit and ONNX slice inference
with proper fallback mapping and security calculation.
"""
import logging
import json
import os
from pathlib import Path
from typing import Optional

from dsperse.src.utils.utils import Utils
from dsperse.src.slice.utils.converter import Converter

logger = logging.getLogger(__name__)

class RunnerAnalyzer:
    SIZE_LIMIT = 1000 * 1024 * 1024  # 1000MB

    def __init__(self):
        """Stateless analyzer. Use static methods."""
        pass

    # ---------- Small path helpers ----------
    @staticmethod
    def rel_from_payload(path: str) -> Optional[str]:
        if not path:
            return None
        parts = str(path).split(os.sep)
        try:
            i = parts.index('payload')
            return os.path.join(*parts[i:])
        except ValueError:
            return None

    @staticmethod
    def with_slice_prefix(rel_path: Optional[str], slice_dirname: str) -> Optional[str]:
        if not rel_path:
            return None
        return os.path.join(slice_dirname, rel_path)


    @staticmethod
    def load_slices_metadata(slices_dir: Path):
        """Load model-level slices metadata from <slices_dir>/metadata.json."""
        try:
            with open(Path(slices_dir) / 'metadata.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load slices metadata from {slices_dir}: {e}")
            raise


    @staticmethod
    def process_slices(slices_data):
        """
        Build the slices dictionary with metadata for each slice.
        Reads per-slice metadata from slice_#/metadata.json and adapts to new layout.
        Emits EZKL artifact paths in run metadata as 'slice_#/payload/...'.
        """

        slices = {}
        for slice_data in slices_data:
            slice_idx = slice_data.get('index')
            slice_key = f"slice_{slice_idx}"

            onnx_slice_path = slice_data.get('path', '')
            if not onnx_slice_path:
                logger.warning(f"No ONNX slice path for slice {slice_idx}")

            # Resolve per-slice metadata path
            slice_meta_path = slice_data.get('slice_metadata')
            if not slice_meta_path and onnx_slice_path:
                try:
                    p = Path(onnx_slice_path)
                    # Expecting .../slice_#/payload/slice_#.onnx -> slice dir is parent of payload
                    slice_dir = p.parent.parent if p.parent.name == 'payload' else p.parent
                    candidate = slice_dir / 'metadata.json'
                    if candidate.exists():
                        slice_meta_path = str(candidate.resolve())
                except Exception:
                    slice_meta_path = None

            # Load slice-level metadata if available
            slice_level_meta = {}
            if slice_meta_path and os.path.exists(slice_meta_path):
                try:
                    with open(slice_meta_path, 'r') as sf:
                        slice_level_meta = json.load(sf)
                except Exception as e:
                    logger.warning(f"Failed to read slice metadata at {slice_meta_path}: {e}")

            # Extract IO shapes and deps
            io_meta = slice_level_meta.get('io', {}) if isinstance(slice_level_meta, dict) else {}
            deps_meta = slice_level_meta.get('deps', {}) if isinstance(slice_level_meta, dict) else {}
            input_shape = io_meta.get('input_shape') or slice_data.get('shape', {}).get('tensor_shape', {}).get('input', ["batch_size", "unknown"])
            output_shape = io_meta.get('output_shape') or slice_data.get('shape', {}).get('tensor_shape', {}).get('output', ["batch_size", "unknown"])

            # Extract EZKL compilation paths (prefer per-slice, then model-level nested, then flat)
            comp = slice_level_meta.get('compilation', {}).get('ezkl', {}) if isinstance(slice_level_meta, dict) else {}
            files = comp.get('files', {}) if isinstance(comp, dict) else {}
            model_comp = slice_data.get('compilation', {}).get('ezkl', {}) if isinstance(slice_data, dict) else {}
            model_files = model_comp.get('files', {}) if isinstance(model_comp, dict) else {}
            flat_model_ezkl = slice_data.get('ezkl', {}) if isinstance(slice_data, dict) else {}

            compiled_circuit_path = (
                files.get('compiled_circuit')
                or model_files.get('compiled_circuit')
                or flat_model_ezkl.get('compiled')
                or flat_model_ezkl.get('compiled_circuit')
            )
            settings_path = files.get('settings') or model_files.get('settings') or flat_model_ezkl.get('settings')
            pk_path = files.get('pk_key') or model_files.get('pk_key') or flat_model_ezkl.get('pk_key')
            vk_path = files.get('vk_key') or model_files.get('vk_key') or flat_model_ezkl.get('vk_key')

            # compiled_flag is True if per-slice marks compiled or model-level marks compiled, or if we have a compiled artifact path
            compiled_flag = bool(comp.get('compiled', False) or model_comp.get('compiled', False) or compiled_circuit_path)

            # Resolve absolute paths for existence checks using slice dir
            slice_dir_abs = None
            if slice_meta_path:
                try:
                    slice_dir_abs = Path(slice_meta_path).parent
                except Exception:
                    slice_dir_abs = None
            def _resolve_abs(p: str) -> str | None:
                if not p:
                    return None
                p_str = str(p)
                if os.path.isabs(p_str):
                    return p_str
                if slice_dir_abs is None:
                    return p_str
                # If starts with slice_key, strip it
                parts = p_str.split(os.sep)
                if parts and parts[0] == slice_key:
                    parts = parts[1:]
                    p_str = os.path.join(*parts) if parts else ''
                return str((slice_dir_abs / p_str).resolve())

            compiled_abs = _resolve_abs(compiled_circuit_path)
            settings_abs = _resolve_abs(settings_path)
            pk_abs = _resolve_abs(pk_path)
            vk_abs = _resolve_abs(vk_path)

            # Check existence on disk (when slices are extracted as directories)
            circuit_exists = bool(compiled_abs) and os.path.exists(compiled_abs) and bool(settings_abs) and os.path.exists(settings_abs)
            keys_exist = bool(pk_abs) and os.path.exists(pk_abs) and bool(vk_abs) and os.path.exists(vk_abs)

            # Determine if artifacts are only packaged (e.g., .dslice files present, no slice dir on disk)
            slice_dir_exists = bool(slice_dir_abs and slice_dir_abs.exists())
            paths_present = bool(compiled_circuit_path) and bool(settings_path)
            keys_present = bool(pk_path) and bool(vk_path)
            packaged_only = (not slice_dir_exists) and paths_present and keys_present and compiled_flag

            # Determine circuit size (only when files exist on disk)
            circuit_size = 0
            if circuit_exists:
                try:
                    circuit_size = Path(compiled_abs).stat().st_size
                except Exception:
                    circuit_size = 0

            # Trust metadata when packaged_only; otherwise require physical files and size limit
            if compiled_flag:
                if packaged_only:
                    use_circuit = True
                else:
                    use_circuit = circuit_exists and keys_exist and circuit_size <= RunnerAnalyzer.SIZE_LIMIT
            else:
                use_circuit = False

            # Normalize ONNX path to 'slice_#/payload/...'
            def _resolve_abs_for_onnx(p: str) -> str:
                if not p:
                    return None
                p_str = str(p)
                if os.path.isabs(p_str):
                    return p_str
                if slice_dir_abs is None:
                    return p_str
                parts = p_str.split(os.sep)
                if parts and parts[0] == slice_key:
                    parts = parts[1:]
                    p_str = os.path.join(*parts) if parts else ''
                return str((slice_dir_abs / p_str).resolve())

            onnx_abs = _resolve_abs_for_onnx(onnx_slice_path)
            onnx_rel = RunnerAnalyzer.rel_from_payload(onnx_abs) or RunnerAnalyzer.rel_from_payload(onnx_slice_path)
            normalized_onnx_path = RunnerAnalyzer.with_slice_prefix(onnx_rel, slice_key) if onnx_rel else onnx_slice_path

            # Build slice metadata
            slice_metadata = {
                "path": normalized_onnx_path,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "ezkl_compatible": True,
                "ezkl": use_circuit,
                "circuit_size": circuit_size,
                "dependencies": slice_data.get('dependencies') or deps_meta,
                "parameters": slice_data.get('parameters', 0)
            }

            # Add circuit paths (emit as 'slice_#/payload/...')
            if circuit_exists or packaged_only:
                comp_rel = RunnerAnalyzer.rel_from_payload(compiled_abs) or RunnerAnalyzer.rel_from_payload(compiled_circuit_path)
                set_rel = RunnerAnalyzer.rel_from_payload(settings_abs) or RunnerAnalyzer.rel_from_payload(settings_path)
                slice_metadata.update({
                    "circuit_path": RunnerAnalyzer.with_slice_prefix(comp_rel, slice_key),
                    "settings_path": RunnerAnalyzer.with_slice_prefix(set_rel, slice_key)
                })
                if keys_exist or (packaged_only and keys_present):
                    vk_rel = RunnerAnalyzer.rel_from_payload(vk_abs) or RunnerAnalyzer.rel_from_payload(vk_path)
                    pk_rel = RunnerAnalyzer.rel_from_payload(pk_abs) or RunnerAnalyzer.rel_from_payload(pk_path)
                    slice_metadata.update({
                        "vk_path": RunnerAnalyzer.with_slice_prefix(vk_rel, slice_key),
                        "pk_path": RunnerAnalyzer.with_slice_prefix(pk_rel, slice_key)
                    })

            # Include slice metadata path as relative
            slice_metadata["slice_metadata_path"] = os.path.join(slice_key, "metadata.json")

            slices[slice_key] = slice_metadata

        return slices

    @staticmethod
    def build_run_metadata(slices_metadata: dict) -> dict:
        """Assemble run metadata dict from model-level slices metadata.
        Expects `slices_metadata` to contain a top-level 'slices' list as produced by slicing.
        """
        slices_data = (slices_metadata or {}).get('slices', [])
        slices = RunnerAnalyzer.process_slices(slices_data)
        execution_chain = RunnerAnalyzer._build_execution_chain(slices)
        circuit_slices = RunnerAnalyzer._build_circuit_slices(slices)
        overall_security = RunnerAnalyzer._calculate_security(slices)
        return {
            "overall_security": overall_security,
            "slices": slices,
            "execution_chain": execution_chain,
            "circuit_slices": circuit_slices,
        }

    @staticmethod
    def _build_execution_chain(slices: dict):
        """
        Build the execution chain with proper node connections and fallback mapping,
        using new slice_* ids and per-slice metadata.
        Note: artifact paths in run metadata may be 'slice_#/payload/...'; we should not
        perform filesystem existence checks here. Trust the computed 'ezkl' flag.
        """
        # Order slices by numeric index extracted from key 'slice_#'
        ordered_keys = sorted(slices.keys(), key=lambda k: int(str(k).split('_')[-1])) if slices else []

        execution_chain = {
            "head": ordered_keys[0] if ordered_keys else None,
            "nodes": {},
            "fallback_map": {}
        }

        for i, slice_key in enumerate(ordered_keys):
            meta = slices.get(slice_key, {})
            circuit_path = meta.get('circuit_path')
            onnx_path = meta.get('path')
            has_circuit = circuit_path is not None and circuit_path != ""
            has_keys = (meta.get('pk_path') is not None) and (meta.get('vk_path') is not None)
            use_circuit = bool(meta.get('ezkl')) and has_circuit and has_keys

            next_slice = ordered_keys[i + 1] if i < len(ordered_keys) - 1 else None
            execution_chain["nodes"][slice_key] = {
                "slice_id": slice_key,
                "primary": circuit_path if use_circuit else onnx_path,
                "fallback": onnx_path,
                "use_circuit": use_circuit,
                "next": next_slice,
                "circuit_path": circuit_path if has_circuit else None,
                "onnx_path": onnx_path
            }

            if has_circuit and onnx_path:
                execution_chain["fallback_map"][circuit_path] = onnx_path
            elif onnx_path:
                execution_chain["fallback_map"][slice_key] = onnx_path

        return execution_chain

    @staticmethod
    def _build_circuit_slices(slices):
        """
        Build dictionary tracking which slices use circuits.
        """
        circuit_slices = {}
        for slice_key, slice_data in slices.items():
            # Trust the computed 'ezkl' flag which already considers compiled, keys, and size limits
            circuit_slices[slice_key] = bool(slice_data.get("ezkl", False))

        return circuit_slices

    @staticmethod
    def get_execution_chain(run_metadata: dict):
        """Return (head, nodes) from run metadata's execution_chain."""
        ec = (run_metadata or {}).get("execution_chain") or {}
        return ec.get("head"), ec.get("nodes") or {}

    @staticmethod
    def _calculate_security(slices):
        if not slices:
            return 0.0
        total_slices = len(slices)
        circuit_slices = sum(1 for slice_data in slices.values() if slice_data.get("ezkl", False))
        return round((circuit_slices / total_slices) * 100, 1)

    @staticmethod
    def _normalize_to_dirs(slice_path: str):
        """Normalize input to (model_root, slices_dir, original_format).
        Handles:
        - slices dir containing slice_* dirs
        - slices dir containing .dslice files + metadata.json (no conversion)
        - single slice directory (payload + metadata.json)
        - model root containing 'slices/'
        - single .dslice file or .dsperse archive (convert to dirs temporarily)
        """
        path_obj = Path(slice_path)
        original_format = 'dirs'

        # Directory-first handling for readability
        if path_obj.is_dir():
            # Case: model root with 'slices/metadata.json'
            if (path_obj / 'slices' / 'metadata.json').exists():
                sdir = (path_obj / 'slices').resolve()
                # Mixed layout allowed: .dslice files under slices/
                return sdir, 'dirs'

            # Case: provided a slices directory directly
            if (path_obj / 'metadata.json').exists():
                # If this directory has .dslice files at root, do NOT convert. Treat as slices dir.
                try:
                    has_dslice_files = any(f.is_file() and f.suffix == '.dslice' for f in path_obj.iterdir())
                except Exception:
                    has_dslice_files = False
                if has_dslice_files:
                    return path_obj.resolve(), 'dirs'

            # If it contains slice_* directories, treat as slices dir
            try:
                if any(d.is_dir() and d.name.startswith('slice_') for d in path_obj.iterdir()):
                    return path_obj.resolve(), 'dirs'
            except Exception:
                pass

            # If it is a single slice directory (has metadata.json + payload)
            if (path_obj / 'metadata.json').exists() and (path_obj / 'payload').exists():
                return path_obj.resolve(), 'dirs'

        # File-based handling (or unknown dir): detect and convert when needed
        detected = None
        try:
            detected = Converter.detect_type(path_obj)
        except Exception:
            detected = None

        # Only convert when the source itself is a file, or an explicit compressed type
        if path_obj.is_file() and detected in ['dslice', 'dsperse']:
            original_format = detected
            logger.info(f"Converting {path_obj} to directory format")
            converted = Converter.convert(str(path_obj), output_type="dirs", cleanup=False)
            sdir = Path(converted)
            return sdir.resolve(), original_format

        # Directory recognized by Converter as 'dirs' (slice dir or slices folder)
        if detected == 'dirs':
            sdir = path_obj
            # If this looks like a slices folder (has metadata.json), parent is model root
            model_root = sdir.parent if (sdir / 'metadata.json').exists() else sdir.parent
            return sdir.resolve(), 'dirs'

        # Fallbacks
        if path_obj.is_dir() and (path_obj / 'slices').is_dir():
            sdir = (path_obj / 'slices').resolve()
            return sdir, 'dirs'

        return (path_obj.parent / 'slices').resolve(), 'dirs'

    @staticmethod
    def _has_model_metadata(path: Path) -> bool:
        p = Path(path)
        # True if provided a slices directory with metadata.json, or a model root with slices/metadata.json
        return (p / "metadata.json").exists() or (p / "slices" / "metadata.json").exists()

    @staticmethod
    def _build_from_model_metadata(slices_dir: Path) -> dict:
        smeta = RunnerAnalyzer.load_slices_metadata(slices_dir)
        run_meta = RunnerAnalyzer.build_run_metadata(smeta)
        try:
            run_meta["model_path"] = str(slices_dir.parent.resolve())
        except Exception:
            pass
        return run_meta

    @staticmethod
    def _build_from_per_slice_dirs(slices_dir: Path) -> dict:
        # Collect slice_* directories (or treat slices_dir itself if single-slice)
        try:
            slice_dirs = sorted(
                [d for d in slices_dir.iterdir() if d.is_dir() and d.name.startswith("slice_")],
                key=lambda d: int(d.name.split('_')[-1])
            )
        except Exception:
            slice_dirs = []
        if not slice_dirs and (slices_dir / "metadata.json").exists() and (slices_dir / "payload").exists():
            slice_dirs = [slices_dir]

        slices_data = []
        for d in slice_dirs:
            name = d.name
            try:
                idx = int(name.split('_')[-1])
            except Exception:
                continue
            # Find ONNX in payload
            onnx_path = None
            try:
                payload = d / "payload"
                if payload.is_dir():
                    cand = next((p for p in payload.glob("*.onnx")), None)
                    if cand:
                        onnx_path = os.path.join(name, "payload", cand.name)
            except Exception:
                pass
            slices_data.append({
                "index": idx,
                "path": onnx_path,
                "slice_metadata": str((d / "metadata.json").resolve())
            })

        slices = RunnerAnalyzer.process_slices(slices_data)
        execution_chain = RunnerAnalyzer._build_execution_chain(slices)
        circuit_slices = RunnerAnalyzer._build_circuit_slices(slices)
        overall_security = RunnerAnalyzer._calculate_security(slices)
        return {
            "overall_security": overall_security,
            "slices": slices,
            "execution_chain": execution_chain,
            "circuit_slices": circuit_slices,
        }

    @staticmethod
    def generate_run_metadata(slice_path: str, save_path=None):
        """
        Build run-metadata from a slices source (dirs/.dslice/.dsperse) and save it.
        - Normalizes inputs to dirs temporarily when needed (no cleanup).
        - Prefers model-level slices/metadata.json when present; otherwise scans per-slice dirs.
        - Emits paths normalized as 'slice_#/payload/...'.
        - Saves to save_path or default '<parent_of_slice_path>/run/metadata.json'.
        - Converts back to original packaging when original_format != 'dirs'.
        Returns the run-metadata dict.
        """

        original_format = 'dirs'
        slices_dir: Optional[Path] = None
        packaging_type = 'dirs'
        source_path = str(Path(slice_path).resolve())
        # Check for model-level metadata before attempting normalization/conversion
        if RunnerAnalyzer._has_model_metadata(Path(slice_path)):
            p = Path(slice_path)
            slices_dir = p if (p / 'metadata.json').exists() else (p / 'slices')
            run_meta = RunnerAnalyzer._build_from_model_metadata(slices_dir)
            # Determine packaging type from contents of slices_dir
            try:
                has_slice_dirs = any(d.is_dir() and d.name.startswith('slice_') for d in slices_dir.iterdir())
                has_dslice = any(f.is_file() and f.suffix == '.dslice' for f in slices_dir.iterdir())
                packaging_type = 'dirs' if has_slice_dirs or not has_dslice else 'dslice'
            except Exception:
                packaging_type = 'dirs'
        else:
            slices_dir, original_format = RunnerAnalyzer._normalize_to_dirs(slice_path)
            run_meta = RunnerAnalyzer._build_from_per_slice_dirs(slices_dir)
            packaging_type = original_format or 'dirs'

        # Ensure model_path is present and points to model root
        try:
            if slices_dir is not None:
                sd = Path(slices_dir)
                if sd.name.startswith("slice_"):
                    model_root = sd.parent.parent
                elif sd.name == "slices":
                    model_root = sd.parent
                else:
                    model_root = sd.parent
                run_meta["model_path"] = str(model_root.resolve())
        except Exception:
            pass

        # Attach packaging metadata
        run_meta["packaging_type"] = packaging_type
        run_meta["source_path"] = source_path

        # Save
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        run_meta["run_directory"] = str(save_path.parent)
        Utils.save_metadata_file(run_meta, save_path)

        # Convert back if we converted in
        if original_format != 'dirs':
            try:
                Converter.convert(str(slices_dir), output_type=original_format, cleanup=True)
            except Exception:
                logger.warning("Failed to convert slices back to original format; continuing.")

        return run_meta


if __name__ == "__main__":
    model_choice = 1
    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet"
    }
    model_dir = base_paths[model_choice]
    model_path = Path(model_dir).resolve()
    print(f"Model path: {model_path}")
    out = RunnerAnalyzer.generate_run_metadata(str(model_path))
    print(json.dumps(out, indent=2)[:500] + "...")
