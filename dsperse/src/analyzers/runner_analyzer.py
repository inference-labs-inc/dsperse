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
    def process_slices(slices_dir, slices_data):
        """
        Build the slices dictionary with metadata for each slice.
        This is now a dispatcher that routes to:
        - per-model processing when `slices_data` entries already contain slice metadata
          (i.e., loaded from slices/metadata.json)
        - per-slice processing when `slices_data` entries contain a path to a per-slice
          metadata.json (i.e., discovered from slice_* directories)
        """

        # Normalize input to a list
        if len(slices_data[0]) == 3:
            if isinstance(slices_data, dict):
                slices_data = [slices_data]
            return RunnerAnalyzer._process_slices_per_slice(slices_dir, slices_data)
        else:
            return RunnerAnalyzer._process_slices_model(slices_dir, slices_data)

    @staticmethod
    def _process_slices_model(slices_dir: Path, slices_list: list[dict]) -> dict:
        """
        Build slices dict using model-level metadata entries from slices/metadata.json.
        Trust the provided metadata; do not perform filesystem checks.
        """
        slices: dict[str, dict] = {}

        for item in slices_list:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            slice_key = f"slice_{idx}"

            # Normalize ONNX path to 'slice_#/payload/...'
            rel_from_meta = item.get("relative_path") or item.get("path")
            rel_payload = RunnerAnalyzer.rel_from_payload(rel_from_meta) or rel_from_meta
            onnx_path = RunnerAnalyzer.with_slice_prefix(rel_payload, slice_key)

            # Shapes and dependencies
            tensor_shape = (item.get("shape") or {}).get("tensor_shape") or {}
            input_shape = tensor_shape.get("input")
            output_shape = tensor_shape.get("output")
            dependencies = item.get("dependencies") or {}
            parameters = item.get("parameters", 0)

            # Check for compilation info (JSTprove first, then EZKL)
            compilation = item.get("compilation") or {}
            jst_comp = compilation.get("jstprove") or {}
            ezkl_comp = compilation.get("ezkl") or {}
            
            # Prefer JSTprove if available, otherwise EZKL
            if jst_comp.get("compiled"):
                backend = "jstprove"
                files = jst_comp.get("files") or {}
                compiled_flag = True
                # Accept both keys: prefer 'compiled' (current), fallback to legacy 'circuit'
                compiled_rel = files.get("compiled") or files.get("circuit")
                settings_rel = files.get("settings")
                pk_rel = None
                vk_rel = None
            else:
                backend = "ezkl"
                files = ezkl_comp.get("files") or {}
                compiled_flag = bool(ezkl_comp.get("compiled", False))
                # Accept both keys: 'compiled_circuit' and legacy 'compiled'
                compiled_rel = files.get("compiled_circuit") or files.get("compiled")
                settings_rel = files.get("settings")
                pk_rel = files.get("pk_key")
                vk_rel = files.get("vk_key")

            def _norm(rel: Optional[str]) -> Optional[str]:
                if not rel:
                    return None
                return RunnerAnalyzer.with_slice_prefix(RunnerAnalyzer.rel_from_payload(rel) or rel, slice_key)

            circuit_path = _norm(compiled_rel)
            settings_path = _norm(settings_rel)
            pk_path = _norm(pk_rel)
            vk_path = _norm(vk_rel)

            # Also compute alternate backend circuit paths to enable multi-level fallbacks in run-metadata
            jst_files = (jst_comp or {}).get("files") or {}
            ezkl_files = (ezkl_comp or {}).get("files") or {}
            jst_circuit_rel = jst_files.get("compiled") or jst_files.get("circuit")
            ezkl_circuit_rel = ezkl_files.get("compiled_circuit") or ezkl_files.get("compiled")
            jstprove_circuit_path = _norm(jst_circuit_rel)
            ezkl_circuit_path = _norm(ezkl_circuit_rel)

            slice_meta_rel = item.get("slice_metadata_relative_path") or os.path.join(slice_key, "metadata.json")

            slices[slice_key] = {
                "path": onnx_path,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "ezkl_compatible": True,
                "ezkl": bool(compiled_flag),
                "backend": backend,
                "circuit_size": 0,  # unknown without touching filesystem; keep 0
                "dependencies": dependencies,
                "parameters": parameters,
                "circuit_path": circuit_path,
                "settings_path": settings_path,
                "vk_path": vk_path,
                "pk_path": pk_path,
                "slice_metadata_path": slice_meta_rel,
                # extra paths for multi-level fallback planning
                "jstprove_circuit_path": jstprove_circuit_path,
                "ezkl_circuit_path": ezkl_circuit_path,
            }

        return slices

    @staticmethod
    def _process_slices_per_slice(slices_dir: Path, slices_data_list: list[dict]) -> dict:
        """
        Build slices dict by reading each per-slice metadata.json referenced by entries.
        Trust the provided metadata for each slice; do not perform filesystem checks.
        """
        slices: dict[str, dict] = {}

        for entry in slices_data_list:
            meta_path = entry.get("slice_metadata")
            parent_dir = os.path.dirname(entry.get("path")).split(os.sep)[0]
            
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load slice metadata at {meta_path}: {e}")
                continue

            # Expect standard format with single slice in meta["slices"][0]
            slice = (meta.get("slices"))[0]

            idx = slice.get("index")
            slice_key = f"slice_{idx}"

            # Normalize ONNX path
            onnx_path = os.path.join(parent_dir, slice.get("relative_path"))

            # Shapes and dependencies
            tensor_shape = (slice.get("shape") or {}).get("tensor_shape") or {}
            input_shape = tensor_shape.get("input")
            output_shape = tensor_shape.get("output")
            dependencies = slice.get("dependencies") or {}
            parameters = slice.get("parameters", 0)

            # Check for compilation info (JSTprove first, then EZKL)
            compilation = slice.get("compilation") or {}
            jst_comp = compilation.get("jstprove") or {}
            ezkl_comp = compilation.get("ezkl") or {}
            
            if jst_comp.get("compiled"):
                backend = "jstprove"
                files = jst_comp.get("files") or {}
                compiled_flag = True
                if files:
                    circuit_rel = files.get("compiled") or files.get("circuit")
                    circuit_path = os.path.join(parent_dir, circuit_rel) if circuit_rel else None
                    settings_rel = files.get("settings")
                    settings_path = os.path.join(parent_dir, settings_rel) if settings_rel else None
                else:
                    circuit_path = None
                    settings_path = None
                pk_path = None
                vk_path = None
            else:
                backend = "ezkl"
                files = ezkl_comp.get("files") or {}
                compiled_flag = bool(ezkl_comp.get("compiled", False))
                if files:
                    circuit_path = os.path.join(parent_dir, files.get("compiled_circuit") or files.get("compiled"))
                    settings_path = os.path.join(parent_dir, files.get("settings"))
                    pk_path = os.path.join(parent_dir, files.get("pk_key"))
                    vk_path = os.path.join(parent_dir, files.get("vk_key"))
                else:
                    circuit_path = None
                    settings_path = None
                    pk_path = None
                    vk_path = None

            # Compute alternate backend circuit paths (slice-prefixed) for multi-level fallback
            jst_files = (jst_comp or {}).get("files") or {}
            ezkl_files = (ezkl_comp or {}).get("files") or {}
            jst_rel = jst_files.get("compiled") or jst_files.get("circuit")
            ezkl_rel = ezkl_files.get("compiled_circuit") or ezkl_files.get("compiled")
            jstprove_circuit_path = os.path.join(parent_dir, jst_rel) if jst_rel else None
            ezkl_circuit_path = os.path.join(parent_dir, ezkl_rel) if ezkl_rel else None

            slices[slice_key] = {
                "path": onnx_path,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "ezkl_compatible": True,
                "ezkl": bool(compiled_flag),
                "backend": backend,
                "circuit_size": 0,
                "dependencies": dependencies,
                "parameters": parameters,
                "circuit_path": circuit_path,
                "settings_path": settings_path,
                "vk_path": vk_path,
                "pk_path": pk_path,
                "slice_metadata_path": meta_path,
                # extra paths for multi-level fallback planning
                "jstprove_circuit_path": jstprove_circuit_path,
                "ezkl_circuit_path": ezkl_circuit_path,
            }

        return slices

    @staticmethod
    def build_run_metadata(slices_dir, slices_metadata: dict) -> dict:
        """Assemble run metadata dict from model-level slices metadata.
        Expects `slices_metadata` to contain a top-level 'slices' list as produced by slicing.
        """
        slices_data = (slices_metadata or {}).get('slices', [])
        slices = RunnerAnalyzer.process_slices(slices_dir, slices_data)
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
            # Map from primary path to an ordered list of fallback paths (e.g., [ezkl_circuit, onnx])
            "fallback_map": {}
        }

        for i, slice_key in enumerate(ordered_keys):
            meta = slices.get(slice_key, {})
            circuit_path = meta.get('circuit_path')
            onnx_path = meta.get('path')
            backend = meta.get('backend', 'ezkl')
            jst_circuit = meta.get('jstprove_circuit_path')
            ezkl_circuit = meta.get('ezkl_circuit_path')
            has_circuit = circuit_path is not None and circuit_path != ""
            has_keys = (meta.get('pk_path') is not None) and (meta.get('vk_path') is not None)
            # JSTprove doesn't require pk/vk keys; EZKL does
            use_circuit = bool(meta.get('ezkl')) and has_circuit and (backend == 'jstprove' or has_keys)

            next_slice = ordered_keys[i + 1] if i < len(ordered_keys) - 1 else None
            # Build ordered fallbacks: prefer EZKL circuit (when primary is JSTprove), then ONNX
            fallbacks = []
            if backend == 'jstprove' and ezkl_circuit:
                fallbacks.append(ezkl_circuit)
            # Always ensure ONNX is the last fallback
            if onnx_path:
                fallbacks.append(onnx_path)

            execution_chain["nodes"][slice_key] = {
                "slice_id": slice_key,
                "primary": circuit_path if use_circuit else onnx_path,
                "fallbacks": fallbacks if use_circuit else ([onnx_path] if onnx_path else []),
                "use_circuit": use_circuit,
                "next": next_slice,
                "circuit_path": circuit_path if has_circuit else None,
                "onnx_path": onnx_path,
                "backend": backend
            }

            # Populate fallback_map with ordered list
            if use_circuit and circuit_path:
                execution_chain["fallback_map"][circuit_path] = fallbacks
            elif onnx_path:
                execution_chain["fallback_map"][slice_key] = [onnx_path]

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

        if path_obj.is_dir() and detected == 'dslice':
            # a folder of dslice files, for each dslice, convert to dirs
            converted = Converter.convert(str(path_obj), output_type="dirs", cleanup=False)
            sdir = Path(converted)
            return sdir.resolve(), 'dslice'

        return (path_obj.parent / 'slices').resolve(), 'dirs'

    @staticmethod
    def _has_model_metadata(path: Path) -> bool:
        return ((path / "metadata.json").exists() or (path / "slices" / "metadata.json").exists()) and not (
                    path / "payload").exists()

    @staticmethod
    def _build_from_model_metadata(slices_dir: Path) -> dict:
        smeta = RunnerAnalyzer.load_slices_metadata(slices_dir)
        run_meta = RunnerAnalyzer.build_run_metadata(slices_dir, smeta)
        try:
            run_meta["model_path"] = str(slices_dir.parent.resolve())
        except Exception:
            pass
        return run_meta

    @staticmethod
    def _build_from_per_slice_dirs(slices_dir: Path) -> dict:
        subdirs = [d for d in slices_dir.iterdir() if d.is_dir()] if slices_dir.is_dir() else []

        # If only payload directory found, go up one level unless this is a single slice dir (has its own metadata.json)
        if len(subdirs) == 1 and subdirs[0].name == "payload" and not (slices_dir / "metadata.json").exists():
            slices_dir = slices_dir.parent
            subdirs = [d for d in slices_dir.iterdir() if d.is_dir()]

        # Treat a directory with its own metadata.json + payload/ as a single-slice dir,
        # regardless of other subdirectories present (e.g., payload only)
        if (slices_dir / "metadata.json").exists() and (slices_dir / "payload").exists():
            subdirs = [slices_dir]

        slices_data = []
        for d in subdirs:
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path, "r") as f:
                meta = json.load(f)
            s0 = meta["slices"][0]
            idx = int(s0["index"])  # index from metadata
            # two lines to get relative path and filename directly from metadata
            relpath = s0.get("relative_path") or s0.get("path")
            filename = s0.get("filename")
            payload_rel = RunnerAnalyzer.rel_from_payload(relpath) or relpath or os.path.join("payload", filename)
            onnx_path = RunnerAnalyzer.with_slice_prefix(payload_rel, d.name)
            slices_data.append({
                "index": idx,
                "path": onnx_path,
                "slice_metadata": str(meta_path.resolve()),
            })

        if not slices_data:
            raise Exception(f"No valid slices found under {slices_dir}. Each slice directory must include metadata.json.")

        slices = RunnerAnalyzer.process_slices(slices_dir, slices_data)
        head_nodes = RunnerAnalyzer._build_execution_chain(slices)
        circuit_slices = RunnerAnalyzer._build_circuit_slices(slices)
        overall_security = RunnerAnalyzer._calculate_security(slices)
        return {
            "overall_security": overall_security,
            "slices": slices,
            "execution_chain": head_nodes,
            "circuit_slices": circuit_slices,
        }

    @staticmethod
    def generate_run_metadata(slice_path: Path, save_path=None, original_format=None):
        """
        Build run-metadata from a slices source (dirs/.dslice/.dsperse) and save it.
        - Normalizes inputs to dirs temporarily when needed (no cleanup).
        - Prefers model-level slices/metadata.json when present; otherwise scans per-slice dirs.
        - Emits paths normalized as 'slice_#/payload/...'.
        - Saves to save_path or default '<parent_of_slice_path>/run/metadata.json'.
        - Converts back to original packaging when original_format != 'dirs'.
        Returns the run-metadata dict.
        """

        if RunnerAnalyzer._has_model_metadata(slice_path):
            run_meta = RunnerAnalyzer._build_from_model_metadata(slice_path)
        else:
            run_meta = RunnerAnalyzer._build_from_per_slice_dirs(slice_path)

        # Attach packaging metadata
        run_meta["packaging_type"] = original_format
        run_meta["source_path"] = str(slice_path)

        # Save
        if save_path is None:
            base = Path(slice_path).resolve()
            # If slice_path is a model root or slices dir, put run/metadata.json next to it
            base_dir = base if base.is_dir() else base.parent
            save_path = base_dir / "run" / "metadata.json"
        else:
            save_path = Path(save_path).resolve()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        run_meta["run_directory"] = str(save_path.parent)
        Utils.save_metadata_file(run_meta, save_path)

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
