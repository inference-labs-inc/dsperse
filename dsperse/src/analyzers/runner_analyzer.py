"""
Generates execution chain metadata for EzKL circuit and ONNX slice inference
with proper fallback mapping and security calculation.
"""
import logging
import json
import os
from pathlib import Path
from typing import Optional

from dsperse.src.analyzers.schema import SliceMetadata, RunSliceMetadata, Compilation, Dependencies, TilingInfo, Backend, ExecutionNode, ExecutionChain, RunMetadata
from dsperse.src.utils.utils import Utils
from dsperse.src.slice.utils.converter import Converter

logger = logging.getLogger(__name__)

class RunnerAnalyzer:
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
        # Avoid duplicate prefixing if the path already starts with the slice directory name
        p_str = str(rel_path)
        if p_str.startswith(slice_dirname + os.sep) or p_str == slice_dirname:
            return rel_path
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
        if isinstance(slices_data, dict):
            slices_data = [slices_data]

        first = slices_data[0] if slices_data else {}
        is_per_slice = isinstance(first, dict) and "slice_metadata" in first and "dependencies" not in first
        if is_per_slice:
            return RunnerAnalyzer._process_slices_per_slice(slices_dir, slices_data)
        return RunnerAnalyzer._process_slices_model(slices_dir, slices_data)

    @staticmethod
    def _process_slices_model(slices_dir: Path, slices_list: list[dict]) -> dict[str, RunSliceMetadata]:
        """
        Build slices dict using model-level metadata entries from slices/metadata.json.
        Trust the provided metadata; do not perform filesystem checks.
        """
        slices: dict[str, RunSliceMetadata] = {}

        for item in slices_list:
            if not isinstance(item, dict):
                continue

            meta = SliceMetadata.from_dict(item)
            slice_key = f"slice_{meta.index}"

            rel_from_meta = meta.relative_path or meta.path
            rel_payload = RunnerAnalyzer.rel_from_payload(rel_from_meta) or rel_from_meta
            onnx_path = RunnerAnalyzer.with_slice_prefix(rel_payload, slice_key)

            def _norm(rel: Optional[str]) -> Optional[str]:
                if not rel:
                    return None
                return RunnerAnalyzer.with_slice_prefix(RunnerAnalyzer.rel_from_payload(rel) or rel, slice_key)

            jst = meta.compilation.jstprove
            ezkl_comp = meta.compilation.ezkl

            if jst.compiled:
                backend = Backend.JSTPROVE
                compiled_flag = True
                compiled_rel = jst.files.compiled
                settings_rel = jst.files.settings
                pk_rel = None
                vk_rel = None
            elif ezkl_comp.compiled:
                backend = Backend.EZKL
                compiled_flag = True
                compiled_rel = ezkl_comp.files.compiled
                settings_rel = ezkl_comp.files.settings
                pk_rel = ezkl_comp.files.pk_key
                vk_rel = ezkl_comp.files.vk_key
            else:
                backend = Backend.ONNX
                compiled_flag = False
                compiled_rel = None
                settings_rel = None
                pk_rel = None
                vk_rel = None

            circuit_path = _norm(compiled_rel)
            settings_path = _norm(settings_rel)
            pk_path = _norm(pk_rel)
            vk_path = _norm(vk_rel)

            jstprove_circuit_path = _norm(jst.files.compiled)
            ezkl_circuit_path = _norm(ezkl_comp.files.compiled)
            jstprove_settings_path = _norm(jst.files.settings)
            ezkl_settings_path = _norm(ezkl_comp.files.settings)
            ezkl_pk_path = _norm(ezkl_comp.files.pk_key)
            ezkl_vk_path = _norm(ezkl_comp.files.vk_key)

            # Ensure tiling and channel split paths are also prefixed
            if meta.tiling:
                if meta.tiling.tile:
                    meta.tiling.tile.path = _norm(meta.tiling.tile.path)
                if meta.tiling.tiles:
                    for t in meta.tiling.tiles:
                        t.path = _norm(t.path)

            if meta.channel_split:
                for group in meta.channel_split.groups:
                    group.transform_paths(_norm)
                if meta.channel_split.bias_path:
                    meta.channel_split.bias_path = _norm(meta.channel_split.bias_path)

            slice_meta_rel = item.get("slice_metadata_relative_path") or os.path.join(slice_key, "metadata.json")

            slices[slice_key] = RunSliceMetadata(
                path=onnx_path or "",
                input_shape=meta.input_shape,
                output_shape=meta.output_shape,
                ezkl_compatible=True,
                ezkl=bool(compiled_flag),
                backend=backend,
                circuit_size=0,
                dependencies=meta.dependencies,
                parameters=meta.parameters,
                circuit_path=circuit_path,
                settings_path=settings_path,
                vk_path=vk_path,
                pk_path=pk_path,
                slice_metadata_path=slice_meta_rel,
                jstprove_circuit_path=jstprove_circuit_path,
                ezkl_circuit_path=ezkl_circuit_path,
                jstprove_settings_path=jstprove_settings_path,
                ezkl_settings_path=ezkl_settings_path,
                ezkl_pk_path=ezkl_pk_path,
                ezkl_vk_path=ezkl_vk_path,
                tiling=meta.tiling,
                channel_split=meta.channel_split,
            )

        return slices

    @staticmethod
    def _process_slices_per_slice(slices_dir: Path, slices_data_list: list[dict]) -> dict[str, RunSliceMetadata]:
        """
        Build slices dict by reading each per-slice metadata.json referenced by entries.
        Trust the provided metadata for each slice; do not perform filesystem checks.
        """
        slices: dict[str, RunSliceMetadata] = {}

        for entry in slices_data_list:
            meta_path = entry.get("slice_metadata")
            parent_dir = os.path.dirname(entry.get("path")).split(os.sep)[0]

            try:
                with open(meta_path, "r") as f:
                    raw_meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load slice metadata at {meta_path}: {e}")
                continue

            slice_raw = (raw_meta.get("slices") or [])[0]
            meta = SliceMetadata.from_dict(slice_raw)
            slice_key = f"slice_{meta.index}"

            onnx_path = os.path.join(parent_dir, meta.relative_path)

            def _join(rel: Optional[str]) -> Optional[str]:
                if not rel:
                    return None
                if rel.split(os.sep)[0] == parent_dir:
                    return rel
                return os.path.join(parent_dir, rel)

            jst = meta.compilation.jstprove
            ezkl_comp = meta.compilation.ezkl

            if jst.compiled:
                backend = Backend.JSTPROVE
                compiled_flag = True
                circuit_path = _join(jst.files.compiled)
                settings_path = _join(jst.files.settings)
                pk_path = None
                vk_path = None
            elif ezkl_comp.compiled:
                backend = Backend.EZKL
                compiled_flag = True
                circuit_path = _join(ezkl_comp.files.compiled)
                settings_path = _join(ezkl_comp.files.settings)
                pk_path = _join(ezkl_comp.files.pk_key)
                vk_path = _join(ezkl_comp.files.vk_key)
            else:
                backend = Backend.ONNX
                compiled_flag = False
                circuit_path = None
                settings_path = None
                pk_path = None
                vk_path = None

            jstprove_circuit_path = _join(jst.files.compiled)
            ezkl_circuit_path = _join(ezkl_comp.files.compiled)
            jstprove_settings_path = _join(jst.files.settings)
            ezkl_settings_path = _join(ezkl_comp.files.settings)
            ezkl_pk_path = _join(ezkl_comp.files.pk_key)
            ezkl_vk_path = _join(ezkl_comp.files.vk_key)

            # Recursively join nested paths for tiling and channel split
            if meta.tiling:
                if meta.tiling.tile:
                    meta.tiling.tile.path = _join(meta.tiling.tile.path)
                if meta.tiling.tiles:
                    for t in meta.tiling.tiles:
                        t.path = _join(t.path)

            if meta.channel_split:
                for group in meta.channel_split.groups:
                    group.transform_paths(_join)
                if meta.channel_split.bias_path:
                    meta.channel_split.bias_path = _join(meta.channel_split.bias_path)

            slices[slice_key] = RunSliceMetadata(
                path=onnx_path,
                input_shape=meta.input_shape,
                output_shape=meta.output_shape,
                ezkl_compatible=True,
                ezkl=bool(compiled_flag),
                backend=backend,
                circuit_size=0,
                dependencies=meta.dependencies,
                parameters=meta.parameters,
                circuit_path=circuit_path,
                settings_path=settings_path,
                vk_path=vk_path,
                pk_path=pk_path,
                slice_metadata_path=meta_path,
                jstprove_circuit_path=jstprove_circuit_path,
                ezkl_circuit_path=ezkl_circuit_path,
                jstprove_settings_path=jstprove_settings_path,
                ezkl_settings_path=ezkl_settings_path,
                ezkl_pk_path=ezkl_pk_path,
                ezkl_vk_path=ezkl_vk_path,
                tiling=meta.tiling,
                channel_split=meta.channel_split,
            )

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
        run_meta = RunMetadata(
            slices=slices,
            execution_chain=execution_chain,
            circuit_slices=circuit_slices,
            overall_security=overall_security,
        )
        return run_meta.to_dict()

    @staticmethod
    def _build_execution_chain(slices: dict[str, RunSliceMetadata]) -> ExecutionChain:
        """
        Build the execution chain with proper node connections and fallback mapping,
        using new slice_* ids and per-slice metadata.
        Note: artifact paths in run metadata may be 'slice_#/payload/...'; we should not
        perform filesystem existence checks here. Trust the computed 'ezkl' flag.
        """
        ordered_keys = sorted(slices.keys(), key=lambda k: int(str(k).split('_')[-1])) if slices else []

        nodes: dict[str, ExecutionNode] = {}
        fallback_map: dict[str, list[str]] = {}

        for i, slice_key in enumerate(ordered_keys):
            meta = slices.get(slice_key)
            if meta is None:
                continue
            circuit_path = meta.circuit_path
            onnx_path = meta.path
            backend = meta.backend or Backend.ONNX
            jst_circuit = meta.jstprove_circuit_path
            ezkl_circuit = meta.ezkl_circuit_path
            has_jst = bool(jst_circuit)
            has_ezkl = bool(ezkl_circuit) and (bool(meta.ezkl_vk_path) or bool(meta.vk_path))
            use_circuit = has_jst or has_ezkl

            next_slice = ordered_keys[i + 1] if i < len(ordered_keys) - 1 else None
            fallbacks = []
            if backend == Backend.JSTPROVE and ezkl_circuit:
                fallbacks.append(ezkl_circuit)
            if onnx_path:
                fallbacks.append(onnx_path)

            nodes[slice_key] = ExecutionNode(
                slice_id=slice_key,
                primary=circuit_path if use_circuit else onnx_path,
                fallbacks=fallbacks if use_circuit else ([onnx_path] if onnx_path else []),
                use_circuit=use_circuit,
                next=next_slice,
                circuit_path=circuit_path if circuit_path else None,
                onnx_path=onnx_path,
                backend=backend,
            )

            if use_circuit and circuit_path:
                fallback_map[circuit_path] = fallbacks
            elif onnx_path:
                fallback_map[slice_key] = [onnx_path]

        return ExecutionChain(
            head=ordered_keys[0] if ordered_keys else None,
            nodes=nodes,
            fallback_map=fallback_map,
        )

    @staticmethod
    def _build_circuit_slices(slices: dict[str, RunSliceMetadata]):
        """
        Build dictionary tracking which slices use circuits.
        """
        circuit_slices = {}
        for slice_key, slice_data in slices.items():
            circuit_slices[slice_key] = slice_data.ezkl

        return circuit_slices

    @staticmethod
    def get_execution_chain(run_metadata: dict):
        """Return (head, nodes) from run metadata's execution_chain."""
        ec = (run_metadata or {}).get("execution_chain") or {}
        return ec.get("head"), ec.get("nodes") or {}

    @staticmethod
    def _calculate_security(slices: dict[str, RunSliceMetadata]):
        if not slices:
            return 0.0
        total_slices = len(slices)
        circuit_slices = sum(1 for slice_data in slices.values() if slice_data.ezkl)
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
        execution_chain = RunnerAnalyzer._build_execution_chain(slices)
        circuit_slices = RunnerAnalyzer._build_circuit_slices(slices)
        overall_security = RunnerAnalyzer._calculate_security(slices)
        run_meta = RunMetadata(
            slices=slices,
            execution_chain=execution_chain,
            circuit_slices=circuit_slices,
            overall_security=overall_security,
        )
        return run_meta.to_dict()

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

    @staticmethod
    def initialize_run_metadata(slices_path: Path, run_dir: Path = None, output_path: str = None, format: str = "dirs") -> tuple[Path, bool, RunMetadata]:
        """Set up the run directory and prepare the run metadata, handling resumes if applicable."""
        import time
        from dsperse.src.run.utils.runner_utils import RunnerUtils

        # 1. Determine base run directory logic
        requested_dir = Path(output_path) if output_path else run_dir
        resume = False
        
        if requested_dir:
            if (requested_dir / "metadata.json").exists():
                actual_run_dir = requested_dir
                resume = True
            elif requested_dir.exists() and not (requested_dir / "metadata.json").exists():
                latest = RunnerUtils.find_most_recent_run(requested_dir)
                if latest:
                    actual_run_dir = latest
                    resume = True
                else:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    actual_run_dir = requested_dir / f"run_{ts}"
                    resume = False
            else:
                actual_run_dir = requested_dir
                resume = actual_run_dir.exists() and (actual_run_dir / "metadata.json").exists()
        else:
            ts = time.strftime('%Y%m%d_%H%M%S')
            base_dir = slices_path.parent
            if base_dir.name == "slices":
                base_dir = base_dir.parent
            actual_run_dir = base_dir / "run" / f"run_{ts}"
            resume = False

        # 2. Load or generate metadata
        if resume:
            metadata_path = actual_run_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                run_metadata = RunMetadata.from_dict(json.load(f))
            logger.info(f"Resuming from existing run directory: {actual_run_dir}")
        else:
            actual_run_dir.mkdir(parents=True, exist_ok=True)
            save_path = actual_run_dir / "metadata.json"
            run_metadata = RunMetadata.from_dict(
                RunnerAnalyzer.generate_run_metadata(slices_path, save_path, format)
            )
            logger.info(f"Started new run in directory: {actual_run_dir}")
            
        return actual_run_dir, resume, run_metadata
