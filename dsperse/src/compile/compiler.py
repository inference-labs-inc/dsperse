"""
Compiler orchestrator module.

This module provides a unified interface for compiling models of different types.
It orchestrates the compilation process by delegating to the appropriate compiler implementation
based on the model type.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.compile.utils.compiler_utils import CompilerUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class Compiler:
    """
    Orchestrator class for compiling models of different types.
    
    This class provides a unified interface for compiling models by delegating
    to the appropriate compiler implementation based on the model type.
    """

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize the Compiler with a specific backend configuration.

        Args:
            backend: Backend specification. Can be:
                - None: Use jstprove with fallback to ezkl then onnx (tries jstprove first)
                - "jstprove" or "ezkl": Use specific backend for all layers
                - "0,2:jstprove;3-4:ezkl": Per-layer backend specification
        """
        self.backend_spec = backend
        self.layer_backends = {}  # Map layer index -> backend name
        self.use_fallback = False

        # Parse backend specification
        if backend is None:
            # Default: use fallback logic (try both jstprove and ezkl, then onnx)
            self.default_backend = None  # Will try both
            self.use_fallback = True
        elif ':' in str(backend):
            # Per-layer specification like "0,2:jstprove;3-4:ezkl"
            # Unspecified layers use default fallback, specified layers try their backend first
            self.default_backend = None
            self.use_fallback = True  # Enable fallback for both specified and unspecified layers
            self._parse_layer_backends(backend)
        else:
            # Simple backend name - no fallback, use only this backend
            self.default_backend = backend.lower()
            self.use_fallback = False

        # Initialize backends (lazy loading to avoid errors if not used)
        self._jstprove = None
        self._ezkl = None

    def _parse_layer_backends(self, spec: str):
        """Parse layer-specific backend specification like '0,2:jstprove;3-4:ezkl'"""
        parts = spec.split(';')
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
            layers_str, backend_name = part.split(':', 1)
            backend_name = backend_name.strip().lower()

            # Reuse existing layer parsing utility
            layer_indices = CompilerUtils.parse_layers(layers_str)
            if layer_indices:
                for idx in layer_indices:
                    self.layer_backends[idx] = backend_name

    def _get_jstprove(self):
        """Lazy initialization of JSTprove backend"""
        if self._jstprove is None:
            try:
                self._jstprove = JSTprove()
            except Exception as e:
                logger.warning(f"Failed to initialize JSTprove: {e}")
                return None
        return self._jstprove

    def _get_ezkl(self):
        """Lazy initialization of EZKL backend"""
        if self._ezkl is None:
            try:
                self._ezkl = EZKL()
            except Exception as e:
                logger.warning(f"Failed to initialize EZKL: {e}")
                return None
        return self._ezkl

    def _get_backend_for_layer(self, layer_idx: int):
        """Get the backend instance for a specific layer"""
        # Check if layer has specific backend assigned
        if layer_idx in self.layer_backends:
            backend_name = self.layer_backends[layer_idx]
            if backend_name == "jstprove":
                return self._get_jstprove(), "jstprove"
            else:
                return self._get_ezkl(), "ezkl"
        elif self.default_backend is None:
            # Default: try both backends (will be handled in fallback logic)
            return None, None
        else:
            # Simple backend specified
            if self.default_backend == "jstprove":
                return self._get_jstprove(), "jstprove"
            else:
                return self._get_ezkl(), "ezkl"

    # Keep backward compatibility properties
    @property
    def backend(self):
        # Return ezkl for backward compatibility
        return self._get_ezkl()

    @property
    def backend_name(self):
        return self.default_backend or "ezkl"

    @property
    def ezkl(self):
        return self._get_ezkl()


    def _compile_slice(self, idx: int, slice_data: dict, base_path: str):
        """
        Function for compiling a single slice with fallback support.
        Tries jstprove -> ezkl -> onnx (skip) if fallback is enabled.
        
        Args:
            idx: Slice index
            slice_data: Dictionary containing slice information
            base_path: Base path for resolving relative paths
        """
        slice_path = slice_data.get('path')
        if slice_path and os.path.exists(slice_path):
            pass  # Use the full path if exists
        elif slice_data.get('relative_path'):
            slice_path = os.path.join(base_path, slice_data.get('relative_path'))
            if not os.path.exists(slice_path):
                logger.warning(f"Slice file not found for {slice_path}")
                raise FileNotFoundError(f"Slice file not found for {slice_path}")
        else:
            logger.error(f"No valid path found for slice")
            raise FileNotFoundError(f"No valid path found for slice")

        # Get the backend for this specific layer
        backend, backend_name = self._get_backend_for_layer(idx)

        # Build list of backends to try
        backends_to_try = []
        if backend is not None:
            # Specific backend assigned to this layer
            backends_to_try = [(backend, backend_name)]
            if self.use_fallback:
                # Add fallback: try other backend, then onnx
                if backend_name == "jstprove":
                    ezkl = self._get_ezkl()
                    if ezkl:
                        backends_to_try.append((ezkl, "ezkl"))
                elif backend_name == "ezkl":
                    jst = self._get_jstprove()
                    if jst:
                        backends_to_try.append((jst, "jstprove"))
                backends_to_try.append((None, "onnx"))
        elif self.use_fallback:
            # No specific backend for this layer, use default fallback chain
            # (jstprove -> ezkl -> onnx)
            jst = self._get_jstprove()
            ezkl = self._get_ezkl()
            if jst:
                backends_to_try.append((jst, "jstprove"))
            if ezkl:
                backends_to_try.append((ezkl, "ezkl"))
            backends_to_try.append((None, "onnx"))
        else:
            # No backend specified and no fallback - skip compilation (use pure ONNX)
            backends_to_try = [(None, "onnx")]

        success = False
        compilation_data = {}
        used_backend = None

        for try_backend, try_backend_name in backends_to_try:
            if try_backend is None:
                # Skip compilation - will use onnx at runtime
                logger.info(f"Slice {idx}: Skipping ZK compilation, will use ONNX at runtime")
                success = True
                used_backend = "onnx"
                compilation_data = {"skipped": True, "reason": "fallback_to_onnx"}
                break

            backend_dir = try_backend_name
            slice_output_path = os.path.join(os.path.dirname(slice_path), backend_dir)

            calibration_input = os.path.join(
                os.path.dirname(slice_path),
                backend_dir,
                f"calibration.json"
            ) if os.path.exists(os.path.join(os.path.dirname(slice_path), backend_dir, "calibration.json")) else None

            try:
                logger.info(f"Slice {idx}: Trying {try_backend_name}...")
                compilation_data = try_backend.compilation_pipeline(
                    slice_path,
                    slice_output_path,
                    input_file_path=calibration_input
                )
                success = CompilerUtils.is_ezkl_compilation_successful(compilation_data)
                if success:
                    used_backend = try_backend_name
                    logger.info(f"Slice {idx}: {try_backend_name} compilation succeeded")
                    break
                else:
                    logger.warning(f"Slice {idx}: {try_backend_name} compilation failed, trying fallback...")
            except Exception as e:
                logger.warning(f"Slice {idx}: {try_backend_name} error: {e}, trying fallback...")
                if not self.use_fallback:
                    raise

    def _resolve_slice_path(self, slice_data: dict, base_path: str) -> str:
        """Resolve absolute path to a slice file from metadata and base path."""
        slice_path = slice_data.get('path')
        if slice_path and os.path.exists(slice_path):
            return slice_path
        if slice_data.get('relative_path'):
            slice_path = os.path.join(base_path, slice_data.get('relative_path'))
            if os.path.exists(slice_path):
                return slice_path
        logger.error("No valid path found for slice")
        raise FileNotFoundError("No valid path found for slice")

    def _compile_ezkl_slice(self, idx: int, slice_data: dict, base_path: str) -> tuple[bool, Dict[str, Any]]:
        """
        Compile a single slice with the EZKL backend.

        Returns: (success, file_paths)
        """
        backend = self._get_ezkl()
        if backend is None:
            raise RuntimeError("EZKL backend is not available")

        slice_path = self._resolve_slice_path(slice_data, base_path)
        backend_dir = "ezkl"
        slice_output_path = os.path.join(os.path.dirname(slice_path), backend_dir)

        calibration_input = os.path.join(
            os.path.dirname(slice_path), backend_dir, "calibration.json"
        ) if os.path.exists(os.path.join(os.path.dirname(slice_path), backend_dir, "calibration.json")) else None

        logger.info(f"Slice {idx}: Compiling with EZKL...")
        compilation_data = backend.compilation_pipeline(
            slice_path,
            slice_output_path,
            input_file_path=calibration_input
        )
        success = CompilerUtils.is_ezkl_compilation_successful(compilation_data)
        # Normalize file paths for metadata (relative under payload and include calibration)
        file_paths = CompilerUtils.get_relative_paths(compilation_data, calibration_input)
        return success, file_paths

    def _compile_jstprove_slice(self, idx: int, slice_data: dict, base_path: str) -> tuple[bool, Dict[str, Any]]:
        """
        Compile a single slice with the JSTprove backend.

        Returns: (success, file_paths)
        """
        backend = self._get_jstprove()
        if backend is None:
            raise RuntimeError("JSTprove backend is not available")

        slice_path = self._resolve_slice_path(slice_data, base_path)
        backend_dir = "jstprove"
        slice_output_path = os.path.join(os.path.dirname(slice_path), backend_dir)

        calibration_input = os.path.join(
            os.path.dirname(slice_path), backend_dir, "calibration.json"
        ) if os.path.exists(os.path.join(os.path.dirname(slice_path), backend_dir, "calibration.json")) else None

        logger.info(f"Slice {idx}: Compiling with JSTprove...")
        compilation_data = backend.compilation_pipeline(
            slice_path,
            slice_output_path,
            input_file_path=calibration_input
        )
        success = CompilerUtils.is_ezkl_compilation_successful(compilation_data)
        # Normalize file paths for metadata (relative under payload and include calibration)
        file_paths = CompilerUtils.get_relative_paths(compilation_data, calibration_input)
        return success, file_paths

    def _compile_model(self, model_file_path: str, input_file_path: Optional[str] = None) -> str:
        """
        Compile a single ONNX model file (not sliced) with backend fallback support.
        """
        if not os.path.isfile(model_file_path):
            raise ValueError(f"model_path must be a file: {model_file_path}")
        output_path_root = os.path.splitext(model_file_path)[0]
        
        # Build list of backends to try (same logic as _compile_slice)
        backends_to_try = []
        if self.default_backend:
            # Specific backend requested
            if self.default_backend == "jstprove":
                jst = self._get_jstprove()
                if jst:
                    backends_to_try.append((jst, "jstprove"))
            else:
                ezkl = self._get_ezkl()
                if ezkl:
                    backends_to_try.append((ezkl, "ezkl"))
            if self.use_fallback:
                # Add fallback options
                if self.default_backend == "jstprove":
                    ezkl = self._get_ezkl()
                    if ezkl:
                        backends_to_try.append((ezkl, "ezkl"))
                else:
                    jst = self._get_jstprove()
                    if jst:
                        backends_to_try.append((jst, "jstprove"))
        elif self.use_fallback:
            # Default fallback: jstprove -> ezkl
            jst = self._get_jstprove()
            ezkl = self._get_ezkl()
            if jst:
                backends_to_try.append((jst, "jstprove"))
            if ezkl:
                backends_to_try.append((ezkl, "ezkl"))
        else:
            # No backend specified, no fallback - use EZKL as default
            ezkl = self._get_ezkl()
            if ezkl:
                backends_to_try.append((ezkl, "ezkl"))

        if not backends_to_try:
            raise RuntimeError("No backends available for compilation")

        # Try each backend until one succeeds
        for try_backend, try_backend_name in backends_to_try:
            circuit_folder = os.path.join(os.path.dirname(output_path_root), try_backend_name)
            os.makedirs(circuit_folder, exist_ok=True)
            try:
                logger.info(f"Compiling model with {try_backend_name}...")
                compilation_data = try_backend.compilation_pipeline(
                    model_file_path, circuit_folder, input_file_path=input_file_path
                )
                success = CompilerUtils.is_ezkl_compilation_successful(compilation_data)
                if success:
                    logger.info(f"Compilation completed with {try_backend_name}. Output saved to {circuit_folder}")
                    return circuit_folder
                else:
                    logger.warning(f"{try_backend_name} compilation failed, trying fallback...")
            except Exception as e:
                logger.warning(f"{try_backend_name} error: {e}, trying fallback...")
                if not self.use_fallback:
                    raise

        raise RuntimeError("All backends failed to compile the model")


    def _compile_slices(self, dir_path: str, input_file_path: Optional[str] = None, layer_indices=None):
        # Load metadata
        metadata_path = Utils.find_metadata_path(dir_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        base_path = os.path.dirname(metadata_path)
        slices_data = metadata.get('slices', [])

        # Phase 1: Run ONNX inference chain for setting calibration (if input file exists)
        if input_file_path:
            CompilerUtils.run_onnx_inference_chain(slices_data, base_path, input_file_path)

        # Phase 2: Compile layers
        compiled_count = 0
        skipped_count = 0
        backend_stats: Dict[int, list[str]] = {}  # Track which backends succeeded for each slice

        for idx, slice_data in enumerate(slices_data):
            if layer_indices is not None and idx not in layer_indices:
                logger.info(f"Skipping ZK compilation for slice {idx} (not in specified layers) - will use pure ONNX at runtime")
                skipped_count += 1
                continue

            logger.info(f"Compiling slice {idx}...")

            # Decide which backends to compile for this slice
            backends_to_build: list[str]
            if idx in self.layer_backends:
                backends_to_build = [self.layer_backends[idx]]
            elif self.default_backend in {"jstprove", "ezkl"} and not self.use_fallback:
                backends_to_build = [self.default_backend]
            else:
                # Default behavior: try both
                backends_to_build = ["jstprove", "ezkl"]

            successful_backends: list[str] = []

            for be in backends_to_build:
                try:
                    if be == "jstprove":
                        success, file_paths = self._compile_jstprove_slice(idx, slice_data, base_path)
                        version = self._jstprove.get_version() if hasattr(self._jstprove, 'get_version') and self._jstprove else None
                    elif be == "ezkl":
                        success, file_paths = self._compile_ezkl_slice(idx, slice_data, base_path)
                        version = self._ezkl.get_version() if hasattr(self._ezkl, 'get_version') and self._ezkl else None
                    else:
                        logger.warning(f"Unknown backend '{be}' requested for slice {idx}, skipping")
                        continue

                    compiled_count += 1

                    # Prefix model-level file paths with slice dir (keep slice-level unchanged)
                    sdn = (slice_data.get('relative_path') or '').split(os.sep)[0] or os.path.basename(os.path.dirname(os.path.dirname(slice_data.get('path') or '')))
                    pref_files = ({k: (os.path.join(sdn, v) if isinstance(v, str) and not v.startswith(sdn + os.sep) else v) for k, v in (file_paths or {}).items()} if sdn else file_paths)
                    comp_block = {
                        "compiled": bool(success),
                        "compilation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "backend": be,
                        "backend_version": version,
                        "files": pref_files
                    }

                    if isinstance(slice_data, dict):
                        if 'compilation' not in slice_data or not isinstance(slice_data.get('compilation'), dict):
                            slice_data['compilation'] = {}
                        slice_data['compilation'][be] = comp_block

                    # Update slice-level metadata file as well (if present)
                    slice_meta_path: Optional[Path] = None
                    if slice_data.get('slice_metadata') and os.path.exists(slice_data.get('slice_metadata')):
                        slice_meta_path = Path(slice_data.get('slice_metadata'))
                    elif slice_data.get('slice_metadata_relative_path') and os.path.exists(os.path.join(base_path, slice_data.get('slice_metadata_relative_path'))):
                        slice_meta_path = Path(os.path.join(base_path, slice_data.get('slice_metadata_relative_path')))

                    if slice_meta_path is not None:
                        try:
                            CompilerUtils.update_slice_metadata(idx, slice_meta_path, bool(success), file_paths, backend_name=be)
                        except Exception as e:
                            logger.warning(f"Failed to update slice metadata for slice {idx} backend {be}: {e}")

                    if success:
                        successful_backends.append(be)
                        logger.info(f"Completed slice {idx} with {be}")
                    else:
                        logger.error(f"Slice {idx}: {be} compilation unsuccessful")
                except Exception as e:
                    logger.error(f"Slice {idx}: {be} compilation error: {e}. Continuing with others if any.")
                    continue

            backend_stats[idx] = successful_backends

            # If none succeeded, mark ONNX fallback for visibility
            if not successful_backends:
                if isinstance(slice_data, dict):
                    if 'compilation' not in slice_data or not isinstance(slice_data.get('compilation'), dict):
                        slice_data['compilation'] = {}
                    slice_data['compilation']['onnx'] = {
                        "compiled": True,
                        "compilation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "backend": "onnx",
                        "backend_version": None,
                        "files": {"skipped": True, "reason": "fallback_to_onnx"}
                    }

            # Save model-level metadata (or single slice metadata)
            Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))

        # Log summary
        backend_summary: Dict[str, int] = {}
        for _idx, backends in backend_stats.items():
            for be in backends:
                backend_summary[be] = backend_summary.get(be, 0) + 1
        summary_str = ", ".join(f"{k}: {v}" for k, v in backend_summary.items())
        if skipped_count > 0:
            logger.info(f"Compilation completed. ZK compiled: {compiled_count} slices ({summary_str}). Skipped: {skipped_count} slices (will use pure ONNX at runtime)")
        else:
            logger.info(f"Compilation completed. ZK compiled: {compiled_count} slices. Backends used: {summary_str}")


    def compile(self, model_path: str, input_file: Optional[str] = None, layers: Optional[str] = None):
        """
        Compile the model, deciding between whole-model or sliced-model compilation.

        Args:
            model_path: Path to the ONNX model file or a directory containing slices/metadata
            input_file: Optional path to input file for calibration
            layers: Optional string specifying which layers to compile (e.g., "3, 20-22").
                    Only applicable to sliced models.

        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path does not exist: {model_path}")
        logger.info(f"Compiling: {model_path}")

        layer_indices = CompilerUtils.parse_layers(layers) if layers else None
        if layer_indices:
            logger.info(f"Will compile only layers with indices: {layer_indices}")
        else:
            # No layers specified: compile ALL layers with default fallback
            logger.info("No layers specified. Will compile all layers with default fallback (jstprove -> ezkl -> onnx).")

        is_sliced, slice_path, type = CompilerUtils.is_sliced_model(model_path)
        if is_sliced:
            # Convert to dirs if needed
            if type != "dirs":
                slice_path = Converter.convert(model_path, output_type="dirs", cleanup=True)

            self._compile_slices(slice_path, input_file_path=input_file, layer_indices=layer_indices)

            # Convert back to original type if needed
            if type != "dirs":
                slice_path = Converter.convert(slice_path, output_type=type, cleanup=True)

            return slice_path

        elif os.path.isfile(model_path) and model_path.lower().endswith('.onnx'):
            return self._compile_model(model_path, input_file_path=input_file)
        else:
            raise ValueError(f"Invalid model path: {model_path}. Must be either a sliced model or an .onnx file")


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/age",
        5: "../models/version"
    }
    abs_path = os.path.abspath(base_paths[model_choice])
    model_dir = abs_path
    slices_dir = os.path.join(abs_path, "slices")
    # slices_dir = os.path.join(slices_dir, "slice_0.dslice")
    input_file = os.path.join(model_dir, "input.json")

    compiler = Compiler()
    result = compiler.compile(model_path=slices_dir)#, input_file=input_file, layers="3, 4")
    print(f"Compilation finished.")
