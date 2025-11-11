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

    def __init__(self, compiler_impl):
        """
        Initialize the Compiler with a specific implementation.

        Args:
            compiler_impl: The compiler implementation to use
        """
        self.compiler_impl = compiler_impl

    @staticmethod
    def create(model_path: str) -> 'Compiler':
        """
        Factory method to create a Compiler instance based on the model type.
        
        Args:
            model_path: Path to the model file or directory
            
        Returns:
            A Compiler instance
            
        Raises:
            ValueError: If the model type is not supported
        """
        # Check if the path is a file or directory
        if os.path.isfile(model_path):
            model_file = model_path
            model_dir = os.path.dirname(model_path)
            if not model_dir:  # If the directory is empty (e.g., just "model.onnx")
                model_dir = ".."
        else:
            model_dir = model_path
            model_file = None
            
        # Determine model type
        is_onnx = False
        
        # Check if it's an ONNX model
        if model_file and model_file.lower().endswith('.onnx'):
            is_onnx = True
        elif os.path.exists(os.path.join(model_dir, "model.onnx")):
            is_onnx = True
            model_file = os.path.join(model_dir, "model.onnx")
        # Check if it's a directory with metadata.json (sliced model)
        elif os.path.isdir(model_path) and (os.path.exists(os.path.join(model_path, "metadata.json")) or 
                                           os.path.exists(os.path.join(model_path, "slices", "metadata.json"))):
            is_onnx = True
            
        # Create appropriate compiler
        if is_onnx:
            logger.info(f"Creating ONNX compiler for model: {model_path}")
            return Compiler(EZKL())
        else:
            # For now, we only support ONNX models as per requirements
            # In the future, this can be extended to support other model types
            raise ValueError(f"Unsupported model type at path: {model_path}")

    def _compile_selected_layers(self, slices_data: list, metadata: Dict[str, Any], metadata_path: str,
                                 input_file_path: Optional[str] = None, layer_indices=None):
        """
        Phase 2: Compile the selected layers/slices.

        Args:
            slices_data: List of slice metadata
            metadata: Full metadata dictionary
            metadata_path: Path to metadata.json file
            input_file_path: Path to the initial input file
            layer_indices: List of layer indices to compile (None means compile all)

        Returns:
            Tuple of (compiled_count, skipped_count, slice_output_path)
        """
        compiled_count = 0
        skipped_count = 0
        slice_output_path = None

        for idx, slice_data in enumerate(slices_data):
            if layer_indices is not None and idx not in layer_indices:
                logger.info(f"Skipping compilation for slice {idx} as it's not in the specified layers")
                skipped_count += 1
                continue

            slice_path = slice_data.get('path')
            if not slice_path or not os.path.exists(slice_path):
                logger.warning(f"Slice file not found for index {idx}: {slice_path}")
                continue

            # Prepare concise progress information similar to slicer output
            deps = slice_data.get('dependencies', {}) if isinstance(slice_data, dict) else {}
            input_names = deps.get('filtered_inputs') or deps.get('input') or []
            output_names = deps.get('output') or []
            try:
                logger.info(f"Compiling slice {idx}: {input_names} -> {output_names}")
            except Exception:
                logger.info(f"Compiling slice {idx}")

            slice_output_path = os.path.join(os.path.dirname(slice_path), "ezkl")
            os.makedirs(slice_output_path, exist_ok=True)

            # See if calibration file exists for this slice
            calibration_input = input_file_path if idx == 0 else os.path.join(
                os.path.dirname(slices_data[idx].get('path')),
                "ezkl",
                f"calibration.json"
            )

            # Run compilation
            compilation_data = self.compiler_impl.compilation_pipeline(
                slice_path,
                slice_output_path,
                input_file_path=calibration_input,
                slice_details=slice_data
            )

            # Determine if compilation was successful (based on absolute paths)
            compilation_successful = CompilerUtils.is_compilation_successful(compilation_data)

            # Compute payload-relative paths and calibration
            payload_rel, calibration_rel = CompilerUtils.compute_payload_and_calibration_rel(compilation_data, calibration_input)

            # Prepare slice-level compilation data (payload-relative paths)
            slice_level_comp_data = CompilerUtils.apply_payload_rel_to_comp_data(compilation_data, payload_rel)

            # Update per-slice metadata (at slice level, not payload level)
            slice_dir, slice_metadata_path = CompilerUtils.get_slice_dirs(slice_path)
            CompilerUtils.update_slice_metadata(
                slice_metadata_path,
                slice_level_comp_data,
                compilation_successful,
                calibration_rel
            )

            # Build model-level metadata with 'slice_#/' prefix
            slice_dirname = os.path.basename(slice_dir)  # e.g., 'slice_4'
            model_level_ezkl = CompilerUtils.build_model_level_ezkl(payload_rel, calibration_rel, slice_dirname, compilation_data)

            # Update model-level metadata for this slice entry (flat ezkl with paths)
            # slice_data['ezkl'] = model_level_ezkl

            # Also mirror per-slice compilation schema at model level for consistency
            try:
                files = {
                    "settings": model_level_ezkl.get('settings'),
                    "compiled_circuit": model_level_ezkl.get('compiled') or model_level_ezkl.get('compiled_circuit'),
                    "vk_key": model_level_ezkl.get('vk_key'),
                    "pk_key": model_level_ezkl.get('pk_key'),
                    "calibration": model_level_ezkl.get('calibration')
                }
                ezkl_version = EZKL.get_version()
                comp_block = {
                    "compiled": bool(compilation_successful),
                    "compilation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ezkl_version": ezkl_version,
                    "files": files
                }
                # Attach under 'compilation.ezkl'
                if isinstance(slice_data, dict):
                    if 'compilation' not in slice_data or not isinstance(slice_data.get('compilation'), dict):
                        slice_data['compilation'] = {}
                    slice_data['compilation']['ezkl'] = comp_block
            except Exception as e:
                logger.warning(f"Failed to add model-level compilation block for slice {idx}: {e}")

            compiled_count += 1
            logger.info(f"Completed slice {idx}")

            # Save model-level metadata
            Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))

        return compiled_count, skipped_count, slice_output_path


    def _compile_model(self, model_file_path: str, input_file_path: Optional[str] = None) -> str:
        if not os.path.isfile(model_file_path):
            raise ValueError(f"model_path must be a file: {model_file_path}")
        output_path_root = os.path.splitext(model_file_path)[0]
        circuit_folder = os.path.join(os.path.dirname(output_path_root), "model")
        os.makedirs(circuit_folder, exist_ok=True)
        # Call backend pipeline
        self.compiler_impl.compilation_pipeline(model_file_path, circuit_folder, input_file_path=input_file_path)
        logger.info(f"Compilation completed. Output saved to {circuit_folder}")
        return circuit_folder


    def _compile_slices(self, dir_path: str, input_file_path: Optional[str] = None, layer_indices=None) -> str:
        # Convert to dirs if needed
        path_obj = Path(dir_path)
        original_format = None
        if path_obj.is_file() or Converter.detect_type(path_obj) in ['dslice', 'dsperse']:
            original_format = 'dslice' if Converter.detect_type(path_obj) == 'dslice' else 'dsperse'
            logger.info(f"Converting {dir_path} to directory format")
            dir_path = Converter.convert(dir_path, output_type="dirs", cleanup=False)

        # Load metadata
        metadata_path = Utils.find_metadata_path(dir_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        slices_data = metadata.get('slices', [])

        # Phase 1: Run ONNX inference chain for setting calibration (if input file exists)
        if input_file_path:
            CompilerUtils.run_onnx_inference_chain(slices_data, input_file_path)

        # Phase 2: Compile selected layers
        compiled_count, skipped_count, slice_output_path = self._compile_selected_layers(
            slices_data, metadata, metadata_path, input_file_path, layer_indices
        )

        # Convert back to original format if needed
        final_path = None
        if original_format:
            logger.info(f"Converting back to {original_format} format")
            final_path = Converter.convert(dir_path, output_type=original_format, cleanup=True)

        # Determine default output if not packaged
        if not final_path:
            if slice_output_path:
                output_dir = os.path.dirname(slice_output_path)
            else:
                output_dir = os.path.dirname(metadata_path)
            final_path = output_dir

        logger.info(f"Compilation of slices completed. Compiled {compiled_count} slices, skipped {skipped_count} slices.")
        try:
            p = Path(final_path)
            if p.is_file():
                logger.info(f"Output packaged at {final_path}")
            else:
                logger.info(f"Output saved under {final_path}")
        except Exception:
            logger.info(f"Output location: {final_path}")
        return final_path


    def compile(self, model_path: str, input_file: Optional[str] = None, layers: Optional[str] = None):
        """
        Compile the model, deciding between whole-model or sliced-model compilation.

        Args:
            model_path: Path to the ONNX model file or a directory containing slices/metadata
            input_file: Optional path to input file for calibration
            layers: Optional string specifying which layers to compile (e.g., "3, 20-22").
                    Only applicable to sliced models.

        Returns:
            The path to the directory where compilation results are saved, or metadata updates path for slices.
        """
        logger.info(f"Compiling: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path does not exist: {model_path}")

        layer_indices = CompilerUtils.parse_layers(layers) if layers else None
        if layer_indices:
            logger.info(f"Will compile only layers with indices: {layer_indices}")
        elif layers:
            logger.info("No valid layer indices parsed. Will compile all layers.")

        is_sliced, slice_path = CompilerUtils.is_sliced_model(model_path)
        if is_sliced:
            return self._compile_slices(slice_path, input_file_path=input_file, layer_indices=layer_indices)
        elif os.path.isfile(model_path) and model_path.lower().endswith('.onnx'):
            if layer_indices:
                logger.warning("Layer selection is only supported for sliced models, not single ONNX files.")
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
    input_file = os.path.join(model_dir, "input.json")
    # input_file = None
    # Compile via orchestrator
    model_path = os.path.abspath(model_dir)
    compiler = Compiler.create(model_path=model_path)
    result_dir = compiler.compile(model_path=model_path, input_file=input_file, layers="3, 4")
    print(f"Compilation finished.")
