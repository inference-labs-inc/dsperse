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

    def __init__(self):
        """
        Initialize the Compiler with a specific implementation.

        Args:
            compiler_impl: The compiler implementation to use
        """
        self.ezkl = EZKL()


    def _compile_slice(self, slice_data, base_path: str):
        """
        Function for compiling a single slice.
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

        slice_output_path = os.path.join(os.path.dirname(slice_path), "ezkl")

        calibration_input = os.path.join(
            os.path.dirname(slice_path),
            "ezkl",
            f"calibration.json"
        ) if os.path.exists(os.path.join(os.path.dirname(slice_path), "ezkl", "calibration.json")) else None

        compilation_data = self.ezkl.compilation_pipeline(
            slice_path,
            slice_output_path,
            input_file_path=calibration_input
        )

        success = CompilerUtils.is_ezkl_compilation_successful(compilation_data)
        file_paths = CompilerUtils.get_relative_paths(compilation_data, calibration_input)

        if slice_data.get('slice_metadata') and os.path.exists(slice_data.get('slice_metadata')):
            path = Path(slice_data.get('slice_metadata'))
            CompilerUtils.update_slice_metadata(path, success, file_paths)
        elif slice_data.get('slice_metadata_relative_path') and os.path.exists(os.path.join(base_path, slice_data.get('slice_metadata_relative_path'))):
            path = Path(os.path.join(base_path, slice_data.get('slice_metadata_relative_path')))
            CompilerUtils.update_slice_metadata(path, success, file_paths)

        return success, file_paths

    def _compile_model(self, model_file_path: str, input_file_path: Optional[str] = None) -> str:
        if not os.path.isfile(model_file_path):
            raise ValueError(f"model_path must be a file: {model_file_path}")
        output_path_root = os.path.splitext(model_file_path)[0]
        circuit_folder = os.path.join(os.path.dirname(output_path_root), "model")
        os.makedirs(circuit_folder, exist_ok=True)
        # Call backend pipeline
        self.ezkl.compilation_pipeline(model_file_path, circuit_folder, input_file_path=input_file_path)
        logger.info(f"Compilation completed. Output saved to {circuit_folder}")
        return circuit_folder


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

        for idx, slice_data in enumerate(slices_data):
            if layer_indices is not None and idx not in layer_indices:
                logger.info(f"Skipping compilation for slice {idx} as it's not in the specified layers")
                skipped_count += 1
                continue

            logger.info(f"Compiling slice {idx}...")

            success, file_paths = self._compile_slice(slice_data, base_path)

            compiled_count += 1
            logger.info(f"Completed slice {idx}")

            comp_block = {
                "compiled": bool(success),
                "compilation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ezkl_version": EZKL.get_version(),
                "files": file_paths
            }
            # Attach under 'compilation.ezkl'
            if isinstance(slice_data, dict):
                if 'compilation' not in slice_data or not isinstance(slice_data.get('compilation'), dict):
                    slice_data['compilation'] = {}
                slice_data['compilation']['ezkl'] = comp_block

            # Save model-level metadata (or single slice metadata)
            Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))

        logger.info(f"Compilation of slices completed. Compiled {compiled_count} slices, skipped {skipped_count} slices.")


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
            logger.info("Will compile all layers.")

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
