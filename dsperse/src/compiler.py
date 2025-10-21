"""
Compiler orchestrator module.

This module provides a unified interface for compiling models of different types.
It orchestrates the compilation process by delegating to the appropriate compiler implementation
based on the model type.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from dsperse.src.backends.ezkl import EZKL
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils
from dsperse.src.run.runner import Runner

logger = logging.getLogger(__name__)

class Compiler:
    """
    Orchestrator class for compiling models of different types.
    
    This class provides a unified interface for compiling models by delegating
    to the appropriate compiler implementation based on the model type.
    """
    
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
                model_dir = "."
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
    
    def __init__(self, compiler_impl):
        """
        Initialize the Compiler with a specific implementation.
        
        Args:
            compiler_impl: The compiler implementation to use
        """
        self.compiler_impl = compiler_impl


    @staticmethod
    def _parse_layers(layers_str: Optional[str]):
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
                detected_type = Converter._detect_type(path_obj)
                if detected_type in ['dirs', 'dslice', 'dsperse']:
                    return True, str(path_obj)
            except ValueError:
                pass

        return False, None


    @staticmethod
    def _update_slice_metadata(slice_metadata_path: str, compilation_data: Dict[str, Any],
                               compilation_successful: bool, calibration_path: Optional[str] = None):
        """
        Update the per-slice metadata.json file with compilation results.

        Args:
            slice_metadata_path: Path to the slice's metadata.json file
            compilation_data: Dictionary containing compilation results from EZKL
            compilation_successful: Boolean indicating if compilation was successful
            calibration_path: Optional path to the calibration file used
        """
        # Load existing slice metadata or create new
        if os.path.exists(slice_metadata_path):
            with open(slice_metadata_path, 'r') as f:
                slice_metadata = json.load(f)
        else:
            slice_metadata = {}

        # Get EZKL version
        ezkl_version = EZKL.get_version()

        # Create compilation info nested under 'ezkl'
        ezkl_compilation_info = {
            "compiled": compilation_successful,
            "compilation_timestamp": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
            "ezkl_version": ezkl_version,
            "files": {
                "settings": compilation_data.get('settings'),
                "compiled_circuit": compilation_data.get('compiled'),
                "vk_key": compilation_data.get('vk_key'),
                "pk_key": compilation_data.get('pk_key'),
                "calibration": calibration_path
            }
        }

        # Add any errors if present
        errors = {k: v for k, v in compilation_data.items() if k.endswith('_error')}
        if errors:
            ezkl_compilation_info["errors"] = errors

        # Ensure compilation section exists
        if 'compilation' not in slice_metadata:
            slice_metadata['compilation'] = {}

        # Update slice metadata with ezkl nested under compilation
        slice_metadata['compilation']['ezkl'] = ezkl_compilation_info

        # Save updated slice metadata
        with open(slice_metadata_path, 'w') as f:
            json.dump(slice_metadata, f, indent=2)

        logger.debug(f"Updated slice metadata at {slice_metadata_path}")


    @staticmethod
    def _run_onnx_inference_chain(segments: list, input_file_path: Optional[str] = None):
        """
        Phase 1: Run ONNX inference chain to generate calibration files.

        Args:
            segments: List of segment metadata
            input_file_path: Path to the initial input file
        """
        current_input = input_file_path
        if current_input and os.path.exists(current_input):
            logger.info("Running ONNX inference chain to generate calibration files")
            for idx, segment in enumerate(segments):
                segment_path = segment.get('path')
                if not segment_path or not os.path.exists(segment_path):
                    logger.warning(f"Segment file not found for index {idx}: {segment_path}")
                    continue

                segment_output_path = os.path.join(os.path.dirname(segment_path), "ezkl")
                os.makedirs(segment_output_path, exist_ok=True)

                # Run ONNX inference to generate calibration data
                output_tensor_path = os.path.join(segment_output_path, f"slice_{idx}_calibration.json")
                logger.info(f"Running ONNX inference for slice {idx} with input file {current_input}")
                success, tensor, exec_info = Runner.run_onnx_segment(
                    slice_info={"path": segment_path},
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

    def _compile_selected_layers(self, segments: list, metadata: Dict[str, Any], metadata_path: str,
                                 input_file_path: Optional[str] = None, layer_indices=None):
        """
        Phase 2: Compile the selected layers/segments.

        Args:
            segments: List of segment metadata
            metadata: Full metadata dictionary
            metadata_path: Path to metadata.json file
            input_file_path: Path to the initial input file
            layer_indices: List of layer indices to compile (None means compile all)

        Returns:
            Tuple of (compiled_count, skipped_count, segment_output_path)
        """
        compiled_count = 0
        skipped_count = 0
        segment_output_path = None

        for idx, segment in enumerate(segments):
            if layer_indices is not None and idx not in layer_indices:
                logger.info(f"Skipping compilation for segment {idx} as it's not in the specified layers")
                skipped_count += 1
                continue

            segment_path = segment.get('path')
            if not segment_path or not os.path.exists(segment_path):
                logger.warning(f"Slice file not found for index {idx}: {segment_path}")
                continue

            # Prepare concise progress information similar to slicer output
            deps = segment.get('dependencies', {}) if isinstance(segment, dict) else {}
            input_names = deps.get('filtered_inputs') or deps.get('input') or []
            output_names = deps.get('output') or []
            try:
                logger.info(f"Compiling segment {idx}: {input_names} -> {output_names}")
            except Exception:
                logger.info(f"Compiling segment {idx}")

            segment_output_path = os.path.join(os.path.dirname(segment_path), "ezkl")
            os.makedirs(segment_output_path, exist_ok=True)

            # See if calibration file exists for this slice
            calibration_input = input_file_path if idx == 0 else os.path.join(
                os.path.dirname(segments[idx - 1].get('path')),
                "ezkl",
                f"segment_{idx}_calibration.json"
            )

            # Run compilation
            compilation_data = self.compiler_impl.compilation_pipeline(
                segment_path,
                segment_output_path,
                input_file_path=calibration_input,
                segment_details=segment
            )

            # Update model-level metadata (changed from 'ezkl_circuitization' to 'ezkl')
            segment['ezkl'] = compilation_data

            # Determine if compilation was successful
            compilation_successful = all([
                compilation_data.get('compiled') and os.path.exists(compilation_data['compiled']),
                compilation_data.get('vk_key') and os.path.exists(compilation_data['vk_key']),
                compilation_data.get('pk_key') and os.path.exists(compilation_data['pk_key']),
                not any(k.endswith('_error') for k in compilation_data.keys())
            ])

            # Update per-slice metadata (at slice level, not payload level)
            slice_dir = os.path.dirname(os.path.dirname(segment_path))  # Go up two levels from payload/slice_X.onnx
            slice_metadata_path = os.path.join(slice_dir, "metadata.json")
            self._update_slice_metadata(
                slice_metadata_path,
                compilation_data,
                compilation_successful,
                calibration_input if calibration_input and os.path.exists(calibration_input) else None
            )

            compiled_count += 1
            logger.info(f"Completed segment {idx}")

            # Save model-level metadata
            Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))

        return compiled_count, skipped_count, segment_output_path


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

        segments = metadata.get('segments', [])

        # Phase 1: Run ONNX inference chain for setting calibration (if input file exists)
        if input_file_path:
            self._run_onnx_inference_chain(segments, input_file_path)

        # Phase 2: Compile selected layers
        compiled_count, skipped_count, segment_output_path = self._compile_selected_layers(
            segments, metadata, metadata_path, input_file_path, layer_indices
        )

        # Convert back to original format if needed
        if original_format:
            logger.info(f"Converting back to {original_format} format")
            dir_path = Converter.convert(dir_path, output_type=original_format, cleanup=True)

        if segment_output_path:
            output_dir = os.path.dirname(segment_output_path)
        else:
            output_dir = os.path.dirname(metadata_path)
        logger.info(f"Compilation of slices completed. Compiled {compiled_count} segments, skipped {skipped_count} segments.")
        logger.info(f"Output saved to {os.path.dirname(output_dir)}")
        return output_dir


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

        layer_indices = self._parse_layers(layers) if layers else None
        if layer_indices:
            logger.info(f"Will compile only layers with indices: {layer_indices}")
        elif layers:
            logger.info("No valid layer indices parsed. Will compile all layers.")

        is_sliced, slice_path = self._is_sliced_model(model_path)
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
    model_choice = 1  # Change this to test different models

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
