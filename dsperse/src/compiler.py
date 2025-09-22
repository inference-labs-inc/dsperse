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
from dsperse.src.utils.utils import Utils
from dsperse.src.runner import Runner

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
        
    def compile(self, model_path: str, input_file: Optional[str] = None, layers: Optional[str] = None) -> Dict[str, Any]:
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

        if os.path.isdir(model_path) and (os.path.exists(os.path.join(model_path, "metadata.json")) or os.path.exists(os.path.join(model_path, "slices", "metadata.json"))):
            return self._compile_slices(model_path, input_file_path=input_file, layer_indices=layer_indices)
        elif os.path.isfile(model_path) and model_path.lower().endswith('.onnx'):
            if layer_indices:
                logger.warning("Layer selection is only supported for sliced models, not single ONNX files.")
            return self._compile_model(model_path, input_file_path=input_file)
        else:
            raise ValueError(f"Invalid model path: {model_path}. Must be either a directory containing metadata.json or an .onnx file")

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
        if not os.path.isdir(dir_path):
            raise ValueError(f"path must be a directory: {dir_path}")
        # Find metadata.json
        metadata_path = os.path.join(dir_path, "metadata.json")
        if not os.path.exists(metadata_path):
            alt = os.path.join(dir_path, "slices", "metadata.json")
            if os.path.exists(alt):
                metadata_path = alt
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {dir_path} or its slices subdirectory")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        segments = metadata.get('segments', [])
        segment_output_path = None
        compiled_count = 0
        skipped_count = 0

        # Phase 1: Run ONNX inference chain if we have input file
        current_input = input_file_path
        if current_input and os.path.exists(current_input):
            logger.info("Running ONNX inference chain to generate calibration files")
            for idx, segment in enumerate(segments):
                segment_path = segment.get('path')
                if not segment_path or not os.path.exists(segment_path):
                    logger.warning(f"Segment file not found for index {idx}: {segment_path}")
                    continue

                segment_output_path = os.path.join(os.path.dirname(segment_path), "ezkl_circuitization")
                os.makedirs(segment_output_path, exist_ok=True)

                # Run ONNX inference to generate calibration data
                output_tensor_path = os.path.join(segment_output_path, f"segment_{idx}_calibration.json")
                logger.info(f"Running ONNX inference for segment {idx} with input file {current_input}")
                success, tensor, exec_info = Runner._run_onnx_segment(
                    slice_info={"path": segment_path},
                    input_tensor_path=Path(current_input),
                    output_tensor_path=Path(output_tensor_path)
                )

                if not success:
                    logger.error(f"ONNX inference failed for segment {idx}: {exec_info.get('error', 'Unknown error')}")
                    return

                current_input = output_tensor_path
                logger.info(f"Generated calibration file: {output_tensor_path}")
        else:
            logger.warning("No input file provided, skipping ONNX inference chain")

        # Phase 2: Compile selected layers
        for idx, segment in enumerate(segments):
            if layer_indices is not None and idx not in layer_indices:
                logger.info(f"Skipping compilation for segment {idx} as it's not in the specified layers")
                skipped_count += 1
                continue

            segment_path = segment.get('path')
            if not segment_path or not os.path.exists(segment_path):
                logger.warning(f"Segment file not found for index {idx}: {segment_path}")
                continue
            # Prepare concise progress information similar to slicer output
            deps = segment.get('dependencies', {}) if isinstance(segment, dict) else {}
            input_names = deps.get('filtered_inputs') or deps.get('input') or []
            output_names = deps.get('output') or []
            try:
                logger.info(f"Compiling segment {idx}: {input_names} -> {output_names}")
            except Exception:
                logger.info(f"Compiling segment {idx}")
            segment_output_path = os.path.join(os.path.dirname(segment_path), "ezkl_circuitization")
            os.makedirs(segment_output_path, exist_ok=True)

            calibration_input = input_file_path if idx == 0 else os.path.join(
                os.path.dirname(segments[idx-1].get('path')),
                "ezkl_circuitization",
                f"segment_{idx-1}_calibration.json"
            )

            if calibration_input and os.path.exists(calibration_input):
                logger.info(f"Compiling segment {idx} with calibration input file {calibration_input}")
            compilation_data = self.compiler_impl.compilation_pipeline(
                segment_path,
                segment_output_path,
                input_file_path=calibration_input,
                segment_details=segment
            )
            segment['ezkl_circuitization'] = compilation_data
            compiled_count += 1
            logger.info(f"Completed segment {idx}")
            Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))


        if segment_output_path:
            output_dir = os.path.dirname(segment_output_path)
        else:
            output_dir = os.path.dirname(metadata_path)
        logger.info(f"Compilation of slices completed. Compiled {compiled_count} segments, skipped {skipped_count} segments.")
        logger.info(f"Output saved to {os.path.dirname(output_dir)}")
        return output_dir


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
    # input_file = os.path.join(model_dir, "input.json")
    input_file = None
    # Compile via orchestrator
    model_path = os.path.abspath(model_dir)
    compiler = Compiler.create(model_path=model_path)
    result_dir = compiler.compile(model_path=model_path, input_file=input_file)
    print(f"Compilation finished.")
