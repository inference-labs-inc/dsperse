"""
Circuitizer orchestrator module.

This module provides a unified interface for circuitizing models of different types.
It orchestrates the circuitization process by delegating to the appropriate circuitizer implementation
based on the model type.
"""

import os
import json
import logging
from typing import Optional, Dict, Any

from src.backends.ezkl import EZKL
from src.utils.utils import Utils

logger = logging.getLogger(__name__)

class Circuitizer:
    """
    Orchestrator class for circuitizing models of different types.
    
    This class provides a unified interface for circuitizing models by delegating
    to the appropriate circuitizer implementation based on the model type.
    """
    
    @staticmethod
    def create(model_path: str) -> 'Circuitizer':
        """
        Factory method to create a Circuitizer instance based on the model type.
        
        Args:
            model_path: Path to the model file or directory
            
        Returns:
            A Circuitizer instance
            
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
            
        # Create appropriate circuitizer
        if is_onnx:
            logger.info(f"Creating ONNX circuitizer for model: {model_path}")
            return Circuitizer(EZKL())
        else:
            # For now, we only support ONNX models as per requirements
            # In the future, this can be extended to support other model types
            raise ValueError(f"Unsupported model type at path: {model_path}")
    
    def __init__(self, circuitizer_impl):
        """
        Initialize the Circuitizer with a specific implementation.
        
        Args:
            circuitizer_impl: The circuitizer implementation to use
        """
        self.circuitizer_impl = circuitizer_impl
        
    def circuitize(self, model_path: str, input_file: Optional[str] = None, layers: Optional[str] = None) -> Dict[str, Any]:
        """
        Circuitize the model, deciding between whole-model or sliced-model circuitization.
        
        Args:
            model_path: Path to the ONNX model file or a directory containing slices/metadata
            input_file: Optional path to input file for calibration
            layers: Optional string specifying which layers to circuitize (e.g., "3, 20-22").
                    Only applicable to sliced models.
            
        Returns:
            The path to the directory where circuitization results are saved, or metadata updates path for slices.
        """
        logger.info(f"Circuitizing: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path does not exist: {model_path}")

        layer_indices = self._parse_layers(layers) if layers else None
        if layer_indices:
            logger.info(f"Will circuitize only layers with indices: {layer_indices}")
        elif layers:
            logger.info("No valid layer indices parsed. Will circuitize all layers.")

        if os.path.isdir(model_path) and (os.path.exists(os.path.join(model_path, "metadata.json")) or os.path.exists(os.path.join(model_path, "slices", "metadata.json"))):
            return self._circuitize_slices(model_path, input_file_path=input_file, layer_indices=layer_indices)
        elif os.path.isfile(model_path) and model_path.lower().endswith('.onnx'):
            if layer_indices:
                logger.warning("Layer selection is only supported for sliced models, not single ONNX files.")
            return self._circuitize_model(model_path, input_file_path=input_file)
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

    def _circuitize_model(self, model_file_path: str, input_file_path: Optional[str] = None) -> str:
        if not os.path.isfile(model_file_path):
            raise ValueError(f"model_path must be a file: {model_file_path}")
        output_path_root = os.path.splitext(model_file_path)[0]
        circuit_folder = os.path.join(os.path.dirname(output_path_root), "model")
        os.makedirs(circuit_folder, exist_ok=True)
        # Call backend pipeline
        self.circuitizer_impl.circuitization_pipeline(model_file_path, circuit_folder, input_file_path=input_file_path)
        logger.info(f"Circuitization completed. Output saved to {circuit_folder}")
        return circuit_folder

    def _circuitize_slices(self, dir_path: str, input_file_path: Optional[str] = None, layer_indices=None) -> str:
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
        circuitized_count = 0
        skipped_count = 0

        for idx, segment in enumerate(segments):
            if layer_indices is not None and idx not in layer_indices:
                logger.info(f"Skipping segment {idx} as it's not in the specified layers")
                skipped_count += 1
                continue
            segment_path = segment.get('path')
            if not segment_path or not os.path.exists(segment_path):
                logger.warning(f"Segment file not found for index {idx}: {segment_path}")
                continue
            segment_output_path = os.path.join(os.path.dirname(segment_path), "ezkl_circuitization")
            # Run pipeline and get data
            circuitization_data = self.circuitizer_impl.circuitization_pipeline(
                segment_path,
                segment_output_path,
                input_file_path=input_file_path,
                segment_details=segment
            )
            segment['ezkl_circuitization'] = circuitization_data
            circuitized_count += 1
            Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))

        if segment_output_path:
            output_dir = os.path.dirname(segment_output_path)
        else:
            output_dir = os.path.dirname(metadata_path)
        logger.info(f"Circuitization of slices completed. Circuitized {circuitized_count} segments, skipped {skipped_count} segments.")
        logger.info(f"Output saved to {os.path.dirname(output_dir)}")
        return output_dir

if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet",
        4: "models/yolov3"
    }
    abs_path = os.path.abspath(base_paths[model_choice])
    model_dir = abs_path
    slices_dir = os.path.join(abs_path, "slices")

    # Circuitize via orchestrator
    model_path = os.path.abspath(model_dir)
    circuitizer = Circuitizer.create(model_path=model_path)
    result_dir = circuitizer.circuitize(model_path=model_path)
    print(f"Circuitization finished. Output at: {result_dir}")