"""
Slicer orchestrator module.

This module provides a unified interface for slicing models of different types.
It orchestrates the slicing process by delegating to the appropriate slicer implementation
based on the model type.
"""

import os
import logging
from typing import Optional

from dsperse.src.utils.slicer_utils.onnx_slicer import OnnxSlicer
# Import ModelSlicer for future use
# from src.slicers.model_slicer import ModelSlicer

logger = logging.getLogger(__name__)

class Slicer:
    """
    Orchestrator class for slicing models of different types.
    
    This class provides a unified interface for slicing models by delegating
    to the appropriate slicer implementation based on the model type.
    """
    
    @staticmethod
    def create(model_path: str, save_path: Optional[str] = None) -> 'Slicer':
        """
        Factory method to create a Slicer instance based on the model type.
        
        Args:
            model_path: Path to the model file or directory
            save_path: Optional path to save the model analysis
            
        Returns:
            A Slicer instance
            
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
            
        # Create appropriate slicer
        if is_onnx:
            logger.info(f"Creating ONNX slicer for model: {model_file}")
            return Slicer(OnnxSlicer(model_file, save_path))
        else:
            # For now, we only support ONNX models as per requirements
            # In the future, this can be extended to support other model types
            raise ValueError(f"Unsupported model type at path: {model_path}")
    
    def __init__(self, slicer_impl):
        """
        Initialize the Slicer with a specific implementation.
        
        Args:
            slicer_impl: The slicer implementation to use
        """
        self.slicer_impl = slicer_impl
        
    def slice_model(self, output_path: Optional[str] = None, **kwargs):
        """
        Slice the model using the appropriate slicer implementation.
        
        Args:
            output_path: Directory to save the sliced model
            **kwargs: Additional arguments to pass to the slicer implementation
            
        Returns:
            The result of the slicing operation
        """
        logger.info(f"Slicing model to output path: {output_path}")
        return self.slicer_impl.slice_model(output_path=output_path)


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1 # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/age",
        5: "../models/version"
    }

    # Resolve paths
    abs_path = os.path.abspath(base_paths[model_choice])
    model_file = os.path.join(abs_path, "model.onnx")
    output_dir = os.path.join(abs_path, "slices")

    try:
        # Initialize slicer via orchestrator (auto-selects ONNX slicer)
        slicer = Slicer.create(model_path=model_file, save_path=abs_path)

        # Run slicing
        print(f"Slicing model at {model_file} to {output_dir}...")
        slices = slicer.slice_model(output_path=output_dir)

        # Display results
        print("\nSlicing completed!")
        if isinstance(slices, list):
            print(f"Created {len(slices)} segments.")
            # Optionally display first few slice paths
            preview = slices[:]
            if preview:
                print("Sample slice files:")
                for p in preview:
                    print(f"  {p}")
        else:
            print("Slicing returned no slice list. Check logs for details.")

    except Exception as e:
        print(f"Error during slicing: {e}")

