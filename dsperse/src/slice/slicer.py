"""
Slicer orchestrator module.

This module provides a unified interface for slicing models of different types.
It orchestrates the slicing process by delegating to the appropriate slicer implementation
based on the model type.
"""

import os
import logging
from typing import Optional

from dsperse.src.slice.onnx_slicer import OnnxSlicer
from dsperse.src.slice.utils.converter import Converter


logger = logging.getLogger(__name__)

class Slicer:
    """
    Orchestrator class for slicing models of different types.
    
    This class provides a unified interface for slicing models by delegating
    to the appropriate slicer implementation based on the model type.
    """
    
    def __init__(self, slicer_impl):
        """
        Initialize the Slicer with a specific implementation.
        
        Args:
            slicer_impl: The slicer implementation to use
        """
        self.slicer_impl = slicer_impl


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
        # For now, we only support ONNX models.
        # In the future, this can be extended to support other model types.
        logger.info(f"Creating ONNX slicer for model: {model_path}")
        return Slicer(OnnxSlicer(model_path, save_path))


    def slice_model(self, output_path: Optional[str] = None, output_type: str = "dirs"):
        """
        Slice the model using the appropriate slicer implementation, then optionally convert output.

        Args:
            output_path: Directory to save the sliced model
            output_type: One of {'dsperse', 'dslice', 'dirs'}

        Returns:
            The result of the slicing operation (list of slice paths from slicer_impl)
        """
        if not output_path:
            raise ValueError("output_path must be provided for slicing")

        logger.info(f"Slicing model to output path: {output_path}")
        result = self.slicer_impl.slice_model(output_path=output_path)

        if output_type != "dirs":
            Converter.convert(output_path, output_type)

        return result



if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2 # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../../models/doom",
        2: "../../models/net",
        3: "../../models/resnet",
        4: "../../models/age",
        5: "../../models/version",
        6: "../../models/bert",
        7: "../../models/roberta"
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
        slices = slicer.slice_model(output_path=output_dir, output_type="dirs")

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

