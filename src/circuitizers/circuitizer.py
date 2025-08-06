"""
Circuitizer orchestrator module.

This module provides a unified interface for circuitizing models of different types.
It orchestrates the circuitization process by delegating to the appropriate circuitizer implementation
based on the model type.
"""

import os
import logging
from typing import Optional, Dict, Any

from src.circuitizers.ezkl_circuitizer import EZKLCircuitizer

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
            return Circuitizer(EZKLCircuitizer())
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
        
    def circuitize(self, model_path: str, input_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Circuitize the model using the appropriate circuitizer implementation.
        
        Args:
            model_path: Path to the model file or directory
            input_file: Optional path to input file for calibration
            
        Returns:
            The result of the circuitization operation
        """
        logger.info(f"Circuitizing model: {model_path}")
        return self.circuitizer_impl.circuitize(model_path=model_path, input_file=input_file)