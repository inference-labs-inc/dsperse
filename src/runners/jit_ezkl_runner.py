import json
import os
import subprocess
import threading
import time
import random
from typing import Any, Set, Dict, List

import torch
from pathlib import Path

from torch import Tensor

from src.runners import runner_utils
from src.runners.runner_utils import RunnerUtils
from src.utils.model_utils import ModelUtils
from src.runners.ezkl_runner import EzklRunner

env = os.environ


class JITEzklRunner(EzklRunner):
    def __init__(self, model_directory: str):
        super().__init__(model_directory)
        
    def random_layer_testing(self, sliced_model: str, circuitized_sliced_model: str, input_data: Dict) -> Dict:
        """
        Perform Just-in-Time testing on randomly selected layers.
        
        Args:
            sliced_model: Path to the sliced model metadata
            circuitized_sliced_model: Path to the circuitized sliced model
            input_data: Input data for the model
            
        Returns:
            Dict containing results for each layer (circuitized or inference)
        """
        # Load metadata for both sliced and circuitized models
        sliced_metadata = ModelUtils.load_metadata(sliced_model)
        circuitized_metadata = ModelUtils.load_metadata(circuitized_sliced_model)
        
        # Get total number of slices
        num_slices = len(sliced_metadata['segments'])
        
        # TODO: Implement whitelist/blacklist logic here
        # For now, assume all layers can be tested
        blacklisted_layers = set()  # Can be populated based on requirements
        
        # Generate random number of slices to test (between 1 and total slices)
        num_slices_to_test = random.randint(1, num_slices)
        
        # Generate random set of layer indices to test
        layers_to_test = set()
        while len(layers_to_test) < num_slices_to_test:
            layer_idx = random.randint(0, num_slices - 1)
            if layer_idx not in blacklisted_layers:
                layers_to_test.add(layer_idx)
        
        # Initialize output dictionary
        output_results = {
            'layer_results': {},
            'selected_layers': list(layers_to_test),
            'total_time': 0,
            'memory_usage': {}
        }
        
        # Start timing
        start_time = time.time()
        current_input = input_data
        
        # Process each layer
        for layer_idx in range(num_slices):
            layer_start_time = time.time()
            
            if layer_idx in layers_to_test:
                # Run circuitized layer through EZKL witness generation
                print(f"Running layer {layer_idx} through EZKL witness generation")
                layer_result = self.run_circuitized_layer(
                    layer_idx,
                    circuitized_sliced_model,
                    current_input
                )
                output_results['layer_results'][layer_idx] = {
                    'type': 'circuitized',
                    'time': time.time() - layer_start_time,
                    'output': layer_result['output'],
                    'witness_path': layer_result['witness_path']
                }
            else:
                # Run regular inference on the sliced layer
                print(f"Running layer {layer_idx} through regular inference")
                layer_result = self.run_inference_layer(
                    layer_idx,
                    current_input
                )
                output_results['layer_results'][layer_idx] = {
                    'type': 'inference',
                    'time': time.time() - layer_start_time,
                    'output': layer_result
                }
            
            # Update input for next layer
            current_input = {
                'input_data': layer_result['output'] if isinstance(layer_result, dict) else layer_result
            }
        
        # Calculate total time
        output_results['total_time'] = time.time() - start_time
        
        return output_results

    def run_circuitized_layer(self, layer_idx: int, circuitized_model_path: str, input_data: Dict) -> Dict:
        """Run a single layer through the circuitized (EZKL) pipeline."""
        segment_name = f"segment_{layer_idx}"
        segment_dir = os.path.join(circuitized_model_path, "slices", segment_name)
        
        # Reshape input data based on layer type and index
        if layer_idx == 0:  # First conv layer
            input_shape = [1, 3, 32, 32]
        elif layer_idx == 1:  # Second conv layer
            input_shape = [1, 6, 28, 28]
        elif layer_idx == 2:  # First FC layer
            input_shape = [1, 16, 5, 5]  # Will be flattened by the layer
        elif layer_idx == 3:  # Second FC layer
            input_shape = [1, 120]
        else:  # Last FC layer
            input_shape = [1, 84]
        
        # Save input data to a JSON file
        input_file = os.path.join(segment_dir, f"{segment_name}_input.json")
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        
        # Format input data according to EZKL's expected format
        input_tensor = torch.tensor(input_data['input_data']).reshape(input_shape)
        ezkl_input = {
            "input_data": input_tensor.flatten().tolist()
        }
        with open(input_file, 'w') as f:
            json.dump(ezkl_input, f)
        
        # Check if witness already exists
        witness_file = os.path.join(segment_dir, f"{segment_name}_witness.json")
        model_file = os.path.join(segment_dir, f"{segment_name}_model.compiled")
        
        if not os.path.exists(witness_file):
            print(f"Generating witness for layer {layer_idx}...")
            subprocess.run(
                [
                    "ezkl",
                    "gen-witness",
                    "--compiled-circuit", model_file,
                    "--data", input_file,
                    "--output", witness_file
                ],
                check=True
            )
        else:
            print(f"Using cached witness for layer {layer_idx}")
        
        # Read witness output
        with open(witness_file, 'r') as f:
            witness_data = json.load(f)
            
        return {
            'output': witness_data["output"],
            'witness_path': witness_file
        }

    def run_inference_layer(self, layer_idx: int, input_data: Dict) -> Dict:
        """Run a single layer through regular PyTorch inference."""
        # Load the model
        model_path = os.path.join(os.path.dirname(self.sliced_model_path), f"segment_{layer_idx}", f"segment_{layer_idx}.pt")
        model = torch.load(model_path)
        
        # Reshape input data based on layer index
        if layer_idx == 0:  # First conv layer
            input_shape = [1, 3, 32, 32]
        elif layer_idx == 1:  # Second conv layer
            input_shape = [1, 6, 28, 28]
        elif layer_idx == 2:  # First FC layer
            input_shape = [1, 16, 5, 5]
        elif layer_idx == 3:  # Second FC layer
            input_shape = [1, 120]
        else:  # Last FC layer
            input_shape = [1, 84]
        
        # Convert input data to tensor and reshape
        input_tensor = torch.tensor(input_data).reshape(input_shape)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert output to list
        return output.flatten().tolist()

    def _get_segment_class(self, idx: int):
        """Get the appropriate segment class based on the model type and index."""
        if self.model_directory.endswith('net'):
            from src.models.net.model import (
                Conv1Segment as NetConv1,
                Conv2Segment as NetConv2,
                FC1Segment as NetFC1,
                FC2Segment as NetFC2,
                FC3Segment as NetFC3
            )
            mapping = {
                0: NetConv1,
                1: NetConv2,
                2: NetFC1,
                3: NetFC2,
                4: NetFC3
            }
        else:  # doom model
            from src.models.doom.model import (
                Conv1Segment,
                Conv2Segment,
                Conv3Segment,
                FC1Segment,
                FC2Segment
            )
            mapping = {
                0: Conv1Segment,
                1: Conv2Segment,
                2: Conv3Segment,
                3: FC1Segment,
                4: FC2Segment
            }
        
        segment_class = mapping.get(idx)
        if segment_class is None:
            raise ValueError(f"No corresponding class found for segment index {idx}")
        return segment_class

    def generate_onnx_model(self, layer_idx: int, sliced_model_path: str, output_dir: str):
        """Generate ONNX model for a layer."""
        metadata = ModelUtils.load_metadata(sliced_model_path)
        segment_info = metadata['segments'][layer_idx]
        
        # Load the segment model
        segment_path = os.path.join(sliced_model_path, segment_info['filename'])
        segment_class = self._get_segment_class(layer_idx)
        
        segment_model = segment_class()
        segment_model.load_state_dict(torch.load(segment_path))
        segment_model.eval()
        
        # Create dummy input based on layer type
        if segment_info['type'] == 'conv':
            if layer_idx == 0:  # First conv layer
                dummy_input = torch.randn(1, segment_info['in_features'], 32, 32)
            else:  # Other conv layers
                prev_out_channels = metadata['segments'][layer_idx - 1]['out_features']
                dummy_input = torch.randn(1, prev_out_channels, 16, 16)  # After pooling
        elif segment_info['type'] == 'fc':
            if 'input_reshape' in segment_info:
                # Handle flattening from conv to fc
                from_shape = segment_info['input_reshape']['from_shape']
                batch_size = 1
                from_shape[0] = batch_size
                dummy_input = torch.randn(*from_shape)
            else:
                # Regular FC layer
                dummy_input = torch.randn(1, segment_info['layers'][0]['in_features'])
        
        # Export to ONNX
        segment_name = f"segment_{layer_idx}"
        onnx_path = os.path.join(output_dir, segment_name, f"{segment_name}.onnx")
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        torch.onnx.export(
            segment_model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        return onnx_path

    def setup_ezkl_environment(self, sliced_model_path: str, circuitized_model_path: str):
        """Set up the EZKL environment by generating ONNX models and compiling them."""
        metadata = ModelUtils.load_metadata(sliced_model_path)
        num_segments = len(metadata['segments'])
        
        for layer_idx in range(num_segments):
            # Generate ONNX model
            onnx_path = self.generate_onnx_model(layer_idx, sliced_model_path, circuitized_model_path)
            
            # Generate settings file
            segment_name = f"segment_{layer_idx}"
            segment_dir = os.path.join(circuitized_model_path, "slices", segment_name)
            settings_path = os.path.join(segment_dir, f"{segment_name}_settings.json")
            
            subprocess.run(
                [
                    "ezkl",
                    "gen-settings",
                    "--model", onnx_path,
                    "--settings-path", settings_path
                ],
                check=True
            )
            
            # Compile model
            model_path = os.path.join(segment_dir, f"{segment_name}_model.compiled")
            subprocess.run(
                [
                    "ezkl",
                    "compile-circuit",
                    "--model", onnx_path,
                    "--settings-path", settings_path,
                    "--compiled-circuit", model_path
                ],
                check=True
            )
            
        return True


if __name__ == "__main__":
    # Example usage
    model_choice = 2  # 1 for doom, 2 for net
    
    base_paths = {
        1: "src/models/doom",
        2: "src/models/net"
    }
    
    model_dir = base_paths[model_choice]
    runner = JITEzklRunner(model_dir)
    
    # Use existing input file
    with open(os.path.join(model_dir, "input.json"), 'r') as f:
        input_data = json.load(f)
    
    # Paths to sliced and circuitized models
    sliced_model_path = os.path.join(model_dir, "slices")
    circuitized_model_path = os.path.join(model_dir, "ezkl")
    
    # Set up EZKL environment (generate ONNX models and compile them)
    print("Setting up EZKL environment...")
    runner.setup_ezkl_environment(sliced_model_path, circuitized_model_path)
    print("EZKL environment setup complete.")
    
    # Run random layer testing
    print("Running random layer testing...")
    results = runner.random_layer_testing(
        sliced_model_path,
        circuitized_model_path,
        input_data
    )
    
    print("Results:", json.dumps(results, indent=2)) 