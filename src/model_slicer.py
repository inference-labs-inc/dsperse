import os
from typing import Optional, Dict, List
import time
import json
import torch
import enum


def try_model_slicer(model_path: str, output_dir: Optional[str] = None,
                     strategy: str = "layer_type", max_segments: Optional[int] = None) -> None:
    """
    Test function to demonstrate the ModelSlicer functionality

    Args:
        model_path: Path to the model file (.pth or .pt)
        output_dir: Directory to save sliced model files (defaults to model directory)
        strategy: Slicing strategy to use
        max_segments: Maximum number of segments to create
    """
    from src.utils.model_utils import ModelUtils

    print(f"\n{'=' * 60}")
    print(f"Testing ModelSlicer on: {os.path.basename(model_path)}")
    print(f"{'=' * 60}")

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensuring output directory exists: {output_dir}")

    # Time the operation
    start_time = time.time()

    # Create ModelUtils instance for analysis
    model_utils = ModelUtils(model_path)

    # Print model summary before slicing
    print("\n[1] Analyzing model structure...")
    analysis = model_utils.analyze_model(verbose=False)

    print(f"  Model type: {analysis.get('model_type', 'Unknown')}")
    print(f"  Parameter count: {analysis.get('total_parameters', 0):,}")
    print(f"  Layer count: {len(analysis.get('layers', []))}")

    # Get layer types summary
    layer_types = {}
    for layer in analysis.get('layers', []):
        layer_type = layer.get('type', 'unknown')
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

    print("  Layer type distribution:")
    for layer_type, count in layer_types.items():
        print(f"    - {layer_type}: {count}")

    # Create and use the ModelSlicer
    print("\n[2] Slicing model...")
    slicer = ModelSlicer(model_utils)
    result = slicer.slice(model_path, output_dir, strategy, max_segments)

    # Print results
    if result['success']:
        elapsed_time = time.time() - start_time
        print(f"\n✓ Model successfully sliced in {elapsed_time:.2f} seconds")
        print(f"  Output directory: {result['output_dir']}")

        # Print segment details
        print("\n[3] Slice results:")
        print(f"  Total segments: {len(result['segments'])}")

        for i, segment in enumerate(result['segments']):
            print(f"\n  Segment {i}: {segment['filename']}")
            print(f"    Type: {segment['type']}")

            # Get layer details
            layer_info = None
            for layer in analysis.get('layers', []):
                if layer['name'] in [l['name'] for l in segment.get('layers', [])]:
                    layer_info = layer
                    break

            if layer_info:
                # Determine input/output sizes based on layer type
                if layer_info.get('type') == 'linear':
                    in_size = layer_info.get('in_features', 'N/A')
                    out_size = layer_info.get('out_features', 'N/A')
                elif layer_info.get('type') == 'conv':
                    in_size = layer_info.get('in_channels', 'N/A')
                    out_size = layer_info.get('out_channels', 'N/A')
                else:
                    in_size = 'N/A'
                    out_size = 'N/A'

                print(f"    Input size: {in_size}")
                print(f"    Output size: {out_size}")

                # Calculate parameters based on layer type
                if layer_info.get('type') == 'linear' and isinstance(in_size, int) and isinstance(out_size, int):
                    weights_params = in_size * out_size
                    has_bias = 'bias' in layer_info.get('parameters', {})
                    bias_params = out_size if has_bias else 0

                    print(f"    Weight parameters: {weights_params:,}")
                    print(f"    Bias parameters: {bias_params:,}")

                elif layer_info.get('type') == 'conv' and isinstance(in_size, int) and isinstance(out_size, int):
                    kernel_h, kernel_w = layer_info.get('kernel_size', (1, 1))
                    weights_params = in_size * out_size * kernel_h * kernel_w
                    has_bias = 'bias' in layer_info.get('parameters', {})
                    bias_params = out_size if has_bias else 0

                    print(f"    Weight parameters: {weights_params:,}")
                    print(f"    Bias parameters: {bias_params:,}")
                    print(f"    Kernel size: {kernel_h}x{kernel_w}")

                print(f"    Total parameters: {segment['parameters']:,}")
            else:
                print(f"    Layers: {segment['layer_count']}")
                print(f"    Parameters: {segment['parameters']:,}")

        # Calculate total parameters in segments for verification
        total_params = sum(segment['parameters'] for segment in result['segments'])
        if total_params == analysis.get('total_parameters', 0):
            print(f"\n✓ Parameter count verification: All parameters accounted for ({total_params:,})")
        else:
            print(f"\n⚠ Parameter count mismatch: Original={analysis.get('total_parameters', 0):,}, "
                  f"Sliced={total_params:,}")

        print(f"\nMetadata saved to: {result['metadata_path']}")
    else:
        print(f"\n✗ Error slicing model: {result.get('error', 'Unknown error')}")


class ModelSlicer:
    """
    Class for slicing PyTorch models into separate layer files
    """

    def __init__(self, model_utils=None):
        """
        Initialize ModelSlicer

        Args:
            model_utils: Optional ModelUtils instance to use
        """
        self.model_utils = model_utils

    def _initialize_model_utils(self, model_path: str) -> Optional[Dict]:
        """Initialize model utilities and load the model"""
        from src.utils.model_utils import ModelUtils

        # Create ModelUtils instance if not provided
        if self.model_utils is None:
            self.model_utils = ModelUtils(model_path)
            if not self.model_utils.load_model():
                return {'success': False, 'error': f"Failed to load model from {model_path}"}
        return None

    def _prepare_output_directory(self, model_path: str, output_dir: Optional[str] = None) -> str:
        """Set up and create output directory for sliced model files"""
        if output_dir is None:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            output_dir = os.path.join(model_dir, f"{model_name}_sliced")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _get_model_segments(self, layers: List[Dict], slice_points: List[int]) -> List[Dict]:
        """Divide model into segments based on slice points"""
        segments = []
        start_idx = 0

        # Add the ending point to make iteration easier
        all_points = sorted(slice_points + [len(layers) - 1])

        # Process each segment
        for i, end_idx in enumerate(all_points):
            # Skip invalid slice points
            if end_idx >= len(layers) or end_idx < start_idx:
                continue

            segment_layers = layers[start_idx:end_idx + 1]
            if not segment_layers:
                continue

            # Determine segment type based on layers
            segment_type = self._determine_segment_type(segment_layers)

            # Create segment info
            segment = {
                'index': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'type': segment_type,
                'layers': segment_layers,
            }

            segments.append(segment)
            start_idx = end_idx + 1

        return segments

    def _process_and_save_segment(self, segment: Dict, output_dir: str) -> Dict:
        """Process a segment, save it, and return segment information"""
        segment_type = segment['type']
        segment_idx = segment['index'] - 1  # Convert to 0-based index

        # Generate filename: {type}_{index}.pt
        filename = f"{segment_type}_{segment_idx}.pt"
        output_path = os.path.join(output_dir, filename)

        # Extract segment state dict
        segment_dict = self._extract_segment_state_dict(
            self.model_utils.state_dict,
            segment['layers']
        )

        # Save segment
        torch.save(segment_dict, output_path)

        # Create segment info with basic details
        segment_info = {
            'index': segment_idx,
            'type': segment_type,
            'filename': filename,
            'path': output_path,
            'layer_count': len(segment['layers']),
            'parameters': sum(layer.get('size', 0) for layer in segment['layers']),
            'layers': segment['layers']  # Include all layer info
        }

        # Add feature information
        self._add_feature_information(segment_info, segment)

        return segment_info

    def _add_feature_information(self, segment_info: Dict, segment: Dict):
        """Add input/output feature and activation information to segment info"""
        segment_type = segment['type']

        if segment['layers']:
            first_layer = segment['layers'][0]
            last_layer = segment['layers'][-1]

            # Add input/output features if available
            if segment_type == 'linear':
                in_features = first_layer.get('in_features')
                out_features = last_layer.get('out_features')

                if in_features is not None:
                    segment_info['in_features'] = in_features
                if out_features is not None:
                    segment_info['out_features'] = out_features

            # For conv layers
            elif segment_type == 'conv':
                in_features = first_layer.get('in_channels')
                out_features = last_layer.get('out_channels')

                if in_features is not None:
                    segment_info['in_features'] = in_features
                if out_features is not None:
                    segment_info['out_features'] = out_features

            # Add activation if available in the last layer
            if 'activation' in last_layer:
                segment_info['activation'] = last_layer['activation']

    def _create_and_save_metadata(self, model_path: str, output_dir: str, analysis: Dict,
                                  strategy: str, saved_segments: List[Dict], slice_points: List[int]) -> str:
        """Create and save metadata for the sliced model"""
        model_type = analysis.get('model_type', 'unknown')
        # Convert model_type to string if it's an Enum
        if isinstance(model_type, enum.Enum):
            model_type = str(model_type)

        metadata = {
            'original_model': model_path,
            'model_type': model_type,
            'total_parameters': analysis.get('total_parameters', 0),
            'slicing_strategy': strategy,
            'segments': saved_segments,
            'slice_points': slice_points
        }

        # Save metadata to JSON file
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata_path

    def slice(self, model_path: str, output_dir: Optional[str] = None,
              strategy: str = "layer_type", max_segments: Optional[int] = None) -> Dict:
        """
        Slice a model into separate layer files

        Args:
            model_path: Path to the model file (.pth or .pt)
            output_dir: Directory to save sliced model files (defaults to model directory)
            strategy: Slicing strategy to use
            max_segments: Maximum number of segments to create

        Returns:
            Dict with slicing results and metadata
        """
        # Initialize model utilities
        error_result = self._initialize_model_utils(model_path)
        if error_result:
            return error_result

        # Set up output directory
        output_dir = self._prepare_output_directory(model_path, output_dir)

        # Get model analysis
        analysis = self.model_utils.analyze_model(verbose=False)
        layers = analysis.get('layers', [])

        if not layers:
            return {'success': False, 'error': 'No layers found in model'}

        # Add activation information to layers
        self._gather_activation_information(model_path, layers)

        # Get slice points based on strategy
        slice_points = self.model_utils.get_slice_points(strategy, max_segments)

        # Get model segments
        segments = self._get_model_segments(layers, slice_points)

        # Process and save segments
        saved_segments = [self._process_and_save_segment(segment, output_dir)
                          for segment in segments]

        # Create and save metadata
        metadata_path = self._create_and_save_metadata(
            model_path, output_dir, analysis, strategy, saved_segments, slice_points
        )

        return {
            'success': True,
            'output_dir': output_dir,
            'segments': saved_segments,
            'metadata_path': metadata_path
        }

    def _determine_segment_type(self, layers: List[Dict]) -> str:
        """
        Determine the primary type of a segment based on its layers

        Args:
            layers: List of layer dictionaries

        Returns:
            String representing the segment type (fc, conv, etc.)
        """
        # Count layer types
        type_counts = {}
        for layer in layers:
            layer_type = layer.get('type', 'unknown')
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1

        # Remove 'unknown' type if there are other types
        if len(type_counts) > 1 and 'unknown' in type_counts:
            del type_counts['unknown']

        # Get the most common type
        if not type_counts:
            return 'misc'

        # Map internal type names to output names
        type_mapping = {
            'linear': 'fc',
            'conv': 'conv',
            'norm': 'norm',
            'embedding': 'emb',
            'unknown': 'misc'
        }

        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        return type_mapping.get(most_common_type, most_common_type)

    def _extract_segment_state_dict(self, full_state_dict: Dict, layers: List[Dict]) -> Dict:
        """
        Extract a portion of the state dict corresponding to specified layers

        Args:
            full_state_dict: The complete model state dict
            layers: List of layer dictionaries to extract

        Returns:
            Dictionary containing only the state dict entries for the specified layers
        """
        segment_dict = {}

        # Collect all layer names to extract
        layer_names = [layer['name'] for layer in layers]

        # Extract relevant keys from state dict
        for key, value in full_state_dict.items():
            # Check if this parameter belongs to one of our layers
            for layer_name in layer_names:
                if key.startswith(layer_name + '.') or key == layer_name:
                    segment_dict[key] = value
                    break

        return segment_dict

    def _gather_activation_information(self, model_path: str, layers: List[Dict]) -> Dict[str, str]:
        """
        Gather activation function information for layers using multiple strategies.

        Args:
            model_path: Path to the model file
            layers: List of layer dictionaries from model analysis

        Returns:
            Dictionary mapping layer names to activation function names
        """
        activations = {}

        # Strategy 1: Check if we have a model object for direct extraction
        if hasattr(self.model_utils, 'model') and self.model_utils.model is not None:
            try:
                activations = self._extract_activation_functions(self.model_utils.model)
            except Exception as e:
                print(f"Warning: Failed to extract activations from model: {e}")

        # Strategy 2: Check for a config file with the same name as the model file
        if not activations:
            try:
                # Try to find and load a config file
                model_dir = os.path.dirname(model_path)
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                potential_config_paths = [
                    os.path.join(model_dir, f"{model_name}_config.json"),
                    os.path.join(model_dir, "config.json"),
                    os.path.join(model_dir, f"{model_name}.json"),
                    os.path.join(model_dir, "test_config.json"),  # For test models
                ]

                for config_path in potential_config_paths:
                    if os.path.exists(config_path):
                        print(f"Found configuration file: {config_path}")
                        with open(config_path, 'r') as f:
                            config = json.load(f)

                        # Extract activations from config
                        if 'layers' in config:
                            for layer_name, layer_info in config['layers'].items():
                                if 'activation' in layer_info:
                                    activations[layer_name] = layer_info['activation']
                        break
            except Exception as e:
                print(f"Warning: Failed to extract activations from config: {e}")

        # Strategy 3: Try to infer activations from layer names as a last resort
        if not activations:
            print("Warning: No activation information found, inferring from layer structure")
            activations = self._infer_activations_from_layers(layers)

        # Add activations to layer information
        activation_count = 0
        for layer in layers:
            layer_name = layer.get('name')
            if layer_name in activations:
                layer['activation'] = activations[layer_name]
                activation_count += 1

        if activation_count > 0:
            print(f"Added activation information to {activation_count} layers")

        return activations

    def _extract_activation_functions(self, model) -> Dict[str, str]:
        """
        Extract activation functions from a PyTorch model.

        Args:
            model: PyTorch model object

        Returns:
            Dictionary mapping layer names to activation function names
        """
        activations = {}

        # Handle case when model is None
        if model is None:
            return activations

        # Get all named modules
        for name, module in model.named_modules():
            # Skip the model itself
            if name == '':
                continue

            # Check for common activation functions
            if isinstance(module, torch.nn.ReLU):
                activations[name] = "ReLU"
            elif isinstance(module, torch.nn.LeakyReLU):
                activations[name] = "LeakyReLU"
            elif isinstance(module, torch.nn.PReLU):
                activations[name] = "PReLU"
            elif isinstance(module, torch.nn.ELU):
                activations[name] = "ELU"
            elif isinstance(module, torch.nn.SELU):
                activations[name] = "SELU"
            elif isinstance(module, torch.nn.GELU):
                activations[name] = "GELU"
            elif isinstance(module, torch.nn.Sigmoid):
                activations[name] = "Sigmoid"
            elif isinstance(module, torch.nn.Tanh):
                activations[name] = "Tanh"
            elif isinstance(module, torch.nn.Softmax):
                activations[name] = "Softmax"
            elif isinstance(module, torch.nn.Softplus):
                activations[name] = "Softplus"
            elif isinstance(module, torch.nn.Softsign):
                activations[name] = "Softsign"

            # For sequential modules, check if they contain activation functions
            if isinstance(module, torch.nn.Sequential):
                for i, submodule in enumerate(module):
                    sub_name = f"{name}.{i}"
                    if isinstance(submodule, torch.nn.ReLU):
                        activations[sub_name] = "ReLU"
                    elif isinstance(submodule, torch.nn.LeakyReLU):
                        activations[sub_name] = "LeakyReLU"
                    # ... and so on for other activation types

        return activations

    def _infer_activations_from_layers(self, layers: List[Dict]) -> Dict[str, str]:
        """
        Attempt to infer activation functions from layer names and structure.

        Args:
            layers: List of layer dictionaries from model analysis

        Returns:
            Dictionary mapping layer names to inferred activation function names
        """
        activations = {}

        # Common patterns in layer naming
        relu_patterns = ["relu", "ReLU"]
        sigmoid_patterns = ["sigmoid", "Sigmoid"]
        tanh_patterns = ["tanh", "Tanh"]
        leaky_relu_patterns = ["leaky", "LeakyReLU"]
        elu_patterns = ["elu", "ELU"]
        softmax_patterns = ["softmax", "Softmax"]

        for i, layer in enumerate(layers):
            layer_name = layer.get('name', '')

            # Check for activation in layer name
            if any(pattern in layer_name for pattern in relu_patterns):
                activations[layer_name] = "ReLU"
            elif any(pattern in layer_name for pattern in sigmoid_patterns):
                activations[layer_name] = "Sigmoid"
            elif any(pattern in layer_name for pattern in tanh_patterns):
                activations[layer_name] = "Tanh"
            elif any(pattern in layer_name for pattern in leaky_relu_patterns):
                activations[layer_name] = "LeakyReLU"
            elif any(pattern in layer_name for pattern in elu_patterns):
                activations[layer_name] = "ELU"
            elif any(pattern in layer_name for pattern in softmax_patterns):
                activations[layer_name] = "Softmax"

            # For layers without explicit activations in names,
            # make educated guesses based on layer type and position
            if layer_name not in activations:
                layer_type = layer.get('type')
                if layer_type == 'conv' and i < len(layers) - 1:
                    activations[layer_name] = "ReLU"  # Common default for conv layers
                elif layer_type == 'linear':
                    # For the last layer in classification models, often Softmax
                    if i == len(layers) - 1:
                        # If the output dimension is small (typical for classification)
                        if layer.get('out_features', 0) < 100:
                            activations[layer_name] = "Softmax"
                    else:
                        # For hidden layers, ReLU is a common choice
                        activations[layer_name] = "ReLU"

        return activations


# Example usage:
if __name__ == "__main__":
    # Choose which model to test
    model_choice = 6  # Change this to test different models

    if model_choice == 1:
        # Doom model
        model_dir = "models/doom"
        model_path = os.path.join(model_dir, "doom.pth")
        output_folder = os.path.join(model_dir, "output")
        strategy = "single_layer"  # Slice by layer types (fc, conv, etc.)

    elif model_choice == 2:
        # Test model
        model_dir = "models/test_model"
        model_path = os.path.join(model_dir, "test_model.pth")
        output_folder = os.path.join(model_dir, "output")
        strategy = "single_layer"  # Create evenly balanced slices

    elif model_choice == 3:
        # Embedded test model
        model_dir = "models/test_model_embedded"
        model_path = os.path.join(model_dir, "test_model_embedded.pth")
        output_folder = os.path.join(model_dir, "output")
        strategy = "single_layer"  # Slice at layer type transitions

    elif model_choice == 4:
        # Transformer model
        model_path = "/path/to/transformer/model.pth"
        output_folder = "/path/to/transformer/model_slices"
        strategy = "layer_type"

    elif model_choice == 5:
        # Test model
        model_dir = "models/test_model_with_biases"
        model_path = os.path.join(model_dir, "test_model.pth")
        output_folder = os.path.join(model_dir, "output")
        strategy = "layer_type"  # Create evenly balanced slices

    elif model_choice == 6:
        # Test model
        model_dir = "models/test_cnn_model_with_biases"
        model_path = os.path.join(model_dir, "test_cnn_model.pth")
        output_folder = os.path.join(model_dir, "output")
        strategy = "layer_type"  # Create evenly balanced slices

    # Run the test with the selected model
    try_model_slicer(
        model_path=model_path,
        output_dir=output_folder,
        strategy=strategy,
        max_segments=None  # Set to a number to limit segments
    )
