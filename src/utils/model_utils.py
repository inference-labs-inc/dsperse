import enum

import numpy as np
import os
import re
import torch
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union


class ModelType(enum.Enum):
    SEQUENTIAL = "sequential"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    FCNN = "fcnn"  # Add this new type for fully connected networks
    HYBRID = "hybrid"  # Add this for mixed architecture models (e.g., CNN + FCNN)
    UNKNOWN = "unknown"



class ModelUtils:
    """Utility class for model analysis and inspection"""

    def __init__(self, model_path=None):
        """
        Initialize ModelUtils with a path to a model file

        Args:
            model_path: Path to the model file (.pth, .pt, etc.)
        """
        self.model_path = model_path
        self.state_dict = None
        self.model_type = None

    def load_model(self) -> bool:
        """
        Load the model state dictionary from the provided path

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"Error: Invalid model path {self.model_path}")
            return False

        try:
            # Load with torch.load and appropriate map_location
            self.state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            print(f"Loaded model with type: {type(self.state_dict)}")

            # Handle different state dict formats
            if isinstance(self.state_dict, dict):
                if 'state_dict' in self.state_dict:
                    self.state_dict = self.state_dict['state_dict']
                    print("Using 'state_dict' key from loaded dictionary")
                elif 'model_state_dict' in self.state_dict:
                    self.state_dict = self.state_dict['model_state_dict']
                    print("Using 'model_state_dict' key from loaded dictionary")
                elif 'net' in self.state_dict:
                    self.state_dict = self.state_dict['net']
                    print("Using 'net' key from loaded dictionary")

                # Debug - print the keys
                if isinstance(self.state_dict, dict):
                    print(f"State dict keys: {list(self.state_dict.keys())[:5]} (showing first 5)")
                else:
                    print(f"State dict is not a dictionary but {type(self.state_dict)}")

            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def analyze_model(self, verbose=True, output_file=None) -> Dict:
        """
        Analyze model architecture and extract key information for slicing

        Args:
            verbose: Whether to print analysis results
            output_file: Optional file path to write verbose output

        Returns:
            Dict containing model analysis results
        """
        if self.state_dict is None:
            if not self.load_model():
                return {"error": "Failed to load model"}

        # Core analysis steps
        analysis = {}

        # 1. Detect model type
        self.model_type = self._detect_model_type()
        analysis["model_type"] = self.model_type

        # 2. Extract layer structure
        layers = self._extract_layers()
        analysis["layers"] = layers

        # 3. Identify key model characteristics
        analysis["total_parameters"] = self._count_parameters()
        analysis["layer_groups"] = self._identify_layer_groups(layers)

        # Print results if requested
        if verbose:
            self._print_analysis(analysis, output_file)

        return analysis

    def _detect_model_type(self) -> ModelType:
        """
        Detect the type of neural network architecture

        Returns:
            ModelType enum value
        """
        if self.state_dict is None:
            return ModelType.UNKNOWN

        if not isinstance(self.state_dict, dict):
            print(f"Warning: state_dict is not a dictionary but {type(self.state_dict)}")
            return ModelType.UNKNOWN

        keys = list(self.state_dict.keys())

        # Check for transformer patterns
        has_transformer = any(pattern in key for key in keys
                              for pattern in ['attention', 'mha', 'self_attn', 'encoder.layers', 'decoder.layers'])

        # Check for CNN patterns
        has_cnn = self._has_cnn_layers(keys)

        # Check for linear/FCNN patterns
        has_linear = any('linear' in key.lower() or
                         ('weight' in key.lower() and not any(c in key.lower() for c in ['conv', 'attention']))
                         for key in keys)

        # Detect if layers are organized in a sequential pattern
        is_sequential = any(re.match(r'^\w+\.\d+\.', key) for key in keys)

        # Apply detection logic with priority
        if has_transformer:
            if has_cnn or has_linear:
                return ModelType.HYBRID  # Transformer + something else
            return ModelType.TRANSFORMER

        if has_cnn:
            if has_linear:
                # Check if this is actually a hybrid model (CNN + FCNN)
                # This is common in models like DOOM where conv layers feed into FC layers
                conv_keys = [k for k in keys if any(c in k.lower() for c in ['conv', 'bn', 'pool'])]
                linear_keys = [k for k in keys if 'linear' in k.lower() or 'fc' in k.lower()]

                # If we have significant presence of both types, it's likely a hybrid
                if len(conv_keys) > 1 and len(linear_keys) > 1:
                    return ModelType.HYBRID

                # Otherwise, still primarily a CNN (with classification head)
                return ModelType.CNN
            return ModelType.CNN

        if has_linear:
            return ModelType.FCNN

        if is_sequential:
            return ModelType.SEQUENTIAL

        return ModelType.UNKNOWN

    def _has_cnn_layers(self, keys) -> bool:
        """
        Check if the model has convolutional layers

        Args:
            keys: List of state dict keys

        Returns:
            bool: True if CNN layers are detected
        """
        # Check for common CNN naming patterns
        if any(pattern in key for key in keys
               for pattern in ['conv', 'features']):
            return True

        # Check for 4D weight tensors (typical for conv layers)
        for key in keys:
            if key.endswith('.weight') and key in self.state_dict:
                tensor = self.state_dict[key]
                if torch.is_tensor(tensor) and len(tensor.shape) == 4:
                    return True

        return False

    def _extract_layers(self) -> List[Dict]:
        """
        Extract layer information from state dict

        Returns:
            List of dicts with layer details
        """
        if self.state_dict is None or not isinstance(self.state_dict, dict):
            print("Warning: Cannot extract layers - state dict is None or not a dictionary")
            return []

        layers = []
        layer_params = {}

        # Group parameters by layer
        for key, tensor in self.state_dict.items():
            if not torch.is_tensor(tensor):
                continue

            # Extract layer name (without parameter suffix)
            layer_parts = key.split('.')
            param_name = layer_parts[-1]  # weight, bias, etc.

            if len(layer_parts) == 1:
                # Single parameter (rare case)
                layer_name = layer_parts[0]
            else:
                # Try to find a sensible layer name
                # For most models, the last part is the parameter name (weight/bias)
                if param_name in ['weight', 'bias', 'running_mean', 'running_var']:
                    layer_name = '.'.join(layer_parts[:-1])
                else:
                    # If not a standard param name, use the full key
                    layer_name = key

            # Initialize tracking dict for this layer if needed
            if layer_name not in layer_params:
                layer_params[layer_name] = {
                    'name': layer_name,
                    'parameters': {},
                    'shape': None,
                    'type': None,
                    'size': 0
                }

            # Add parameter info
            layer_params[layer_name]['parameters'][param_name] = {
                'shape': list(tensor.shape),
                'size': tensor.numel()
            }
            layer_params[layer_name]['size'] += tensor.numel()

            # Try to determine layer type from parameter shapes
            if param_name == 'weight':
                shape = tensor.shape
                if len(shape) == 4:  # Conv layer
                    layer_params[layer_name]['type'] = 'conv'
                    layer_params[layer_name]['shape'] = shape
                    layer_params[layer_name]['in_channels'] = shape[1]
                    layer_params[layer_name]['out_channels'] = shape[0]
                    layer_params[layer_name]['kernel_size'] = (shape[2], shape[3])
                elif len(shape) == 2:  # FC layer
                    layer_params[layer_name]['type'] = 'linear'
                    layer_params[layer_name]['shape'] = shape
                    layer_params[layer_name]['in_features'] = shape[1]
                    layer_params[layer_name]['out_features'] = shape[0]
                elif len(shape) == 1:  # Norm layer or embedding
                    if 'norm' in layer_name or 'bn' in layer_name:
                        layer_params[layer_name]['type'] = 'norm'
                    elif 'embedding' in layer_name:
                        layer_params[layer_name]['type'] = 'embedding'
                    else:
                        layer_params[layer_name]['type'] = 'unknown'
                    layer_params[layer_name]['shape'] = shape

        # Convert to sorted list
        layers = list(layer_params.values())

        # Try to sort layers in logical order
        return self._sort_layers(layers)

    def _sort_layers(self, layers: List[Dict]) -> List[Dict]:
        """
        Sort layers in a likely execution order

        Args:
            layers: List of layer dictionaries

        Returns:
            Sorted list of layer dictionaries
        """

        # First try grouping by numerical prefixes
        def extract_prefix_and_number(name):
            match = re.match(r'([a-zA-Z_]+)(\d+)(\..*)?', name)
            if match:
                prefix, number, suffix = match.groups()
                return prefix, int(number), suffix or ''
            return name, float('inf'), ''

        # Group layers by common prefixes
        prefix_groups = {}
        for layer in layers:
            prefix, number, suffix = extract_prefix_and_number(layer['name'])
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append((number, suffix, layer))

        # Sort each group and flatten
        sorted_layers = []
        for prefix, group in prefix_groups.items():
            group.sort()  # Sort by number then suffix
            sorted_layers.extend(layer for _, _, layer in group)

        return sorted_layers

    def _count_parameters(self) -> int:
        """
        Count total number of parameters in the model

        Returns:
            Total parameter count
        """
        if self.state_dict is None or not isinstance(self.state_dict, dict):
            return 0

        return sum(tensor.numel() for tensor in self.state_dict.values()
                   if torch.is_tensor(tensor))

    def _identify_layer_groups(self, layers: List[Dict]) -> Dict:
        """
        Group layers by type and identify potential slicing points

        Args:
            layers: List of layer dictionaries

        Returns:
            Dictionary of layer groups and potential slicing points
        """
        groups = {
            'conv': [],
            'linear': [],
            'norm': [],
            'embedding': [],
            'other': []
        }

        # Group layers by type
        for layer in layers:
            layer_type = layer.get('type', 'other')
            if layer_type in groups:
                groups[layer_type].append(layer['name'])
            else:
                groups['other'].append(layer['name'])

        # Identify potential slicing points (transitions between layer types)
        slicing_points = []
        prev_type = None

        for layer in layers:
            layer_type = layer.get('type')
            # Type transitions are good slicing points
            if prev_type and layer_type != prev_type:
                slicing_points.append({
                    'after': prev_type,
                    'before': layer_type,
                    'layer_name': layer['name']
                })
            prev_type = layer_type

        return {
            'groups': groups,
            'potential_slicing_points': slicing_points
        }

    def _print_analysis(self, analysis: Dict, output_file=None):
        """
        Print analysis results to console or file

        Args:
            analysis: Model analysis dictionary
            output_file: Optional file path for output
        """
        output = []  # Collect output lines

        # Header
        output.append("=" * 60)
        output.append("MODEL ARCHITECTURE ANALYSIS")
        output.append("=" * 60)

        # Model type
        output.append(f"\nModel Type: {analysis['model_type'].value}")

        # Parameters
        total_params = analysis['total_parameters']
        if total_params > 1_000_000:
            params_str = f"{total_params / 1_000_000:.2f}M"
        elif total_params > 1_000:
            params_str = f"{total_params / 1_000:.1f}K"
        else:
            params_str = str(total_params)
        output.append(f"Total Parameters: {params_str} ({total_params:,})")

        # Layer summary
        layers = analysis['layers']
        output.append(f"\nLayers: {len(layers)}")

        # Layer type distribution
        layer_groups = analysis['layer_groups']['groups']
        output.append("\nLayer Types:")
        for group_name, group_layers in layer_groups.items():
            if group_layers:
                output.append(f"  - {group_name.capitalize()}: {len(group_layers)}")

        # Potential slicing points
        slicing_points = analysis['layer_groups']['potential_slicing_points']
        if slicing_points:
            output.append("\nPotential Slicing Points:")
            for i, point in enumerate(slicing_points):
                output.append(f"  {i + 1}. After {point['after']} before {point['before']} at {point['layer_name']}")

        # Details of each layer
        output.append("\nLayer Details:")
        for i, layer in enumerate(layers):
            layer_type = layer.get('type', 'unknown')
            layer_name = layer['name']
            params = layer['size']

            # Format layer details based on type
            if layer_type == 'conv':
                details = f"{layer_type.upper()} | {layer.get('in_channels')}→{layer.get('out_channels')} | k={layer.get('kernel_size')}"
            elif layer_type == 'linear':
                details = f"{layer_type.upper()} | {layer.get('in_features')}→{layer.get('out_features')}"
            else:
                shape_str = str(layer.get('shape', ''))
                details = f"{layer_type.upper()} | {shape_str}"

            output.append(f"  {i + 1}. {layer_name:<30} | {params:,} params | {details}")

        # Footer
        output.append("\n" + "=" * 60)

        # Write to file or print to console
        full_output = "\n".join(output)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(full_output)
            print(f"Analysis written to {output_file}")
        else:
            print(full_output)

    def get_layer_at_index(self, index: int) -> Dict:
        """
        Get information about a specific layer by index

        Args:
            index: Index of the layer to retrieve

        Returns:
            Dictionary with layer information or empty dict if index is invalid
        """
        if self.state_dict is None:
            if not self.load_model():
                return {}

        layers = self._extract_layers()
        if 0 <= index < len(layers):
            return layers[index]
        return {}

    def get_layer_by_name(self, name: str) -> Dict:
        """
        Get information about a specific layer by name

        Args:
            name: Name of the layer to retrieve

        Returns:
            Dictionary with layer information or empty dict if name is not found
        """
        if self.state_dict is None:
            if not self.load_model():
                return {}

        layers = self._extract_layers()
        for layer in layers:
            if layer['name'] == name:
                return layer
        return {}

    def verify_slice_integrity(self, slice_points: List[int]) -> Dict:
        """
        Verify the integrity of proposed model slices

        Args:
            slice_points: List of indices where the model should be sliced

        Returns:
            Dictionary with verification results
        """
        if self.state_dict is None:
            if not self.load_model():
                return {'valid': False, 'error': 'Model not loaded'}

        layers = self._extract_layers()
        total_layers = len(layers)

        # Verify all slice points are valid indices
        valid_points = []
        invalid_points = []
        for point in slice_points:
            if 0 <= point < total_layers:
                valid_points.append(point)
            else:
                invalid_points.append(point)

        # Calculate resulting segments
        segments = []
        prev_point = 0
        for point in sorted(valid_points):
            segments.append((prev_point, point))
            prev_point = point + 1
        segments.append((prev_point, total_layers))

        # Calculate parameters and layer counts per segment
        segment_stats = []
        for i, (start, end) in enumerate(segments):
            segment_layers = layers[start:end]
            param_count = sum(layer['size'] for layer in segment_layers)
            layer_types = {}
            for layer in segment_layers:
                layer_type = layer.get('type', 'unknown')
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

            segment_stats.append({
                'segment': i + 1,
                'start_layer': start,
                'end_layer': end - 1,
                'layer_count': end - start,
                'parameter_count': param_count,
                'layer_types': layer_types
            })

        return {
            'valid': len(invalid_points) == 0,
            'total_layers': total_layers,
            'valid_slice_points': valid_points,
            'invalid_slice_points': invalid_points,
            'segment_count': len(segments),
            'segment_stats': segment_stats
        }

    def estimate_slice_sizes(self, slice_points: List[int]) -> Dict:
        """
        Estimate the size of each model slice based on proposed slice points

        Args:
            slice_points: List of indices where the model should be sliced

        Returns:
            Dictionary with size estimates for each slice
        """
        if self.state_dict is None:
            if not self.load_model():
                return {'error': 'Model not loaded'}

        verification = self.verify_slice_integrity(slice_points)
        if not verification['valid']:
            return {'error': 'Invalid slice points', 'details': verification}

        # Calculate memory footprint for each segment
        for segment in verification['segment_stats']:
            # Estimate size in MB (parameters * 4 bytes for float32)
            size_bytes = segment['parameter_count'] * 4
            size_mb = size_bytes / (1024 * 1024)
            segment['estimated_size_mb'] = round(size_mb, 2)

        return {
            'total_segments': verification['segment_count'],
            'segments': verification['segment_stats']
        }

    def get_slice_points(self, strategy: str = "layer_type", max_segments: int = None) -> List[int]:
        """
        Get recommended model slicing points based on specified strategy

        Args:
            strategy: Slicing strategy ("layer_type", "balanced", or "transitions")
            max_segments: Optional maximum number of segments to create (caps the number of slice points)

        Returns:
            List of recommended slice point indices
        """
        if self.state_dict is None:
            if not self.load_model():
                return []

        layers = self._extract_layers()
        if not layers:
            return []

        total_layers = len(layers)
        slice_points = []

        if strategy == "single_layer":
            # Slice after every layer except the last one
            for i in range(total_layers - 1):
                slice_points.append(i)

        elif strategy == "layer_type":
            # Slice after each fully connected and convolutional layer
            for i, layer in enumerate(layers):
                # Skip if it's the last layer
                if i == total_layers - 1:
                    continue

                layer_type = layer.get('type', 'unknown')
                # Slice after linear (FC) and conv layers
                if layer_type in ['linear', 'conv']:
                    slice_points.append(i)

        elif strategy == "balanced":
            # Create evenly sized segments
            if max_segments and max_segments > 1:
                segment_size = total_layers // max_segments
                for i in range(1, max_segments):
                    point = i * segment_size - 1
                    if 0 <= point < total_layers - 1:  # Avoid slicing at the very end
                        slice_points.append(point)

        elif strategy == "transitions":
            # Slice at transitions between different layer types
            prev_type = None
            for i, layer in enumerate(layers):
                # Skip if it's the last layer
                if i == total_layers - 1:
                    continue

                layer_type = layer.get('type', 'unknown')
                if prev_type and layer_type != prev_type:
                    slice_points.append(i - 1)  # Slice after the previous layer
                prev_type = layer_type

        else:
            # Default: hybrid approach prioritizing FC and Conv layers but also keeping segments balanced
            # Find all FC and Conv layers
            type_slices = []
            for i, layer in enumerate(layers):
                if i == total_layers - 1:
                    continue

                layer_type = layer.get('type', 'unknown')
                if layer_type in ['linear', 'conv']:
                    type_slices.append(i)

            # If we have a reasonable number of type-based slices, use them
            if type_slices and (not max_segments or len(type_slices) <= max_segments - 1):
                slice_points = type_slices
            else:
                # Otherwise fall back to balanced slicing
                if max_segments and max_segments > 1:
                    segment_size = total_layers // max_segments
                    for i in range(1, max_segments):
                        point = i * segment_size - 1
                        if 0 <= point < total_layers - 1:
                            slice_points.append(point)

        # Limit the number of slice points if max_segments is specified
        if max_segments and len(slice_points) >= max_segments:
            # Prioritize evenly distributed slice points
            if len(slice_points) > max_segments - 1:
                step = len(slice_points) // (max_segments - 1)
                indices = list(range(0, len(slice_points), step))[:max_segments - 1]
                slice_points = [slice_points[i] for i in indices]

        # Ensure slice points are unique and sorted
        return sorted(list(set(slice_points)))


# Example usage:
if __name__ == "__main__":
    # model_dir = "models/test_model_embedded"
    # model_path = os.path.join(model_dir, "test_model_embedded.pth")

    model_dir = "../models/doom"
    model_path = os.path.join(model_dir, "doom.pth")

    print(f"Analyzing model: {model_path}")
    model_utils = ModelUtils(model_path)
    result = model_utils.analyze_model(True)
