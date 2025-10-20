import enum
import json
import os
import re
from pathlib import Path
from typing import Dict, List
import torch


class ModelType(enum.Enum):
    SEQUENTIAL = "sequential"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    FCNN = "fcnn"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"



class ModelUtils:
    def __init__(self, model_path=None, verbose=False, state_dict=None, model_type=None):
        self.model_path = model_path
        self.state_dict = None
        self.model_type = None

    def load_model(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"Error: Invalid model path {self.model_path}")
            return False

        try:
            # Load with torch.load and appropriate map_location
            self.state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))

            # Handle different state dict formats
            if isinstance(self.state_dict, dict):
                if 'state_dict' in self.state_dict:
                    self.state_dict = self.state_dict['state_dict']
                elif 'model_state_dict' in self.state_dict:
                    self.state_dict = self.state_dict['model_state_dict']
                elif 'net' in self.state_dict:
                    self.state_dict = self.state_dict['net']

                # Debug - print the keys
                # if isinstance(self.state_dict, dict):
                #     print(f"State dict keys: {list(self.state_dict.keys())[:5]} (showing first 5)")
                # else:
                #     print(f"State dict is not a dictionary but {type(self.state_dict)}")

            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def analyze_model(self, verbose=True, output_file=None) -> Dict:
        """
        Analyzes the state and structure of a loaded model. This method performs
        several critical tasks including detection of the model type, extraction
        of layer structure, and identification of significant model
        characteristics such as the total number of parameters and groupings
        of layers. Additionally, the results of the analysis can be optionally
        printed to the console or written to an output file.
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
        Determines the model type based on the structure and keys of the `state_dict`.

        This method inspects the `state_dict` attribute of the containing object
        and identifies the type of model based on the presence of specific patterns in the keys.
        The function assumes the model to be one among `TRANSFORMER`, `CNN`, `FCNN`,
        `HYBRID`, `SEQUENTIAL`, or `UNKNOWN`. Detection prioritizes transformers,
        followed by hybrids, CNNs, FCNNs, or sequential models.

        Model type definitions:
          - `TRANSFORMER`: Indicates the presence of keys suggesting transformer-like architecture.
          - `CNN`: Indicates convolutional neural network patterns in the structure.
          - `FCNN`: Suggests a fully connected network (linear).
          - `HYBRID`: Combination of transformers or CNN with other structures.
          - `SEQUENTIAL`: Models with a clearly sequential pattern in their layers.
          - `UNKNOWN`: Could not classify the model reliably into any of the above categories.

        Warning messages are printed if `state_dict` is neither a dictionary nor `None`.
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
            # Relax the hybrid condition for CNN+Linear, since many CNNs have linear layers
            return ModelType.CNN

        if has_linear:
            return ModelType.FCNN

        if is_sequential:
            return ModelType.SEQUENTIAL

        return ModelType.UNKNOWN

    def _has_cnn_layers(self, keys) -> bool:
        """
        Determines whether a set of keys belongs to convolutional neural network (CNN)
        layers. The method identifies CNN layers based on key naming patterns or by
        detecting specific properties of their weight tensors. It uses common CNN
        naming conventions and checks for 4D weight tensor shapes, a typical
        characteristic of convolutional layers.
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
        Extracts and organizes layer information from the state dictionary of a model. This
        method processes a model's state dictionary to identify and group parameters by their
        corresponding layers. Each layer is characterized by its name, type (e.g., convolutional,
        fully connected, normalization, etc.), shape, and associated parameters, such as weights
        and biases. Additionally, for convolutional and fully connected layers, detailed
        metadata such as kernel size, in/out channels, and features are extracted.
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
                    layer_params[layer_name]['stride'] = (1, 1)
                    layer_params[layer_name]['padding'] = (0, 0)
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

    @staticmethod
    def _sort_layers(layers: List[Dict]) -> List[Dict]:
        """
        Sorts a list of layer dictionaries based on numerical prefixes extracted from their names. Layer names are grouped
        by common prefixes, sorted by number, followed by suffix within each group. Groups are then flattened in the order
        of their prefixes.
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
        Counts the total number of parameters in the state dictionary.
        """
        if self.state_dict is None or not isinstance(self.state_dict, dict):
            return 0

        return sum(tensor.numel() for tensor in self.state_dict.values()
                   if torch.is_tensor(tensor))

    @staticmethod
    def _identify_layer_groups(layers: List[Dict]) -> Dict:
        """
        Groups and classifies layers into predefined categories based on their type, and identifies
        potential transition points (or slicing points) between different layer types. This helps in
        structuring neural network models for better organization and analysis.
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

    @staticmethod
    def _print_analysis(analysis: Dict, output_file=None):
        """
        Prints a detailed analysis of a model's architecture, including parameter counts, layer type
        distribution, and individual layer details. The analysis can be either printed to the console
        or written to a specified output file.
        """
        output = ["=" * 60, "MODEL ARCHITECTURE ANALYSIS", "=" * 60,
                  f"\nModel Type: {analysis['model_type'].value}"]  # Collect output lines

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


    def get_slice_points(self, strategy: str = "layer_type", max_segments: int = None) -> List[int]:
        """
        Determines slice points for dividing a set of neural network layers into segments
        based on the specified slicing strategy. This can be used to partition the layers
        within a model for various purposes such as distributed training, optimization,
        or analysis. Different strategies offer flexibility in how layers are grouped,
        including single-layer segmentation, type-based transitions, balanced groups,
        or hybrid approaches.
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

    @staticmethod
    def save_tensor_to_json(tensor_data, filename="input_data_reshaped.json", model_directory: str = None):
        """Saves the given tensor data to a JSON file in the model directory."""
        output_path = os.path.join(model_directory, "generated_inputs", filename)

        try:
            if isinstance(tensor_data, torch.Tensor):
                data_list = tensor_data.detach().cpu().tolist()
            elif hasattr(tensor_data, "tolist"):
                data_list = tensor_data.tolist()
            else:
                data_list = list(tensor_data)

            json_output = {"input_data": data_list}

            with open(output_path, 'w') as f:
                json.dump(json_output, f, indent=4)

        except Exception as e:
            print(f"Error: Could not write tensor data to {output_path}. Reason: {e}")


    @staticmethod
    def load_metadata(folder_path):
        """Load the model metadata from the metadata.json file."""
        metadata_path = Path(folder_path) / "metadata.json"
        if not metadata_path.exists():
            print(f"Required metadata.json file not found at: {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata


# Example usage:
if __name__ == "__main__":
    # model_dir = "models/test_model_embedded"
    # model_path = os.path.join(model_dir, "test_model_embedded.pth")

    model_dir = "../models/net"
    model_path = os.path.join(model_dir, "model.pth")

    print(f"Analyzing model: {model_path}")
    model_utils = ModelUtils(model_path)
    result = model_utils.analyze_model(True)
