import enum
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch


class ModelType(enum.Enum):
    SEQUENTIAL = "sequential"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    FCNN = "fcnn"  # Add this new type for fully connected networks
    HYBRID = "hybrid"  # Add this for mixed architecture models (e.g., CNN + FCNN)
    UNKNOWN = "unknown"



class ModelUtils:
    """
    ModelUtils class is a utility for managing and analyzing neural network model files.

    The class provides functionalities for loading model state dictionaries, analyzing
    their architecture, and identifying key characteristics such as the type of network,
    layer structures, and parameter details. It primarily operates on PyTorch model files.
    """

    def __init__(self, model_path=None):
        """
        Class representing a model with support for managing its path, state, and type.

        The class allows defining an optional model path during initialization. It
        maintains additional attributes to store the state dictionary and type of
        the model.

        """
        self.model_path = model_path
        self.state_dict = None
        self.model_type = None

    def load_model(self) -> bool:
        """
        Loads a machine learning model from the specified file path using PyTorch,
        validates its existence, and manages different expected formats of state
        dictionaries. This method is essential for initializing and preparing the
        model for inference or further training.

        :raises ValueError: If the specified model path is invalid or does not exist.
        :raises RuntimeError: If the model cannot be loaded due to other
            unexpected issues during the loading process.

        :return: True if the model is successfully loaded and processed,
            otherwise False.
        :rtype: bool
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
        Analyzes the state and structure of a loaded model. This method performs
        several critical tasks including detection of the model type, extraction
        of layer structure, and identification of significant model
        characteristics such as the total number of parameters and groupings
        of layers. Additionally, the results of the analysis can be optionally
        printed to the console or written to an output file.

        :param verbose: Whether detailed analysis results should be printed
            to the console. Defaults to True.
        :type verbose: bool
        :param output_file: Optional path to a file where the analysis results
            should be saved. If None, the results are not written to a file.
        :type output_file: str or None
        :return: A dictionary containing the results of the model analysis,
            including information on the model's type, its layer structure,
            key characteristics such as total parameters, and identified
            layer groups. If the model fails to load, an error message is
            returned instead.
        :rtype: Dict
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
        Determines whether a set of keys belongs to convolutional neural network (CNN)
        layers. The method identifies CNN layers based on key naming patterns or by
        detecting specific properties of their weight tensors. It uses common CNN
        naming conventions and checks for 4D weight tensor shapes, a typical
        characteristic of convolutional layers.

        :param keys: A list of keys representing the layer names or identifiers to
            be checked.
        :type keys: list[str]
        :return: A boolean value indicating whether any of the provided keys belongs
            to a CNN layer.
        :rtype: bool
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

        The extracted layer information is returned as a sorted list of dictionaries, where
        each dictionary represents a layer with its details. Sorting aims to maintain a logical
        order in the layer sequence, which is useful for model inspection and debugging.

        :return: A list of dictionaries, where each dictionary contains organized information
                 about a model's layer, including its name, type, parameters, and sizes.
        :rtype: List[Dict]
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

        :param layers: A list of dictionaries where each dictionary represents a layer. Each layer dictionary must contain
                       a 'name' key, which is a string used to determine its sorting order.
        :type layers: List[Dict]
        :return: A list of dictionaries representing the layers sorted first by numerical prefix, then by any suffix.
        :rtype: List[Dict]
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

        The method computes the sum of elements for tensors stored in the
        state dictionary. If the state dictionary is not set or if it is not a
        valid dictionary, the method returns 0.

        :param self: The object instance containing the state dictionary.
        :return: The total count of parameters in the state dictionary, or 0
            if the state dictionary is None or invalid.
        :rtype: int
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

        :param layers: A list of dictionaries where each dictionary represents a layer in the network.
            Each dictionary is expected to have at least the following keys:
                - 'type': A string indicating the type of the layer.
                - 'name': A string representing the name of the layer.
        :return: A dictionary with the following structure:
            - 'groups': A dictionary categorizing layer names by their types. The supported types are:
                'conv', 'linear', 'norm', 'embedding', and 'other'. Layers not matching predefined
                types default to the 'other' category.
            - 'potential_slicing_points': A list of dictionaries that mark transition points between
                layers of different types. Each dictionary contains:
                    - 'after': The type of the preceding layer.
                    - 'before': The type of the succeeding layer.
                    - 'layer_name': The name of the layer where the transition occurs.
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
        Prints a detailed analysis of a model's architecture, including parameter counts, layer type
        distribution, and individual layer details. The analysis can be either printed to the console
        or written to a specified output file.

        :param analysis: A dictionary containing details about the model analysis. The dictionary must
            include keys like 'model_type', 'total_parameters', 'layers', and 'layer_groups', among others.
            The 'model_type' key should contain an enum value representing the type of model, and
            'layer_groups' must contain layer type distribution along with potential slicing points.
        :param output_file: The path to an optional file where the analysis will be written. If not
            provided, the analysis will be printed to the console.

        :return: None.
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
        Retrieves a specific layer from the list of extracted layers using its index.

        Detailed Description:
        This method accesses the model's layers by extracting them if they are available
        within the state dictionary. It checks if the provided index is within the valid
        range of the available layers. If the index is valid, it returns the corresponding
        layer as a dictionary. If the index is invalid or the model fails to load, it
        returns an empty dictionary.

        :param index: The position in the layers' list to retrieve.
        :type index: int
        :return: A dictionary representing the layer at the specified index, or an
                 empty dictionary if the index is out of range or the model cannot be
                 loaded.
        :rtype: Dict
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
        Retrieve a specific layer by name from the model's layers.

        The method searches for a layer with the specified name within the extracted
        layers of the model. If the state dictionary is not already loaded, it attempts
        to load the model first. If loading fails or no layer matches the given name, an
        empty dictionary is returned.

        :param name: Name of the layer to search for in the model's layers.
        :type name: str
        :return: A dictionary representation of the layer with the specified name, or an
            empty dictionary if not found.
        :rtype: Dict
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
        Verifies the integrity of the provided slice points against the model
        layers and calculates the resulting slices, their statistics, and
        validity.

        The slice points are validated as indices that properly divide the
        model's layers into logical segments. For each segment, the
        number of layers, total parameter count, and the count of
        different layer types are computed.

        :param slice_points: A list of integers indicating the indices to divide
            the layers into segments.
        :return: A dictionary containing verification and segmentation results:
            - 'valid': A boolean indicating if all slice points are valid.
            - 'total_layers': Total number of layers in the model.
            - 'valid_slice_points': List of valid slice points.
            - 'invalid_slice_points': List of invalid slice points.
            - 'segment_count': Number of resulting segments based on slice
              points.
            - 'segment_stats': List of dictionaries containing detailed
              statistics for each segment, including:
                - The segment number (1-indexed).
                - The start and end layer indices for the segment.
                - The number of layers in the segment.
                - The total count of parameters in the segment from all
                  included layers.
                - A dictionary describing the count of each layer type in
                  the segment.

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
        Estimates the memory footprint sizes of slices defined by the given slice points.

        This method analyzes the points where a large model is split for distributed
        processing or storage, calculates the memory footprint for each segment in megabytes,
        and returns detailed statistics along with total segment count. The slices are verified
        before estimation to ensure integrity of the specified segment points.

        :param slice_points: A list of integers representing the points at which the model
            is sliced.
        :type slice_points: List[int]
        :return: A dictionary containing the total number of segments, memory footprint
            estimates for each segment in MB, and error details if either the model
            fails to load or the segments are found to be invalid during verification.
        :rtype: Dict
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
        Determines slice points for dividing a set of neural network layers into segments
        based on the specified slicing strategy. This can be used to partition the layers
        within a model for various purposes such as distributed training, optimization,
        or analysis. Different strategies offer flexibility in how layers are grouped,
        including single-layer segmentation, type-based transitions, balanced groups,
        or hybrid approaches.

        :param strategy: The slicing strategy to use for determining slice points.
            Supported strategies are:

            - `"single_layer"`: Slices occur after every individual layer except the last one.
            - `"layer_type"`: Slices occur after fully connected ('linear') and convolutional ('conv') layers.
            - `"balanced"`: Divides layers into evenly-sized segments based on `max_segments`.
            - `"transitions"`: Places slices at transitions between different layer types.
            - Default (hybrid strategy): Balances slices across fully connected/convolutional
              layers or evenly distributes slices if `max_segments` applies.

        :param max_segments: The maximum number of segments to partition the layers into.
            When specified, ensures the slice points do not exceed the allowed number of
            segments. If omitted, strategies determine the slicing freely.

        :return: A list of sorted and unique integer indices representing the slice points
            in the set of layers. Each index corresponds to the position after which
            a slice occurs. Returns an empty list if no valid slicing points are found
            or the layer set is empty.
        :rtype: List[int]
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
    def check_model_structure(model):
        """
        Check the structure of a loaded model to determine how it should be used for inference.

        Parameters:
            model: The loaded model object

        Returns:
            dict: A dictionary containing information about the model structure:
                - 'type': 'callable', 'state_dict', 'dict_with_model', or 'unknown'
                - 'callable_model': The callable model if found, otherwise None
                - 'component_name': Name of the callable component if found in a dictionary
        """
        result = {
            'type': 'unknown',
            'callable_model': None,
            'component_name': None
        }

        # Check if it's already a callable model
        if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
            result['type'] = 'callable'
            result['callable_model'] = model
            return result

        # Check if it's a state dict (dictionary of tensors)
        if isinstance(model, dict):
            # Check if it's a state dict with parameters
            if all(isinstance(v, torch.Tensor) for v in model.values()):
                result['type'] = 'state_dict'
                return result

            # Check if it's a dictionary containing a model or state_dict
            if 'model' in model and hasattr(model['model'], 'forward'):
                result['type'] = 'dict_with_model'
                result['callable_model'] = model['model']
                result['component_name'] = 'model'
                return result

            if 'state_dict' in model:
                result['type'] = 'state_dict'
                return result

            # Look for any callable model in the dictionary
            for key, value in model.items():
                if hasattr(value, 'forward') and callable(getattr(value, 'forward')):
                    result['type'] = 'dict_with_model'
                    result['callable_model'] = value
                    result['component_name'] = key
                    return result

        return result

    def to_onnx(self, example_input, onnx_file_path=None, input_names=None, output_names=None, opset_version=11):
        """Exports the model to an ONNX formatted file."""

        model = self.load_model()
        if model is None:
            raise ValueError("Failed to load model—ONNX export aborted.")

        model.eval()

        input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        if onnx_file_path is None:
            base_path = os.path.dirname(self.model_path) if self.model_path else "."
            onnx_file_path = os.path.join(base_path, "onnx", "model.onnx")
            os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

        torch.onnx.export(
            model,
            example_input,
            onnx_file_path,
            export_params=True,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"Model successfully exported to ONNX at '{onnx_file_path}'")

    @staticmethod
    def get_input_shape(input_file_path: str) -> tuple:
        """
        Returns the input shape based on a provided input data file (JSON or other formats).

        Args:
            input_file_path (str): Path to the file containing input data.
                                   Currently supports JSON files.

        Returns:
            tuple: The explicit shape of the input data, excluding batch size.

        Raises:
            FileNotFoundError: If the provided input file path doesn't exist.
            ValueError: if file content is invalid or unsupported.
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"The specified input file does not exist: {input_file_path}")

        file_extension = os.path.splitext(input_file_path)[1].lower()

        if file_extension == '.json':
            with open(input_file_path, 'r') as file:
                data = json.load(file)

            if 'input_data' not in data:
                raise ValueError("JSON file must contain an 'input_data' key.")

            input_data = np.array(data['input_data'])

            # check if input has batch dimension explicitly (common practice)
            if input_data.ndim == 1:
                input_shape = input_data.shape
            else:
                input_shape = input_data.shape[1:]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Currently, only JSON is supported.")

        print(f"Explicitly determined input shape from file: {input_shape}")
        return input_shape

    @staticmethod
    def preprocess_input(input_path):
        """
        Preprocess input data from JSON.

        Parameters:
            input_path (str): Path to input JSON file

        Returns:
            torch.Tensor: Preprocessed input tensor
        """
        try:

            # Load JSON data
            with open(input_path, 'r') as f:
                input_data = json.load(f)

            print(f"Loaded input data: {type(input_data)}")

            # Extract input data
            if isinstance(input_data, dict):
                if 'input_data' in input_data:
                    print("Found 'input_data' key in input JSON")
                    input_data = input_data['input_data']
                elif 'input' in input_data:
                    print("Found 'input' key in input JSON")
                    input_data = input_data['input']

            # Convert to tensor
            if isinstance(input_data, list):
                if isinstance(input_data[0], list):
                    # 2D input
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                else:
                    # 1D input
                    input_tensor = torch.tensor([input_data], dtype=torch.float32)
            else:
                raise ValueError("Expected input data to be a list or nested list")

            print(f"Input tensor shape: {input_tensor.shape}")
            return input_tensor

        except Exception as e:
            print(f"Error preprocessing input: {e}")
            return None


# Example usage:
if __name__ == "__main__":
    # model_dir = "models/test_model_embedded"
    # model_path = os.path.join(model_dir, "test_model_embedded.pth")

    model_dir = "../models/test_cnn_model_with_biases"
    model_path = os.path.join(model_dir, "test_cnn_model.pth")

    print(f"Analyzing model: {model_path}")
    model_utils = ModelUtils(model_path)
    result = model_utils.analyze_model(True)
