import os
from typing import Optional, Dict, List
import time
import json
import torch
import enum

from src.utils.model_utils import ModelUtils


class ModelSlicer:
    """
    A utility class for slicing and processing machine learning models into distinct
    segments based on layer information.

    This class provides methods to manage machine learning model slicing by
    segmenting layers and saving processed model parts. It integrates the
    functionalities of the `ModelUtils` class to load and interact with
    models, handle directory preparation for saving outputs, extract model segments,
    and save corresponding metadata.

    :ivar model_utils: An instance of the `ModelUtils` class used for model-related
          operations, such as loading and accessing layer information.
    :type model_utils: Optional
    """

    def __init__(self, model_utils:ModelUtils=None):
        """
        Initialize ModelSlicer

        Args:
            model_utils: Optional ModelUtils instance to use
        """
        self.model_utils = model_utils

    def _initialize_model_utils(self, model_path: str) -> Optional[Dict]:
        """
        Initializes the ModelUtils instance for managing the machine learning model.

        The method checks if the given 'model_utils' property of the object is set.
        If it is not set, it will create a new instance of the `ModelUtils` class
        using the provided 'model_path' argument. It will then attempt to load the
        model using the `load_model()` method of `ModelUtils`. If the loading of the
        model fails, the method returns a dictionary containing error details. If the
        loading succeeds, it returns `None`.

        :param model_path: The path to the model file.
        :type model_path: str
        :return: A dictionary with error details if the model loading fails, or None
            if successful.
        :rtype: Optional[Dict]
        """
        if self.model_utils is None:
            self.model_utils = ModelUtils(model_path)
            if not self.model_utils.load_model():
                return {'success': False, 'error': f"Failed to load model from {model_path}"}
        return None

    @staticmethod
    def _prepare_output_directory(model_path: str, output_dir: Optional[str] = None) -> str:
        """
        Prepares and returns the output directory where files will be stored. If the output directory
        is not specified, it creates a directory within the path of the model file with a suffix
        "_sliced". Ensures the directory exists before returning its path.

        :param model_path: Path to the model file that determines the default output directory
                           if output_dir is not provided.
        :type model_path: str
        :param output_dir: Path to the output directory where files should be stored. If None,
                           a default directory will be created in the same location as the
                           model file.
        :type output_dir: Optional[str]
        :return: Path to the prepared output directory.
        :rtype: str
        """
        if output_dir is None:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            output_dir = os.path.join(model_dir, f"{model_name}_sliced")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _get_model_segments(self, layers: List[Dict], slice_points: List[int]) -> List[Dict]:
        """
        Extracts segments from the given model layers based on specified slice points. Each
        segment represents a contiguous portion of the layers marked by the slice points.
        The function ensures the segments are non-overlapping and skips invalid slice
        points. Each segment is categorized by its type using a helper method.

        :param layers: List of layer dictionaries representing the model layers.
        :param slice_points: List of integers indicating the slice points where segments
            should start or end.
        :return: A list of dictionaries, each containing details about a segment, such
            as its index, start index, end index, type, and layers included.
        """
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
        """
        Processes a specified model segment and saves its state dictionary to a specified
        directory. The segment information, including metadata and additional features,
        is also created and returned.

        :param segment: A dictionary containing the segment to be processed, which includes
              details such as `type`, `index`, and `layers`.
        :param output_dir: A string specifying the directory where the processed segment's
              state dictionary will be saved.
        :return: A dictionary containing processed information about the segment, including
              details such as index, type, filename, path, layer count, total parameters,
              and layers.
        """
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

    @staticmethod
    def _add_feature_information(segment_info: Dict, segment: Dict):
        """
        Adds feature information to the given ``segment_info`` dictionary based on the provided ``segment`` details.
        The function processes the type and layer information of the segment and updates ``segment_info`` with
        input features, output features, and activation functions, if applicable.

        :param segment_info: A dictionary to store the updated feature information, which includes
            input features, output features, and activation function, based on the analyzed segment data.
        :param segment: A dictionary containing the type and layer details of the segment. It includes
            information about the segment's type (e.g., 'linear', 'conv') and layers, from which features
            like ``in_features``, ``out_features``, ``in_channels``, ``out_channels``, and
            activation function may be extracted.
        :return: None
        """
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

    @staticmethod
    def _create_and_save_metadata(model_path: str, output_dir: str, analysis: Dict,
                                  strategy: str, saved_segments: List[Dict], slice_points: List[int]) -> str:
        """
        Generates metadata for a trained model, including its attributes, segmentation
        strategies, and analysis details, and then saves it to a JSON file in the
        specified output directory.

        The metadata includes details such as the original model path, model type,
        total number of parameters, slicing strategy used, saved segments, and slice
        points. This function writes the metadata in JSON format for later retrieval
        and returns the path to the saved file.

        :param model_path: Path to the original trained model on the filesystem.
        :type model_path: str
        :param output_dir: Directory where the metadata JSON file will be saved.
        :type output_dir: str
        :param analysis: Dictionary containing analysis details of the model with
            keys like 'model_type' and 'total_parameters'.
        :type analysis: Dict
        :param strategy: Slicing strategy used for the model's segmentation.
        :type strategy: str
        :param saved_segments: List of dictionaries containing information about the
            saved segments of the model.
        :type saved_segments: List[Dict]
        :param slice_points: Sequence of integers representing model slicing points.
        :type slice_points: List[int]
        :return: Path to the created metadata JSON file.
        :rtype: str
        """
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
        Slices a model into smaller segments based on the provided slicing strategy and
        maximum number of segments. It processes and saves the segments to the output
        directory and generates metadata associated with the slicing operation.

        :param model_path: Path to the model file that needs to be sliced.
        :type model_path: str
        :param output_dir: Directory where sliced model segments and metadata will be stored.
                           If not specified, a default output directory is used.
        :type output_dir: Optional[str]
        :param strategy: Strategy for slicing the model, e.g., "layer_type". Determines how
                         the model layers are divided into segments.
        :type strategy: str
        :param max_segments: Maximum number of segments to create during the slicing process.
                             If not provided, slicing defaults to determining the number of
                             segments based on the strategy.
        :type max_segments: Optional[int]
        :return: Dictionary containing the outcome of the slicing operation:
                 - `success` (bool): Indicates whether the slicing was successful.
                 - `output_dir` (str): Path to the directory where segments and metadata are saved.
                 - `segments` (list): List of saved model segments with their respective
                                      processing outputs.
                 - `metadata_path` (str): Path to the metadata file describing the slicing operation.
        :rtype: Dict
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

    @staticmethod
    def _determine_segment_type(layers: List[Dict]) -> str:
        """
        Determines the segment type of given list of layers based on their frequencies
        and maps them to standardized output names.

        The function calculates the frequency of each layer type, removes the 'unknown' type if at
        least one known type is present, and determines the most common type. It then maps the internal
        type names to corresponding standardized segment types such as 'fc', 'conv', 'norm', or others.
        If no valid type is found, it defaults to 'misc'.

        :param layers: A list of dictionaries where each dictionary represents a layer with its
            attributes and an optional "type" key indicating the type of the layer.
        :type layers: List[Dict]
        :return: A string representing the most common segment type mapped to a standardized name.
            If no recognized type is determined, 'misc' is returned.
        :rtype: str
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

    @staticmethod
    def _extract_segment_state_dict(full_state_dict: Dict, layers: List[Dict]) -> Dict:
        """
        Extracts a subset of a state dictionary based on specified layers. This function iterates through
        the given ``full_state_dict`` and filters out keys associated with the layers provided in the
        ``layers`` parameter. The resulting dictionary will include only the keys that correspond to
        these layers.

        :param full_state_dict: The full state dictionary containing keys and associated layer values.
        :param layers: List of dictionaries where each dictionary contains information about a layer. Each
            layer must have a ``name`` key, representing the layer's identifier.
        :return: A new dictionary containing the keys and values from the ``full_state_dict`` that match
            the specified layers.
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
        Gathers activation information for given layers of a model using multiple strategies including extraction
        from an existing model object, searching configuration files, and inferring from layer structure.

        :param model_path: Path to the model file, typically required for locating associated configuration files.
        :type model_path: str
        :param layers: A list of dictionaries representing the layers of the model. Each dictionary may contain
                       keys such as 'name', and additional entries will be updated based on inferred activation
                       functions.
        :type layers: List[Dict]
        :return: A dictionary mapping layer names to their respective activation functions. If no activation
                 information could be determined, it will return an empty dictionary.
        :rtype: Dict[str, str]
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

    @staticmethod
    def _extract_activation_functions(model) -> Dict[str, str]:
        """
        Extracts and identifies activation functions present within a given PyTorch model.

        The function iterates through all the named submodules of the model and checks each module
        against a set of known activation function classes provided by PyTorch's `torch.nn` module.
        It collects and records these activation functions along with their corresponding hierarchical
        names within the model.

        In the case of sequential modules, the function further explores its children and checks for
        activation functions nested within them. This allows identification of activation layers used
        within composite structures like `torch.nn.Sequential`. If the input model is `None`, the function
        returns an empty dictionary.

        :param model: The PyTorch model, potentially containing activation functions, as an instance of `torch.nn.Module`.
        :return: A dictionary where keys represent the hierarchical names of the modules containing
                 activation functions, and values represent the corresponding activation type as a
                 string.
        :rtype: Dict[str, str]
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

    @staticmethod
    def _infer_activations_from_layers(layers: List[Dict]) -> Dict[str, str]:
        """
        Infers and maps activation functions based on the naming patterns and types of layers
        provided. The function identifies common activation functions by checking specific
        keywords in the layer names or guessing them based on layer types and positions within
        the network. Assumptions are made for commonly used defaults in certain types of layers.

        :param layers: List of dictionaries representing layers in a model. Each dictionary
            must include at least the 'name' field. Optionally, the 'type' and 'out_features'
            fields can also be provided to improve accuracy of activation inference.
            Example of a layer dictionary:
                {
                    "name": "layer1_relu",
                    "type": "conv",
                    "out_features": 128
                }

        :return: A dictionary mapping layer names to their inferred activation functions. The
            keys of the dictionary are layer names, and the values are corresponding activation
            function names such as "ReLU", "Sigmoid", or "Softmax".
        :rtype: Dict[str, str]
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
