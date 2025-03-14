import base64
import io
import json
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader


class ModelVisualizer:
    """
    Provides functionality to visualize a neural network model based on its architecture and the metadata provided.

    The ModelVisualizer class enables visualization of different types of neural networks,
    such as convolutional neural networks (CNNs), fully connected neural networks (FCNNs),
    and hybrid models combining CNN layers and FCNN layers. The visualization is generated
    using matplotlib and includes support for selective rendering of specific layers or segments
    of the network. Metadata describing the model's structure is loaded from a JSON file, and the
    visualization can optionally be saved to a file or returned as a matplotlib.figure object.

    :ivar metadata: The parsed metadata from the given metadata.json file.
    :type metadata: dict
    :ivar model_type: The type of the model (e.g., 'CNN', 'FCNN', 'HYBRID').
    :type model_type: str
    :ivar segments: List of segments of the model loaded from the metadata file.
    :type segments: list
    :ivar fig: The matplotlib figure object for the current visualization.
    :type fig: matplotlib.figure.Figure or None
    :ivar ax: The matplotlib axes object for the current visualization.
    :type ax: matplotlib.axes._axes.Axes or None
    """

    def __init__(self, metadata_path):
        """
        Initialize the visualizer with a path to metadata.json.

        Args:
            metadata_path (str): Path to the metadata.json file
        """
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.model_type = self.metadata.get("model_type", "unknown")
        self.segments = self.metadata.get("segments", [])
        self.fig = None
        self.ax = None

    def visualize(self, output_path=None, start_segment=None, end_segment=None):
        """
        Visualizes the neural network model based on its type and selected segment range

        Args:
            output_path: Optional path to save the visualization image to
            start_segment: First segment index to visualize (0-indexed), None for start from beginning
            end_segment: Last segment index to visualize (inclusive), None for all remaining segments

        Note:
            Segment indices correspond to the "index" field in the metadata file.
            For example, segment index 2 will visualize the segment with "index": 2 in the metadata.
            The output layer is NOT automatically added - only the specified segments are shown.

        Returns:
            Path to the saved visualization file or the matplotlib figure
        """
        if not self.metadata:
            print("No metadata available. Cannot visualize.")
            return None

        # Get model type from metadata and clean it
        model_type_raw = self.metadata.get('model_type', '')
        if '.' in model_type_raw:
            # Extract just the type name without the enum prefix
            self.model_type = model_type_raw.split('.')[-1]
        else:
            self.model_type = model_type_raw

        # Extract all layers from the model
        all_layers = self._extract_all_layers()

        if not all_layers:
            print("No layers found in model metadata.")
            return None

        # Filter layers by segment index
        if start_segment is not None or end_segment is not None:
            # Default values if not specified
            start_segment_idx = start_segment if start_segment is not None else 0
            # Get the last segment index from the metadata
            max_segment_idx = max([layer.get('segment_idx', 0) for layer in all_layers]) if all_layers else 0
            end_segment_idx = end_segment if end_segment is not None else max_segment_idx

            # Validate indices
            start_segment_idx = max(0, min(start_segment_idx, max_segment_idx))
            end_segment_idx = max(start_segment_idx, min(end_segment_idx, max_segment_idx))

            # Filter layers by segment_idx
            selected_layers = [layer for layer in all_layers
                               if start_segment_idx <= layer.get('segment_idx', 0) <= end_segment_idx]
        else:
            # If no filtering, use all layers
            selected_layers = all_layers

        # Check the model type and visualize accordingly
        if 'HYBRID' == self.model_type:
            print("This is Hybrid model type.")
            self.fig = self._visualize_hybrid_layers(selected_layers)
        elif 'CNN' == self.model_type:
            print("This is CNN model type.")
            self.fig = self._visualize_cnn_layers(selected_layers)
        elif 'FCNN' == self.model_type:
            print("This is FCNN model type.")
            self.fig = self._visualize_fcnn_layers(selected_layers)
        else:
            print(f"Unsupported model type: {self.model_type}")
            return None

        # Save the visualization if output_path is provided
        if output_path and self.fig:
            try:
                self.fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {output_path}")
            except Exception as e:
                print(f"Error saving visualization: {e}")

        return output_path if output_path else self.fig

    def _visualize_hybrid_layers(self, selected_layers: list[dict] | None = None):
        """
        Visualizes the hybrid layer structure combining CNN and FCNN components.

        This method produces a detailed 2D visual representation of the layers in a
        neural network, including both CNN and FCNN components. CNN layers are
        visualized using fixed dimensions while FCNN layers follow a dynamic strategy
        based on the number of neurons. Connections between layers and transitions
        (e.g., Flatten layer, CNN to FCNN transition) are also visualized. Options
        are provided to customize the visualization based on specific layers or
        sections to highlight.

        :param selected_layers: A list of dictionaries containing specific layers
            to visualize or None to visualize the entire network.
        :type selected_layers: list[dict] | None
        :return: A matplotlib figure representing the hybrid layer visualization.
        :rtype: matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(20, 12))

        # Visual settings
        layer_spacing = 4.5  # Spacing between layers (matching FCNN visualization)
        neuron_radius = 0.25  # Size of neuron nodes
        node_color = 'lightblue'  # Color for nodes

        # Set fixed sizes for CNN layers
        cnn_height = 3.0  # Fixed height for CNN layers
        cnn_center_y = 0.5  # Center position

        # Track current position and previous layer info for connections
        x_pos = 0.5
        prev_layer_info = None
        cnn_end_x = 0

        # Get all segments in order
        all_segments = self.metadata.get('segments', [])

        # Separate CNN and FCNN segments
        cnn_segments = [seg for seg in all_segments if seg.get('type') == 'conv']
        fc_segments = [seg for seg in all_segments if seg.get('type') == 'fc']

        # Process all CNN segments with fixed height
        cnn_layer_counter = 0
        for seg_idx, segment in enumerate(cnn_segments):
            if 'layers' in segment:
                for layer_idx, layer in enumerate(segment['layers']):
                    x_pos, layer_info = self._draw_conv_layer_2d(
                        ax, layer, x_pos, seg_idx, len(cnn_segments),
                        plt.cm.viridis, cnn_center_y, cnn_height
                    )

                    # Add layer name at the bottom
                    layer_name = layer.get('name', f"CNN{cnn_layer_counter}")
                    y_pos_label = cnn_center_y - (cnn_height / 2) - 0.3  # Position below the layer
                    ax.text(x_pos - 0.25, y_pos_label, layer_name,
                            ha='center', va='top', fontsize=10, zorder=4)

                    if prev_layer_info:
                        self._draw_arrow_between_layers(ax, prev_layer_info, layer_info)

                    prev_layer_info = layer_info
                    cnn_layer_counter += 1
                    x_pos += 0.5  # Add some spacing

            # Add pooling if this isn't the last CNN segment
            if seg_idx < len(cnn_segments) - 1:
                x_pos, layer_info = self._draw_pooling_layer_2d(
                    ax, x_pos, plt.cm.viridis, cnn_center_y, cnn_height
                )

                # Label the pooling layer
                y_pos_label = cnn_center_y - (cnn_height / 2) - 0.3
                ax.text(x_pos - 0.25, y_pos_label, "MaxPool",
                        ha='center', va='top', fontsize=10, zorder=4)

                if prev_layer_info:
                    self._draw_arrow_between_layers(ax, prev_layer_info, layer_info)

                prev_layer_info = layer_info
                x_pos += 0.5  # Add some spacing

        # Store CNN ending position to calculate proper transition
        cnn_end_x = x_pos

        # Add flatten transition if we have both CNN and FC segments
        if cnn_segments and fc_segments:
            # Add extra spacing before flatten layer
            x_pos += 1.0

            x_pos, layer_info = self._draw_flatten_transition_2d(
                ax, x_pos, plt.cm.viridis, cnn_center_y, cnn_height
            )

            # Label the flatten layer
            y_pos_label = cnn_center_y - (cnn_height / 2) - 0.3
            ax.text(x_pos - 0.25, y_pos_label, "Flatten",
                    ha='center', va='top', fontsize=10, zorder=4)

            if prev_layer_info:
                self._draw_arrow_between_layers(ax, prev_layer_info, layer_info)

            prev_layer_info = layer_info

            # Add more spacing after flatten layer to separate CNN and FC sections
            x_pos += 1.5

        # Now handle FC segments using the FCNN style visualization
        if fc_segments:
            # Extract all FC layers
            fc_layers = []
            for segment in fc_segments:
                if 'layers' in segment:
                    for layer in segment['layers']:
                        fc_layers.append(layer)

            if not fc_layers:
                # No FC layers to visualize
                self._format_plot_2d(ax, x_pos)
                return fig

            # Calculate neuron counts for each layer
            neuron_counts = []
            weights = []

            # First collect input features for each layer
            for layer in fc_layers:
                if 'in_features' in layer:
                    neuron_counts.append(layer['in_features'])
                else:
                    segment = next((seg for seg in fc_segments if seg.get('layers')
                                    and layer in seg['layers']), None)
                    if segment and 'in_features' in segment:
                        neuron_counts.append(segment['in_features'])
                    else:
                        print(f"Warning: Missing in_features for layer: {layer}")
                        neuron_counts.append(1)

                # Extract weights if available
                if 'parameters' in layer and 'weight' in layer.get('parameters', {}):
                    weights.append(layer['parameters']['weight'])
                else:
                    weights.append(None)

            # Add output features of the last layer
            if fc_layers:
                last_layer = fc_layers[-1]
                if 'out_features' in last_layer:
                    neuron_counts.append(last_layer['out_features'])
                else:
                    segment = next((seg for seg in fc_segments if seg.get('layers')
                                    and last_layer in seg['layers']), None)
                    if segment and 'out_features' in segment:
                        neuron_counts.append(segment['out_features'])
                    else:
                        print(f"Warning: Missing out_features for last layer: {last_layer}")
                        neuron_counts.append(1)

            # Dynamic visualization strategy for FCNN
            max_neurons = max(neuron_counts) if neuron_counts else 1

            # Adjust the threshold based on maximum neurons
            neuron_threshold = 20  # Base threshold for small networks

            # Scale threshold based on max_neurons but with limits
            if max_neurons > 100:
                neuron_threshold = 30
            elif max_neurons > 500:
                neuron_threshold = 50

            # Sampling strategy for different size ranges
            def get_sampling_strategy(count):
                if count <= neuron_threshold:
                    return {'show_all': True, 'sample': count}  # Show all
                elif count <= 100:
                    return {'show_all': False, 'sample': min(count, neuron_threshold)}  # Sample some
                elif count <= 1000:
                    return {'show_all': False, 'sample': min(count, neuron_threshold)}  # Sample fewer
                else:
                    return {'show_all': False, 'sample': min(count, neuron_threshold)}  # Sample even fewer

            # Add a visual separator between CNN and FCNN parts
            if cnn_segments:
                # Draw a vertical dashed line between CNN and FCNN
                separator_x = x_pos - 1.0
                ax.axvline(x=separator_x, ymin=0.1, ymax=0.9, color='gray',
                           linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

                # Add a transition label
                ax.text(separator_x, 0.05, "CNN → FCNN Transition",
                        ha='center', va='bottom', fontsize=10, color='gray',
                        bbox=dict(boxstyle="round,pad=0.3", fc="whitesmoke", ec="gray", alpha=0.8),
                        zorder=4)

            # Store layer positions for connections
            layer_infos = []

            # Draw each FC layer
            for i, neurons_count in enumerate(neuron_counts):
                layer_x = x_pos + i * layer_spacing

                # Apply sampling strategy
                strategy = get_sampling_strategy(neurons_count)
                neurons_to_draw = strategy['sample']
                draw_ellipsis = not strategy['show_all']

                neuron_positions = []

                # Calculate total height needed for neurons
                total_height = neurons_to_draw * 1.5 * neuron_radius * 2
                start_y = 0.5 - (total_height / 2)

                # Draw neurons
                for j in range(neurons_to_draw):
                    # Calculate neuron position
                    neuron_y = start_y + j * 1.5 * neuron_radius * 2

                    # Draw neuron
                    circle = plt.Circle((layer_x, neuron_y), neuron_radius, color=node_color, zorder=4)
                    ax.add_patch(circle)
                    neuron_positions.append((layer_x, neuron_y))

                # Draw ellipsis if needed
                if draw_ellipsis:
                    ellipsis_y = start_y + neurons_to_draw * 1.5 * neuron_radius * 2 + neuron_radius * 2
                    ax.text(layer_x, ellipsis_y, "...", ha='center', va='center', fontsize=12, zorder=4)

                    # Draw the actual count
                    ax.text(layer_x, ellipsis_y + neuron_radius * 3,
                            f"({neurons_count} total)",
                            ha='center', va='center', fontsize=8, zorder=4)

                # Draw layer label - Use proper index for FC layers to avoid numbering issues
                if i < len(fc_layers):
                    # Get the actual name from the layer or use FC{i} for correct sequential numbering
                    layer_name = fc_layers[i].get('name', f"FC{i}")
                    ax.text(layer_x, start_y - neuron_radius * 3,
                            layer_name, ha='center', va='center', fontsize=10, zorder=4)

                # Save layer info
                layer_info = {
                    'x': layer_x,
                    'positions': neuron_positions,
                    'name': f"FC{i}" if i < len(fc_layers) else "Output",
                    'type': 'fc'
                }
                layer_infos.append(layer_info)

            # Draw connections between FC layers
            for i in range(len(layer_infos) - 1):
                from_layer = layer_infos[i]
                to_layer = layer_infos[i + 1]

                # Get weight values if available
                weight_values = None
                if i < len(weights) and weights[i] is not None:
                    if 'data' in weights[i]:
                        weight_values = weights[i]['data']
                    elif 'tensor' in weights[i]:
                        weight_values = weights[i]['tensor']
                    elif 'shape' in weights[i]:
                        shape = weights[i]['shape']
                        if len(shape) == 2:
                            weight_values = np.random.randn(shape[0], shape[1])

                # Draw connections
                for from_idx, from_pos in enumerate(from_layer['positions']):
                    for to_idx, to_pos in enumerate(to_layer['positions']):
                        color = 'gray'  # Default color

                        if weight_values is not None and from_idx < weight_values.shape[1] and to_idx < \
                                weight_values.shape[
                                    0]:
                            weight = weight_values[to_idx, from_idx]
                            cmap = plt.cm.bwr
                            norm_weight = max(-1, min(1, weight))
                            color = cmap((norm_weight + 1) / 2)

                        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                                color=color, linewidth=0.5, alpha=0.6, zorder=3)

            # Update final x_pos
            if layer_infos:
                x_pos = layer_infos[-1]['x'] + layer_spacing / 2

        # Format the plot
        self._format_plot_2d(ax, x_pos)
        return fig

    def _visualize_cnn_layers(self, layers: list[dict]) -> matplotlib.figure.Figure:

        """
            Visualizes the structure of convolutional neural network layers, including
            convolution, pooling, and flatten layers. The function dynamically arranges
            and annotates components based on the provided layer configuration for
            visual clarity.

            The visualization will adapt to the structure, spacing, and styles based on
            the layers provided in the `layers` parameter. The output is a matplotlib
            figure detailing the sequence and connectivity of the layers.

            This function is intended for debug and presentation purposes.

            :param layers: A list of dictionaries where each dictionary contains
                information about a specific layer in the CNN. The dictionary should
                specify at least the `type` of the layer. Additional details such as
                `segment_idx` or `segment_activation` can be provided for labeling and
                additional annotations.
            :type layers: List[Dict[str, Any]]

            :return: A matplotlib figure object representing the visualized CNN layers.
            :rtype: matplotlib.figure.Figure
            """
        # Debug check
        if not layers:
            print("Warning: No layers provided to visualize")
            # Return empty figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, "No layers to visualize", ha='center', va='center')
            ax.axis('off')
            return fig

        # Increase figure size and use 16:9 aspect ratio
        fig, ax = plt.subplots(figsize=(20, 12))

        # Visual settings
        layer_spacing = 3.0  # Increased from 1.5 for better visibility
        color_map = plt.cm.viridis
        standard_center_y = 0.5
        standard_height = 0.8

        # Track position for layer placement and connections
        x_pos = 0
        prev_layer_info = None

        # Add a label list to track layer labels
        layer_labels = []

        # Iterate through layers
        for i, layer in enumerate(layers):
            layer_type = layer.get('type', '')
            segment_idx = layer.get('segment_idx', i)

            # Store the starting x-position for this layer
            layer_start_x = x_pos

            if layer_type in ['conv2d', 'conv1d']:
                # Draw convolutional layer
                x_pos, layer_info = self._draw_conv_layer_2d(
                    ax, layer, x_pos, i, len(layers),
                    color_map, standard_center_y, standard_height
                )

                # Draw arrow if not first layer
                if prev_layer_info:
                    self._draw_arrow_between_layers(ax, prev_layer_info, layer_info)

                prev_layer_info = layer_info

                # Add layer label
                activation = layer.get('segment_activation', '')
                layer_labels.append({
                    'x': (layer_start_x + x_pos) / 2,  # Center point of the layer
                    'text': f"Layer {segment_idx}",
                    'activation': activation
                })

                # Check if we need to add pooling after this layer
                if i < len(layers) - 1:
                    next_layer = layers[i + 1]
                    next_type = next_layer.get('type', '')

                    if next_type in ['maxpool2d', 'avgpool2d']:
                        # Store the starting x-position for pooling layer
                        pooling_start_x = x_pos

                        # Draw pooling and get its position info
                        x_pos, pooling_info = self._draw_pooling_layer_2d(
                            ax, x_pos, color_map, standard_center_y, standard_height
                        )

                        # Draw arrow to connect
                        if layer_info:
                            self._draw_arrow_between_layers(ax, layer_info, pooling_info)

                        prev_layer_info = pooling_info

                        # Add pooling label (using the next layer's segment index)
                        pool_segment_idx = next_layer.get('segment_idx', i + 1)
                        pool_activation = next_layer.get('segment_activation', '')
                        layer_labels.append({
                            'x': (pooling_start_x + x_pos) / 2,  # Center point of the pooling layer
                            'text': f"Layer {pool_segment_idx}",
                            'activation': pool_activation,
                            'subtext': f"({next_type})"
                        })

            elif layer_type in ['maxpool2d', 'avgpool2d']:
                # Draw pooling layer if not handled in the previous iteration
                x_pos, pooling_info = self._draw_pooling_layer_2d(
                    ax, x_pos, color_map, standard_center_y, standard_height
                )

                # Draw arrow if not first layer
                if prev_layer_info:
                    self._draw_arrow_between_layers(ax, prev_layer_info, pooling_info)

                prev_layer_info = pooling_info

                # Add layer label
                activation = layer.get('segment_activation', '')
                layer_labels.append({
                    'x': (layer_start_x + x_pos) / 2,  # Center point of the layer
                    'text': f"Layer {segment_idx}",
                    'activation': activation,
                    'subtext': f"({layer_type})"
                })

            elif layer_type == 'flatten':
                # Draw flatten transition
                x_pos, layer_info = self._draw_flatten_transition_2d(
                    ax, x_pos, color_map, standard_center_y, standard_height
                )

                # Draw arrow if not first layer
                if prev_layer_info:
                    self._draw_arrow_between_layers(ax, prev_layer_info, layer_info)

                prev_layer_info = layer_info

                # Add layer label
                activation = layer.get('segment_activation', '')
                layer_labels.append({
                    'x': (layer_start_x + x_pos) / 2,  # Center point of the layer
                    'text': f"Layer {segment_idx}",
                    'activation': activation,
                    'subtext': "(flatten)"
                })

        # Calculate the bottom position for labels
        y_min, y_max = ax.get_ylim()
        label_y = y_min - 0.3  # Position below the visualization

        # Add layer labels
        for label_info in layer_labels:
            # Layer number text
            ax.text(
                label_info['x'],
                label_y,
                label_info['text'],
                ha='center',
                va='top',
                fontsize=10
            )

            # Add subtext if available (layer type)
            if 'subtext' in label_info and label_info['subtext']:
                ax.text(
                    label_info['x'],
                    label_y - 0.2,
                    label_info['subtext'],
                    ha='center',
                    va='top',
                    fontsize=9,
                    color='gray'
                )

            # Activation function text (on a new line)
            if label_info['activation']:
                ax.text(
                    label_info['x'],
                    label_y - 0.4 if 'subtext' not in label_info else label_y - 0.6,
                    f"({label_info['activation']})",
                    ha='center',
                    va='top',
                    fontsize=9
                )

        # Format the plot - pass label_y to adjust bottom margin
        self._format_plot_2d(ax, x_pos + layer_spacing, True, label_y)

        # Add title
        ax.set_title('Convolutional Neural Network Architecture', fontsize=14)

        # Use 16:9 aspect ratio
        ax.set_aspect(16 / 9)

        return fig

    def _visualize_fcnn_layers(self, layers):
        """
        Visualizes the Fully Connected Neural Network (FCNN) layer structure, including neuron
        nodes and inter-layer connections. This visualization provides an overview of the
        network architecture, connection relationships, and possibly the weight distribution
        (if weights are provided).

        The function is capable of handling layers with missing specifications (e.g.,
        `in_features` or `out_features`) by adopting default settings or fallbacks. For
        layers with large numbers of neurons, scaling and ellipsis are applied to keep the
        visualization interpretable. Additionally, random or default weights can be used if
        none are provided for inter-layer connections.

        This visualization makes use of matplotlib for drawing the network diagram, including
        customized settings for layer spacing, node size, and connection intensity based on
        weights. The connections between layers are color-encoded to depict strength (weak
        connections in blue and strong connections in red) when weight information is available.

        :param layers: List of FCNN layers to visualize. Each layer can be specified as a
            dictionary or object containing attributes like `in_features`,
            `out_features`, and optional `parameters` with `weight` details.
        :type layers: list

        :raises TypeError: If `layers` is not a list or contains invalid layer data.
        :raises ValueError: If random weight generation fails or unexpected issues
            arise during plotting.

        :return: Matplotlib figure object containing the FCNN visualization. An empty
            figure with a warning message is returned if no layers are provided.
        :rtype: matplotlib.figure.Figure
        """
        # Debug check
        if not layers:
            print("Warning: No layers provided to visualize")
            # Return empty figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, "No layers to visualize", ha='center', va='center')
            ax.axis('off')
            return fig

        # Increase figure width and height for better spacing
        fig, ax = plt.subplots(figsize=(20, 12))  # Wider and taller

        # Calculate neuron counts for each layer
        neuron_counts = []
        weights = []
        using_random_weights = False

        # First collect all input features for nodes
        for layer in layers:
            if 'in_features' in layer:
                neuron_counts.append(layer['in_features'])
            elif hasattr(layer, 'in_features'):
                neuron_counts.append(layer.in_features)
            else:
                print(f"Warning: Layer missing in_features: {layer}")
                # Fallback to a default value
                neuron_counts.append(1)

            # Extract weights if available
            if 'parameters' in layer and 'weight' in layer['parameters']:
                weights.append(layer['parameters']['weight'])
            else:
                weights.append(None)
                using_random_weights = True  # Mark that we'll need to use random weights

        # Check if the last layer is the final layer of the model
        is_final_layer = False
        if layers:
            last_layer = layers[-1]
            # Check if this is the last layer in the model by checking its segment index
            all_layers = self._extract_all_layers() if hasattr(self, '_extract_all_layers') else []
            if all_layers:
                max_segment_idx = max([layer.get('segment_idx', 0) for layer in all_layers])
                is_final_layer = last_layer.get('segment_idx', -1) == max_segment_idx

            # If this is the final layer OR if we're specifically asked to show the output layer,
            # add the output features as the last column
            if is_final_layer:
                if 'out_features' in last_layer:
                    neuron_counts.append(last_layer['out_features'])
                elif hasattr(last_layer, 'out_features'):
                    neuron_counts.append(last_layer.out_features)
                else:
                    print(f"Warning: Last layer missing out_features: {last_layer}")
                    neuron_counts.append(1)  # Default fallback

        # Visual settings
        layer_spacing = 4.5  # Further increased spacing between layers
        neuron_radius = 0.25  # Slightly smaller nodes
        node_color = 'lightblue'  # Consistent color for all nodes

        # Calculate scale factor for dense layers
        max_neurons = max(neuron_counts) if neuron_counts else 1
        # Only show ellipsis for layers with more than 100 neurons
        neuron_threshold = 100
        scale_factor = 1.0
        if max_neurons > neuron_threshold:
            scale_factor = neuron_threshold / max_neurons

        # Store layer positions for connections
        layer_infos = []

        # Draw each FC layer
        for i, neurons_count in enumerate(neuron_counts):
            layer_x = i * layer_spacing

            # Determine how many neurons to draw
            draw_ellipsis = neurons_count > neuron_threshold
            neurons_to_draw = neurons_count if not draw_ellipsis else int(neurons_count * scale_factor)

            neuron_positions = []

            # Calculate total height needed for neurons, but use a more compact layout
            total_height = neurons_to_draw * 1.8 * neuron_radius  # Reduced vertical spacing
            if draw_ellipsis:
                total_height += 1.8 * neuron_radius  # Space for ellipsis

            # Calculate starting y-position to center neurons vertically
            start_y = -total_height / 2

            # Draw neurons in this layer
            for j in range(neurons_to_draw):
                neuron_y = start_y + j * 1.8 * neuron_radius + neuron_radius
                circle = plt.Circle(
                    (layer_x, neuron_y),
                    neuron_radius,
                    color=node_color if i < len(neuron_counts) - 1 or not is_final_layer else 'lightgreen',
                    # Highlight output layer
                    ec='black',
                    lw=0.5
                )
                ax.add_patch(circle)
                neuron_positions.append(neuron_y)

            # Add ellipsis for large layers
            if draw_ellipsis:
                ellipsis_y = start_y + neurons_to_draw * 1.8 * neuron_radius + neuron_radius
                ax.text(layer_x, ellipsis_y, "...", ha='center', va='center', fontsize=10)

            layer_infos.append({
                'x': layer_x,
                'positions': neuron_positions,
                'count': neurons_count
            })

        # Import ColorMap for weight coloring
        from matplotlib.colors import LinearSegmentedColormap

        # Create custom colormap: blue (weak) -> white (medium) -> red (strong)
        weight_cmap = LinearSegmentedColormap.from_list(
            'weight_cmap',
            ['#2050C0', '#FFFFFF', '#C02050']
        )

        # Draw connections between layers - but only between actual layers
        # and to output layer if it's the final layer
        connections_to_draw = len(layer_infos) - 1
        for i in range(connections_to_draw):
            source_layer = layer_infos[i]
            target_layer = layer_infos[i + 1]

            # Get weights for this layer if available
            weight_matrix = None
            if i < len(weights) and weights[i] is not None:
                if isinstance(weights[i], dict) and 'shape' in weights[i]:
                    # Try to create a mock weight matrix for visualization
                    out_features = weights[i]['shape'][0] if len(weights[i]['shape']) >= 1 else target_layer['count']
                    in_features = weights[i]['shape'][1] if len(weights[i]['shape']) >= 2 else source_layer['count']
                    # Create a normalized random weight matrix for visualization if actual values aren't available
                    weight_matrix = np.random.randn(out_features, in_features)
                    # Normalize to [-1, 1] range for color mapping
                    weight_matrix = 2 * (weight_matrix - weight_matrix.min()) / (
                            weight_matrix.max() - weight_matrix.min()) - 1
            else:
                # Use random weights with a moderate distribution
                out_features = target_layer['count']
                in_features = source_layer['count']
                np.random.seed(i)  # Set seed for reproducibility
                weight_matrix = np.random.randn(out_features, in_features)
                # Normalize to [-1, 1] range for color mapping
                weight_matrix = 2 * (weight_matrix - weight_matrix.min()) / (
                            weight_matrix.max() - weight_matrix.min()) - 1

            # Draw connections between neurons
            src_indices = list(range(min(len(source_layer['positions']), source_layer['count'])))
            tgt_indices = list(range(min(len(target_layer['positions']), target_layer['count'])))

            # Scale down the number of connections if there are too many
            if len(src_indices) * len(tgt_indices) > 1000:
                src_indices = src_indices[:20]  # Limit to first 20 source neurons
                tgt_indices = tgt_indices[:20]  # Limit to first 20 target neurons

            for src_idx in src_indices:
                src_y = source_layer['positions'][src_idx]
                for tgt_idx in tgt_indices:
                    tgt_y = target_layer['positions'][tgt_idx]

                    # Get weight value for color and alpha
                    weight_val = 0
                    alpha = 0.2  # Base opacity
                    if weight_matrix is not None and src_idx < weight_matrix.shape[1] and tgt_idx < weight_matrix.shape[
                        0]:
                        weight_val = weight_matrix[tgt_idx, src_idx]
                        # Adjust alpha based on absolute weight value (stronger connections are more visible)
                        alpha = 0.1 + 0.4 * abs(weight_val)

                    # Map weight value to color
                    line_color = weight_cmap((weight_val + 1) / 2)  # Map from [-1, 1] to [0, 1] for colormap

                    line = plt.Line2D(
                        [source_layer['x'], target_layer['x']],
                        [src_y, tgt_y],
                        color=line_color,
                        alpha=alpha,
                        linewidth=0.5
                    )
                    ax.add_line(line)

        # Add labels at the bottom
        bottom_margin = 0.8

        # Find the lowest point of the graph
        min_y = min([min(layer['positions']) if layer['positions'] else 0 for layer in layer_infos])
        label_y = min_y - bottom_margin

        # Add labels for each layer
        for i in range(len(layers)):
            if i >= len(layer_infos):
                break

            layer_x = layer_infos[i]['x']
            segment_idx = layers[i].get('segment_idx', i)
            activation = layers[i].get('segment_activation', '')

            # Layer number text
            ax.text(layer_x, label_y, f"Layer {segment_idx}", ha='center', va='top', fontsize=10)

            # Activation function text (on a new line)
            if activation:
                ax.text(layer_x, label_y - 0.4, f"({activation})", ha='center', va='top', fontsize=9)

        # Add a label for the output layer if we're showing it
        if is_final_layer and len(layer_infos) > len(layers):
            output_layer_x = layer_infos[-1]['x']
            ax.text(output_layer_x, label_y, "Output", ha='center', va='top', fontsize=10)

        # Set equal aspect ratio and limits
        ax.autoscale_view()

        # Add a bit of padding
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min - 1, x_max + 1)
        # Add extra space at the bottom for labels and potentially the random weights message
        ax.set_ylim(min(y_min, label_y - 2.5), y_max + 0.5)

        # If using random weights, add a note at the bottom
        if using_random_weights:
            ax.text(
                (x_min + x_max) / 2,  # Center horizontally
                label_y - 1.8,  # Position below layer labels
                "Note: Connection colors based on randomly generated weights (blue=weak, red=strong)",
                ha='center',
                va='top',
                fontsize=9,
                style='italic',
                alpha=0.7
            )

        # Add a small colorbar to show weight legend
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Create a small axes for the colorbar
        cax = fig.add_axes([0.92, 0.3, 0.015, 0.4])  # [left, bottom, width, height]
        norm = Normalize(vmin=-1, vmax=1)
        sm = ScalarMappable(cmap=weight_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Weight Strength', fontsize=9)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Negative', 'Neutral', 'Positive'])

        ax.axis('off')
        ax.set_title('Fully Connected Neural Network Architecture', fontsize=14)

        # Use a 16:9 aspect ratio
        ax.set_aspect(16 / 9)

        return fig

    def _extract_all_layers(self):
        """
        Extracts all layers from the metadata, augmenting each layer with
        additional information such as the type and activation of its containing
        segment, as well as its index within that segment.

        The method iterates through the metadata's 'segments' entry, if it exists,
        and processes the contained 'layers' entries. Each layer is enriched with
        metadata about its parent segment and added to the returned collection.

        :return: A list of dictionaries, each representing a layer enriched with
                 additional information about its segment and position.
        :rtype: list[dict]
        """
        all_layers = []

        if 'segments' in self.metadata:
            for segment_idx, segment in enumerate(self.metadata['segments']):
                segment_type = segment.get('type', '')
                segment_activation = segment.get('activation', '')

                if 'layers' in segment:
                    for layer_idx, layer in enumerate(segment['layers']):
                        # Create a copy with segment information
                        layer_with_segment = layer.copy()
                        layer_with_segment['segment_type'] = segment_type
                        layer_with_segment['segment_activation'] = segment_activation
                        layer_with_segment['segment_idx'] = segment_idx
                        layer_with_segment['layer_idx_in_segment'] = layer_idx
                        all_layers.append(layer_with_segment)

        return all_layers

    def _draw_conv_layer_2d(self, ax, layer, x_pos, seg_idx, total_segments, color_map,
                            center_y=0.5, standard_height=0.8):
        """
        Draws a visual representation of a 2D convolutional layer segment on the specified
        matplotlib axis. This method is designed to render both the visual dimensions
        of the convolutional layer and its descriptive annotations, such as in/out channels
        and kernel size. Elements of the visualization include stacked layers, grid lines,
        text details, and color gradients to represent depth.

        :param ax: Matplotlib axis used to draw the layer visualization.
        :type ax: matplotlib.axes.Axes
        :param layer: Dictionary containing layer attributes such as `in_channels`,
                      `out_channels`, and possibly `shape` (kernel height and width).
                      Optional activation and name attributes may also be present.
        :type layer: dict
        :param x_pos: X-coordinate for the position of the current layer.
        :type x_pos: float
        :param seg_idx: Segment index used for varying layer visual properties (e.g., color).
        :type seg_idx: int
        :param total_segments: Total number of segments available for visualization.
        :type total_segments: int
        :param color_map: Color map used for gradient rendering of the layer.
        :type color_map: matplotlib.colors.Colormap
        :param center_y: Central y-coordinate around which the layer is visually centered.
        :type center_y: float, optional
        :param standard_height: Standardized height for rendering the visual layer.
                                The width is scaled proportionally to
                                maintain the aspect ratio.
        :type standard_height: float, optional
        :return: A tuple containing the updated X-coordinate for the next layer and
                 a dictionary with positional data for the current layer.
        :rtype: tuple[float, dict] or tuple[float, None]
        """
        from matplotlib import patheffects

        # Skip if missing required info
        if 'in_channels' not in layer or 'out_channels' not in layer:
            return x_pos, None

        # Get output dimensions
        out_channels = layer.get('out_channels', 1)

        # Get shape information
        if 'shape' in layer:
            shape = layer['shape']
            kernel_h, kernel_w = shape[-2], shape[-1]
        else:
            kernel_h, kernel_w = 3, 3

        # Calculate visual dimensions (maintain aspect ratio but with standard height)
        aspect_ratio = kernel_w / kernel_h
        visual_height = standard_height
        visual_width = visual_height * aspect_ratio

        # Calculate y_offset to center the layer at center_y
        y_offset = center_y - (visual_height / 2)

        # Determine number of matrices to stack
        num_matrices = min(5, out_channels)

        # Base color for this segment
        base_color = color_map(0.3 + 0.5 * seg_idx / max(1, total_segments - 1))

        # Draw stacked matrices - from back to front
        for i in range(num_matrices):
            # Small offset for stacked effect (only horizontal, not vertical)
            x_offset = x_pos + 0.015 * (num_matrices - i - 1)

            # Darker matrices at the back, brighter at front
            matrix_color = color_map(0.2 + 0.6 * i / max(1, num_matrices - 1))
            alpha = 0.5 + 0.4 * i / max(1, num_matrices - 1)

            # Draw the matrix as a rectangle
            rect = plt.Rectangle(
                (x_offset, y_offset),
                visual_width,
                visual_height,
                facecolor=matrix_color,
                edgecolor='black',
                alpha=alpha,
                linewidth=0.7
            )
            ax.add_patch(rect)

            # Add grid lines for the front matrix only
            if i == num_matrices - 1:
                self._add_grid_lines_2d(ax, x_offset, y_offset, visual_width, visual_height)

        # Add layer information above the layer
        layer_name = layer.get('name', f"Conv{seg_idx}")
        in_channels = layer.get('in_channels', 1)
        activation = layer.get('activation', "")

        # Layer info text (positioned above the layer)
        ax.text(
            x_pos + visual_width / 2,
            y_offset + visual_height + 0.05,
            f"{layer_name} • {in_channels}→{out_channels}\n{kernel_h}×{kernel_w} {activation}",
            ha='center', va='bottom',
            fontsize=8,
            color='black',
            weight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=2, foreground='white')
            ]
        )

        # Store layer position information for arrow connections
        layer_info = {
            'start_x': x_pos,
            'end_x': x_pos + visual_width,
            'height': visual_height,
            'y_offset': y_offset,
            'center_y': center_y,  # Use the exact center_y passed in
            'width': visual_width
        }

        # Update position for next layer with reduced spacing
        return x_pos + 0.5 + visual_width, layer_info

    def _draw_pooling_layer_2d(self, ax, x_pos, color_map, center_y=0.5, standard_height=0.8):
        """
        Draws a 2D pooling layer representation on a matplotlib axis.

        This function visualizes a 2D pooling layer typically used in
        convolutional neural networks by drawing stacked rectangles
        (horizontally offset) to represent the pooling operation.
        It adds annotations ("MaxPool") to label the pooling layer
        and returns the updated x position after the layer and its related
        information.

        :param ax: The matplotlib axis on which the pooling layer will be drawn.
            This axis serves as the canvas for drawing the pooling layer.
        :param x_pos: The x-coordinate on the axis where the pooling layer starts.
        :param color_map: A colormap function used to determine the color of the pooling
            layers for visualization consistency.
        :param center_y: The y-coordinate of the center position of the layer. Defaults
            to 0.5.
        :param standard_height: The standard visualization height for the layer from
            which the pooling layer's height is derived. Defaults to 0.8.
        :return: A tuple where the first element is the updated x position after
            the pooling layer, and the second element is a dictionary containing
            information about the drawn layer's position properties.
        """
        from matplotlib import patheffects

        # Use a smaller height for pooling layers but maintain the center position
        pool_height = standard_height * 0.7
        pool_width = pool_height  # Make it square

        # Calculate y_offset to center at center_y
        y_offset = center_y - (pool_height / 2)

        pooling_color = color_map(0.2)

        # Draw stacked pooling layers (horizontal stacking only)
        for i in range(3):
            # Small offset (horizontal only)
            x_offset = x_pos + 0.01 * (3 - i - 1)

            # Alpha and color
            alpha = 0.5 + 0.4 * i / 2

            # Draw the pooling layer as a rectangle
            rect = plt.Rectangle(
                (x_offset, y_offset),
                pool_width,
                pool_height,
                facecolor=pooling_color,
                edgecolor='black',
                alpha=alpha,
                linewidth=0.7
            )
            ax.add_patch(rect)

        # Add pooling label above the layer
        ax.text(
            x_pos + pool_width / 2,
            y_offset + pool_height + 0.05,
            "MaxPool",
            ha='center', va='bottom',
            fontsize=7,
            color='black',
            weight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=2, foreground='white')
            ]
        )

        # Store layer position information for arrow connections
        layer_info = {
            'start_x': x_pos,
            'end_x': x_pos + pool_width,
            'height': pool_height,
            'y_offset': y_offset,
            'center_y': center_y,  # Use the exact center_y passed in
            'width': pool_width
        }

        # Update position
        return x_pos + 0.5 + pool_width, layer_info

    def _draw_flatten_transition_2d(self, ax, x_pos, color_map, center_y=0.5, standard_height=0.8):
        """
        Draws a visual representation of a 2D flatten transition for neural network diagrams.

        The method draws stacked progressively narrower rectangles to depict the flattening
        layer transition in a neural network. It adds the representation of the flattened layer
        to the provided axis, and updates the layer's positional information for further diagram
        connections. The color and style of the transition are defined by the provided color
        map, and users can adjust the layer's height and center position.

        :param ax: Matplotlib axis object where the flatten layer transition will be drawn.
        :param x_pos: Position along the x-axis where the flatten transition starts.
        :param color_map: An instance of a matplotlib color map used to fill the layer's
            rectangles.
        :param center_y: Vertical center position of the flatten layer transition. Default is 0.5.
        :param standard_height: Standard height value used to scale the flatten layer
            transition for consistent sizing. Default is 0.8.
        :return: Tuple containing the updated x-axis position after the flatten transition and
            a dictionary with metadata about the layer's position and dimensions for further
            use.

        """
        from matplotlib import patheffects

        # Use standard dimensions but maintain center position
        flatten_height = standard_height * 0.75
        flatten_width = flatten_height

        # Calculate y_offset to center at center_y
        y_offset = center_y - (flatten_height / 2)

        transition_color = color_map(0.5)

        # Draw flatten transition with progressive shapes (horizontal stacking only)
        for i in range(4):
            # Small horizontal offset
            x_offset = x_pos + 0.01 * (4 - i - 1)

            # Width narrows as we go higher in stack
            width_factor = 1.0 - 0.15 * i / 3
            height_factor = 1.0  # Keep height constant for horizontal alignment

            # Alpha and color
            alpha = 0.5 + 0.4 * i / 3

            # Draw the transition as a rectangle
            rect = plt.Rectangle(
                (x_offset, y_offset),
                flatten_width * width_factor,
                flatten_height * height_factor,
                facecolor=transition_color,
                edgecolor='black',
                alpha=alpha,
                linewidth=0.7
            )
            ax.add_patch(rect)

        # Add flattening effect with horizontal lines
        for i in range(3):
            y_level = y_offset + 0.2 * i * flatten_height
            ax.plot(
                [x_pos + 0.1, x_pos + flatten_width - 0.1],
                [y_level, center_y],
                color='black',
                alpha=0.6,
                linewidth=0.5
            )

        # Add label above the layer
        ax.text(
            x_pos + flatten_width / 2,
            y_offset + flatten_height + 0.05,
            "FLATTEN\n→ FC NETWORK",
            ha='center', va='bottom',
            fontsize=8,
            color='black',
            weight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=2, foreground='white')
            ]
        )

        # Store layer position information for arrow connections
        layer_info = {
            'start_x': x_pos,
            'end_x': x_pos + flatten_width,
            'height': flatten_height,
            'y_offset': y_offset,
            'center_y': center_y,  # Use the exact center_y passed in
            'width': flatten_width
        }

        # Update position
        return x_pos + 0.5 + flatten_width, layer_info

    def _draw_arrow_between_layers(self, ax, prev_layer, curr_layer):
        """
        Draws a straight horizontal arrow between two layers on a given axis using matplotlib's FancyArrowPatch tool. The
        arrow starts from the center of the previous layer's right side and ends at the center of the current layer's left side.

        :param ax: The axis where the arrow will be drawn.
        :type ax: matplotlib.axes.Axes
        :param prev_layer: Dictionary containing the positional and dimensional attributes of the previous layer. Must include
            keys 'end_x' (float) and 'center_y' (float) for arrow start point computation.
        :type prev_layer: dict
        :param curr_layer: Dictionary containing the positional and dimensional attributes of the current layer. Must include
            key 'start_x' (float) for arrow end point computation.
        :type curr_layer: dict
        :return: None
        :rtype: None
        """
        from matplotlib.patches import FancyArrowPatch

        # Calculate arrow start and end points
        start_x = prev_layer['end_x']
        start_y = prev_layer['center_y']  # Use center_y from previous layer

        end_x = curr_layer['start_x']
        end_y = prev_layer['center_y']  # Use same y as start for horizontal arrow

        # Create a straight arrow
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle="fancy,head_width=6,head_length=8,tail_width=0.5",
            facecolor='gray',
            edgecolor='black',
            linewidth=0.8,
            alpha=0.7,
            zorder=1
        )

        # Add the arrow to the plot
        ax.add_patch(arrow)

    def _add_grid_lines_2d(self, ax, x, y, width, height, num_lines=4):
        """
        Add grid lines to a 2D space within the given rectangular area on the `ax` plot.

        This function generates evenly spaced horizontal and vertical grid lines based on
        the specified number of divisions (`num_lines`) within the provided rectangle defined
        by its top-left corner (`x`, `y`) and dimensions (`width`, `height`). The grid lines
        are styled using black color, semi-transparency, and a thin line width.

        :param ax: The axes object (matplotlib Axes) on which the grid lines will be drawn.
        :param x: The x-coordinate of the top-left corner of the rectangular area.
        :param y: The y-coordinate of the top-left corner of the rectangular area.
        :param width: The width of the rectangular area where grid lines will be drawn.
        :param height: The height of the rectangular area where grid lines will be drawn.
        :param num_lines: The number of grid lines to draw for each direction (horizontal
            and vertical) within the rectangular area. The default is 4.
        :return: None
        """
        # Horizontal grid lines
        for j in range(1, num_lines):
            y_line = y + height * j / num_lines
            ax.plot(
                [x, x + width],
                [y_line, y_line],
                color='black',
                alpha=0.3,
                linewidth=0.5
            )

        # Vertical grid lines
        for j in range(1, num_lines):
            x_line = x + width * j / num_lines
            ax.plot(
                [x_line, x_line],
                [y, y + height],
                color='black',
                alpha=0.3,
                linewidth=0.5
            )

    def _format_plot_2d(self, ax, max_x, is_cnn=False, label_y=None):
        """
        Formats a 2D plot with specific layout settings. This method adjusts axis limits to include margins, hides axis ticks
        and spines, and sets the title based on the type of network.

        :param ax: Axis object for the plot to be formatted.
        :type ax: matplotlib.axes.Axes
        :param max_x: The maximum value for the x-axis.
        :type max_x: float
        :param is_cnn: Flag indicating if the plot refers to a Convolutional Neural Network. Defaults to False.
        :type is_cnn: bool, optional
        :param label_y: Optional reference value for the y-axis to ensure sufficient margin below. Defaults to None.
        :type label_y: float, optional
        :return: None. Modifies the plot axis in place.
        """
        # Set the limits with some margins
        margin = 0.5
        ax.set_xlim(-margin, max_x + margin)

        # Get current y limits
        y_min, y_max = ax.get_ylim()

        # If we have a label_y value, make sure there's enough space below
        if label_y is not None:
            bottom_margin = abs(label_y - y_min) + 1.0  # Extra space below labels
            ax.set_ylim(y_min - bottom_margin, y_max + margin)
        else:
            ax.set_ylim(y_min - margin, y_max + margin)

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add type-specific title
        title = "Convolutional Neural Network" if is_cnn else "Neural Network Architecture"
        ax.set_title(title, fontsize=14)

    def get_model_summary(self):
        """
        Generates a comprehensive summary of the model, including its type, total parameters,
        input size dimensions, segment count, and a breakdown of segment types. This method
        extracts information based on the model structure and metadata defined in the class.

        :return: A dictionary containing the following keys:
            - model_type (str): The type of the model.
            - total_parameters (int): The total number of parameters in the model.
            - input_size (Union[List[int], int, None]): Dimensions of the input, which can be
              either a list for convolutional models or an integer for fully connected models.
              None is returned if input dimensions cannot be determined.
            - segment_count (int): The total number of segments in the model.
            - segment_types (Dict[str, int]): A mapping of segment types to their occurrence counts.
        :rtype: Dict[str, Union[str, int, List[int], Dict[str, int], None]]
        """
        segment_types = {}
        for segment in self.segments:
            segment_type = segment.get("type", "unknown")
            segment_types[segment_type] = segment_types.get(segment_type, 0) + 1

        # Determine input size based on model type
        input_size = 0
        if self.segments:
            first_segment = self.segments[0]
            if first_segment.get("type") == "conv" and "layers" in first_segment and first_segment["layers"]:
                first_layer = first_segment["layers"][0]
                # For CNN, get input dimensions correctly
                if "in_channels" in first_layer and "height" in first_layer and "width" in first_layer:
                    # Format input size as [channels, height, width]
                    input_size = [
                        first_layer.get("in_channels", 0),
                        first_layer.get("height", 0),
                        first_layer.get("width", 0)
                    ]
            elif first_segment.get("type") == "fc" and "in_features" in first_segment:
                # For FCNN, use in_features
                input_size = first_segment.get("in_features", 0)

        return {
            "model_type": self.model_type,
            "total_parameters": self.metadata.get("total_parameters", 0),
            "input_size": input_size,
            "segment_count": len(self.segments),
            "segment_types": segment_types
        }

    def create_html_report(self, output_path=None):
        """
        Generates an HTML report summarizing the structure, layers, and parameters of the
        neural network model. This report includes detailed layer-level statistics,
        visualizations, and model metadata presented in a structured format.

        :param output_path: The path to save the generated HTML report. If not provided, the
            default value `model_report.html` is used.
        :type output_path: str | None

        :return: The file path to the saved HTML report.
        :rtype: str | None
        """
        if self.fig is None:
            print("No visualization available. Run visualize() first.")
            return None

        # Get model summary
        model_summary = self.get_model_summary()

        # Calculate model stats
        total_params = 0
        total_layers = len(self.segments)
        input_size = 0
        output_size = 0

        # Determine model type based on segments
        model_type_display = "Fully Connected Neural Network"  # Default

        # Check if we have convolutional layers
        has_conv = any(segment.get("type") == "conv" for segment in self.segments)
        has_fc = any(segment.get("type") == "fc" for segment in self.segments)

        if has_conv and has_fc:
            model_type_display = "Hybrid Neural Network"
        elif has_conv:
            model_type_display = "Convolutional Neural Network"

        # More robust input size detection
        input_size_display = "0"  # Default if detection fails

        # Try multiple approaches to get the input size
        if self.segments:
            first_segment = self.segments[0]

            # For CNN, check in_channels in the first layer, not in the segment
            if first_segment["type"] == "conv":
                # Check if there are layers with input dimensions
                if "layers" in first_segment and first_segment["layers"]:
                    first_layer = first_segment["layers"][0]

                    # Look for in_channels in the first_layer
                    if "in_channels" in first_layer:
                        in_channels = first_layer["in_channels"]
                        # For CNN, the input size should be formatted properly
                        input_size = in_channels

                        # Try to find input dimensions
                        input_height = first_layer.get("height", None)
                        input_width = first_layer.get("width", None)

                        # Check if we have dimensions
                        if input_height is not None and input_width is not None:
                            input_size_display = f"{in_channels} × {input_height} × {input_width}"
                            input_size = in_channels * input_height * input_width
                        else:
                            # Check if we can infer dimensions from weights shape
                            if "parameters" in first_layer and "weight" in first_layer["parameters"]:
                                weight_shape = first_layer["parameters"]["weight"]["shape"]
                                if len(weight_shape) == 4:  # [out_channels, in_channels, kernel_h, kernel_w]
                                    # We at least know the number of input channels
                                    input_size_display = f"{in_channels} channels"

                            # If we have more information from metadata, use that format
                            if "input_shape" in self.metadata:
                                input_shape = self.metadata["input_shape"]
                                if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
                                    input_size_display = f"{input_shape[0]} × {input_shape[1]} × {input_shape[2]}"
                                    input_size = input_shape[0] * input_shape[1] * input_shape[2]
                            else:
                                # If we don't have full dimensions, just show channels
                                input_size_display = f"{in_channels} channels"
                                input_size = in_channels
                    else:
                        # Fallback if in_channels not found in first_layer
                        input_size_display = "Unknown"
                        input_size = 0

            # For FC networks, look at the first layer's in_features
            elif first_segment["type"] == "fc" and "in_features" in first_segment:
                input_size = first_segment["in_features"]
                input_size_display = str(input_size)

        layer_stats = []
        activation_functions = set()
        output_neurons = 0

        for i, segment in enumerate(self.segments):
            seg_type = segment.get("type", "unknown")

            # Track activations
            activation = segment.get("activation", "None")
            if activation:
                activation_functions.add(activation)

            # Generate layer name
            layer_name = f"Layer {i + 1}: {seg_type.upper()}"
            if "name" in segment:
                layer_name = segment["name"]

            # Calculate parameters
            weights_count = 0
            biases_count = 0
            params_count = 0

            if "parameters" in segment and isinstance(segment["parameters"], dict):
                for param_name, param_info in segment["parameters"].items():
                    if isinstance(param_info, dict) and "size" in param_info:
                        if "weight" in param_name.lower():
                            weights_count = param_info["size"]
                        elif "bias" in param_name.lower():
                            biases_count = param_info["size"]
                        params_count += param_info["size"]
            elif "parameters" in segment:
                params_count = segment["parameters"]

            # Create layer info matching your template structure
            layer_info = {"name": layer_name, "type": seg_type.upper(),
                          "in_features": segment.get("in_features", segment.get("in_channels", "-")),
                          "out_features": segment.get("out_features", segment.get("out_channels", "-")),
                          "activation": activation.upper() if activation else "None",
                          "weights_count": f"{weights_count:,}", "biases_count": f"{biases_count:,}",
                          "params_count": f"{params_count:,}", "details": {}}

            # Add all additional details from your original code

            # Add layer-specific details based on type
            if seg_type == "conv":
                if "kernel_size" in segment:
                    layer_info["details"]["kernel_size"] = segment["kernel_size"]
                if "stride" in segment:
                    layer_info["details"]["stride"] = segment["stride"]
                if "padding" in segment:
                    layer_info["details"]["padding"] = segment["padding"]
                if "out_channels" in segment:
                    layer_info["details"]["out_channels"] = segment["out_channels"]

            elif seg_type == "fc":
                if "in_features" in segment:
                    layer_info["details"]["in_features"] = segment["in_features"]
                if "out_features" in segment:
                    layer_info["details"]["out_features"] = segment["out_features"]

            elif seg_type == "pool":
                if "pool_type" in segment:
                    layer_info["details"]["pool_type"] = segment["pool_type"]
                if "kernel_size" in segment:
                    layer_info["details"]["kernel_size"] = segment["kernel_size"]
                if "stride" in segment:
                    layer_info["details"]["stride"] = segment["stride"]

            elif seg_type == "dropout":
                if "p" in segment:
                    layer_info["details"]["probability"] = segment["p"]

            # Add input and output shapes if available
            if "input_shape" in segment:
                layer_info["details"]["input_shape"] = segment["input_shape"]
            if "output_shape" in segment:
                layer_info["details"]["output_shape"] = segment["output_shape"]

            layer_stats.append(layer_info)

            # Track output size
            if segment == self.segments[-1]:  # Last segment
                if seg_type == "fc":
                    output_neurons = segment.get("out_features", 0)
                elif seg_type == "conv":
                    output_neurons = segment.get("out_channels", 0)

        # Get image data as base64 for embedding in HTML
        # Save the current figure to a BytesIO object
        img_data = io.BytesIO()
        self.fig.savefig(img_data, format="png", dpi=150, bbox_inches='tight')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.read()).decode("utf-8")

        # Get path to templates directory
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("report_template.html")


        # Calculate if visualization needs horizontal scrolling
        fig_size_inches = self.fig.get_size_inches()
        actual_width = fig_size_inches[0] * 100  # approximate pixels
        visible_width = min(actual_width, 900)  # max width for display
        is_scrollable = actual_width > visible_width

        # Render the template with our data
        html_content = template.render(
            img_data=img_base64,
            summary=model_summary,
            model_type=self.model_type,
            model_type_display=model_type_display,
            input_size_display=input_size_display,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_scrollable=is_scrollable,
            visible_width=int(visible_width),
            actual_width=int(actual_width),
            total_params=self.format_number(total_params),
            total_layers=total_layers,
            input_size=input_size,
            output_size=output_size,
            layer_stats=layer_stats,
            activation_functions=list(activation_functions) if activation_functions else ["None"]
        )

        # Save the HTML file
        if output_path is None:
            output_path = "model_report.html"

        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"HTML report saved to: {output_path}")
        return output_path

    @staticmethod
    def format_number(num):
        """Format large numbers for readability"""
        if num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)


def try_model_visualizer(metadata_path, output_dir=None, report_name=None, visualization_name=None):
    """
    Function to test the ModelVisualizer on a given model.

    Args:
        metadata_path (str): Path to the metadata.json file
        output_dir (str, optional): Directory to save outputs. If None, uses model's directory.
        report_name (str, optional): Name for the HTML report file. Default: "model_report.html"
        visualization_name (str, optional): Name for the visualization image. Default: "model_visualization.png"
    """
    import os

    print(f"Loading model metadata from: {metadata_path}")

    # Create the visualizer
    visualizer = ModelVisualizer(metadata_path)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(metadata_path)

    os.makedirs(output_dir, exist_ok=True)

    # Determine file names
    if visualization_name is None:
        visualization_name = "model_visualization.png"

    if report_name is None:
        report_name = "model_report.html"

    visualization_path = os.path.join(output_dir, visualization_name)
    report_path = os.path.join(output_dir, report_name)

    # Create visualization
    print(f"Creating visualization: {visualization_path}")
    vis_path = visualizer.visualize(visualization_path)
    print(f"Visualization saved to: {vis_path}")

    # Create HTML report
    print(f"Generating HTML report: {report_path}")
    report_path = visualizer.create_html_report(report_path)
    print(f"HTML report saved to: {report_path}")

    # Print model summary
    summary = visualizer.get_model_summary()
    print("\nModel Summary:")
    print(f"  - Model Type: {summary['model_type']}")
    print(f"  - Total Parameters: {summary['total_parameters']:,}")
    print(f"  - Segment Count: {summary['segment_count']}")
    print(f"  - Segment Types: {', '.join([f'{k} ({v})' for k, v in summary['segment_types'].items()])}")

    # Try to open the visualization if we're in an environment that supports it
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
    except:
        print("Could not automatically open the HTML report.")


# Example usage:
if __name__ == "__main__":
    import os

    # Choose which model to visualize
    model_choice = 1  # Change this to test different models

    if model_choice == 1:
        # FCNN model
        model_dir = "models/test_model"
        metadata_path = os.path.join(model_dir, "output/metadata.json")
        output_dir = os.path.join(model_dir, "visualization")

    elif model_choice == 2:
        # CNN model
        model_dir = "models/test_cnn_model_with_biases"
        metadata_path = os.path.join(model_dir, "output/metadata.json")
        output_dir = os.path.join(model_dir, "visualization")

    elif model_choice == 3:
        # Transformer model
        model_dir = "models/transformer_model"
        metadata_path = os.path.join(model_dir, "metadata.json")
        output_dir = os.path.join(model_dir, "visualization")

    elif model_choice == 4:
        # Hybrid model (CNN + FCNN)
        model_dir = "models/doom"
        metadata_path = os.path.join(model_dir, "output/metadata.json")
        output_dir = os.path.join(model_dir, "visualization")

    elif model_choice == 5:
        # Custom model path
        metadata_path = "path/to/your/custom/model/metadata.json"
        output_dir = "path/to/your/custom/model/visualization"

    # Run the visualizer with the selected model
    try_model_visualizer(
        metadata_path=metadata_path,
        output_dir=output_dir,
        report_name="model_report.html",
        visualization_name="model_visualization.png"
    )
