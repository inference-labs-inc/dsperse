import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import os

def load_results(file_path: str) -> dict:
    """Load the test results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_metadata(model_dir: str) -> dict:
    """Load model metadata to get layer sizes."""
    metadata_path = os.path.join(model_dir, "slices", "metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)

def calculate_layer_sizes(metadata: dict) -> Dict[int, int]:
    """Calculate the size (number of parameters) for each layer."""
    layer_sizes = {}
    for i, segment in enumerate(metadata['segments']):
        size = 0
        for layer in segment['layers']:
            if 'weight' in layer:
                size += layer['weight']['size']
            if 'bias' in layer:
                size += layer['bias']['size']
        layer_sizes[i] = size
    return layer_sizes

def create_layer_frequency_plot(results: dict, output_path: str):
    """Create a bar plot showing layer selection frequency."""
    layer_freq = results['layer_frequency']
    layers = sorted(layer_freq.keys())
    frequencies = [layer_freq[layer] for layer in layers]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(layers, frequencies)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title('Layer Selection Frequency Across All Seeds')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Times Selected')
    plt.xticks(layers)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    plt.close()

def create_selection_matrix(results: dict, output_path: str):
    """Create a heatmap showing layer selection patterns for each seed."""
    test_results = results['test_results']
    seeds = [result['seed'] for result in test_results]
    all_layers = sorted(results['all_selected_layers'])
    
    # Create selection matrix
    selection_matrix = np.zeros((len(seeds), len(all_layers)))
    for i, result in enumerate(test_results):
        for layer in result['selected_layers']:
            selection_matrix[i, all_layers.index(layer)] = 1
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(selection_matrix, 
                xticklabels=all_layers,
                yticklabels=seeds,
                cmap='YlOrRd',
                cbar=False)
    
    plt.title('Layer Selection Patterns by Seed')
    plt.xlabel('Layer Index')
    plt.ylabel('Seed')
    
    plt.savefig(output_path)
    plt.close()

def create_inference_time_plot(results: dict, output_path: str):
    """Create a box plot showing inference time distribution for each layer."""
    test_results = results['test_results']
    layer_times = {}
    
    # Collect inference times for each layer
    for result in test_results:
        for layer in result['selected_layers']:
            if layer not in layer_times:
                layer_times[layer] = []
            layer_times[layer].append(result['inference_time'])
    
    # Create box plot
    plt.figure(figsize=(12, 6))
    data = [layer_times[layer] for layer in sorted(layer_times.keys())]
    plt.boxplot(data, labels=sorted(layer_times.keys()))
    
    plt.title('Inference Time Distribution by Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Inference Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    plt.close()

def create_size_vs_time_plot(results: dict, layer_sizes: Dict[int, int], output_path: str):
    """Create a scatter plot showing relationship between layer size and inference time."""
    test_results = results['test_results']
    sizes = []
    times = []
    layers = []
    
    # Collect size and time data
    for result in test_results:
        for layer in result['selected_layers']:
            sizes.append(layer_sizes[layer])
            times.append(result['inference_time'])
            layers.append(layer)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(sizes, times, c=layers, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(scatter, label='Layer Index')
    
    plt.title('Layer Size vs Inference Time')
    plt.xlabel('Layer Size (number of parameters)')
    plt.ylabel('Inference Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    plt.close()

def create_variance_analysis(results: dict, output_path: str):
    """Create a plot showing variance in inference time for different layer combinations."""
    test_results = results['test_results']
    num_layers = []
    times = []
    
    # Collect data about number of layers and inference times
    for result in test_results:
        num_layers.append(len(result['selected_layers']))
        times.append(result['inference_time'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(num_layers, times, alpha=0.6)
    
    # Add trend line
    z = np.polyfit(num_layers, times, 1)
    p = np.poly1d(z)
    plt.plot(sorted(set(num_layers)), p(sorted(set(num_layers))), "r--", alpha=0.8)
    
    plt.title('Variance in Inference Time by Number of Selected Layers')
    plt.xlabel('Number of Selected Layers')
    plt.ylabel('Total Inference Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    plt.close()

def create_combined_visualization(results: dict, metadata: dict, output_dir: str):
    """Create all visualizations and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate layer sizes
    layer_sizes = calculate_layer_sizes(metadata)
    
    # Create all visualizations
    create_layer_frequency_plot(
        results,
        os.path.join(output_dir, 'layer_frequency.png')
    )
    
    create_selection_matrix(
        results,
        os.path.join(output_dir, 'selection_patterns.png')
    )
    
    create_inference_time_plot(
        results,
        os.path.join(output_dir, 'inference_time_distribution.png')
    )
    
    create_size_vs_time_plot(
        results,
        layer_sizes,
        os.path.join(output_dir, 'size_vs_time.png')
    )
    
    create_variance_analysis(
        results,
        os.path.join(output_dir, 'variance_analysis.png')
    )
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # Model configuration
    model_choice = 2  # 1 for doom, 2 for net
    base_paths = {
        1: "src/models/doom",
        2: "src/models/net"
    }
    model_dir = base_paths[model_choice]
    
    # Load results and metadata
    results = load_results('jit_test_results.json')
    metadata = load_metadata(model_dir)
    
    # Create visualizations
    create_combined_visualization(results, metadata, 'visualization_results')
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 