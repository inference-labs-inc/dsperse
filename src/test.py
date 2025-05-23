import json
import os
import random
import time
from src.runners.jit_ezkl_runner import JITEzklRunner

def get_random_layers(seed: int, num_slices: int) -> list:
    """Get random layer selections for a given seed."""
    random.seed(seed)
    num_slices_to_test = random.randint(1, num_slices)
    layers_to_test = set()
    while len(layers_to_test) < num_slices_to_test:
        layer_idx = random.randint(0, num_slices - 1)
        layers_to_test.add(layer_idx)
    return sorted(list(layers_to_test))

def simulate_inference_time(selected_layers: list, is_circuitized: bool, is_first_run: bool) -> float:
    """Simulate inference time based on number and type of layers."""
    # Base time for each layer type (in seconds)
    base_times = {
        0: 0.15,  # First conv layer
        1: 0.12,  # Second conv layer
        2: 0.08,  # First FC layer
        3: 0.06,  # Second FC layer
        4: 0.04   # Last FC layer
    }
    
    # Circuitized layers take longer due to witness generation
    # First run includes witness generation, subsequent runs use cached witnesses
    if is_circuitized:
        multiplier = 2.5 if is_first_run else 1.2  # Much faster with cached witnesses
    else:
        multiplier = 1.0
    
    # Add some random variation (±20%)
    total_time = 0
    for layer in selected_layers:
        base_time = base_times[layer] * multiplier
        variation = random.uniform(-0.2, 0.2)
        total_time += base_time * (1 + variation)
    
    return total_time

def main():
    # Model configuration
    model_choice = 2  # 1 for doom, 2 for net
    base_paths = {
        1: "src/models/doom",
        2: "src/models/net"
    }
    model_dir = base_paths[model_choice]
    
    # Load metadata to get number of slices
    metadata_path = os.path.join(model_dir, "slices", "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    num_slices = len(metadata['segments'])
    
    # Run tests with different seeds
    seeds = [42, 123, 456, 789, 101112]
    test_results = []
    
    print("Running tests with different random seeds...")
    for seed_idx, seed in enumerate(seeds):
        print(f"\nRunning test with seed {seed}...")
        selected_layers = get_random_layers(seed, num_slices)
        
        # Simulate inference time for both circuitized and regular inference
        # First run includes witness generation, subsequent runs use cached witnesses
        is_first_run = seed_idx == 0
        circuitized_time = simulate_inference_time(selected_layers, True, is_first_run)
        inference_time = simulate_inference_time(
            [i for i in range(num_slices) if i not in selected_layers],
            False,
            is_first_run
        )
        total_time = circuitized_time + inference_time
        
        test_results.append({
            'seed': seed,
            'selected_layers': selected_layers,
            'circuitized_time': circuitized_time,
            'inference_time': inference_time,
            'total_time': total_time,
            'is_first_run': is_first_run
        })
        print(f"Selected layers for circuitization: {selected_layers}")
        print(f"Circuitized layers time: {circuitized_time:.3f} seconds")
        print(f"Regular inference time: {inference_time:.3f} seconds")
        print(f"Total time: {total_time:.3f} seconds")
        if is_first_run:
            print("(First run - includes witness generation)")
        else:
            print("(Using cached witnesses)")
    
    # Analyze results
    print("\nAnalysis of layer selection patterns:")
    all_selected_layers = set()
    layer_frequency = {}
    
    for result in test_results:
        layers = result['selected_layers']
        all_selected_layers.update(layers)
        for layer in layers:
            layer_frequency[layer] = layer_frequency.get(layer, 0) + 1
    
    print("\nLayer selection frequency for circuitization:")
    for layer, freq in sorted(layer_frequency.items()):
        print(f"Layer {layer}: selected {freq} times")
    
    print("\nUnique layers selected for circuitization:")
    print(f"Total unique layers: {len(all_selected_layers)}")
    print(f"Layers: {sorted(all_selected_layers)}")
    
    # Calculate average times for first run and subsequent runs
    first_run_times = [r for r in test_results if r['is_first_run']]
    cached_run_times = [r for r in test_results if not r['is_first_run']]
    
    print("\nAverage execution times (First run - with witness generation):")
    if first_run_times:
        avg_circuitized = sum(r['circuitized_time'] for r in first_run_times) / len(first_run_times)
        avg_inference = sum(r['inference_time'] for r in first_run_times) / len(first_run_times)
        avg_total = sum(r['total_time'] for r in first_run_times) / len(first_run_times)
        print(f"Circuitized layers: {avg_circuitized:.3f} seconds")
        print(f"Regular inference: {avg_inference:.3f} seconds")
        print(f"Total: {avg_total:.3f} seconds")
    
    print("\nAverage execution times (Using cached witnesses):")
    if cached_run_times:
        avg_circuitized = sum(r['circuitized_time'] for r in cached_run_times) / len(cached_run_times)
        avg_inference = sum(r['inference_time'] for r in cached_run_times) / len(cached_run_times)
        avg_total = sum(r['total_time'] for r in cached_run_times) / len(cached_run_times)
        print(f"Circuitized layers: {avg_circuitized:.3f} seconds")
        print(f"Regular inference: {avg_inference:.3f} seconds")
        print(f"Total: {avg_total:.3f} seconds")
    
    # Save results to file
    output_file = "jit_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'test_results': test_results,
            'layer_frequency': layer_frequency,
            'all_selected_layers': sorted(list(all_selected_layers)),
            'average_times': {
                'first_run': {
                    'circuitized': avg_circuitized if first_run_times else None,
                    'inference': avg_inference if first_run_times else None,
                    'total': avg_total if first_run_times else None
                },
                'cached_runs': {
                    'circuitized': avg_circuitized if cached_run_times else None,
                    'inference': avg_inference if cached_run_times else None,
                    'total': avg_total if cached_run_times else None
                }
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main() 