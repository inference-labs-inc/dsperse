import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent.parent

def load_test_results():
    """Load the test results from JSON file."""
    with open(BASE_DIR / "jit_test_results.json", 'r') as f:
        return json.load(f)

def get_random_layers(seed: int, num_slices: int, num_selected: int) -> list:
    """Get random layer selections for a given seed and number of layers to select."""
    random.seed(seed)
    layers = set()
    while len(layers) < num_selected:
        layers.add(random.randint(0, num_slices - 1))
    return sorted(list(layers))

def calculate_security_level(num_layers, agent_runs):
    """Calculate security level based on number of circuitized layers and agent runs."""
    if num_layers == 0 or agent_runs == 0:
        return 0.0
    return (1 - 1 / (2 ** (agent_runs * num_layers))) * 100

def create_model_visualization(selected_layers, total_layers):
    """Create a visualization of the model with selected layers highlighted."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create boxes for each layer
    box_width = 0.8
    box_height = 0.6
    spacing = 0.6
    
    for i in range(total_layers):
        x = i * (box_width + spacing)
        y = 0
        
        # Different colors for different layer types
        if i == 0 or i == 1:
            # Convolutional layer: triangle
            color = '#4F8BC9' if i in selected_layers else '#BFD7ED'
            triangle = plt.Polygon([
                [x + box_width/2, y + box_height],
                [x, y],
                [x + box_width, y]
            ], closed=True, facecolor=color, edgecolor='#2C3E50', alpha=0.9)
            ax.add_patch(triangle)
        else:
            # FC layer: rectangle
            color = '#7D6B91' if i in selected_layers else '#D6CDEA'
            rect = plt.Rectangle((x, y), box_width, box_height, 
                               facecolor=color, edgecolor='#2C3E50', alpha=0.9)
            ax.add_patch(rect)
        
        # Add layer label
        ax.text(x + box_width/2, y + box_height/2, f'Layer {i}',
                ha='center', va='center', fontsize=12, color='#222')
    
    # Set plot properties
    ax.set_xlim(-0.5, total_layers * (box_width + spacing) - spacing + 0.5)
    ax.set_ylim(-0.5, 1.2)
    ax.axis('off')
    
    return fig

def simulate_inference_time(selected_layers, total_layers, is_circuitized, is_first_run):
    base_times = {0: 0.15, 1: 0.12, 2: 0.08, 3: 0.06, 4: 0.04}
    if is_circuitized:
        multiplier = 2.5 if is_first_run else 1.2
    else:
        multiplier = 1.0
    total_time = 0
    for layer in selected_layers:
        base_time = base_times[layer] * multiplier
        variation = random.uniform(-0.2, 0.2)
        total_time += base_time * (1 + variation)
    return total_time

def main():
    st.set_page_config(layout="wide")
    
    # Title and description
    st.title("JIT EZKL Model Visualization")
    st.markdown("""
    This interface allows you to:
    1. Select the number of layers to circuitize
    2. Set the number of agent runs
    3. View the security level and its formula
    4. See the model visualization
    5. Compare performance metrics
    """)
    
    # Load test results
    results = load_test_results()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Layer selection slider
        total_layers = 5  # Total number of layers in the model
        num_selected = st.slider(
            "Number of layers to circuitize",
            min_value=0,
            max_value=total_layers,
            value=2,
            help="Select how many layers to circuitize"
        )
        agent_runs = st.number_input("Number of agent runs", min_value=1, value=3, step=1)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        selected_layers = get_random_layers(seed, total_layers, num_selected) if num_selected > 0 else []
        
        # Calculate security level
        security_level = calculate_security_level(num_selected, agent_runs)
        st.metric("Security Level", f"{security_level:.2f}%")
        st.markdown(r"""
        **Security Level Formula:**
        $$(1 - \frac{1}{2^{(\text{agent runs} \times \text{layers})}}) \times 100\%$$
        """)
        st.markdown(f"**Selected layers (random by seed):** {selected_layers}")
        
        # Create and display model visualization
        fig = create_model_visualization(selected_layers, total_layers)
        st.pyplot(fig)
        
        # Legend
        st.markdown("""
        **Legend:**
        - <span style='color:#4F8BC9;font-weight:bold'>&#9651; Triangle</span>: Convolutional Layer
        - <span style='color:#7D6B91;font-weight:bold'>&#9632; Rectangle</span>: Fully Connected Layer
        <br>
        <span style='color:#4F8BC9;font-weight:bold'>Filled</span>: Circuitized (selected)
        <span style='color:#BFD7ED;font-weight:bold'>Unfilled</span>: Not circuitized (conv)
        <span style='color:#7D6B91;font-weight:bold'>Filled</span>: Circuitized (FC)
        <span style='color:#D6CDEA;font-weight:bold'>Unfilled</span>: Not circuitized (FC)
        """, unsafe_allow_html=True)
    
    with col2:
        # Performance metrics
        st.subheader("Performance Metrics")
        
        # Simulate times for current selection
        col_a, col_b = st.columns(2)
        for label, is_first in zip(["First Run", "Cached Run"], [True, False]):
            with (col_a if is_first else col_b):
                st.markdown(f"**{label}**")
                circuitized_time = simulate_inference_time(selected_layers, total_layers, True, is_first) if selected_layers else 0.0
                inference_layers = [i for i in range(total_layers) if i not in selected_layers]
                inference_time = simulate_inference_time(inference_layers, total_layers, False, is_first) if inference_layers else 0.0
                total_time = circuitized_time + inference_time
                st.metric("Total Time", f"{total_time:.3f}s")
                st.metric("Circuitized Layers", f"{circuitized_time:.3f}s")
                st.metric("Regular Inference", f"{inference_time:.3f}s")
        
        # Add space for future agent runner
        st.markdown("---")
        st.subheader("Agent Runner")
        st.markdown("""
        Space reserved for agent runner implementation.
        This area will be used to display agent-related controls and metrics.
        """)

if __name__ == "__main__":
    main() 