#!/usr/bin/env python3
"""
Simple Multiply Circuit for Segment 5 (EZKL)
Creates a basic multiply circuit that EZKL can handle for the smaller segment_5 model.
"""
import os
import numpy as np
import onnx
import onnx_graphsurgeon as gs

def create_simple_multiply_circuit_segment5():
    """Create a simple multiply circuit for segment_5 that EZKL can handle."""
    
    # Input shapes: [N, C, H, W] - using smaller dimensions for segment_5
    input_shape = [1, 64, 56, 56]  # Typical for later ResNet layers
    
    # Create input variables
    input_real = gs.Variable("input_real", dtype=np.float32, shape=input_shape)
    input_imag = gs.Variable("input_imag", dtype=np.float32, shape=input_shape)
    
    # Create kernel constants (simplified for testing)
    kernel_real = gs.Constant("kernel_real", np.ones(input_shape, dtype=np.float32) * 0.1)
    kernel_imag = gs.Constant("kernel_imag", np.ones(input_shape, dtype=np.float32) * 0.1)
    
    # Create output variables
    output_real = gs.Variable("output_real", dtype=np.float32, shape=input_shape)
    output_imag = gs.Variable("output_imag", dtype=np.float32, shape=input_shape)
    
    # Create a simple graph with direct operations
    graph = gs.Graph(opset=18, inputs=[input_real, input_imag], outputs=[output_real, output_imag])
    
    # Add simple multiply operations
    # Real part: input_real * kernel_real - input_imag * kernel_imag
    temp1 = gs.Variable("temp1", dtype=np.float32, shape=input_shape)
    temp2 = gs.Variable("temp2", dtype=np.float32, shape=input_shape)
    
    graph.nodes.append(gs.Node(op="Mul", inputs=[input_real, kernel_real], outputs=[temp1]))
    graph.nodes.append(gs.Node(op="Mul", inputs=[input_imag, kernel_imag], outputs=[temp2]))
    graph.nodes.append(gs.Node(op="Sub", inputs=[temp1, temp2], outputs=[output_real]))
    
    # Imaginary part: input_real * kernel_imag + input_imag * kernel_real
    temp3 = gs.Variable("temp3", dtype=np.float32, shape=input_shape)
    temp4 = gs.Variable("temp4", dtype=np.float32, shape=input_shape)
    
    graph.nodes.append(gs.Node(op="Mul", inputs=[input_real, kernel_imag], outputs=[temp3]))
    graph.nodes.append(gs.Node(op="Mul", inputs=[input_imag, kernel_real], outputs=[temp4]))
    graph.nodes.append(gs.Node(op="Add", inputs=[temp3, temp4], outputs=[output_imag]))
    
    return graph

def main():
    """Create and save the simple multiply circuit for segment_5."""
    output_dir = "src/models/resnet/ezkl_circuits/segment_5"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating simple multiply circuit for segment_5 (EZKL)...")
    
    # Create the circuit
    graph = create_simple_multiply_circuit_segment5()
    
    # Save the circuit
    output_path = os.path.join(output_dir, "simple_multiply_segment5.onnx")
    onnx.save(gs.export_onnx(graph), output_path)
    
    print(f"✅ Simple multiply circuit for segment_5 saved to: {output_path}")
    print(f"   - Inputs: {[inp.name for inp in graph.inputs]}")
    print(f"   - Outputs: {[out.name for out in graph.outputs]}")
    print(f"   - Nodes: {len(graph.nodes)}")
    print(f"   - Input shape: {graph.inputs[0].shape}")

if __name__ == "__main__":
    main()
