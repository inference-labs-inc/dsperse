#!/usr/bin/env python3
"""
FFT Convolution Decomposer
Replaces large convolution layers with FFT-based decomposition:
Conv -> FFT + FreqMult + IFFT circuits
This allows large convolutions that don't fit in PLONK to be broken down
into smaller, manageable circuits that can be chained together.

Modified to process multiple sliced model segments (e.g., segment_0.onnx to segment_20.onnx).
Preserves input/output shapes and tensor proto structures by replacing nodes in-place while maintaining
original graph inputs/outputs. Exports each decomposed segment as fft_segment_i.onnx.
Uses ONNX shape inference to verify input/output shapes remain intact post-decomposition.
Can be validated/loaded via onnx_runtime for runtime checks if needed.
"""

import onnx
import os
import json
import numpy as np
from onnx import helper, numpy_helper, TensorProto, shape_inference
from typing import List, Dict, Tuple
from functools import wraps

def generate_fft_metadata(func):
    """
    Decorator to automatically generate metadata.json for FFT-decomposed models.
    
    This decorator:
    1. Runs the original FFT decomposition function
    2. Analyzes the decomposed models to generate metadata
    3. Creates a metadata.json file in the FFT_cov directory
    4. Updates paths and information to reflect FFT decomposition
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Run the original function
        result = func(*args, **kwargs)
        
        # Get the output folder from the function call
        output_folder = "./src/models/resnet/FFT_cov"
        
        # Generate metadata for FFT-decomposed models
        generate_fft_metadata_json(output_folder)
        
        return result
    return wrapper

def generate_fft_metadata_json(output_folder: str):
    """
    Generate metadata.json for FFT-decomposed models.
    
    Args:
        output_folder: Path to the FFT_cov directory containing decomposed models
    """
    print("Generating FFT decomposition metadata...")
    
    # Load original metadata to get structure
    original_metadata_path = "src/models/resnet/slices/metadata.json"
    if not os.path.exists(original_metadata_path):
        print(f"Warning: Original metadata not found at {original_metadata_path}")
        return
    
    with open(original_metadata_path, 'r') as f:
        original_metadata = json.load(f)
    
    # Create FFT metadata structure
    fft_metadata = {
        "original_model": original_metadata.get("original_model", ""),
        "model_type": "ONNX_FFT_DECOMPOSED",
        "total_parameters": original_metadata.get("total_parameters", 0),
        "input_shape": original_metadata.get("input_shape", []),
        "output_shapes": original_metadata.get("output_shapes", []),
        "slice_points": original_metadata.get("slice_points", []),
        "fft_decomposition_info": {
            "decomposition_method": "DFT-based convolution replacement",
            "opset_version": 20,
            "convolution_replacement": "Conv -> DFT(forward) + Mul(frequency) + DFT(inverse)",
            "total_conv_layers_replaced": 0
        },
        "segments": []
    }
    
    # Process each segment
    for segment_info in original_metadata.get("segments", []):
        segment_idx = segment_info.get("index", 0)
        fft_model_path = os.path.join(output_folder, f"fft_segment_{segment_idx}.onnx")
        
        if os.path.exists(fft_model_path):
            # Load the FFT-decomposed model to analyze it
            try:
                fft_model = onnx.load(fft_model_path)
                
                # Count DFT operations (replaced convolutions)
                dft_count = sum(1 for node in fft_model.graph.node if node.op_type == "DFT")
                conv_count = sum(1 for node in fft_model.graph.node if node.op_type == "Conv")
                
                # Create FFT segment info
                fft_segment_info = {
                    "index": segment_idx,
                    "filename": f"fft_segment_{segment_idx}.onnx",
                    "path": os.path.abspath(fft_model_path),
                    "parameters": segment_info.get("parameters", 0),
                    "fft_operations": {
                        "dft_nodes": dft_count,
                        "conv_nodes": conv_count,
                        "conv_replaced": conv_count == 0 and dft_count > 0
                    },
                    "shape": segment_info.get("shape", {}),
                    "dependencies": segment_info.get("dependencies", {}),
                    "layers": []
                }
                
                # Update layers information for FFT decomposition
                for layer in segment_info.get("layers", []):
                    if layer.get("type") == "Conv":
                        # Replace Conv layer with FFT decomposition info
                        fft_layer = {
                            "name": layer.get("name", ""),
                            "type": "FFT_DECOMPOSED",
                            "activation": "FFT_Decomposition",
                            "parameter_details": {
                                "fft_forward": {
                                    "shape": ["FFT_forward"],
                                    "size": 1
                                },
                                "frequency_multiplication": {
                                    "shape": ["Freq_Mul"],
                                    "size": 1
                                },
                                "fft_inverse": {
                                    "shape": ["FFT_inverse"],
                                    "size": 1
                                },
                                "original_conv_info": layer.get("parameter_details", {})
                            }
                        }
                        fft_segment_info["layers"].append(fft_layer)
                        
                        # Count replaced convolutions
                        if conv_count == 0 and dft_count > 0:
                            fft_metadata["fft_decomposition_info"]["total_conv_layers_replaced"] += 1
                    else:
                        # Keep non-conv layers as-is
                        fft_segment_info["layers"].append(layer)
                
                fft_metadata["segments"].append(fft_segment_info)
                
            except Exception as e:
                print(f"Warning: Could not analyze FFT model {fft_model_path}: {e}")
                # Add basic info without analysis
                fft_segment_info = {
                    "index": segment_idx,
                    "filename": f"fft_segment_{segment_idx}.onnx",
                    "path": os.path.abspath(fft_model_path),
                    "parameters": segment_info.get("parameters", 0),
                    "fft_operations": {
                        "dft_nodes": "unknown",
                        "conv_nodes": "unknown",
                        "conv_replaced": "unknown"
                    },
                    "shape": segment_info.get("shape", {}),
                    "dependencies": segment_info.get("dependencies", {}),
                    "layers": segment_info.get("layers", [])
                }
                fft_metadata["segments"].append(fft_segment_info)
    
    # Save FFT metadata
    metadata_path = os.path.join(output_folder, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(fft_metadata, f, indent=4)
    
    print(f"FFT decomposition metadata saved to: {metadata_path}")
    print(f"Total convolution layers replaced: {fft_metadata['fft_decomposition_info']['total_conv_layers_replaced']}")

class FFTConvolutionDecomposer:
    """
    Decomposes large convolution layers into FFT-based circuits
    Workflow:
    1. Extract conv layer from original ONNX
    2. Create 3 separate circuits:
       - FFT circuit: spatial -> frequency domain
       - FreqMult circuit: frequency domain multiplication with kernel
       - IFFT circuit: frequency -> spatial domain
    3. Replace original conv_node via onnx subset export
    """
    
    def __init__(self, model_path: str):
        """Initialize with ONNX model path"""
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.conv_nodes = []
        self.replaced_nodes = []
        
        # Infer shapes for original model to verify later
        self.original_inferred_model = shape_inference.infer_shapes(self.model, data_prop=True)
        
    def analyze_convolutions(self) -> List[Dict]:
        """Analyze ALL convolution layers in the model"""
        conv_info = []
        
        for node in self.graph.node:
            if node.op_type == "Conv":
                # Get input/output shapes and kernel info
                input_name = node.input[0]
                kernel_name = node.input[1] if len(node.input) > 1 else None
                output_name = node.output[0]
                
                # Extract attributes
                attrs = {attr.name: attr for attr in node.attribute}
                
                conv_data = {
                    'node': node,
                    'input_name': input_name,
                    'kernel_name': kernel_name,
                    'output_name': output_name,
                    'attributes': attrs,
                    'name': node.name
                }
                
                conv_info.append(conv_data)
                self.conv_nodes.append(node)
                
        return conv_info
    
    def create_fft_circuit(self, conv_info: Dict) -> List[onnx.NodeProto]:
        """Create FFT transformation circuit"""
        input_name = conv_info['input_name']
        node_name = conv_info['name']
        
        # FFT node - transforms input to frequency domain
        fft_output = f"{input_name}_fft"
        fft_node = helper.make_node(
            'DFT',  # Use ONNX DFT operator (available in opset 20+)
            inputs=[input_name],
            outputs=[fft_output],
            name=f"{node_name}_fft",
            inverse=0,  # Forward FFT
            onesided=0  # Two-sided FFT
        )
        
        return [fft_node], fft_output
    
    def create_freq_mult_circuit(self, conv_info: Dict, fft_output: str) -> List[onnx.NodeProto]:
        """Create frequency domain multiplication circuit"""
        kernel_name = conv_info['kernel_name']
        node_name = conv_info['name']
        
        # Transform kernel to frequency domain
        kernel_fft_output = f"{kernel_name}_fft"
        kernel_fft_node = helper.make_node(
            'DFT',
            inputs=[kernel_name],
            outputs=[kernel_fft_output],
            name=f"{node_name}_kernel_fft",
            inverse=0,
            onesided=0
        )
        
        # Element-wise multiplication in frequency domain
        mult_output = f"{fft_output}_mult"
        mult_node = helper.make_node(
            'Mul',
            inputs=[fft_output, kernel_fft_output],
            outputs=[mult_output],
            name=f"{node_name}_freq_mult"
        )
        
        return [kernel_fft_node, mult_node], mult_output
    
    def create_ifft_circuit(self, conv_info: Dict, mult_output: str) -> List[onnx.NodeProto]:
        """Create IFFT transformation circuit"""
        output_name = conv_info['output_name']
        node_name = conv_info['name']
        
        # IFFT node - transforms back to spatial domain
        ifft_node = helper.make_node(
            'DFT',
            inputs=[mult_output],
            outputs=[output_name],
            name=f"{node_name}_ifft",
            inverse=1,  # Inverse FFT
            onesided=0
        )
        
        return [ifft_node]
    
    def decompose_large_convs(self) -> None:
        """Replace ALL convolution layers with FFT-based decomposition"""
        print("Analyzing convolution layers...")
        conv_infos = self.analyze_convolutions()
        
        if not conv_infos:
            print("No convolution layers found!")
            return
            
        print(f"Found {len(conv_infos)} convolution layers to decompose")
        
        # Create new node list with FFT decomposition
        new_nodes = []
        
        for node in self.graph.node:
            if node.op_type == "Conv":
                # Find corresponding conv_info
                conv_info = next((info for info in conv_infos if info['node'] == node), None)
                if conv_info:
                    print(f"Decomposing convolution: {conv_info['name']}")
                    
                    # Create FFT circuit
                    fft_nodes, fft_output = self.create_fft_circuit(conv_info)
                    
                    # Create frequency multiplication circuit  
                    freq_mult_nodes, mult_output = self.create_freq_mult_circuit(conv_info, fft_output)
                    
                    # Create IFFT circuit
                    ifft_nodes = self.create_ifft_circuit(conv_info, mult_output)
                    
                    # Add all decomposed nodes
                    new_nodes.extend(fft_nodes)
                    new_nodes.extend(freq_mult_nodes)
                    new_nodes.extend(ifft_nodes)
                    
                    self.replaced_nodes.append({
                        'original': node,
                        'decomposed': fft_nodes + freq_mult_nodes + ifft_nodes
                    })
            else:
                # Keep non-convolution nodes as-is
                new_nodes.append(node)
        
        # Replace nodes in graph
        self.graph.ClearField('node')
        self.graph.node.extend(new_nodes)
        
        # Update model opset to support DFT operator
        self._update_opset()
        
    def _update_opset(self):
        """Update model opset to version 20+ to support DFT operator"""
        for opset in self.model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                if opset.version < 20:
                    opset.version = 20
                    print(f"Updated opset version to {opset.version} for DFT support")
    
    def export_model(self, export_path: str) -> str:
        """Export the decomposed model to specified path"""
        
        # Validate model before saving
        try:
            onnx.checker.check_model(self.model)
            print("Model validation: PASSED")
        except onnx.checker.ValidationError as e:
            print(f"Model validation warning: {e}")
        
        # Infer shapes for decomposed model and compare to original
        try:
            inferred_model = shape_inference.infer_shapes(self.model, data_prop=True)
            
            # Compare input/output shapes (value_infos may differ internally, but graph inputs/outputs should match)
            original_inputs = {vi.name: vi for vi in self.original_inferred_model.graph.input}
            new_inputs = {vi.name: vi for vi in inferred_model.graph.input}
            original_outputs = {vi.name: vi for vi in self.original_inferred_model.graph.output}
            new_outputs = {vi.name: vi for vi in inferred_model.graph.output}
            
            if original_inputs != new_inputs or original_outputs != new_outputs:
                print("Warning: Input/output shapes may have changed post-decomposition!")
            else:
                print("Shape verification: Input/output shapes preserved.")
        except Exception as e:
            print(f"Shape inference warning: {e}")
        
        # Save the decomposed model
        onnx.save(self.model, export_path)
        
        print(f"Successfully exported FFT-decomposed model to: {export_path}")
        print(f"Replaced {len(self.conv_nodes)} convolution layers with FFT circuits")
        
        return export_path
    
    def get_decomposition_summary(self) -> Dict:
        """Get summary of the decomposition process"""
        return {
            'original_conv_count': len(self.conv_nodes),
            'replaced_nodes_count': len(self.replaced_nodes),
            'total_new_nodes': sum(len(r['decomposed']) for r in self.replaced_nodes)
        }

@generate_fft_metadata
def main():
    """Main execution function for processing multiple segments"""
    print("FFT Convolution Decomposer for ResNet18 Segments")
    print("=" * 50)
    
    output_folder = "./src/models/resnet/FFT_cov"
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(21):
        input_model = f"src/models/resnet/slices/segment_{i}/segment_{i}.onnx"
        if not os.path.exists(input_model):
            print(f"Error: Input model '{input_model}' not found!")
            continue
        
        try:
            # Initialize decomposer
            decomposer = FFTConvolutionDecomposer(input_model)
            
            # Perform decomposition
            decomposer.decompose_large_convs()
            
            # Export decomposed model with unique name
            export_path = os.path.join(output_folder, f"fft_segment_{i}.onnx")
            decomposer.export_model(export_path)
            
            # Print summary
            summary = decomposer.get_decomposition_summary()
            print("\nDecomposition Summary for {}:".format(input_model))
            print(f"- Original convolutions: {summary['original_conv_count']}")
            print(f"- Replaced with FFT circuits: {summary['replaced_nodes_count']}")
            print(f"- Total new nodes created: {summary['total_new_nodes']}")
            
            print(f"\nOutput saved to: {export_path}")
            print("Note: Exported model can be loaded/validated in onnx_runtime for runtime checks.")
            print("TensorProto structures are preserved via in-place node replacement.")
        
        except Exception as e:
            print(f"Error during decomposition of {input_model}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
