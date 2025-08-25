#!/usr/bin/env python3
"""
FFT Metadata Generator
Generates metadata.json for FFT-decomposed models by adapting the original metadata
to reflect the transformation from Conv nodes to DFT → Mul → DFT operations.
Now supports flattened nested segments with names like segment_0_1, segment_0_2, etc.
"""

import json
import os
import onnx
from typing import Dict, List, Any
from pathlib import Path

class FFTMetadataGenerator:
    """
    Generates metadata for FFT-decomposed models by analyzing the original metadata
    and the FFT-decomposed ONNX files to create updated metadata.
    Supports both regular FFT segments and flattened nested segments.
    """
    
    def __init__(self, original_metadata_path: str, fft_models_dir: str, nested_slices_dir: str = None):
        """
        Initialize with paths to original metadata and FFT-decomposed models.
        
        Args:
            original_metadata_path: Path to original slices metadata.json
            fft_models_dir: Directory containing FFT-decomposed ONNX files
            nested_slices_dir: Directory containing flattened nested slices (optional)
        """
        self.original_metadata_path = original_metadata_path
        self.fft_models_dir = fft_models_dir
        self.nested_slices_dir = nested_slices_dir
        self.original_metadata = None
        self.fft_metadata = None
        
    def load_original_metadata(self) -> Dict[str, Any]:
        """Load the original metadata.json file."""
        with open(self.original_metadata_path, 'r') as f:
            self.original_metadata = json.load(f)
        return self.original_metadata
    
    def analyze_fft_model(self, model_path: str) -> Dict[str, Any]:
        """
        Analyze an FFT-decomposed ONNX model to extract node information.
        
        Args:
            model_path: Path to the FFT-decomposed ONNX file
            
        Returns:
            Dict containing analysis of the FFT model
        """
        model = onnx.load(model_path)
        graph = model.graph
        
        # Count different node types
        node_counts = {}
        dft_nodes = []
        conv_nodes = []
        
        for node in graph.node:
            op_type = node.op_type
            node_counts[op_type] = node_counts.get(op_type, 0) + 1
            
            if op_type == "DFT":
                dft_nodes.append({
                    "name": node.name,
                    "inverse": next((attr.i for attr in node.attribute if attr.name == "inverse"), 0),
                    "onesided": next((attr.i for attr in node.attribute if attr.name == "onesided"), 0)
                })
            elif op_type == "Conv":
                conv_nodes.append(node.name)
        
        return {
            "node_counts": node_counts,
            "dft_nodes": dft_nodes,
            "conv_nodes": conv_nodes,
            "total_nodes": len(graph.node),
            "has_fft_decomposition": len(dft_nodes) > 0
        }
    
    def transform_segment_metadata(self, original_segment: Dict[str, Any], fft_analysis: Dict[str, Any], segment_suffix: str = "") -> Dict[str, Any]:
        """
        Transform a segment's metadata to reflect FFT decomposition.
        
        Args:
            original_segment: Original segment metadata
            fft_analysis: Analysis of the corresponding FFT model
            segment_suffix: Suffix to add to segment names (e.g., "_1", "_2" for nested segments)
            
        Returns:
            Transformed segment metadata
        """
        # Start with a copy of the original segment
        transformed = original_segment.copy()
        
        # Update the path to point to FFT model
        segment_index = transformed["index"]
        if segment_suffix and not transformed.get("filename", "").startswith("segment_"):
            # Only add suffix if filename doesn't already have the correct format
            transformed["filename"] = f"segment_{segment_index}{segment_suffix}.onnx"
            transformed["path"] = os.path.join(self.fft_models_dir, f"segment_{segment_index}{segment_suffix}.onnx")
        else:
            # Use the filename as-is (already correctly formatted)
            transformed["path"] = os.path.join(self.fft_models_dir, transformed["filename"])
        
        # Add FFT-specific information
        transformed["fft_decomposition"] = {
            "has_fft": fft_analysis["has_fft_decomposition"],
            "dft_node_count": len(fft_analysis["dft_nodes"]),
            "conv_node_count": len(fft_analysis["conv_nodes"]),
            "total_nodes": fft_analysis["total_nodes"],
            "node_type_breakdown": fft_analysis["node_counts"]
        }
        
        # Transform layers to reflect FFT decomposition
        if fft_analysis["has_fft_decomposition"]:
            transformed["layers"] = self._transform_layers_for_fft(transformed["layers"], fft_analysis)
        
        # Update parameters count if needed (FFT operations don't add parameters)
        # But we might want to note the transformation
        if fft_analysis["has_fft_decomposition"]:
            transformed["fft_transformation"] = "Conv → DFT → Mul → DFT"
        
        return transformed
    
    def _transform_layers_for_fft(self, original_layers: List[Dict], fft_analysis: Dict) -> List[Dict]:
        """
        Transform layer metadata to reflect FFT decomposition.
        
        Args:
            original_layers: Original layers metadata
            fft_analysis: FFT analysis results
            
        Returns:
            Transformed layers metadata
        """
        transformed_layers = []
        
        for layer in original_layers:
            if layer["type"] == "Conv":
                # Replace Conv layer with FFT decomposition layers
                fft_layer = {
                    "name": f"{layer['name']}_fft",
                    "type": "DFT",
                    "activation": "FFT (Forward)",
                    "parameter_details": {
                        "fft_type": "forward",
                        "original_conv": layer["name"],
                        "transformation": "Conv → FFT"
                    }
                }
                
                mul_layer = {
                    "name": f"{layer['name']}_freq_mult",
                    "type": "Mul",
                    "activation": "Frequency Multiplication",
                    "parameter_details": {
                        "operation": "element_wise_multiplication",
                        "domain": "frequency",
                        "original_conv": layer["name"]
                    }
                }
                
                ifft_layer = {
                    "name": f"{layer['name']}_ifft",
                    "type": "DFT",
                    "activation": "FFT (Inverse)",
                    "parameter_details": {
                        "fft_type": "inverse",
                        "original_conv": layer["name"],
                        "transformation": "FFT → Spatial"
                    }
                }
                
                transformed_layers.extend([fft_layer, mul_layer, ifft_layer])
            else:
                # Keep non-conv layers as-is
                transformed_layers.append(layer)
        
        return transformed_layers
    
    def generate_fft_metadata(self) -> Dict[str, Any]:
        """
        Generate complete metadata for FFT-decomposed models.
        
        Returns:
            Complete FFT metadata dictionary
        """
        if not self.original_metadata:
            self.load_original_metadata()
        
        # Start with the original metadata structure
        fft_metadata = self.original_metadata.copy()
        
        # Update the model path and add FFT information
        fft_metadata["original_model"] = os.path.join(self.fft_models_dir, "fft_decomposed_resnet.onnx")
        fft_metadata["model_type"] = "ONNX_FFT"
        fft_metadata["fft_decomposition_info"] = {
            "description": "ResNet model with convolution layers decomposed into FFT operations",
            "transformation": "Conv → DFT → Mul → DFT",
            "opset_version": 20,
            "fft_operator": "ONNX DFT operator (opset 20+)"
        }
        
        # Transform each segment
        transformed_segments = []
        for segment in fft_metadata["segments"]:
            segment_index = segment["index"]
            
            # Check if we have nested slices for this segment
            if self.nested_slices_dir:
                nested_dir = os.path.join(self.nested_slices_dir, f"segment_{segment_index}_nested")
                if os.path.exists(nested_dir):
                    # Process nested segments
                    nested_segments = self._process_nested_segments(segment, nested_dir, segment_index)
                    transformed_segments.extend(nested_segments)
                    continue
            
            # Process regular FFT segment
            fft_model_path = os.path.join(self.fft_models_dir, f"fft_segment_{segment_index}.onnx")
            
            if os.path.exists(fft_model_path):
                # Analyze the FFT model
                fft_analysis = self.analyze_fft_model(fft_model_path)
                
                # Transform the segment metadata
                transformed_segment = self.transform_segment_metadata(segment, fft_analysis)
                transformed_segments.append(transformed_segment)
            else:
                # If FFT model doesn't exist, keep original but mark as missing
                segment["fft_decomposition"] = {
                    "has_fft": False,
                    "error": "FFT model not found"
                }
                transformed_segments.append(segment)
        
        fft_metadata["segments"] = transformed_segments
        
        return fft_metadata
    
    def _process_nested_segments(self, original_segment: Dict[str, Any], nested_dir: str, segment_index: int) -> List[Dict[str, Any]]:
        """
        Process nested segments for a given original segment.
        
        Args:
            original_segment: Original segment metadata
            nested_dir: Directory containing nested segments
            segment_index: Index of the original segment
            
        Returns:
            List of transformed nested segment metadata
        """
        nested_segments = []
        
        # Find all nested segment files
        nested_files = [f for f in os.listdir(nested_dir) if f.endswith('.onnx') and f.startswith('segment_')]
        nested_files.sort()  # Ensure consistent ordering
        
        for nested_file in nested_files:
            nested_segment_num = nested_file.split('_')[1].split('.')[0]
            nested_model_path = os.path.join(nested_dir, nested_file)
            
            try:
                # Analyze the nested FFT model
                fft_analysis = self.analyze_fft_model(nested_model_path)
                
                # Create nested segment metadata
                nested_segment = original_segment.copy()
                nested_segment["index"] = f"{segment_index}_{nested_segment_num}"
                nested_segment["filename"] = f"segment_{segment_index}_{nested_segment_num}.onnx"
                nested_segment["path"] = nested_model_path
                nested_segment["parent_segment"] = segment_index
                nested_segment["nested_level"] = int(nested_segment_num)
                
                # Transform the segment metadata
                transformed_nested_segment = self.transform_segment_metadata(
                    nested_segment, fft_analysis, f"_{nested_segment_num}"
                )
                
                nested_segments.append(transformed_nested_segment)
                
            except Exception as e:
                print(f"Warning: Could not process nested segment {nested_file}: {e}")
                continue
        
        return nested_segments
    
    def save_fft_metadata(self, output_path: str = None) -> str:
        """
        Save the generated FFT metadata to a file.
        
        Args:
            output_path: Path to save the metadata (defaults to FFT models directory)
            
        Returns:
            Path where metadata was saved
        """
        if output_path is None:
            output_path = os.path.join(self.fft_models_dir, "metadata.json")
        
        fft_metadata = self.generate_fft_metadata()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(fft_metadata, f, indent=4)
        
        return output_path

def main():
    """Main function to generate FFT metadata."""
    # Paths
    original_metadata_path = "src/models/resnet/slices/metadata.json"
    fft_models_dir = "src/models/resnet/FFT_cov"
    nested_slices_dir = "src/models/resnet/flattened_nested_slices"
    
    # Check if paths exist
    if not os.path.exists(original_metadata_path):
        print(f"Error: Original metadata not found at {original_metadata_path}")
        return
    
    if not os.path.exists(fft_models_dir):
        print(f"Error: FFT models directory not found at {fft_models_dir}")
        return
    
    # Generate FFT metadata
    generator = FFTMetadataGenerator(original_metadata_path, fft_models_dir, nested_slices_dir)
    
    try:
        output_path = generator.save_fft_metadata()
        print(f"✅ FFT metadata generated successfully!")
        print(f"📁 Saved to: {output_path}")
        
        # Print summary
        fft_metadata = generator.generate_fft_metadata()
        total_segments = len(fft_metadata["segments"])
        fft_segments = sum(1 for seg in fft_metadata["segments"] 
                          if seg.get("fft_decomposition", {}).get("has_fft", False))
        
        print(f"\n📊 Summary:")
        print(f"   Total segments: {total_segments}")
        print(f"   FFT-decomposed: {fft_segments}")
        print(f"   Model type: {fft_metadata['model_type']}")
        
        # Check for nested segments
        nested_segments = [seg for seg in fft_metadata["segments"] if "parent_segment" in seg]
        if nested_segments:
            print(f"   Nested segments: {len(nested_segments)}")
            print(f"   Parent segments: {len(set(seg['parent_segment'] for seg in nested_segments))}")
        
    except Exception as e:
        print(f"❌ Error generating FFT metadata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
