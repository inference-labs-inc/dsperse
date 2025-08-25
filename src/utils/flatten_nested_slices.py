#!/usr/bin/env python3
"""
Flatten Nested Slices
Flattens the nested slice structure and renames segments to segment_X_Y format.
Updates metadata to chain the segments together.
"""

import os
import json
import shutil
import glob
from typing import Dict, List, Any

def flatten_nested_slices(nested_slices_dir: str, output_dir: str):
    """
    Flatten the nested slices structure and rename segments.
    
    Args:
        nested_slices_dir: Directory containing nested slices
        output_dir: Output directory for flattened segments
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 Flattening nested slices structure...")
    print(f"📁 Input: {nested_slices_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Find all nested directories
    nested_dirs = sorted([d for d in os.listdir(nested_slices_dir) if d.startswith('nested_')])
    
    all_segments = []
    segment_mapping = {}
    
    for nested_dir in nested_dirs:
        segment_num = nested_dir.split('_')[1]
        nested_path = os.path.join(nested_slices_dir, nested_dir)
        
        # Find all nested segments in this directory
        nested_segment_dirs = sorted([d for d in os.listdir(nested_path) if d.startswith('nested_segment_')])
        
        for nested_segment_dir in nested_segment_dirs:
            nested_segment_num = nested_segment_dir.split('_')[2]
            source_path = os.path.join(nested_path, nested_segment_dir, f"{nested_segment_dir}.onnx")
            
            if os.path.exists(source_path):
                # Create new name: segment_X_Y
                new_name = f"segment_{segment_num}_{nested_segment_num}"
                dest_path = os.path.join(output_dir, f"{new_name}.onnx")
                
                # Copy the file
                shutil.copy2(source_path, dest_path)
                
                # Store mapping and segment info
                segment_mapping[f"{segment_num}_{nested_segment_num}"] = {
                    "original_nested": nested_dir,
                    "original_segment": nested_segment_dir,
                    "source_path": source_path,
                    "dest_path": dest_path
                }
                
                all_segments.append({
                    "name": new_name,
                    "path": dest_path,
                    "original_parent": segment_num,
                    "original_nested": nested_segment_num
                })
                
                print(f"   📄 {nested_dir}/{nested_segment_dir} → {new_name}")
    
    print(f"\n✅ Flattened {len(all_segments)} segments")
    return all_segments, segment_mapping

def create_chained_metadata(all_segments: List[Dict], output_dir: str, original_metadata_path: str = None):
    """
    Create metadata that chains the flattened segments together.
    
    Args:
        all_segments: List of all flattened segments
        output_dir: Output directory for metadata
        original_metadata_path: Path to original metadata for reference
    """
    
    # Create chained metadata structure
    chained_metadata = {
        "model_type": "ONNX_FFT_CHAINED",
        "description": "ResNet model with FFT-decomposed convolutions, flattened and chained segments",
        "total_segments": len(all_segments),
        "segments": [],
        "chain_order": [],
        "fft_decomposition_info": {
            "description": "Each Conv → FFT decomposition split into granular segments",
            "transformation": "Conv → DFT → Mul → DFT",
            "segmenting_strategy": "Flattened nested structure with chained execution"
        }
    }
    
    # Group segments by original parent
    segments_by_parent = {}
    for segment in all_segments:
        parent = segment["original_parent"]
        if parent not in segments_by_parent:
            segments_by_parent[parent] = []
        segments_by_parent[parent].append(segment)
    
    # Create segment metadata and chain order
    segment_index = 0
    for parent_num in sorted(segments_by_parent.keys()):
        parent_segments = sorted(segments_by_parent[parent_num], key=lambda x: int(x["original_nested"]))
        
        for segment in parent_segments:
            # Load ONNX model to get input/output info
            try:
                import onnx
                model = onnx.load(segment["path"])
                
                # Extract input/output shapes
                input_shapes = []
                for input_info in model.graph.input:
                    shape = []
                    for dim in input_info.type.tensor_type.shape.dim:
                        if dim.dim_value > 0:
                            shape.append(dim.dim_value)
                        else:
                            shape.append("batch_size")
                    input_shapes.append(shape)
                
                output_shapes = []
                for output_info in model.graph.output:
                    shape = []
                    for dim in output_info.type.tensor_type.shape.dim:
                        if dim.dim_value > 0:
                            shape.append(dim.dim_value)
                        else:
                            shape.append("batch_size")
                    output_shapes.append(shape)
                
                # Get node types
                node_types = [node.op_type for node in model.graph.node]
                
                segment_metadata = {
                    "index": segment_index,
                    "name": segment["name"],
                    "path": segment["path"],
                    "original_parent": segment["original_parent"],
                    "original_nested": segment["original_nested"],
                    "input_shapes": input_shapes,
                    "output_shapes": output_shapes,
                    "node_types": node_types,
                    "node_count": len(model.graph.node),
                    "execution_order": segment_index
                }
                
                chained_metadata["segments"].append(segment_metadata)
                chained_metadata["chain_order"].append(segment["name"])
                
                segment_index += 1
                
            except Exception as e:
                print(f"   ⚠️  Warning: Could not analyze {segment['name']}: {e}")
                # Add basic metadata without analysis
                segment_metadata = {
                    "index": segment_index,
                    "name": segment["name"],
                    "path": segment["path"],
                    "original_parent": segment["original_parent"],
                    "original_nested": segment["original_nested"],
                    "error": f"Could not analyze: {e}"
                }
                chained_metadata["segments"].append(segment_metadata)
                chained_metadata["chain_order"].append(segment["name"])
                segment_index += 1
    
    # Save chained metadata
    metadata_path = os.path.join(output_dir, "chained_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(chained_metadata, f, indent=4)
    
    print(f"📋 Created chained metadata: {metadata_path}")
    return chained_metadata

def main():
    """Main function to flatten nested slices and create chained metadata."""
    nested_slices_dir = "./src/models/resnet/nested_slices"
    output_dir = "./src/models/resnet/flattened_slices"
    
    print("🔪 Flatten Nested Slices - Creating chained segment structure")
    print("=" * 70)
    
    # Check if nested slices exist
    if not os.path.exists(nested_slices_dir):
        print(f"❌ Error: Nested slices directory not found: {nested_slices_dir}")
        print("Please run the nested FFT slicer first.")
        return
    
    # Flatten the nested structure
    all_segments, segment_mapping = flatten_nested_slices(nested_slices_dir, output_dir)
    
    if not all_segments:
        print("❌ No segments found to flatten!")
        return
    
    # Create chained metadata
    chained_metadata = create_chained_metadata(all_segments, output_dir)
    
    # Print summary
    print(f"\n🎉 Flattening complete!")
    print(f"📊 Summary:")
    print(f"   Total segments: {len(all_segments)}")
    print(f"   Output directory: {output_dir}")
    print(f"   Metadata file: chained_metadata.json")
    
    # Show chain order
    print(f"\n🔗 Execution Chain Order:")
    for i, segment_name in enumerate(chained_metadata["chain_order"]):
        print(f"   {i}: {segment_name}")
    
    print(f"\n📁 Check output in: {output_dir}")

if __name__ == "__main__":
    main()
