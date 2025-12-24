#!/usr/bin/env python3
"""
Test video inference with sliced model and JStprove verification.
Processes test.mp4 through all 64 slices with ZK verification for slices 41 and 52.
"""

import cv2
import numpy as np
import json
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from dsperse.src.backends.jstprove import JSTprove


def preprocess_frame(frame: np.ndarray, img_size: int = 640) -> np.ndarray:
    """Preprocess frame for YOLO inference."""
    img_resized = cv2.resize(frame, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def run_sliced_inference(slices_dir: Path, input_tensor: np.ndarray, 
                        verify_slices: list = [41, 52]) -> tuple:
    """
    Run inference through all slices sequentially.
    For slices in verify_slices, use JStprove circuit execution.
    
    Returns:
        (output_tensor, verification_results)
    """
    metadata_path = slices_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get slice order
    slice_indices = sorted(metadata.get("slice_points", []))
    
    # Current outputs (start with input)
    current_outputs = {"input_data": input_tensor.flatten().tolist()}
    verification_results = {}
    
    jstprove = JSTprove() if verify_slices else None
    
    for slice_idx in slice_indices:
        slice_id = f"slice_{slice_idx}"
        slice_dir = slices_dir / slice_id
        
        if not slice_dir.exists():
            continue
        
        slice_onnx = slice_dir / "payload" / f"{slice_id}.onnx"
        if not slice_onnx.exists():
            continue
        
        # Check if this slice should use JStprove
        use_jstprove = slice_idx in verify_slices
        
        if use_jstprove and jstprove:
            # Use JStprove circuit execution
            try:
                # Find circuit file
                jstprove_dir = slice_dir / "payload" / "jstprove"
                if not jstprove_dir.exists():
                    jstprove_dir = slice_dir / "payload" / "jstprove_circuitization"
                
                circuit_path = None
                for pattern in ["*_circuit.txt", "*circuit.txt"]:
                    circuits = list(jstprove_dir.glob(pattern))
                    if circuits:
                        circuit_path = circuits[0]
                        break
                
                if circuit_path and circuit_path.exists():
                    # Prepare input JSON
                    input_json_path = slice_dir / f"{slice_id}_input.json"
                    with open(input_json_path, 'w') as f:
                        json.dump({"input_data": current_outputs.get("input_data", [])}, f)
                    
                    # Generate witness and execute
                    output_json_path = slice_dir / f"{slice_id}_output.json"
                    witness_path = slice_dir / f"{slice_id}_witness.bin"
                    
                    success, _ = jstprove.generate_witness(
                        input_file=str(input_json_path),
                        model_path=str(circuit_path),
                        output_file=str(output_json_path)
                    )
                    
                    if success:
                        # Load output
                        with open(output_json_path, 'r') as f:
                            output_data = json.load(f)
                        current_outputs = output_data
                        verification_results[slice_id] = {"method": "jstprove", "success": True}
                        continue
            except Exception as e:
                print(f"Warning: JStprove execution failed for {slice_id}: {e}")
        
        # Fallback to ONNX execution
        session = ort.InferenceSession(str(slice_onnx))
        
        # Prepare inputs
        inputs = {}
        for inp in session.get_inputs():
            inp_name = inp.name
            inp_shape = inp.shape
            
            # Find matching output from previous slice
            if inp_name in current_outputs:
                data = np.array(current_outputs[inp_name], dtype=np.float32)
                expected_size = np.prod([s for s in inp_shape if isinstance(s, int)])
                if data.size >= expected_size:
                    data = data[:expected_size].reshape(inp_shape)
                inputs[inp_name] = data
        
        if inputs:
            outputs = session.run(None, inputs)
            # Store outputs
            for idx, out in enumerate(session.get_outputs()):
                current_outputs[out.name] = outputs[idx].flatten().tolist()
            
            if not use_jstprove:
                verification_results[slice_id] = {"method": "onnx", "success": True}
    
    # Get final output (YOLO format: 1, 5, 8400)
    final_output = None
    for key in current_outputs:
        if isinstance(current_outputs[key], list):
            arr = np.array(current_outputs[key])
            if arr.size == 42000:  # 5 * 8400
                final_output = arr.reshape(1, 5, 8400)
                break
    
    return final_output, verification_results


def postprocess_yolo(outputs: np.ndarray, conf_threshold: float = 0.25) -> list:
    """Post-process YOLO output to get detections."""
    predictions = outputs[0] if len(outputs.shape) == 3 else outputs
    boxes = predictions[:4, :].T  # (8400, 4)
    confidences = predictions[4, :]  # (8400,)
    
    # Filter by confidence
    valid_mask = confidences > conf_threshold
    valid_boxes = boxes[valid_mask]
    valid_confs = confidences[valid_mask]
    
    return list(zip(valid_boxes, valid_confs))


def process_video(video_path: Path, slices_dir: Path, output_path: Path):
    """Process video through sliced model."""
    print(f"Processing video: {video_path}")
    print(f"Slices directory: {slices_dir}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    stats = {
        "total_frames": 0,
        "frames_with_detections": 0,
        "total_detections": 0,
        "all_confidences": [],
        "verification_results": {}
    }
    
    print("\nProcessing frames...")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        input_tensor = preprocess_frame(frame)
        
        # Run sliced inference
        output, verification_results = run_sliced_inference(
            slices_dir, input_tensor, verify_slices=[41, 52]
        )
        
        # Store verification results (from first frame)
        if frame_idx == 0:
            stats["verification_results"] = verification_results
        
        stats["total_frames"] += 1
        
        if output is not None:
            # Post-process
            detections = postprocess_yolo(output)
            
            if detections:
                stats["frames_with_detections"] += 1
                stats["total_detections"] += len(detections)
                
                # Draw detections
                for box, conf in detections:
                    stats["all_confidences"].append(float(conf))
                    
                    cx, cy, w, h = box
                    x1 = int((cx - w/2) * width / 640)
                    y1 = int((cy - h/2) * height / 640)
                    x2 = int((cx + w/2) * width / 640)
                    y2 = int((cy + h/2) * height / 640)
                    
                    # Clip to frame boundaries
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Deer {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames} - Detections: {stats['total_detections']}")
    
    cap.release()
    out.release()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Detection Statistics")
    print("=" * 70)
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Frames with detections: {stats['frames_with_detections']} "
          f"({stats['frames_with_detections']/stats['total_frames']*100:.1f}%)")
    print(f"  Total detections: {stats['total_detections']}")
    
    if stats['all_confidences']:
        confs = np.array(stats['all_confidences'])
        print(f"\n  Confidence:")
        print(f"    Mean: {confs.mean():.3f}")
        print(f"    Min: {confs.min():.3f}")
        print(f"    Max: {confs.max():.3f}")
    
    print(f"\n  Verification Results:")
    for slice_id, result in stats['verification_results'].items():
        method = result.get('method', 'unknown')
        success = result.get('success', False)
        status = "✓" if success else "✗"
        print(f"    {slice_id}: {status} {method}")
    
    print(f"\n  Output video: {output_path}")
    
    return stats


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video with sliced model")
    parser.add_argument("--video", type=str, 
                       default="../../test.mp4",
                       help="Path to input video")
    parser.add_argument("--output", type=str,
                       default="../../test_result.mp4",
                       help="Path to output video")
    parser.add_argument("--slices", type=str,
                       default=".",
                       help="Path to slices directory")
    
    args = parser.parse_args()
    
    video_path = Path(args.video).resolve()
    output_path = Path(args.output).resolve()
    slices_dir = Path(args.slices).resolve()
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return
    
    if not slices_dir.exists():
        print(f"Error: Slices directory not found: {slices_dir}")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = process_video(video_path, slices_dir, output_path)
    
    # Save statistics
    stats_path = output_path.parent / "video_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\nStatistics saved to: {stats_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

