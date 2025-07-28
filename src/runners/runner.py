"""
Runner for EzKL Circuit and ONNX Inference
"""

import json
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import src.runners.onnx_runner as onnx_runner
import src.runners.ezkl_runner as ezkl_runner

class Runner:
    def __init__(self, slice_output_dir: str):
        self.slice_output_dir = Path(slice_output_dir)
        self.run_metadata_path = self.slice_output_dir / "run_metadata.json"
        
        if not self.run_metadata_path.exists():
            raise FileNotFoundError(f"run_metadata.json not found at {self.run_metadata_path}")
        
        with open(self.run_metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def run(self, input_tensor: torch.Tensor) -> dict:
        """Run inference through the chain of segments."""
        execution_chain = self.metadata.get("execution_chain", {})
        current_slice_id = execution_chain.get("head")
        current_tensor = input_tensor
        slice_results = {}
        
        # Chain execution
        while current_slice_id:
            slice_node = execution_chain["nodes"][current_slice_id]
            segment_dir = self.slice_output_dir / current_slice_id
            segment_dir.mkdir(exist_ok=True)
            
            # Write input for this segment
            input_file = segment_dir / "input.json"
            onnx_runner.write_input(current_tensor, str(input_file))
            
            # Execute segment based on circuit availability
            if slice_node.get("use_circuit", False):
                slice_info = self.metadata["slices"][current_slice_id]
                current_tensor, execution_info = self._run_ezkl_segment(
                    slice_info, current_tensor, str(segment_dir)
                )
                slice_results[current_slice_id] = execution_info
            else:
                current_tensor = self._run_onnx_segment(
                    slice_node["fallback"], current_tensor, str(segment_dir)
                )
                slice_results[current_slice_id] = {
                    "method": "onnx_only",
                    "attempted_ezkl": False
                }
            
            # Read output and prepare for next segment
            output_file = segment_dir / "output.json"
            if output_file.exists():
                current_tensor = onnx_runner.read_output(str(output_file))
            
            current_slice_id = slice_node.get("next")
        
        # Final processing
        probabilities = F.softmax(current_tensor, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        
        results = {
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
            "tensor_shape": list(current_tensor.shape),
            "slice_results": slice_results
        }
        
        # Save inference output
        self._save_inference_output(results)
        
        return results

    def _run_onnx_segment(self, onnx_path: str, input_tensor: torch.Tensor, segment_dir: str) -> torch.Tensor:
        """Run ONNX inference for a segment."""
        return onnx_runner.run_slice(onnx_path, input_tensor, segment_dir)

    def _run_ezkl_segment(self, slice_info: dict, input_tensor: torch.Tensor, segment_dir: str) -> tuple:
        """Run EZKL inference for a segment with fallback to ONNX."""
        start_time = time.time()
        
        # Attempt EZKL execution
        output_tensor, exec_info = ezkl_runner.run_slice(slice_info, input_tensor, segment_dir)
        
        # Add timing information
        exec_info["execution_time"] = time.time() - start_time
        if exec_info.get("verified", False):
            exec_info["verification_time"] = f"{exec_info['execution_time']:.3f}s"
        
        return output_tensor, exec_info
    
    def _save_inference_output(self, results: dict):
        """Save inference_output.json with execution details."""
        model_name = self.metadata.get("model_name", "unknown")
        slice_results = results.get("slice_results", {})
        
        # Count execution methods
        ezkl_complete = sum(1 for r in slice_results.values() 
                           if r.get("method") == "ezkl_circuit_complete")
        total_slices = len(slice_results)
        
        # Build execution results
        execution_results = []
        for slice_id, exec_info in slice_results.items():
            result_entry = {
                "slice_id": slice_id,
                "method": exec_info.get("method", "unknown")
            }
            
            # Add EZKL-specific details if available
            if exec_info.get("witness_path"):
                result_entry["witness_path"] = exec_info["witness_path"]
            if exec_info.get("proof_path"):
                result_entry["proof_path"] = exec_info["proof_path"]
            if exec_info.get("verification_time"):
                result_entry["verification_time"] = exec_info["verification_time"]
            
            execution_results.append(result_entry)
        
        # Calculate security percentage
        security_percent = (ezkl_complete / total_slices * 100) if total_slices > 0 else 0
        
        # Build output structure
        inference_output = {
            "model_name": model_name,
            "prediction": results["prediction"],
            "probabilities": results["probabilities"],
            "execution_chain": {
                "total_slices": total_slices,
                "ezkl_verified_slices": ezkl_complete,
                "overall_security": f"{security_percent:.1f}%",
                "execution_results": execution_results
            },
            "performance_comparison": {
                "note": "Full ONNX vs verified chain comparison would require separate pure ONNX run"
            }
        }
        
        # Save to file
        output_path = self.slice_output_dir / "inference_output.json"
        with open(output_path, 'w') as f:
            json.dump(inference_output, f, indent=2)
        
        print(f"Inference output saved to: {output_path}")

def main():
    """Test runner with different models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on sliced models")
    parser.add_argument("--model", type=int, default=2, choices=[1, 2, 3],
                       help="Model choice: 1=doom, 2=net, 3=resnet")
    args = parser.parse_args()
    
    # Model configurations
    model_configs = {
        1: ("kubz/output/doom_slices", "kubz/src/models/doom/input.json"),
        2: ("kubz/output/net_slices", "kubz/src/models/net/input.json"),
        3: ("kubz/output/resnet_slices", "kubz/src/models/resnet/input.json")
    }
    
    slice_output_dir, input_json_path = model_configs[args.model]
    
    # Check if sliced model exists
    if not Path(slice_output_dir).exists():
        print(f"Sliced model not found at {slice_output_dir}")
        print("Please run slicing first or check the path")
        return
    
    # Initialize runner
    runner = Runner(slice_output_dir)
    
    # Load input
    input_tensor = onnx_runner.read_input(input_json_path)
    
    # Reshape based on metadata
    expected_shape = runner.metadata.get("input_shape", [])
    if expected_shape and isinstance(expected_shape[0], list):
        shape = [1] + expected_shape[0][1:]
        input_tensor = input_tensor.reshape(shape)
    
    # Run inference
    print(f"Running inference on model...")
    results = runner.run(input_tensor)
    
    # Display results
    print(f"\nPrediction: {results['prediction']}")
    print(f"Execution summary:")
    for slice_id, info in results["slice_results"].items():
        print(f"  {slice_id}: {info['method']}")

if __name__ == "__main__":
    main()