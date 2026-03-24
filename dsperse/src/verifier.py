"""
Orchestration for verifying proofs.
"""

import os
import json
import time
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.utils.backend_manager import BackendManager

class Verifier:
    """
    Orchestrator for verifying model execution proofs.
    """
    
    def __init__(self, backend=None):
        """
        Initialize the verifier.

        Args:
            backend: Backend name to use for verification ('ezkl' or 'jstprove')
        """
        self.backend = BackendManager.get_backend(backend_name=backend)
    
    def verify_run(self, run_results_path, metadata_path):
        """
        Verify the proofs in a run.
        
        Args:
            run_results_path (str): Path to the run_results.json file
            metadata_path (str): Path to the metadata.json file
            
        Returns:
            dict: Updated run results with verification information
        """
        # Load the run results and metadata
        with open(run_results_path, 'r') as f:
            run_results = json.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        verify_start = time.time()
        # Initialize counters
        verified_segments = 0
        total_ezkl_segments = 0
        
        # Process each segment in the execution results
        for segment in run_results["execution_chain"]["execution_results"]:
            segment_id = segment["segment_id"]
            
            # Skip segments that don't have proof_execution
            if "proof_execution" not in segment:
                continue
            
            # Skip segments where proof generation was not successful
            if not segment["proof_execution"]["success"]:
                continue
            
            # Get the proof file path
            proof_path = segment["proof_execution"]["proof_file"]
            if not os.path.exists(proof_path):
                print(f"Warning: Proof file not found for {segment_id}: {proof_path}")
                continue
            
            total_ezkl_segments += 1
            
            # Get the segment metadata
            segment_metadata = metadata["slices"].get(segment_id)
            if not segment_metadata:
                print(f"Warning: Metadata for segment {segment_id} not found")
                continue
            
            # Get the paths for verification
            settings_path = segment_metadata.get("settings_path")
            vk_path = segment_metadata.get("vk_path")
            circuit_path = segment_metadata.get("circuit_path")
            
            # For JSTprove, we need additional paths
            witness_execution = segment.get("witness_execution", {})
            witness_method = witness_execution.get("method", "")
            
            # Get input and output paths from run directory
            run_dir = os.path.dirname(run_results_path)
            segment_dir = os.path.join(run_dir, segment_id)
            input_path = os.path.join(segment_dir, "input.json")
            output_path = os.path.join(segment_dir, "output.json")
            
            # For JSTprove, use witness_file instead of output_file
            if witness_method == "jstprove_gen_witness":
                output_file = witness_execution.get("output_file", "")
                if output_file.endswith("output.json"):
                    witness_path = output_file.replace("output.json", "output_witness.bin")
                else:
                    witness_path = witness_execution.get("witness_file") or output_file.replace(".json", "_witness.bin")
            else:
                witness_path = witness_execution.get("output_file")
            
            # For JSTprove, use circuit.txt instead of ONNX if available
            if witness_method == "jstprove_gen_witness" and circuit_path and circuit_path.endswith('.onnx'):
                circuit_dir = os.path.dirname(circuit_path)
                circuit_txt = os.path.join(circuit_dir, "jstprove_circuitization", os.path.basename(circuit_path).replace('.onnx', '_circuit.txt'))
                if os.path.exists(circuit_txt):
                    circuit_path = circuit_txt
            
            # Verify the proof
            print(f"Verifying proof for {segment_id}...")
            start_time = time.time()
            
            # Check if backend is JSTprove
            if hasattr(self.backend, '__class__') and 'JSTprove' in str(type(self.backend)):
                verify_success = self.backend.verify(
                    proof_path=proof_path,
                    circuit_path=circuit_path,
                    input_path=input_path,
                    output_path=output_path,
                    witness_path=witness_path,
                    settings_path=settings_path,
                    vk_path=vk_path
                )
            else:
                verify_success = self.backend.verify(
                    proof_path=proof_path,
                    settings_path=settings_path,
                    vk_path=vk_path
                )
            verify_time = time.time() - start_time
            
            # Create or update verification_execution object
            verification_execution = {
                "verified": verify_success,
                "success": verify_success,
                "verification_time": verify_time
            }
            
            if verify_success:
                verified_segments += 1
                print(f"Successfully verified {segment_id}")
                # Display individual segment verification time immediately
                print(f"  {segment_id}: {verify_time:.2f}s")
            else:
                print(f"Failed to verify {segment_id}")
            
            # Update the segment with the verification information
            segment["verification_execution"] = verification_execution
        
        verify_elapsed = time.time() - verify_start
        # Update the execution_chain with the new counters
        # Keep the existing ezkl_witness_slices and ezkl_proved_slices
        run_results["execution_chain"]["ezkl_verified_slices"] = verified_segments
        run_results["execution_chain"]["verify_total_time"] = verify_elapsed
        
        # Save the updated run results
        with open(run_results_path, 'w') as f:
            json.dump(run_results, f, indent=2)
        
        return run_results


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet"
    }

    # Get model directory
    model_dir = os.path.abspath(base_paths[model_choice])
    
    # Get run directory - use the latest run in the model's run directory
    run_dir = os.path.join(model_dir, "run")
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found at {run_dir}")
        exit(1)
    
    # Find the latest run
    run_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith("run_")])
    if not run_dirs:
        print(f"Error: No runs found in {run_dir}")
        exit(1)
    
    latest_run = run_dirs[-1]
    run_path = os.path.join(run_dir, latest_run)
    
    # Construct paths for run_results.json and metadata.json
    run_results_path = os.path.join(run_path, "run_result.json")
    metadata_path = os.path.join(run_dir, "metadata.json")
    
    if not os.path.exists(run_results_path):
        print(f"Error: run_result.json not found at {run_results_path}")
        exit(1)
    
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found at {metadata_path}")
        exit(1)
    
    # Initialize verifier
    verifier = Verifier()
    
    # Run verification
    print(f"Verifying run {latest_run} for model {base_paths[model_choice]}...")
    results = verifier.verify_run(run_results_path, metadata_path)
    
    # Display results
    print(f"\nVerification completed!")
    print(f"Verified segments: {results['execution_chain']['ezkl_verified_slices']} of {results['execution_chain']['ezkl_proved_slices']}")
    
    # Print details for each segment
    print("\nSegment details:")
    for segment in results["execution_chain"]["execution_results"]:
        segment_id = segment["segment_id"]
        if "verification_execution" in segment:
            verified = segment["verification_execution"]["verified"]
            status = "Success" if verified else "Failed"
            time_taken = segment["verification_execution"]["verification_time"]
            print(f"  {segment_id}: {status} (Time: {time_taken:.2f}s)")