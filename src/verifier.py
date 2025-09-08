"""
Orchestration for verifying proofs.
"""

import os
import json
import time
from pathlib import Path
from src.backends.ezkl import EZKL

class Verifier:
    """
    Orchestrator for verifying model execution proofs.
    """
    
    def __init__(self):
        """
        Initialize the verifier.
        """
        self.ezkl_runner = EZKL()
    
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
            settings_path = segment_metadata["settings_path"]
            vk_path = segment_metadata["vk_path"]
            
            # Verify the proof
            print(f"Verifying proof for {segment_id}...")
            start_time = time.time()
            verify_success = self.ezkl_runner.verify(
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
        
        # Update the execution_chain with the new counters
        # Keep the existing ezkl_witness_slices and ezkl_proved_slices
        run_results["execution_chain"]["ezkl_verified_slices"] = verified_segments
        
        # Save the updated run results
        with open(run_results_path, 'w') as f:
            json.dump(run_results, f, indent=2)
        
        return run_results


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet"
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