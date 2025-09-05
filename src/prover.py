"""
Orchestration for various provers.
"""

import os
import json
import time
from pathlib import Path
from src.backends.ezkl import EZKL

class Prover:
    """
    Orchestrator for proving model execution segments.
    """
    
    def __init__(self):
        """
        Initialize the prover.
        """
        self.ezkl_runner = EZKL()
    
    def prove_run(self, run_results_path, metadata_path):
        """
        Prove the segments in a run.
        
        Args:
            run_results_path (str): Path to the run_results.json file
            metadata_path (str): Path to the metadata.json file
            
        Returns:
            dict: Updated run results with proof information
        """
        # Load the run results and metadata
        with open(run_results_path, 'r') as f:
            run_results = json.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get the run directory from the run_results_path
        run_dir = os.path.dirname(run_results_path)
        
        # Initialize counters
        proved_segments = 0
        total_ezkl_segments = 0

        # Check if any circuit files exist before proceeding
        has_circuit_files = False
        for segment in run_results["execution_chain"]["execution_results"]:
            segment_id = segment["segment_id"]
            witness_execution = segment["witness_execution"]
            if witness_execution["method"] == "ezkl_gen_witness" and witness_execution["success"]:
                segment_metadata = metadata["slices"].get(segment_id)
                if segment_metadata and segment_metadata.get("circuit_path") and os.path.exists(segment_metadata["circuit_path"]):
                    has_circuit_files = True
                    break

        if not has_circuit_files:
            raise ValueError("No circuit files found. Please run 'dsperse circuitize' first to generate circuit files before attempting to prove.")

        # Process each segment in the execution results
        for segment in run_results["execution_chain"]["execution_results"]:
            segment_id = segment["segment_id"]
            
            # Create a witness_execution object from the existing segment data
            witness_execution = segment["witness_execution"]
            # Only process segments with method "ezkl_gen_witness" and success=true
            if witness_execution["method"] == "ezkl_gen_witness" and witness_execution["success"]:
                total_ezkl_segments += 1
                
                # Get the segment metadata
                segment_metadata = metadata["slices"].get(segment_id)
                if not segment_metadata:
                    print(f"Warning: Metadata for segment {segment_id} not found")
                    continue
                
                # Get the paths for verification
                witness_path = witness_execution["output_file"]
                model_path = segment_metadata["circuit_path"]
                pk_path = segment_metadata["pk_path"]

                # Check if circuit file exists
                if model_path is None:
                    print(f"Warning: No circuit file found for segment {segment_id} (circuit_path is null)")
                    continue
                if not os.path.exists(model_path):
                    print(f"Warning: Circuit file not found for segment {segment_id}: {model_path}")
                    continue

                # Create proof directory and path
                proof_dir = os.path.join(run_dir, segment_id)
                os.makedirs(proof_dir, exist_ok=True)
                proof_path = os.path.join(proof_dir, "proof.json")

                # Generate proof
                print(f"Generating proof for {segment_id}...")
                start_time = time.time()
                prove_success, prove_result = self.ezkl_runner.prove(
                    witness_path=witness_path,
                    model_path=model_path,
                    proof_path=proof_path,
                    pk_path=pk_path
                )
                prove_time = time.time() - start_time
                
                # Create proof_execution object
                proof_execution = {
                    "proof_file": proof_path,
                    "success": prove_success,
                    "proof_generation_time": prove_time
                }
                
                # Add error message if proof generation failed
                if not prove_success:
                    print(f"Failed to generate proof for {segment_id}: {prove_result}")
                    proof_execution["error"] = f"Proof generation failed: {prove_result}"
                else:
                    # If proof generation was successful, increment proved_segments
                    proved_segments += 1
                
                # Update the segment with the new structure
                segment.clear()
                segment.update({
                    "segment_id": segment_id,
                    "witness_execution": witness_execution,
                    "proof_execution": proof_execution
                })
            else:
                # For segments that are not ezkl_gen_witness or not successful,
                # just restructure them to use the new format
                segment.clear()
                segment.update({
                    "segment_id": segment_id,
                    "witness_execution": witness_execution
                })
        
        # Update the execution_chain with the new counters
        run_results["execution_chain"]["ezkl_witness_slices"] = total_ezkl_segments
        run_results["execution_chain"]["ezkl_proved_slices"] = proved_segments
        # Set verified_slices to 0 as verification is not performed in the prove command
        run_results["execution_chain"]["ezkl_verified_slices"] = 0
        
        # Remove the old verification section if it exists
        if "verification" in run_results:
            del run_results["verification"]
        
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
    
    # Initialize prover
    prover = Prover()
    
    # Run proving
    print(f"Proving run {latest_run} for model {base_paths[model_choice]}...")
    results = prover.prove_run(run_results_path, metadata_path)
    
    # Display results
    print(f"\nProving completed!")
    print(f"Proved segments: {results['execution_chain']['ezkl_proved_slices']} of {results['execution_chain']['ezkl_witness_slices']}")
    
    # Print details for each segment
    print("\nSegment details:")
    for segment in results["execution_chain"]["execution_results"]:
        segment_id = segment["segment_id"]
        if "proof_execution" in segment:
            success = segment["proof_execution"]["success"]
            status = "Success" if success else "Failed"
            time_taken = segment["proof_execution"]["proof_generation_time"]
            print(f"  {segment_id}: {status} (Time: {time_taken:.2f}s)")