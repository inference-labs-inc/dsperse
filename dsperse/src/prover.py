"""
Orchestration for various provers.
"""

import os
import json
import time
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL


class Prover:
    """
    Orchestrator for proving model execution slices.
    """

    def __init__(self):
        """
        Initialize the prover.
        """
        self.ezkl_runner = EZKL()

    # ------------------------
    # Internal helpers
    # ------------------------
    def _load_run_and_metadata(self, run_results_path, metadata_path):
        with open(run_results_path, "r") as f:
            run_results = json.load(f)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return run_results, metadata

    def _has_circuit_files(self, run_results, metadata):
        for slice_result in run_results.get("execution_chain", {}).get(
            "execution_results", []
        ):
            slice_id = slice_result.get("slice_id")
            witness_execution = slice_result.get("witness_execution", {})
            if witness_execution.get(
                "method"
            ) == "ezkl_gen_witness" and witness_execution.get("success"):
                slice_metadata = metadata.get("slices", {}).get(slice_id)
                if (
                    slice_metadata
                    and slice_metadata.get("circuit_path")
                    and os.path.exists(slice_metadata["circuit_path"])
                ):
                    return True
        return False

    def _process_slice(self, slice_result, metadata, run_dir):
        """Process a single slice; returns tuple (updated_slice, counters_delta).
        counters_delta: (ezkl_witness_increment, proved_increment)
        """
        slice_id = slice_result["slice_id"]
        witness_execution = slice_result["witness_execution"]

        if witness_execution.get(
            "method"
        ) != "ezkl_gen_witness" or not witness_execution.get("success"):
            # Just normalize structure
            return {"slice_id": slice_id, "witness_execution": witness_execution}, (
                0,
                0,
            )

        # It's an EZKL slice we should try to prove
        ezkl_witness_inc = 1

        slice_metadata = metadata.get("slices", {}).get(slice_id)
        if not slice_metadata:
            print(f"Warning: Metadata for slice {slice_id} not found")
            return {"slice_id": slice_id, "witness_execution": witness_execution}, (
                ezkl_witness_inc,
                0,
            )

        witness_path = witness_execution.get("output_file")
        model_path = slice_metadata.get("circuit_path")
        pk_path = slice_metadata.get("pk_path")
        settings_path = slice_metadata.get("settings_path")

        # Validate circuit path
        if model_path is None:
            print(
                f"Warning: No circuit file found for slice {slice_id} (circuit_path is null)"
            )
            return {"slice_id": slice_id, "witness_execution": witness_execution}, (
                ezkl_witness_inc,
                0,
            )
        if not os.path.exists(model_path):
            print(
                f"Warning: Circuit file not found for slice {slice_id}: {model_path}"
            )
            return {"slice_id": slice_id, "witness_execution": witness_execution}, (
                ezkl_witness_inc,
                0,
            )

        # Prepare proof path
        proof_dir = os.path.join(run_dir, slice_id)
        os.makedirs(proof_dir, exist_ok=True)
        proof_path = os.path.join(proof_dir, "proof.json")

        # Generate proof
        print(f"Generating proof for {slice_id}...")
        start_time = time.time()
        prove_success, prove_result = self.ezkl_runner.prove(
            witness_path=witness_path,
            model_path=model_path,
            proof_path=proof_path,
            pk_path=pk_path,
            settings_path=settings_path
        )
        prove_time = time.time() - start_time

        proof_execution = {
            "proof_file": proof_path,
            "success": prove_success,
            "proof_generation_time": prove_time,
        }
        if not prove_success:
            print(f"Failed to generate proof for {slice_id}: {prove_result}")
            proof_execution["error"] = f"Proof generation failed: {prove_result}"
            proved_inc = 0
        else:
            proved_inc = 1
            print(f"  {slice_id}: {prove_time:.2f}s")

        updated_slice = {
            "slice_id": slice_id,
            "witness_execution": witness_execution,
            "proof_execution": proof_execution,
        }
        return updated_slice, (ezkl_witness_inc, proved_inc)

    def _finalize_run_results(self, run_results, proved_slices, total_ezkl_slices):
        run_results["execution_chain"]["ezkl_witness_slices"] = total_ezkl_slices
        run_results["execution_chain"]["ezkl_proved_slices"] = proved_slices
        run_results["execution_chain"]["ezkl_verified_slices"] = 0
        if "verification" in run_results:
            del run_results["verification"]
        return run_results

    def _save_run_results(self, run_results_path, run_results):
        with open(run_results_path, "w") as f:
            json.dump(run_results, f, indent=2)

    # ------------------------
    # Public API
    # ------------------------
    def prove_run(self, run_results_path, metadata_path):
        """
        Prove the slices in a run.

        Args:
            run_results_path (str): Path to the run_results.json file
            metadata_path (str): Path to the metadata.json file

        Returns:
            dict: Updated run results with proof information
        """
        run_results, metadata = self._load_run_and_metadata(
            run_results_path, metadata_path
        )
        run_dir = os.path.dirname(run_results_path)

        # Pre-check circuits exist
        if not self._has_circuit_files(run_results, metadata):
            raise ValueError(
                "No circuit files found. Please run 'dsperse circuitize' first to generate circuit files before attempting to prove."
            )

        proved_slices = 0
        total_ezkl_slices = 0
        updated_slices = []
        for slice_result in run_results["execution_chain"]["execution_results"]:
            updated_slice, (w_inc, p_inc) = self._process_slice(
                slice_result, metadata, run_dir
            )
            updated_slices.append(updated_slice)
            total_ezkl_slices += w_inc
            proved_slices += p_inc

        run_results["execution_chain"]["execution_results"] = updated_slices
        run_results = self._finalize_run_results(
            run_results, proved_slices, total_ezkl_slices
        )
        self._save_run_results(run_results_path, run_results)
        return run_results


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    # Model configurations
    base_paths = {1: "../models/doom", 2: "../models/net", 3: "../models/resnet"}

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
    print(
        f"Proved slices: {results['execution_chain']['ezkl_proved_slices']} of {results['execution_chain']['ezkl_witness_slices']}"
    )

    # Print details for each slice
    print("\nSlice details:")
    for slice_result in results["execution_chain"]["execution_results"]:
        slice_id = slice_result["slice_id"]
        if "proof_execution" in slice_result:
            success = slice_result["proof_execution"]["success"]
            status = "Success" if success else "Failed"
            time_taken = slice_result["proof_execution"]["proof_generation_time"]
            print(f"  {slice_id}: {status} (Time: {time_taken:.2f}s)")
