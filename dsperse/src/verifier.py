"""
Orchestration for verifying proofs.
"""

import os
import time
import logging
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class Verifier:
    """
    Orchestrator for verifying model execution proofs.
    """
    
    def __init__(self):
        """
        Initialize the verifier.
        """
        self.ezkl_runner = EZKL()

    def verify_dirs(self, run_path: str | Path, dirs_path: str | Path) -> dict:
        """Verify proofs for circuit-capable slices.
        - Loads run metadata from run_path/metadata.json.
        - Gets per-slice proof path exclusively from run_results.execution_chain.execution_results[].proof_execution.proof_file.
        - Resolves settings and vk relative to the slice directory.
        - Persists verification_execution entries into run_results.json and updates ezkl_verified_slices.
        Returns updated run_results dict.
        """
        run_path = Path(run_path)
        dirs_path = Utils.dirs_root_from(Path(dirs_path))
        metadata = Utils.load_run_metadata(run_path)
        run_results = Utils.load_run_results(run_path)

        # Build quick map from run_results to prefer existing proof paths
        proof_paths_by_slice = {}
        try:
            for entry in run_results.get("execution_chain", {}).get("execution_results", []):
                sid = entry.get("slice_id") or entry.get("segment_id")
                pe = entry.get("proof_execution", {}) if isinstance(entry, dict) else {}
                if sid and pe and pe.get("proof_file"):
                    proof_paths_by_slice[sid] = pe.get("proof_file")
        except Exception:
            pass

        verifs: dict[str, dict] = {}
        total = 0

        for slice_id, meta in Utils.iter_circuit_slices(metadata):
            # Determine proof path preference order
            if slice_id in proof_paths_by_slice:
                proof_path = Path(proof_paths_by_slice[slice_id])
            else:
                logger.warning(f"Skipping {slice_id}: proof path not recorded in run_results")
                continue

            if not proof_path.exists():
                logger.warning(f"Skipping {slice_id}: proof not found at {proof_path}")
                continue

            total += 1
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)

            # Resolve verification artifacts
            settings_path = Utils.resolve_under_slice(slice_dir, meta.get("settings_path"))
            vk_path = Utils.resolve_under_slice(slice_dir, meta.get("vk_path"))

            if not settings_path or not os.path.exists(settings_path):
                logger.warning(f"Skipping {slice_id}: settings file not found ({settings_path})")
                continue
            if not vk_path or not os.path.exists(vk_path):
                logger.warning(f"Skipping {slice_id}: verification key not found ({vk_path})")
                continue

            start = time.time()
            success = self.ezkl_runner.verify(
                proof_path=str(proof_path),
                settings_path=settings_path,
                vk_path=vk_path,
            )
            elapsed = time.time() - start

            verifs[slice_id] = {
                "success": bool(success),
                "time_sec": elapsed,
                "error": None if success else "verification_failed",
            }

        # Persist to run_results.json
        run_results, verified_count = Utils.merge_execution_into_run_results(run_results, verifs, "verification")
        exec_chain = run_results.setdefault("execution_chain", {})
        exec_chain["ezkl_verified_slices"] = int(verified_count)
        # Save updates
        Utils.save_run_results(run_path, run_results)
        # Update metadata
        Utils.update_metadata_after_execution(run_path, total, verified_count, "verification")
        return run_results

    def verify_dslice(self, run_path: str | Path, dslice_path: str | Path) -> dict:
        temp_dirs = Converter.convert(str(dslice_path), output_type="dirs", cleanup=False)

        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        result = self.verify_dirs(run_path, dirs_root)

        Converter.convert(str(dirs_root), output_type="dslice", cleanup=False)

        return result

    def verify_dsperse(self, run_path: str | Path, dsperse_path: str | Path) -> dict:
        temp_dirs = Converter.convert(dsperse_path, output_type="dirs", cleanup=False)

        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        result = self.verify_dirs(run_path, dirs_root)

        Converter.convert(str(dirs_root), output_type="dsperse", cleanup=False)

        return result

    def verify(self, run_path: str | Path, model_path: str | Path) -> dict:
        # Validate run path
        Utils.load_run_metadata(run_path)

        detected = Converter.detect_type(model_path)

        if detected == "dslice":
            return self.verify_dslice(run_path, model_path)
        if detected == "dsperse":
            return self.verify_dsperse(run_path, model_path)
        if detected == "dirs":
            return self.verify_dirs(run_path, model_path)
        raise ValueError(f"Unsupported data type for verification: {detected}")


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet"
    }

    # Get model directory
    model_dir = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(model_dir, "slices")
    slices_dir = os.path.join(slices_dir, "slice_0.dslice")  # give a single slice to test
    
    # Get run directory - use the latest run in the model's run directory
    run_dir = os.path.join(model_dir, "run")
    
    # Find the latest run
    run_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith("run_")])
    
    latest_run = run_dirs[-1]
    run_path = os.path.join(run_dir, latest_run)
    
    # Initialize verifier
    verifier = Verifier()
    
    # Run verification
    print(f"Verifying run {latest_run} for model {base_paths[model_choice]}...")
    results = verifier.verify(run_path, slices_dir)
    
    # Display results
    print(f"\nVerification completed!")
    print(f"Verified slices: {results['execution_chain']['ezkl_verified_slices']} of {results['execution_chain']['ezkl_proved_slices']}")
    
    # Print details for each slice
    print("\nSlice details:")
    for slice_result in results["execution_chain"]["execution_results"]:
        slice_id = slice_result["slice_id"]
        if "verification_execution" in slice_result:
            verified = slice_result["verification_execution"]["verified"]
            status = "Success" if verified else "Failed"
            time_taken = slice_result["verification_execution"]["verification_time"]
            print(f"  {slice_id}: {status} (Time: {time_taken:.2f}s)")