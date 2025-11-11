"""
Orchestration for various provers.
"""
import logging
import os
import json
import time
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class Prover:
    """
    Orchestrator for proving model execution slices.
    """

    def __init__(self):
        """
        Initialize the prover.
        """
        self.ezkl_runner = EZKL()

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

    @staticmethod
    def _resolve_rel_path(p: str, slice_dir: str) -> str:
        """Resolve a metadata path to an absolute path.
        - Absolute paths are returned as-is.
        - `slice_#/...` has the leading slice dir removed and resolved under `slice_dir`.
        - Any other relative path is resolved relative to `slice_dir`.
        """
        if not p:
            return None
        p_str = str(p)
        # Absolute path
        if os.path.isabs(p_str):
            return p_str
        # If starts with this slice directory name, strip it
        sd_name = os.path.basename(os.path.abspath(slice_dir))
        parts = p_str.split(os.sep)
        if parts and parts[0] == sd_name:
            parts = parts[1:]
            p_str = os.path.join(*parts) if parts else ''
        # Resolve relative to slice_dir
        return os.path.abspath(os.path.join(slice_dir, p_str))

    @staticmethod
    def _extract_artifacts(meta: dict, slice_dir: str):
        """Extract compiled circuit, pk, and settings paths from slice metadata.
        Supports both the new nested compilation schema and legacy flat keys.
        Returns tuple (model_path, pk_path, settings_path).
        """
        model_path = pk_path = settings_path = None

        # Preferred nested schema
        comp = (meta or {}).get('compilation', {})
        ezkl_comp = (comp or {}).get('ezkl', {})
        files = (ezkl_comp or {}).get('files', {})
        if files:
            model_path = files.get('compiled_circuit') or files.get('compiled')
            pk_path = files.get('pk_key')
            settings_path = files.get('settings')

        # Legacy flat keys fallback
        model_path = model_path or meta.get('circuit_path') or meta.get('compiled')
        pk_path = pk_path or meta.get('pk_path')
        settings_path = settings_path or meta.get('settings_path')

        # Resolve to absolute paths relative to the slice directory
        model_path = Prover._resolve_rel_path(model_path, slice_dir) if model_path else None
        pk_path = Prover._resolve_rel_path(pk_path, slice_dir) if pk_path else None
        settings_path = Prover._resolve_rel_path(settings_path, slice_dir) if settings_path else None

        return model_path, pk_path, settings_path

    def prove_single_slice(self, input_slice, witness_file, output_path=None):
        """
        Proves a single slice (dslice file or slice directory) using the specified
        witness file and saves the proof to the provided output path.

        Behavior:
        - If `input_slice` is a `.dslice` file, it is converted to a slice directory
          (without cleanup) before proving.
        - Slice-level metadata is read to locate the compiled circuit, proving key,
          and settings. Relative paths like `payload/...` are resolved against the
          slice directory. Paths like `slice_#/payload/...` are also supported by
          stripping the slice prefix.
        - If `output_path` is not provided, the proof is written next to the witness
          file as `proof.json`.

        Args:
            input_slice (str): Path to a slice directory or a `.dslice` file.
            witness_file (str): Path to the witness JSON file required for proving.
            output_path (str, optional): Path where the proof will be saved. If it is
                a directory, the proof will be saved as `<dir>/proof.json`.

        Returns:
            str: Path to the generated proof file.
        """

        witness_file_path = os.path.abspath(str(witness_file))
        if not os.path.exists(witness_file_path):
            raise FileNotFoundError(f"Witness file not found: {witness_file_path}")

        # If output_path points to a directory, write proof.json inside it; if None, next to witness.
        if output_path is None:
            proof_path = os.path.join(os.path.dirname(witness_file_path), "proof.json")
        else:
            output_path = str(output_path)
            if os.path.isdir(output_path) or (not os.path.splitext(output_path)[1]):
                proof_path = os.path.join(output_path, "proof.json")
            else:
                proof_path = output_path
        os.makedirs(os.path.dirname(proof_path), exist_ok=True)

        # If input slice is a dslice, convert to directory
        dir_path = str(input_slice)
        original_format = None
        in_path_obj = Path(dir_path)
        detected_type = Converter.detect_type(in_path_obj) if in_path_obj.exists() else None
        if detected_type == 'dslice':
            original_format = 'dslice'
            logger.info(f"Converting {in_path_obj} to directory format for proving")
            dir_path = Converter.convert(str(in_path_obj), output_type="dirs", cleanup=False)
            in_path_obj = Path(dir_path)

        # At this point, dir_path should be a slice directory
        # Load slice-level metadata
        metadata_path = Utils.find_metadata_path(str(dir_path))
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        slice_dir = os.path.dirname(metadata_path)
        model_path, pk_path, settings_path = Prover._extract_artifacts(metadata, slice_dir)

        # Validate required artifacts
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Compiled circuit not found for slice at {slice_dir}. Got: {model_path}")
        if not pk_path or not os.path.exists(pk_path):
            raise FileNotFoundError(f"Proving key not found for slice at {slice_dir}. Got: {pk_path}")
        if settings_path and not os.path.exists(settings_path):
            # Not fatal, but warn; EZKL.prove uses it for SRS convenience
            logger.warning(f"Settings file not found at {settings_path}; proceeding without it.")
            settings_path = None

        logger.info(
            f"Proving slice at {slice_dir} with witness {witness_file_path}, model {model_path}, pk {pk_path}, proof -> {proof_path}"
        )
        start_time = time.time()
        success, result = self.ezkl_runner.prove(
            witness_path=witness_file_path,
            model_path=model_path,
            proof_path=proof_path,
            pk_path=pk_path,
            settings_path=settings_path,
        )
        elapsed = time.time() - start_time

        if not success:
            logger.error(f"Proof generation failed for slice at {slice_dir}: {result}")
            raise RuntimeError(f"Proof generation failed: {result}")

        logger.info(f"Proof generated for slice at {slice_dir} -> {proof_path} in {elapsed:.2f}s")

        # Note: We do not convert back to the original dslice format automatically.
        # If needed, the caller can perform conversion after proving.
        return proof_path


    def prove_full_run(self, slices_path, run_path): # TODO: Metadata or slices/dslices/dsperse file path
        """
        Prove the slices in a run.

        Args:
            run_results_path (str): Path to the run_results.json file
            metadata_path (str): Path to the metadata.json file

        Returns:
            dict: Updated run results with proof information
        """

        # TODO: If the input file is a dsperse file, extract metadata and slices
        file_type = Converter.detect_type(slices_path)

        run_results_path = os.path.join(run_path, "run_results.json")
        metadata_path = os.path.join(run_path, "metadata.json")

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

        # todo: convert to back to original format

        return run_results


    def prove(self, slices, input_file, output_path=None):
        """
        Prove the slices in a run. This function will delegate to the appropriate prover.
        One for a full run, and one for a single slice, depending on the input.

        Args:
            slices: can be a single dslice file or a dsperse file or a directory containing slices
            input_file: either a single generated witness file a run directory containing the whole chain of inputs and outputs

        Returns:

        """
        # if the input slices is a single dslice file and input file ends in .json
        if input_file.endswith(".json") and Path(slices).suffix == '.dslice':
            return self.prove_single_slice(input_slice=slices, witness_file=input_file, output_path=output_path)
        elif Path(input_file).is_dir() and (Path(slices).is_dir()) or (Path(slices).is_file() and Path(slices).suffix == '.dsperse'):
            return self.prove_full_run(slices=slices, run_directory=input_file, output_path=output_path)
        else:
            raise ValueError("Invalid input. Please provide a single dslice file or a dsperse file or a directory containing slices.")


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/age",
        5: "../models/version"
    }

    model_dir = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(model_dir, "slices") # slices dir, or single slice, or dsperse file
    input_file = os.path.join(model_dir, "input.json") # Path to input file for this slice, or whole model if not provided

    # Get run directory - use the latest run in the model's run directory
    run_dir = os.path.join(model_dir, "run")
    if not os.path.exists(run_dir):
        print(f"Run directory not found at {run_dir}, assuming input file provided.")

    # Find the latest run
    run_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith("run_")])
    if not run_dirs:
        print(f"Error: No runs found in {run_dir}")
        exit(1)

    latest_run = run_dirs[-1]
    run_path = os.path.join(run_dir, latest_run)

    # Construct paths for run_results.json and metadata.json
    # TODO: Change the inputs for a full run.
    run_results_path = os.path.join(run_path, "run_result.json")
    run_metadata_path = os.path.join(run_dir, "metadata.json")

    # Initialize prover
    prover = Prover()

    # TODO: make a function that would prove only one slice
    # Run proving
    print(f"Proving run {latest_run} for model {base_paths[model_choice]}...")
    results = prover.prove(slice_path=slices_dir, run_path=run_path)

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
