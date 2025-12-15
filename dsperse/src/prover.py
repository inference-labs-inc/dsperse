"""
Orchestration for various provers.
"""
import logging
import os
import time
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
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
        try:
            self.jstprove_runner = JSTprove()
        except Exception:
            self.jstprove_runner = None

    # ------------------------ Small helpers (no behavior change) ------------------------
    @staticmethod
    def _resolve_slice_artifacts(slice_dir: Path, meta: dict) -> tuple[str | None, str | None, str | None]:
        """Resolve circuit, pk, settings paths under a given slice directory.
        Returns (circuit_path, pk_path, settings_path), any may be None.
        """
        circuit_rel = meta.get("circuit_path") or meta.get("compiled")
        circuit_path = Utils.resolve_under_slice(slice_dir, circuit_rel)
        pk_path = Utils.resolve_under_slice(slice_dir, meta.get("pk_path"))
        settings_path = Utils.resolve_under_slice(slice_dir, meta.get("settings_path"))

        if settings_path and not os.path.exists(settings_path):
            logger.warning(f"Settings file not found at {settings_path}; proceeding without it.")
            settings_path = None

        return circuit_path, pk_path, settings_path

    @staticmethod
    def _get_witness_backend_from_run(run_path: Path, slice_id: str) -> str | None:
        """Inspect run_results.json to see which backend produced the witness for a slice.
        Returns 'jstprove', 'ezkl', or None if not found.
        """
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        exec_chain = (rr or {}).get("execution_chain") or {}
        exec_results = exec_chain.get("execution_results") or []
        for entry in exec_results:
            if entry.get("slice_id") == slice_id:
                w = entry.get("witness_execution") or {}
                method = (w.get("method") or "").lower()
                if method.startswith("jstprove"):
                    return "jstprove"
                if method.startswith("ezkl"):
                    return "ezkl"
        return None

    def _select_proving_backend(self, run_path: Path, slice_id: str, meta: dict) -> str:
        """Choose proving backend with this priority: witness backend → meta backend → jstprove.
        Returns 'jstprove' or 'ezkl'.
        """
        from_run = self._get_witness_backend_from_run(run_path, slice_id)
        if from_run in ("jstprove", "ezkl"):
            return from_run
        meta_backend = (meta.get("backend") or "").lower()
        if meta_backend in ("jstprove", "ezkl"):
            return meta_backend
        return "jstprove"

    @staticmethod
    def _get_witness_file_from_run(run_path: Path, slice_id: str) -> str | None:
        """Return concrete witness file path recorded by runner for a slice, if any."""
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        exec_chain = (rr or {}).get("execution_chain") or {}
        exec_results = exec_chain.get("execution_results") or []
        for entry in exec_results:
            if entry.get("slice_id") == slice_id:
                w = entry.get("witness_execution") or {}
                wf = w.get("witness_file") or w.get("witness_path")
                return wf
        return None

    def _prove_with_backend(
        self,
        backend: str,
        witness_path: Path,
        circuit_path: str,
        proof_path: Path,
        pk_path: str | None,
        settings_path: str | None,
    ) -> tuple[bool, str, str | Path | None]:
        """Dispatch to the requested backend. Returns (success, method, result_or_error)."""
        if backend == "jstprove":
            if self.jstprove_runner is None:
                return False, "jstprove_prove", "JSTprove CLI not available"
            try:
                ok, res = self.jstprove_runner.prove(
                    witness_path=str(witness_path),
                    circuit_path=str(circuit_path),
                    proof_path=str(proof_path),
                )
                return ok, "jstprove_prove", res
            except Exception as e:
                return False, "jstprove_prove", str(e)

        # EZKL requires pk
        if not pk_path or not os.path.exists(pk_path):
            return False, "ezkl_prove", f"Proving key not found at {pk_path}"
        try:
            ok, res = self.ezkl_runner.prove(
                witness_path=str(witness_path),
                model_path=str(circuit_path),
                proof_path=str(proof_path),
                pk_path=str(pk_path),
                settings_path=settings_path,
            )
            return ok, "ezkl_prove", res
        except Exception as e:
            return False, "ezkl_prove", str(e)


    def prove_dirs(self, run_path: str | Path, dirs_path: str | Path, output_path: str | Path | None = None) -> dict:
        """Prove all circuit-capable slices (JSTprove preferred, fallback EZKL) given a slices directory layout.
        - Reads run metadata from `run_path/metadata.json`.
        - For each circuit slice, locates witness at `run_path/slice_#/output.json`.
        - Resolves compiled circuit (and pk for EZKL) relative to the slice directory.
        - Writes proofs to `<output_root or run_path>/slice_#/proof.json`.
        Returns a summary dict.
        """
        run_path = Path(run_path)
        dirs_path = Utils.dirs_root_from(Path(dirs_path))
        metadata = Utils.load_run_metadata(run_path)

        proofs: dict[str, dict] = {}
        total = 0
        proved_jst = 0
        proved_ezkl = 0

        # Get slices to prove from metadata (prefer Utils helper; fallback to execution_chain)
        try:
            slices_iter = list(Utils.iter_circuit_slices(metadata))
        except Exception:
            slices_iter = []

        if not slices_iter:
            nodes = ((metadata or {}).get("execution_chain") or {}).get("nodes", {})
            all_slices = (metadata or {}).get("slices", {})
            slices_iter = [(sid, all_slices.get(sid, {})) for sid, node in nodes.items() if node.get("use_circuit")]

        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to prove under run {run_path}. Nothing to do.")
            # Return existing run_results unchanged to avoid silent no-op
            return Utils.load_run_results(run_path)

        for slice_id, meta in slices_iter:
            total += 1
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)
            circuit_path, pk_path, settings_path = self._resolve_slice_artifacts(slice_dir, meta)

            # Determine witness path strictly based on backend that generated it
            preferred = self._select_proving_backend(Path(run_path), slice_id, meta)
            if preferred == "jstprove":
                # Prefer runner-recorded witness file; else default to 'output_witness.bin' next to output.json
                wf = self._get_witness_file_from_run(Path(run_path), slice_id)
                witness_path = Path(wf) if wf else (Path(run_path) / slice_id / "output_witness.bin")
            else:
                # EZKL uses the output.json written by runner as witness input for proving
                witness_path = Utils.witness_path_for(run_path, slice_id)
            if not witness_path.exists():
                logger.warning(f"Skipping {slice_id}: witness not found at {witness_path}")
                continue

            proof_path = Utils.proof_output_path(run_path, slice_id, output_path)
            os.makedirs(proof_path.parent, exist_ok=True)

            # Validate circuit
            if not circuit_path or not os.path.exists(circuit_path):
                logger.warning(f"Skipping {slice_id}: compiled circuit not found ({circuit_path})")
                continue

            # Decide backend: strictly use witness backend (or meta fallback); no cross-backend fallback
            preferred = self._select_proving_backend(Path(run_path), slice_id, meta)

            start = time.time()
            success = False
            method = None
            result = None

            # Attempt preferred backend only (no cross-backend fallback)
            success, method, result = self._prove_with_backend(
                preferred, Path(witness_path), str(circuit_path), Path(proof_path), pk_path, settings_path
            )

            elapsed = time.time() - start

            proofs[slice_id] = {
                "success": success,
                "proof_path": str(proof_path),
                "time_sec": elapsed,
                "method": method or "unknown",
                "attempted_jstprove": preferred == "jstprove",
                "attempted_ezkl": preferred == "ezkl",
                "error": None if success else (str(result) if result is not None else "Unknown error"),
            }
            if success:
                if method == "jstprove_prove":
                    proved_jst += 1
                elif method == "ezkl_prove":
                    proved_ezkl += 1
            else:
                logger.error(f"Proof failed for {slice_id}: {result}")

        # Persist updates to run_results.json
        run_results = Utils.load_run_results(run_path)
        run_results, _ = Utils.merge_execution_into_run_results(run_results, proofs, "proof")
        exec_chain = run_results.setdefault("execution_chain", {})
        exec_chain["jstprove_proved_slices"] = int(proved_jst)
        exec_chain["ezkl_proved_slices"] = int(proved_ezkl)
        # Reset verified counters since proofs changed (legacy behavior kept for EZKL; JSTprove placeholder)
        exec_chain["ezkl_verified_slices"] = exec_chain.get("ezkl_verified_slices", 0) if proved_ezkl == 0 else 0

        # Save run_results
        Utils.save_run_results(run_path, run_results)
        # Update metadata with a small proof summary
        Utils.update_metadata_after_execution(run_path, total, proved_jst + proved_ezkl, "proof")
        return run_results

    def prove_dslice(self, run_path: str | Path, dslice_path: str | Path, output_path: str | Path | None = None) -> dict:
        """Convert dslice -> dirs, delegate to `prove_dirs`, then convert back to dslice."""
        temp_dirs = Converter.convert(dslice_path, output_type="dirs", cleanup=False)

        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        summary = self.prove_dirs(run_path, dirs_root, output_path)

        # Convert back to dslice packaging (non-destructive)
        Converter.convert(str(dirs_root), output_type="dslice", cleanup=False)

        return summary

    def prove_dsperse(self, run_path: str | Path, dsperse_path: str | Path, output_path: str | Path | None = None) -> dict:
        """Convert dsperse -> dirs, delegate to `prove_dirs`, then convert back to dsperse."""
        temp_dirs = Converter.convert(dsperse_path, output_type="dirs", cleanup=False)

        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        summary = self.prove_dirs(run_path, dirs_root, output_path)

        Converter.convert(str(dirs_root), output_type="dsperse", cleanup=False)

        return summary

    def prove(self, run_path: str | Path, model_dir: str | Path, output_path: str | Path | None = None) -> dict:
        """Route to the appropriate prove path based on `model_dir` packaging.
        Supports two modes for run_path:
        - Run-root mode: run_path contains metadata.json (prove all circuit-capable slices)
        - Single-slice mode: run_path contains input.json and output.json (prove just this slice)
        """
        run_path = Path(run_path)

        # Determine run mode
        is_run_root = (run_path / "metadata.json").exists()
        is_slice_run = (run_path / "input.json").exists() and (run_path / "output.json").exists()

        detected = Converter.detect_type(model_dir)

        if is_run_root:
            # Existing behavior
            Utils.load_run_metadata(run_path)
            if detected == "dslice":
                return self.prove_dslice(run_path, model_dir, output_path)
            if detected == "dsperse":
                return self.prove_dsperse(run_path, model_dir, output_path)
            if detected == "dirs":
                return self.prove_dirs(run_path, model_dir, output_path)
            raise ValueError(f"Unsupported data type for proving: {detected}")

        # Single-slice mode: no run metadata, but we must have per-slice files
        if is_slice_run:
            # Ensure we operate on a directory layout for the provided slices/model path
            dirs_model_path = model_dir
            if detected != "dirs":
                dirs_model_path = Converter.convert(str(model_dir), output_type="dirs", cleanup=False)

            # Prove using the directory layout, while remembering original packaging for reconversion
            result = self._prove_single_slice(run_path, dirs_model_path, detected)

            # Convert back to the original packaging if needed
            if detected != "dirs":
                Converter.convert(str(Utils.dirs_root_from(Path(dirs_model_path))), output_type=detected, cleanup=True)

            return result

        # Otherwise invalid run_dir
        raise FileNotFoundError(f"Run path invalid; expected run-root (metadata.json) or per-slice (input.json + output.json) at {run_path}")

    def _prove_single_slice(self, run_path: Path, model_dir: str | Path, detected: str) -> dict:
        """Internal: prove exactly one slice using a per-slice run directory.
        Expects `<run_path>/input.json` and `<run_path>/output.json` to exist and `model_dir` to
        resolve to a single-slice metadata source (slice dir or .dslice).
        Writes `proof.json` and updates/creates `run_results.json` in `run_path`.
        """
        # Normalize/expand model_dir into run-like metadata so we can get artifact paths
        sdir = Path(model_dir)

        # Generate run-like metadata strictly from provided slices path
        run_meta = RunnerAnalyzer.generate_run_metadata(Path(sdir if sdir.is_dir() else Path(sdir)), save_path=None, original_format=detected)
        model_slices = (run_meta or {}).get("slices", {})

        # Expect exactly one slice in provided metadata for single-slice proving
        if len(model_slices) != 1:
            raise ValueError(f"Slices path must represent exactly one slice for single-slice proving; found {len(model_slices)} in {model_dir}")

        # Extract the only slice id and meta
        (slice_id, meta), = model_slices.items()

        # Resolve artifacts relative to the provided slices path root
        dirs_root = Utils.dirs_root_from(Path(model_dir))
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)
        model_path_res, pk_path_res, settings_path_res = self._resolve_slice_artifacts(slice_dir, meta)

        # Validate required inputs (circuit and witness); pk may be absent for JSTprove
        if not model_path_res or not os.path.exists(model_path_res):
            raise FileNotFoundError(f"Compiled circuit not found for {slice_id} at {model_path_res}")

        witness_path = Path(run_path) / "output.json"
        if not witness_path.exists():
            raise FileNotFoundError(f"Witness not found at {witness_path}")

        proof_path = Path(run_path) / "proof.json"
        proof_path.parent.mkdir(parents=True, exist_ok=True)

        preferred = self._select_proving_backend(run_path, slice_id, meta)
        # Adjust witness path per backend: JSTprove expects a dedicated witness file
        if preferred == "jstprove":
            wf = self._get_witness_file_from_run(run_path, slice_id)
            witness_path = Path(wf) if wf else (Path(run_path) / "output_witness.bin")
        else:
            witness_path = Path(run_path) / "output.json"
        if not witness_path.exists():
            raise FileNotFoundError(f"Witness not found at {witness_path}")

        start = time.time()
        success, method, result = self._prove_with_backend(
            preferred, witness_path, str(model_path_res), proof_path, pk_path_res, settings_path_res
        )
        elapsed = time.time() - start

        # Merge into run_results.json located in the per-slice dir
        proofs = {
            slice_id: {
                "success": bool(success),
                "proof_path": str(proof_path),
                "time_sec": elapsed,
                "method": method or "unknown",
                "attempted_jstprove": preferred == "jstprove",
                "attempted_ezkl": preferred == "ezkl",
                "error": None if success else str(result),
            }
        }
        run_results = Utils.load_run_results(Path(run_path))
        run_results, _ = Utils.merge_execution_into_run_results(run_results, proofs, "proof")
        exec_chain = run_results.setdefault("execution_chain", {})
        if method == "jstprove_prove":
            exec_chain["jstprove_proved_slices"] = int(exec_chain.get("jstprove_proved_slices", 0)) + 1
        elif method == "ezkl_prove":
            exec_chain["ezkl_proved_slices"] = int(exec_chain.get("ezkl_proved_slices", 0)) + 1

        existing_witness = int(exec_chain.get("ezkl_witness_slices", 0) or 0)
        if existing_witness == 0:
            exec_chain["ezkl_witness_slices"] = len(proofs)
        # Save run_results.json next to the per-slice files
        Utils.save_run_results(Path(run_path), run_results)

        return run_results


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
    # slices_dir = os.path.join(slices_dir, "slice_0")  # give a single slice to test

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

    # run_path = '/Volumes/SSD/Users/dan/Projects/dsperse/dsperse/models/net/run/run_20251214_234920/slice_0'

    # Initialize prover
    prover = Prover()

    # Run proving
    print(f"Proving run {latest_run} for model {base_paths[model_choice]}...")
    results = prover.prove(run_path, slices_dir)

    # Display results
    print(f"\nProving completed!")

    # Print details for each slice
    print("\nSlice details:")
    for slice_result in results["execution_chain"]["execution_results"]:
        slice_id = slice_result["slice_id"]
        if "proof_execution" in slice_result:
            success = slice_result["proof_execution"]["success"]
            status = "Success" if success else "Failed"
            time_taken = slice_result["proof_execution"]["proof_generation_time"]
            print(f"  {slice_id}: {status} (Time: {time_taken:.2f}s)")
