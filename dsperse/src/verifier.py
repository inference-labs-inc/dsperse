"""
Orchestration for verifying proofs.
"""

import os
import time
import logging
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
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
        try:
            self.jstprove_runner = JSTprove()
        except Exception:
            self.jstprove_runner = None

    # ------------------------ Small helpers (mirror Prover) ------------------------
    @staticmethod
    def _get_witness_backend_from_run(run_path: Path, slice_id: str) -> str | None:
        """Inspect run_results.json to see which backend produced the witness for a slice."""
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

    @staticmethod
    def _select_verification_backend(run_path: Path, slice_id: str, meta: dict) -> str:
        """Prefer the backend that produced the witness; else meta backend; default jstprove."""
        from_run = Verifier._get_witness_backend_from_run(run_path, slice_id)
        if from_run in ("jstprove", "ezkl"):
            return from_run
        meta_backend = (meta.get("backend") or "").lower()
        if meta_backend in ("jstprove", "ezkl"):
            return meta_backend
        return "jstprove"

    def verify_dirs(self, run_path: str | Path, dirs_path: str | Path) -> dict:
        """Verify proofs for circuit-capable slices (JSTprove and EZKL).
        - Loads run metadata from run_path/metadata.json.
        - Locates per-slice proof path from run_results (fallback to <run>/slice_#/proof.json).
        - Selects backend aligned with witness backend; no cross-backend verification.
        - Persists per-slice verification into run_results.json and updates per-backend counters.
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
        jst_verified = 0
        ezkl_verified = 0

        # Build slice iterator with graceful fallback (mirror Prover)
        try:
            slices_iter = list(Utils.iter_circuit_slices(metadata))
        except Exception:
            slices_iter = []
        if not slices_iter:
            nodes = ((metadata or {}).get("execution_chain") or {}).get("nodes", {})
            all_slices = (metadata or {}).get("slices", {})
            slices_iter = [(sid, all_slices.get(sid, {})) for sid, node in nodes.items() if node.get("use_circuit")]

        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to verify under run {run_path}. Nothing to do.")
            return run_results

        for slice_id, meta in slices_iter:
            # Determine proof path preference order
            if slice_id in proof_paths_by_slice:
                proof_path = Path(proof_paths_by_slice[slice_id])
            else:
                # Fallback to standard location
                proof_path = Path(run_path) / slice_id / "proof.json"
            if not proof_path.exists():
                logger.warning(f"Skipping {slice_id}: proof not found at {proof_path}")
                continue

            total += 1
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)

            # Choose backend strictly (no cross-backend)
            preferred = self._select_verification_backend(run_path, slice_id, meta)

            start = time.time()
            success = False
            error_msg = None
            method = None

            if preferred == "jstprove":
                if self.jstprove_runner is None:
                    logger.warning("JSTprove CLI not available; cannot verify JSTprove proof")
                    success = False
                    error_msg = "JSTprove CLI not available"
                    method = "jstprove_verify"
                else:
                    # Resolve JSTprove artifacts
                    circuit_path = Utils.resolve_under_slice(slice_dir, meta.get("circuit_path") or meta.get("compiled"))
                    input_path = Path(run_path) / slice_id / "input.json"
                    output_path = Path(run_path) / slice_id / "output.json"
                    wf = self._get_witness_file_from_run(run_path, slice_id)
                    witness_path = Path(wf) if wf else (Path(run_path) / slice_id / "output_witness.bin")

                    # Validate required files
                    missing = [p for p in [circuit_path, input_path, output_path, witness_path] if not p or not Path(p).exists()]
                    if missing:
                        success = False
                        error_msg = f"Missing files for JSTprove verify: {', '.join(map(str, missing))}"
                        method = "jstprove_verify"
                    else:
                        try:
                            success = self.jstprove_runner.verify(
                                proof_path=str(proof_path),
                                circuit_path=str(circuit_path),
                                input_path=str(input_path),
                                output_path=str(output_path),
                                witness_path=str(witness_path),
                            )
                            method = "jstprove_verify"
                        except Exception as e:
                            success = False
                            error_msg = str(e)
                            method = "jstprove_verify"
                if success:
                    jst_verified += 1
            else:
                # EZKL verify path
                settings_path = Utils.resolve_under_slice(slice_dir, meta.get("settings_path"))
                vk_path = Utils.resolve_under_slice(slice_dir, meta.get("vk_path"))
                if not settings_path or not os.path.exists(settings_path):
                    logger.warning(f"Skipping {slice_id}: settings file not found ({settings_path})")
                    continue
                if not vk_path or not os.path.exists(vk_path):
                    logger.warning(f"Skipping {slice_id}: verification key not found ({vk_path})")
                    continue
                try:
                    success = self.ezkl_runner.verify(
                        proof_path=str(proof_path),
                        settings_path=settings_path,
                        vk_path=vk_path,
                    )
                    method = "ezkl_verify"
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    method = "ezkl_verify"
                if success:
                    ezkl_verified += 1

            elapsed = time.time() - start

            verifs[slice_id] = {
                "success": bool(success),
                "time_sec": elapsed,
                "method": method or "unknown",
                "attempted_jstprove": preferred == "jstprove",
                "attempted_ezkl": preferred == "ezkl",
                "error": None if success else (error_msg or "verification_failed"),
            }

        # Persist to run_results.json
        run_results, verified_count = Utils.merge_execution_into_run_results(run_results, verifs, "verification")
        exec_chain = run_results.setdefault("execution_chain", {})
        exec_chain["jstprove_verified_slices"] = int(jst_verified)
        exec_chain["ezkl_verified_slices"] = int(ezkl_verified)
        # Save updates
        Utils.save_run_results(run_path, run_results)
        # Update metadata
        Utils.update_metadata_after_execution(run_path, total, (jst_verified + ezkl_verified), "verification")
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
        """Verify proofs.
        Supports:
        - Run-root mode: run_path contains metadata.json (verify across slices using run_results proof paths)
        - Single-slice mode: run_path contains input.json and output.json (verify proof.json in-place)
        """
        run_path = Path(run_path)

        is_run_root = (run_path / "metadata.json").exists()
        is_slice_run = (run_path / "input.json").exists() and (run_path / "output.json").exists()

        detected = Converter.detect_type(model_path)

        if is_run_root:
            # Normal multi-slice flow
            Utils.load_run_metadata(run_path)
            if detected == "dslice":
                return self.verify_dslice(run_path, model_path)
            if detected == "dsperse":
                return self.verify_dsperse(run_path, model_path)
            if detected == "dirs":
                return self.verify_dirs(run_path, model_path)
            raise ValueError(f"Unsupported data type for verification: {detected}")

        if is_slice_run:
            # Ensure we operate on directory layout for the provided model/slice path
            dirs_model_path = model_path
            if detected != "dirs":
                dirs_model_path = Converter.convert(str(model_path), output_type="dirs", cleanup=False)

            result = self._verify_single_slice(run_path, dirs_model_path, detected)

            # Convert back to the original packaging if needed
            if detected != "dirs":
                from dsperse.src.utils.utils import Utils as _Utils
                root = _Utils.dirs_root_from(Path(dirs_model_path))
                Converter.convert(str(root), output_type=detected, cleanup=True)

            return result

        raise FileNotFoundError(f"Run path invalid; expected run-root (metadata.json) or per-slice (input.json + output.json) at {run_path}")

    def _verify_single_slice(self, run_path: Path, model_path: str | Path, detected: str) -> dict:
        """Internal: verify exactly one slice using a per-slice run directory.
        Expects `<run_path>/input.json` and `<run_path>/output.json` and proof.json to exist and `model_path` to
        resolve to a single-slice metadata source (slice dir or .dslice).
        Updates/creates `run_results.json` in `run_path`.
        """
        sdir = Path(model_path)

        run_meta = RunnerAnalyzer.generate_run_metadata(Path(sdir if sdir.is_dir() else Path(sdir)), save_path=None, original_format=detected)
        model_slices = (run_meta or {}).get("slices", {})
        if len(model_slices) != 1:
            raise ValueError(f"Slices path must represent exactly one slice for single-slice verification; found {len(model_slices)} in {model_path}")

        (slice_id, meta), = model_slices.items()

        dirs_root = Utils.dirs_root_from(Path(model_path))
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)
        proof_path = Path(run_path) / "proof.json"
        if not proof_path.exists():
            raise FileNotFoundError(f"Proof file not found at {proof_path}. Run 'prove' for this slice first.")

        # Select backend like multi-slice
        preferred = self._select_verification_backend(run_path, slice_id, meta)

        start = time.time()
        success = False
        error_msg = None
        method = None

        if preferred == "jstprove":
            if self.jstprove_runner is None:
                raise RuntimeError("JSTprove CLI not available; cannot verify JSTprove proof")
            circuit_path = Utils.resolve_under_slice(slice_dir, meta.get("circuit_path") or meta.get("compiled"))
            input_path = Path(run_path) / "input.json"
            output_path = Path(run_path) / "output.json"
            wf = self._get_witness_file_from_run(run_path, slice_id)
            witness_path = Path(wf) if wf else (Path(run_path) / "output_witness.bin")
            missing = [p for p in [circuit_path, input_path, output_path, witness_path] if not p or not Path(p).exists()]
            if missing:
                raise FileNotFoundError(f"Missing files for JSTprove verify: {', '.join(map(str, missing))}")
            success = self.jstprove_runner.verify(
                proof_path=str(proof_path),
                circuit_path=str(circuit_path),
                input_path=str(input_path),
                output_path=str(output_path),
                witness_path=str(witness_path),
            )
            method = "jstprove_verify"
        else:
            vk_path_res = Utils.resolve_under_slice(slice_dir, meta.get("vk_path"))
            settings_path_res = Utils.resolve_under_slice(slice_dir, meta.get("settings_path"))
            if not vk_path_res or not os.path.exists(vk_path_res):
                raise FileNotFoundError(f"Verification key not found for {slice_id} at {vk_path_res}")
            if settings_path_res and not os.path.exists(settings_path_res):
                raise FileNotFoundError(f"Settings file not found for {slice_id} at {settings_path_res}")
            success = self.ezkl_runner.verify(
                proof_path=str(proof_path),
                settings_path=settings_path_res,
                vk_path=vk_path_res,
            )
            method = "ezkl_verify"

        elapsed = time.time() - start

        verifs = {
            slice_id: {
                "success": bool(success),
                "time_sec": elapsed,
                "method": method,
                "attempted_jstprove": preferred == "jstprove",
                "attempted_ezkl": preferred == "ezkl",
                "error": None if success else (error_msg or "verification_failed"),
            }
        }
        run_results = Utils.load_run_results(Path(run_path))
        run_results, verified_count = Utils.merge_execution_into_run_results(run_results, verifs, "verification")
        exec_chain = run_results.setdefault("execution_chain", {})
        if method == "jstprove_verify":
            exec_chain["jstprove_verified_slices"] = int(exec_chain.get("jstprove_verified_slices", 0)) + int(success)
        elif method == "ezkl_verify":
            exec_chain["ezkl_verified_slices"] = int(exec_chain.get("ezkl_verified_slices", 0)) + int(success)
        Utils.save_run_results(Path(run_path), run_results)
        return run_results


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
    # slices_dir = os.path.join(slices_dir, "slice_0.dslice")  # give a single slice to test
    
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
    ec = results.get('execution_chain', {})
    verified_total = int(ec.get('jstprove_verified_slices', 0)) + int(ec.get('ezkl_verified_slices', 0))
    proved_total = int(ec.get('jstprove_proved_slices', 0)) + int(ec.get('ezkl_proved_slices', 0))
    print(f"Verified slices: {verified_total} of {proved_total}")
    
    # Print details for each slice
    print("\nSlice details:")
    for slice_result in results.get("execution_chain", {}).get("execution_results", []):
        slice_id = slice_result.get("slice_id")
        ve = slice_result.get("verification_execution")
        if ve:
            status = "Success" if ve.get("success") else "Failed"
            time_taken = ve.get("time_sec", 0.0)
            print(f"  {slice_id}: {status} (Time: {time_taken:.2f}s)")