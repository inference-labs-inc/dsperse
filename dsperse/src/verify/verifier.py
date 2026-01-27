"""
Orchestration for verifying proofs.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union, List, Tuple

from dsperse.src.analyzers.schema import TilingInfo, RunSliceMetadata, TileResult, SliceResult, Backend, ExecutionMethod, ExecutionChain, RunMetadata
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.verify.utils.verifier_utils import VerifierUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class Verifier:
    """
    Orchestrator for verifying model execution proofs.
    """

    def __init__(self, parallel: int = 1):
        """
        Initialize the verifier.

        Args:
            parallel: Number of parallel processes for verification (default: 1)
        """
        self.parallel = max(1, parallel)
        try:
            self.ezkl_runner = EZKL()
        except RuntimeError:
            self.ezkl_runner = None
            logger.warning("EZKL CLI not available. EZKL backend will be disabled.")

        try:
            self.jstprove_runner = JSTprove()
        except Exception:
            self.jstprove_runner = None
            logger.warning("JSTprove CLI not available. JSTprove backend will be disabled.")


    def verify_dirs(self, run_path: str | Path, dirs_path: str | Path, backend: str | None = None) -> dict:
        """Verify proofs for circuit-capable slices (JSTprove and EZKL)."""
        # --- Initialization ---
        run_path, dirs_path = Path(run_path), Utils.dirs_root_from(Path(dirs_path))
        metadata = VerifierUtils.initialize_verify_metadata(run_path, dirs_path)
        run_results = Utils.load_run_results(run_path)

        # --- Slice Filtering ---
        slices_iter = VerifierUtils.filter_verifiable_slices(metadata, backend, run_path)
        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to verify under run {run_path}. Nothing to do.")
            return run_results

        # --- Work Item Preparation ---
        proof_paths = VerifierUtils.get_proof_paths(run_results)
        work_items = self._prepare_work_items(slices_iter, dirs_path, run_path, proof_paths)
        if not work_items:
            return run_results

        # --- Execution ---
        results = self._execute_verification(work_items)

        # --- Result Processing & Finalization ---
        verifs, jst_verified, ezkl_verified = self._process_results(results)
        return VerifierUtils.finalize_verify_results(run_path, verifs, jst_verified, ezkl_verified, len(work_items))

    @staticmethod
    def _prepare_work_items(slices_iter: list, dirs_path: Path, run_path: Path, proof_paths: dict) -> list:
        """Prepare verification tasks for each slice."""
        work_items = []
        for slice_id, meta in slices_iter:
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)
            preferred = VerifierUtils.select_backend(run_path, slice_id, meta)
            tiling = meta.tiling

            circuit_path = Utils.resolve_under_slice(slice_dir, meta.jstprove_circuit_path or meta.circuit_path)
            settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
            vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)

            if tiling:
                proof_path, input_path, output_path, witness_path = None, None, None, None
            else:
                proof_path = Path(proof_paths.get(slice_id, run_path / slice_id / "proof.json"))
                if not proof_path.exists():
                    logger.warning(f"Skipping {slice_id}: proof not found at {proof_path}")
                    continue

                input_path, output_path = run_path / slice_id / "input.json", run_path / slice_id / "output.json"
                wf = VerifierUtils.get_witness_file(run_path, slice_id)
                witness_path = Path(wf) if wf else (run_path / slice_id / "output_witness.bin")

                if preferred == Backend.EZKL:
                    if not settings_path or not os.path.exists(settings_path):
                        logger.warning(f"Skipping {slice_id}: settings file not found ({settings_path})")
                        continue
                    if not vk_path or not os.path.exists(vk_path):
                        logger.warning(f"Skipping {slice_id}: verification key not found ({vk_path})")
                        continue

            work_items.append((
                slice_id, preferred, proof_path, circuit_path, settings_path, vk_path,
                input_path, output_path, witness_path, tiling.to_dict() if tiling else None,
                str(run_path), str(slice_dir)
            ))
        return work_items

    def _execute_verification(self, work_items: list) -> list:
        """Execute verification tasks in parallel or sequentially."""
        if self.parallel > 1 and len(work_items) > 1:
            logger.info(f"Verifying {len(work_items)} slices with {self.parallel} parallel processes...")
            results = []
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(VerifierUtils.verify_slice_logic, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    slice_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Slice {slice_id} verification failed: {e}")
                        results.append(SliceResult(slice_id=slice_id, success=False, error=str(e)).to_dict())
            return results
        else:
            return [VerifierUtils.verify_slice_logic(item) for item in work_items]

    @staticmethod
    def _process_results(results: list) -> tuple[dict, int, int]:
        """Aggregate verification results and count successful backend executions."""
        verifs = {}
        jst_verified, ezkl_verified = 0, 0
        for result in results:
            slice_id = result['slice_id']
            verifs[slice_id] = result
            if result['success']:
                method = result.get('method')
                if method == ExecutionMethod.JSTPROVE_VERIFY:
                    jst_verified += 1
                elif method == ExecutionMethod.EZKL_VERIFY:
                    ezkl_verified += 1
        return verifs, jst_verified, ezkl_verified

    def verify_dslice(self, run_path: str | Path, dslice_path: str | Path, backend: str | None = None) -> dict:
        """Verify slices packaged in .dslice format."""
        return self._verify_packaged(run_path, dslice_path, "dslice", backend)

    def verify_dsperse(self, run_path: str | Path, dsperse_path: str | Path, backend: str | None = None) -> dict:
        """Verify slices packaged in .dsperse format."""
        return self._verify_packaged(run_path, dsperse_path, "dsperse", backend)

    def _verify_packaged(self, run_path: str | Path, pkg_path: str | Path, pkg_type: str, backend: str | None) -> dict:
        """Internal: Verify slices from a packaged format by temporarily converting to dirs."""
        temp_dirs = Converter.convert(pkg_path, output_type="dirs", cleanup=False)
        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        try:
            summary = self.verify_dirs(run_path, dirs_root, backend=backend)
        finally:
            Converter.convert(str(dirs_root), output_type=pkg_type, cleanup=True)
        return summary

    def verify(self, run_path: str | Path, model_path: str | Path, backend: str | None = None,
               tiles_range: range | list[int] | None = None) -> dict:
        """Verify proofs (supports full runs, packaged formats, and single slices)."""
        # --- Initialization & Root Detection ---
        run_path = Path(run_path)
        is_run_root = (run_path / "metadata.json").exists() or (run_path / "run_results.json").exists()

        # --- Run Type Detection ---
        is_slice_run = ((run_path / "input.json").exists() and (run_path / "output.json").exists()) or \
                       (run_path / "split").exists() or (run_path / "tile_0").exists()

        detected = Converter.detect_type(model_path)

        # --- Dispatch to Specific Handler ---
        if is_run_root:
            return self._handle_run_root_verifying(run_path, model_path, detected, backend)

        if is_slice_run:
            return self._handle_single_slice_verifying(run_path, model_path, detected, backend, tiles_range)

        raise FileNotFoundError(f"Run path invalid at {run_path}")

    def _handle_run_root_verifying(self, run_path: Path, model_path: str | Path, detected: str, backend: str | None) -> dict:
        """Internal: Handle verification for a full model run root."""
        if detected == "dslice":
            return self.verify_dslice(run_path, model_path, backend=backend)
        if detected == "dsperse":
            return self.verify_dsperse(run_path, model_path, backend=backend)
        if detected == "dirs":
            return self.verify_dirs(run_path, model_path, backend=backend)
        raise ValueError(f"Unsupported data type: {detected}")

    def _handle_single_slice_verifying(self, run_path: Path, model_path: str | Path, detected: str, 
                                       backend: str | None, tiles_range: range | list[int] | None) -> dict:
        """Internal: Handle verification for a single slice run directory."""
        if backend not in (Backend.JSTPROVE, Backend.EZKL):
            raise ValueError("Single-slice verification requires explicit backend.")
        
        dirs_model_path = model_path
        if detected != "dirs":
            dirs_model_path = Converter.convert(str(model_path), output_type="dirs", cleanup=False)

        result = self._verify_single_slice(run_path, dirs_model_path, detected, backend, tiles_range=tiles_range)
        
        if detected != "dirs":
            Converter.convert(str(Utils.dirs_root_from(Path(dirs_model_path))), output_type=detected, cleanup=True)
        return result

    @staticmethod
    def _verify_single_slice(run_path: Path, model_path: str | Path, detected: str, backend: str,
                             tiles_range: range | list[int] | None = None) -> dict:
        """Internal: verify exactly one slice (detects tiling)."""
        # --- Initialization ---
        run_path, dirs_root = Path(run_path), Utils.dirs_root_from(Path(model_path))
        run_meta = VerifierUtils.initialize_verify_metadata(run_path, dirs_root)

        if len(run_meta.slices) != 1:
            raise ValueError("Slices path must represent exactly one slice.")

        # --- Artifact Resolution ---
        (slice_id, meta), = run_meta.slices.items()
        preferred = (backend or "").lower()
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)
        
        # --- Execution ---
        if meta.tiling:
            success, tile_verifs = VerifierUtils.verify_tile_batch(slice_id, run_path, meta.tiling.num_tiles, preferred, slice_dir, meta, tiles_range=tiles_range)
            method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY
            verifs = {
                slice_id: SliceResult(
                    slice_id=slice_id,
                    success=bool(success),
                    method=method,
                    tiles=[TileResult.from_dict(t) if isinstance(t, dict) else t for t in tile_verifs],
                    error=None if success else "One or more tiles failed verification",
                ).to_dict()
            }
        else:
            proof_path = run_path / "proof.json"
            if not proof_path.exists(): raise FileNotFoundError(f"Proof file not found at {proof_path}")
            
            input_path, output_path = run_path / "input.json", run_path / "output.json"
            wf = VerifierUtils.get_witness_file(run_path, slice_id)
            witness_path = Path(wf) if wf else (run_path / "output_witness.bin")
            
            circuit_path = Utils.resolve_under_slice(slice_dir, meta.jstprove_circuit_path or meta.circuit_path)
            settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
            vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)

            ok = VerifierUtils.verify_with_backend(preferred, str(proof_path), str(circuit_path), str(input_path), str(output_path), str(witness_path), settings_path, vk_path)
            method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY
            verifs = {
                slice_id: SliceResult(
                    slice_id=slice_id,
                    success=bool(ok),
                    method=method,
                    error=None if ok else "verification_failed",
                ).to_dict()
            }

        # --- Result Persistence ---
        jst_v = 1 if method == ExecutionMethod.JSTPROVE_VERIFY and verifs[slice_id]['success'] else 0
        ezkl_v = 1 if method == ExecutionMethod.EZKL_VERIFY and verifs[slice_id]['success'] else 0
        return VerifierUtils.finalize_verify_results(run_path, verifs, jst_v, ezkl_v, 1)


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
    slices_dir = os.path.join(model_dir, "slices")
    slices_dir = os.path.join(slices_dir, "slice_0")  # give a single slice to test
    
    # Get run directory - use the latest run in the model's run directory
    run_dir = os.path.join(model_dir, "run")

    # Find the latest run
    run_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith("run_")])
    
    latest_run = run_dirs[-1]
    run_path = os.path.join(run_dir, latest_run)
    run_path = os.path.join(run_path, "slice_0")
    
    # Initialize verifier
    verifier = Verifier()
    
    # Run verification
    print(f"Verifying run {latest_run} for model {base_paths[model_choice]}...")
    results = verifier.verify(run_path, slices_dir, backend="jstprove",  tiles_range=[0, 1])
    
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