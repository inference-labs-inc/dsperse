"""
Orchestration for various provers.
"""
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, Union

from dsperse.src.analyzers.schema import TilingInfo, RunSliceMetadata, TileResult, SliceResult, Backend, ExecutionMethod, ExecutionChain, RunMetadata
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.prove.utils.prover_utils import ProverUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class Prover:
    """
    Orchestrator for proving model execution slices.
    """

    def __init__(self, parallel: int = 1):
        """
        Initialize the prover.

        Args:
            parallel: Number of parallel processes for proof generation (default: 1)
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

    def prove_dirs(self, run_path: str | Path, dirs_path: str | Path, output_path: str | Path | None = None,
                   backend: str | None = None) -> dict:
        """Prove all circuit-capable slices given a slices directory layout."""
        # --- Initialization ---
        run_path, dirs_path = Path(run_path), Utils.dirs_root_from(Path(dirs_path))
        metadata = ProverUtils.initialize_prove_metadata(run_path, dirs_path)

        # --- Slice Filtering ---
        slices_iter = ProverUtils.filter_provable_slices(metadata, backend, run_path)
        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to prove under run {run_path}.")
            return Utils.load_run_results(run_path)

        # --- Work Item Preparation ---
        work_items = self._prepare_work_items(slices_iter, dirs_path, run_path, output_path)
        if not work_items:
            return Utils.load_run_results(run_path)

        # --- Execution ---
        results = self._execute_proving(work_items)

        # --- Result Processing & Finalization ---
        proofs, proved_jst, proved_ezkl = self._process_results(results)
        return ProverUtils.finalize_prove_results(run_path, proofs, proved_jst, proved_ezkl, len(work_items))

    @staticmethod
    def _prepare_work_items(slices_iter: list, dirs_path: Path, run_path: Path, output_path: str | Path | None) -> list:
        """Prepare proving tasks for each slice."""
        work_items = []
        for slice_id, meta in slices_iter:
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)
            preferred = ProverUtils.select_backend(run_path, slice_id, meta)
            circuit_path, pk_path, settings_path = ProverUtils.resolve_prove_artifacts(slice_dir, meta, preferred)

            tiling = meta.tiling
            if tiling:
                witness_path = None
                proof_path = None
            else:
                if preferred == Backend.JSTPROVE:
                    wf = ProverUtils.get_witness_file(run_path, slice_id)
                    witness_path = Path(wf) if wf else (run_path / slice_id / "output_witness.bin")
                else:
                    witness_path = Utils.witness_path_for(run_path, slice_id)

                if not witness_path.exists():
                    logger.warning(f"Skipping {slice_id}: witness not found at {witness_path}")
                    continue

                proof_path = Utils.proof_output_path(run_path, slice_id, output_path)

                if not circuit_path or not os.path.exists(circuit_path):
                    logger.warning(f"Skipping {slice_id}: compiled circuit not found ({circuit_path})")
                    continue

            work_items.append((
                slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path,
                tiling.to_dict() if tiling else None, str(run_path), str(slice_dir)
            ))
        return work_items

    def _execute_proving(self, work_items: list) -> list:
        """Execute proving tasks in parallel or sequentially."""
        if self.parallel > 1 and len(work_items) > 1:
            logger.info(f"Proving {len(work_items)} slices with {self.parallel} parallel processes...")
            results = []
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(ProverUtils.prove_slice_logic, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    slice_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Slice {slice_id} proving failed: {e}")
                        results.append({'slice_id': slice_id, 'success': False, 'method': None, 'error': str(e)})
            return results
        else:
            return [ProverUtils.prove_slice_logic(item) for item in work_items]

    @staticmethod
    def _process_results(results: list) -> tuple[dict, int, int]:
        """Aggregate proof results and count successful backend executions."""
        proofs = {}
        proved_jst, proved_ezkl = 0, 0
        for result in results:
            slice_id = result['slice_id']
            proofs[slice_id] = result

            if result['success']:
                method = result.get('method')
                if method == ExecutionMethod.JSTPROVE_PROVE:
                    proved_jst += 1
                elif method == ExecutionMethod.EZKL_PROVE:
                    proved_ezkl += 1
            else:
                logger.error(f"Proof failed for {slice_id}: {result.get('error')}")
        return proofs, proved_jst, proved_ezkl

    def prove_dslice(self, run_path: str | Path, dslice_path: str | Path, output_path: str | Path | None = None,
                     backend: str | None = None) -> dict:
        """Prove slices packaged in .dslice format."""
        return self._prove_packaged(run_path, dslice_path, "dslice", output_path, backend)

    def prove_dsperse(self, run_path: str | Path, dsperse_path: str | Path, output_path: str | Path | None = None,
                      backend: str | None = None) -> dict:
        """Prove slices packaged in .dsperse format."""
        return self._prove_packaged(run_path, dsperse_path, "dsperse", output_path, backend)

    def _prove_packaged(self, run_path: str | Path, pkg_path: str | Path, pkg_type: str, 
                        output_path: str | Path | None, backend: str | None) -> dict:
        """Internal: Prove slices from a packaged format by temporarily converting to dirs."""
        temp_dirs = Converter.convert(pkg_path, output_type="dirs", cleanup=False)
        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        try:
            summary = self.prove_dirs(run_path, dirs_root, output_path, backend=backend)
        finally:
            Converter.convert(str(dirs_root), output_type=pkg_type, cleanup=True)
        return summary

    def prove(self, run_path: str | Path, model_dir: str | Path, output_path: str | Path | None = None,
              backend: str | None = None, tiles_range: range | list[int] | None = None) -> dict:
        """Route to the appropriate prove path based on `model_dir` packaging."""
        # --- Initialization & Root Detection ---
        run_path = Path(run_path)
        is_run_root = (run_path / "metadata.json").exists() or (run_path / "run_results.json").exists()

        # --- Run Type Detection ---
        # Improved slice run detection (standard or tiled)
        is_slice_run = ((run_path / "input.json").exists() and (run_path / "output.json").exists()) or \
                       (run_path / "split").exists() or (run_path / "tile_0").exists()

        detected = Converter.detect_type(model_dir)

        # --- Dispatch to Specific Handler ---
        if is_run_root:
            return self._handle_run_root_proving(run_path, model_dir, detected, output_path, backend)

        if is_slice_run:
            return self._handle_single_slice_proving(run_path, model_dir, detected, backend, tiles_range)

        raise FileNotFoundError(f"Run path invalid at {run_path}")

    def _handle_run_root_proving(self, run_path: Path, model_dir: str | Path, detected: str, 
                                 output_path: str | Path | None, backend: str | None) -> dict:
        """Internal: Handle proving for a full model run root."""
        if detected == "dslice":
            return self.prove_dslice(run_path, model_dir, output_path, backend=backend)
        if detected == "dsperse":
            return self.prove_dsperse(run_path, model_dir, output_path, backend=backend)
        if detected == "dirs":
            return self.prove_dirs(run_path, model_dir, output_path, backend=backend)
        raise ValueError(f"Unsupported data type: {detected}")

    def _handle_single_slice_proving(self, run_path: Path, model_dir: str | Path, detected: str, 
                                     backend: str | None, tiles_range: range | list[int] | None) -> dict:
        """Internal: Handle proving for a single slice run directory."""
        if backend not in (Backend.JSTPROVE, Backend.EZKL):
            raise ValueError("Single-slice proving requires explicit backend: 'jstprove' or 'ezkl'.")
        
        dirs_model_path = model_dir
        if detected != "dirs":
            dirs_model_path = Converter.convert(str(model_dir), output_type="dirs", cleanup=False)

        result = self._prove_single_slice(run_path, dirs_model_path, detected, backend, tiles_range=tiles_range)
        
        if detected != "dirs":
            Converter.convert(str(Utils.dirs_root_from(Path(dirs_model_path))), output_type=detected, cleanup=True)
        return result

    @staticmethod
    def _prove_single_slice(run_path: Path, model_dir: str | Path, detected: str, backend: str,
                            tiles_range: range | list[int] | None = None) -> dict:
        """Internal: prove exactly one slice (detects tiling)."""
        # --- Initialization ---
        model_dir_path = Path(model_dir)
        run_path, dirs_root = Path(run_path), Utils.dirs_root_from(model_dir_path)
        run_meta = ProverUtils.initialize_prove_metadata(run_path, dirs_root)

        target_name = model_dir_path.name
        if target_name.startswith("slice_") and target_name in run_meta.slices and len(run_meta.slices) > 1:
            run_meta.slices = {target_name: run_meta.slices[target_name]}

        if len(run_meta.slices) != 1:
            raise ValueError(f"Slices path must represent exactly one slice; found {len(run_meta.slices)}")

        # --- Artifact Resolution ---
        (slice_id, meta), = run_meta.slices.items()
        preferred = (backend or "").lower()
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)
        circuit_path, pk_path, settings_path = ProverUtils.resolve_prove_artifacts(slice_dir, meta, preferred)

        # --- Witness & Proof Path Logic ---
        if meta.tiling:
            witness_path, proof_path = None, None
        else:
            if preferred == Backend.JSTPROVE:
                wf = ProverUtils.get_witness_file(run_path, slice_id)
                witness_path = Path(wf) if wf else (run_path / "output_witness.bin")
            else:
                witness_path = run_path / "output.json"
            proof_path = run_path / "proof.json"

        # --- Execution ---
        result_dict = ProverUtils.execute_slice_proving(
            slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path,
            meta.tiling.to_dict() if meta.tiling else None, str(run_path), tiles_range=tiles_range
        )

        # --- Result Packaging & Persistence ---
        method = result_dict.get('method')
        proved_jst = 1 if method == ExecutionMethod.JSTPROVE_PROVE else 0
        proved_ezkl = 1 if method == ExecutionMethod.EZKL_PROVE else 0
        return ProverUtils.finalize_prove_results(run_path, {slice_id: result_dict}, proved_jst, proved_ezkl, 1)


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/age",
        5: "../models/version"
    }

    model_dir = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(model_dir, "slices") # slices dir, or single slice, or dsperse file
    slices_dir = os.path.join(slices_dir, "slice_0")  # give a single slice to test

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
    run_path = os.path.join(run_path, "slice_0")

    # Initialize prover
    prover = Prover()

    # Run proving
    print(f"Proving run {latest_run} for model {base_paths[model_choice]}...")
    results = prover.prove(run_path, slices_dir, backend="jstprove", tiles_range=[0, 1])

    # Display results
    print(f"\nProving completed!")

    # Print details for each slice
    print("\nSlice details:")
    for slice_result in results["execution_chain"]["execution_results"]:
        # print(f"\n{slice_result}:")
        slice_id = slice_result["slice_id"]
        if "proof_execution" in slice_result:
            success = slice_result["proof_execution"]["success"]
            status = "Success" if success else "Failed"
            time_taken = slice_result["proof_execution"]["time_sec"]
            print(f"  {slice_id}: {status} (Time: {time_taken:.2f}s)")
