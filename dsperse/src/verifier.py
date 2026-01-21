"""
Orchestration for verifying proofs.
"""

import os
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.metadata.schema import TilingInfo, RunSliceMetadata, TileResult, SliceResult, Backend, ExecutionMethod, ExecutionChain, RunMetadata
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


def _verify_slice_worker(args: tuple) -> dict:
    """
    Worker function for parallel verification.
    Must be module-level for pickling by ProcessPoolExecutor.
    """
    (slice_id, preferred, proof_path, circuit_path, settings_path, vk_path, input_path, output_path, witness_path, tiling_info, run_path, _slice_dir) = args

    result = SliceResult(slice_id=slice_id, success=False)

    start = time.time()

    if tiling_info:
        tiling = TilingInfo.from_dict(tiling_info)
        num_tiles = tiling.num_tiles

        if num_tiles <= 0:
            result.success = False
            result.method = None
            result.tiles = []
            result.error = "no_tiles"
            result.time_sec = time.time() - start
            return result.to_dict()

        tile_verifs = []
        method = None

        for tile_idx in range(num_tiles):
            tile_name = f"tile_{tile_idx}"
            tile_run_dir = Path(run_path) / slice_id / tile_name
            tile_proof_path = tile_run_dir / "proof.json"

            if not tile_proof_path.exists():
                tile_verifs.append(TileResult(tile_idx=tile_idx, success=False, error="proof_missing"))
                continue

            try:
                if preferred == Backend.JSTPROVE:
                    from dsperse.src.backends.jstprove import JSTprove
                    tile_input_path = tile_run_dir / "input.json"
                    tile_output_path = tile_run_dir / "output.json"
                    tile_witness_path = tile_run_dir / "output_witness.bin"

                    missing = [p for p in [circuit_path, tile_input_path, tile_output_path, tile_witness_path] if not p or not Path(p).exists()]
                    if missing:
                        tile_verifs.append(TileResult(tile_idx=tile_idx, success=False, error=f"Missing files: {', '.join(map(str, missing))}"))
                        continue

                    backend = JSTprove()
                    ok = backend.verify(
                        proof_path=str(tile_proof_path),
                        circuit_path=str(circuit_path),
                        input_path=str(tile_input_path),
                        output_path=str(tile_output_path),
                        witness_path=str(tile_witness_path),
                    )
                    method = ExecutionMethod.JSTPROVE_VERIFY
                else:
                    from dsperse.src.backends.ezkl import EZKL
                    backend = EZKL()
                    ok = backend.verify(
                        proof_path=str(tile_proof_path),
                        settings_path=settings_path,
                        vk_path=vk_path,
                    )
                    method = ExecutionMethod.EZKL_VERIFY

                tile_verifs.append(TileResult(tile_idx=tile_idx, success=ok, method=method))
            except Exception as e:
                tile_verifs.append(TileResult(tile_idx=tile_idx, success=False, error=str(e)))

        result.success = all(v.success for v in tile_verifs)
        result.method = method
        result.tiles = tile_verifs
        result.error = None if result.success else "One or more tiles failed verification"
    else:
        try:
            if preferred == Backend.JSTPROVE:
                from dsperse.src.backends.jstprove import JSTprove
                missing = [p for p in [circuit_path, input_path, output_path, witness_path] if not p or not Path(p).exists()]
                if missing:
                    result.error = f"Missing files for JSTprove verify: {', '.join(map(str, missing))}"
                    result.method = ExecutionMethod.JSTPROVE_VERIFY
                else:
                    backend = JSTprove()
                    ok = backend.verify(
                        proof_path=str(proof_path),
                        circuit_path=str(circuit_path),
                        input_path=str(input_path),
                        output_path=str(output_path),
                        witness_path=str(witness_path),
                    )
                    result.success = ok
                    result.method = ExecutionMethod.JSTPROVE_VERIFY
            else:
                from dsperse.src.backends.ezkl import EZKL
                backend = EZKL()
                ok = backend.verify(
                    proof_path=str(proof_path),
                    settings_path=settings_path,
                    vk_path=vk_path,
                )
                result.success = ok
                result.method = ExecutionMethod.EZKL_VERIFY
        except Exception as e:
            result.error = str(e)
            result.method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY

    result.time_sec = time.time() - start
    return result.to_dict()


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

    # ------------------------ Small helpers (mirror Prover) ------------------------
    @staticmethod
    def _get_witness_backend_from_run(run_path: Path, slice_id: str) -> str | None:
        """Inspect run_results.json to see which backend produced the witness for a slice."""
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        exec_chain = ExecutionChain.from_dict((rr or {}).get("execution_chain"))
        entry = exec_chain.get_result_for_slice(slice_id)
        if not entry or not entry.witness_execution:
            return None
        w = entry.witness_execution
        method = (w.method or "").lower()
        if method == ExecutionMethod.TILED and w.tiles:
            method = (w.tiles[0].method or "").lower()
        if method.startswith(Backend.JSTPROVE):
            return Backend.JSTPROVE
        if method.startswith(Backend.EZKL):
            return Backend.EZKL
        return None

    @staticmethod
    def _get_witness_file_from_run(run_path: Path, slice_id: str) -> str | None:
        """Return concrete witness file path recorded by runner for a slice, if any."""
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        exec_chain = ExecutionChain.from_dict((rr or {}).get("execution_chain"))
        entry = exec_chain.get_result_for_slice(slice_id)
        if not entry or not entry.witness_execution:
            return None
        return entry.witness_execution.witness_file

    @staticmethod
    def _select_verification_backend(run_path: Path, slice_id: str, meta: RunSliceMetadata) -> str:
        """Prefer the backend that produced the witness; else meta backend; default jstprove."""
        from_run = Verifier._get_witness_backend_from_run(run_path, slice_id)
        if from_run in (Backend.JSTPROVE, Backend.EZKL):
            return from_run
        meta_backend = (meta.backend or "").lower()
        if meta_backend in (Backend.JSTPROVE, Backend.EZKL):
            return meta_backend
        return Backend.JSTPROVE

    def _verify_tile(
            self,
            tile_idx: int,
            slice_id: str,
            run_path: Path,
            preferred_backend: str,
            slice_dir: Path,
            meta: RunSliceMetadata,
    ) -> tuple[bool, str | None]:
        """Verify a single tile within a slice."""
        tile_name = f"tile_{tile_idx}"

        if (run_path / slice_id / tile_name).exists():
            tile_run_dir = run_path / slice_id / tile_name
        else:
            tile_run_dir = run_path / tile_name

        tile_proof_path = tile_run_dir / "proof.json"
        if not tile_proof_path.exists():
            logger.warning(f"Proof missing for {slice_id}/{tile_name}, skipping")
            return False, "proof_missing"

        if preferred_backend == Backend.JSTPROVE:
            if self.jstprove_runner is None:
                return False, "jstprove_unavailable"
            circuit_path = Utils.resolve_under_slice(slice_dir, meta.jstprove_circuit_path or meta.circuit_path)
            input_path = tile_run_dir / "input.json"
            output_path = tile_run_dir / "output.json"
            tile_witness_path = tile_run_dir / "output_witness.bin"

            missing = [p for p in [circuit_path, input_path, output_path, tile_witness_path] if
                       not p or not Path(p).exists()]
            if missing:
                return False, f"Missing files for tile verify: {', '.join(map(str, missing))}"
            try:
                ok = self.jstprove_runner.verify(
                    proof_path=str(tile_proof_path),
                    circuit_path=str(circuit_path),
                    input_path=str(input_path),
                    output_path=str(output_path),
                    witness_path=str(tile_witness_path),
                )
                return ok, None if ok else "verification_failed"
            except Exception as e:
                return False, str(e)
        else:
            if self.ezkl_runner is None:
                return False, "ezkl_unavailable"
            settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
            vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)
            try:
                ok = self.ezkl_runner.verify(proof_path=str(tile_proof_path), settings_path=settings_path,
                                             vk_path=vk_path)
                return ok, None if ok else "verification_failed"
            except Exception as e:
                return False, str(e)

    def _verify_tile_batch(
            self,
            slice_id: str,
            run_path: Path,
            num_tiles: int,
            preferred_backend: str,
            slice_dir: Path,
            meta: RunSliceMetadata,
            tiles_range: range | list[int] | None = None,
    ) -> tuple[bool, list[dict]]:
        """Verify a subset or all tiles for a slice."""
        target_tiles = tiles_range if tiles_range is not None else range(num_tiles)
        logger.info(f"Verifying tiled slice {slice_id} (indices: {list(target_tiles)})...")
        tile_verifs = []

        for tile_idx in target_tiles:
            start = time.time()
            ok, res = self._verify_tile(tile_idx, slice_id, run_path, preferred_backend, slice_dir, meta)
            tile_verifs.append({
                "tile_idx": tile_idx,
                "success": ok,
                "time_sec": time.time() - start,
                "error": res
            })

        success = all(v["success"] for v in tile_verifs)
        return success, tile_verifs

    def verify_dirs(self, run_path: str | Path, dirs_path: str | Path, backend: str | None = None) -> dict:
        """Verify proofs for circuit-capable slices (JSTprove and EZKL)."""
        run_path = Path(run_path)
        dirs_path = Utils.dirs_root_from(Path(dirs_path))
        metadata = RunMetadata.from_dict(Utils.load_run_metadata(run_path))
        run_results = Utils.load_run_results(run_path)

        proof_paths_by_slice = {}
        try:
            results_chain = ExecutionChain.from_dict((run_results or {}).get("execution_chain"))
            for entry in results_chain.execution_results:
                if entry.proof_execution and entry.proof_execution.proof_path:
                    proof_paths_by_slice[entry.slice_id] = entry.proof_execution.proof_path
        except Exception:
            pass

        verifs: dict[str, dict] = {}
        jst_verified = 0
        ezkl_verified = 0

        slices_iter = list(metadata.iter_circuit_slices())
        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to verify under run {run_path}. Nothing to do.")
            return run_results

        if backend in (Backend.JSTPROVE, Backend.EZKL):
            filtered = []
            for slice_id, meta in slices_iter:
                wb = self._get_witness_backend_from_run(Path(run_path), slice_id)
                if wb == backend:
                    filtered.append((slice_id, meta))
            slices_iter = filtered
            if not slices_iter:
                logger.info(f"No slices found with witness backend '{backend}' to verify under run {run_path}.")
                return run_results

        work_items = []
        for slice_id, meta in slices_iter:
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)
            preferred = self._select_verification_backend(run_path, slice_id, meta)
            tiling = meta.tiling

            circuit_path = Utils.resolve_under_slice(slice_dir, meta.circuit_path)
            settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
            vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)

            if tiling:
                proof_path = None
                input_path = None
                output_path = None
                witness_path = None
            else:
                if slice_id in proof_paths_by_slice:
                    proof_path = Path(proof_paths_by_slice[slice_id])
                else:
                    proof_path = Path(run_path) / slice_id / "proof.json"

                if not proof_path.exists():
                    logger.warning(f"Skipping {slice_id}: proof not found at {proof_path}")
                    continue

                input_path = Path(run_path) / slice_id / "input.json"
                output_path = Path(run_path) / slice_id / "output.json"
                wf = self._get_witness_file_from_run(run_path, slice_id)
                witness_path = Path(wf) if wf else (Path(run_path) / slice_id / "output_witness.bin")

                if preferred == Backend.EZKL:
                    if not settings_path or not os.path.exists(settings_path):
                        logger.warning(f"Skipping {slice_id}: settings file not found ({settings_path})")
                        continue
                    if not vk_path or not os.path.exists(vk_path):
                        logger.warning(f"Skipping {slice_id}: verification key not found ({vk_path})")
                        continue

            work_items.append((slice_id, preferred, proof_path, circuit_path, settings_path, vk_path, input_path, output_path, witness_path, tiling.to_dict() if tiling else None, str(run_path), str(slice_dir)))

        total = len(work_items)

        if self.parallel > 1 and len(work_items) > 1:
            logger.info(f"Verifying {len(work_items)} slices with {self.parallel} parallel processes...")
            results = []
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(_verify_slice_worker, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    slice_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Slice {slice_id} verification failed: {e}")
                        results.append(SliceResult(slice_id=slice_id, success=False, error=str(e)).to_dict())
        else:
            results = [_verify_slice_worker(item) for item in work_items]

        for result in results:
            slice_id = result['slice_id']
            verifs[slice_id] = result

            if result['success']:
                method = result.get('method')
                if method == ExecutionMethod.JSTPROVE_VERIFY:
                    jst_verified += 1
                elif method == ExecutionMethod.EZKL_VERIFY:
                    ezkl_verified += 1

        run_results, verified_count = Utils.merge_execution_into_run_results(run_results, verifs, "verification")
        exec_chain = run_results.setdefault("execution_chain", {})
        exec_chain["jstprove_verified_slices"] = int(jst_verified)
        exec_chain["ezkl_verified_slices"] = int(ezkl_verified)
        Utils.save_run_results(run_path, run_results)
        Utils.update_metadata_after_execution(run_path, total, (jst_verified + ezkl_verified), "verification")
        return run_results

    def verify_dslice(self, run_path: str | Path, dslice_path: str | Path, backend: str | None = None) -> dict:
        temp_dirs = Converter.convert(str(dslice_path), output_type="dirs", cleanup=False)
        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        result = self.verify_dirs(run_path, dirs_root, backend=backend)
        Converter.convert(str(dirs_root), output_type="dslice", cleanup=False)
        return result

    def verify_dsperse(self, run_path: str | Path, dsperse_path: str | Path, backend: str | None = None) -> dict:
        temp_dirs = Converter.convert(dsperse_path, output_type="dirs", cleanup=False)
        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        result = self.verify_dirs(run_path, dirs_root, backend=backend)
        Converter.convert(str(dirs_root), output_type="dsperse", cleanup=False)
        return result

    def verify(self, run_path: str | Path, model_path: str | Path, backend: str | None = None,
               tiles_range: range | list[int] | None = None) -> dict:
        """Verify proofs (supports full runs, packaged formats, and single slices)."""
        run_path = Path(run_path)
        is_run_root = (run_path / "metadata.json").exists()
        is_slice_run = ((run_path / "input.json").exists() and (run_path / "output.json").exists()) or \
                       (run_path / "split").exists() or (run_path / "tile_0").exists()

        detected = Converter.detect_type(model_path)

        if is_run_root:
            Utils.load_run_metadata(run_path)
            if detected == "dslice": return self.verify_dslice(run_path, model_path, backend=backend)
            if detected == "dsperse": return self.verify_dsperse(run_path, model_path, backend=backend)
            if detected == "dirs": return self.verify_dirs(run_path, model_path, backend=backend)
            raise ValueError(f"Unsupported data type: {detected}")

        if is_slice_run:
            if backend not in (Backend.JSTPROVE, Backend.EZKL):
                raise ValueError("Single-slice verification requires explicit backend.")
            dirs_model_path = model_path
            if detected != "dirs":
                dirs_model_path = Converter.convert(str(model_path), output_type="dirs", cleanup=False)
            result = self._verify_single_slice(run_path, dirs_model_path, detected, backend, tiles_range=tiles_range)
            if detected != "dirs":
                Converter.convert(str(Utils.dirs_root_from(Path(dirs_model_path))), output_type=detected, cleanup=True)
            return result

        raise FileNotFoundError(f"Run path invalid at {run_path}")

    def _verify_single_slice(self, run_path: Path, model_path: str | Path, detected: str, backend: str,
                             tiles_range: range | list[int] | None = None) -> dict:
        """Internal: verify exactly one slice (detects tiling)."""
        sdir = Path(model_path)
        run_meta_dict = RunnerAnalyzer.generate_run_metadata(Path(sdir if sdir.is_dir() else Path(sdir)), save_path=None,
                                                              original_format=detected)
        run_meta = RunMetadata.from_dict(run_meta_dict)
        if len(run_meta.slices) != 1:
            raise ValueError("Slices path must represent exactly one slice.")

        (slice_id, meta), = run_meta.slices.items()
        preferred = (backend or "").lower()
        dirs_root = Utils.dirs_root_from(Path(model_path))
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)

        tiling = meta.tiling
        if tiles_range is not None and not tiling:
            logger.warning(f"tiles_range provided for non-tiled slice {slice_id}; ignoring")
            tiles_range = None
        if tiling:
            num_tiles = tiling.num_tiles
            start = time.time()
            success, tile_verifs = self._verify_tile_batch(slice_id, Path(run_path), num_tiles, preferred, slice_dir,
                                                           meta, tiles_range=tiles_range)
            elapsed = time.time() - start
            verifs = {
                slice_id: SliceResult(
                    slice_id=slice_id,
                    success=bool(success),
                    time_sec=elapsed,
                    method=ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY,
                    tiles=[TileResult.from_dict(t) if isinstance(t, dict) else t for t in tile_verifs],
                    error=None if success else "One or more tiles failed verification",
                ).to_dict()
            }
        else:
            proof_path = Path(run_path) / "proof.json"
            if not proof_path.exists(): raise FileNotFoundError(f"Proof file not found at {proof_path}")
            start = time.time()
            success = False
            error_msg = None
            if preferred == Backend.JSTPROVE:
                circuit_path = Utils.resolve_under_slice(slice_dir, meta.jstprove_circuit_path or meta.circuit_path)
                input_path, output_path = Path(run_path) / "input.json", Path(run_path) / "output.json"
                wf = self._get_witness_file_from_run(run_path, slice_id)
                witness_path = Path(wf) if wf else (Path(run_path) / "output_witness.bin")
                try:
                    success = self.jstprove_runner.verify(str(proof_path), str(circuit_path), str(input_path),
                                                          str(output_path), str(witness_path))
                except Exception as e:
                    error_msg = str(e)
            else:
                vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)
                settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
                try:
                    success = self.ezkl_runner.verify(str(proof_path), settings_path=settings_path, vk_path=vk_path)
                except Exception as e:
                    error_msg = str(e)

            elapsed = time.time() - start
            verifs = {
                slice_id: SliceResult(
                    slice_id=slice_id,
                    success=bool(success),
                    time_sec=elapsed,
                    method=ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY,
                    error=None if success else (error_msg or "verification_failed"),
                ).to_dict()
            }

        run_results = Utils.load_run_results(Path(run_path))
        run_results, _ = Utils.merge_execution_into_run_results(run_results, verifs, "verification")
        exec_chain = run_results.setdefault("execution_chain", {})
        if preferred == Backend.JSTPROVE:
            exec_chain["jstprove_verified_slices"] = int(exec_chain.get("jstprove_verified_slices", 0)) + int(success)
        else:
            exec_chain["ezkl_verified_slices"] = int(exec_chain.get("ezkl_verified_slices", 0)) + int(success)
        Utils.save_run_results(Path(run_path), run_results)
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