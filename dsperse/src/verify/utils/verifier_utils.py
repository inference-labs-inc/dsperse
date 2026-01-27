import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union, List

from dsperse.src.analyzers.schema import Backend, RunSliceMetadata, ExecutionChain, ExecutionMethod, RunMetadata, TilingInfo, TileResult, SliceResult
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class VerifierUtils:
    """Utility class for verification operations."""

    @staticmethod
    def get_witness_backend(run_path: Path, slice_id: str) -> str | None:
        """Inspect run_results.json to see which backend produced the witness for a slice."""
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        if not rr:
            return None
        exec_chain = ExecutionChain.from_dict(rr.get("execution_chain"))
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
    def get_witness_file(run_path: Path, slice_id: str) -> str | None:
        """Return concrete witness file path recorded by runner for a slice, if any."""
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        if not rr:
            return None
        exec_chain = ExecutionChain.from_dict(rr.get("execution_chain"))
        entry = exec_chain.get_result_for_slice(slice_id)
        if not entry or not entry.witness_execution:
            return None
        return entry.witness_execution.witness_file

    @staticmethod
    def select_backend(run_path: Path, slice_id: str, meta: RunSliceMetadata) -> str:
        """Prefer the backend that produced the witness; else meta backend; default jstprove."""
        from_run = VerifierUtils.get_witness_backend(run_path, slice_id)
        if from_run in (Backend.JSTPROVE, Backend.EZKL):
            return from_run
        meta_backend = (meta.backend or "").lower()
        if meta_backend in (Backend.JSTPROVE, Backend.EZKL):
            return meta_backend
        return Backend.JSTPROVE

    @staticmethod
    def initialize_verify_metadata(run_path: Path, dirs_path: Path) -> RunMetadata:
        """Initialize metadata for verification."""
        slices_metadata_path = dirs_path / "metadata.json"
        if slices_metadata_path.exists():
            from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
            run_meta_dict = RunnerAnalyzer.generate_run_metadata(dirs_path, save_path=None, original_format="dirs")
            return RunMetadata.from_dict(run_meta_dict)
        else:
            return RunMetadata.from_dict(Utils.load_run_metadata(run_path))

    @staticmethod
    def filter_verifiable_slices(metadata: RunMetadata, backend: Optional[str], run_path: Path) -> list[tuple[str, RunSliceMetadata]]:
        """Filter circuit-capable slices that match the requested backend for verification."""
        slices_iter = list(metadata.iter_circuit_slices())
        if not backend or backend not in (Backend.JSTPROVE, Backend.EZKL):
            return slices_iter

        filtered = []
        for slice_id, meta in slices_iter:
            wb = VerifierUtils.get_witness_backend(run_path, slice_id)
            if wb == backend:
                filtered.append((slice_id, meta))
        return filtered

    @staticmethod
    def get_proof_paths(run_results: dict) -> dict[str, str]:
        """Map slice IDs to proof paths from run results."""
        proof_paths = {}
        try:
            results_chain = ExecutionChain.from_dict((run_results or {}).get("execution_chain"))
            for entry in results_chain.execution_results:
                if entry.proof_execution and entry.proof_execution.proof_path:
                    proof_paths[entry.slice_id] = entry.proof_execution.proof_path
        except Exception:
            pass
        return proof_paths

    @staticmethod
    def finalize_verify_results(run_path: Path, verifs: dict, jst_verified: int, ezkl_verified: int, total: int):
        """Update run results and metadata after verification completion."""
        run_results = Utils.load_run_results(run_path)
        run_results, _ = Utils.merge_execution_into_run_results(run_results, verifs, "verification")
        exec_chain = run_results.setdefault("execution_chain", {})
        exec_chain["jstprove_verified_slices"] = int(jst_verified)
        exec_chain["ezkl_verified_slices"] = int(ezkl_verified)
        Utils.save_run_results(run_path, run_results)
        Utils.update_metadata_after_execution(run_path, total, (jst_verified + ezkl_verified), "verification")
        return run_results

    @staticmethod
    def verify_with_backend(
        backend_name: str,
        proof_path: str,
        circuit_path: Optional[str],
        input_path: Optional[str],
        output_path: Optional[str],
        witness_path: Optional[str],
        settings_path: Optional[str] = None,
        vk_path: Optional[str] = None,
    ) -> bool:
        """Dispatch verification to the appropriate backend."""
        if backend_name == Backend.JSTPROVE:
            from dsperse.src.backends.jstprove import JSTprove
            backend = JSTprove()
            return backend.verify(
                proof_path=proof_path,
                circuit_path=circuit_path,
                input_path=input_path,
                output_path=output_path,
                witness_path=witness_path,
            )
        else:
            from dsperse.src.backends.ezkl import EZKL
            backend = EZKL()
            return backend.verify(
                proof_path=proof_path,
                settings_path=settings_path,
                vk_path=vk_path,
            )

    @staticmethod
    def verify_tile(
        tile_idx: int,
        slice_id: str,
        run_path: Path,
        preferred_backend: str,
        slice_dir: Path,
        meta: RunSliceMetadata,
    ) -> tuple[bool, str | None]:
        """Verify a single tile proof."""
        tile_name = f"tile_{tile_idx}"
        tile_run_dir = run_path / slice_id / tile_name if (run_path / slice_id / tile_name).exists() else run_path / tile_name
        tile_proof_path = tile_run_dir / "proof.json"

        if not tile_proof_path.exists():
            logger.warning(f"Proof missing for {slice_id}/{tile_name}, skipping")
            return False, "proof_missing"

        try:
            if preferred_backend == Backend.JSTPROVE:
                circuit_path = Utils.resolve_under_slice(slice_dir, meta.jstprove_circuit_path or meta.circuit_path)
                input_path = tile_run_dir / "input.json"
                output_path = tile_run_dir / "output.json"
                tile_witness_path = tile_run_dir / "output_witness.bin"

                missing = [p for p in [circuit_path, input_path, output_path, tile_witness_path] if not p or not Path(p).exists()]
                if missing:
                    return False, f"Missing files for tile verify: {', '.join(map(str, missing))}"

                ok = VerifierUtils.verify_with_backend(Backend.JSTPROVE, str(tile_proof_path), str(circuit_path), str(input_path), str(output_path), str(tile_witness_path))
                return ok, None if ok else "verification_failed"
            else:
                settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
                vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)
                ok = VerifierUtils.verify_with_backend(Backend.EZKL, str(tile_proof_path), None, None, None, None, str(settings_path), str(vk_path))
                return ok, None if ok else "verification_failed"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def verify_tile_batch(
        slice_id: str,
        run_path: Path,
        num_tiles: int,
        preferred_backend: str,
        slice_dir: Path,
        meta: RunSliceMetadata,
        tiles_range: Optional[Union[range, list[int]]] = None,
    ) -> tuple[bool, list[dict]]:
        """Verify proofs for a batch of tiles."""
        target_tiles = tiles_range if tiles_range is not None else range(num_tiles)
        logger.info(f"Verifying tiled slice {slice_id} (indices: {list(target_tiles)})...")
        tile_verifs = []

        for tile_idx in target_tiles:
            start_time = time.time()
            ok, res = VerifierUtils.verify_tile(tile_idx, slice_id, run_path, preferred_backend, slice_dir, meta)
            tile_verifs.append({
                "tile_idx": tile_idx,
                "success": ok,
                "time_sec": time.time() - start_time,
                "error": res
            })

        success = all(v["success"] for v in tile_verifs)
        return success, tile_verifs

    @staticmethod
    def parse_tiles_range(tiles_str: str | None) -> range | list[int] | None:
        """Parse a tile string into a range or list of integers."""
        if not tiles_str:
            return None

        if '-' in tiles_str:
            try:
                start, end = map(int, tiles_str.split('-'))
                return range(start, end + 1)
            except ValueError:
                return None

        if ',' in tiles_str:
            try:
                return [int(x.strip()) for x in tiles_str.split(',')]
            except ValueError:
                return None

        try:
            return [int(tiles_str)]
        except ValueError:
            return None

    @staticmethod
    def verify_slice_logic(args: tuple) -> dict:
        """Core logic for verifying a single slice (handles tuple for pool compatibility)."""
        (slice_id, preferred, proof_path, circuit_path, settings_path, vk_path, input_path, output_path, witness_path, tiling_info, run_path, slice_dir) = args
        
        result = SliceResult(slice_id=slice_id, success=False)
        start_time = time.time()

        if tiling_info:
            tiling = TilingInfo.from_dict(tiling_info)
            success, tile_verifs = VerifierUtils.verify_tile_batch(slice_id, Path(run_path), tiling.num_tiles, preferred, Path(slice_dir), RunSliceMetadata.from_dict({'tiling': tiling_info}))
            result.success = success
            result.method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY
            result.tiles = [TileResult.from_dict(v) for v in tile_verifs]
            result.error = None if success else "One or more tiles failed verification"
        else:
            try:
                ok = VerifierUtils.verify_with_backend(preferred, str(proof_path), circuit_path, input_path, output_path, witness_path, settings_path, vk_path)
                result.success = ok
                result.method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY
                result.error = None if ok else "verification_failed"
            except Exception as e:
                result.error = str(e)
                result.method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY

        result.time_sec = time.time() - start_time
        return result.to_dict()
