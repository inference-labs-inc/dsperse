import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

from dsperse.src.analyzers.schema import Backend, RunSliceMetadata, ExecutionChain, ExecutionMethod, RunMetadata, TilingInfo, TileResult, SliceResult
from dsperse.src.backends.dispatch import WITNESS_FILENAME, instantiate_backend, resolve_circuit_path
from dsperse.src.utils.utils import Utils
from dsperse.src.utils.pipeline_utils import get_witness_backend, get_witness_file, select_backend, initialize_stage_metadata, filter_circuit_slices

logger = logging.getLogger(__name__)

class VerifierUtils:

    @staticmethod
    def get_proof_paths(run_results: dict) -> dict[str, str]:
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
        backend = instantiate_backend(backend_name)
        if backend_name == Backend.JSTPROVE:
            return backend.verify(
                proof_path=proof_path,
                circuit_path=circuit_path,
                input_path=input_path,
                output_path=output_path,
                witness_path=witness_path,
            )
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
        tile_name = f"tile_{tile_idx}"
        tile_run_dir = run_path / slice_id / tile_name if (run_path / slice_id / tile_name).exists() else run_path / tile_name
        tile_proof_path = tile_run_dir / "proof.json"

        if not tile_proof_path.exists():
            logger.warning(f"Proof missing for {slice_id}/{tile_name}, skipping")
            return False, "proof_missing"

        try:
            if preferred_backend == Backend.JSTPROVE:
                circuit_path = Utils.resolve_under_slice(slice_dir, resolve_circuit_path(meta, preferred_backend))
                input_path = tile_run_dir / "input.json"
                output_path = tile_run_dir / "output.json"
                tile_witness_path = tile_run_dir / WITNESS_FILENAME[Backend.JSTPROVE]

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
    def verify_slice_logic(args: tuple) -> dict:
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
