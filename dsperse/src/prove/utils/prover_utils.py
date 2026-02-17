import logging
import os
from pathlib import Path
from typing import Optional, Union

from dsperse.src.analyzers.schema import Backend, RunSliceMetadata, ExecutionMethod, TilingInfo, TileResult, SliceResult
from dsperse.src.backends.dispatch import WITNESS_FILENAME, instantiate_backend, resolve_circuit_path
from dsperse.src.utils.utils import Utils
from dsperse.src.utils.pipeline_utils import get_witness_file, select_backend, initialize_stage_metadata, filter_circuit_slices

logger = logging.getLogger(__name__)

class ProverUtils:

    @staticmethod
    def resolve_prove_artifacts(
        slice_dir: Path,
        meta: RunSliceMetadata,
        backend: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        circuit_path = Utils.resolve_under_slice(slice_dir, resolve_circuit_path(meta, backend))
        pk_path = Utils.resolve_under_slice(slice_dir, meta.pk_path)
        settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)

        if settings_path and not os.path.exists(settings_path):
            logger.warning(f"Settings file not found at {settings_path}; proceeding without it.")
            settings_path = None

        return circuit_path, pk_path, settings_path

    @staticmethod
    def finalize_prove_results(run_path: Path, proofs: dict, proved_jst: int, proved_ezkl: int, total: int):
        run_results = Utils.load_run_results(run_path)
        run_results, _ = Utils.merge_execution_into_run_results(run_results, proofs, "proof")
        exec_chain = run_results.setdefault("execution_chain", {})
        exec_chain["jstprove_proved_slices"] = int(proved_jst)
        exec_chain["ezkl_proved_slices"] = int(proved_ezkl)
        exec_chain["ezkl_verified_slices"] = exec_chain.get("ezkl_verified_slices", 0) if proved_ezkl == 0 else 0

        Utils.save_run_results(run_path, run_results)
        Utils.update_metadata_after_execution(run_path, total, proved_jst + proved_ezkl, "proof")
        return run_results

    @staticmethod
    def prove_with_backend(
        backend_name: str,
        witness_path: Path,
        circuit_path: str,
        proof_path: Path,
        pk_path: str | None,
        settings_path: str | None,
    ) -> tuple[bool, str, str | Path | None]:
        if backend_name == Backend.JSTPROVE:
            method = ExecutionMethod.JSTPROVE_PROVE
        elif backend_name == Backend.EZKL:
            method = ExecutionMethod.EZKL_PROVE
        else:
            return False, "unknown", f"Unknown backend: {backend_name}"

        if backend_name == Backend.EZKL and (not pk_path or not os.path.exists(pk_path)):
            return False, method, f"Proving key not found at {pk_path}"

        try:
            backend = instantiate_backend(backend_name)
            if backend_name == Backend.JSTPROVE:
                ok, res = backend.prove(
                    witness_path=str(witness_path),
                    circuit_path=str(circuit_path),
                    proof_path=str(proof_path),
                )
            else:
                ok, res = backend.prove(
                    witness_path=str(witness_path),
                    model_path=str(circuit_path),
                    proof_path=str(proof_path),
                    pk_path=str(pk_path),
                    settings_path=settings_path,
                )
            return ok, method, res
        except Exception as e:
            return False, method, str(e)

    @staticmethod
    def prove_tile(
        tile_idx: int,
        slice_id: str,
        run_path: Path,
        preferred_backend: str,
        circuit_path: str,
        pk_path: str | None,
        settings_path: str | None,
        output_path: str | Path | None = None,
    ) -> dict:
        import time
        tile_name = f"tile_{tile_idx}"

        if (run_path / slice_id / tile_name).exists():
            tile_run_dir = run_path / slice_id / tile_name
            proof_rel_path = os.path.join(slice_id, tile_name)
        else:
            tile_run_dir = run_path / tile_name
            proof_rel_path = tile_name

        tile_witness_path = tile_run_dir / WITNESS_FILENAME[preferred_backend]

        tile_proof_path = Utils.proof_output_path(run_path, proof_rel_path, output_path)
        os.makedirs(tile_proof_path.parent, exist_ok=True)

        if not tile_witness_path.exists():
            logger.warning(f"Witness missing for {slice_id}/{tile_name}, skipping")
            return {"tile_idx": tile_idx, "success": False, "error": "witness_missing", "time_sec": 0}

        if not circuit_path or not os.path.exists(circuit_path):
            logger.error(f"Circuit not found for tile {slice_id}/{tile_name}: {circuit_path}")
            return {"tile_idx": tile_idx, "success": False, "error": "circuit_missing", "time_sec": 0}

        tile_start = time.time()
        ok, method, res = ProverUtils.prove_with_backend(
            preferred_backend, tile_witness_path, str(circuit_path), tile_proof_path, pk_path, settings_path
        )
        tile_elapsed = time.time() - tile_start

        return {
            "tile_idx": tile_idx,
            "success": ok,
            "proof_path": str(tile_proof_path),
            "time_sec": tile_elapsed,
            "method": method,
            "error": None if ok else str(res)
        }

    @staticmethod
    def prove_tile_batch(
            slice_id: str,
            run_path: Path,
            num_tiles: int,
            preferred_backend: str,
            circuit_path: str,
            pk_path: str | None,
            settings_path: str | None,
            output_path: str | Path | None = None,
            tiles_range: Union[range, list[int], None] = None,
    ) -> tuple[bool, str | None, list[dict]]:
        target_tiles = tiles_range if tiles_range is not None else range(num_tiles)
        logger.info(f"Proving tiled slice {slice_id} (indices: {list(target_tiles)})...")
        tile_results = []
        method = None

        for tile_idx in target_tiles:
            res = ProverUtils.prove_tile(
                tile_idx, slice_id, run_path, preferred_backend, circuit_path, pk_path, settings_path, output_path
            )
            tile_results.append(res)
            if res["success"]:
                method = res["method"]

        success = all(r["success"] for r in tile_results)
        return success, method, tile_results

    @staticmethod
    def prove_slice_logic(args: tuple) -> dict:
        (slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path, tiling_info, run_path) = args
        return ProverUtils.execute_slice_proving(
            slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path, tiling_info, run_path
        )

    @staticmethod
    def execute_slice_proving(
        slice_id: str,
        preferred: str,
        witness_path: Optional[Union[str, Path]],
        circuit_path: str,
        proof_path: Optional[Union[str, Path]],
        pk_path: Optional[str],
        settings_path: Optional[str],
        tiling_info: Optional[dict],
        run_path: str,
        tiles_range: Optional[Union[range, list[int]]] = None
    ) -> dict:
        import time
        result = SliceResult(
            slice_id=slice_id,
            success=False,
            proof_path=str(proof_path) if (proof_path and not tiling_info) else None,
        )

        start = time.time()

        if tiling_info:
            tiling = TilingInfo.from_dict(tiling_info)
            success, method, tile_results = ProverUtils.prove_tile_batch(
                slice_id, Path(run_path), tiling.num_tiles, preferred, str(circuit_path), pk_path, settings_path, tiles_range=tiles_range
            )
            result.success = success
            result.method = method
            result.tiles = [TileResult.from_dict(t) if isinstance(t, dict) else t for t in tile_results]
            result.error = None if success else "One or more tiles failed to prove"
        else:
            if not proof_path:
                raise ValueError(f"proof_path required for non-tiled slice {slice_id}")
            os.makedirs(Path(proof_path).parent, exist_ok=True)
            ok, method, res = ProverUtils.prove_with_backend(
                preferred, Path(witness_path), str(circuit_path), Path(proof_path), pk_path, settings_path
            )
            result.success = ok
            result.method = method
            result.error = None if ok else str(res)

        result.time_sec = time.time() - start
        return result.to_dict()
