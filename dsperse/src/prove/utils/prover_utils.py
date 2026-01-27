import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

from dsperse.src.analyzers.schema import Backend, RunSliceMetadata, ExecutionChain, ExecutionMethod, RunMetadata, TilingInfo, TileResult, SliceResult
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class ProverUtils:
    """Utility class for proving operations."""

    @staticmethod
    def resolve_prove_artifacts(
        slice_dir: Path,
        meta: RunSliceMetadata,
        backend: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """Resolve circuit, pk, and settings paths under a given slice directory."""
        b = (backend or "").lower()
        if b == Backend.EZKL:
            circuit_rel = meta.ezkl_circuit_path or meta.circuit_path
            pk_rel = meta.pk_path
            settings_rel = meta.settings_path
        elif b == Backend.JSTPROVE:
            circuit_rel = meta.jstprove_circuit_path or meta.circuit_path
            pk_rel = meta.pk_path
            settings_rel = meta.settings_path
        else:
            circuit_rel = meta.circuit_path
            pk_rel = meta.pk_path
            settings_rel = meta.settings_path

        circuit_path = Utils.resolve_under_slice(slice_dir, circuit_rel)
        pk_path = Utils.resolve_under_slice(slice_dir, pk_rel)
        settings_path = Utils.resolve_under_slice(slice_dir, settings_rel)

        if settings_path and not os.path.exists(settings_path):
            logger.warning(f"Settings file not found at {settings_path}; proceeding without it.")
            settings_path = None

        return circuit_path, pk_path, settings_path

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
    def select_backend(run_path: Path, slice_id: str, meta: RunSliceMetadata) -> str:
        """Choose proving backend based on witness backend, metadata, or default to jstprove."""
        from_run = ProverUtils.get_witness_backend(run_path, slice_id)
        if from_run in (Backend.JSTPROVE, Backend.EZKL):
            return from_run
        meta_backend = (meta.backend or "").lower()
        if meta_backend in (Backend.JSTPROVE, Backend.EZKL):
            return meta_backend
        return Backend.JSTPROVE

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
    def initialize_prove_metadata(run_path: Path, dirs_path: Path) -> RunMetadata:
        """Initialize metadata for proving from either slices directory or run directory."""
        slices_metadata_path = dirs_path / "metadata.json"
        if slices_metadata_path.exists():
            from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
            run_meta_dict = RunnerAnalyzer.generate_run_metadata(dirs_path, save_path=None, original_format="dirs")
            return RunMetadata.from_dict(run_meta_dict)
        else:
            return RunMetadata.from_dict(Utils.load_run_metadata(run_path))

    @staticmethod
    def filter_provable_slices(metadata: RunMetadata, backend: Optional[str], run_path: Path) -> list[tuple[str, RunSliceMetadata]]:
        """Filter circuit-capable slices that match the requested backend."""
        slices_iter = list(metadata.iter_circuit_slices())
        if not backend or backend not in (Backend.JSTPROVE, Backend.EZKL):
            return slices_iter

        filtered = []
        for slice_id, meta in slices_iter:
            wb = ProverUtils.get_witness_backend(run_path, slice_id)
            if wb == backend:
                filtered.append((slice_id, meta))
        return filtered

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
    def finalize_prove_results(run_path: Path, proofs: dict, proved_jst: int, proved_ezkl: int, total: int):
        """Update run results and metadata after proving completion."""
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
        """Dispatch to the requested backend for proving."""
        if backend_name == Backend.JSTPROVE:
            try:
                from dsperse.src.backends.jstprove import JSTprove
                backend = JSTprove()
                ok, res = backend.prove(
                    witness_path=str(witness_path),
                    circuit_path=str(circuit_path),
                    proof_path=str(proof_path),
                )
                return ok, ExecutionMethod.JSTPROVE_PROVE, res
            except Exception as e:
                return False, ExecutionMethod.JSTPROVE_PROVE, str(e)

        if not pk_path or not os.path.exists(pk_path):
            return False, ExecutionMethod.EZKL_PROVE, f"Proving key not found at {pk_path}"
        try:
            from dsperse.src.backends.ezkl import EZKL
            backend = EZKL()
            ok, res = backend.prove(
                witness_path=str(witness_path),
                model_path=str(circuit_path),
                proof_path=str(proof_path),
                pk_path=str(pk_path),
                settings_path=settings_path,
            )
            return ok, ExecutionMethod.EZKL_PROVE, res
        except Exception as e:
            return False, ExecutionMethod.EZKL_PROVE, str(e)

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
        """Prove a single tile within a slice."""
        import time
        tile_name = f"tile_{tile_idx}"

        # Determine path layout: run_path/slice_id/tile_N (run-root) or run_path/tile_N (single-slice)
        if (run_path / slice_id / tile_name).exists():
            tile_run_dir = run_path / slice_id / tile_name
            proof_rel_path = os.path.join(slice_id, tile_name)
        else:
            tile_run_dir = run_path / tile_name
            proof_rel_path = tile_name

        if preferred_backend == Backend.JSTPROVE:
            tile_witness_path = tile_run_dir / "output_witness.bin"
        else:
            tile_witness_path = tile_run_dir / "output.json"

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
        """Prove a subset or all tiles for a slice and return aggregated results."""
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
        """Worker function logic for proving a single slice (handles tuple for pool compatibility)."""
        (slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path, tiling_info, run_path, slice_dir) = args
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
        """Core logic for proving a single slice (tiled or standard)."""
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
