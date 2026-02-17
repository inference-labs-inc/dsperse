import logging
from pathlib import Path
from typing import Optional, Union

from dsperse.src.analyzers.schema import Backend, ExecutionChain, ExecutionMethod, RunSliceMetadata, RunMetadata
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


def _backend_from_entry(entry) -> str | None:
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


def _load_exec_chain(run_path: Path) -> ExecutionChain | None:
    try:
        rr = Utils.load_run_results(run_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.debug(f"Could not load execution chain from {run_path}: {e}")
        return None
    if not rr:
        return None
    return ExecutionChain.from_dict(rr.get("execution_chain"))


def get_witness_backend(run_path: Path, slice_id: str) -> str | None:
    chain = _load_exec_chain(run_path)
    if not chain:
        return None
    return _backend_from_entry(chain.get_result_for_slice(slice_id))


def get_witness_file(run_path: Path, slice_id: str) -> str | None:
    chain = _load_exec_chain(run_path)
    if not chain:
        return None
    entry = chain.get_result_for_slice(slice_id)
    if not entry or not entry.witness_execution:
        return None
    return Utils.resolve_path(entry.witness_execution.witness_file, run_path)


def select_backend(run_path: Path, slice_id: str, meta: RunSliceMetadata) -> str:
    from_run = get_witness_backend(run_path, slice_id)
    if from_run in (Backend.JSTPROVE, Backend.EZKL):
        return from_run
    meta_backend = (meta.backend or "").lower()
    if meta_backend in (Backend.JSTPROVE, Backend.EZKL):
        return meta_backend
    return Backend.JSTPROVE


def initialize_stage_metadata(run_path: Path, dirs_path: Path) -> RunMetadata:
    slices_metadata_path = dirs_path / "metadata.json"
    if slices_metadata_path.exists():
        from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
        run_meta_dict = RunnerAnalyzer.generate_run_metadata(dirs_path, save_path=None, original_format="dirs")
        return RunMetadata.from_dict(run_meta_dict)
    else:
        return RunMetadata.from_dict(Utils.load_run_metadata(run_path))


def filter_circuit_slices(metadata: RunMetadata, backend: Optional[str], run_path: Path) -> list[tuple[str, RunSliceMetadata]]:
    slices_iter = list(metadata.iter_circuit_slices())
    if not backend or backend not in (Backend.JSTPROVE, Backend.EZKL):
        return slices_iter
    chain = _load_exec_chain(run_path)
    if not chain:
        return slices_iter
    filtered = []
    for slice_id, meta in slices_iter:
        wb = _backend_from_entry(chain.get_result_for_slice(slice_id))
        if wb == backend:
            filtered.append((slice_id, meta))
    return filtered


def finalize_stage_results(run_path: Path, results: dict, stage: str, key_suffix: str,
                           jst_count: int, ezkl_count: int, total: int) -> dict:
    run_results = Utils.load_run_results(run_path)
    run_results, _ = Utils.merge_execution_into_run_results(run_results, results, stage)
    for entry in run_results.get("execution_chain", {}).get("execution_results", []):
        pe = entry.get("proof_execution")
        if pe:
            if pe.get("proof_file"):
                pe["proof_file"] = Utils.relativize_path(pe["proof_file"], run_path)
            for tp in pe.get("tile_proofs_info") or []:
                if tp.get("proof_path"):
                    tp["proof_path"] = Utils.relativize_path(tp["proof_path"], run_path)
    exec_chain = run_results.setdefault("execution_chain", {})
    exec_chain[f"jstprove_{key_suffix}_slices"] = int(jst_count)
    exec_chain[f"ezkl_{key_suffix}_slices"] = int(ezkl_count)
    Utils.save_run_results(run_path, run_results)
    Utils.update_metadata_after_execution(run_path, total, jst_count + ezkl_count, stage)
    return run_results


def parse_tiles_range(tiles_str: str | None) -> range | list[int] | None:
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
