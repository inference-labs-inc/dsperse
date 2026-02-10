import logging
from pathlib import Path
from typing import Optional, Union

from dsperse.src.analyzers.schema import Backend, ExecutionChain, ExecutionMethod, RunSliceMetadata, RunMetadata
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


def get_witness_backend(run_path: Path, slice_id: str) -> str | None:
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


def get_witness_file(run_path: Path, slice_id: str) -> str | None:
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
    filtered = []
    for slice_id, meta in slices_iter:
        wb = get_witness_backend(run_path, slice_id)
        if wb == backend:
            filtered.append((slice_id, meta))
    return filtered


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
