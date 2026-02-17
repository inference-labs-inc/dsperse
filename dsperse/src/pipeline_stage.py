import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dsperse.src.analyzers.schema import Backend
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


def detect_run_type(run_path: str | Path) -> tuple[bool, bool]:
    p = Path(run_path)
    is_run_root = (p / "metadata.json").exists() or (p / "run_results.json").exists()
    is_slice_run = ((p / "input.json").exists() and (p / "output.json").exists()) or \
                   (p / "split").exists() or (p / "tile_0").exists()
    return is_run_root, is_slice_run


class PipelineStage:

    def __init__(self, parallel: int = 1):
        self.parallel = max(1, parallel)

    def _dispatch(self, run_path: str | Path, model_dir: str | Path,
                  backend: str | None = None, output_path: str | Path | None = None,
                  tiles_range: range | list[int] | None = None) -> dict:
        run_path = Path(run_path)
        is_run_root, is_slice_run = detect_run_type(run_path)
        detected = Converter.detect_type(model_dir)

        # is_slice_run first: a per-slice dir may also have run_results.json
        # from a prior prove step, which would incorrectly match is_run_root.
        if is_slice_run:
            return self._handle_single_slice(run_path, model_dir, detected, backend, tiles_range)
        if is_run_root:
            return self._handle_run_root(run_path, model_dir, detected, output_path, backend)
        raise FileNotFoundError(f"Run path invalid at {run_path}")

    def _handle_run_root(self, run_path: Path, model_dir: str | Path, detected: str,
                         output_path: str | Path | None, backend: str | None) -> dict:
        if detected in ("dslice", "dsperse"):
            return self._execute_packaged(run_path, model_dir, detected, output_path, backend)
        if detected == "dirs":
            return self._execute_dirs(run_path, model_dir, output_path, backend)
        raise ValueError(f"Unsupported data type: {detected}")

    def _handle_single_slice(self, run_path: Path, model_dir: str | Path, detected: str,
                             backend: str | None, tiles_range: range | list[int] | None) -> dict:
        if backend not in (Backend.JSTPROVE, Backend.EZKL):
            raise ValueError("Single-slice mode requires explicit backend: 'jstprove' or 'ezkl'.")
        dirs_model_path = model_dir
        if detected != "dirs":
            dirs_model_path = Converter.convert(str(model_dir), output_type="dirs", cleanup=False)
        try:
            result = self._execute_single_slice(run_path, dirs_model_path, backend, tiles_range)
        finally:
            if detected != "dirs":
                Converter.convert(str(Utils.dirs_root_from(Path(dirs_model_path))), output_type=detected, cleanup=True)
        return result

    def _execute_packaged(self, run_path: str | Path, pkg_path: str | Path, pkg_type: str,
                          output_path: str | Path | None, backend: str | None) -> dict:
        temp_dirs = Converter.convert(str(pkg_path), output_type="dirs", cleanup=False)
        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        try:
            summary = self._execute_dirs(run_path, dirs_root, output_path, backend)
        finally:
            Converter.convert(str(dirs_root), output_type=pkg_type, cleanup=True)
        return summary

    def _run_work_items(self, work_items: list, worker_func, label: str, error_result_func) -> list:
        if self.parallel > 1 and len(work_items) > 1:
            logger.info(f"{label} {len(work_items)} slices with {self.parallel} parallel processes...")
            results = []
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(worker_func, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    slice_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.exception(f"Slice {slice_id} {label.lower()} failed: {e}")
                        results.append(error_result_func(slice_id, e))
            return results

        results = []
        for item in work_items:
            slice_id = item[0]
            try:
                results.append(worker_func(item))
            except Exception as e:
                logger.exception(f"Slice {slice_id} {label.lower()} failed: {e}")
                results.append(error_result_func(slice_id, e))
        return results

    def _execute_dirs(self, run_path, dirs_path, output_path, backend) -> dict:
        raise NotImplementedError

    def _execute_single_slice(self, run_path, model_dir, backend, tiles_range) -> dict:
        raise NotImplementedError
