import logging
from pathlib import Path

from dsperse.src.analyzers.schema import Backend
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class PipelineStage:

    def __init__(self, parallel: int = 1):
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

    def _dispatch(self, run_path: str | Path, model_dir: str | Path,
                  backend: str | None = None, output_path: str | Path | None = None,
                  tiles_range: range | list[int] | None = None) -> dict:
        run_path = Path(run_path)
        is_run_root = (run_path / "metadata.json").exists() or (run_path / "run_results.json").exists()
        is_slice_run = ((run_path / "input.json").exists() and (run_path / "output.json").exists()) or \
                       (run_path / "split").exists() or (run_path / "tile_0").exists()
        detected = Converter.detect_type(model_dir)

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
        result = self._execute_single_slice(run_path, dirs_model_path, backend, tiles_range)
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

    def _execute_dirs(self, run_path, dirs_path, output_path, backend) -> dict:
        raise NotImplementedError

    def _execute_single_slice(self, run_path, model_dir, detected, backend, tiles_range) -> dict:
        raise NotImplementedError
