"""
Async storage drain - writes to fast storage, drains to persistent storage in background.
"""

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

logger = logging.getLogger(__name__)


def get_free_gb(path: Path) -> float:
    """Get free space in GB."""
    try:
        stat = os.statvfs(path)
        return (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
    except Exception:
        return 0.0


class AsyncDrain:
    """Background thread that moves completed data from hot to cold storage."""

    def __init__(self, hot_path: Path, cold_path: Path):
        self.hot_path = Path(hot_path)
        self.cold_path = Path(cold_path)
        self._queue: Queue[str] = Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self.moved_count = 0
        self.bytes_moved = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"Drain started: {self.hot_path} -> {self.cold_path}")

    def stop(self, wait: bool = True):
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=60.0)

    def queue(self, relative_path: str):
        with self._lock:
            if relative_path not in self._pending:
                self._pending.add(relative_path)
                self._queue.put(relative_path)

    def _loop(self):
        while not self._stop.is_set():
            try:
                path = self._queue.get(timeout=0.5)
                self._move(path)
                with self._lock:
                    self._pending.discard(path)
            except Empty:
                continue

    def _move(self, relative_path: str):
        src = self.hot_path / relative_path
        dst = self.cold_path / relative_path
        if not src.exists():
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                size = sum(f.stat().st_size for f in src.rglob('*') if f.is_file())
                shutil.rmtree(src)
            else:
                shutil.copy2(src, dst)
                size = src.stat().st_size
                src.unlink()
            self.moved_count += 1
            self.bytes_moved += size
        except Exception as e:
            logger.error(f"Drain failed for {relative_path}: {e}")

    def flush(self, timeout: float = 120.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if not self._pending and self._queue.empty():
                    return True
            time.sleep(0.1)
        return False

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


class TieredStorage:
    """Hot storage with async drain to cold storage."""

    def __init__(self, hot_path: Path, cold_path: Path, min_free_gb: float = 5.0):
        self.hot_path = Path(hot_path)
        self.cold_path = Path(cold_path)
        self.min_free_gb = min_free_gb
        self._drain: Optional[AsyncDrain] = None
        self._run_id: Optional[str] = None

    def initialize(self, run_id: str) -> Path:
        self._run_id = run_id
        hot_run = self.hot_path / run_id
        cold_run = self.cold_path / run_id
        hot_run.mkdir(parents=True, exist_ok=True)
        cold_run.mkdir(parents=True, exist_ok=True)

        self._drain = AsyncDrain(hot_run, cold_run)
        self._drain.start()

        hot_free = get_free_gb(self.hot_path)
        cold_free = get_free_gb(self.cold_path)
        logger.info(f"Storage: hot={self.hot_path} ({hot_free:.1f}GB), cold={self.cold_path} ({cold_free:.1f}GB)")
        return hot_run

    def mark_complete(self, relative_path: str):
        if self._drain:
            self._drain.queue(relative_path)

    def wait_for_space(self, timeout: float = 300.0) -> bool:
        """Wait until hot storage has min_free_gb available."""
        if not self._drain:
            return True
        start = time.time()
        while time.time() - start < timeout:
            free = get_free_gb(self.hot_path)
            if free >= self.min_free_gb:
                return True
            pending = self._drain.pending_count
            if pending == 0:
                return free > 0
            logger.info(f"Hot storage: {free:.1f}GB free, waiting for drain ({pending} pending)")
            time.sleep(2.0)
        return False

    def shutdown(self):
        if self._drain:
            logger.info("Flushing drain queue...")
            self._drain.flush()
            self._drain.stop()
            logger.info(f"Drain complete: {self._drain.moved_count} items, {self._drain.bytes_moved / 1e9:.2f}GB")

    @property
    def hot_run_dir(self) -> Optional[Path]:
        return self.hot_path / self._run_id if self._run_id else None

    @property
    def cold_run_dir(self) -> Optional[Path]:
        return self.cold_path / self._run_id if self._run_id else None
