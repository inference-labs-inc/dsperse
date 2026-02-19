import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class WorkItem:
    circuit_id: str
    run_uid: str
    slice_num: str
    task_id: str
    is_tile: bool
    tile_idx: int | None
    proof_system: str
    inputs: dict | list | None
    outputs: dict | list | None
    run_source: str
    completed: bool = False


@dataclass
class SliceExecResult:
    slice_id: str
    inputs_data: dict | list | None = None
    outputs_data: dict | list | None = None
    tiling: dict | None = None


@dataclass
class Run:
    run_uid: str
    circuit_id: str
    inputs: dict | list | None
    run_source: str
    max_tiles: int | None
    status: RunStatus = RunStatus.PENDING
    work_items: list[WorkItem] = field(default_factory=list)
    slice_results: dict[str, SliceExecResult] = field(default_factory=dict)
    applied_results: dict[str, dict] = field(default_factory=dict)
    error: str | None = None


class RunState:
    def __init__(self):
        self._runs: dict[str, Run] = {}

    def create_run(
        self,
        circuit_id: str,
        inputs: dict | list | None,
        run_source: str,
        max_tiles: int | None,
    ) -> Run:
        run_uid = uuid.uuid4().hex[:16]
        run = Run(
            run_uid=run_uid,
            circuit_id=circuit_id,
            inputs=inputs,
            run_source=run_source,
            max_tiles=max_tiles,
        )
        self._runs[run_uid] = run
        return run

    def get_run(self, run_uid: str) -> Run | None:
        return self._runs.get(run_uid)

    def get_pending_work(self, run_uid: str) -> list[WorkItem]:
        run = self._runs.get(run_uid)
        if not run:
            return []
        return [w for w in run.work_items if not w.completed]

    def mark_work_completed(self, run_uid: str, slice_num: str, tile_idx: int | None = None):
        run = self._runs.get(run_uid)
        if not run:
            return
        for item in run.work_items:
            if item.slice_num == slice_num and item.tile_idx == tile_idx:
                item.completed = True
                break

    def all_work_done(self, run_uid: str) -> bool:
        run = self._runs.get(run_uid)
        if not run:
            return True
        return all(w.completed for w in run.work_items)

    def generate_all_requests(self) -> list[dict]:
        items = []
        for run in self._runs.values():
            if run.status in (RunStatus.FAILED, RunStatus.COMPLETE):
                continue
            for w in run.work_items:
                if w.completed:
                    continue
                items.append({
                    "circuit_id": w.circuit_id,
                    "run_uid": w.run_uid,
                    "slice_num": w.slice_num,
                    "task_id": w.task_id,
                    "is_tile": w.is_tile,
                    "tile_idx": w.tile_idx,
                    "proof_system": w.proof_system,
                    "inputs": w.inputs,
                    "outputs": w.outputs,
                    "run_source": w.run_source,
                })
        return items
