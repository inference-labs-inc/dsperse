"""
IncrementalRunner for distributed slice execution.

This module provides an IncrementalRunner that allows external systems (like subnet validators)
to execute slices incrementally, where each slice's computation happens remotely and outputs
are fed back to continue the chain.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
from dsperse.src.analyzers.schema import (
    Backend,
    ExecutionMethod,
    RunMetadata,
    RunSliceMetadata,
    Dependencies,
)
from dsperse.src.backends.onnx_models import OnnxModels
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


@dataclass
class SliceTask:
    """Represents a slice that needs to be executed remotely."""

    slice_id: str
    slice_index: int
    inputs: dict[str, Any]
    input_tensor_names: list[str]
    output_tensor_names: list[str]
    use_circuit: bool
    backend: str
    is_tiled: bool = False
    tile_count: int = 0
    circuit_path: Optional[str] = None
    onnx_path: Optional[str] = None
    metadata: Optional[RunSliceMetadata] = None


@dataclass
class SliceResult:
    """Result from a remotely executed slice."""

    slice_id: str
    success: bool
    outputs: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    proof: Optional[Any] = None
    proof_time: float = 0.0


@dataclass
class IncrementalRunState:
    """State for an incremental run."""

    run_metadata: RunMetadata
    slices_path: Path
    tensor_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    current_slice_id: Optional[str] = None
    completed_slices: list[str] = field(default_factory=list)
    failed_slices: list[str] = field(default_factory=list)


class IncrementalRunner:
    """
    Runner for incremental/distributed slice execution.

    Unlike the standard Runner which executes all slices locally, IncrementalRunner
    yields SliceTasks that can be executed remotely. The caller is responsible for
    executing each slice and feeding outputs back.

    Usage:
        runner = IncrementalRunner()
        state = runner.initialize(slice_path, input_data)

        for task in runner.iter_tasks(state):
            if task.use_circuit:
                # Send to remote miner for execution + proof
                result = send_to_miner(task)
                runner.apply_result(state, result)
            else:
                # Execute ONNX-only slice locally
                result = runner.execute_onnx_slice(state, task)
                runner.apply_result(state, result)

        final_output = runner.get_final_output(state)
    """

    def __init__(self):
        self._onnx_sessions: dict[str, Any] = {}

    def initialize(
        self,
        slice_path: str | Path,
        input_data: dict | torch.Tensor,
        run_dir: Optional[str | Path] = None,
    ) -> IncrementalRunState:
        """
        Initialize an incremental run.

        Args:
            slice_path: Path to slices directory or .dsperse/.dslice file
            input_data: Model input as dict (with 'input_data' key) or tensor
            run_dir: Optional directory for run outputs

        Returns:
            IncrementalRunState that tracks the run's progress
        """
        slice_path = Path(slice_path)
        if not slice_path.exists():
            raise FileNotFoundError(f"Slice path not found: {slice_path}")

        detected_format = Converter.detect_type(slice_path)
        if detected_format != "dirs":
            slice_path = Path(Converter.convert(str(slice_path), output_type="dirs"))

        run_dir_path, _, run_metadata = RunnerAnalyzer.initialize_run_metadata(
            slice_path,
            run_dir=Path(run_dir) if run_dir else None,
            output_path=None,
            format=detected_format,
        )

        if isinstance(input_data, dict):
            input_tensor = Utils.dict_to_tensor(input_data)
        else:
            input_tensor = input_data

        head_id = run_metadata.execution_chain.head
        first_slice_meta = run_metadata.get_slice(head_id)
        first_slice_inputs = (
            first_slice_meta.dependencies.filtered_inputs if first_slice_meta else []
        )
        model_input_name = first_slice_inputs[0] if first_slice_inputs else "input"

        tensor_cache = {model_input_name: input_tensor}

        return IncrementalRunState(
            run_metadata=run_metadata,
            slices_path=slice_path,
            tensor_cache=tensor_cache,
            current_slice_id=head_id,
        )

    def iter_tasks(self, state: IncrementalRunState) -> Iterator[SliceTask]:
        """
        Iterate over slices that need execution.

        Yields SliceTask objects for each slice in execution order.
        The caller should execute each task and call apply_result() before
        continuing iteration.

        Args:
            state: The run state from initialize()

        Yields:
            SliceTask for each slice needing execution
        """
        nodes = state.run_metadata.execution_chain.nodes
        slice_index = 0

        while state.current_slice_id:
            slice_id = state.current_slice_id
            node = nodes.get(slice_id)
            if not node:
                logger.error(f"Slice {slice_id} not found in execution chain")
                break

            meta = state.run_metadata.get_slice(slice_id)
            if not meta:
                logger.error(f"Metadata not found for slice {slice_id}")
                break

            inputs = self._prepare_slice_inputs(state, meta)

            task = SliceTask(
                slice_id=slice_id,
                slice_index=slice_index,
                inputs=inputs,
                input_tensor_names=meta.dependencies.filtered_inputs,
                output_tensor_names=meta.dependencies.output,
                use_circuit=node.use_circuit,
                backend=node.backend,
                is_tiled=meta.tiling is not None,
                tile_count=meta.tiling.num_tiles if meta.tiling else 0,
                circuit_path=node.circuit_path,
                onnx_path=node.onnx_path or meta.path,
                metadata=meta,
            )

            yield task
            slice_index += 1

    def _prepare_slice_inputs(
        self, state: IncrementalRunState, meta: RunSliceMetadata
    ) -> dict[str, Any]:
        """Prepare input data for a slice from the tensor cache."""
        inputs = {}
        for input_name in meta.dependencies.filtered_inputs:
            if input_name in state.tensor_cache:
                tensor = state.tensor_cache[input_name]
                if hasattr(tensor, "tolist"):
                    inputs[input_name] = tensor.tolist()
                else:
                    inputs[input_name] = tensor
            else:
                logger.warning(f"Input {input_name} not found in tensor cache")

        if len(inputs) == 1:
            key = list(inputs.keys())[0]
            return {"input_data": inputs[key]}

        return {"input_data": inputs}

    def apply_result(self, state: IncrementalRunState, result: SliceResult) -> bool:
        """
        Apply a slice execution result to the run state.

        Updates the tensor cache with outputs and advances to the next slice.

        Args:
            state: The run state
            result: The execution result from remote or local execution

        Returns:
            True if result was applied successfully
        """
        if result.slice_id != state.current_slice_id:
            logger.error(
                f"Result slice_id {result.slice_id} doesn't match "
                f"current slice {state.current_slice_id}"
            )
            return False

        if not result.success:
            state.failed_slices.append(result.slice_id)
            logger.error(f"Slice {result.slice_id} failed: {result.error}")
            return False

        meta = state.run_metadata.get_slice(result.slice_id)
        if meta and result.outputs:
            self._update_tensor_cache(state, meta, result.outputs)

        state.completed_slices.append(result.slice_id)

        nodes = state.run_metadata.execution_chain.nodes
        node = nodes.get(state.current_slice_id)
        state.current_slice_id = node.next if node else None

        return True

    def _update_tensor_cache(
        self,
        state: IncrementalRunState,
        meta: RunSliceMetadata,
        outputs: dict[str, Any],
    ) -> None:
        """Update tensor cache with slice outputs."""
        output_data = outputs.get("output_data") or outputs.get("output") or outputs

        if isinstance(output_data, dict):
            for name, value in output_data.items():
                tensor = self._to_tensor(value)
                if tensor is not None:
                    state.tensor_cache[name] = tensor
        else:
            output_names = meta.dependencies.output
            tensor = self._to_tensor(output_data)
            if tensor is not None and output_names:
                state.tensor_cache[output_names[0]] = tensor

    def _to_tensor(self, value: Any) -> Optional[torch.Tensor]:
        """Convert value to tensor."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        try:
            return torch.tensor(value)
        except Exception as e:
            logger.warning(f"Failed to convert to tensor: {e}")
            return None

    def execute_onnx_slice(
        self, state: IncrementalRunState, task: SliceTask
    ) -> SliceResult:
        """
        Execute an ONNX-only slice locally.

        Used for slices that don't require circuit proofs.

        Args:
            state: The run state
            task: The slice task to execute

        Returns:
            SliceResult with execution outcome
        """
        try:
            onnx_path = task.onnx_path
            if not onnx_path:
                meta = state.run_metadata.get_slice(task.slice_id)
                if meta:
                    onnx_path = RunnerUtils.resolve_relative_path(
                        meta.path, state.slices_path / task.slice_id
                    )

            if not onnx_path or not Path(onnx_path).exists():
                return SliceResult(
                    slice_id=task.slice_id,
                    success=False,
                    error=f"ONNX model not found: {onnx_path}",
                )

            input_tensor = None
            for name in task.input_tensor_names:
                if name in state.tensor_cache:
                    input_tensor = state.tensor_cache[name]
                    break

            if input_tensor is None:
                return SliceResult(
                    slice_id=task.slice_id,
                    success=False,
                    error="No input tensor available",
                )

            success, result = OnnxModels.run_inference(
                model_path=onnx_path,
                input_tensor=input_tensor,
            )

            if not success:
                return SliceResult(
                    slice_id=task.slice_id,
                    success=False,
                    error=str(result),
                )

            output_tensor = RunnerUtils.extract_output_tensor(result)
            outputs = {
                "output_data": (
                    output_tensor.tolist()
                    if hasattr(output_tensor, "tolist")
                    else output_tensor
                )
            }

            return SliceResult(
                slice_id=task.slice_id,
                success=True,
                outputs=outputs,
            )

        except Exception as e:
            logger.exception(f"ONNX execution failed for {task.slice_id}")
            return SliceResult(
                slice_id=task.slice_id,
                success=False,
                error=str(e),
            )

    def get_final_output(self, state: IncrementalRunState) -> Optional[torch.Tensor]:
        """
        Get the final output tensor after all slices complete.

        Args:
            state: The completed run state

        Returns:
            Final output tensor or None if run incomplete
        """
        if state.current_slice_id is not None:
            logger.warning("Run not complete, some slices remaining")
            return None

        nodes = state.run_metadata.execution_chain.nodes
        last_slice_id = None
        for slice_id in state.completed_slices:
            if nodes.get(slice_id) and nodes[slice_id].next is None:
                last_slice_id = slice_id
                break

        if not last_slice_id:
            last_slice_id = (
                state.completed_slices[-1] if state.completed_slices else None
            )

        if last_slice_id:
            meta = state.run_metadata.get_slice(last_slice_id)
            if meta and meta.dependencies.output:
                output_name = meta.dependencies.output[0]
                return state.tensor_cache.get(output_name)

        return None

    def is_complete(self, state: IncrementalRunState) -> bool:
        """Check if the run has completed all slices."""
        return state.current_slice_id is None and not state.failed_slices

    def get_progress(self, state: IncrementalRunState) -> dict:
        """Get progress information for the run."""
        total_slices = len(state.run_metadata.execution_chain.nodes)
        return {
            "total_slices": total_slices,
            "completed": len(state.completed_slices),
            "failed": len(state.failed_slices),
            "remaining": total_slices
            - len(state.completed_slices)
            - len(state.failed_slices),
            "current_slice": state.current_slice_id,
            "is_complete": self.is_complete(state),
        }
