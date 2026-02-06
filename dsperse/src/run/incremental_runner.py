"""
IncrementalRunner for distributed slice execution.

This module provides an IncrementalRunner that allows external systems (like subnet validators)
to execute slices incrementally, where each slice's computation happens remotely and outputs
are fed back to continue the chain.
"""

import io
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
from dsperse.src.analyzers.schema import (
    Backend,
    ExecutionMethod,
    RunMetadata,
    RunSliceMetadata,
    Dependencies,
    TilingInfo,
)
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.backends.onnx_models import OnnxModels
from dsperse.src.run.tile_executor import TileExecutor
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

try:
    from python.core.utils.witness_utils import (
        extract_io_from_witness,
        load_witness,
        ZKProofSystems,
    )
    HAS_WITNESS_UTILS = True
except ImportError:
    HAS_WITNESS_UTILS = False

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
class TileTask:
    """Represents an individual tile of a tiled slice."""

    task_id: str
    slice_id: str
    tile_idx: int
    inputs: dict[str, Any]
    use_circuit: bool
    backend: str
    circuit_path: Optional[str] = None
    tiling_info: Optional[Any] = None
    metadata: Optional[RunSliceMetadata] = None


@dataclass
class SliceResult:
    """Result from a remotely executed slice.

    For circuit slices, only proof and witness are required - outputs are
    extracted from the witness after verification. For ONNX-only slices,
    outputs should be provided directly.
    """

    slice_id: str
    success: bool
    outputs: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    proof: Optional[bytes] = None
    witness: Optional[bytes] = None
    proof_time: float = 0.0


@dataclass
class TileResult:
    """Result from a tile execution."""

    task_id: str
    slice_id: str
    tile_idx: int
    success: bool
    outputs: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    proof: Optional[bytes] = None
    witness: Optional[bytes] = None


@dataclass
class PendingTiledSlice:
    """Tracks pending tiles for a tiled slice."""

    slice_id: str
    total_tiles: int
    completed_tiles: dict[int, TileResult] = field(default_factory=dict)
    failed_tiles: list[int] = field(default_factory=list)
    tiling_info: Optional[Any] = None
    metadata: Optional[RunSliceMetadata] = None

    @property
    def is_complete(self) -> bool:
        return len(self.completed_tiles) + len(self.failed_tiles) >= self.total_tiles

    @property
    def all_success(self) -> bool:
        return len(self.completed_tiles) == self.total_tiles and len(self.failed_tiles) == 0


@dataclass
class IncrementalRunState:
    """State for an incremental run."""

    run_metadata: RunMetadata
    slices_path: Path
    tensor_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    current_slice_id: Optional[str] = None
    completed_slices: list[str] = field(default_factory=list)
    failed_slices: list[str] = field(default_factory=list)
    pending_tiled_slice: Optional[PendingTiledSlice] = None


class IncrementalRunner:
    """
    Runner for incremental/distributed slice execution.

    Unlike the standard Runner which executes all slices locally, IncrementalRunner
    yields SliceTasks that can be executed remotely. The caller is responsible for
    executing each slice and feeding outputs back.

    Security Model:
        Circuit slices (use_circuit=True):
            - CAN be sent to miners
            - Outputs are EXTRACTED from the witness after proof verification
            - Miner-provided `result.outputs` is IGNORED
            - Witness contains public_inputs = [inputs..., outputs..., scale]
              which are cryptographically bound to the proof

        ONNX-only slices (use_circuit=False):
            - MUST be run locally by the validator
            - There is no proof, so miner outputs cannot be verified
            - NEVER send these to miners - they could return anything

    Usage:
        runner = IncrementalRunner(verify_proofs=True)
        state = runner.initialize(slice_path, input_data)

        for task in runner.iter_tasks(state):
            if task.use_circuit:
                # Safe to send to miner - outputs verified via proof
                result = send_to_miner(task)
                if not runner.apply_result(state, result):
                    handle_verification_failure(task)
            else:
                # MUST run locally - no proof means no verification
                result = runner.execute_onnx_slice(state, task)
                runner.apply_result(state, result)

        final_output = runner.get_final_output(state)
    """

    def __init__(self, verify_proofs: bool = True):
        """
        Initialize the IncrementalRunner.

        Args:
            verify_proofs: If True, verify proofs before accepting results.
                          Should always be True in production for security.
        """
        self._onnx_sessions: dict[str, Any] = {}
        self._verify_proofs = verify_proofs

        self._jstprove_runner: Optional[JSTprove] = None
        self._ezkl_runner: Optional[EZKL] = None

        if verify_proofs:
            try:
                self._jstprove_runner = JSTprove()
            except Exception:
                logger.warning("JSTprove not available for verification")

            try:
                self._ezkl_runner = EZKL()
            except Exception:
                logger.warning("EZKL not available for verification")

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
            for key in ["input_data", "input", "data", "inputs"]:
                if key in input_data:
                    input_tensor = torch.tensor(input_data[key])
                    break
            else:
                if len(input_data) == 1:
                    input_tensor = torch.tensor(next(iter(input_data.values())))
                else:
                    raise ValueError(f"Cannot find input tensor in dict keys: {list(input_data.keys())}")
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

    def iter_tasks(
        self, state: IncrementalRunState
    ) -> Iterator[SliceTask | TileTask]:
        """
        Iterate over slices/tiles that need execution.

        For non-tiled slices: yields SliceTask objects.
        For tiled slices: yields N TileTask objects (one per tile).

        The caller should execute each task and call apply_result() or
        apply_tile_result() before continuing iteration.

        Args:
            state: The run state from initialize()

        Yields:
            SliceTask for non-tiled slices, TileTask for each tile of tiled slices
        """
        nodes = state.run_metadata.execution_chain.nodes
        slice_index = 0

        while state.current_slice_id:
            if state.pending_tiled_slice and not state.pending_tiled_slice.is_complete:
                return

            slice_id = state.current_slice_id
            node = nodes.get(slice_id)
            if not node:
                logger.error(f"Slice {slice_id} not found in execution chain")
                break

            meta = state.run_metadata.get_slice(slice_id)
            if not meta:
                logger.error(f"Metadata not found for slice {slice_id}")
                break

            if meta.tiling and meta.tiling.num_tiles > 1:
                yield from self._expand_tiled_slice(state, slice_id, node, meta)
            else:
                inputs = self._prepare_slice_inputs(state, meta)
                task = SliceTask(
                    slice_id=slice_id,
                    slice_index=slice_index,
                    inputs=inputs,
                    input_tensor_names=meta.dependencies.filtered_inputs,
                    output_tensor_names=meta.dependencies.output,
                    use_circuit=node.use_circuit,
                    backend=node.backend,
                    is_tiled=False,
                    tile_count=0,
                    circuit_path=node.circuit_path,
                    onnx_path=node.onnx_path or meta.path,
                    metadata=meta,
                )
                yield task

            slice_index += 1

    def _expand_tiled_slice(
        self,
        state: IncrementalRunState,
        slice_id: str,
        node: Any,
        meta: RunSliceMetadata,
    ) -> Iterator[TileTask]:
        """Expand a tiled slice into individual tile tasks."""
        tiling = meta.tiling
        if not tiling:
            return

        tile_executor = TileExecutor(state.slices_path, state.tensor_cache)
        input_tensor = tile_executor.get_input_tensor(slice_id, tiling, meta)
        tile_executor.split_into_tiles(slice_id, tiling, input_tensor)

        state.pending_tiled_slice = PendingTiledSlice(
            slice_id=slice_id,
            total_tiles=tiling.num_tiles,
            tiling_info=tiling,
            metadata=meta,
        )

        slice_idx = tiling.slice_idx

        for tile_idx in range(tiling.num_tiles):
            cache_name = f"tile_{slice_idx}_{tile_idx}_in"
            tile_tensor = state.tensor_cache.get(cache_name)

            if tile_tensor is None:
                logger.error(f"Tile input {cache_name} not found in cache")
                continue

            tile_inputs = {"input_data": tile_tensor.tolist()}

            yield TileTask(
                task_id=f"{slice_id}_tile_{tile_idx}",
                slice_id=slice_id,
                tile_idx=tile_idx,
                inputs=tile_inputs,
                use_circuit=node.use_circuit,
                backend=node.backend,
                circuit_path=node.circuit_path,
                tiling_info=tiling,
                metadata=meta,
            )

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

        return inputs

    def apply_result(self, state: IncrementalRunState, result: SliceResult) -> bool:
        """
        Apply a slice execution result to the run state.

        For circuit slices: verifies proof and extracts outputs FROM the witness
        (outputs are cryptographically bound to the proof).
        For ONNX-only slices: uses provided outputs directly.

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
        nodes = state.run_metadata.execution_chain.nodes
        node = nodes.get(state.current_slice_id)

        if not meta or not node:
            logger.error(f"Metadata not found for slice {result.slice_id}")
            state.failed_slices.append(result.slice_id)
            return False

        outputs_to_use = result.outputs

        if node.use_circuit and self._verify_proofs:
            verified_outputs = self._verify_and_extract_outputs(state, meta, node, result)
            if verified_outputs is None:
                state.failed_slices.append(result.slice_id)
                logger.error(f"Proof verification failed for {result.slice_id}")
                return False
            outputs_to_use = verified_outputs

        if outputs_to_use:
            validation_error = self._validate_outputs(meta, outputs_to_use)
            if validation_error:
                state.failed_slices.append(result.slice_id)
                logger.error(f"Output validation failed for {result.slice_id}: {validation_error}")
                return False

            self._update_tensor_cache(state, meta, outputs_to_use)

        state.completed_slices.append(result.slice_id)
        state.current_slice_id = node.next if node else None

        return True

    def apply_tile_result(self, state: IncrementalRunState, result: TileResult) -> bool:
        """
        Apply a tile execution result to the run state.

        For circuit tiles: verifies proof and extracts outputs FROM the witness.
        For ONNX-only tiles: uses provided outputs directly.

        When all tiles for a slice complete, automatically reconstructs the
        full output and advances to the next slice.

        Args:
            state: The run state
            result: The tile execution result

        Returns:
            True if result was applied successfully
        """
        pending = state.pending_tiled_slice
        if not pending:
            logger.error(f"No pending tiled slice for tile result {result.task_id}")
            return False

        if result.slice_id != pending.slice_id:
            logger.error(
                f"Tile result slice_id {result.slice_id} doesn't match "
                f"pending slice {pending.slice_id}"
            )
            return False

        if not result.success:
            pending.failed_tiles.append(result.tile_idx)
            logger.error(f"Tile {result.task_id} failed: {result.error}")
            return False

        meta = pending.metadata
        tiling = pending.tiling_info
        nodes = state.run_metadata.execution_chain.nodes
        node = nodes.get(pending.slice_id)

        if not meta or not node or not tiling:
            logger.error(f"Missing metadata for tiled slice {pending.slice_id}")
            pending.failed_tiles.append(result.tile_idx)
            return False

        outputs_to_use = result.outputs

        if node.use_circuit and self._verify_proofs:
            verified_outputs = self._verify_and_extract_tile_outputs(
                state, meta, node, result, tiling
            )
            if verified_outputs is None:
                pending.failed_tiles.append(result.tile_idx)
                logger.error(f"Tile proof verification failed for {result.task_id}")
                return False
            outputs_to_use = verified_outputs

        if outputs_to_use:
            self._store_tile_output(state, tiling, result.tile_idx, outputs_to_use)

        pending.completed_tiles[result.tile_idx] = result

        if pending.is_complete:
            return self._finalize_tiled_slice(state)

        return True

    def _verify_and_extract_tile_outputs(
        self,
        state: IncrementalRunState,
        meta: RunSliceMetadata,
        node: Any,
        result: TileResult,
        tiling: TilingInfo,
    ) -> Optional[dict[str, Any]]:
        """Verify tile proof and extract outputs from witness."""
        if not result.proof:
            logger.error(f"Missing proof for tile {result.task_id}")
            return None

        if not result.witness:
            logger.error(f"Missing witness for tile {result.task_id}")
            return None

        backend = (node.backend or meta.backend or "").lower()
        circuit_path = node.circuit_path or meta.jstprove_circuit_path or meta.ezkl_circuit_path

        if not circuit_path:
            logger.error(f"No circuit path for tile {result.task_id}")
            return None

        circuit_path = RunnerUtils.resolve_relative_path(circuit_path, state.slices_path)
        if not circuit_path or not Path(circuit_path).exists():
            logger.error(f"Circuit not found: {circuit_path}")
            return None

        tile_size = tiling.tile_size
        halo = tiling.halo
        c_in = tiling.c_in or 1
        tile_h = tile_size + 2 * halo[0]
        tile_w = tile_size + 2 * halo[1]
        num_inputs = c_in * tile_h * tile_w

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                witness_path = tmp / "witness.bin"

                with open(witness_path, "wb") as f:
                    f.write(result.witness if isinstance(result.witness, bytes) else result.witness.encode())

                if backend == Backend.JSTPROVE:
                    return self._extract_jstprove_tile_outputs(witness_path, num_inputs, tiling)
                elif backend == Backend.EZKL:
                    return self._extract_ezkl_outputs(witness_path, meta)
                else:
                    logger.error(f"Unknown backend for tile output extraction: {backend}")
                    return None

        except Exception as e:
            logger.exception(f"Tile proof verification error for {result.task_id}: {e}")
            return None

    def _extract_jstprove_tile_outputs(
        self,
        witness_path: Path,
        num_inputs: int,
        tiling: TilingInfo,
    ) -> Optional[dict[str, Any]]:
        """Extract outputs from JSTprove witness for a tile."""
        if not HAS_WITNESS_UTILS:
            logger.error("JSTprove witness_utils not available")
            return None

        try:
            witness_data = load_witness(str(witness_path), ZKProofSystems.Expander)
            extracted = extract_io_from_witness(witness_data, num_inputs)
            if extracted is None:
                logger.error("Failed to extract I/O from tile witness")
                return None

            return {
                "output": extracted["outputs"],
                "rescaled_output": extracted["rescaled_outputs"],
                "raw_output": extracted["raw_outputs"],
            }

        except Exception as e:
            logger.exception(f"Failed to extract JSTprove tile outputs: {e}")
            return None

    def _store_tile_output(
        self,
        state: IncrementalRunState,
        tiling: TilingInfo,
        tile_idx: int,
        outputs: dict[str, Any],
    ) -> None:
        """Store tile output in tensor cache for later reconstruction."""
        output_data = outputs.get("output_data") or outputs.get("output") or outputs

        tensor = self._to_tensor(output_data)
        if tensor is None:
            logger.warning(f"Failed to convert tile {tile_idx} output to tensor")
            return

        c_out = tiling.c_out
        if tiling.tile and tiling.tile.conv_out:
            h_out, w_out = tiling.tile.conv_out
        else:
            h_out, w_out = 0, 0

        if c_out and h_out and w_out and tensor.numel() == (1 * c_out * h_out * w_out):
            tensor = tensor.reshape(1, c_out, h_out, w_out)

        cache_name = f"tile_{tiling.slice_idx}_{tile_idx}_out"
        state.tensor_cache[cache_name] = tensor

    def _finalize_tiled_slice(self, state: IncrementalRunState) -> bool:
        """Finalize a tiled slice after all tiles complete."""
        pending = state.pending_tiled_slice
        if not pending:
            return False

        if pending.failed_tiles:
            state.failed_slices.append(pending.slice_id)
            logger.error(
                f"Tiled slice {pending.slice_id} failed: "
                f"{len(pending.failed_tiles)} tiles failed"
            )
            state.pending_tiled_slice = None
            return False

        try:
            tile_executor = TileExecutor(state.slices_path, state.tensor_cache)
            tile_executor.reconstruct_from_tiles(pending.slice_id, pending.tiling_info)
        except Exception as e:
            state.failed_slices.append(pending.slice_id)
            logger.exception(f"Failed to reconstruct tiled slice {pending.slice_id}: {e}")
            state.pending_tiled_slice = None
            return False

        nodes = state.run_metadata.execution_chain.nodes
        node = nodes.get(pending.slice_id)

        state.completed_slices.append(pending.slice_id)
        state.current_slice_id = node.next if node else None
        state.pending_tiled_slice = None

        return True

    def _validate_outputs(self, meta: RunSliceMetadata, outputs: dict[str, Any]) -> Optional[str]:
        """Validate output data for shape correctness and NaN/Inf values."""
        output_data = outputs.get("output_data") or outputs.get("output") or outputs

        if output_data is None:
            return "No output data provided"

        try:
            if isinstance(output_data, dict):
                for name, value in output_data.items():
                    err = self._validate_tensor_value(value, name)
                    if err:
                        return err
            else:
                err = self._validate_tensor_value(output_data, "output")
                if err:
                    return err

            if meta.output_shape:
                tensor = self._to_tensor(
                    output_data if not isinstance(output_data, dict)
                    else next(iter(output_data.values()))
                )
                if tensor is not None:
                    expected_shape = meta.output_shape[0] if meta.output_shape else None
                    if expected_shape:
                        expected_numel = 1
                        for dim in expected_shape:
                            if isinstance(dim, int):
                                expected_numel *= dim
                        if tensor.numel() != expected_numel:
                            return f"Output size mismatch: got {tensor.numel()}, expected {expected_numel}"

        except Exception as e:
            return f"Validation error: {e}"

        return None

    def _validate_tensor_value(self, value: Any, name: str) -> Optional[str]:
        """Check a tensor value for NaN/Inf."""
        try:
            arr = np.asarray(value, dtype=np.float32)
            if np.any(np.isnan(arr)):
                return f"NaN values detected in {name}"
            if np.any(np.isinf(arr)):
                return f"Inf values detected in {name}"
        except Exception:
            pass
        return None

    def _verify_and_extract_outputs(
        self,
        state: IncrementalRunState,
        meta: RunSliceMetadata,
        node: Any,
        result: SliceResult,
    ) -> Optional[dict[str, Any]]:
        """
        Verify the proof and extract outputs from the witness.

        The outputs are extracted FROM the witness's public_inputs, which are
        cryptographically bound to the proof. This ensures the miner cannot
        claim different outputs than what was actually computed.

        Returns:
            Extracted outputs dict if verification succeeds, None otherwise.
        """
        if not result.proof:
            logger.error(f"Missing proof for circuit slice {result.slice_id}")
            return None

        if not result.witness:
            logger.error(f"Missing witness for circuit slice {result.slice_id}")
            return None

        backend = (node.backend or meta.backend or "").lower()
        circuit_path = node.circuit_path or meta.jstprove_circuit_path or meta.ezkl_circuit_path

        if not circuit_path:
            logger.error(f"No circuit path for slice {result.slice_id}")
            return None

        circuit_path = RunnerUtils.resolve_relative_path(circuit_path, state.slices_path)
        if not circuit_path or not Path(circuit_path).exists():
            logger.error(f"Circuit not found: {circuit_path}")
            return None

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                proof_path = tmp / "proof.bin"
                witness_path = tmp / "witness.bin"

                with open(proof_path, "wb") as f:
                    f.write(result.proof if isinstance(result.proof, bytes) else result.proof.encode())
                with open(witness_path, "wb") as f:
                    f.write(result.witness if isinstance(result.witness, bytes) else result.witness.encode())

                extracted_outputs = self._extract_outputs_from_witness(
                    witness_path, meta, backend
                )
                if extracted_outputs is None:
                    logger.error(f"Failed to extract outputs from witness for {result.slice_id}")
                    return None

                inputs = self._prepare_slice_inputs(state, meta)
                input_path = tmp / "input.json"
                output_path = tmp / "output.json"

                with open(input_path, "w") as f:
                    json.dump(inputs, f)
                with open(output_path, "w") as f:
                    json.dump(extracted_outputs, f)

                if backend == Backend.JSTPROVE and self._jstprove_runner:
                    verified = self._jstprove_runner.verify(
                        proof_path=proof_path,
                        circuit_path=circuit_path,
                        input_path=input_path,
                        output_path=output_path,
                        witness_path=witness_path,
                    )
                elif backend == Backend.EZKL and self._ezkl_runner:
                    settings_path = meta.ezkl_settings_path or meta.settings_path
                    vk_path = meta.ezkl_vk_path or meta.vk_path
                    if settings_path:
                        settings_path = RunnerUtils.resolve_relative_path(settings_path, state.slices_path)
                    if vk_path:
                        vk_path = RunnerUtils.resolve_relative_path(vk_path, state.slices_path)
                    verified = self._ezkl_runner.verify(
                        proof_path=proof_path,
                        settings_path=settings_path,
                        vk_path=vk_path,
                    )
                else:
                    logger.error(f"No verifier available for backend {backend}")
                    return None

                if not verified:
                    logger.error(f"Proof verification failed for {result.slice_id}")
                    return None

                return extracted_outputs

        except Exception as e:
            logger.exception(f"Proof verification error for {result.slice_id}: {e}")
            return None

    def _extract_outputs_from_witness(
        self,
        witness_path: Path,
        meta: RunSliceMetadata,
        backend: str,
    ) -> Optional[dict[str, Any]]:
        """
        Extract output values from the witness file.

        For JSTprove/Expander, the witness public_inputs contain:
        [input_values..., output_values..., scale_base, scale_exponent]
        """
        if backend == Backend.JSTPROVE:
            return self._extract_jstprove_outputs(witness_path, meta)
        elif backend == Backend.EZKL:
            return self._extract_ezkl_outputs(witness_path, meta)
        else:
            logger.error(f"Unknown backend for output extraction: {backend}")
            return None

    def _extract_jstprove_outputs(
        self,
        witness_path: Path,
        meta: RunSliceMetadata,
    ) -> Optional[dict[str, Any]]:
        """Extract outputs from JSTprove/Expander witness format."""
        if not HAS_WITNESS_UTILS:
            logger.error("JSTprove witness_utils not available")
            return None

        try:
            witness_data = load_witness(str(witness_path), ZKProofSystems.Expander)

            num_inputs = sum(
                np.prod([d for d in shape if isinstance(d, int)])
                for shape in (meta.input_shape or [])
            )
            num_inputs = int(num_inputs) if num_inputs > 0 else 0

            extracted = extract_io_from_witness(witness_data, num_inputs)
            if extracted is None:
                logger.error("Failed to extract I/O from witness")
                return None

            return {
                "output": extracted["outputs"],
                "rescaled_output": extracted["rescaled_outputs"],
                "raw_output": extracted["raw_outputs"],
            }

        except Exception as e:
            logger.exception(f"Failed to extract JSTprove outputs: {e}")
            return None

    def _extract_ezkl_outputs(
        self,
        witness_path: Path,
        meta: RunSliceMetadata,
    ) -> Optional[dict[str, Any]]:
        """Extract outputs from EZKL witness format."""
        try:
            with open(witness_path, "r") as f:
                witness_data = json.load(f)

            if "pretty_elements" in witness_data:
                rescaled = witness_data["pretty_elements"].get("rescaled_outputs", [[]])[0]
                return {"output": rescaled, "rescaled_output": rescaled}

            if "outputs" in witness_data:
                return {"output": witness_data["outputs"]}

            logger.error("Could not parse EZKL witness format")
            return None

        except Exception as e:
            logger.exception(f"Failed to extract EZKL outputs: {e}")
            return None

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
                for oname in output_names:
                    state.tensor_cache[oname] = tensor

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
            meta = state.run_metadata.get_slice(task.slice_id)
            if meta:
                onnx_path = RunnerUtils.resolve_relative_path(
                    onnx_path or meta.path, state.slices_path / task.slice_id
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
