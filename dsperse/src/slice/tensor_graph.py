"""
TensorGraph: Complete dependency mapping for ONNX models.

Builds a graph of tensor producers and consumers to properly handle
skip connections and cross-slice dependencies during slicing and inference.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import onnx


@dataclass
class TensorInfo:
    name: str
    producer_node: Optional[str] = None
    producer_node_idx: Optional[int] = None
    consumers: list[tuple[str, int]] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    is_initializer: bool = False


class TensorGraph:
    def __init__(self, model: onnx.ModelProto | str | Path):
        if isinstance(model, (str, Path)):
            self.model = onnx.load(str(model))
        else:
            self.model = model
        self.graph = self.model.graph

        self.tensors: dict[str, TensorInfo] = {}
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}

        self._build_graph()

    def _build_graph(self):
        for inp in self.graph.input:
            self.tensors[inp.name] = TensorInfo(
                name=inp.name,
                is_input=True
            )

        for init in self.graph.initializer:
            if init.name in self.tensors:
                self.tensors[init.name].is_initializer = True
            else:
                self.tensors[init.name] = TensorInfo(
                    name=init.name,
                    is_initializer=True
                )

        for idx, node in enumerate(self.graph.node):
            node_name = node.name or f"{node.op_type}_{idx}"
            self.node_to_idx[node_name] = idx
            self.idx_to_node[idx] = node_name

            for output in node.output:
                if output:
                    if output not in self.tensors:
                        self.tensors[output] = TensorInfo(name=output)
                    self.tensors[output].producer_node = node_name
                    self.tensors[output].producer_node_idx = idx

            for inp in node.input:
                if inp:
                    if inp not in self.tensors:
                        self.tensors[inp] = TensorInfo(name=inp)
                    self.tensors[inp].consumers.append((node_name, idx))

        for out in self.graph.output:
            if out.name in self.tensors:
                self.tensors[out.name].is_output = True

    def get_slice_inputs(self, start_idx: int, end_idx: int) -> list[str]:
        """Get all external inputs needed by nodes in range [start_idx, end_idx)."""
        internal_outputs = set()
        for idx in range(start_idx, end_idx):
            node = self.graph.node[idx]
            for output in node.output:
                if output:
                    internal_outputs.add(output)

        external_inputs = []
        seen = set()
        for idx in range(start_idx, end_idx):
            node = self.graph.node[idx]
            for inp in node.input:
                if inp and inp not in internal_outputs and inp not in seen:
                    info = self.tensors.get(inp)
                    if info and not info.is_initializer:
                        external_inputs.append(inp)
                        seen.add(inp)

        return external_inputs

    def get_slice_outputs(self, start_idx: int, end_idx: int) -> list[str]:
        """Get outputs from slice that are used by later nodes or are model outputs."""
        outputs = []
        for idx in range(start_idx, end_idx):
            node = self.graph.node[idx]
            for output in node.output:
                if not output:
                    continue
                info = self.tensors.get(output)
                if not info:
                    continue

                if info.is_output:
                    outputs.append(output)
                    continue

                for consumer_name, consumer_idx in info.consumers:
                    if consumer_idx >= end_idx:
                        outputs.append(output)
                        break

        return outputs

    def get_skip_connections(self, slice_ranges: list[tuple[int, int]]) -> dict[int, list[str]]:
        """
        For each slice, find tensors that skip over it (produced before, consumed after).

        Returns: {slice_idx: [tensor_names that skip over this slice]}
        """
        skip_connections = {}

        for slice_idx, (start, end) in enumerate(slice_ranges):
            skips = []
            for tensor_name, info in self.tensors.items():
                if info.is_initializer or info.producer_node_idx is None:
                    continue

                produced_before = info.producer_node_idx < start

                consumed_after = any(
                    consumer_idx >= end
                    for _, consumer_idx in info.consumers
                )

                if produced_before and consumed_after:
                    skips.append(tensor_name)

            skip_connections[slice_idx] = skips

        return skip_connections

    def get_tensor_lifetime(self, tensor_name: str) -> tuple[int, int]:
        """Get (first_use_idx, last_use_idx) for a tensor."""
        info = self.tensors.get(tensor_name)
        if not info:
            return (-1, -1)

        first_use = info.producer_node_idx if info.producer_node_idx is not None else 0

        if info.consumers:
            last_use = max(idx for _, idx in info.consumers)
        else:
            last_use = first_use

        if info.is_output:
            last_use = len(self.graph.node)

        return (first_use, last_use)

    def get_tensors_needed_at(self, node_idx: int) -> list[str]:
        """Get all non-initializer tensors needed as input at a specific node."""
        if node_idx >= len(self.graph.node):
            return []

        node = self.graph.node[node_idx]
        needed = []
        for inp in node.input:
            if inp:
                info = self.tensors.get(inp)
                if info and not info.is_initializer:
                    needed.append(inp)
        return needed

    def get_tensors_alive_at(self, node_idx: int) -> list[str]:
        """Get all tensors that must be kept in memory at a specific node index."""
        alive = []
        for tensor_name, info in self.tensors.items():
            if info.is_initializer:
                continue
            first, last = self.get_tensor_lifetime(tensor_name)
            if first <= node_idx <= last:
                alive.append(tensor_name)
        return alive

    def build_slice_dependencies(self, slice_ranges: list[tuple[int, int]]) -> dict:
        """
        Build complete dependency info for a set of slices.

        Returns: {
            slice_idx: {
                "inputs": [tensor_names needed from outside],
                "outputs": [tensor_names produced for later slices],
                "skip_inputs": [tensor_names from non-adjacent earlier slices],
            }
        }
        """
        result = {}

        for slice_idx, (start, end) in enumerate(slice_ranges):
            inputs = self.get_slice_inputs(start, end)
            outputs = self.get_slice_outputs(start, end)

            skip_inputs = []
            if slice_idx > 0:
                prev_end = slice_ranges[slice_idx - 1][1]
                for inp in inputs:
                    info = self.tensors.get(inp)
                    if info and info.producer_node_idx is not None:
                        if info.producer_node_idx < prev_end - 1:
                            skip_inputs.append(inp)

            result[slice_idx] = {
                "inputs": inputs,
                "outputs": outputs,
                "skip_inputs": skip_inputs,
                "node_range": (start, end),
            }

        return result

    def __repr__(self):
        return f"TensorGraph({len(self.tensors)} tensors, {len(self.graph.node)} nodes)"
