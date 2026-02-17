from __future__ import annotations

from typing import TYPE_CHECKING

from dsperse.src.analyzers.schema import Backend
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove

if TYPE_CHECKING:
    from dsperse.src.analyzers.schema import RunSliceMetadata

WITNESS_FILENAME = {
    Backend.JSTPROVE: "output_witness.bin",
    Backend.EZKL: "output.json",
}


def instantiate_backend(backend_name: str):
    b = (backend_name or "").lower()
    if b == Backend.JSTPROVE:
        return JSTprove()
    if b == Backend.EZKL:
        return EZKL()
    raise ValueError(f"Unknown backend: {backend_name}")


def resolve_circuit_path(meta: "RunSliceMetadata", backend: str) -> str | None:
    b = (backend or "").lower()
    if b == Backend.JSTPROVE:
        return meta.jstprove_circuit_path or meta.circuit_path
    if b == Backend.EZKL:
        return meta.ezkl_circuit_path or meta.circuit_path
    return meta.circuit_path
