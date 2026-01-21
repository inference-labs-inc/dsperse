"""
Typed metadata schema for DSperse.

This module defines strongly-typed dataclasses for all metadata structures,
enforcing consistent access patterns across the codebase.
"""

from dataclasses import dataclass, field, asdict
from enum import StrEnum
from typing import Optional
from pathlib import Path


class Backend(StrEnum):
    JSTPROVE = "jstprove"
    EZKL = "ezkl"
    ONNX = "onnx"
    AUTO = "auto"


class ExecutionMethod(StrEnum):
    JSTPROVE_GEN_WITNESS = "jstprove_gen_witness"
    EZKL_GEN_WITNESS = "ezkl_gen_witness"
    ONNX_ONLY = "onnx_only"
    ONNX_MULTI_INPUT = "onnx_multi_input"
    TILED = "tiled"
    JSTPROVE_FALLBACK_ONNX = "jstprove_fallback_onnx"
    EZKL_FALLBACK_ONNX = "ezkl_fallback_onnx"
    JSTPROVE_PROVE = "jstprove_prove"
    EZKL_PROVE = "ezkl_prove"
    JSTPROVE_VERIFY = "jstprove_verify"
    EZKL_VERIFY = "ezkl_verify"


@dataclass
class TensorShape:
    """Shape info extracted from ONNX graph I/O."""
    input: list[list[int | str]] = field(default_factory=list)
    output: list[list[int | str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "TensorShape":
        if not d:
            return cls()
        return cls(
            input=d.get("input", []),
            output=d.get("output", []),
        )


@dataclass
class WeightShape:
    """Shape info from weight parameters."""
    input: list[int] = field(default_factory=list)
    output: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "WeightShape":
        if not d:
            return cls()
        return cls(
            input=d.get("input", []),
            output=d.get("output", []),
        )


@dataclass
class SliceShape:
    """Combined shape info for a slice."""
    tensor_shape: TensorShape = field(default_factory=TensorShape)
    weight_shape: WeightShape = field(default_factory=WeightShape)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SliceShape":
        if not d:
            return cls()
        return cls(
            tensor_shape=TensorShape.from_dict(d.get("tensor_shape")),
            weight_shape=WeightShape.from_dict(d.get("weight_shape")),
        )


@dataclass
class Dependencies:
    """Input/output dependencies for a slice."""
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    filtered_inputs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Dependencies":
        if not d:
            return cls()
        return cls(
            input=d.get("input", []),
            output=d.get("output", []),
            filtered_inputs=d.get("filtered_inputs", []),
        )


@dataclass
class TileInfo:
    """Nested tile artifact info within tiling config."""
    path: str = ""
    conv_out: tuple[int, int] = (0, 0)

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["TileInfo"]:
        if not d:
            return None
        conv_out = d.get("conv_out", [0, 0])
        return cls(
            path=d.get("path", ""),
            conv_out=(conv_out[0], conv_out[1]) if isinstance(conv_out, list) else conv_out,
        )

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "conv_out": list(self.conv_out),
        }


@dataclass
class TilingInfo:
    """Tiling configuration for a slice."""
    slice_idx: int = 0
    tile_size: int = 0
    num_tiles: int = 1
    tiles_y: int = 1
    tiles_x: int = 1
    halo: tuple[int, int] = (0, 0)
    out_tile: tuple[int, int] = (0, 0)
    stride: tuple[int, int] = (1, 1)
    c_in: int = 0
    c_out: int = 0
    input_name: str = "input"
    output_name: str = "output"
    tile: Optional[TileInfo] = None

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["TilingInfo"]:
        if not d:
            return None
        halo = d.get("halo", [0, 0])
        out_tile = d.get("out_tile", [0, 0])
        stride = d.get("stride", [1, 1])
        return cls(
            slice_idx=d.get("slice_idx", 0),
            tile_size=d.get("tile_size", 0),
            num_tiles=d.get("num_tiles", 1),
            tiles_y=d.get("tiles_y", 1),
            tiles_x=d.get("tiles_x", 1),
            halo=(halo[0], halo[1]) if isinstance(halo, list) else halo,
            out_tile=(out_tile[0], out_tile[1]) if isinstance(out_tile, list) else out_tile,
            stride=(stride[0], stride[1]) if isinstance(stride, list) else stride,
            c_in=d.get("c_in", 0),
            c_out=d.get("c_out", 0),
            input_name=d.get("input_name", "input"),
            output_name=d.get("output_name", "output"),
            tile=TileInfo.from_dict(d.get("tile")),
        )

    def to_dict(self) -> dict:
        d = {
            "slice_idx": self.slice_idx,
            "tile_size": self.tile_size,
            "num_tiles": self.num_tiles,
            "tiles_y": self.tiles_y,
            "tiles_x": self.tiles_x,
            "halo": list(self.halo),
            "out_tile": list(self.out_tile),
            "stride": list(self.stride),
            "c_in": self.c_in,
            "c_out": self.c_out,
            "input_name": self.input_name,
            "output_name": self.output_name,
        }
        if self.tile:
            d["tile"] = self.tile.to_dict()
        return d


@dataclass
class CompilationFiles:
    """Paths to compiled circuit artifacts."""
    compiled: Optional[str] = None
    settings: Optional[str] = None
    pk_key: Optional[str] = None
    vk_key: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "CompilationFiles":
        if not d:
            return cls()
        return cls(
            compiled=d.get("compiled") or d.get("compiled_circuit") or d.get("circuit"),
            settings=d.get("settings"),
            pk_key=d.get("pk_key"),
            vk_key=d.get("vk_key"),
        )


@dataclass
class BackendCompilation:
    """Compilation status for a single backend."""
    compiled: bool = False
    tiled: bool = False
    files: CompilationFiles = field(default_factory=CompilationFiles)
    compilation_timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "BackendCompilation":
        if not d:
            return cls()
        files_dict = d.get("files", {})
        if d.get("tiled") and "tile_0" in files_dict:
            files_dict = files_dict.get("tile_0", {})
        return cls(
            compiled=d.get("compiled", False),
            tiled=d.get("tiled", False),
            files=CompilationFiles.from_dict(files_dict),
            compilation_timestamp=d.get("compilation_timestamp"),
        )


@dataclass
class Compilation:
    """Compilation info for all backends."""
    jstprove: BackendCompilation = field(default_factory=BackendCompilation)
    ezkl: BackendCompilation = field(default_factory=BackendCompilation)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Compilation":
        if not d:
            return cls()
        return cls(
            jstprove=BackendCompilation.from_dict(d.get(Backend.JSTPROVE)),
            ezkl=BackendCompilation.from_dict(d.get(Backend.EZKL)),
        )


@dataclass
class SliceMetadata:
    """
    Full metadata for a single slice.

    This is the authoritative schema for slice metadata stored in:
    - slices/metadata.json (model-level, multiple slices)
    - slice_#/metadata.json (per-slice, single slice)
    """
    index: int
    filename: str = ""
    path: str = ""
    relative_path: str = ""
    parameters: int = 0
    shape: SliceShape = field(default_factory=SliceShape)
    dependencies: Dependencies = field(default_factory=Dependencies)
    layers: list[dict] = field(default_factory=list)
    tiling: Optional[TilingInfo] = None
    compilation: Compilation = field(default_factory=Compilation)
    dsperse_version: Optional[str] = None
    opset_version: Optional[int] = None
    slice_metadata: Optional[str] = None
    slice_metadata_relative_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SliceMetadata":
        return cls(
            index=d.get("index", 0),
            filename=d.get("filename", ""),
            path=d.get("path", ""),
            relative_path=d.get("relative_path", ""),
            parameters=d.get("parameters", 0),
            shape=SliceShape.from_dict(d.get("shape")),
            dependencies=Dependencies.from_dict(d.get("dependencies")),
            layers=d.get("layers", []),
            tiling=TilingInfo.from_dict(d.get("tiling")),
            compilation=Compilation.from_dict(d.get("compilation")),
            dsperse_version=d.get("dsperse_version"),
            opset_version=d.get("opset_version"),
            slice_metadata=d.get("slice_metadata"),
            slice_metadata_relative_path=d.get("slice_metadata_relative_path"),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.tiling is None:
            del d["tiling"]
        return d

    @property
    def output_shape(self) -> list[list[int | str]]:
        """Convenience accessor for output tensor shapes."""
        return self.shape.tensor_shape.output

    @property
    def input_shape(self) -> list[list[int | str]]:
        """Convenience accessor for input tensor shapes."""
        return self.shape.tensor_shape.input

    @property
    def output_names(self) -> list[str]:
        """Convenience accessor for output tensor names."""
        return self.dependencies.output

    @property
    def input_names(self) -> list[str]:
        """Convenience accessor for filtered input tensor names."""
        return self.dependencies.filtered_inputs


@dataclass
class RunSliceMetadata:
    """
    Flattened slice metadata for runtime execution.

    This is the schema used in run/metadata.json after transformation
    by RunnerAnalyzer.process_slices(). Key differences from SliceMetadata:
    - Shapes are flattened to top-level (input_shape, output_shape)
    - Circuit paths are normalized to slice-prefixed format
    - Backend selection info is exposed
    """
    path: str
    input_shape: list[list[int | str]] = field(default_factory=list)
    output_shape: list[list[int | str]] = field(default_factory=list)
    dependencies: Dependencies = field(default_factory=Dependencies)
    parameters: int = 0
    tiling: Optional[TilingInfo] = None
    backend: str = Backend.ONNX
    ezkl: bool = False
    ezkl_compatible: bool = True
    circuit_size: int = 0
    circuit_path: Optional[str] = None
    settings_path: Optional[str] = None
    vk_path: Optional[str] = None
    pk_path: Optional[str] = None
    jstprove_circuit_path: Optional[str] = None
    ezkl_circuit_path: Optional[str] = None
    jstprove_settings_path: Optional[str] = None
    ezkl_settings_path: Optional[str] = None
    ezkl_pk_path: Optional[str] = None
    ezkl_vk_path: Optional[str] = None
    slice_metadata_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunSliceMetadata":
        return cls(
            path=d.get("path", ""),
            input_shape=d.get("input_shape", []),
            output_shape=d.get("output_shape", []),
            dependencies=Dependencies.from_dict(d.get("dependencies")),
            parameters=d.get("parameters", 0),
            tiling=TilingInfo.from_dict(d.get("tiling")),
            backend=d.get("backend", Backend.ONNX),
            ezkl=d.get("ezkl", False),
            ezkl_compatible=d.get("ezkl_compatible", True),
            circuit_size=d.get("circuit_size", 0),
            circuit_path=d.get("circuit_path"),
            settings_path=d.get("settings_path"),
            vk_path=d.get("vk_path"),
            pk_path=d.get("pk_path"),
            jstprove_circuit_path=d.get("jstprove_circuit_path"),
            ezkl_circuit_path=d.get("ezkl_circuit_path"),
            jstprove_settings_path=d.get("jstprove_settings_path"),
            ezkl_settings_path=d.get("ezkl_settings_path"),
            ezkl_pk_path=d.get("ezkl_pk_path"),
            ezkl_vk_path=d.get("ezkl_vk_path"),
            slice_metadata_path=d.get("slice_metadata_path"),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.tiling is None:
            del d["tiling"]
        return d

    def get_target_shape(self, index: int = 0) -> list[int]:
        """Get target output shape for reshaping, replacing dynamic dims with 1."""
        if not self.output_shape or index >= len(self.output_shape):
            return []
        return [d if isinstance(d, int) else 1 for d in self.output_shape[index]]


@dataclass
class ModelMetadata:
    """Top-level metadata for sliced model."""
    original_model: str = ""
    model_type: str = ""
    total_parameters: int = 0
    input_shape: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    slice_points: list[int] = field(default_factory=list)
    slices: list[SliceMetadata] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMetadata":
        slices_raw = d.get("slices", [])
        slices = [SliceMetadata.from_dict(s) for s in slices_raw]
        return cls(
            original_model=d.get("original_model", ""),
            model_type=d.get("model_type", ""),
            total_parameters=d.get("total_parameters", 0),
            input_shape=d.get("input_shape", []),
            output_shapes=d.get("output_shapes", []),
            slice_points=d.get("slice_points", []),
            slices=slices,
        )

    def to_dict(self) -> dict:
        return {
            "original_model": self.original_model,
            "model_type": self.model_type,
            "total_parameters": self.total_parameters,
            "input_shape": self.input_shape,
            "output_shapes": self.output_shapes,
            "slice_points": self.slice_points,
            "slices": [s.to_dict() for s in self.slices],
        }

    @classmethod
    def load(cls, path: str | Path) -> "ModelMetadata":
        """Load metadata from JSON file."""
        import json
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file."""
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TileResult:
    """Result of a single tile operation (proving/verification/execution)."""
    tile_idx: int
    success: bool
    error: Optional[str] = None
    method: Optional[str] = None
    time_sec: float = 0.0
    proof_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"tile_idx": self.tile_idx, "success": self.success}
        if self.error is not None:
            d["error"] = self.error
        if self.method is not None:
            d["method"] = self.method
        if self.time_sec > 0:
            d["time_sec"] = self.time_sec
        if self.proof_path is not None:
            d["proof_path"] = self.proof_path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TileResult":
        return cls(
            tile_idx=d.get("tile_idx", 0),
            success=d.get("success", False),
            error=d.get("error"),
            method=d.get("method"),
            time_sec=d.get("time_sec", 0.0),
            proof_path=d.get("proof_path"),
        )


@dataclass
class ExecutionInfo:
    """Execution info for witness generation. Only includes fields that are actually read by consumers."""
    method: str
    success: bool = False
    error: Optional[str] = None
    witness_file: Optional[str] = None
    tiles: list["TileResult"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"method": self.method, "success": self.success}
        if self.error is not None:
            d["error"] = self.error
        if self.witness_file is not None:
            d["witness_file"] = self.witness_file
        if self.tiles:
            d["tile_exec_infos"] = [t.to_dict() for t in self.tiles]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionInfo":
        tiles_raw = d.get("tiles") or d.get("tile_exec_infos") or []
        return cls(
            method=d.get("method", "unknown"),
            success=d.get("success", False),
            error=d.get("error"),
            witness_file=d.get("witness_file") or d.get("witness_path"),
            tiles=[TileResult.from_dict(t) for t in tiles_raw],
        )


@dataclass
class SliceResult:
    """Result of a slice-level operation (proving/verification/execution)."""
    slice_id: str
    success: bool
    method: Optional[str] = None
    error: Optional[str] = None
    proof_path: Optional[str] = None
    time_sec: float = 0.0
    tiles: list[TileResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"slice_id": self.slice_id, "success": self.success}
        if self.method is not None:
            d["method"] = self.method
        if self.error is not None:
            d["error"] = self.error
        if self.proof_path is not None:
            d["proof_path"] = self.proof_path
        if self.time_sec > 0:
            d["time_sec"] = self.time_sec
        if self.tiles:
            d["tiles"] = [t.to_dict() for t in self.tiles]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SliceResult":
        tiles_raw = d.get("tiles") or d.get("tile_proofs_info") or d.get("tile_verifs_info") or []
        return cls(
            slice_id=d.get("slice_id", ""),
            success=d.get("success", False),
            method=d.get("method"),
            error=d.get("error"),
            proof_path=d.get("proof_path"),
            time_sec=d.get("time_sec", 0.0),
            tiles=[TileResult.from_dict(t) for t in tiles_raw],
        )


@dataclass
class ExecutionNode:
    """A node in the execution chain representing a slice's execution configuration."""
    slice_id: str
    primary: Optional[str] = None
    fallbacks: list[str] = field(default_factory=list)
    use_circuit: bool = False
    next: Optional[str] = None
    circuit_path: Optional[str] = None
    onnx_path: Optional[str] = None
    backend: str = Backend.ONNX

    def to_dict(self) -> dict:
        return {
            "slice_id": self.slice_id,
            "primary": self.primary,
            "fallbacks": self.fallbacks,
            "use_circuit": self.use_circuit,
            "next": self.next,
            "circuit_path": self.circuit_path,
            "onnx_path": self.onnx_path,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionNode":
        return cls(
            slice_id=d.get("slice_id", ""),
            primary=d.get("primary"),
            fallbacks=d.get("fallbacks", []),
            use_circuit=d.get("use_circuit", False),
            next=d.get("next"),
            circuit_path=d.get("circuit_path"),
            onnx_path=d.get("onnx_path"),
            backend=d.get("backend", Backend.ONNX),
        )


@dataclass
class ExecutionResultEntry:
    """Result entry for a slice's execution in the pipeline (witness + proof + verification)."""
    slice_id: str
    witness_execution: Optional[ExecutionInfo] = None
    proof_execution: Optional[SliceResult] = None
    verification_execution: Optional[SliceResult] = None

    def to_dict(self) -> dict:
        d: dict = {"slice_id": self.slice_id}
        if self.witness_execution:
            d["witness_execution"] = self.witness_execution.to_dict()
        if self.proof_execution:
            d["proof_execution"] = self.proof_execution.to_dict()
        if self.verification_execution:
            d["verification_execution"] = self.verification_execution.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionResultEntry":
        witness_raw = d.get("witness_execution")
        proof_raw = d.get("proof_execution")
        verif_raw = d.get("verification_execution")
        return cls(
            slice_id=d.get("slice_id", ""),
            witness_execution=ExecutionInfo.from_dict(witness_raw) if witness_raw else None,
            proof_execution=SliceResult.from_dict(proof_raw) if proof_raw else None,
            verification_execution=SliceResult.from_dict(verif_raw) if verif_raw else None,
        )


@dataclass
class ExecutionChain:
    """The execution chain structure containing nodes and results."""
    head: Optional[str] = None
    nodes: dict[str, ExecutionNode] = field(default_factory=dict)
    fallback_map: dict[str, list[str]] = field(default_factory=dict)
    execution_results: list[ExecutionResultEntry] = field(default_factory=list)
    jstprove_proved_slices: int = 0
    ezkl_proved_slices: int = 0
    jstprove_verified_slices: int = 0
    ezkl_verified_slices: int = 0

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "fallback_map": self.fallback_map,
            "execution_results": [e.to_dict() for e in self.execution_results],
            "jstprove_proved_slices": self.jstprove_proved_slices,
            "ezkl_proved_slices": self.ezkl_proved_slices,
            "jstprove_verified_slices": self.jstprove_verified_slices,
            "ezkl_verified_slices": self.ezkl_verified_slices,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "ExecutionChain":
        if not d:
            return cls()
        nodes_raw = d.get("nodes", {})
        results_raw = d.get("execution_results", [])
        return cls(
            head=d.get("head"),
            nodes={k: ExecutionNode.from_dict(v) for k, v in nodes_raw.items()},
            fallback_map=d.get("fallback_map", {}),
            execution_results=[ExecutionResultEntry.from_dict(e) for e in results_raw],
            jstprove_proved_slices=d.get("jstprove_proved_slices", 0),
            ezkl_proved_slices=d.get("ezkl_proved_slices", 0),
            jstprove_verified_slices=d.get("jstprove_verified_slices", 0),
            ezkl_verified_slices=d.get("ezkl_verified_slices", 0),
        )

    def get_result_for_slice(self, slice_id: str) -> Optional[ExecutionResultEntry]:
        """Find execution result entry for a specific slice."""
        for entry in self.execution_results:
            if entry.slice_id == slice_id:
                return entry
        return None


@dataclass
class RunMetadata:
    """Top-level run metadata structure containing slices and execution chain."""
    slices: dict[str, RunSliceMetadata] = field(default_factory=dict)
    execution_chain: ExecutionChain = field(default_factory=ExecutionChain)
    circuit_slices: dict[str, bool] = field(default_factory=dict)
    overall_security: float = 0.0
    packaging_type: Optional[str] = None
    source_path: Optional[str] = None
    run_directory: Optional[str] = None
    model_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "slices": {k: v.to_dict() for k, v in self.slices.items()},
            "execution_chain": self.execution_chain.to_dict(),
            "circuit_slices": self.circuit_slices,
            "overall_security": self.overall_security,
            "packaging_type": self.packaging_type,
            "source_path": self.source_path,
            "run_directory": self.run_directory,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "RunMetadata":
        if not d:
            return cls()
        slices_raw = d.get("slices", {})
        return cls(
            slices={k: RunSliceMetadata.from_dict(v) for k, v in slices_raw.items()},
            execution_chain=ExecutionChain.from_dict(d.get("execution_chain")),
            circuit_slices=d.get("circuit_slices", {}),
            overall_security=d.get("overall_security", 0.0),
            packaging_type=d.get("packaging_type"),
            source_path=d.get("source_path"),
            run_directory=d.get("run_directory"),
            model_path=d.get("model_path"),
        )

    def get_slice(self, slice_id: str) -> Optional[RunSliceMetadata]:
        """Get metadata for a specific slice."""
        return self.slices.get(slice_id)

    def iter_circuit_slices(self):
        """Iterate over slices that have circuits (use_circuit=True in execution chain)."""
        for slice_id, node in self.execution_chain.nodes.items():
            if node.use_circuit:
                meta = self.slices.get(slice_id)
                if meta:
                    yield slice_id, meta
