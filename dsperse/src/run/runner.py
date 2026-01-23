"""
Runner for EzKL Circuit and ONNX Inference
"""

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.backends.onnx_models import OnnxModels
from dsperse.src.metadata.schema import RunSliceMetadata, TilingInfo, ChannelSplitInfo, Dependencies, ExecutionInfo, TileResult, Backend, ExecutionMethod, RunMetadata
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils
from dsperse.src.utils.storage import TieredStorage

logger = logging.getLogger(__name__)


def _run_single_tile_worker(args: dict) -> dict:
    """Worker function for parallel tile execution. Must be at module level for pickling."""
    import json
    from pathlib import Path
    from dsperse.src.backends.jstprove import JSTprove
    from dsperse.src.backends.ezkl import EZKL
    from dsperse.src.run.utils.runner_utils import RunnerUtils
    from dsperse.src.metadata.schema import ExecutionMethod

    tile_idx = args['tile_idx']
    tile_in = args['tile_in']
    tile_out = args['tile_out']
    has_jst = args['has_jst']
    has_ezkl = args['has_ezkl']
    slices_path = Path(args['slices_path'])

    try:
        if has_jst:
            jst_runner = JSTprove()
            circuit_path = RunnerUtils.resolve_relative_path(args['jstprove_circuit_path'], slices_path)
            success, result = jst_runner.generate_witness(tile_in, circuit_path, tile_out)
            if success:
                output_tensor = RunnerUtils.extract_output_tensor(result)
                if output_tensor is not None:
                    return {
                        'tile_idx': tile_idx,
                        'success': True,
                        'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                        'method': ExecutionMethod.JSTPROVE_GEN_WITNESS
                    }
            return {'tile_idx': tile_idx, 'success': False, 'error': f'JSTprove failed: {result}'}

        elif has_ezkl:
            ezkl_runner = EZKL()
            circuit_path = RunnerUtils.resolve_relative_path(args['ezkl_circuit_path'], slices_path)
            vk_path = RunnerUtils.resolve_relative_path(args['vk_path'], slices_path) if args['vk_path'] else None
            settings_path = RunnerUtils.resolve_relative_path(args['settings_path'], slices_path) if args['settings_path'] else None
            success, result = ezkl_runner.generate_witness(tile_in, circuit_path, tile_out, vk_path, settings_path)
            if success:
                output_tensor = RunnerUtils.extract_output_tensor(result)
                if output_tensor is not None:
                    return {
                        'tile_idx': tile_idx,
                        'success': True,
                        'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                        'method': ExecutionMethod.EZKL_GEN_WITNESS
                    }
            return {'tile_idx': tile_idx, 'success': False, 'error': f'EZKL failed: {result}'}

        return {'tile_idx': tile_idx, 'success': False, 'error': 'No backend available'}

    except Exception as e:
        return {'tile_idx': tile_idx, 'success': False, 'error': str(e)}

class Runner:
    def __init__(self, run_metadata_path: str = None, save_metadata_path: str = None, parallel_tiles: int = 1,
                 circuit_cache_dir: str = None, hot_storage: str = None, cold_storage: str = None):
        """Initialize the Runner.

        Args:
            run_metadata_path: Path to existing run metadata
            save_metadata_path: Path where to save generated metadata
            parallel_tiles: Number of parallel tile execution processes
            circuit_cache_dir: Directory for caching circuit files (e.g., RAM disk)
            hot_storage: Fast storage path (e.g., RAM disk) for run outputs
            cold_storage: Persistent storage path (e.g., SSD) - data drains here async
        """
        self._provided_run_metadata_path = run_metadata_path
        self._save_metadata_path = save_metadata_path
        self.run_metadata = None
        self.last_run_dir: Path | None = None
        self.force_backend: str | None = None
        self.parallel_tiles = max(1, parallel_tiles)
        self.circuit_cache_dir = Path(circuit_cache_dir) if circuit_cache_dir else None
        self._cached_circuits: dict[str, Path] = {}

        self._tiered_storage: TieredStorage | None = None
        if hot_storage and cold_storage:
            self._tiered_storage = TieredStorage(Path(hot_storage), Path(cold_storage))

        try:
            self.ezkl_runner = EZKL()
        except RuntimeError:
            self.ezkl_runner = None
            logger.warning("EZKL CLI not available. EZKL backend will be disabled.")

        try:
            self.jstprove_runner = JSTprove()
        except RuntimeError:
            self.jstprove_runner = None
            logger.warning("JSTprove CLI not available. JSTprove backend will be disabled.")

    def run(self, input_json_path, slice_path: str, output_path: str = None, backend: str | None = None) -> dict:
        """Run inference through the chain using run/metadata.json.

        slice_path can be provided here (preferred) or at construction time for backward compatibility.
        
        Args:
            input_json_path: Path to the input JSON tensor file
            slice_path: Path to the slices directory or packaged slices (.dsperse/.dslice)
            output_path: Optional path where run data/results should be saved
            backend: Optional backend selector ('jstprove' | 'ezkl' | 'onnx').
                     - When provided, applies only at run-time and only affects slices that
                       have multiple circuit backends compiled. If 'onnx', skips circuit backends.
        """
        # Ensure slices path is available and valid
        if slice_path is None or not Path(slice_path).exists():
            raise Exception("A valid path must be provided for slices")
        self.slices_path = Path(slice_path)

        # convert to dirs
        format = Converter.detect_type(self.slices_path)
        if format != "dirs":
            slices_path = Converter.convert(str(self.slices_path), output_type="dirs")
            self.slices_path = Path(slices_path)

        # Generate run metadata if needed
        self._generate_run_metadata(format)

        # Apply optional one-shot backend override for this run only
        prev_forced = self.force_backend
        if backend is not None:
            self.force_backend = backend

        try:
            # run inference
            results = self._run(input_json_path=input_json_path, output_path=output_path)
        finally:
            # Restore previous forced backend setting
            self.force_backend = prev_forced

        if format != "dirs":
            self.slices_path = Converter.convert(str(self.slices_path), output_type=format, cleanup=True)

        return results


    def _run_ezkl_slice(self, meta: RunSliceMetadata, input_tensor_path, output_witness_path, slice_dir: Path = None):
        """Run EZKL inference for a slice with fallback to ONNX."""
        if self.ezkl_runner is None:
            return False, "EZKL CLI not available", ExecutionInfo(
                method=ExecutionMethod.EZKL_GEN_WITNESS, success=False, error='EZKL CLI not available'
            )

        model_path = meta.circuit_path
        vk_path = meta.vk_path
        settings_path = meta.settings_path

        if model_path and not os.path.isabs(str(model_path)):
            model_path = RunnerUtils.resolve_relative_path(model_path, slice_dir)
        if vk_path and not os.path.isabs(str(vk_path)):
            vk_path = RunnerUtils.resolve_relative_path(vk_path, slice_dir)
        if settings_path and not os.path.isabs(str(settings_path)):
            settings_path = RunnerUtils.resolve_relative_path(settings_path, slice_dir)

        try:
            success, output_tensor = self.ezkl_runner.generate_witness(
                input_file=input_tensor_path,
                model_path=model_path,
                output_file=output_witness_path,
                vk_path=vk_path,
                settings_path=settings_path
            )
        except Exception as e:
            success = False
            output_tensor = str(e)

        exec_info = ExecutionInfo(
            method=ExecutionMethod.EZKL_GEN_WITNESS,
            success=success,
            error=None if success else (output_tensor if isinstance(output_tensor, str) else "Unknown EZKL error"),
            witness_file=str(output_witness_path),
        )

        return success, output_tensor, exec_info

    def _run_jstprove_slice(self, meta: RunSliceMetadata, input_tensor_path, output_witness_path, slice_dir: Path = None):
        """Run JSTprove inference for a slice with fallback to ONNX."""
        if self.jstprove_runner is None:
            return False, "JSTprove CLI not available", ExecutionInfo(
                method=ExecutionMethod.JSTPROVE_GEN_WITNESS, success=False, error='JSTprove CLI not available'
            )

        circuit_path = meta.circuit_path
        if circuit_path and not os.path.isabs(str(circuit_path)):
            circuit_path = RunnerUtils.resolve_relative_path(circuit_path, slice_dir)

        try:
            witness_file_path = Path(output_witness_path).with_name("output_witness.bin")
            success, output_tensor = self.jstprove_runner.generate_witness(
                input_file=input_tensor_path,
                model_path=circuit_path,
                output_file=output_witness_path,
            )
        except Exception as e:
            success = False
            output_tensor = str(e)
            witness_file_path = Path(output_witness_path).with_name("output_witness.bin")

        exec_info = ExecutionInfo(
            method=ExecutionMethod.JSTPROVE_GEN_WITNESS,
            success=success,
            error=None if success else (output_tensor if isinstance(output_tensor, str) else "Unknown JSTprove error"),
            witness_file=str(witness_file_path),
        )

        return success, output_tensor, exec_info

    def _save_inference_output(self, results, output_path):
        """Save inference_output.json with execution details."""
        model_path = self.run_metadata.model_path or "unknown"
        slice_results = results.get("slice_results", {})

        def _get_method(r):
            return r.method if isinstance(r, ExecutionInfo) else r.get("method", "")

        def _get_tiles(r):
            if isinstance(r, ExecutionInfo):
                return r.tiles
            return r.get("tiles") or r.get("tile_exec_infos") or []

        def _count_tile_method(tiles, method_prefix):
            return sum(1 for t in tiles if (t.method if isinstance(t, TileResult) else t.get("method", "")).startswith(method_prefix))

        ezkl_complete = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.EZKL_GEN_WITNESS)
        jstprove_complete = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.JSTPROVE_GEN_WITNESS)
        jstprove_tiled_slices = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.TILED and _count_tile_method(_get_tiles(r), Backend.JSTPROVE) > 0)
        ezkl_tiled_slices = sum(1 for r in slice_results.values() if _get_method(r) == ExecutionMethod.TILED and _count_tile_method(_get_tiles(r), Backend.EZKL) > 0)
        jstprove_complete += jstprove_tiled_slices
        ezkl_complete += ezkl_tiled_slices
        total_slices = len(slice_results)

        execution_results = []
        for slice_id, exec_info in slice_results.items():
            if isinstance(exec_info, ExecutionInfo):
                witness_execution = exec_info.to_dict()
            else:
                witness_execution = {
                    "method": exec_info.get("method", "unknown"),
                    "success": exec_info.get("success", False),
                    "witness_file": exec_info.get("witness_file") or exec_info.get("witness_path"),
                    "tile_exec_infos": exec_info.get("tile_exec_infos", []),
                }
                if exec_info.get("error"):
                    witness_execution["error"] = exec_info["error"]

            execution_results.append({"slice_id": slice_id, "witness_execution": witness_execution})

        # Calculate security percentage (any circuit backend counts as secure)
        circuit_slices = ezkl_complete + jstprove_complete
        security_percent = (circuit_slices / total_slices * 100) if total_slices > 0 else 0

        # Build output structure
        inference_output = {
            "model_path": model_path,
            "output": results["output"],
            "tensor_shape": results["tensor_shape"],
            "execution_chain": {
                "total_slices": total_slices,
                "jstprove_witness_slices": jstprove_complete,
                "ezkl_witness_slices": ezkl_complete,
                "overall_security": f"{security_percent:.1f}%",
                "execution_results": execution_results
            },
            "performance_comparison": {
                "note": "Full ONNX vs verified chain comparison would require separate pure ONNX run"
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(inference_output, f, indent=2)

    def _generate_run_metadata(self, format: str = "dirs"):
        if self._provided_run_metadata_path:
            with open(self._provided_run_metadata_path, 'r') as f:
                self.run_metadata = RunMetadata.from_dict(json.load(f))
        else:
            if self._save_metadata_path:
                save_path = Path(self._save_metadata_path)
            else:
                ts = time.strftime('%Y%m%d_%H%M%S')
                base_dir = self.slices_path.parent
                if base_dir.name == "slices":
                    base_dir = base_dir.parent
                save_path = base_dir / "run" / f"run_{ts}" / "metadata.json"
            self.run_metadata = RunMetadata.from_dict(RunnerAnalyzer.generate_run_metadata(self.slices_path, save_path, format))

    def _cache_circuit(self, circuit_path: str, slice_id: str) -> str:
        """Copy circuit directory to cache for faster loading. Returns cached path or original if no cache configured.

        JSTprove circuits are directories containing multiple files (circuit.txt, quantized_model.onnx, metadata.json, etc).
        This method copies the entire parent directory and returns the path to the circuit file within the cache.
        """
        import shutil
        if not self.circuit_cache_dir or not circuit_path:
            return circuit_path

        src = Path(circuit_path)
        if not src.exists():
            return circuit_path

        if src.is_file():
            src_dir = src.parent
            filename = src.name
        else:
            src_dir = src
            filename = None

        cache_key = f"{slice_id}_circuit_dir"
        if cache_key in self._cached_circuits:
            cached_dir = self._cached_circuits[cache_key]
            if cached_dir.exists():
                return str(cached_dir / filename) if filename else str(cached_dir)

        self.circuit_cache_dir.mkdir(parents=True, exist_ok=True)
        dest_dir = self.circuit_cache_dir / f"{slice_id}_tiles"

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
        logger.info(f"Cached circuit directory for {slice_id}: {src_dir} -> {dest_dir}")

        self._cached_circuits[cache_key] = dest_dir
        return str(dest_dir / filename) if filename else str(dest_dir)

    def _clear_circuit_cache(self, slice_id: str = None):
        """Clear cached circuits. If slice_id provided, only clear that slice's cache."""
        import shutil
        if slice_id:
            cache_key = f"{slice_id}_circuit_dir"
            if cache_key in self._cached_circuits:
                cached = self._cached_circuits.pop(cache_key)
                if cached.exists():
                    if cached.is_dir():
                        shutil.rmtree(cached)
                    else:
                        cached.unlink()
        else:
            for cached in self._cached_circuits.values():
                if cached.exists():
                    if cached.is_dir():
                        shutil.rmtree(cached)
                    else:
                        cached.unlink()
            self._cached_circuits.clear()

    def _run_tiled_slice(self, slice_id: str, meta: RunSliceMetadata, tensor_cache: dict, run_dir: Path) -> ExecutionInfo:
        """Run a tiled slice: split (Python) → tiles (ONNX) → concat (Python)."""
        tiling = meta.tiling
        if not tiling:
            return ExecutionInfo(method=ExecutionMethod.TILED, success=False, error='missing_tiling_info')

        input_tensor = self._get_tiled_input_tensor(slice_id, tiling, meta, tensor_cache)

        self._run_tiling_split(slice_id, tiling, input_tensor, run_dir, tensor_cache)
        tile_results = self._run_tiling_parallel_tiles(slice_id, tiling, meta, run_dir, tensor_cache)
        self._run_tiling_concat(slice_id, tiling, run_dir, tensor_cache)

        return ExecutionInfo(
            method=ExecutionMethod.TILED,
            success=True,
            tiles=tile_results,
        )

    def _get_tiled_input_tensor(self, slice_id: str, tiling: TilingInfo, meta: RunSliceMetadata, tensor_cache: dict) -> torch.Tensor:
        input_name = tiling.input_name or (meta.dependencies.filtered_inputs[0] if meta.dependencies.filtered_inputs else "input")
        input_tensor = tensor_cache.get(input_name)
        if input_tensor is None:
            raise ValueError(f"Missing input tensor '{input_name}' for tiled slice {slice_id}")
        return input_tensor

    def _run_tiling_split(self, slice_id: str, tiling: TilingInfo, input_tensor: torch.Tensor, run_dir: Path, tensor_cache: dict) -> float:
        """Pure Python split: pad input and extract tiles."""
        start_time = time.time()

        slice_idx = tiling.slice_idx
        tile_size = tiling.tile_size
        halo_h, halo_w = tiling.halo
        tiles_y = tiling.tiles_y
        tiles_x = tiling.tiles_x
        num_tiles = tiling.num_tiles

        tile_with_halo_h = tile_size + 2 * halo_h
        tile_with_halo_w = tile_size + 2 * halo_w

        padded = F.pad(input_tensor, (halo_w, halo_w, halo_h, halo_h), mode='constant', value=0)

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                tile_idx = ty * tiles_x + tx
                y_start = ty * tile_size
                x_start = tx * tile_size
                y_end = y_start + tile_with_halo_h
                x_end = x_start + tile_with_halo_w

                tile = padded[:, :, y_start:y_end, x_start:x_end]
                cache_name = f"tile_{slice_idx}_{tile_idx}_in"
                tensor_cache[cache_name] = tile

        split_time = time.time() - start_time
        logger.info(f"Split {slice_id} completed in {split_time:.3f}s, produced {num_tiles} tiles")
        return split_time

    def _run_tiling_parallel_tiles(self, slice_id: str, tiling: TilingInfo, meta: RunSliceMetadata, run_dir: Path, tensor_cache: dict) -> list[TileResult]:
        import onnxruntime as ort
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor, as_completed

        num_tiles = tiling.num_tiles
        tile_info = tiling.tile
        tile_onnx_path = RunnerUtils.resolve_relative_path(tile_info.path, self.slices_path) if tile_info else None
        slice_idx = tiling.slice_idx

        if tile_onnx_path is None or not Path(tile_onnx_path).exists():
            raise ValueError(
                f"Tile ONNX path not found for {slice_id} (slice_idx={slice_idx}, num_tiles={num_tiles}): {tile_onnx_path}"
            )

        effective_backend = self.force_backend if self.force_backend else Backend.AUTO
        has_jst = effective_backend in (Backend.JSTPROVE, Backend.AUTO) and bool(meta.jstprove_circuit_path) and getattr(self, "jstprove_runner", None)
        has_ezkl = effective_backend in (Backend.EZKL, Backend.AUTO) and bool(meta.ezkl_circuit_path)

        tile_start = time.time()
        tile_exec_infos = []

        if has_jst or has_ezkl:
            backend_name = Backend.JSTPROVE if has_jst else Backend.EZKL
            slice_specific_dir = self.slices_path / slice_id

            jst_circuit_path = meta.jstprove_circuit_path
            ezkl_circuit_path = meta.ezkl_circuit_path
            if self.circuit_cache_dir:
                if has_jst and jst_circuit_path:
                    resolved_jst = RunnerUtils.resolve_relative_path(jst_circuit_path, self.slices_path)
                    if resolved_jst:
                        jst_circuit_path = self._cache_circuit(resolved_jst, slice_id)
                if has_ezkl and ezkl_circuit_path:
                    resolved_ezkl = RunnerUtils.resolve_relative_path(ezkl_circuit_path, self.slices_path)
                    if resolved_ezkl:
                        ezkl_circuit_path = self._cache_circuit(resolved_ezkl, f"{slice_id}_ezkl")

            tile_args_list = []
            for tile_idx in range(num_tiles):
                cache_input_name = f"tile_{slice_idx}_{tile_idx}_in"
                tile_tensor = tensor_cache.get(cache_input_name)
                if tile_tensor is None:
                    raise ValueError(f"Missing tile input tensor '{cache_input_name}' for slice {slice_id}")

                tile_run_dir = run_dir / slice_id / f"tile_{tile_idx}"
                tile_run_dir.mkdir(parents=True, exist_ok=True)
                tile_in = tile_run_dir / "input.json"
                tile_out = tile_run_dir / "output.json"
                Utils.write_input(tile_tensor, str(tile_in))

                tile_args_list.append({
                    'tile_idx': tile_idx,
                    'tile_in': str(tile_in),
                    'tile_out': str(tile_out),
                    'tile_onnx_path': str(tile_onnx_path),
                    'jstprove_circuit_path': jst_circuit_path,
                    'ezkl_circuit_path': ezkl_circuit_path,
                    'settings_path': meta.settings_path,
                    'vk_path': meta.vk_path,
                    'slice_specific_dir': str(slice_specific_dir),
                    'slices_path': str(self.slices_path),
                    'has_jst': has_jst,
                    'has_ezkl': has_ezkl,
                    'c_out': tiling.c_out,
                    'conv_out': tiling.tile.conv_out if tiling.tile else (0, 0),
                })

            parallel_count = min(self.parallel_tiles, num_tiles)
            if parallel_count > 1:
                logger.info(f"Running {num_tiles} tiles with {backend_name} circuits using {parallel_count} parallel processes")
                with ProcessPoolExecutor(max_workers=parallel_count) as executor:
                    futures = {executor.submit(_run_single_tile_worker, args): args['tile_idx'] for args in tile_args_list}
                    results_map = {}
                    for future in as_completed(futures):
                        tile_idx = futures[future]
                        try:
                            results_map[tile_idx] = future.result()
                        except Exception as e:
                            results_map[tile_idx] = {'success': False, 'error': str(e), 'tile_idx': tile_idx}

                for tile_idx in range(num_tiles):
                    result = results_map[tile_idx]
                    cache_output_name = f"tile_{slice_idx}_{tile_idx}_out"
                    if result['success'] and result.get('output_tensor') is not None:
                        output_tensor = torch.tensor(result['output_tensor'])
                        c_out = tiling.c_out
                        h_out, w_out = tiling.tile.conv_out if tiling.tile else (0, 0)
                        if c_out and h_out and w_out:
                            expected_numel = 1 * c_out * h_out * w_out
                            if output_tensor.numel() == expected_numel:
                                output_tensor = output_tensor.reshape(1, c_out, h_out, w_out)
                        tensor_cache[cache_output_name] = output_tensor
                        tile_exec_infos.append(TileResult(tile_idx=tile_idx, success=True, method=result.get('method', 'unknown')))
                    else:
                        tile_exec_infos.append(TileResult(tile_idx=tile_idx, success=False, error=result.get('error', 'unknown')))
                        raise RuntimeError(f"Tile {tile_idx} execution failed: {result.get('error', 'unknown')}")
            else:
                logger.info(f"Running {num_tiles} tiles with {backend_name} circuits (sequential)")
                for args in tile_args_list:
                    tile_idx = args['tile_idx']
                    cache_output_name = f"tile_{slice_idx}_{tile_idx}_out"

                    tile_meta = RunSliceMetadata(
                        path=args['tile_onnx_path'],
                        jstprove_circuit_path=args['jstprove_circuit_path'],
                        ezkl_circuit_path=args['ezkl_circuit_path'],
                        settings_path=args['settings_path'],
                        vk_path=args['vk_path'],
                        dependencies=meta.dependencies,
                    )

                    ok, result, t_info = RunnerUtils.execute_slice(
                        self, tile_meta, Path(args['tile_in']), Path(args['tile_out']), slice_specific_dir
                    )

                    output_tensor = RunnerUtils.extract_output_tensor(result) if ok else None
                    t_method = t_info.method if isinstance(t_info, ExecutionInfo) else t_info.get('method', 'unknown')
                    t_error = t_info.error if isinstance(t_info, ExecutionInfo) else t_info.get('error', 'unknown')

                    if ok and output_tensor is not None:
                        if isinstance(output_tensor, torch.Tensor) and output_tensor.dim() < 4:
                            c_out = tiling.c_out
                            h_out, w_out = tiling.tile.conv_out if tiling.tile else (0, 0)
                            if c_out and h_out and w_out:
                                expected_numel = 1 * c_out * h_out * w_out
                                if output_tensor.numel() == expected_numel:
                                    output_tensor = output_tensor.reshape(1, c_out, h_out, w_out)
                        tensor_cache[cache_output_name] = output_tensor
                        tile_exec_infos.append(TileResult(tile_idx=tile_idx, success=True, method=t_method))
                    else:
                        tile_exec_infos.append(TileResult(tile_idx=tile_idx, success=False, error=t_error))
                        raise RuntimeError(f"Tile {tile_idx} execution failed: {t_error}")
        else:
            session = ort.InferenceSession(str(tile_onnx_path))
            input_name = session.get_inputs()[0].name

            for tile_idx in range(num_tiles):
                cache_input_name = f"tile_{slice_idx}_{tile_idx}_in"
                cache_output_name = f"tile_{slice_idx}_{tile_idx}_out"

                tile_tensor = tensor_cache.get(cache_input_name)
                if tile_tensor is None:
                    raise ValueError(f"Missing tile input tensor '{cache_input_name}' for slice {slice_id}")

                if isinstance(tile_tensor, torch.Tensor):
                    input_arr = tile_tensor.detach().cpu().numpy().astype(np.float32)
                else:
                    input_arr = np.asarray(tile_tensor, dtype=np.float32)

                outputs = session.run(None, {input_name: input_arr})
                output_arr = outputs[0]

                output_tensor = torch.from_numpy(output_arr)
                tensor_cache[cache_output_name] = output_tensor

                tile_exec_infos.append(TileResult(tile_idx=tile_idx, success=True, method=ExecutionMethod.ONNX_ONLY))

        tile_time = time.time() - tile_start
        logger.info(f"All {num_tiles} tiles completed in {tile_time:.2f}s ({tile_time / num_tiles * 1000:.1f}ms avg)")
        return tile_exec_infos

    def _run_tiling_concat(self, slice_id: str, tiling: TilingInfo, run_dir: Path, tensor_cache: dict) -> float:
        """Pure Python concat: reassemble tiles into full output."""
        concat_start = time.time()

        slice_idx = tiling.slice_idx
        tiles_y = tiling.tiles_y
        tiles_x = tiling.tiles_x
        output_name = tiling.output_name

        rows = []
        for ty in range(tiles_y):
            row_tiles = []
            for tx in range(tiles_x):
                tile_idx = ty * tiles_x + tx
                cache_name = f"tile_{slice_idx}_{tile_idx}_out"
                tile = tensor_cache.get(cache_name)
                if tile is None:
                    raise ValueError(f"Missing tile output tensor '{cache_name}' for concat")
                row_tiles.append(tile)
            row = torch.cat(row_tiles, dim=3)
            rows.append(row)

        output = torch.cat(rows, dim=2)
        tensor_cache[output_name] = output

        concat_time = time.time() - concat_start
        logger.info(f"Concat {slice_id} completed in {concat_time:.3f}s, output shape {list(output.shape)}")
        return concat_time

    def _run_channel_group(self, group, group_input: torch.Tensor, run_dir: Path, output_shape: tuple) -> tuple[torch.Tensor, str]:
        """Run a single channel group, returning (output_tensor, method_used)."""
        import numpy as np
        import onnxruntime as ort

        forced = self.force_backend
        has_jst = bool(group.jstprove_circuit_path) and self.jstprove_runner
        has_ezkl = bool(group.ezkl_circuit_path) and group.vk_path

        if isinstance(group_input, torch.Tensor):
            input_arr = group_input.detach().cpu().numpy().astype(np.float32)
        else:
            input_arr = np.asarray(group_input, dtype=np.float32)

        group_dir = run_dir / f"channel_group_{group.group_idx}"
        group_dir.mkdir(parents=True, exist_ok=True)
        in_file = group_dir / "input.json"
        out_file = group_dir / "output.json"
        Utils.write_input(torch.from_numpy(input_arr), in_file)

        def _extract_and_reshape(result):
            tensor = RunnerUtils.extract_output_tensor(result)
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.dim() < 4:
                expected = np.prod(output_shape)
                if tensor.numel() == expected:
                    return tensor.reshape(output_shape)
            return tensor

        if forced == Backend.ONNX or (not has_jst and not has_ezkl):
            group_onnx_path = RunnerUtils.resolve_relative_path(group.path, self.slices_path)
            session = ort.InferenceSession(str(group_onnx_path))
            ort_input_name = session.get_inputs()[0].name
            outputs = session.run(None, {ort_input_name: input_arr})
            return torch.from_numpy(outputs[0]), ExecutionMethod.ONNX_ONLY

        if (forced == Backend.JSTPROVE and has_jst) or (has_jst and forced != Backend.EZKL):
            circuit_path = RunnerUtils.resolve_relative_path(group.jstprove_circuit_path, self.slices_path)
            success, result = self.jstprove_runner.generate_witness(str(in_file), circuit_path, str(out_file))
            if success:
                tensor = _extract_and_reshape(result)
                if tensor is not None:
                    return tensor, ExecutionMethod.JSTPROVE_GEN_WITNESS
            logger.warning(f"JSTprove failed for group {group.group_idx}, falling back")

        if has_ezkl:
            circuit_path = RunnerUtils.resolve_relative_path(group.ezkl_circuit_path, self.slices_path)
            vk_path = RunnerUtils.resolve_relative_path(group.vk_path, self.slices_path)
            settings_path = RunnerUtils.resolve_relative_path(group.settings_path, self.slices_path) if group.settings_path else None
            ezkl_in = RunnerUtils._flatten_input_for_ezkl(in_file)
            success, result = self.ezkl_runner.generate_witness(str(ezkl_in), circuit_path, str(out_file), vk_path, settings_path)
            if success:
                tensor = _extract_and_reshape(result)
                if tensor is not None:
                    return tensor, ExecutionMethod.EZKL_GEN_WITNESS
            logger.warning(f"EZKL failed for group {group.group_idx}, falling back")

        group_onnx_path = RunnerUtils.resolve_relative_path(group.path, self.slices_path)
        session = ort.InferenceSession(str(group_onnx_path))
        outputs = session.run(None, {session.get_inputs()[0].name: input_arr})
        return torch.from_numpy(outputs[0]), ExecutionMethod.ONNX_ONLY

    def _run_channel_split_slice(self, slice_id: str, meta: RunSliceMetadata, tensor_cache: dict, run_dir: Path) -> ExecutionInfo:
        import numpy as np

        channel_split = meta.channel_split
        if not channel_split:
            return ExecutionInfo(method="channel_split", success=False, error="missing_channel_split_info")

        input_name = channel_split.input_name or (meta.dependencies.filtered_inputs[0] if meta.dependencies.filtered_inputs else "input")
        input_tensor = tensor_cache.get(input_name)
        if input_tensor is None:
            raise ValueError(f"Missing input tensor '{input_name}' for channel-split slice {slice_id}")

        output_shape = (1, channel_split.c_out, channel_split.h, channel_split.w)

        partial_outputs = []
        methods_used = []
        for group in channel_split.groups:
            group_input = input_tensor[:, group.c_start:group.c_end, :, :]
            output, method = self._run_channel_group(group, group_input, run_dir, output_shape)
            partial_outputs.append(output)
            methods_used.append(method)

        summed = partial_outputs[0]
        for po in partial_outputs[1:]:
            summed = summed + po

        if channel_split.bias_path:
            bias_path = RunnerUtils.resolve_relative_path(channel_split.bias_path, self.slices_path)
            if Path(bias_path).exists():
                bias = np.load(bias_path)
                bias_tensor = torch.from_numpy(bias).reshape(1, -1, 1, 1)
                summed = summed + bias_tensor

        output_name = channel_split.output_name or (meta.dependencies.output[0] if meta.dependencies.output else "output")
        tensor_cache[output_name] = summed

        primary_method = methods_used[0] if len(set(methods_used)) == 1 else "channel_split_mixed"
        logger.info(f"Channel split {slice_id}: {channel_split.num_groups} groups via {primary_method}, output {list(summed.shape)}")

        return ExecutionInfo(method=f"channel_split:{primary_method}", success=True)

    def _run(self, output_path=None, input_json_path=None):
        exec_chain = self.run_metadata.execution_chain
        head = exec_chain.head
        nodes = exec_chain.nodes

        if self._tiered_storage:
            run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            run_dir = self._tiered_storage.initialize(run_id)
            self.last_run_dir = self._tiered_storage.cold_run_dir
        else:
            run_dir = RunnerUtils.make_run_dir(self.run_metadata, output_path, self.slices_path)
            self.last_run_dir = run_dir

        input_tensor = Utils.read_input(input_json_path)
        first_slice_meta = self.run_metadata.get_slice(head)
        first_slice_inputs = first_slice_meta.dependencies.filtered_inputs if first_slice_meta else []
        model_input_name = first_slice_inputs[0] if first_slice_inputs else "input"
        tensor_cache = {model_input_name: input_tensor}

        current_slice_id = head
        slice_results = {}
        final_tensor = None

        while current_slice_id:
            info = self.run_metadata.get_slice(current_slice_id)
            slice_dir = self.slices_path

            if self._tiered_storage:
                self._tiered_storage.wait_for_space()

            if info.channel_split:
                logger.info(f"Running channel-split slice {current_slice_id} with {info.channel_split.num_groups} groups")
                exec_info = self._run_channel_split_slice(current_slice_id, info, tensor_cache, run_dir)
                ok = exec_info.success
                output_names = info.dependencies.output
                if ok and output_names:
                    final_tensor = tensor_cache.get(output_names[0])
            elif info.tiling:
                logger.info(f"Running tiled slice {current_slice_id} with parallel tiles")
                exec_info = self._run_tiled_slice(current_slice_id, info, tensor_cache, run_dir)
                ok = exec_info.success
                output_names = info.dependencies.output
                if ok and output_names:
                    final_tensor = tensor_cache.get(output_names[0])
            else:
                filtered_inputs = [n for n in info.dependencies.filtered_inputs if n]
                output_names = info.dependencies.output
                node = nodes[current_slice_id]

                input_name = filtered_inputs[0] if filtered_inputs else "input"
                current_tensor = tensor_cache.get(input_name, input_tensor)

                use_circuit = node.use_circuit and self.force_backend != Backend.ONNX

                if use_circuit:
                    in_file, out_file = RunnerUtils.prepare_slice_io(run_dir, current_slice_id)
                    Utils.write_input(current_tensor, str(in_file))
                    ok, result, exec_info = RunnerUtils.execute_slice(self, info, in_file, out_file, slice_dir)
                else:
                    if len(filtered_inputs) > 1:
                        missing = [n for n in filtered_inputs if n not in tensor_cache]
                        if missing:
                            raise ValueError(f"Missing input tensors for {current_slice_id}: {missing}")
                        extra_tensors = {name: tensor_cache[name] for name in filtered_inputs}
                    else:
                        extra_tensors = {input_name: current_tensor}
                    ok, result, exec_info = RunnerUtils.run_onnx_multi_input_slice(info, None, slice_dir, extra_tensors)

                logger.info(f"[{current_slice_id}] ok={ok}, result type={type(result).__name__}")
                if ok and result is not None:
                    if isinstance(result, dict) and 'output_tensors' in result:
                        for oname, tensor in result['output_tensors'].items():
                            tensor_cache[oname] = tensor
                        first_output = list(result['output_tensors'].values())[0] if result['output_tensors'] else None
                        if first_output is not None:
                            final_tensor = first_output
                        else:
                            final_tensor = RunnerUtils.extract_output_tensor(result)
                    else:
                        out_tensor = RunnerUtils.extract_output_tensor(result)
                        if isinstance(out_tensor, torch.Tensor):
                            target_shape = info.get_target_shape()
                            if target_shape:
                                expected_numel = torch.prod(torch.tensor(target_shape)).item()
                                if out_tensor.numel() == expected_numel:
                                    out_tensor = out_tensor.reshape(target_shape)
                                    logger.info(f"[{current_slice_id}] Reshaped output to {target_shape}")
                                else:
                                    logger.warning(f"[{current_slice_id}] Cannot reshape: got {out_tensor.numel()} elements, expected {expected_numel}")
                            for oname in output_names:
                                tensor_cache[oname] = out_tensor
                            final_tensor = out_tensor

            slice_results[current_slice_id] = exec_info
            if not ok:
                err = exec_info.error if isinstance(exec_info, ExecutionInfo) else exec_info.get('error', 'unknown')
                raise Exception(f"Inference failed for {current_slice_id}: {err}")

            if self._tiered_storage:
                self._tiered_storage.mark_complete(current_slice_id)
            if self.circuit_cache_dir:
                self._clear_circuit_cache(current_slice_id)

            current_slice_id = nodes[current_slice_id].next

        if final_tensor is None:
            if slice_results:
                logger.warning("No output tensor produced by execution chain, using input tensor")
            final_tensor = input_tensor

        results = {
            "output": final_tensor.tolist(),
            "tensor_shape": list(final_tensor.shape),
            "slice_results": slice_results,
        }

        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_inference_output(results, run_dir / "run_results.json")

        if self._tiered_storage:
            self._tiered_storage.mark_complete("run_results.json")
            self._tiered_storage.shutdown()

        return results



if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../../models/doom",
        2: "../../models/net",
        3: "../../models/resnet",
        4: "../../models/yolov3",
        5: "../../models/age",
    }

    # Get model directory
    abs_path = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(abs_path, "slices")
    # slices_dir = os.path.join(slices_dir, "slice_0")
    input_json = os.path.join(abs_path, "input.json")
    run_metadata_path = None

    saved_run_metadata_path = None

    print(f"saves run metadata to {saved_run_metadata_path}")

    # Initialize runner (auto-generates run metadata if needed). Slices dir is now passed to run(...).
    runner = Runner(run_metadata_path=run_metadata_path, save_metadata_path=saved_run_metadata_path)

    # Run inference
    print(f"Running inference on model {base_paths[model_choice]}...")
    results = runner.run(input_json, slice_path=slices_dir)#, backend="onnx")

    # Display results
    print(f"\nOutput shape: {results['tensor_shape']}")
    print("Execution summary:")
    for slice_id, info in results["slice_results"].items():
        print(f"  {slice_id}: {info['method']}")
