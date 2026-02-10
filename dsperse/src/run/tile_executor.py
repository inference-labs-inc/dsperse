import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dsperse.src.analyzers.schema import (
    Backend, ExecutionMethod, RunSliceMetadata, TileResult, TilingInfo
)
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class TileExecutor:
    """Handles all tiling operations for sliced model execution."""

    def __init__(self, slices_path: Path, tensor_cache: dict):
        self.slices_path = slices_path
        self.tensor_cache = tensor_cache

    def get_input_tensor(self, slice_id: str, tiling: TilingInfo, meta: RunSliceMetadata) -> torch.Tensor:
        input_name = tiling.input_name or (
            meta.dependencies.filtered_inputs[0] if meta.dependencies.filtered_inputs else "input"
        )
        input_tensor = self.tensor_cache.get(input_name)
        if input_tensor is None:
            raise ValueError(f"Missing input tensor '{input_name}' for tiled slice {slice_id}")
        return input_tensor

    def split_into_tiles(self, slice_id: str, tiling: TilingInfo, input_tensor: torch.Tensor) -> float:
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
                self.tensor_cache[cache_name] = tile

        split_time = time.time() - start_time
        logger.info(f"Split {slice_id} completed in {split_time:.3f}s, produced {num_tiles} tiles")
        return split_time

    def reconstruct_from_tiles(self, slice_id: str, tiling: TilingInfo) -> float:
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
                tile = self.tensor_cache.get(cache_name)
                if tile is None:
                    raise ValueError(f"Missing tile output tensor '{cache_name}' for concat")
                row_tiles.append(tile)
            row = torch.cat(row_tiles, dim=3)
            rows.append(row)

        output = torch.cat(rows, dim=2)
        self.tensor_cache[output_name] = output

        concat_time = time.time() - concat_start
        logger.info(f"Concat {slice_id} completed in {concat_time:.3f}s, output shape {list(output.shape)}")
        return concat_time

    def get_execution_config(
        self, tiling: TilingInfo, meta: RunSliceMetadata, backend: str = None, has_jst_runner: bool = False
    ) -> dict:
        num_tiles = tiling.num_tiles
        has_per_tile_onnx = tiling.tiles and len(tiling.tiles) == num_tiles

        if has_per_tile_onnx:
            tile_onnx_path = self._resolve_path(tiling.tiles[0].path)
        elif tiling.tile:
            tile_onnx_path = self._resolve_path(tiling.tile.path)
        else:
            tile_onnx_path = None

        effective_backend = backend if backend else Backend.AUTO
        has_jst = (
            effective_backend in (Backend.JSTPROVE, Backend.AUTO)
            and bool(meta.jstprove_circuit_path)
            and has_jst_runner
        )
        has_ezkl = (
            effective_backend in (Backend.EZKL, Backend.AUTO)
            and bool(meta.ezkl_circuit_path)
            and (meta.ezkl_vk_path or meta.vk_path)
        )

        return {
            'tile_onnx_path': tile_onnx_path,
            'has_per_tile_onnx': has_per_tile_onnx,
            'effective_backend': effective_backend,
            'has_jst': has_jst,
            'has_ezkl': has_ezkl,
            'backend_name': Backend.JSTPROVE if has_jst else (Backend.EZKL if has_ezkl else Backend.ONNX)
        }

    def prepare_tasks(
        self, num_tiles: int, slice_idx: int, tiling: TilingInfo, meta: RunSliceMetadata,
        run_dir: Path, config: dict
    ) -> list[dict]:
        tile_args_list = []
        slice_specific_dir = self.slices_path / f"slice_{slice_idx}"

        for tile_idx in range(num_tiles):
            cache_input_name = f"tile_{slice_idx}_{tile_idx}_in"
            tile_tensor = self.tensor_cache.get(cache_input_name)
            if tile_tensor is None:
                raise ValueError(f"Missing tile input tensor '{cache_input_name}'")

            tile_run_dir = run_dir / f"slice_{slice_idx}" / f"tile_{tile_idx}"
            tile_run_dir.mkdir(parents=True, exist_ok=True)
            tile_in, tile_out = tile_run_dir / "input.json", tile_run_dir / "output.json"
            tile_input_name = tiling.input_name or "tile_in"
            Utils.write_input(tile_tensor, str(tile_in), tile_input_name)

            this_tile_onnx = (
                self._resolve_path(tiling.tiles[tile_idx].path) if config['has_per_tile_onnx']
                else config['tile_onnx_path']
            )
            conv_out = (
                tiling.tiles[tile_idx].conv_out if config['has_per_tile_onnx']
                else (tiling.tile.conv_out if tiling.tile else (0, 0))
            )

            tile_args_list.append({
                'tile_idx': tile_idx,
                'tile_in': str(tile_in),
                'tile_out': str(tile_out),
                'tile_onnx_path': str(this_tile_onnx),
                'jstprove_circuit_path': meta.jstprove_circuit_path,
                'ezkl_circuit_path': meta.ezkl_circuit_path,
                'settings_path': meta.settings_path,
                'vk_path': meta.vk_path,
                'jstprove_settings_path': meta.jstprove_settings_path,
                'ezkl_settings_path': meta.ezkl_settings_path,
                'ezkl_vk_path': meta.ezkl_vk_path,
                'ezkl_pk_path': meta.ezkl_pk_path,
                'slice_specific_dir': str(slice_specific_dir),
                'slices_path': str(self.slices_path),
                'has_jst': config['has_jst'],
                'has_ezkl': config['has_ezkl'],
                'c_out': tiling.c_out,
                'conv_out': conv_out,
            })
        return tile_args_list

    def prepare_tasks_in_memory(
        self, num_tiles: int, slice_idx: int, tiling: TilingInfo, meta: RunSliceMetadata,
        run_dir: Path, config: dict
    ) -> list[dict]:
        tile_args_list = []
        slice_specific_dir = self.slices_path / f"slice_{slice_idx}"

        for tile_idx in range(num_tiles):
            cache_input_name = f"tile_{slice_idx}_{tile_idx}_in"
            tile_tensor = self.tensor_cache.get(cache_input_name)
            if tile_tensor is None:
                raise ValueError(f"Missing tile input tensor '{cache_input_name}'")

            tile_run_dir = run_dir / f"slice_{slice_idx}" / f"tile_{tile_idx}"
            tile_run_dir.mkdir(parents=True, exist_ok=True)
            tile_out = tile_run_dir / "output.json"

            this_tile_onnx = (
                self._resolve_path(tiling.tiles[tile_idx].path) if config['has_per_tile_onnx']
                else config['tile_onnx_path']
            )
            conv_out = (
                tiling.tiles[tile_idx].conv_out if config['has_per_tile_onnx']
                else (tiling.tile.conv_out if tiling.tile else (0, 0))
            )

            tile_args_list.append({
                'tile_idx': tile_idx,
                'tile_tensor': tile_tensor,
                'tile_out': str(tile_out),
                'tile_onnx_path': str(this_tile_onnx),
                'jstprove_circuit_path': meta.jstprove_circuit_path,
                'ezkl_circuit_path': meta.ezkl_circuit_path,
                'settings_path': meta.settings_path,
                'vk_path': meta.vk_path,
                'jstprove_settings_path': meta.jstprove_settings_path,
                'ezkl_settings_path': meta.ezkl_settings_path,
                'ezkl_vk_path': meta.ezkl_vk_path,
                'ezkl_pk_path': meta.ezkl_pk_path,
                'slice_specific_dir': str(slice_specific_dir),
                'slices_path': str(self.slices_path),
                'has_jst': config['has_jst'],
                'has_ezkl': config['has_ezkl'],
                'c_out': tiling.c_out,
                'conv_out': conv_out,
            })
        return tile_args_list

    def process_result(self, result, tiling: TilingInfo, slice_idx: int, tile_idx: int):
        output_tensor = self._extract_output_tensor(result)
        if output_tensor is None:
            return None

        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.tensor(output_tensor)

        c_out = tiling.c_out
        h_out, w_out = tiling.tile.conv_out if tiling.tile else (0, 0)
        if c_out and h_out and w_out and output_tensor.numel() == (1 * c_out * h_out * w_out):
            output_tensor = output_tensor.reshape(1, c_out, h_out, w_out)

        self.tensor_cache[f"tile_{slice_idx}_{tile_idx}_out"] = output_tensor
        return output_tensor

    def collect_outputs(self, results_map: dict, num_tiles: int, slice_idx: int, tiling: TilingInfo) -> list[TileResult]:
        tile_exec_infos = []
        for tile_idx in range(num_tiles):
            result = results_map[tile_idx]
            if result['success']:
                output = self.process_result(result, tiling, slice_idx, tile_idx)
                if output is not None:
                    tile_exec_infos.append(TileResult(tile_idx=tile_idx, success=True, method=result.get('method', 'unknown')))
                else:
                    raise RuntimeError(f"Tile {tile_idx} produced no output tensor")
            else:
                raise RuntimeError(f"Tile {tile_idx} execution failed: {result.get('error', 'unknown')}")
        return tile_exec_infos

    def _resolve_path(self, p: str) -> str | None:
        from dsperse.src.run.utils.runner_utils import RunnerUtils
        return RunnerUtils.resolve_relative_path(p, self.slices_path)

    @staticmethod
    def _extract_output_tensor(result):
        return RunnerUtils.extract_output_tensor(result)

    @staticmethod
    def execute_worker(args: dict) -> dict:
        """Execute a single tile in a separate process. Must be static for pickling."""
        tile_idx = args['tile_idx']
        tile_in = args['tile_in']
        tile_out = args['tile_out']
        has_jst = args['has_jst']
        has_ezkl = args['has_ezkl']
        slices_path = Path(args['slices_path'])

        try:
            if has_jst:
                from dsperse.src.backends.jstprove import JSTprove
                from dsperse.src.run.utils.runner_utils import RunnerUtils
                jst_runner = JSTprove()
                circuit_path = RunnerUtils.resolve_relative_path(args['jstprove_circuit_path'], slices_path)
                success, result = jst_runner.generate_witness(tile_in, circuit_path, tile_out)
                if success:
                    output_tensor = TileExecutor._extract_output_tensor(result)
                    if output_tensor is not None:
                        return {
                            'tile_idx': tile_idx,
                            'success': True,
                            'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                            'method': ExecutionMethod.JSTPROVE_GEN_WITNESS
                        }
                return {'tile_idx': tile_idx, 'success': False, 'error': f'JSTprove failed: {result}'}

            elif has_ezkl:
                from dsperse.src.backends.ezkl import EZKL
                from dsperse.src.run.utils.runner_utils import RunnerUtils
                ezkl_runner = EZKL()
                circuit_path = RunnerUtils.resolve_relative_path(args['ezkl_circuit_path'], slices_path)
                vk_path = RunnerUtils.resolve_relative_path(args.get('ezkl_vk_path') or args.get('vk_path'), slices_path)
                settings_path = RunnerUtils.resolve_relative_path(args.get('ezkl_settings_path') or args.get('settings_path'), slices_path)
                success, result = ezkl_runner.generate_witness(tile_in, circuit_path, tile_out, vk_path, settings_path)
                if success:
                    output_tensor = TileExecutor._extract_output_tensor(result)
                    if output_tensor is not None:
                        return {
                            'tile_idx': tile_idx,
                            'success': True,
                            'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                            'method': ExecutionMethod.EZKL_GEN_WITNESS
                        }
                return {'tile_idx': tile_idx, 'success': False, 'error': f'EZKL failed: {result}'}

            tile_onnx_path = args.get('tile_onnx_path')
            if tile_onnx_path:
                from dsperse.src.backends.onnx_models import OnnxModels
                success, result = OnnxModels.run_inference(tile_in, tile_onnx_path, tile_out)
                if success:
                    output_tensor = TileExecutor._extract_output_tensor(result)
                    return {
                        'tile_idx': tile_idx,
                        'success': True,
                        'output_tensor': output_tensor.tolist() if hasattr(output_tensor, 'tolist') else output_tensor,
                        'method': ExecutionMethod.ONNX_ONLY
                    }
                return {'tile_idx': tile_idx, 'success': False, 'error': f'ONNX failed: {result}'}

            return {'tile_idx': tile_idx, 'success': False, 'error': 'No backend available'}

        except Exception as e:
            return {'tile_idx': tile_idx, 'success': False, 'error': str(e)}
