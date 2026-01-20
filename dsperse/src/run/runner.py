"""
Runner for EzKL Circuit and ONNX Inference
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.backends.onnx_models import OnnxModels
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, run_metadata_path: str = None, save_metadata_path: str = None):
        """Initialize the Runner.

        We keep run_metadata_path and save_metadata_path at instantiation as requested.
        """
        self._provided_run_metadata_path = run_metadata_path
        self._save_metadata_path = save_metadata_path
        self.run_metadata = None
        # Expose the last run directory to callers (e.g., CLI) for user messaging
        self.last_run_dir: Path | None = None
        # Optional: force a specific backend at runtime ('jstprove' | 'ezkl' | 'onnx')
        self.force_backend: str | None = None

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

        self.tiled_info = None


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


    def _run_ezkl_slice(self, slice_info: dict, input_tensor_path, output_witness_path, slice_dir: Path = None):
        """Run EZKL inference for a slice with fallback to ONNX.
        Accepts paths possibly formatted as `slice_#/payload/...` or `payload/...` and resolves them
        under the provided `slice_dir` if necessary.
        """
        if self.ezkl_runner is None:
            return False, "EZKL CLI not available", {
                'success': False,
                'method': 'ezkl_gen_witness',
                'error': 'EZKL CLI not available',
                'attempted_ezkl': True
            }

        model_path = slice_info.get("circuit_path")
        vk_path = slice_info.get("vk_path")
        settings_path = slice_info.get("settings_path")

        # Resolve possibly relative paths
        if model_path and not os.path.isabs(str(model_path)):
            model_path = RunnerUtils.resolve_relative_path(model_path, slice_dir)
        if vk_path and not os.path.isabs(str(vk_path)):
            vk_path = RunnerUtils.resolve_relative_path(vk_path, slice_dir)
        if settings_path and not os.path.isabs(str(settings_path)):
            settings_path = RunnerUtils.resolve_relative_path(settings_path, slice_dir)

        start_time = time.time()
        # Attempt EZKL execution, but ensure we catch any exceptions to allow fallback
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

        end_time = time.time()
        exec_info = {
            'success': success,
            'method': 'ezkl_gen_witness',
            'execution_time': end_time - start_time,
            'witness_path': str(output_witness_path),
            'attempted_ezkl': True
        }

        if success:
            exec_info['input_file'] = str(input_tensor_path.resolve())
            exec_info['output_file'] = str(output_witness_path.resolve())
        else:
            # When EZKL fails, output_tensor contains the error string or exception message
            exec_info['error'] = output_tensor if isinstance(output_tensor, str) else "Unknown EZKL error"

        return success, output_tensor, exec_info

    def _run_jstprove_slice(self, slice_info: dict, input_tensor_path, output_witness_path, slice_dir: Path = None):
        """Run JSTprove inference for a slice with fallback to ONNX.
        Accepts paths possibly formatted as `slice_#/payload/...` or `payload/...` and resolves them
        under the provided `slice_dir` if necessary.
        """
        if self.jstprove_runner is None:
            return False, "JSTprove CLI not available", {'success': False, 'method': 'jstprove_gen_witness', 'error': 'JSTprove CLI not available'}

        circuit_path = slice_info.get("circuit_path")
        settings_path = slice_info.get("settings_path")

        # Resolve possibly relative paths
        if circuit_path and not os.path.isabs(str(circuit_path)):
            circuit_path = RunnerUtils.resolve_relative_path(circuit_path, slice_dir)

        start_time = time.time()
        # Attempt JSTprove execution
        try:
            # Record the actual JSTprove witness file name produced by the backend
            # Convention: runner writes outputs to <run_dir>/slice_#/output.json and JSTprove
            # emits a binary witness next to it named 'output_witness.bin'.
            witness_file_path = Path(output_witness_path).with_name("output_witness.bin")
            success, output_tensor = self.jstprove_runner.generate_witness(
                input_file=input_tensor_path,
                model_path=circuit_path,
                output_file=output_witness_path,
            )
        except Exception as e:
            success = False
            output_tensor = str(e)

        end_time = time.time()
        exec_info = {
            'success': success,
            'method': 'jstprove_gen_witness',
            'execution_time': end_time - start_time,
            'witness_path': str(output_witness_path),
            'witness_file': str(witness_file_path),
            'attempted_jstprove': True
        }

        if success:
            exec_info['input_file'] = str(input_tensor_path.resolve())
            exec_info['output_file'] = str(output_witness_path.resolve())
        else:
            exec_info['error'] = output_tensor if isinstance(output_tensor, str) else "Unknown JSTprove error"

        return success, output_tensor, exec_info

    def _save_inference_output(self, results, output_path):
        """Save inference_output.json with execution details."""
        model_path = self.run_metadata.get("model_path", "unknown")
        slice_results = results.get("slice_results", {})

        # Count execution methods
        ezkl_complete = sum(
            1 for r in slice_results.values()
            if r.get("method") == "ezkl_gen_witness"
        )
        jstprove_complete = sum(
            1 for r in slice_results.values()
            if r.get("method") == "jstprove_gen_witness"
        )
        total_slices = len(slice_results)

        # Build execution results
        execution_results = []
        for slice_id, exec_info in slice_results.items():
            # Create witness_execution object to nest execution data
            witness_execution = {
                "method": exec_info.get("method", "unknown"),
                "execution_time": exec_info.get("execution_time", 0),
                "attempted_ezkl": exec_info.get("attempted_ezkl", False),
                "attempted_jstprove": exec_info.get("attempted_jstprove", False),
                "success": exec_info.get("success", False),
                "input_file": exec_info.get("input_file", "unknown"),
                "output_file": exec_info.get("output_file", "unknown"),
                "tile_exec_infos": exec_info.get("tile_exec_infos", []),
            }
            # Propagate error message if present (e.g., EZKL failure reason before fallback)
            if "error" in exec_info and exec_info["error"]:
                witness_execution["error"] = exec_info["error"]

            # Create result_entry with segment_id and witness_execution
            result_entry = {
                "slice_id": slice_id,
                "witness_execution": witness_execution
            }

            execution_results.append(result_entry)

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
                self.run_metadata = json.load(f)
        else:
            if self._save_metadata_path:
                save_path = Path(self._save_metadata_path)
            else:
                ts = time.strftime('%Y%m%d_%H%M%S')
                base_dir = self.slices_path.parent
                if base_dir.name == "slices":
                    base_dir = base_dir.parent
                save_path = base_dir / "run" / f"run_{ts}" / "metadata.json"
            self.run_metadata = RunnerAnalyzer.generate_run_metadata(self.slices_path, save_path, format)

    def _run_tiled_slice(self, slice_id: str, slice_info: dict, tensor_cache: dict, run_dir: Path) -> dict:
        """Run a tiled slice: split (Python) → tiles (ONNX) → concat (Python)."""
        tiling = slice_info.get("tiling")
        if not tiling:
            return {'success': False, 'error': 'missing_tiling_info'}

        input_tensor = self._get_tiled_input_tensor(slice_id, tiling, slice_info, tensor_cache)

        split_time = self._run_tiling_split(slice_id, tiling, input_tensor, run_dir, tensor_cache)
        tile_time, tile_exec_infos = self._run_tiling_parallel_tiles(slice_id, tiling, slice_info, run_dir, tensor_cache)
        concat_time = self._run_tiling_concat(slice_id, tiling, run_dir, tensor_cache)

        total_time = split_time + tile_time + concat_time

        return {
            'success': True,
            'method': 'tiled',
            'execution_time': total_time,
            'num_tiles': tiling["num_tiles"],
            'split_time': split_time,
            'tile_time': tile_time,
            'concat_time': concat_time,
            'tile_exec_infos': tile_exec_infos,
            'tiling': tiling,
        }

    def _get_tiled_input_tensor(self, slice_id: str, tiling: dict, slice_info: dict, tensor_cache: dict) -> torch.Tensor:
        input_name = tiling.get("input_name") or slice_info.get("dependencies", {}).get("filtered_inputs", ["input"])[0]
        input_tensor = tensor_cache.get(input_name)
        if input_tensor is None:
            model_input_name = self.run_metadata.get("input_shape", [{}])[0].get("name", "images")
            input_tensor = tensor_cache.get(model_input_name)

        if input_tensor is None:
            raise ValueError(f"Missing input tensor '{input_name}' for tiled slice {slice_id}")
        return input_tensor

    def _run_tiling_split(self, slice_id: str, tiling: dict, input_tensor: torch.Tensor, run_dir: Path, tensor_cache: dict) -> float:
        """Pure Python split: pad input and extract tiles."""
        start_time = time.time()

        slice_idx = tiling.get("slice_idx", 0)
        tile_size = tiling["tile_size"]
        halo_h, halo_w = tiling["halo"]
        tiles_y = tiling["tiles_y"]
        tiles_x = tiling["tiles_x"]
        num_tiles = tiling["num_tiles"]

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

    def _run_single_tile(self, tile_idx: int, slice_id: str, slice_info: dict, tiling: dict, tile_onnx_path: str, run_dir: Path, tensor_cache: dict, cache_lock: threading.Lock):
        slice_idx = tiling.get("slice_idx", 0)
        cache_input_name = f"tile_{slice_idx}_{tile_idx}_in"
        cache_output_name = f"tile_{slice_idx}_{tile_idx}_out"

        tile_run_dir = run_dir / slice_id / f"tile_{tile_idx}"
        tile_run_dir.mkdir(parents=True, exist_ok=True)
        tile_in = tile_run_dir / "input.json"
        tile_out = tile_run_dir / "output.json"

        tile_tensor = tensor_cache.get(cache_input_name)
        if tile_tensor is None:
            raise ValueError(f"Missing tile input tensor '{cache_input_name}' for slice {slice_id}")
        Utils.write_input(tile_tensor, str(tile_in))

        tile_slice_info = dict(slice_info)
        tile_slice_info["path"] = tile_onnx_path
        tile_node_info = {"use_circuit": True}

        ok, result, t_info = RunnerUtils.execute_slice(
            self, tile_node_info, tile_slice_info, tile_in, tile_out, self.slices_path
        )

        output_tensor = None
        if ok and result is not None:
            if isinstance(result, dict) and 'output_tensors' in result:
                output_tensor = result['output_tensors'].get("tile_out")
                if output_tensor is None and result['output_tensors']:
                    output_tensor = list(result['output_tensors'].values())[0]
            else:
                output_tensor = result.get('logits', result) if isinstance(result, dict) else result

            if isinstance(output_tensor, torch.Tensor) and output_tensor.dim() < 4:
                c_out = tiling.get("c_out")
                h_out, w_out = tiling["tile"].get("conv_out", [0, 0])
                if c_out and h_out and w_out:
                    expected_numel = 1 * c_out * h_out * w_out
                    if output_tensor.numel() == expected_numel:
                        output_tensor = output_tensor.reshape(1, c_out, h_out, w_out)
                        logger.debug(f"Reshaped tile {tile_idx} output to [1, {c_out}, {h_out}, {w_out}]")

        if ok and output_tensor is not None:
            with cache_lock:
                tensor_cache[cache_output_name] = output_tensor

        return tile_idx, cache_output_name, ok, output_tensor, t_info

    def _run_tiling_parallel_tiles(self, slice_id: str, tiling: dict, slice_info: dict, run_dir: Path, tensor_cache: dict) -> tuple[float, list]:
        import onnxruntime as ort
        import numpy as np

        num_tiles = tiling["num_tiles"]
        tile_info = tiling["tile"]
        tile_onnx_path = RunnerUtils.resolve_relative_path(tile_info["path"], self.slices_path)
        slice_idx = tiling.get("slice_idx", 0)

        tile_start = time.time()

        # Create ONE session for all tiles (major optimization)
        session = ort.InferenceSession(str(tile_onnx_path))
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        tile_exec_infos = []
        for tile_idx in range(num_tiles):
            cache_input_name = f"tile_{slice_idx}_{tile_idx}_in"
            cache_output_name = f"tile_{slice_idx}_{tile_idx}_out"

            tile_tensor = tensor_cache.get(cache_input_name)
            if tile_tensor is None:
                raise ValueError(f"Missing tile input tensor '{cache_input_name}' for slice {slice_id}")

            # Convert to numpy for ONNX
            if isinstance(tile_tensor, torch.Tensor):
                input_arr = tile_tensor.detach().cpu().numpy().astype(np.float32)
            else:
                input_arr = np.asarray(tile_tensor, dtype=np.float32)

            # Run inference
            outputs = session.run(None, {input_name: input_arr})
            output_arr = outputs[0]

            # Convert back to tensor and cache
            output_tensor = torch.from_numpy(output_arr)
            tensor_cache[cache_output_name] = output_tensor

            tile_exec_infos.append({'tile_idx': tile_idx, 'success': True})

        tile_time = time.time() - tile_start
        logger.info(f"All {num_tiles} tiles completed in {tile_time:.2f}s ({tile_time / num_tiles * 1000:.1f}ms avg)")
        return tile_time, tile_exec_infos

    def _run_tiling_concat(self, slice_id: str, tiling: dict, run_dir: Path, tensor_cache: dict) -> float:
        """Pure Python concat: reassemble tiles into full output."""
        concat_start = time.time()

        slice_idx = tiling.get("slice_idx", 0)
        tiles_y = tiling["tiles_y"]
        tiles_x = tiling["tiles_x"]
        output_name = tiling["output_name"]

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


    def _run(self, output_path=None, input_json_path=None):
        head, nodes = RunnerAnalyzer.get_execution_chain(self.run_metadata)
        run_dir = RunnerUtils.make_run_dir(self.run_metadata, output_path, self.slices_path)
        self.last_run_dir = run_dir

        input_tensor = Utils.read_input(input_json_path)
        model_input_name = self.run_metadata.get("input_shape", [{}])[0].get("name", "images")
        tensor_cache = {model_input_name: input_tensor}

        current_slice_id = head
        slice_results = {}
        final_tensor = None

        while current_slice_id:
            info = self.run_metadata["slices"][current_slice_id]
            slice_dir = self.slices_path

            # Check if this slice is tiled
            tiling = info.get("tiling")

            if tiling:
                logger.info(f"Running tiled slice {current_slice_id} with parallel tiles")
                exec_info = self._run_tiled_slice(current_slice_id, info, tensor_cache, run_dir)
                ok = exec_info.get('success', False)
                output_names = info.get("dependencies", {}).get("output", [])
                if ok and output_names:
                    final_tensor = tensor_cache.get(output_names[0])
            else:
                filtered_inputs = [n for n in info.get("dependencies", {}).get("filtered_inputs", []) if n]
                output_names = info.get("dependencies", {}).get("output", [])
                node = nodes[current_slice_id]

                input_name = filtered_inputs[0] if filtered_inputs else "input"
                current_tensor = tensor_cache.get(input_name, input_tensor)

                use_circuit = node.get("use_circuit") and self.force_backend != 'onnx'

                if use_circuit:
                    in_file, out_file = RunnerUtils.prepare_slice_io(run_dir, current_slice_id)
                    Utils.write_input(current_tensor, str(in_file))
                    ok, result, exec_info = RunnerUtils.execute_slice(self, node, info, in_file, out_file, slice_dir)
                else:
                    if len(filtered_inputs) > 1:
                        missing = [n for n in filtered_inputs if n not in tensor_cache]
                        if missing:
                            raise ValueError(f"Missing input tensors for {current_slice_id}: {missing}")
                        extra_tensors = {name: tensor_cache[name] for name in filtered_inputs}
                    else:
                        extra_tensors = {input_name: current_tensor}
                    ok, result, exec_info = RunnerUtils.run_onnx_multi_input_slice(info, None, slice_dir, extra_tensors)

                if ok and result:
                    if isinstance(result, dict) and 'output_tensors' in result:
                        for oname, tensor in result['output_tensors'].items():
                            tensor_cache[oname] = tensor
                        first_output = list(result['output_tensors'].values())[0] if result['output_tensors'] else None
                        final_tensor = first_output if first_output is not None else result.get('logits')
                    else:
                        out_tensor = result.get('logits', result) if isinstance(result, dict) else result
                        if isinstance(out_tensor, torch.Tensor):
                            expected_shape = info.get("output_shape")
                            if expected_shape and len(expected_shape) > 0:
                                target_shape = [d if isinstance(d, int) else 1 for d in expected_shape[0]]
                                if out_tensor.numel() == torch.prod(torch.tensor(target_shape)).item():
                                    out_tensor = out_tensor.reshape(target_shape)
                            for oname in output_names:
                                tensor_cache[oname] = out_tensor
                            final_tensor = out_tensor

            slice_results[current_slice_id] = exec_info
            if not ok:
                raise Exception(f"Inference failed for {current_slice_id}: {exec_info.get('error', 'unknown')}")
            current_slice_id = nodes[current_slice_id].get("next")

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
    print(f"\nPrediction: {results['prediction']}")
    print("Execution summary:")
    for slice_id, info in results["slice_results"].items():
        print(f"  {slice_id}: {info['method']}")
