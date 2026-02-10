import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

from dsperse.src.analyzers.schema import (
    Backend, ExecutionInfo, ExecutionMethod, RunMetadata, RunSliceMetadata, TileResult
)
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class RunnerUtils:

    @staticmethod
    def extract_output_tensor(result):
        """Extract output tensor from execution result, checking 'output', 'logits', then 'output_tensor' keys."""
        if result is None:
            return None
        if isinstance(result, dict):
            for key in ['output', 'logits', 'output_tensor']:
                if key in result and result[key] is not None:
                    return result[key]
            return None
        return result

    @staticmethod
    def prepare_slice_io(run_dir: Path, slice_id: str) -> tuple[Path, Path]:
        slice_run_dir = run_dir / slice_id
        slice_run_dir.mkdir(parents=True, exist_ok=True)
        in_file = slice_run_dir / "input.json"
        out_file = slice_run_dir / "output.json"
        return in_file, out_file

    @staticmethod
    def initialize_run_state(input_json_path, run_metadata: RunMetadata, head_id: str):
        """Prepare initial input tensor and tensor cache."""
        input_tensor = Utils.read_input(input_json_path)
        first_slice_meta = run_metadata.get_slice(head_id)
        first_slice_inputs = first_slice_meta.dependencies.filtered_inputs if first_slice_meta else []
        model_input_name = first_slice_inputs[0] if first_slice_inputs else "input"
        return input_tensor, {model_input_name: input_tensor}

    @staticmethod
    def finalize_run_results(run_metadata, input_tensor, final_tensor, slice_results, run_dir):
        """Aggregate results and save the final run summary."""
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
        RunnerUtils.save_run_results(run_metadata, results, run_dir / "run_results.json")
        return results

    @staticmethod
    def resolve_relative_path(p: str, base_dir: Path) -> Optional[str]:
        if not p:
            return None
        p_str = str(p)
        if os.path.isabs(p_str):
            return p_str

        sd_name = os.path.basename(os.path.abspath(str(base_dir))) if base_dir else None
        parts = p_str.split(os.sep)
        if sd_name and parts and parts[0] == sd_name:
            parts = parts[1:]
            p_str = os.path.join(*parts) if parts else ''

        if not base_dir:
            return os.path.abspath(p_str)

        path = (base_dir / p_str).resolve()
        if path.exists():
            return str(path)

        alt_path = (base_dir.parent / p_str).resolve()
        if alt_path.exists():
            return str(alt_path)

        return str(path)

    @staticmethod
    def prepare_jstprove_meta(meta: RunSliceMetadata) -> RunSliceMetadata:
        """Prepare metadata with JSTprove-specific circuit path and settings."""
        return replace(meta,
            circuit_path=meta.jstprove_circuit_path or meta.circuit_path,
            settings_path=meta.jstprove_settings_path or meta.settings_path
        )

    @staticmethod
    def prepare_ezkl_meta(meta: RunSliceMetadata) -> RunSliceMetadata:
        """Prepare metadata with EZKL-specific circuit path and keys."""
        return replace(meta,
            circuit_path=meta.ezkl_circuit_path or meta.circuit_path,
            settings_path=meta.ezkl_settings_path or meta.settings_path,
            vk_path=meta.ezkl_vk_path or meta.vk_path,
            pk_path=meta.ezkl_pk_path or meta.pk_path
        )

    @staticmethod
    def flatten_input_for_ezkl(in_file: Path) -> Path:
        """Create flattened rank-2 input file for EZKL, return path to it."""
        with open(in_file, 'r') as f:
            data = json.load(f)
        if "input_data" in data:
            tensor_data = data["input_data"]
        elif "input" in data:
            tensor_data = data["input"]
        elif isinstance(data, dict) and len(data) == 1:
            tensor_data = next(iter(data.values()))
        else:
            tensor_data = []
        tensor = torch.tensor(tensor_data)
        flattened = tensor.flatten().tolist()
        ezkl_file = in_file.parent / "input_ezkl.json"
        with open(ezkl_file, 'w') as f:
            json.dump({"input_data": [flattened]}, f)
        return ezkl_file

    @staticmethod
    def _get_file_path() -> str:
        """Get the parent directory path of the current file."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def preprocess_input(input_path: str, model_directory: str = None, save_reshape: bool = False) -> torch.Tensor:
        """Preprocess input data from JSON."""
        if os.path.isfile(input_path):
            with open(input_path, 'r') as f:
                input_data = json.load(f)
        else:
            input_path = os.path.join(RunnerUtils._get_file_path(), input_path)
            logger.warning(f"Input file not found. Trying relative path: {input_path}")
            with open(input_path, 'r') as f:
                input_data = json.load(f)

        if isinstance(input_data, dict):
            if 'input_data' in input_data:
                input_data = input_data['input_data']
            elif 'input' in input_data:
                input_data = input_data['input']
            elif len(input_data) == 1:
                input_data = next(iter(input_data.values()))

        if isinstance(input_data, list) and len(input_data) == 0:
            raise ValueError("Input data list is empty")

        if isinstance(input_data, list):
            if isinstance(input_data[0], list):
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                input_tensor = torch.tensor([input_data], dtype=torch.float32)
        else:
            raise ValueError("Expected input data to be a list or nested list")

        return input_tensor

    @staticmethod
    def process_final_output(torch_tensor):
        """Return raw output tensor. Model-specific post-processing is caller's responsibility."""
        return {"output": torch_tensor}

    @staticmethod
    def save_to_file_flattened(input_tensor: torch.Tensor, file_path: str):
        tensor_data = input_tensor.flatten().tolist()
        file_dir = os.path.dirname(file_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
        data = {"input_data": [tensor_data]}
        with open(file_path, 'w') as f:
            json.dump(data, f)


    @staticmethod
    def save_run_results(run_metadata: RunMetadata, results: dict, output_path: str):
        """Save the final inference output and execution metadata to a JSON file."""
        model_path = run_metadata.model_path or "unknown"
        slice_results = results.get("slice_results", {})

        def _get_method(r):
            return r.method if isinstance(r, ExecutionInfo) else r.get("method", "")

        def _get_tiles(r):
            if isinstance(r, ExecutionInfo):
                return r.tiles
            return r.get("tiles") or r.get("tile_exec_infos") or []

        def _count_tile_method(tiles, method_prefix):
            return sum(1 for t in tiles if ((t.method or "") if isinstance(t, TileResult) else (t.get("method") or "")).startswith(method_prefix))

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

        circuit_slices = ezkl_complete + jstprove_complete
        security_percent = (circuit_slices / total_slices * 100) if total_slices > 0 else 0

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

    @staticmethod
    def find_most_recent_run(base_dir: Path) -> Path | None:
        """Search a directory for the latest run subdirectory based on naming convention."""
        run_dirs = sorted(base_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
        return run_dirs[0] if run_dirs else None


    @staticmethod
    def try_resume_slice(run_dir: Path, slice_id: str) -> tuple[bool, torch.Tensor | None, ExecutionInfo | None]:
        """Attempts to load cached results for a slice if it was previously completed."""
        slice_run_dir = run_dir / slice_id
        output_file = slice_run_dir / "output.json"
        if not output_file.exists():
            output_file = slice_run_dir / "slice_output.json"

        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    cached_data = json.load(f)
                cached_tensor = torch.tensor(cached_data['output'])
                exec_info = ExecutionInfo(
                    method=cached_data.get('method', 'resumed'),
                    success=True
                )
                logger.info(f"[resume] {slice_id}: loaded cached output")
                return True, cached_tensor, exec_info
            except Exception as e:
                logger.warning(f"Failed to load cached output for {slice_id}: {e}")
        return False, None, None

    @staticmethod
    def prepare_slice_input(info: RunSliceMetadata, tensor_cache: dict, model_input_tensor: torch.Tensor, in_file: Path, skip_write: bool = False) -> torch.Tensor:
        filtered_inputs = [n for n in info.dependencies.filtered_inputs if n]
        input_name = filtered_inputs[0] if filtered_inputs else "input"
        current_tensor = tensor_cache.get(input_name)
        if current_tensor is None:
            raise ValueError(
                f"Input tensor '{input_name}' not found in tensor cache. "
                f"Available: {list(tensor_cache.keys())}"
            )
        if not skip_write:
            Utils.write_input(current_tensor, str(in_file), input_name)
        return current_tensor

    @staticmethod
    def process_inference_result(slice_id: str, info: RunSliceMetadata, ok: bool, result, exec_info, tensor_cache: dict) -> torch.Tensor | None:
        """Handles reshaping, tensor cache updates, and final tensor extraction from inference results."""
        final_tensor = None
        output_names = info.dependencies.output

        logger.info(f"[{slice_id}] ok={ok}, result type={type(result).__name__}")
        if not ok or result is None:
            return None

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
                        logger.info(f"[{slice_id}] Reshaped output to {target_shape}")
                    else:
                        logger.warning(f"[{slice_id}] Cannot reshape: got {out_tensor.numel()} elements, expected {expected_numel}")
                for oname in output_names:
                    tensor_cache[oname] = out_tensor
                final_tensor = out_tensor
        return final_tensor

    @staticmethod
    def save_intermediate_output(out_file: Path, tensor: torch.Tensor, exec_info: ExecutionInfo):
        """Saves intermediate inference result to a JSON file."""
        method = exec_info.method if isinstance(exec_info, ExecutionInfo) else exec_info.get('method', 'unknown')
        raw_output = None
        if out_file.exists():
            try:
                with open(out_file, 'r') as f:
                    existing = json.load(f)
                if isinstance(existing.get('output'), list):
                    raw_output = existing['output']
            except Exception:
                pass
        data = {'output': tensor.tolist(), 'method': str(method)}
        if raw_output is not None:
            data['raw_output'] = raw_output
        with open(out_file, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def resolve_inference_paths(meta: RunSliceMetadata, slice_dir: Path) -> dict:
        """Resolve all circuit-related paths for a slice."""
        return {
            'circuit': RunnerUtils.resolve_relative_path(meta.circuit_path, slice_dir),
            'vk': RunnerUtils.resolve_relative_path(meta.vk_path, slice_dir),
            'settings': RunnerUtils.resolve_relative_path(meta.settings_path, slice_dir)
        }
