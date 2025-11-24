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
from dsperse.src.backends.onnx_models import OnnxModels
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, slice_path: str, run_metadata_path: str = None, save_metadata_path: str = None):
        self.slices_path = Path(slice_path)

        if not slice_path or not Path(slice_path).exists():
            raise Exception("A path of a dslice file or dsperse file is required.")

        # Load or generate run metadata from the new RunnerAnalyzer only
        if not run_metadata_path:
            if save_metadata_path:
                save_path = Path(save_metadata_path)
            else:
                ts = time.strftime('%Y%m%d_%H%M%S')
                # Save metadata inside a unique run_id folder when no explicit path is provided
                base_dir = Path(slice_path).parent
                if base_dir.name == "slices":
                    base_dir = base_dir.parent
                save_path = base_dir / "run" / f"run_{ts}" / "metadata.json"
            self.run_metadata = RunnerAnalyzer.generate_run_metadata(slice_path, save_path)
        else:
            with open(run_metadata_path, 'r') as f:
                self.run_metadata = json.load(f)

        self.ezkl_runner = EZKL()


    def run(self, input_json_path, output_path: str = None) -> dict:
        """Run inference through the chain using run/metadata.json."""
        self.metadata = getattr(self, "run_metadata", {})
        m = self.metadata or {}
        slices_root, self._original_format, model_path = RunnerUtils.normalize_for_runtime(m, self.slices_path)
        self.slices_path = slices_root
        head, nodes = RunnerAnalyzer.get_execution_chain(m)
        run_dir = RunnerUtils.make_run_dir(m, output_path, model_path)

        current_slice_id = head
        current_tensor = Utils.read_input(input_json_path)
        slice_results = {}

        while current_slice_id:
            slice_dir = self.slices_path / current_slice_id
            _, in_file, out_file = RunnerUtils.prepare_slice_io(run_dir, current_slice_id)
            Utils.write_input(current_tensor, str(in_file))
            info = m["slices"][current_slice_id]
            ok, tensor, exec_info = RunnerUtils.execute_slice(self, nodes[current_slice_id], info, in_file, out_file, slice_dir)
            slice_results[current_slice_id] = exec_info
            if not ok:
                raise Exception(f"Inference failed for {current_slice_id}: {exec_info.get('error','unknown')}")
            current_tensor = RunnerUtils.filter_tensor(info, tensor)
            current_slice_id = nodes[current_slice_id].get("next")

        probs = F.softmax(current_tensor, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        results = {
            "prediction": pred,
            "probabilities": probs.tolist(),
            "tensor_shape": list(current_tensor.shape),
            "slice_results": slice_results,
        }

        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_inference_output(results, run_dir / "run_results.json")
        RunnerUtils.repackage_if_needed(self._original_format, self.slices_path)
        return results

    @staticmethod
    def run_onnx_slice(slice_info: dict, input_tensor_path, output_tensor_path, slice_dir: Path = None):
        """Run ONNX inference for a slice.
        Accepts `slice_info['path']` possibly as `slice_#/payload/...` or absolute; resolves under `slice_dir` when provided.
        """
        onnx_path = slice_info.get("path")

        # Resolve possibly relative path
        def _resolve_rel_path(p: str, base_dir: Path) -> str:
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
            return str((Path(base_dir) / p_str).resolve()) if base_dir else os.path.abspath(p_str)

        if onnx_path and not os.path.isabs(str(onnx_path)):
            onnx_path = _resolve_rel_path(onnx_path, slice_dir)

        start_time = time.time()
        success, result = OnnxModels.run_inference(model_path=onnx_path, input_file=input_tensor_path, output_file=output_tensor_path)

        end_time = time.time()
        exec_info = {'success': success, 'method': 'onnx_only', 'execution_time': end_time - start_time, 'output_tensor_path': str(output_tensor_path)}

        if success:
            exec_info['input_file'] = str(input_tensor_path.resolve())
            exec_info['output_file'] = str(output_tensor_path.resolve())

        return success, result, exec_info

    def _run_ezkl_slice(self, slice_info: dict, input_tensor_path, output_witness_path, slice_dir: Path = None):
        """Run EZKL inference for a slice with fallback to ONNX.
        Accepts paths possibly formatted as `slice_#/payload/...` or `payload/...` and resolves them
        under the provided `slice_dir` if necessary.
        """
        def _resolve_rel_path(p: str, base_dir: Path) -> str:
            if not p:
                return None
            p_str = str(p)
            if os.path.isabs(p_str):
                return p_str
            try:
                sd_name = os.path.basename(os.path.abspath(str(base_dir))) if base_dir else None
            except Exception:
                sd_name = None
            parts = p_str.split(os.sep)
            if sd_name and parts and parts[0] == sd_name:
                parts = parts[1:]
                p_str = os.path.join(*parts) if parts else ''
            if base_dir:
                return str((Path(base_dir) / p_str).resolve())
            return os.path.abspath(p_str)

        model_path = slice_info.get("circuit_path")
        vk_path = slice_info.get("vk_path")
        settings_path = slice_info.get("settings_path")

        # Resolve possibly relative paths
        if model_path and not os.path.isabs(str(model_path)):
            model_path = _resolve_rel_path(model_path, slice_dir)
        if vk_path and not os.path.isabs(str(vk_path)):
            vk_path = _resolve_rel_path(vk_path, slice_dir)
        if settings_path and not os.path.isabs(str(settings_path)):
            settings_path = _resolve_rel_path(settings_path, slice_dir)

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
    
    def _save_inference_output(self, results, output_path):
        """Save inference_output.json with execution details."""
        model_path = self.metadata.get("model_path", "unknown")
        slice_results = results.get("slice_results", {})
        
        # Count execution methods
        ezkl_complete = sum(1 for r in slice_results.values() 
                           if r.get("method") == "ezkl_gen_witness")
        total_slices = len(slice_results)
        
        # Build execution results
        execution_results = []
        for slice_id, exec_info in slice_results.items():
            # Create witness_execution object to nest execution data
            witness_execution = {
                "method": exec_info.get("method", "unknown"),
                "execution_time": exec_info.get("execution_time", 0),
                "attempted_ezkl": exec_info.get("attempted_ezkl", True),
                "success": exec_info.get("success", False),
                "input_file": exec_info.get("input_file", "unknown"),
                "output_file": exec_info.get("output_file", "unknown"),
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
        
        # Calculate security percentage
        security_percent = (ezkl_complete / total_slices * 100) if total_slices > 0 else 0
        
        # Build output structure
        inference_output = {
            "model_path": model_path,
            "prediction": results["prediction"],
            "probabilities": results["probabilities"],
            "execution_chain": {
                "total_slices": total_slices,
                "ezkl_witness_slices": ezkl_complete,
                "overall_security": f"{security_percent:.1f}%",
                "execution_results": execution_results
            },
            "performance_comparison": {
                "note": "Full ONNX vs verified chain comparison would require separate pure ONNX run"
            }
        }
        
        # Save to file
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(inference_output, f, indent=2)


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

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
    # slices_dir = os.path.join(slices_dir, "slice_0.dslice") # give a single slice to test
    input_json = os.path.join(abs_path, "input.json")
    # input_json = "/Volumes/SSD/Users/dan/Projects/dsperse/dsperse/models/net/run/run_20251120_113439/slice_3/input.json"
    run_metadata_path = None #os.path.join(abs_path, "run", "metadata.json") if os.path.exists(
       # os.path.join(abs_path, "run", "metadata.json")) else None

    saved_run_metadata_path = None #os.path.join(abs_path, "run", "metadata.json")

    print(f"saves run metadata to {saved_run_metadata_path}")

    # Initialize runner (auto-generates run metadata if needed)
    runner = Runner(slice_path=slices_dir, run_metadata_path=run_metadata_path, save_metadata_path=saved_run_metadata_path)

    # Run inference
    print(f"Running inference on model {base_paths[model_choice]}...")
    results = runner.run(input_json)

    # Display results
    print(f"\nPrediction: {results['prediction']}")
    print("Execution summary:")
    for slice_id, info in results["slice_results"].items():
        print(f"  {slice_id}: {info['method']}")
