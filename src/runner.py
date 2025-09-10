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
from src.backends.onnx_models import OnnxModels
from src.backends.ezkl import EZKL
from src.analyzers.runner_analyzer import RunnerAnalyzer
from src.utils.utils import Utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Runner:
    def __init__(self, model_path, slices_path=None, metadata_path=None, run_metadata_path=None):
        if not model_path:
            raise ValueError("Please provide a model_path (parent of slices dir) to initialize the runner.")
        self.model_path = model_path
        self.slices_path = Path(slices_path) if slices_path else Path(model_path) / "slices"
        self.metadata_path = Path(metadata_path) if metadata_path else self.slices_path / "metadata.json"
        self.run_metadata_path = Path(run_metadata_path) if run_metadata_path else None
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found at {self.metadata_path}. Please run slicing first.")

        if self.run_metadata_path is None or not self.run_metadata_path.exists():
            logger.info("run metadata not found. Generating...")
            print(f"Generating run metadata at {self.run_metadata_path}")
            runner_metadata = RunnerAnalyzer(self.model_path)
            self.run_metadata_path = runner_metadata.generate_metadata(save_path=self.run_metadata_path)
        
        with open(self.run_metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.ezkl_runner = EZKL()

    def run(self, input_json_path) -> dict:
        """Run inference through the chain of segments."""
        # input_tensor = Utils.read_input(str(input_json_path))
        execution_chain = self.metadata.get("execution_chain", {})
        current_slice_id = execution_chain.get("head")
        current_input = input_json_path
        slice_results = {}

        run_directory = self.metadata.get("run_directory")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        run_dir = Path(run_directory) / run_id

        current_tensor = Utils.read_input(current_input)

        # Chain execution
        while current_slice_id:
            slice_node = execution_chain["nodes"][current_slice_id]
            segment_dir = self.slices_path / current_slice_id
            segment_dir.mkdir(exist_ok=True)

            seg_run_dir = run_dir / current_slice_id
            seg_run_dir.mkdir(parents=True, exist_ok=True)

            # Write input for this segment
            input_file = seg_run_dir / "input.json"
            output_file = seg_run_dir / "output.json"
            Utils.write_input(current_tensor, str(input_file))

            current_slice_metadata = self.metadata["slices"][current_slice_id]
            # Execute segment based on circuit availability
            if slice_node.get("use_circuit"):
                try:
                    success, tensor, ezkl_exec_info = self._run_ezkl_segment(
                        current_slice_metadata, input_file, output_file
                    )
                    if success:
                        slice_results[current_slice_id] = ezkl_exec_info
                    else:
                        raise Exception("EZKL execution failed")
                except Exception as e:
                    ezkl_error = str(e)
                    success, tensor, onnx_exec_info = self._run_onnx_segment(current_slice_metadata, input_file,
                                                                             output_file)
                    # mark as fallback and that EZKL was attempted
                    onnx_exec_info["method"] = "ezkl_fallback_onnx"
                    onnx_exec_info["attempted_ezkl"] = True
                    onnx_exec_info["error"] = ezkl_error
                    slice_results[current_slice_id] = onnx_exec_info

                    if not success:
                        raise Exception(
                            "EzKL fallback to ONNX failed for segment: " + current_slice_id + " with error: " + onnx_exec_info.get(
                                "error", "Unknown error. Check logs for details."))
            else:
                success, tensor, execution_info = self._run_onnx_segment(current_slice_metadata, input_file,
                                                                         output_file)
                execution_info["attempted_ezkl"] = False
                slice_results[current_slice_id] = execution_info

                if not success:
                    raise Exception("ONNX inference failed for segment: " + current_slice_id)

            # filter tensor and make tensor next input.json file
            current_tensor = self._filter_tensor(current_slice_metadata, tensor)
            current_slice_id = slice_node.get("next")

        # Final processing
        probabilities = F.softmax(current_tensor, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

        results = {
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
            "tensor_shape": list(current_tensor.shape),
            "slice_results": slice_results
        }

        # Save inference output
        self._save_inference_output(results, run_dir / "run_result.json")

        return results

    @staticmethod
    def _run_onnx_segment(slice_info: dict, input_tensor_path, output_tensor_path):
        """Run ONNX inference for a segment."""
        onnx_path = slice_info.get("path")
        start_time = time.time()
        success, result =  OnnxModels.run_inference(model_path=onnx_path, input_file=input_tensor_path, output_file=output_tensor_path)

        end_time = time.time()
        exec_info = {'success': success, 'method': 'onnx_only', 'execution_time': end_time - start_time, 'output_tensor_path': str(output_tensor_path)}

        if success:
            exec_info['input_file'] = str(input_tensor_path.resolve())
            exec_info['output_file'] = str(output_tensor_path.resolve())

        return success, result, exec_info

    def _run_ezkl_segment(self, slice_info: dict, input_tensor_path, output_witness_path):
        """Run EZKL inference for a segment with fallback to ONNX."""
        model_path = slice_info.get("circuit_path")
        vk_path = slice_info.get("vk_path")
        start_time = time.time()
        # Attempt EZKL execution
        success, output_tensor = self.ezkl_runner.generate_witness(input_file=input_tensor_path, model_path=model_path, output_file=output_witness_path, vk_path=vk_path)

        end_time = time.time()
        exec_info = {'success': success, 'method': 'ezkl_gen_witness', 'execution_time': end_time - start_time, 'witness_path': str(output_witness_path), 'attempted_ezkl': True}

        if success:
            exec_info['input_file'] = str(input_tensor_path.resolve())
            exec_info['output_file'] = str(output_witness_path.resolve())
        else:
            # When EZKL fails, output_tensor contains the error string per EZKL.generate_witness
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
                "segment_id": slice_id,
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
        with open(output_path, 'w') as f:
            json.dump(inference_output, f, indent=2)


    @staticmethod
    def _filter_tensor(current_slice_metadata, tensor):
        # take the tensor object, and extract the output that is relevant to the next segment
        logits = tensor["logits"]
        probabilities = tensor["probabilities"]
        predicted_action = tensor["predicted_action"]

        # Check the shape using our new function
        expected_shape = current_slice_metadata["output_shape"]
        Runner.check_expected_shape(logits, expected_shape, tensor_name="logits")

        return logits

    @staticmethod
    def check_expected_shape(tensor, expected_shape_data, tensor_name="tensor"):
        """
        Check if the tensor shape matches the expected shape from metadata.

        Args:
            tensor: The PyTorch tensor to check
            expected_shape_data: The shape data from metadata (usually a nested list with possible string placeholders)
            tensor_name: Name of the tensor for logging purposes

        Returns:
            bool: True if shapes match, False otherwise
        """
        # Handle the case where output_shape is a nested list
        if isinstance(expected_shape_data, list) and len(expected_shape_data) > 0:
            # Extract the inner shape list - the first element of output_shape
            shape_values = expected_shape_data[0]

            # Replace string placeholders with actual values from tensor
            expected_elements = 1
            shape_dict = {
                "batch_size": tensor.shape[0] if tensor.dim() > 0 else 1,
                "unk__0": tensor.shape[0] if tensor.dim() > 0 else 1
            }

            # Build the expected shape with placeholders replaced
            expected_shape = []
            for dim in shape_values:
                if isinstance(dim, str):
                    if dim in shape_dict:
                        expected_shape.append(shape_dict[dim])
                        expected_elements *= shape_dict[dim]
                    else:
                        logger.warning(f"Unknown dimension placeholder: {dim}")
                        # Default to using 1 for unknown dimensions
                        expected_shape.append(1)
                        expected_elements *= 1
                else:
                    expected_shape.append(dim)
                    expected_elements *= dim

            # Check total elements
            tensor_elements = torch.numel(tensor)
            if tensor_elements != expected_elements:
                logger.warning(
                    f"{tensor_name} shape {list(tensor.shape)} has {tensor_elements} elements, "
                    f"but expected shape {expected_shape} has {expected_elements} elements"
                )
                return False

            # If the tensor is flattened but should be multidimensional
            if len(tensor.shape) == 1 and len(expected_shape) > 1:
                logger.info(
                    f"{tensor_name} is flattened ({tensor.shape[0]} elements), "
                    f"but expected shape is {expected_shape}"
                )
                return True

            # Check actual dimensions if tensor is not flattened
            if len(tensor.shape) == len(expected_shape):
                for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                    if actual != expected:
                        logger.warning(
                            f"Dimension mismatch at index {i}: {tensor_name} has size {actual}, "
                            f"expected {expected}"
                        )
                        return False
                return True

        # If we can't determine expected shape, just return True
        logger.debug(f"Could not determine precise expected shape for {tensor_name}")
        return True


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 5  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet",
        5: "models/age",
        6: "models/version"
    }

    # Get model directory
    abs_path = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(abs_path, "slices")
    input_json = os.path.join(abs_path, "input.json")
    run_metadata_path = os.path.join(abs_path, "run", "metadata.json") if os.path.exists(
        os.path.join(abs_path, "run", "metadata.json")) else None

    # Initialize runner (auto-generates run metadata if needed)
    runner = Runner(model_path=abs_path, slices_path=slices_dir, run_metadata_path=run_metadata_path)

    # Run inference
    print(f"Running inference on model {base_paths[model_choice]}...")
    results = runner.run(input_json)

    # Display results
    print(f"\nPrediction: {results['prediction']}")
    print("Execution summary:")
    for slice_id, info in results["slice_results"].items():
        print(f"  {slice_id}: {info['method']}")
