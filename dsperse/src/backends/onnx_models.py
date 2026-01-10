import logging
import os
import re

import numpy as np
import onnxruntime as ort
import torch

from dsperse.src.run.utils.runner_utils import RunnerUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OnnxModels:
    def __init__(self):
        self.device = torch.device("cpu")

    @staticmethod
    def _create_session(model_path: str) -> ort.InferenceSession:
        """Create ONNX Runtime session with deterministic settings for ZK proofs."""
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.enable_profiling = False
        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_reuse = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(model_path, opts)

    @staticmethod
    def _process_outputs(session: ort.InferenceSession, raw_output: list, output_file: str) -> dict:
        """Convert raw ONNX outputs to result dict with tensors."""
        model_outputs = session.get_outputs()
        if not model_outputs:
            raise ValueError("Model has no outputs")

        output_tensors = {}
        for i, out in enumerate(model_outputs):
            output_tensors[out.name] = torch.from_numpy(raw_output[i])

        first_output_name = model_outputs[0].name
        output_tensor = output_tensors[first_output_name].float()
        result = RunnerUtils.process_final_output(output_tensor)
        result['output_tensors'] = output_tensors
        RunnerUtils.save_to_file_flattened(result['logits'], output_file)
        return result

    @staticmethod
    def run_inference(input_file: str, model_path: str, output_file: str):
        """Run inference with the ONNX model and return the logits, probabilities, and predictions."""
        try:
            session = OnnxModels._create_session(model_path)
            input_tensor = RunnerUtils.preprocess_input(input_file)
            input_dict = OnnxModels.apply_onnx_shape(model_path, input_tensor)
            raw_output = session.run(None, input_dict)
            return True, OnnxModels._process_outputs(session, raw_output, output_file)
        except Exception as e:
            logger.exception("Error during inference")
            return False, str(e)

    @staticmethod
    def _parse_onnx_type(type_str: str) -> np.dtype:
        """Parse onnxruntime type string like 'tensor(float)' to numpy dtype."""
        type_map = {
            'float': np.float32, 'float16': np.float16, 'double': np.float64,
            'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
            'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
            'bool': np.bool_,
        }
        match = re.search(r'tensor\((\w+)\)', type_str)
        if match:
            return type_map.get(match.group(1), np.float32)
        return np.float32

    @staticmethod
    def run_inference_multi(model_path: str, extra_tensors: dict, output_file: str):
        """Run inference with multiple named inputs. All inputs must be provided in extra_tensors."""
        try:
            session = OnnxModels._create_session(model_path)
            input_dict = {}

            for model_input in session.get_inputs():
                name = model_input.name
                if name not in extra_tensors or extra_tensors[name] is None:
                    raise ValueError(f"Missing required input tensor: {name}")

                t = extra_tensors[name]
                arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
                arr = arr.astype(OnnxModels._parse_onnx_type(model_input.type))

                expected_rank = len(model_input.shape)
                if arr.ndim != expected_rank:
                    raise ValueError(f"Rank mismatch for {name}: got {arr.ndim}D, expected {expected_rank}D")

                for i, (got, expected) in enumerate(zip(arr.shape, model_input.shape)):
                    if isinstance(expected, int) and got != expected:
                        raise ValueError(f"Dim {i} mismatch for {name}: got {got}, expected {expected}")

                input_dict[name] = arr

            raw_output = session.run(None, input_dict)
            return True, OnnxModels._process_outputs(session, raw_output, output_file)
        except Exception as e:
            logger.exception("Error during multi-input inference")
            return False, str(e)

    @staticmethod
    def apply_onnx_shape(model_path, input_tensor, is_numpy=False):
        """
        Reshapes the input tensor to match the expected input shape of the ONNX model.

        Args:
            model_path: Path to the ONNX model
            input_tensor: Input tensor (can be a PyTorch tensor or NumPy array)
            is_numpy: Boolean indicating if input_tensor is already a NumPy array

        Returns:
            Dictionary mapping input names to properly shaped tensors
        """
        try:
            session = OnnxModels._create_session(model_path)
            model_inputs = session.get_inputs()
            logger.info(f"Model expects {len(model_inputs)} input(s)")

            # Convert input to numpy if it's not already
            if not is_numpy:
                if isinstance(input_tensor, torch.Tensor):
                    input_numpy = input_tensor.numpy().astype(np.float32)
                else:
                    input_numpy = np.array(input_tensor, dtype=np.float32)
            else:
                input_numpy = input_tensor.astype(np.float32)

            # Handle multiple inputs
            if len(model_inputs) > 1:
                # If we have a flattened tensor, we need to split it for each input
                result = {}
                total_elements_used = 0

                for i, model_input in enumerate(model_inputs):
                    input_name = model_input.name
                    input_shape = model_input.shape
                    logger.info(f"Input {i + 1}: {input_name} with shape {input_shape}")

                    # Calculate number of elements needed for this input
                    elements_needed = 1
                    final_shape = []

                    for dim in input_shape:
                        if isinstance(dim, int):
                            elements_needed *= dim
                            final_shape.append(dim)
                        elif dim == 'batch_size' or dim.startswith('unk'):
                            batch_size = 1  # Default batch size
                            elements_needed *= batch_size
                            final_shape.append(batch_size)
                        else:
                            # For any other symbolic dimension, default to 1
                            elements_needed *= 1
                            final_shape.append(1)

                    # Extract the portion of the flattened tensor for this input
                    if input_numpy.size > total_elements_used + elements_needed:
                        input_portion = input_numpy.flatten()[total_elements_used:total_elements_used + elements_needed]
                        total_elements_used += elements_needed
                    else:
                        # If we don't have enough elements, use what's left
                        input_portion = input_numpy.flatten()[total_elements_used:]
                        logger.warning(
                            f"Not enough elements for input {input_name}. Expected {elements_needed}, got {input_portion.size}")

                        # Pad with zeros if necessary
                        if input_portion.size < elements_needed:
                            padding = np.zeros(elements_needed - input_portion.size, dtype=np.float32)
                            input_portion = np.concatenate([input_portion, padding])

                    # Reshape to match expected shape
                    reshaped = input_portion.reshape(final_shape)
                    # Ensure float32 for ORT compatibility
                    if reshaped.dtype != np.float32:
                        reshaped = reshaped.astype(np.float32)
                    result[input_name] = reshaped

                return result
            else:
                # Single input case
                input_name = model_inputs[0].name
                input_shape = model_inputs[0].shape
                logger.info(f"Single input: {input_name} with shape {input_shape}")

                # Check if we need to reshape
                if len(input_numpy.shape) != len(input_shape):
                    # Determine the appropriate shape
                    final_shape = []
                    for dim in input_shape:
                        if isinstance(dim, int):
                            final_shape.append(dim)
                        elif dim == 'batch_size' or dim.startswith('unk'):
                            final_shape.append(1)  # Default batch size to 1
                        else:
                            # For any other symbolic dimension, default to 1
                            final_shape.append(1)

                    # Calculate total elements needed
                    elements_needed = np.prod(final_shape)

                    # Check if we have enough elements
                    if input_numpy.size < elements_needed:
                        logger.warning(f"Not enough elements. Expected {elements_needed}, got {input_numpy.size}")

                        # Pad with zeros if necessary
                        flat = input_numpy.flatten()
                        padding = np.zeros(elements_needed - flat.size, dtype=np.float32)
                        input_numpy = np.concatenate([flat, padding])

                    # Reshape the input
                    input_numpy = input_numpy.reshape(final_shape)
                    logger.info(f"Reshaped input to {input_numpy.shape}")
                elif not np.array_equal(input_numpy.shape,
                                        [int(dim) if isinstance(dim, int) else 1 for dim in input_shape]):
                    # If dimensions don't match (after replacing symbolic dims with 1)
                    expected_shape = [int(dim) if isinstance(dim, int) else 1 for dim in input_shape]

                    # Check if total elements match
                    elements_needed = np.prod(expected_shape)
                    if input_numpy.size == elements_needed:
                        # If same number of elements, just reshape
                        input_numpy = input_numpy.reshape(expected_shape)
                        logger.info(f"Reshaped input from {input_numpy.shape} to {expected_shape}")
                    else:
                        # Try to use what we have
                        logger.warning(f"Input shape {input_numpy.shape} doesn't match expected shape {expected_shape}")

                        # Flatten and reshape, padding if necessary
                        flat = input_numpy.flatten()
                        if flat.size < elements_needed:
                            padding = np.zeros(elements_needed - flat.size, dtype=np.float32)
                            flat = np.concatenate([flat, padding])
                        elif flat.size > elements_needed:
                            flat = flat[:elements_needed]

                        input_numpy = flat.reshape(expected_shape)

                # Ensure float32 for ORT compatibility
                if input_numpy.dtype != np.float32:
                    input_numpy = input_numpy.astype(np.float32)
                return {input_name: input_numpy}

        except Exception as e:
            logger.error(f"Error in apply_onnx_shape: {e}")
            # In case of error, return the original tensor with the first input name
            if len(session.get_inputs()) > 0:
                return {session.get_inputs()[0].name: input_numpy}
            else:
                return {"input": input_numpy}


# Example usage
if __name__ == "__main__":

    # Choose which model to test
    model_choice = 1  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet"
    }

    # Get model directory
    abs_path = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(abs_path, "slices")
    input_json = os.path.join(abs_path, "input.json")
    output_json = os.path.join(abs_path, "output.json")

    print(f"Running inference on {abs_path}")

    result = OnnxModels.run_inference(input_file=input_json, model_path=os.path.join(abs_path, "model.onnx"), output_file="output.json")

    print(result)
