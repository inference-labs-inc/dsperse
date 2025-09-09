import logging
import os

import numpy as np
import onnxruntime as ort
import torch

from src.utils.runner_utils.runner_utils import RunnerUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OnnxModels:
    def __init__(self):
        self.device = torch.device("cpu")

    @staticmethod
    def run_inference(input_file: str, model_path: str, output_file: str):
        """
        Run inference with the ONNX model and return the logits, probabilities, and predictions.
        """
        try:
            # Create an ONNX Runtime session
            session = ort.InferenceSession(model_path)

            # Convert PyTorch tensor to numpy array for ONNX Runtime
            input_tensor = RunnerUtils.preprocess_input(input_file)

            # Apply proper shaping based on the ONNX model's expected input
            input_dict = OnnxModels.apply_onnx_shape(model_path, input_tensor)

            # Run inference
            raw_output = session.run(None, input_dict)

            # Convert the output back to a PyTorch tensor
            output_tensor = torch.tensor(raw_output[0])

            # Process the output
            result = RunnerUtils.process_final_output(output_tensor)

            RunnerUtils.save_to_file_flattened(result['logits'], output_file)

            return True, result

        except Exception as e:
            logger.warning(f"Error during inference: {e}")
            return False, None

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
            # Create an ONNX Runtime session to get model metadata
            session = ort.InferenceSession(model_path)

            # Get input details from the model
            model_inputs = session.get_inputs()
            logger.info(f"Model expects {len(model_inputs)} input(s)")

            # Convert input to numpy if it's not already, enforce float32
            if not is_numpy:
                if isinstance(input_tensor, torch.Tensor):
                    input_numpy = input_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    input_numpy = np.array(input_tensor, dtype=np.float32)
            else:
                input_numpy = np.asarray(input_tensor, dtype=np.float32)

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

                    # Extract the portion of the flattened tensor for this input using correct bounds
                    flat = input_numpy.flatten()
                    end_idx = total_elements_used + elements_needed
                    if flat.size >= end_idx:
                        input_portion = flat[total_elements_used:end_idx]
                        total_elements_used = end_idx
                    else:
                        # Use what's left and pad if needed
                        input_portion = flat[total_elements_used:]
                        got = input_portion.size
                        if got < elements_needed:
                            logger.warning(f"Not enough elements for input {input_name}. Expected {elements_needed}, got {got}")
                            padding = np.zeros(elements_needed - got, dtype=np.float32)
                            input_portion = np.concatenate([input_portion, padding]).astype(np.float32, copy=False)
                    
                    # Ensure dtype float32 and reshape to match expected shape
                    input_portion = np.asarray(input_portion, dtype=np.float32).reshape(final_shape)
                    result[input_name] = input_portion

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
                        input_numpy = np.concatenate([flat, padding]).astype(np.float32, copy=False)

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
                            flat = np.concatenate([flat, padding]).astype(np.float32, copy=False)
                        elif flat.size > elements_needed:
                            flat = flat[:elements_needed]

                        input_numpy = flat.reshape(expected_shape)

                # Ensure float32 before returning
                input_numpy = np.asarray(input_numpy, dtype=np.float32)
                return {input_name: input_numpy}

        except Exception as e:
            logger.error(f"Error in apply_onnx_shape: {e}")
            # In case of error, return the original tensor with the first input name, enforce float32
            safe_np = np.asarray(input_numpy, dtype=np.float32)
            try:
                inputs = session.get_inputs()
                if len(inputs) > 0:
                    return {inputs[0].name: safe_np}
            except Exception:
                pass
            return {"input": safe_np}

    @staticmethod
    def _run_inference_raw(model_path: str, input_file: str):
        """
        Internal: Run plain ONNX inference and return full raw outputs.

        Args:
            model_path: Path to the .onnx file
            input_file: Path to input.json

        Returns:
            (success: bool, result: dict)
                result on success:
                    {
                        "outputs": [
                            {
                                "name": <output name>,
                                "shape": [..],
                                "dtype": "float32" (etc),
                                "data": <nested list>
                            }, ...
                        ]
                    }
                result on failure:
                    { "error": <string> }
        """
        try:
            # Build ORT session
            session = ort.InferenceSession(model_path)

            # Read and prepare input
            input_tensor = RunnerUtils.preprocess_input(input_file)
            input_dict = OnnxModels.apply_onnx_shape(model_path, input_tensor)

            # Collect all output names to preserve ordering
            outputs_meta = session.get_outputs()
            output_names = [o.name for o in outputs_meta]

            # Run inference
            raw_outputs = session.run(output_names, input_dict)

            # Process output tensor for prediction
            output_tensor = torch.tensor(raw_outputs[0])
            prediction_result = RunnerUtils.process_final_output(output_tensor)

            # Normalize to JSON-serializable result
            outputs = []
            for meta, arr in zip(outputs_meta, raw_outputs):
                np_arr = np.asarray(arr)
                outputs.append({
                    "name": meta.name,
                    "shape": list(np_arr.shape),
                    "dtype": str(np_arr.dtype),
                    "data": np_arr.tolist(),
                })

            return True, {"prediction": prediction_result, "outputs": outputs}

        except Exception as e:
            logger.warning(f"Error during raw ONNX inference: {e}")
            return False, {"error": str(e)}


# Example usage
if __name__ == "__main__":

    # Choose which model to test
    model_choice = 5  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/yolov3",
        5: "../models/age",
        6: "../models/version"
    }

    # Get model directory
    abs_path = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(abs_path, "slices")
    # input_json = os.path.join(abs_path, "input.json")
    output_json = os.path.join(abs_path, "output.json")
    # model_path = os.path.join(abs_path, "model.onnx")
    input_json = "/Volumes/SSD/Users/dan/Projects/dsperse/src/models/age/run/run_20250905_154933/segment_12/input.json"
    model_path = "/Volumes/SSD/Users/dan/Projects/dsperse/src/models/age/slices/segment_12/segment_12.onnx"
    print(f"Running inference on {abs_path}")

    # result = OnnxModels.run_inference(input_file=input_json, model_path=os.path.join(abs_path, "model.onnx"), output_file="output.json")
    result = OnnxModels._run_inference_raw(model_path=model_path, input_file=input_json)
    print(result)
