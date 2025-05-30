import os
import onnxruntime as ort
import torch

from src.runners.runner_utils import RunnerUtils


class OnnxRunner:
    def __init__(self, model_directory: str,  model_path: str = None):
        self.device = torch.device("cpu")
        self.model_directory = os.path.join(OnnxRunner._get_file_path(), model_directory)
        self.model_path = os.path.join(OnnxRunner._get_file_path(), model_path) if model_path else None

    @staticmethod
    def _get_file_path() -> str:
        """Get the parent directory path of the current file."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def infer(self, mode: str = None, input_path: str = None) -> dict:
        """
        Run inference with the ONNX model.
        Args:
            mode: "sliced" to run layered inference, None or any other value for whole model inference
            input_path: path to the input JSON file, if None uses default input.json in model directory
        Returns:
            dict with inference results
        """
        input_path = input_path if input_path else os.path.join(self.model_directory, "input.json")
        input_tensor = RunnerUtils.preprocess_input(input_path, self.model_directory)

        if mode == "sliced":
            result = self.run_layered_inference(input_tensor)
        else:
            result = self.run_inference(input_tensor)

        return result

    def run_layered_inference(self, input_tensor):
        """
        Run inference with sliced ONNX models and return the logits, probabilities, and predictions.
        """
        try:
            # Get the directory containing the sliced models
            slices_directory = os.path.join(self.model_directory, "onnx_slices", "single_layers")

            # Get the segments this model was split into
            segments = RunnerUtils.get_segments(slices_directory)
            if segments is None:
                return None

            # Process each segment in sequence
            for idx, segment in enumerate(segments):
                segment_path = segment["path"]

                # Create an ONNX Runtime session for this segment
                session = ort.InferenceSession(segment_path)

                # Get the input name for the ONNX model
                input_name = session.get_inputs()[0].name

                # Convert PyTorch tensor to numpy array for ONNX Runtime
                input_numpy = input_tensor.numpy()

                # Run inference on this segment
                raw_output = session.run(None, {input_name: input_numpy})

                # Convert the output back to a PyTorch tensor and use as input for next layer
                input_tensor = torch.tensor(raw_output[0])

            # Process the final output
            result = RunnerUtils.process_final_output(input_tensor)
            return result

        except Exception as e:
            print(f"Error during layered ONNX inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_inference(self, input_tensor):
        """
        Run inference with the ONNX model and return the logits, probabilities, and predictions.
        """
        try:
            # Load the ONNX model
            model_path = os.path.join(self.model_directory, "model.onnx")

            # Create an ONNX Runtime session
            session = ort.InferenceSession(model_path)

            # Get the input name for the ONNX model
            input_name = session.get_inputs()[0].name

            # Convert PyTorch tensor to numpy array for ONNX Runtime
            input_numpy = input_tensor.numpy()

            # Run inference
            raw_output = session.run(None, {input_name: input_numpy})

            # Convert the output back to a PyTorch tensor
            output_tensor = torch.tensor(raw_output[0])

            # Process the output
            result = RunnerUtils.process_final_output(output_tensor)
            return result

        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            import traceback
            traceback.print_exc()
            return None


# Example usage
if __name__ == "__main__":

    # Choose which model to test
    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet",
        4: "models/yolov3"
    }

    model_dir = base_paths[model_choice]
    model_runner = OnnxRunner(model_directory=model_dir)

    if model_choice in [1, 2, 3, 4]:
        # print(model_runner.infer())
        print(model_runner.infer(mode="sliced"))
    else:
        print("Invalid model choice. Please choose 1, 2, 3, or 4.")
