import os

import onnx
import onnxruntime as ort
import torch

from src.runners.runner_utils import RunnerUtils
from src.utils.model_utils import ModelUtils
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.tools.onnx_model_utils import optimize_model, ModelProtoWithShapeInfo
# from onnxruntime.tools.remove_initializer_from_input import remove_initializer_from_input


class OnnxRunner:
    def __init__(self, model_directory: str,  model_path: str = None):
        self.device = torch.device("cpu")
        self.model_directory = os.path.join(OnnxRunner._get_file_path(), model_directory)
        self.model_path = os.path.join(OnnxRunner._get_file_path(), model_path) if model_path else None

    @staticmethod
    def _get_file_path() -> str:
        """Get the parent directory path of the current file."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    def preprocess_onnx_model_slices(self):
        """Remove initializers from inputs in an ONNX model"""
        print("Preprocessing ONNX model...")
        path = self.model_directory + "/onnx_slices/model.onnx"

        model = onnx.load(path)
        # model = optimize_model(self.model_path, output_path=model_path)
        model = ModelProtoWithShapeInfo(path).model_with_shape_info
        # model = remove_initializer_from_input(model)
        onnx.save(model, path)

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
        Run inference with sliced ONNX models using a computational graph approach.
        """
        try:
            # Get the directory containing the sliced models
            slices_directory = os.path.join(self.model_directory, "onnx_slices")

            # Load metadata
            metadata = ModelUtils.load_metadata(slices_directory)
            if metadata is None:
                return None

            # Build computational graph
            comp_graph = self.build_computational_graph(metadata)

            # Dictionary to store all intermediate outputs
            intermediate_outputs = {}

            # Get segments
            segments = metadata.get('segments', [])

            # Process each segment in sequence
            for segment in segments:
                segment_idx = segment['index']
                segment_path = segment['path']

                # Create an ONNX Runtime session for this segment
                session = ort.InferenceSession(segment_path)

                # Prepare inputs for this segment
                input_feed = {}

                # Get required inputs from computational graph
                for input_info in session.get_inputs():
                    input_name = input_info.name

                    # Skip constants/initializers - they're already in the model
                    if input_name in comp_graph[segment_idx]['constants']:
                        continue

                    # Handle original input
                    if comp_graph[segment_idx]['inputs'].get(input_name) == "original_input":
                        input_feed[input_name] = input_tensor.numpy()

                    # Handle intermediate outputs from previous segments
                    elif input_name in intermediate_outputs:
                        input_feed[input_name] = intermediate_outputs[input_name]
                    else:
                        print(f"Warning: Required input '{input_name}' not found in intermediate outputs")

                # Run inference on this segment
                outputs = session.run(None, input_feed)
                outputs = session.run(None, {"leakyRelu": value1, "conv": val2})

                # Store all outputs in our intermediate outputs dictionary
                for i, output_info in enumerate(session.get_outputs()):
                    output_name = output_info.name
                    intermediate_outputs[output_name] = outputs[i]

            # Get the final output (from the last segment's last output)
            final_segment = segments[-1]
            final_output_name = final_segment['dependencies']['output'][-1]
            final_output = intermediate_outputs[final_output_name]

            # Convert to PyTorch tensor and process
            output_tensor = torch.tensor(final_output)
            result = RunnerUtils.process_final_output(output_tensor)
            return result

        except Exception as e:
            print(f"Error during layered ONNX inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    # def run_layered_inference(self, input_tensor):
    #     """
    #     Run inference with sliced ONNX models and return the logits, probabilities, and predictions.
    #     """
    #     try:
    #         # Get the directory containing the sliced models
    #         # First try the new structure (onnx_slices)
    #         slices_directory = os.path.join(self.model_directory, "onnx_slices")
    #
    #         # If that doesn't exist, try the old structure (onnx_slices directly)
    #         if not os.path.exists(slices_directory) or not os.path.exists(os.path.join(slices_directory, "metadata.json")):
    #             slices_directory = os.path.join(self.model_directory, "onnx_slices")
    #
    #         # Get the segments this model was split into
    #         segments = RunnerUtils.get_segments(slices_directory)
    #         if segments is None:
    #             return None
    #
    #         # Process each segment in sequence
    #         for idx, segment in enumerate(segments):
    #             segment_path = segment["path"]
    #
    #             # Create an ONNX Runtime session for this segment
    #             session = ort.InferenceSession(segment_path)
    #
    #             # Get the input name for the ONNX model
    #             input_name = session.get_inputs()[0].name
    #
    #             # Convert PyTorch tensor to numpy array for ONNX Runtime
    #             input_numpy = input_tensor.numpy()
    #
    #             # Run inference on this segment
    #             raw_output = session.run(None, {input_name: input_numpy})
    #
    #             # Convert the output back to a PyTorch tensor and use as input for next layer
    #             input_tensor = torch.tensor(raw_output[0])
    #
    #         # Process the final output
    #         result = RunnerUtils.process_final_output(input_tensor)
    #         return result
    #
    #     except Exception as e:
    #         print(f"Error during layered ONNX inference: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None

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

    @staticmethod
    def build_computational_graph(metadata):
        """
        Build a computational graph dictionary from metadata.json
        """
        segments = metadata.get('segments', [])
        comp_graph = {}

        # Dictionary to track where each tensor comes from
        tensor_sources = {}

        # Process each segment
        for segment in segments:
            segment_idx = segment['index']
            comp_graph[segment_idx] = {
                'inputs': {},
                'outputs': [],
                'constants': {}
            }

            # Record all outputs from this segment
            for output in segment['dependencies']['output']:
                tensor_sources[output] = segment_idx
                comp_graph[segment_idx]['outputs'].append(output)

            # Process inputs for this segment
            for input_name in segment['dependencies']['input']:
                # Check if this is a constant/initializer (starts with "onnx::")
                if input_name.startswith("onnx::"):
                    # This is a constant weight/bias
                    comp_graph[segment_idx]['constants'][input_name] = True
                # Check if this is the original model input
                elif input_name == "x" or input_name == "input":
                    comp_graph[segment_idx]['inputs'][input_name] = "original_input"
                # Otherwise, it's an intermediate tensor from a previous segment
                elif input_name in tensor_sources:
                    source_segment = tensor_sources[input_name]
                    comp_graph[segment_idx]['inputs'][input_name] = source_segment
                else:
                    print(f"Warning: Input {input_name} for segment {segment_idx} has unknown source")

        return comp_graph


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
    # model_runner.preprocess_onnx_model()

    result = model_runner.infer(mode="sliced") # change function and mode when needed
    print(result)
