import os

import torch

from dsperse.src.models.doom.model import DoomAgent, Conv1Segment as doomConv1, Conv2Segment as doomConv2, \
    Conv3Segment as doomConv3, FC1Segment as doomFC1, FC2Segment as doomFC2
from dsperse.src.models.net.model import Net, Conv1Segment as netConv1, Conv2Segment as netConv2, FC1Segment as netFC1, \
    FC2Segment as netFC2, FC3Segment as netFC3
from dsperse.src.run.utils.runner_utils import RunnerUtils

env = os.environ


class ModelRunner:
    def __init__(self, model_directory: str, model_path: str = None):
        self.device = torch.device("cpu")
        self.model_directory = os.path.join(ModelRunner._get_file_path(), model_directory)
        self.model_path = os.path.join(ModelRunner._get_file_path(), model_path) if model_path else None

    @staticmethod
    def _get_file_path() -> str:
        """Get the parent directory path of the current file."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def infer(self, mode: str = None, input_path: str = None) -> dict:

        input_path = input_path if input_path else os.path.join(self.model_directory, "input.json")
        input_tensor = RunnerUtils.preprocess_input(input_path, self.model_directory)

        if mode == "sliced":
            result = self.run_layered_inference(input_tensor)
        else:
            result = self.run_inference(input_tensor)

        return result

    def run_inference(self, input_tensor, model_directory: str = None, model_path: str = None):
        """
        Run inference with the model and return the logits, probabilities, and predictions.
        """
        try:
            # load the model
            self.model_directory = model_directory if model_directory else self.model_directory
            self.model_path = model_path if model_path else os.path.join(self.model_directory, "model.pth")

            checkpoint = torch.load(self.model_path, map_location=self.device)

            # TODO: figure out a way to make this dynamic for any model
            if "doom" in self.model_directory.lower():
                model = DoomAgent(n_actions=7)
            elif "net" in self.model_directory.lower():
                model = Net()
            else:
                raise ValueError("Unsupported model.")

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()

            # Run inference
            with torch.no_grad():
                raw_output = model(input_tensor)

            result = RunnerUtils.process_final_output(raw_output)
            return result

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None


    def run_layered_inference(self, input_tensor, slices_directory: str = None):
        try:
            slices_directory = slices_directory or os.path.join(self.model_directory, "model_slices")
            # get the segments this model was split into
            segments = RunnerUtils.get_segments(slices_directory)
            if segments is None:
                return None

            for idx, segment in enumerate(segments):
                # print(f"\nProcessing segment {idx + 1}/{num_segments}")
                segment_path = segment["path"]

                if "doom" in segment_path:
                    SegmentClass = self._get_doom_segment_class(idx)
                    segment_model = SegmentClass()
                elif 'net' in segment_path:
                    SegmentClass = self._get_net_segment_class(idx)
                    segment_model = SegmentClass()
                else:
                    raise Exception("Invalid type of segment")
                segment_path = os.path.join(self._get_file_path(), segment_path)
                checkpoint = torch.load(segment_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    segment_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    segment_model.load_state_dict(checkpoint)

                segment_model.to(self.device)
                segment_model.eval()

                # Run inference
                with torch.no_grad():
                    raw_output = segment_model(input_tensor)
                    # chain together
                    input_tensor = raw_output

            result = RunnerUtils.process_final_output(raw_output)
            return result

        except Exception as e:
            print(f"Error occurred in layered inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _get_net_segment_class(idx):
        mapping = {
            0: netConv1,
            1: netConv2,
            2: netFC1,
            3: netFC2,
            4: netFC3
        }
        segment_class = mapping.get(idx)
        if segment_class is None:
            raise ValueError(f"No corresponding class found for segment index {idx}")
        return segment_class


    @staticmethod
    def _get_doom_segment_class(idx):
        mapping = {
            0: doomConv1,
            1: doomConv2,
            2: doomConv3,
            3: doomFC1,
            4: doomFC2
        }
        segment_class = mapping.get(idx)
        if segment_class is None:
            raise ValueError(f"No corresponding class found for segment index {idx}")
        return segment_class

# Example usage
if __name__ == "__main__":

    # Choose which model to test
    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net"
    }

    model_dir = base_paths[model_choice]
    model_runner = ModelRunner(model_directory=model_dir)

    if model_choice == 1:
        print(model_runner.infer())
        print(model_runner.infer(mode="sliced"))

    elif model_choice == 2:
        print(model_runner.infer())
        print(model_runner.infer(mode="sliced"))

    else:
        print("Invalid model choice. Please choose 1 or 2.")
