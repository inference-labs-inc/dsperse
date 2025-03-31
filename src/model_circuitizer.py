import json
import math
import os
import subprocess

import torch
from torch import nn

from models.doom.model import DoomAgent, Conv1Segment, Conv2Segment, Conv3Segment, FC1Segment, FC2Segment


class ModelCircuitizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = os.environ

    def circuitize_doom_model(self):
        """Circuitize a model."""
        dummy_input = torch.randn(1, 4, 28, 28, device=self.device)
        model = DoomAgent().to(self.device)
        checkpoint = torch.load("models/doom/doom.pth", map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # generate onnx
        circuit_folder = "models/doom/circuit/"
        os.makedirs(circuit_folder, exist_ok=True)
        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(circuit_folder, "network.onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=18,
        )
        print("Model exported to network.onnx")

        # generate settings.json
        subprocess.run(
            [
                "ezkl",
                "gen-settings",
                "--param-visibility",
                "fixed",
                "--input-visibility",
                "public",
                "--model",
                os.path.join(circuit_folder, "network.onnx"),
                "--settings-path",
                os.path.join(circuit_folder, "settings.json"),
            ],
            env=self.env,
            check=True,
        )

        # generate calibration.json
        subprocess.run(
            [
                "ezkl",
                "calibrate-settings",
                "--model",
                os.path.join(circuit_folder, "network.onnx"),
                "--settings-path",
                os.path.join(circuit_folder, "settings.json"),
                "--data",
                os.path.join(circuit_folder, "calibration.json"),
                "--target", "accuracy"
            ],
            env=self.env,
            check=True,
        )

        # generate model.compiled
        subprocess.run(
            [
                "ezkl",
                "compile-circuit",
                "--model",
                os.path.join(circuit_folder, "network.onnx"),
                "--settings-path",
                os.path.join(circuit_folder, "settings.json"),
                "--compiled-circuit",
                os.path.join(circuit_folder, "model.compiled"),
            ],
            env=self.env,
            check=True
        )

        # generate pk and vk
        subprocess.run(
            [
                "ezkl",
                "setup",
                "--compiled-circuit",
                os.path.join(circuit_folder, "model.compiled"),
                "--vk-path",
                os.path.join(circuit_folder, "vk.key"),
                "--pk-path",
                os.path.join(circuit_folder, "pk.key")
            ],
            env=self.env,
            check=True
        )

    @staticmethod
    def _get_segment_class(idx):
        """Explicit mapping explicitly indexed distinctly clearly and explicitly."""
        mapping = {
            0: Conv1Segment,
            1: Conv2Segment,
            2: Conv3Segment,
            3: FC1Segment,
            4: FC2Segment
        }
        segment_class = mapping.get(idx)
        if segment_class is None:
            raise ValueError(f"No corresponding explicit class explicitly found for segment index {idx}")
        return segment_class

    def _create_dummy_calibration(self, json_filepath, input_tensor):
        """
        Creates a calibration file with properly normalized input in EZKL's expected format.

        Args:
            json_filepath: Path where to save the calibration file
            input_tensor: Either a tensor or shape tuple
        """
        # Check if input_tensor is already a tensor or just a shape tuple
        if not isinstance(input_tensor, torch.Tensor):
            # It's a shape tuple, create a tensor
            input_tensor = torch.randn(*input_tensor, device=self.device)

        # Normalize to [0,1] range
        min_val = float(input_tensor.min())
        max_val = float(input_tensor.max())
        normalized_tensor = (input_tensor - min_val) / (max_val - min_val)

        # Completely flatten the tensor - this is what EZKL expects
        flat_data = normalized_tensor.cpu().view(-1).tolist()

        # Create proper JSON structure with EXACTLY one level of nesting
        # This creates: {"input_data": [[val1, val2, val3, ...]]}
        calibration_data = {"input_data": [flat_data]}

        # Write with proper formatting
        try:
            with open(json_filepath, 'w') as f:
                json_str = json.dumps(calibration_data)
                f.write(json_str)

            # Verify the first part of file
            with open(json_filepath, 'r') as f:
                start = f.read(min(100, os.path.getsize(json_filepath)))
                print(f"Generated JSON starts with: {start}...")

            print(f"✅ Calibration file created successfully: {json_filepath}")
        except Exception as e:
            print(f"❌ Error writing calibration JSON: {e}")
            raise

    def circuitize_sliced_model(self, sliced_dir):
        """circuitize each slice found in the provided directory."""
        metadata_path = "models/doom/output/"
        metadata_path = os.path.join(sliced_dir, "metadata.json")

        # Explicitly load metadata details
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        segments = metadata['segments']
        circuit_folder = "models/doom/output/circuitized_slices/"
        os.makedirs(circuit_folder, exist_ok=True)

        # Iterate distinctly through segments
        for idx, segment_meta in enumerate(segments):
            segment_type = segment_meta['type']
            segment_filename = segment_meta.get('filename')
            segment_path = os.path.join(sliced_dir, segment_filename)

            print(f"\n🚧 Circuitizing segment {idx + 1}/{len(segments)}: {segment_filename}")

            # instantiate the specific segment class
            SegmentClass = self._get_segment_class(idx)
            segment_model = SegmentClass()
            segment_model.load_state_dict(torch.load(segment_path, map_location=self.device))
            segment_model.to(self.device)
            segment_model.eval()

            # Generate dummy input matching segment
            if idx == 0:
                dummy_input_shape = (1, 4, 28, 28)
            elif idx == 1:
                dummy_input_shape = (1, 16, 28, 28)
            elif idx == 2:
                dummy_input_shape = (1, 32, 14, 14)
            elif idx == 3:
                dummy_input_shape = (1, 32, 7, 7)
            elif idx == 4:
                dummy_input_shape = (1, 256)
            else:
                raise ValueError(f"Unknown index: {idx}")

            dummy_input = torch.randn(dummy_input_shape, device=self.device)

            # Export to ONNX
            onnx_filename = os.path.join(circuit_folder, f"segment_{idx}.onnx")
            torch.onnx.export(
                segment_model,
                dummy_input,
                onnx_filename,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=12
            )

            print(f"🚀 Segment {idx + 1} exported to ONNX: {onnx_filename}")
            # generate settings.json
            subprocess.run(
                [
                    "ezkl",
                    "gen-settings",
                    "--param-visibility", "fixed",
                    "--input-visibility", "public",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(circuit_folder, f"segment_{idx}_settings.json")
                ],
                env=self.env,
                check=True
            )

            # Explicitly create dummy calibration, then explicitly run EZKL calibration
            calibration_json_filepath = os.path.join(circuit_folder, f"segment_{idx}_calibration.json")
            self._create_dummy_calibration(calibration_json_filepath, dummy_input)

            if idx == 2:
                # generate calibration.json
                subprocess.run(
                    [
                        "ezkl",
                        "calibrate-settings",
                        "--model", onnx_filename,
                        "--settings-path", os.path.join(circuit_folder, f"segment_{idx}_settings.json"),
                        "--data", os.path.join(circuit_folder, f"segment_{idx}_calibration.json"),
                        "--target", "resources"
                    ],
                    env=self.env,
                    check=True
                )
            else:
                # generate calibration.json
                subprocess.run(
                    [
                        "ezkl",
                        "calibrate-settings",
                        "--model", onnx_filename,
                        "--settings-path", os.path.join(circuit_folder, f"segment_{idx}_settings.json"),
                        "--data", os.path.join(circuit_folder, f"segment_{idx}_calibration.json"),
                        "--target", "accuracy"
                    ],
                    env=self.env,
                    check=True
                )

            # generate model.compiled
            subprocess.run(
                [
                    "ezkl",
                    "compile-circuit",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(circuit_folder, f"segment_{idx}_settings.json"),
                    "--compiled-circuit", os.path.join(circuit_folder, f"segment_{idx}_model.compiled")
                ],
                env=self.env,
                check=True
            )

            # generate pk and vk
            subprocess.run(
                [
                    "ezkl",
                    "setup",
                    "--compiled-circuit", os.path.join(circuit_folder, f"segment_{idx}_model.compiled"),
                    "--vk-path", os.path.join(circuit_folder, f"segment_{idx}_vk.key"),
                    "--pk-path", os.path.join(circuit_folder, f"segment_{idx}_pk.key")
                ],
                env=self.env,
                check=True
            )

            print(f"🎉 EZKL setup complete for segment {idx + 1}\n")


# Example usage
if __name__ == "__main__":
    # model_path = "models/doom/doom.pth"
    model_path = "models/doom/output/"
    input_path = "models/doom/input/input.json"
    model_circuitizer = ModelCircuitizer()
    model_circuitizer.circuitize_doom_model()
    # model_circuitizer.circuitize_sliced_model(model_path)
