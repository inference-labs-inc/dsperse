import json
import os
import subprocess

import torch

from src.models.doom.model import DoomAgent, Conv1Segment, Conv2Segment, Conv3Segment, FC1Segment, FC2Segment
from src.models.net.model import Net, Conv1Segment as NetConv1, Conv2Segment as NetConv2, FC1Segment as NetFC1, \
    FC2Segment as NetFC2, FC3Segment as NetFC3


class ModelCircuitizer:
    def __init__(self, model_directory: str = None, input_file_path: str = None, model_path: str = None,):
        self.model_path = model_path
        self.input_file_path = input_file_path
        self.model_dir = model_directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = os.environ
        
        
    def circuitize_net_model(self, model_directory=None, input_file_path=None, model_path=None):
        """Circuitize an unsliced model."""
        self.model_dir = model_directory if model_directory else self.model_dir
        self.input_file_path = input_file_path if input_file_path else os.path.join(self.model_dir, "input.json")
        self.model_path = model_path if model_path else os.path.join(self.model_dir, "model.pth")
        
        # TODO: find dummy input params from model --> Make the model instantiation + input shape dynamic
        dummy_input = torch.randn(1, 3, 32, 32, device=self.device)
        model = Net().to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        # model.load_state_dict(checkpoint["model_state_dict"])

        # TODO: verify if this is ok -> to load the whole checkpoint as state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        # generate onnx
        circuit_folder = os.path.join(model_dir, "ezkl", "model")
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

        # generate calibration.json - use input.json for this
        input_data = os.path.join(self.model_dir, "input.json")
        subprocess.run(
            [
                "ezkl",
                "calibrate-settings",
                "--model",
                os.path.join(circuit_folder, "network.onnx"),
                "--settings-path",
                os.path.join(circuit_folder, "settings.json"),
                "--data",
                os.path.join(input_data),
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

    def circuitize_doom_model(self):
        """Circuitize a model."""
        dummy_input = torch.randn(1, 4, 28, 28, device=self.device)
        model = DoomAgent().to(self.device)
        checkpoint = torch.load("models/doom/model.pth", map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # generate onnx
        circuit_folder = "models/doom/ezkl/model"
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

        # generate calibration.json - use input.json for this
        input_data = os.path.join(self.model_dir, "input.json")
        subprocess.run(
            [
                "ezkl",
                "calibrate-settings",
                "--model",
                os.path.join(circuit_folder, "network.onnx"),
                "--settings-path",
                os.path.join(circuit_folder, "settings.json"),
                "--data",
                os.path.join(input_data),
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
    def _get_net_segment_class(idx):
        mapping = {
            0: NetConv1,
            1: NetConv2,
            2: NetFC1,
            3: NetFC2,
            4: NetFC3
        }
        segment_class = mapping.get(idx)
        if segment_class is None:
            raise ValueError(f"No corresponding class found for segment index {idx}")
        return segment_class

    @staticmethod
    def _get_doom_segment_class(idx):
        mapping = {
            0: Conv1Segment,
            1: Conv2Segment,
            2: Conv3Segment,
            3: FC1Segment,
            4: FC2Segment
        }
        segment_class = mapping.get(idx)
        if segment_class is None:
            raise ValueError(f"No corresponding class found for segment index {idx}")
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

    @staticmethod
    def _fix_constant_nodes(segment_path, output_path):
        """
        Convert problematic Constant nodes to initializers for EZKL compatibility.
        
        Args:
            segment_path: Path to the original segment ONNX file
            output_path: Path to save the fixed segment ONNX file
        """
        import onnx
        
        # Load the problematic segment
        model = onnx.load(segment_path)
        
        print(f"Fixing Constant nodes in {segment_path}...")
        
        # Work directly with the ONNX model to convert Constant to initializer
        nodes_to_remove = []
        for i, node in enumerate(model.graph.node):
            if node.op_type == 'Constant':
                print(f'Found Constant node: {node.name}')
                
                # Get the constant value tensor
                const_tensor = None
                for attr in node.attribute:
                    if attr.name == 'value':
                        const_tensor = attr.t
                        break
                
                if const_tensor:
                    # Create a unique name for the initializer
                    init_name = node.name.replace('/', '_') + '_as_initializer'
                    
                    # Create a new initializer with the same data
                    new_initializer = onnx.TensorProto()
                    new_initializer.CopyFrom(const_tensor)
                    new_initializer.name = init_name
                    
                    # Add the initializer to the graph
                    model.graph.initializer.append(new_initializer)
                    
                    # Update the node output name to match the initializer
                    if len(node.output) > 0:
                        old_output_name = node.output[0]
                        
                        # Replace all references to the old output with the new initializer name
                        for other_node in model.graph.node:
                            for j, input_name in enumerate(other_node.input):
                                if input_name == old_output_name:
                                    other_node.input[j] = init_name
                        
                        # Mark node for removal
                        nodes_to_remove.append(node)
                        
                        print(f'Converted Constant node to initializer: {init_name}')
        
        # Remove the constant nodes
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        
        print(f'Saving fixed model to {output_path}...')
        onnx.save(model, output_path)

    def circuitize_sliced_doom_model(self, model_directory: str = None):
        """circuitize each slice found in the provided directory."""
        model_directory = model_directory if model_directory else self.model_dir
        metadata_path = os.path.join(model_directory, "onnx_analysis/model_metadata.json")
        segments_path = os.path.join(model_directory, "onnx_slices")
        # Explicitly load metadata details
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        segments = metadata['nodes']
        circuit_folder = os.path.join(model_directory, "ezkl/slices")
        os.makedirs(circuit_folder, exist_ok=True)

        # Process all segments including segment 2 with fixed version
        segments_to_process = [0, 1, 2, 3, 4]  # Include all segments
        print(f"Processing segments: {segments_to_process} (with segment 2 fixed for Constant node compatibility)")

        # Iterate through all segments
        for idx in segments_to_process:
            print(f"\n🔄 Starting circuitization of doom segment {idx}...")
            segment_filename = f"segment_{idx}.onnx"
            segment_path = os.path.join(segments_path, segment_filename)
            slice_output_path = os.path.join(circuit_folder, f"segment_{idx}")
            os.makedirs(slice_output_path, exist_ok=True)

            # Special handling for segment 2 - use fixed version
            if idx == 2:
                fixed_segment_path = os.path.join(segments_path, "segment_2_fixed.onnx")
                if not os.path.exists(fixed_segment_path):
                    # Create fixed version if it doesn't exist
                    self._fix_constant_nodes(segment_path, fixed_segment_path)
                segment_path = fixed_segment_path
                print(f"Using fixed segment 2: {segment_path}")

            # Load the ONNX model directly instead of using PyTorch segments
            import onnx
            onnx_model = onnx.load(segment_path)
            
            # Export to ONNX (already in ONNX format, just copy)
            onnx_filename = os.path.join(slice_output_path, f"segment_{idx}.onnx")
            onnx.save(onnx_model, onnx_filename)

            # Generate dummy input matching segment
            if idx == 0:
                dummy_input_shape = (1, 4, 28, 28)
            elif idx == 1:
                dummy_input_shape = (1, 16, 28, 28)
            elif idx == 2:  # Conv3 segment - from 16 channels to 32 channels
                dummy_input_shape = (1, 16, 28, 28)
            elif idx == 3:  # FC1 segment
                dummy_input_shape = (1, 32, 7, 7)
            elif idx == 4:  # FC2 segment
                dummy_input_shape = (1, 256)
            else:
                raise ValueError(f"Unknown index: {idx}")

            dummy_input = torch.randn(dummy_input_shape, device=self.device)

            # generate settings.json
            subprocess.run(
                [
                    "ezkl",
                    "gen-settings",
                    "--param-visibility", "fixed",
                    "--input-visibility", "public",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(slice_output_path, f"segment_{idx}_settings.json")
                ],
                env=self.env,
                check=True
            )

            calibration_json_filepath = os.path.join(slice_output_path, f"segment_{idx}_calibration.json")
            self._create_dummy_calibration(calibration_json_filepath, dummy_input)

            # generate calibration.json
            subprocess.run(
                [
                    "ezkl",
                    "calibrate-settings",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(slice_output_path, f"segment_{idx}_settings.json"),
                    "--data", os.path.join(slice_output_path, f"segment_{idx}_calibration.json"),
                ],
                env=self.env,
                check=True
            )

            # Set logrows to 21 for all segments
            settings_path = os.path.join(slice_output_path, f"segment_{idx}_settings.json")
            with open(settings_path, 'r') as f:
                settings = json.load(f)

            settings["run_args"]['logrows'] = 21

            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=4)

            # generate model.compiled
            subprocess.run(
                [
                    "ezkl",
                    "compile-circuit",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(slice_output_path, f"segment_{idx}_settings.json"),
                    "--compiled-circuit", os.path.join(slice_output_path, f"segment_{idx}_model.compiled")
                ],
                env=self.env,
                check=True
            )

            # generate pk and vk
            subprocess.run(
                [
                    "ezkl",
                    "setup",
                    "--compiled-circuit", os.path.join(slice_output_path, f"segment_{idx}_model.compiled"),
                    "--vk-path", os.path.join(slice_output_path, f"segment_{idx}_vk.key"),
                    "--pk-path", os.path.join(slice_output_path, f"segment_{idx}_pk.key")
                ],
                env=self.env,
                check=True
            )
            
            print(f"✅ Completed circuitization of doom segment {idx}")

        print(f"\n🎉 Doom model circuitization completed! Processed segments: {segments_to_process}")
        print("✅ All segments including segment 2 (with Constant node fix) have been processed successfully!")

    def circuitize_sliced_net_model(self, model_directory: str = None):
        """circuitize each slice found in the provided directory."""
        model_directory = model_directory if model_directory else self.model_dir
        metadata_path = os.path.join(model_directory, "onnx_analysis/model_metadata.json")
        segments_path = os.path.join(model_directory, "onnx_slices")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        segments = metadata['nodes']
        circuit_folder = os.path.join(model_directory, "ezkl/slices")
        os.makedirs(circuit_folder, exist_ok=True)


        for idx in range(5):  # We know there are 5 segments from the ONNX slicer
            segment_filename = f"segment_{idx}.onnx"
            segment_path = os.path.join(segments_path, segment_filename)
            slice_output_path = os.path.join(circuit_folder, f"segment_{idx}")
            os.makedirs(slice_output_path, exist_ok=True)

            # Load the ONNX model directly instead of using PyTorch segments
            import onnx
            onnx_model = onnx.load(segment_path)
            
            # Export to ONNX (already in ONNX format, just copy)
            onnx_filename = os.path.join(slice_output_path, f"segment_{idx}.onnx")
            onnx.save(onnx_model, onnx_filename)

            # dummy inputs for layers
            if idx == 0:
                dummy_input_shape = (1, 3, 32, 32)
            elif idx == 1:
                dummy_input_shape = (1, 6, 14, 14)
            elif idx == 2:
                dummy_input_shape = (1, 16, 5, 5)
            elif idx == 3:
                dummy_input_shape = (1, 120)
            elif idx == 4:
                dummy_input_shape = (1, 84)
            else:
                raise ValueError(f"Unknown index: {idx}")

            dummy_input = torch.randn(dummy_input_shape, device=self.device)

            # generate settings.json
            subprocess.run(
                [
                    "ezkl",
                    "gen-settings",
                    "--param-visibility", "fixed",
                    "--input-visibility", "public",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(slice_output_path, f"segment_{idx}_settings.json")
                ],
                env=self.env,
                check=True
            )

            calibration_json_filepath = os.path.join(slice_output_path, f"segment_{idx}_calibration.json")
            self._create_dummy_calibration(calibration_json_filepath, dummy_input)

            subprocess.run(
                [
                    "ezkl",
                    "calibrate-settings",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(slice_output_path, f"segment_{idx}_settings.json"),
                    "--data", os.path.join(slice_output_path, f"segment_{idx}_calibration.json"),
                ],
                env=self.env,
                check=True
            )

            # change settings
            # json load setting file and change 'decomp_legs' to 4
            settings_path = os.path.join(slice_output_path, f"segment_{idx}_settings.json")
            with open(settings_path, 'r') as f:
                settings = json.load(f)

            settings["run_args"]['decomp_legs'] = 4
            # Set logrows to 21 for all segments
            settings["run_args"]['logrows'] = 21

            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=4)

            subprocess.run(
                [
                    "ezkl",
                    "compile-circuit",
                    "--model", onnx_filename,
                    "--settings-path", os.path.join(slice_output_path, f"segment_{idx}_settings.json"),
                    "--compiled-circuit", os.path.join(slice_output_path, f"segment_{idx}_model.compiled")
                ],
                env=self.env,
                check=True
            )

            subprocess.run(
                [
                    "ezkl",
                    "setup",
                    "--compiled-circuit", os.path.join(slice_output_path, f"segment_{idx}_model.compiled"),
                    "--vk-path", os.path.join(slice_output_path, f"segment_{idx}_vk.key"),
                    "--pk-path", os.path.join(slice_output_path, f"segment_{idx}_pk.key")
                ],
                env=self.env,
                check=True
            )


# Example usage
# TODO: Integrate into Ezkl runner
if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "src/models/doom",
        2: "src/models/net"
    }

    model_dir = base_paths[model_choice]
    model_circuitizer = ModelCircuitizer(model_directory=model_dir)

    if model_choice == 1:
        # model_circuitizer.circuitize_doom_model()
        model_circuitizer.circuitize_sliced_doom_model()

    elif model_choice == 2:
        # model_circuitizer.circuitize_net_model()  # Skip unsliced version
        model_circuitizer.circuitize_sliced_net_model()
