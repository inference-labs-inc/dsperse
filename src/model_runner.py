import subprocess
import time
import torch
import torch.nn.functional as F
from models.doom.model import DoomAgent, Conv1Segment as doomConv1, Conv2Segment as doomConv2, Conv3Segment as doomConv3, FC1Segment as doomFC1, FC2Segment as doomFC2
from models.net.model import Net, Conv1Segment as netConv1, Conv2Segment as netConv2, FC1Segment as netFC1, FC2Segment as netFC2, FC3Segment as netFC3
import os
import json
from pathlib import Path
from utils.model_utils import ModelUtils

env = os.environ


class ModelRunner:
    def __init__(self, model_directory: str, model_path: str = None):
        # Always use CPU
        self.device = torch.device("cpu")

        # Initialize attributes
        self.model_directory = model_directory
        self.model_path = model_path


    def run_inference(self, input_tensor, model_directory: str = None, model_path: str = None):
        """
        Run inference with the model and apply softmax to get probability distribution.
        """
        try:
            # load the model
            self.model_directory = model_directory if model_directory else self.model_directory
            self.model_path = model_path if model_path else os.path.join(self.model_directory, "model.pth")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            input_tensor = self._process_initial_input_tensor(input_tensor)

            # Create a proper model instance
            if "doom" in self.model_directory.lower():
                model = DoomAgent(n_actions=7)  # Adjust parameters as needed
            elif "net" in self.model_directory.lower():
                model = Net()  # Adjust initialization as needed
            else:
                raise ValueError("Unsupported model.")

            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Move to device and evaluate
            model = model.to(self.device)
            model.eval()

            # Run inference
            with torch.no_grad():
                raw_output = model(input_tensor)

            result = self._process_final_output(raw_output)
            return result

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_initial_input_tensor(self, input_tensor):
        if input_tensor.dim() == 2 and input_tensor.size(1) == 3136 and 'doom' in self.model_directory.lower():
            input_tensor = input_tensor.reshape(1, 4, 28, 28)
        elif input_tensor.dim() == 2 and input_tensor.size(1) == 3072:
            if input_tensor.size(0) == 1:
                # Single sample case
                input_tensor = input_tensor.reshape(1, 3, 32, 32)
            else:
                # Multiple samples case (e.g., batch size 100)
                print(f"Processing only the first sample out of {input_tensor.size(0)}")
                input_tensor = input_tensor[0:1].reshape(1, 3, 32, 32)
        else:
            raise ValueError(f"Input tensor has unsupported dimensions: {input_tensor.shape}")
        return input_tensor

    def run_layered_inference(self, input_tensor, slices_directory: str = None):
        try:
            slices_directory = slices_directory or os.path.join(self.model_directory, "slices")
            # get the segments this model was split into
            segments = self._get_segments(slices_directory)
            if segments is None:
                return None

            input_tensor = self._process_initial_input_tensor(input_tensor)

            num_segments = len(segments)
            # Continue processing segments
            for idx, segment in enumerate(segments):
                print(f"\nProcessing segment {idx + 1}/{num_segments}")
                segment_path = segment["path"]

                if "doom" in segment_path:
                    SegmentClass = self._get_doom_segment_class(idx)
                    segment_model = SegmentClass()
                elif 'net' in segment_path:
                    SegmentClass = self._get_net_segment_class(idx)
                    segment_model = SegmentClass()
                else:
                    raise Exception("Invalid type of segment")

                checkpoint = torch.load(segment_path, map_location=self.device)
                # load state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    segment_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    segment_model.load_state_dict(checkpoint)

                segment_model.to(self.device)
                segment_model.eval()

                # Run inference
                with torch.no_grad():
                    raw_output = segment_model(input_tensor)
                    #chain together
                    input_tensor = raw_output


            return self._process_final_output(raw_output)

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

    def _get_segments(self, slices_directory):
        metadata = self._load_metadata(slices_directory)
        if metadata is None:
            return None

        segments = metadata.get('segments', [])
        if not segments:
            print("No segments found in metadata.json")
            return None

        num_segments = len(segments)
        print(f"Found {num_segments} segments in metadata.json")
        return segments

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

    @staticmethod
    def _load_metadata(folder_path):
        """Load the model metadata from the metadata.json file."""
        metadata_path = Path(folder_path) / "metadata.json"
        if not metadata_path.exists():
            print(f"Required metadata.json file not found at: {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata

    @staticmethod
    def _get_model_details(metadata):
        """Extract basic model details from metadata."""
        model_type = metadata.get('model_type', 'Unknown')
        total_params = metadata.get('total_parameters', 'Unknown')
        slicing_strategy = metadata.get('slicing_strategy', 'Unknown')
        input_shape = metadata.get('input_data_info', {}).get('input_shape', 'Unknown')

        print(f"Model type: {model_type}")
        print(f"Total parameters: {total_params}")
        print(f"Slicing strategy: {slicing_strategy}")
        print(f"Model input shape from metadata: {input_shape}")

        return model_type, total_params, slicing_strategy, input_shape

    def _prepare_initial_input(self, x, input_shape):
        """Temporary fix for original input_shape=[3136] flattened (4,28,28)."""
        #TODO: hard coded --> make dynamic
        if len(input_shape) == 1 and input_shape[0] == 3136 and 'doom' in self.model_directory.lower():
            channels, height, width = 4, 28, 28  #
            x = x.reshape(-1, channels, height, width).to(self.device)
        elif len(input_shape) == 1 and input_shape[0] == 3072:
            channels, height, width = 3, 32, 32  #
            # Check if we have multiple samples (more than one row)
            if x.dim() > 1 and x.size(0) > 1:
                # Select only the first sample
                x = x[0:1]
                print(f"Multiple samples detected. Using only the first sample.")

            x = x.reshape(-1, channels, height, width).to(self.device)

        else:
            raise ValueError(f"Provided input_shape metadata seems incorrect or incomplete: {input_shape}")
        return x

    @staticmethod
    def _get_segment_path(segment, folder_path):
        """Get the file path for a segment."""
        if 'path' in segment:
            segment_path = segment['path']
            if not os.path.exists(segment_path):
                # Try using relative path
                segment_path = os.path.join(folder_path, os.path.basename(segment_path))
        elif 'filename' in segment:
            segment_path = os.path.join(folder_path, segment['filename'])
        else:
            return None

        return segment_path

    def _process_segment(self, segment, x, segment_idx, folder_path):
        """Process a single model segment."""
        segment_type = segment.get('type')
        segment_name = segment.get('name', f"segment_{segment_idx}")

        # Get the segment file path
        segment_path = self._get_segment_path(segment, folder_path)
        if not segment_path or not os.path.exists(segment_path):
            print(f"Segment file not found: {segment_path}")
            return None

        print(f"Loading segment from: {segment_path}")
        segment_data = torch.load(segment_path, map_location=self.device)

        # Load/get input for this segment
        print(f"Input tensor shape before processing: {x.shape}")

        # Validate input shape for segment type
        if segment_type == 'conv' and len(x.shape) != 4:
            print(
                f"Segment {segment_idx} is convolutional but input shape {x.shape} is not 4D [batch, channels, height, width]")
            return None

        # Apply any reshaping specified in the segment
        x = self._apply_segment_reshaping(x, segment)

        # Process based on segment type
        if segment_type == 'conv':
            x = self._process_conv_segment(segment, segment_data, x, segment_name)
        elif segment_type == 'fc' or segment_type == 'linear':
            x = self._process_fc_segment(segment, segment_data, x, segment_name)
        else:
            print(f"Unknown segment type '{segment_type}', skipping")

        return x

    @staticmethod
    def _apply_segment_reshaping(x, segment):
        """Apply any reshaping specified in the segment."""
        if 'input_reshape' in segment:
            reshape_info = segment['input_reshape']
            if reshape_info.get('type') == 'flatten':
                # Flatten to [batch_size, -1]
                original_shape = x.shape
                x = x.reshape(x.size(0), -1)
                print(f"Applied specified flattening: {original_shape} → {x.shape}")

            elif reshape_info.get('type') == 'reshape' and 'shape' in reshape_info:
                target_shape = reshape_info['shape']
                # Replace None/null with batch size
                target_shape = [x.size(0) if dim is None else dim for dim in target_shape]
                x = x.reshape(target_shape)
                print(f"Applied specified reshaping: {x.shape}")

        # Add a fallback for FC layers that don't have explicit reshape info
        elif segment.get('type') == 'fc' and len(x.shape) > 2:
            # This should only happen if no input_reshape was specified
            original_shape = x.shape
            x = x.reshape(x.size(0), -1)
            print(f"Applied default FC flattening: {original_shape} → {x.shape}")

        return x

    def _process_conv_segment(self, segment, segment_data, x, segment_name):
        """Process a convolutional segment."""
        print(f"Processing convolutional segment: {segment_name}")

        # # Hard-coded parameters for the DOOM model
        if segment_name == "segment_0":  # First conv layer
            stride, padding = 1, 1
        elif segment_name in ["segment_1", "segment_2"]:  # Second and third conv layers
            stride, padding = 2, 1
        else:
            stride, padding = 1, 0  # Default values

        for layer_info in segment.get('layers', []):
            layer_name = layer_info.get('name', 'unnamed_layer')

            # Extract weights and biases
            weight, bias = self._extract_weights_and_biases(segment_data, layer_name)

            if weight is None:
                print(f"Available keys in segment_data: {list(segment_data.keys())}")
                print(f"Could not find weights for layer {layer_name}")
                return None

            # # Get convolution parameters from the layer metadata todo: dynamically get stride and padding
            # stride = layer_info.get('stride', 1)
            # padding = layer_info.get('padding', 0)
            #
            # # Ensure proper formatting for stride and padding
            # if isinstance(stride, (list, tuple)):
            #     stride = tuple(stride)
            # elif isinstance(stride, int):
            #     stride = (stride, stride)
            #
            # if isinstance(padding, (list, tuple)):
            #     padding = tuple(padding)
            # elif isinstance(padding, int):
            #     padding = (padding, padding)

            print(f"  Layer {layer_name}: input={x.shape}, weights={weight.shape}, stride={stride}, padding={padding}")

            # Apply convolution
            x = F.conv2d(x, weight, bias, stride=stride, padding=padding)

            # Apply activation function if specified
            x = self._apply_activation(x, layer_info.get('activation'))
            print(f"  Output shape after layer {layer_name}: {x.shape}")

        return x

    def _process_fc_segment(self, segment, segment_data, x, segment_name):
        """Process a fully connected segment."""
        print(f"Processing fully connected segment: {segment_name}")

        for layer_info in segment.get('layers', []):
            layer_name = layer_info.get('name', 'unnamed_layer')

            # Extract weights and biases
            weight, bias = self._extract_weights_and_biases(segment_data, layer_name)

            if weight is None:
                print(f"Available keys in segment_data: {list(segment_data.keys())}")
                print(f"Could not find weights for layer {layer_name}")
                return None

            # Match input dimensions to weight matrix
            # x = self._match_input_dimensions(x, weight, layer_info)

            # Apply linear transformation
            print(f"  Layer {layer_name}: input={x.shape}, weights={weight.shape}")
            # x = F.linear(x, weight, bias)

            # Apply activation function if specified
            x = self._apply_activation(x, layer_info.get('activation'))
            print(f"  Output shape after layer {layer_name}: {x.shape}")

        return x

    def _extract_weights_and_biases(self, segment_data, layer_name):
        """Extract weights and biases from segment data."""
        weight = None
        bias = None

        # Try to find weights in segment_data
        for key in [f"{layer_name}.weight", "weight", f"{layer_name}_weight"]:
            if key in segment_data:
                weight = segment_data[key]
                break

        # Try to find biases in segment_data
        for key in [f"{layer_name}.bias", "bias", f"{layer_name}_bias"]:
            if key in segment_data:
                bias = segment_data[key]
                break

        return weight, bias

    def _match_input_dimensions(self, x, weight, layer_info):
        """Match the input dimensions to the weight matrix dimensions."""
        input_features = x.shape[1]
        expected_features = weight.shape[1]

        if input_features != expected_features:
            print(
                f"Input feature mismatch! Got {input_features} features but weights expect {expected_features} features")

            # Try to get expected feature count from layer info
            if 'input_features' in layer_info:
                input_features_target = layer_info['input_features']
                if input_features_target == expected_features:
                    print(f"Using input_features from layer info: {input_features_target}")

            # Handle dimension mismatch by cropping or padding
            if input_features > expected_features:
                print(f"Cropping input tensor from {input_features} to {expected_features} features")
                x = x[:, :expected_features]
            else:
                print(f"Padding input tensor from {input_features} to {expected_features} features with zeros")
                padding = torch.zeros(x.size(0), expected_features - input_features, device=x.device)
                x = torch.cat([x, padding], dim=1)

        return x

    def _apply_activation(self, x, activation):
        """Apply activation function to tensor."""

        if activation and activation.lower() != 'none':
            if activation.upper() == 'RELU':
                x = F.relu(x)
            elif activation.upper() == 'SIGMOID':
                x = torch.sigmoid(x)
            elif activation.upper() == 'TANH':
                x = torch.tanh(x)
            # elif activation.upper() == 'SOFTMAX':
            #     x = F.softmax(x, dim=1)

        return x

    @staticmethod
    def _process_final_output(torch_tensor):
        """Process the final output of the model."""
        # Verify and print the final output

        # Apply softmax to get probabilities if not already applied
        if len(torch_tensor.shape) != 2:  # Ensure raw output is 2D [batch_size, num_classes]
            print(f"Warning: Raw output shape {torch_tensor.shape} is not as expected. Reshaping to [1, -1].")
            torch_tensor = torch_tensor.reshape(1, -1)

        probabilities = F.softmax(torch_tensor, dim=1)
        predicted_action = torch.argmax(probabilities, dim=1).item()

        result = {
            "logits": torch_tensor,
            "probabilities": probabilities,
            "predicted_action": predicted_action
        }

        return result

    def predict(self, mode:str = None, input_path: str = None) -> dict:
        
        input_path = input_path if input_path else os.path.join(self.model_directory, "input.json") 
        input_tensor = ModelUtils.preprocess_input(input_path)

        if mode == "sliced":
            result = self.run_layered_inference(input_tensor)
        else:
            result = self.run_inference(input_tensor)

        return result

    def generate_witness(self, base_path: str = None, input_file: str = None, model_path: str = None) -> dict:
        base_path = base_path or os.path.join(self.model_directory, "ezkl", "model")
        input_file = input_file or os.path.join(self.model_directory, "input.json")
        model_path = model_path or os.path.join(base_path, "model.compiled")
        witness_path = os.path.join(base_path, "witness.json")
        vk_path = os.path.join(base_path, "vk.key")

        proof_time_start = time.time()
        
        subprocess.run(
            [
                "ezkl",
                "gen-witness",
                "--data", input_file,
                "--compiled-circuit", model_path,
                "--output", witness_path,
                "--vk-path", vk_path
            ],
            env=env,
            check=True
        )

        proof_time_end = time.time()

        print(f"Witness time: {proof_time_end - proof_time_start:.2f} seconds")

        with open(witness_path, "r") as f:
            witness_data = json.load(f)
            # outputs = witness_data["pretty_elements"]["rescaled_outputs"][0]
            outputs = witness_data
            # print(outputs)
        return outputs

    def process_witness_output(self, witness_data):
        """
        Process the witness.json data to get prediction results.
        """
        try:
            rescaled_outputs = witness_data["pretty_elements"]["rescaled_outputs"][0]
        except KeyError:
            print("Error: Could not find rescaled_outputs in witness data")
            return None

        # Convert string values to float and create a tensor
        float_values = [float(val) for val in rescaled_outputs]

        # Create a tensor with shape [1, num_classes] to match batch_size, num_classes format
        tensor_output = torch.tensor([float_values], dtype=torch.float32)

        # Process the tensor through _process_final_output (simulating one segment)
        result = self._process_final_output(tensor_output, 1)

        return result

    def generate_proof(self, base_path: str = None, model_directory: str = None):
        self.model_directory = model_directory or self.model_directory
        base_path = base_path or os.path.join(self.model_directory, "ezkl", "model")
        witness_path = os.path.join(base_path, "witness.json")
        model_path = os.path.join(base_path, "model.compiled")
        pk_path = os.path.join(base_path, "pk.key")
        proof_path = os.path.join(base_path, "proof.json")

        proof_time_start = time.time()

        subprocess.run(
            [
                "ezkl",
                "prove",
                "--check-mode", "unsafe",
                "--witness", witness_path,
                "--compiled-circuit", model_path,
                "--proof-path", proof_path,
                "--pk-path", pk_path
            ],
            env=env,
            check=True
        )

        proof_time_end = time.time()
        print(f"Proving time: {proof_time_end - proof_time_start:.2f} seconds")

        with open(proof_path, "r") as f:
            proof_data = json.load(f)
            outputs = proof_data["pretty_public_inputs"]["rescaled_outputs"][0]

        proof_time = time.time() - proof_time_start
        print(f"Proving time: {proof_time:.2f} seconds")

        return outputs, {"proof_time": proof_time}

    def generate_proof_sliced(self, slices_directory: str = None, model_directory: str = None, metadata_directory: str = None):
        self.model_directory = model_directory or self.model_directory
        slices_directory = slices_directory or os.path.join(self.model_directory, "ezkl", "slices")
        metadata_directory = metadata_directory or os.path.join(self.model_directory, "slices")

        segments = self._get_segments(metadata_directory)
        num_segments = len(segments)

        # Start timing
        start_time = time.time()
        segment_times = {}
        proof_paths = {}

        for segment_idx in range(num_segments):
            segment_start_time = time.time()

            segment_name = f"segment_{segment_idx}"
            segment_model_path = os.path.join(slices_directory, segment_name, f"{segment_name}_model.compiled")
            segment_witness_path = os.path.join(slices_directory, segment_name, f"{segment_name}_witness.json")
            segment_proof_path = os.path.join(slices_directory, segment_name, f"{segment_name}_proof.json")
            segment_pk_path = os.path.join(slices_directory, segment_name, f"{segment_name}_pk.key")

            if not os.path.exists(segment_model_path):
                print(f"ERROR: Model file not found at {segment_model_path}")
                continue

            if not os.path.exists(segment_witness_path):
                print(f"ERROR: Witness file not found at {segment_witness_path}")
                continue


            # Run EZKL proof generation command
            cmd = [
                "ezkl",
                "prove",
                "--compiled-circuit", segment_model_path,
                "--witness", segment_witness_path,
                "--proof-path", segment_proof_path,
                "--pk-path", segment_pk_path,
            ]

            try:
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # Check if proof file was created
                if os.path.exists(segment_proof_path):
                    # Get proof file size
                    proof_size = os.path.getsize(segment_proof_path)

                    # Try to load proof to check if valid
                    try:
                        with open(segment_proof_path, 'r') as f:
                            proof_data = json.load(f)
                        print(f"Loaded proof data with {len(proof_data)} keys")
                    except Exception as e:
                        print(f"Warning: Could not load proof file as JSON: {e}")
                else:
                    print(f"Warning: Proof file not found at {segment_proof_path}")

                # Store proof path and timing
                proof_paths[segment_idx] = segment_proof_path
                segment_time = time.time() - segment_start_time
                segment_times[segment_idx] = segment_time
                print(f"✓ Segment {segment_idx} proof generated in {segment_time:.2f}s")

            except subprocess.CalledProcessError as e:
                print(f"Error generating proof for segment {segment_idx}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Error code: {e.returncode}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                continue

        # Calculate total time
        total_time = time.time() - start_time

        # Return the results
        results = {
            "total_time": total_time,
            "segment_times": segment_times,
            "proof_paths": proof_paths,
            "num_segments_processed": len(segment_times)
        }

        print(f"✓ All segment proofs processed. Total proof generation time: {total_time:.2f}s")

        # Print timing summary
        if segment_times:
            print("\nTiming summary:")
            for segment_idx, time_taken in segment_times.items():
                print(f"  Segment {segment_idx}: {time_taken:.2f}s")

            avg_time = sum(segment_times.values()) / len(segment_times)
            print(f"  Average time per segment: {avg_time:.2f}s")

        return results

    def generate_witness_sliced(self, input_path: str = None) -> dict:
        input_path = input_path or os.path.join(self.model_directory, "input.json")
        metadata_dir = os.path.join(self.model_directory, "slices")

        slices_dir = os.path.join(self.model_directory, "ezkl", "slices")

        # Ensure metadata is loaded
        try:
            metadata = self._load_metadata(metadata_dir)
            num_segments = len(metadata['segments'])

            if num_segments == 0:
                print("No segments found to process!")
                return {"error": "No segments found"}

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {"error": str(e)}

        # Start timing
        start_time = time.time()
        segment_times = {}

        # Load initial input
        try:
            with open(input_path, 'r') as f:
                current_input = json.load(f)
        except Exception as e:
            print(f"Error loading input: {e}")
            return {"error": str(e)}

        # Process each segment
        witness_paths = {}

        for segment_idx in range(num_segments):
            segment_start_time = time.time()

            # Get segment model paths
            segment_name = f"segment_{segment_idx}"
            segment_model_name = f"{segment_name}_model.compiled"
            segment_model_path = os.path.join(slices_dir, segment_name, segment_model_name)

            # Create segment-specific input file
            segment_input_path = os.path.join(slices_dir, segment_name, f"{segment_name}_input.json")
            try:
                with open(segment_input_path, 'w') as f:
                    json.dump(current_input, f)

                # Print info about the size/structure of the input instead of full content
                if isinstance(current_input, dict):
                    print(f"Input file created with {len(current_input)} keys: {list(current_input.keys())}")
                elif isinstance(current_input, list):
                    print(f"Input file created with {len(current_input)} items")
                else:
                    print(f"Input file created with data of type: {type(current_input)}")
            except Exception as e:
                print(f"Error creating input file: {e}")
                continue

            # Create segment-specific witness output path
            segment_witness_path = os.path.join(slices_dir, segment_name, f"{segment_name}_witness.json")

            # Run EZKL witness generation command
            cmd = [
                "ezkl",
                "gen-witness",
                "--compiled-circuit", segment_model_path,
                "--data", segment_input_path,
                "--output", segment_witness_path,
            ]

            try:
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # If successful, prepare for next segment
                if segment_idx < num_segments - 1:
                    try:
                        with open(segment_witness_path, 'r') as f:
                            witness_data = json.load(f)
                    except Exception as e:
                        print(f"Error loading witness file: {e}")
                        continue

                    # Extract the output data to use as input for next segment
                    if 'outputs' in witness_data:
                        # Format the outputs as input for the next segment
                        output_data = [[float(val) for val in
                                       witness_data['pretty_elements']['rescaled_outputs'][0]]]
                        
                        current_input = {"input_data": output_data}

                        # Print summary information about the outputs
                        outputs = output_data
                    else:
                        print(f"WARNING: Witness file does not contain 'outputs' key")
                        print(f"Available keys: {list(witness_data.keys())}")
                        raise ValueError(f"Witness file for segment {segment_idx} does not contain outputs")

                # Store witness path and timing
                witness_paths[segment_idx] = segment_witness_path
                segment_time = time.time() - segment_start_time
                segment_times[segment_idx] = segment_time
                print(f"✓ Segment {segment_idx} witness generated in {segment_time:.2f}s")

            except subprocess.CalledProcessError as e:
                print(f"Error generating witness for segment {segment_idx}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Error: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                break

        # Calculate total time
        total_time = time.time() - start_time

        # Return the results
        results = {
            "total_time": total_time,
            "segment_times": segment_times,
            "witness_paths": witness_paths
        }

        print(f"✓ All segments processed. Total witness generation time: {total_time:.2f}s")
        print(f"Segment times: {segment_times}")

        return results

    def process_sliced_witness_output(self, sliced_witness_result):
        """
        Process the output from generate_witness_sliced to get prediction results.
        """
        witness_paths = sliced_witness_result.get('witness_paths', {})

        if not witness_paths:
            print("Error: No witness paths found in the sliced witness result")
            return None

        last_segment = max(witness_paths.keys())
        final_witness_path = witness_paths[last_segment]

        try:
            with open(final_witness_path, "r") as f:
                witness_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading witness file: {e}")
            return None

        return self.process_witness_output(witness_data)


# Example usage
if __name__ == "__main__":

    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net"
    }

    model_dir = base_paths[model_choice]
    model_runner = ModelRunner(model_directory=model_dir)

    if model_choice == 1:
        model_runner.predict()
        model_runner.predict(mode="sliced")
        model_runner.generate_witness()
        model_runner.generate_witness_sliced()
        model_runner.generate_proof()
        model_runner.generate_proof_sliced()

    elif model_choice == 2:
        model_runner.predict()
        model_runner.predict(mode="sliced")
        model_runner.generate_witness()
        model_runner.generate_witness_sliced()
        model_runner.generate_proof()
        model_runner.generate_proof_sliced()

