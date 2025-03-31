import math
import subprocess
import time
import tracemalloc

import torch
import torch.nn.functional as F
from models.doom.model import DoomAgent  # Import your model class
import os
import json
from pathlib import Path
from utils.model_utils import ModelUtils

env = os.environ

class ModelRunner:
    def __init__(self):
        # Always use CPU
        self.device = torch.device("cpu")

        # Initialize attributes
        self.model = None
        self.model_path = None
        self.is_sliced_model = False

    def load_model(self, model_path):
        """
        Load a PyTorch model from a file.

        Parameters:
            model_path (str): Path to the model file (.pth)
        """
        try:
            self.model_path = model_path

            # Check if model_path is a directory (sliced model) or a file (single model)
            if os.path.isdir(model_path):
                print(f"Loading sliced model from directory: {model_path}")
                self.is_sliced_model = True
                self.model = None  # We'll load segments on demand during layered inference
            else:
                # For single file, load the checkpoint
                print(f"Loading single model from file: {model_path}")
                self.is_sliced_model = False
                checkpoint = torch.load(model_path, map_location=self.device)
                print(f"Loaded model type: {type(checkpoint)}")

                if isinstance(checkpoint, dict):
                    print(f"Dictionary keys: {checkpoint.keys()}")

                self.model = checkpoint
                print(f"Successfully loaded model from {model_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def run_inference(self, input_tensor):
        """
        Run inference with the model and apply softmax to get probability distribution.

        Parameters:
            input_tensor (torch.Tensor): Input tensor for the model

        Returns:
            dict: Dictionary containing raw logits, softmax probabilities, and predicted action
        """
        try:
            print(f"Input tensor shape: {input_tensor.shape}")

            if not self.is_sliced_model:
                # Handle regular model case
                # If the input is flattened (2D) with 3136 elements, reshape it for CNN
                if input_tensor.dim() == 2 and input_tensor.size(1) == 3136:
                    input_tensor = input_tensor.reshape(1, 4, 28, 28)
                    print(f"Reshaped input to: {input_tensor.shape}")

                # Create and initialize model from state dict
                checkpoint = self.model

                # Create a proper model instance
                model = DoomAgent(n_actions=7)  # Adjust parameters as needed

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
            else:
                raise ValueError("Sliced models require the run_layered_inference method.")

        # Apply softmax to the raw output to get probabilities

            result = self._process_final_output(raw_output)
            # probabilities = F.softmax(raw_output, dim=1)
            #
            # # Get the predicted action
            # predicted_action = torch.argmax(probabilities, dim=1).item()
            #
            # # Create the result dictionary
            # result = {
            #     "logits": raw_output,
            #     "probabilities": probabilities,
            #     "predicted_action": predicted_action
            # }

            # Print the results in a more readable format
            # self._print_inference_results(result)

            return result

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_layered_inference(self, input_tensor, folder_path: str):
        print(f"Running layered inference with input shape: {input_tensor.shape}")

        try:
            metadata = self._load_metadata(folder_path)
            if metadata is None:
                return None

            model_details = self._get_model_details(metadata)
            if model_details is None:
                return None

            model_type, total_params, slicing_strategy, input_shape = model_details

            segments = metadata.get('segments', [])
            if not segments:
                print("No segments found in metadata.json")
                return None

            num_segments = len(segments)
            print(f"Found {num_segments} segments in metadata.json")

            # # reshape the initial input using metadata input_shape
            x = input_tensor.to(self.device)
            x = self._prepare_initial_input(x, segments, input_shape)

            # Continue processing segments
            for i, segment in enumerate(segments):
                print(f"\nProcessing segment {i + 1}/{num_segments}")
                x = self._process_segment(segment, x, i, folder_path, num_segments)
                if x is None:
                    return None

                print(f"Output tensor shape after segment {i + 1}: {x.shape}")

            return self._process_final_output(x, num_segments)

        except Exception as e:
            print(f"Error occurred in layered inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_metadata(self, folder_path):
        """Load the model metadata from the metadata.json file."""
        metadata_path = Path(folder_path) / "metadata.json"
        if not metadata_path.exists():
            print(f"Required metadata.json file not found at: {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata

    def _get_model_details(self, metadata):
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

    def _prepare_initial_input(self, x, segments, input_shape):
        """Temporary fix explicitly for original input_shape=[3136] flattened (4,28,28)."""
        #TODO: hard coded for now, but will work on not hardcoded version
        if len(input_shape) == 1 and input_shape[0] == 3136:
            # Explicit temporary reshape clearly known from your original model definition
            channels, height, width = 4, 28, 28  # explicitly correct based on model architecture
            x = x.reshape(-1, channels, height, width).to(self.device)
            print(f"Explicit temporary correct reshape successful: {x.shape}")
        else:
            raise ValueError(f"Explicitly provided input_shape metadata seems incorrect or incomplete: {input_shape}")
        return x

    def _get_segment_path(self, segment, folder_path):
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

    def _process_segment(self, segment, x, segment_idx, folder_path, num_segments):
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

    def _apply_segment_reshaping(self, x, segment):
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
            x = self._match_input_dimensions(x, weight, layer_info)

            # Apply linear transformation
            print(f"  Layer {layer_name}: input={x.shape}, weights={weight.shape}")
            x = F.linear(x, weight, bias)

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

    def _process_final_output(self, x, num_segments=None):
        """Process the final output of the model."""

        # print(f"\nProcessed all {num_segments} segments")

        # Verify and print the final output
        raw_output = x
        print(f"Final output shape: {raw_output.shape}")

        # Apply softmax to get probabilities if not already applied
        if len(raw_output.shape) == 2:  # [batch_size, num_classes]
            probabilities = F.softmax(raw_output, dim=1)
            predicted_action = torch.argmax(probabilities, dim=1).item()
            print(f"Predicted action: {predicted_action}")

            result = {
                "logits": raw_output,
                "probabilities": probabilities,
                "predicted_action": predicted_action
            }

            # Print the results in a more readable format
            # self._print_inference_results(result)

            return result

        else:
            print(
                f"Raw output shape {raw_output.shape} is not as expected for classification ([batch_size, num_classes])")
            return {
                "output": raw_output
            }

    def _print_inference_results(self, result):
        """
        Print inference results in a readable format.

        Parameters:
            result (dict): The inference results dictionary
        """
        import numpy as np

        logits = result["logits"].cpu().numpy()[0]
        probs = result["probabilities"].cpu().numpy()[0]
        action = result["predicted_action"]

        print("\n----- Inference Results -----")
        print(f"Predicted Action: {action}")
        print("\nProbabilities:")

        # Define action labels if known (customize as needed)
        action_labels = ["Move Left", "Move Right", "Attack", "Move Forward",
                         "Move Backward", "Turn Left", "Turn Right"]

        # Print probabilities as percentages
        print(f"{'Action':<15} {'Probability':<15} {'Logit':<15}")
        print("-" * 45)

        for i, (prob, logit) in enumerate(zip(probs, logits)):
            label = action_labels[i] if i < len(action_labels) else f"Action {i}"
            prob_percent = f"{prob * 100:.4f}%"

            # Highlight the highest probability
            if i == action:
                print(f"{label:<15} {prob_percent:<15} {logit:<15.4f} <- SELECTED")
            else:
                print(f"{label:<15} {prob_percent:<15} {logit:<15.4f}")

        print("-----------------------------\n")

    def predict(self, input_path: str = None, model_path: str = None, input_tensor=None) -> dict:
        self.load_model(model_path)

        if input_tensor is None:
            input_tensor = ModelUtils.preprocess_input(input_path)

        if os.path.isdir(model_path): # if self.is_sliced_model
            result = self.run_layered_inference(input_tensor, model_path)
        else:
            result = self.run_inference(input_tensor)

        return result

    def generate_witness(self, circuit_path: str = None, input_file: str = None, model_path: str = None) -> dict:
        circuit_path = circuit_path or "models/doom/circuit/"
        input_file = input_file or os.path.join(circuit_path, "calibration.json")
        model_path = model_path or os.path.join(circuit_path, "model.compiled")
        witness_path = os.path.join(circuit_path, "witness.json")
        vk_path = os.path.join(circuit_path, "vk.key")

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

        with open(witness_path, "r") as f:
            witness_data = json.load(f)
            # outputs = witness_data["pretty_elements"]["rescaled_outputs"][0]
            outputs = witness_data
            # print(outputs)
        return outputs

    def process_witness_output(self, witness_data):
        """
        Process the witness.json data to get prediction results.

        Parameters:
            witness_data (dict): The complete witness.json data

        Returns:
            dict: Prediction results with logits, probabilities, and predicted_action
        """
        # Extract the rescaled outputs from the witness data
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

        # Print detailed results
        # self._print_inference_results(result)

        return result

    def generate_proof(self):
        circuit_path = "models/doom/circuit/"
        witness_path = os.path.join(circuit_path, "witness.json")
        model_path = os.path.join(circuit_path, "model.compiled")
        pk_path = os.path.join(circuit_path, "pk.key")
        proof_path = os.path.join(circuit_path, "proof.json")

        proof_time_start = time.time()
        tracemalloc.start()

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
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Proving time: {proof_time_end - proof_time_start:.2f} seconds")

        with open(proof_path, "r") as f:
            proof_data = json.load(f)
            outputs = proof_data["pretty_public_inputs"]["rescaled_outputs"][0]

        proof_time = time.time() - proof_time_start
        print(f"Proving time: {proof_time:.2f} seconds")

        return outputs, {"proof_time": proof_time}

    def generate_proof_sliced(self):
        print("Starting generate_proofs_sliced...")

        # Get output path for the model
        output_path = "models/doom/output/"
        circuitized_slices_path = os.path.join(output_path, "circuitized_slices/")

        print(f"Output path: {output_path}")
        print(f"Circuitized slices path: {circuitized_slices_path}")

        # Ensure metadata is loaded
        try:
            metadata = self._load_metadata(output_path)

            # Get number of segments from the 'segments' array length if it exists
            if 'segments' in metadata and isinstance(metadata['segments'], list):
                num_segments = len(metadata['segments'])
                print(f"Found {num_segments} segments in metadata['segments']")
            else:
                # Fallback: look for files with segment pattern
                if os.path.exists(circuitized_slices_path):
                    segment_files = [f for f in os.listdir(circuitized_slices_path)
                                     if f.startswith('segment_') and f.endswith('_witness.json')]
                    num_segments = len(set(int(f.split('_')[1]) for f in segment_files if f.split('_')[1].isdigit()))
                    print(f"Determined {num_segments} segments by counting witness files in directory")
                else:
                    num_segments = 0
                    print("WARNING: Could not determine number of segments")

            print(f"Number of segments to process: {num_segments}")

            if num_segments == 0:
                print("No segments found to process!")
                return {"error": "No segments found"}

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {"error": str(e)}

        # Start timing
        start_time = time.time()
        tracemalloc.start()

        segment_times = {}
        proof_paths = {}

        print(f"Starting to process proofs for {num_segments} segments...")

        for segment_idx in range(num_segments):
            print(f"\n--- PROCESSING PROOF FOR SEGMENT {segment_idx} ---")
            segment_start_time = time.time()

            # Get segment file paths
            segment_name = f"segment_{segment_idx}"
            segment_model_path = os.path.join(circuitized_slices_path, f"{segment_name}_model.compiled")
            segment_witness_path = os.path.join(circuitized_slices_path, f"{segment_name}_witness.json")
            segment_proof_path = os.path.join(circuitized_slices_path, f"{segment_name}_proof.json")
            segment_pk_path = os.path.join(circuitized_slices_path, f"{segment_name}_pk.key")

            # Check if all required files exist
            print(f"Checking required files for segment {segment_idx}:")
            print(f"  Model exists: {os.path.exists(segment_model_path)}")
            print(f"  Witness exists: {os.path.exists(segment_witness_path)}")

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
                "--pk-path", segment_pk_path
            ]

            print(f"Running command: {' '.join(cmd)}")

            try:
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Command succeeded")

                # Check if proof file was created
                if os.path.exists(segment_proof_path):
                    # Get proof file size
                    proof_size = os.path.getsize(segment_proof_path)
                    print(f"Proof file created, size: {proof_size} bytes")

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
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Return the results
        results = {
            "total_time": total_time,
            "segment_times": segment_times,
            "proof_paths": proof_paths,
            "num_segments_processed": len(segment_times),
            # "memory_usage_bytes": peak_memory

        }

        print(f"✓ All segment proofs processed. Total proof generation time: {total_time:.2f}s")
        print(f"Successfully generated proofs for {len(segment_times)} of {num_segments} segments")

        # Print timing summary
        if segment_times:
            print("\nTiming summary:")
            for segment_idx, time_taken in segment_times.items():
                print(f"  Segment {segment_idx}: {time_taken:.2f}s")

            avg_time = sum(segment_times.values()) / len(segment_times)
            print(f"  Average time per segment: {avg_time:.2f}s")

        return results

    def generate_witness_sliced(self, input_path: str = None, model_path: str = None) -> dict:

        print("Starting generate_witness_sliced...")

        input_path = input_path or "models/doom/input/input.json"
        # Get output path for the model
        output_path = "models/doom/output/"
        circuitized_slices_path = os.path.join(output_path, "circuitized_slices/")

        print(f"Output path: {output_path}")
        print(f"Circuitized slices path: {circuitized_slices_path}")
        print(f"Checking if directory exists: {os.path.exists(circuitized_slices_path)}")

        # Debug directory contents
        if os.path.exists(circuitized_slices_path):
            print(f"Directory contents: {os.listdir(circuitized_slices_path)}")

        # Ensure metadata is loaded
        try:
            metadata = self._load_metadata(output_path)
            print(f"Loaded metadata: {metadata}")

            # Get number of segments from the 'segments' array length if it exists
            if 'segments' in metadata and isinstance(metadata['segments'], list):
                num_segments = len(metadata['segments'])
                print(f"Found {num_segments} segments in metadata['segments']")
            else:
                # Fallback: look for files with segment pattern
                if os.path.exists(circuitized_slices_path):
                    segment_files = [f for f in os.listdir(circuitized_slices_path)
                                     if f.startswith('segment_') and f.endswith('_model.compiled')]
                    num_segments = len(set(int(f.split('_')[1]) for f in segment_files if f.split('_')[1].isdigit()))
                    print(f"Determined {num_segments} segments by counting files in directory")
                else:
                    num_segments = 0
                    print("WARNING: Could not determine number of segments")

            print(f"Number of segments to process: {num_segments}")

            if num_segments == 0:
                print("No segments found to process!")
                return {"error": "No segments found"}

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {"error": str(e)}

        # Start timing
        start_time = time.time()
        tracemalloc.start()
        segment_times = {}

        # Load initial input
        try:
            print(f"Loading input from: {input_path}")
            with open(input_path, 'r') as f:
                current_input = json.load(f)
            print(f"Loaded input: {current_input}")
        except Exception as e:
            print(f"Error loading input: {e}")
            return {"error": str(e)}

        # Process each segment
        witness_paths = {}

        print(f"Starting to process {num_segments} segments...")

        for segment_idx in range(num_segments):
            print(f"\n--- PROCESSING SEGMENT {segment_idx} ---")
            segment_start_time = time.time()

            # Get segment model paths
            segment_name = f"segment_{segment_idx}"
            segment_model_name = f"{segment_name}_model.compiled"
            segment_model_path = os.path.join(circuitized_slices_path, segment_model_name)

            print(f"Segment model path: {segment_model_path}")
            print(f"Checking if model exists: {os.path.exists(segment_model_path)}")

            # Create segment-specific input file
            segment_input_path = os.path.join(circuitized_slices_path, f"{segment_name}_input.json")
            print(f"Creating input file at: {segment_input_path}")
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
            segment_witness_path = os.path.join(circuitized_slices_path, f"{segment_name}_witness.json")
            print(f"Witness output path: {segment_witness_path}")

            # Get settings file for this segment
            settings_path = os.path.join(os.path.dirname(circuitized_slices_path), f"{segment_name}_settings.json")
            print(f"Settings path: {settings_path}")
            print(f"Settings file exists: {os.path.exists(settings_path)}")

            # Run EZKL witness generation command
            cmd = [
                "ezkl",
                "gen-witness",
                "--compiled-circuit", segment_model_path,
                "--data", segment_input_path,
                "--output", segment_witness_path,
                # "--settings", settings_path
            ]

            print(f"Running command: {' '.join(cmd)}")

            try:
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Command output: {process.stdout}")

                # If successful, prepare for next segment
                if segment_idx < num_segments - 1:
                    print(f"Processing output for next segment...")
                    # Read the witness file to extract the output
                    try:
                        with open(segment_witness_path, 'r') as f:
                            witness_data = json.load(f)
                        print(f"Loaded witness data with {len(witness_data)} keys: {list(witness_data.keys())}")
                    except Exception as e:
                        print(f"Error loading witness file: {e}")
                        continue

                    # Extract the output data to use as input for next segment
                    if 'outputs' in witness_data:
                        # Format the outputs as input for the next segment
                        current_input = {"input_data": witness_data['outputs']}

                        # Print summary information about the outputs
                        outputs = witness_data['outputs']
                        if isinstance(outputs, list):
                            print(f"Prepared input for next segment: list with {len(outputs)} elements")
                            if outputs and isinstance(outputs[0], list):
                                print(f"  First element is a list with {len(outputs[0])} items")
                        elif isinstance(outputs, dict):
                            print(
                                f"Prepared input for next segment: dict with {len(outputs)} keys: {list(outputs.keys())}")
                        else:
                            print(f"Prepared input for next segment: data of type {type(outputs)}")
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
                continue

        # Calculate total time
        total_time = time.time() - start_time
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Return the results
        results = {
            "total_time": total_time,
            "segment_times": segment_times,
            "witness_paths": witness_paths,
            "memory_usage_bytes": peak_memory
        }

        print(f"✓ All segments processed. Total witness generation time: {total_time:.2f}s")
        print(f"Segment times: {segment_times}")

        return results

    def process_sliced_witness_output(self, sliced_witness_result):
        """
        Process the output from generate_witness_sliced to get prediction results.

        This function takes the result from generate_witness_sliced, loads the final
        segment's witness file, and processes it to get the prediction.

        Parameters:
            sliced_witness_result (dict): The result from generate_witness_sliced

        Returns:
            dict: Prediction results with logits, probabilities, and predicted_action
        """
        # Get the witness paths from the result
        witness_paths = sliced_witness_result.get('witness_paths', {})

        if not witness_paths:
            print("Error: No witness paths found in the sliced witness result")
            return None

        # Get the last segment number and its witness path
        last_segment = max(witness_paths.keys())
        final_witness_path = witness_paths[last_segment]

        print(f"Processing final segment {last_segment}, witness file: {final_witness_path}")

        # Load the witness file
        try:
            with open(final_witness_path, "r") as f:
                witness_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading witness file: {e}")
            return None

        # Use the existing function to process the witness data
        return self.process_witness_output(witness_data)


# Example usage
if __name__ == "__main__":
    model_path_full = "models/doom/doom.pth"
    model_path_sliced = "models/doom/output/"
    input_path = "models/doom/input/input.json"
    model_runner = ModelRunner()

    # run regular inference Sliced and Whole
    # output_full = model_runner.predict(input_path, model_path_full)
    # output_sliced = model_runner.predict(input_path, model_path_sliced)
    # print(f"sliced output: {output_sliced}")
    # print(f"output_full: {output_full}")

    # # run ezkl inference Sliced and Whole
    # witness = model_runner.generate_witness()
    sliced_witness = model_runner.generate_witness_sliced()
    # print(f"circuitized model: {model_runner.process_witness_output(witness)}")
    print(f"sliced circuitized model: {model_runner.process_sliced_witness_output(sliced_witness)}")
    #
    # # # run proof Sliced and Whole
    # print(model_runner.generate_proof())
    # print(model_runner.generate_proof_sliced())

