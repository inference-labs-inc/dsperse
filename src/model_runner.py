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
        result = self._process_final_output(tensor_output)

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
                # print(f"✓ Segment {segment_idx} witness generated in {segment_time:.2f}s")

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

