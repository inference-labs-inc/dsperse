import json
import os
import subprocess
import torch
from src.runners.utils.runner_utils import RunnerUtils

env = os.environ

class EzklRunner:
    def __init__(self):
        # ezkl_project_path = os.environ.get('EZKL_PATH')

        # check if ezkl is installed via cli
        try:
            result = subprocess.run(['ezkl', '--version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise Exception("EZKL CLI not found. Please install EZKL first.")
        except FileNotFoundError:
            raise Exception("EZKL CLI not found. Please install EZKL first.")


    def generate_witness(self, input_file: str, model_path: str, output_file: str, vk_path: str):
        # Validate required files exist
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vk_path):
            raise FileNotFoundError(f"Verification key file not found: {vk_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "gen-witness",
                    "--data", input_file,
                    "--compiled-circuit", model_path,
                    "--output", output_file,
                    "--vk-path", vk_path
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                error_msg = f"Witness generation failed with return code {process.returncode}"
                if process.stderr:
                    error_msg += f"\nError: {process.stderr}"
                return False, error_msg

        except subprocess.CalledProcessError as e:
            error_msg = f"Witness generation failed: {e}"
            if e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            return False, error_msg

        # return the processed outputs
        with open(output_file, "r") as f:
            witness_data = json.load(f)
            output = self.process_witness_output(witness_data)

        return True, output

    def prove(self):
        parent_dir = self.src_dir
        witness_path = os.path.join(self.base_path, "model", "witness.json")
        model_path = os.path.join(self.base_path, "model", "model.compiled")
        pk_path = os.path.join(self.base_path, "model", "pk.key")
        proof_path = os.path.join(self.base_path, "model", "proof.json")

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
            cwd=str(parent_dir),
            check=True
        )

        results = proof_path
        return results

    def verify(self) -> bool:
        parent_dir = self.src_dir
        settings_path = os.path.join(self.base_path, "model", "settings.json")
        vk_path = os.path.join(self.base_path, "model", "vk.key")
        proof_path = os.path.join(self.base_path, "model", "proof.json")

        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "verify",
                    "--proof-path", proof_path,
                    "--settings-path", settings_path,
                    "--vk-path", vk_path
                ],
                env=os.environ,  # Use os.environ for environment variables
                cwd=str(parent_dir),
                check=True,
                capture_output=True,
                text=True
            )

            # print("✓ Proof verified successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error verifying proof: {e}")
            return False



    def circuitize_model(self, model_path, output_path):
        # TODO: Implement this
        pass

    def circuitize_slices(self, model_path, layer_name, output_path):
        # TODO: Implement this
        pass

    @staticmethod
    def process_witness_output(witness_data):
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
        output = RunnerUtils.process_final_output(tensor_output)
        return output

if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net"
    }

    model_dir = base_paths[model_choice]
    runner = EzklRunner(model_dir)

    # run Test
    result = runner.generate_witness() # change function and mode when needed
    print(result)
