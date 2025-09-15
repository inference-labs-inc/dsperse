import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path
import pickle
import numpy as np
import json
import torch

import psutil

from kubz.runners.ezkl_runner import EzklRunner
from kubz.runners.jstprove_runner import JSTProveRunner
from kubz.runners.model_runner import ModelRunner
from kubz.runners.runner_utils import RunnerUtils


class ModelTester:
    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.model_runner = ModelRunner(model_directory)
        self.ezkl_runner = EzklRunner(model_directory)
        self.jstprove_runner = JSTProveRunner(model_directory)

    def test_model_accuracy(self, num_runs=10, input_path: str = None):
        """Run normal/ezkl inference on the sliced and whole and compare output from both."""
        # Check if input file path exists
        input_path = (
            input_path
            if input_path
            else os.path.join(self.model_directory, "input.json")
        )

        #  dict of {int run, final output from [original whole model (0-7), original sliced model(0-7), circuitized_whole_model(0-7), circuitized_sliced model(0-7)]}
        results = {}

        # get original input from path, find out its shape so we can generate more inputs
        generated_inputs_directory = os.path.join(
            self.model_directory, "generated_inputs"
        )
        os.makedirs(generated_inputs_directory, exist_ok=True)

        # for num_runs
        for i in range(num_runs):

            # run inference on an original whole model
            original_output = self.model_runner.infer(input_path=input_path)
            original_output = original_output["logits"]

            # run inference on an original sliced model
            sliced_output = self.model_runner.infer(
                input_path=input_path, mode="sliced"
            )
            sliced_output = sliced_output["logits"]

            # run generate_witness on a whole model
            witness_output = self.model_runner.generate_witness(input_file=input_path)
            circuitized_output = self.model_runner.process_witness_output(
                witness_output
            )
            circuitized_output = circuitized_output["logits"]

            # run generate_witness_sliced --> get the output from last witness.json
            # use helper method to fetch final result and run through a softmax
            sliced_witness_output = self.model_runner.generate_witness_sliced(
                input_path=input_path
            )
            sliced_witness_output = self.model_runner.process_sliced_witness_output(
                sliced_witness_output
            )
            sliced_circuitized_output = sliced_witness_output["logits"]

            # add to results
            results[i] = {
                "original_output": original_output,
                "sliced_output": sliced_output,
                "circuitized_output": circuitized_output,
                "sliced_circuitized_output": sliced_circuitized_output,
            }

            # mutate, or randomly generate new input for the next round
            input_path = self._generate_random_input_file(
                input_path, generated_inputs_directory
            )

        # For the results
        accuracies = {
            "original_vs_sliced": [],
            "original_vs_circuitized": [],
            "original_vs_sliced_circuitized": [],
            "circuitized_vs_sliced_circuitized": [],
        }

        for i, result in results.items():
            original_output = result["original_output"].detach().cpu().numpy()
            sliced_output = result["sliced_output"].detach().cpu().numpy()
            circuitized_output = result["circuitized_output"].detach().cpu().numpy()
            sliced_circuitized_output = (
                result["sliced_circuitized_output"].detach().cpu().numpy()
            )

            max_error = 1.0

            # Calculate accuracy as (1 - normalized_error) and convert to percentage
            accuracies["original_vs_sliced"].append(
                float(
                    100
                    * (1 - np.mean(np.abs(original_output - sliced_output)) / max_error)
                )
            )
            accuracies["original_vs_circuitized"].append(
                float(
                    100
                    * (
                        1
                        - np.mean(np.abs(original_output - circuitized_output))
                        / max_error
                    )
                )
            )
            accuracies["original_vs_sliced_circuitized"].append(
                float(
                    100
                    * (
                        1
                        - np.mean(np.abs(original_output - sliced_circuitized_output))
                        / max_error
                    )
                )
            )
            accuracies["circuitized_vs_sliced_circuitized"].append(
                float(
                    100
                    * (
                        1
                        - np.mean(
                            np.abs(circuitized_output - sliced_circuitized_output)
                        )
                        / max_error
                    )
                )
            )

        # Calculate the average accuracies over all runs
        average_accuracies = {
            key: max(0.0, min(100.0, float(np.mean(value))))
            for key, value in accuracies.items()
        }

        # Return the results
        return {"results": results, "accuracies": average_accuracies}

    def test_jst_prove(self, mode: str = "whole"):
        """
        Runs the doom_model circuit command and extracts performance metrics.

        Executes 'Python -m python_testing.circuit_models.doom_model' in the
        '../GravyTesting-Internal' directory relative to the project root.
        Parses the command output to find 'Peak Memory used Overall' and
        'Time elapsed' values.

        Returns:
            dict: A dictionary containing 'peak_memory' and 'time_elapsed' as floats.
                  Returns {'peak_memory': None, 'time_elapsed': None} if values
                  cannot be found or an error occurs.
        Raises:
            FileNotFoundError: If the '../GravyTesting-Internal' directory doesn't exist.
            subprocess.CalledProcessError: If the command execution fails.
            RuntimeError: If parsing fails to find the required metrics.
        """
        # Get the directory of the current script (src/)
        script_dir = Path(__file__).resolve().parent
        # Get the project root directory (kubz/) by going one level up
        project_root = script_dir.parent
        # Construct the absolute path to the target directory
        target_directory_path = (
            project_root.parent / "GravyTesting-Internal"
        )  # Go one more level up for GravyTesting-Internal

        # Determine the command based on the mode
        if mode == "whole":
            # module_name = "python_testing.circuit_models.doom_model"
            # Define the two commands for 'whole' mode
            compile_command = [
                "python",
                "cli.py",
                "--circuit",
                "doom_model",
                "--class",
                "Doom",
                "--compile",
                "--circuit_path",
                "doom_circuit.txt",
            ]
            witness_command = [
                "python",
                "cli.py",
                "--circuit",
                "doom_model",
                "--class",
                "Doom",
                "--gen_witness",
                "--input",
                "inputs/doom_input.json",
                "--output",
                "output/doom_output.json",
                "--circuit_path",
                "doom_circuit.txt",
            ]

        elif mode == "sliced":
            module_name = "python_testing.circuit_models.doom_slices"
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'whole' or 'sliced'.")

        command = ["Python", "-m", module_name]
        results = {"peak_memory": None, "time_elapsed": None}

        try:
            # Ensure the target directory exists before trying to run the command
            if not target_directory_path.is_dir():
                raise FileNotFoundError(
                    f"Calculated directory '{target_directory_path}' does not exist or is not a directory."
                )

            print(
                f"Running command: {' '.join(command)} in {target_directory_path} (mode: {mode})"
            )
            # Execute the command using the calculated absolute path
            process = subprocess.run(
                command,
                cwd=target_directory_path,  # Use the calculated absolute path
                capture_output=True,
                text=True,
                check=True,  # Raise an exception if the command fails
            )

            output = process.stdout
            # print("Command output:")
            # print(output) # Print output for debugging

            # Define regex patterns
            # memory in MB
            memory_pattern = r"Peak Memory used Overall : (\d+(\.\d+)?)"
            time_pattern = r"Time elapsed: (\d+(\.\d+)?) seconds"

            if mode == "whole":
                # Parse single values for 'whole' mode
                memory_match = re.search(memory_pattern, output)
                time_match = re.search(time_pattern, output)

                if memory_match:
                    results["peak_memory"] = float(memory_match.group(1))
                else:
                    print(
                        "Warning: Could not find 'Peak Memory used Overall' in output."
                    )

                if time_match:
                    results["time_elapsed"] = float(time_match.group(1))
                else:
                    print("Warning: Could not find 'Time elapsed' in output.")

            elif mode == "sliced":
                # Parse and aggregate multiple values for 'sliced' mode
                memory_matches = re.findall(memory_pattern, output)
                time_matches = re.findall(time_pattern, output)

                if memory_matches:
                    # Find the maximum peak memory across all slices
                    peak_memory_values = [float(match[0]) for match in memory_matches]
                    results["peak_memory"] = (
                        max(peak_memory_values) if peak_memory_values else None
                    )
                    print(
                        f"Found {len(peak_memory_values)} memory values. Max: {results['peak_memory']}"
                    )
                else:
                    print(
                        "Warning: Could not find any 'Peak Memory used Overall' in output for sliced mode."
                    )

                if time_matches:
                    # Store individual times and calculate the total time
                    time_elapsed_values = [float(match[0]) for match in time_matches]
                    results["slice_times"] = time_elapsed_values
                    results["total_time_elapsed"] = (
                        sum(time_elapsed_values) if time_elapsed_values else None
                    )
                    print(
                        f"Found {len(time_elapsed_values)} time values. Individual: {results['slice_times']}. Total: {results['total_time_elapsed']}"
                    )
                else:
                    print(
                        "Warning: Could not find any 'Time elapsed' in output for sliced mode."
                    )
                    results["slice_times"] = []

        except FileNotFoundError as e:
            print(f"Error: Directory not found. {e}")
            raise  # Re-raise the exception
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            raise  # Re-raise the exception
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise  # Re-raise the exception

        print(f"Extracted results (mode: {mode}): {results}")
        return results

    def test_deep_prove(
        self,
        deep_prove_path: Path,
        verbose=False,
        num_samples: int = 10,
        output_csv_path: str = None,
        onnx_model_path: str = None,
        input_path: str = None,
    ):
        """
        Run DeepProve benchmark tool on an ONNX model.

        Args:
            input_path (str or Path): Path to the input JSON file
            onnx_model_path (str or Path): Path to the ONNX model file
            output_csv_path (str or Path): Path where to save benchmark results
            num_samples (int): Number of samples to process
            verbose (bool): Whether to enable verbose logging

        Returns:
            Path: Path to the output CSV file with benchmark results
        """
        csv_name = "deepprove_benchmark_results.csv"
        input_path = (
            input_path
            if input_path
            else os.path.join(self.model_directory, "deepprove", "input.json")
        )
        onnx_model_path = (
            onnx_model_path
            if onnx_model_path
            else os.path.join(self.model_directory, "deepprove", "model.onnx")
        )
        output_csv_path = (
            output_csv_path
            if output_csv_path
            else os.path.join(self.model_directory, "deepprove", csv_name)
        )

        # TODO: Ensure output/deep prove directory exists
        # output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Base command
        cmd = [
            "cargo",
            "run",
            "--release",
            "--",
            "-i",
            str(Path(input_path).resolve()),
            "-o",
            str(Path(onnx_model_path).resolve()),
            "-b",
            str(Path(output_csv_path).resolve()),
            "-n",
            str(num_samples),
        ]

        # Set up environment variables for verbose output if needed
        env = os.environ.copy()
        if verbose:
            env["RUST_LOG"] = "trace"
            env["RUST_BACKTRACE"] = "1"

        # Print command if verbose
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
            print(
                f"With environment: RUST_LOG={env.get('RUST_LOG', '')}, RUST_BACKTRACE={env.get('RUST_BACKTRACE', '')}"
            )

        # Run the command
        print(f"Running DeepProve benchmark...")
        try:
            # Check if the deep-prove directory exists
            if not deep_prove_path.exists():
                raise FileNotFoundError(
                    f"DeepProve directory not found at {deep_prove_path}"
                )

            result = subprocess.run(
                cmd,
                cwd=str(deep_prove_path),
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            if verbose:
                print("Command output:")
                print(result.stdout)
            print("DeepProve benchmark completed successfully")
            return output_csv_path
        except subprocess.CalledProcessError as e:
            print(f"Error running DeepProve benchmark: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Command error: {e.stderr}")
            raise RuntimeError("DeepProve benchmark failed") from e

    def _generate_random_input_file(self, input_file, save_path):
        """Helper method to generate random values in the input file"""
        # Generate random data with the specified shape
        if "doom" in self.model_directory.lower():
            dummy_input_shape = (1, 4, 28, 28)
        elif "net" in self.model_directory.lower():
            dummy_input_shape = (1, 3, 32, 32)
        else:
            raise ValueError(
                "Unknown input file shape. Please specify the shape manually in the code. "
            )

        random_data = np.random.rand(*dummy_input_shape)

        # Normalize to the range [0, 1] (similar to the example data)
        random_data = random_data.astype(np.float32)

        # Flatten the data to match the expected format
        flattened_data = random_data.flatten().tolist()

        # Create the JSON structure
        input_json = {"input_data": [flattened_data]}

        # Save to the specified path
        output_path = os.path.join(save_path, os.path.basename(input_file))
        with open(output_path, "w") as f:
            json.dump(input_json, f)

        return output_path

    @staticmethod
    def generate_multiple_random_inputs(
        model_directory, num_inputs, save_path, output_filename="input.json"
    ):
        """Generates multiple random inputs and saves them to a single JSON file."""

        # Determine the input shape based on the model directory
        if "doom" in model_directory.lower():
            dummy_input_shape = (1, 4, 28, 28)
        elif "net" in model_directory.lower():
            dummy_input_shape = (1, 3, 32, 32)
        else:
            raise ValueError(
                "Unknown input file shape. Please specify the shape manually or update the logic."
            )

        all_inputs = []
        for _ in range(num_inputs):
            # Generate random data with the specified shape
            random_data = np.random.rand(*dummy_input_shape)
            # Normalize to the range [0, 1] and set type to float32
            random_data = random_data.astype(np.float32)
            # Flatten the data
            flattened_data = random_data.flatten().tolist()
            all_inputs.append(flattened_data)

        # Create the final JSON structure
        output_json = {"input_data": all_inputs}

        # Save to the specified output file
        # Ensure the output directory exists if output_filename includes a path
        output_dir = os.path.join(model_dir, save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(Path(output_dir, output_filename), "w") as f:
            json.dump(output_json, f)

        print(
            f"Successfully generated {num_inputs} inputs and saved to {output_filename}"
        )
        return output_filename

    def run_inference_test(self, input_tensor, mode: str = None):
        """Run inferences and track performance metrics for each input"""

        # Get current process
        process = psutil.Process()

        input_tensor = RunnerUtils.reshape(input_tensor, self.model_directory)

        # Start memory tracker
        start_mem = process.memory_info().rss / 1024 / 1024  # Memory in MB

        # Start timer
        start_time = time.time()

        # Run inference
        if mode == "sliced":
            inference_result = self.model_runner.run_layered_inference(
                input_tensor=input_tensor
            )
        else:
            inference_result = self.model_runner.run_inference(
                input_tensor=input_tensor
            )

        # Stop timer
        end_time = time.time()

        # Get peak memory
        end_mem = process.memory_info().rss / 1024 / 1024
        mem_used = end_mem - start_mem

        print(f"Memory used: {mem_used} MB")
        print(f"Time elapsed: {end_time - start_time} seconds")
        print(f"Output: {inference_result}")

        # Record results
        result = {
            "time_seconds": end_time - start_time,
            "memory_mb": mem_used,
            "output": inference_result["logits"].detach().cpu().numpy().tolist(),
        }

        return result

    def run_ezkl_test(self, input_file: str, mode: str = None):
        """Run EZKL witness test"""
        testing_dir = Path(self.model_directory, "testing")
        with open(Path(self.model_directory, "testing", input_file), "r") as f:
            input_data = json.load(f)

        input_array = np.array(input_data["input_data"])
        input_tensor = torch.from_numpy(input_array)
        input_tensor = RunnerUtils.reshape(input_tensor, self.model_directory)

        torch_input_file = Path(testing_dir, "test_run.json")
        RunnerUtils.save_to_file_flattened(
            input_tensor=input_tensor, file_path=str(torch_input_file)
        )

        results = {}

        # Run inference
        if mode == "sliced":
            inference_result = self.ezkl_runner.generate_witness(
                mode="sliced", input_file=str(torch_input_file)
            )
            proof_result = self.ezkl_runner.prove(mode="sliced")
            verification_result = self.ezkl_runner.verify(mode="sliced")
        else:
            inference_result = self.ezkl_runner.generate_witness(
                input_file=str(torch_input_file)
            )
            proof_result = self.ezkl_runner.prove()
            verification_result = self.ezkl_runner.verify()

        total_time = inference_result["total_time"]

        # Record results
        results["witness"] = {
            "memory_mb": inference_result["memory"],
            "total_time": total_time,
            "layer_times": inference_result.get("segment_times", "N/A"),
            "output": inference_result["result"]["logits"]
            .detach()
            .cpu()
            .numpy()
            .tolist(),
        }

        results["proof"] = {
            "memory_mb": proof_result["memory"],
            "total_time": proof_result["total_time"],
            "layer_times": proof_result.get("segment_times", "N/A"),
        }

        results["verification"] = {
            "memory_mb": verification_result["memory"],
            "total_time": verification_result["total_time"],
            "layer_times": verification_result.get("segment_times", "N/A"),
        }

        return results

    def run_jstprove_test(self, input_file: str, mode: str = None):
        testing_dir = Path(self.model_directory, "testing")
        file_path = Path(testing_dir, input_file)
        with open(Path(self.model_directory, "testing", input_file), "r") as f:
            input_data = json.load(f)

        input_array = np.array(input_data["input_data"])
        input_tensor = torch.from_numpy(input_array)
        input_tensor = RunnerUtils.reshape(input_tensor, self.model_directory)

        torch_input_file = Path(testing_dir, "test_run.json")
        RunnerUtils.save_to_file_flattened(
            input_tensor=input_tensor, file_path=str(torch_input_file)
        )

        results = {}

        # Run inference
        if mode == "sliced":
            inference_result = self.jstprove_runner.generate_witness(
                mode="sliced", input_file=str(torch_input_file.absolute())
            )
            proof_result = self.jstprove_runner.prove(mode="sliced")
            verification_result = self.jstprove_runner.verify(
                mode="sliced", input_file=str(torch_input_file.absolute())
            )
        else:
            inference_result = self.jstprove_runner.generate_witness(
                input_file=str(torch_input_file.absolute())
            )
            proof_result = self.jstprove_runner.prove()
            verification_result = self.jstprove_runner.verify(
                input_file=str(torch_input_file.absolute())
            )

        # Record results
        results["witness"] = {
            "memory_mb": inference_result["memory"],
            "total_time": inference_result["total_time"],
            "layer_times": inference_result.get("segment_times", "N/A"),
            "output": inference_result["result"],
        }

        results["proof"] = {
            "memory_mb": proof_result["memory"],
            "total_time": proof_result["total_time"],
            "layer_times": proof_result.get("segment_times", "N/A"),
        }

        results["verification"] = {
            "memory_mb": verification_result["memory"],
            "total_time": verification_result["total_time"],
            "layer_times": verification_result.get("segment_times", "N/A"),
        }

        return results

    @staticmethod
    def get_cifar_data(input_file):
        """Loads and unpickles CIFAR dataset from input file.

        Args:
            input_file (str): Path to the CIFAR pickle file.

        Returns:
            dict: Dictionary containing 'data' (10000x3072 numpy array) and 'labels' (list of 10000 numbers)
        """
        try:
            # Unpickle data file
            with open(input_file, "rb") as fo:
                data_dict = pickle.load(fo, encoding="bytes")
            return data_dict
        except FileNotFoundError:
            print(
                f"Error: Input file '{input_file}' not found. Please make sure you have the CIFAR dataset and are pointing to the correct file."
            )
            return None

    def test_all(self, num_runs=10, cifar_data_file: str = None):
        """Run all test variations"""

        # data prep
        cifar_data_file = cifar_data_file or os.path.join(
            self.model_directory, "testing", "cifar-10-batches-py", "data_batch_1"
        )
        cifar_data = self.get_cifar_data(cifar_data_file)

        # Ensure testing directory exists
        testing_dir = os.path.join(self.model_directory, "testing")
        os.makedirs(testing_dir, exist_ok=True)
        results_file = os.path.join(testing_dir, "results.json")

        # Initialize results structure
        results = {}

        # for each num run
        for i in range(num_runs):
            # get random cifir image
            random_index = random.randint(0, len(cifar_data[b"data"]) - 1)
            random_input = np.frombuffer(
                cifar_data[b"data"][random_index], dtype=np.uint8
            )
            true_label = cifar_data[b"labels"][
                random_index
            ]  # get the correct answer for this image
            normalized_input = (
                random_input.astype(np.float32) / 127.5
            ) - 1.0  # make bytes into floats

            input_tensor = torch.from_numpy(normalized_input).float().unsqueeze(0)

            RunnerUtils.save_to_file_flattened(
                input_tensor=input_tensor,
                file_path=os.path.join(testing_dir, "random_input.json"),
            )

            # run .pth
            pytorch_results = self.run_inference_test(input_tensor, mode="whole")
            sliced_pytorch_results = self.run_inference_test(
                input_tensor, mode="sliced"
            )
            original_pytorch_results = {
                "whole": pytorch_results,
                "sliced": sliced_pytorch_results,
            }

            # run jstprove
            jstprove_whole_results = self.run_jstprove_test(
                "random_input.json", mode="whole"
            )
            jstprove_sliced_results = self.run_jstprove_test(
                "random_input.json", mode="sliced"
            )
            jstprove_results = {
                "whole": jstprove_whole_results,
                "sliced": jstprove_sliced_results,
            }

            # run ezkl
            ezkl_whole_results = self.run_ezkl_test("random_input.json", mode="whole")
            ezkl_sliced_results = self.run_ezkl_test("random_input.json", mode="sliced")
            ezkl_results = {"whole": ezkl_whole_results, "sliced": ezkl_sliced_results}

            # add data to dict (read and add/dump)
            results[i] = {
                "input_data": str(input_tensor.tolist()),
                "input_data_label": true_label,
                "original_pytorch_results": original_pytorch_results,
                "jstprove_results": jstprove_results,
                "ezkl_results": ezkl_results,
            }

            # save results file (overwrite)
            with open(results_file, "w") as f:
                json.dump(results, f)

        # close things?


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {1: "models/doom", 2: "models/net"}

    model_dir = base_paths[model_choice]
    model_tester = ModelTester(model_dir)

    # TODO: modify this to install path. Maybe env var?
    deep_prove_zkml_path = Path("/Volumes/SSD/projects/deep-prove/zkml")

    # TODO: Add test for ezkl time and memory
    if model_choice == 1:
        # model_tester.run_inference_test("10k_inputs.json", "10k_outputs_sliced.json", mode="sliced")
        # ModelTester.generate_multiple_random_inputs(model_dir, 100, "testing", "100_inputs.json")
        # accuracy = model_tester.test_model_accuracy(num_runs=10)
        # print(f"Model accuracy: {accuracy}")
        # result = model_tester.test_jst_prove()
        result = model_tester.test_jst_prove(mode="sliced")
        # result = model_tester.test_deep_prove(deep_prove_path=deep_prove_zkml_path, verbose=True)
        print(f"Model deep prove result: {result}")

    elif model_choice == 2:
        start_time = time.time()
        # ModelTester.generate_multiple_random_inputs(model_dir, 1, "testing", "1_inputs.json")
        # result = model_tester.run_inference_test("input_data.json", "output_sliced.json", mode="sliced")
        # result = model_tester.run_ezkl_test("1_inputs.json", "1_outputs.json", mode="whole")
        # result = model_tester.run_ezkl_test("100_inputs.json", "100_outputs_sliced.json", mode="sliced")
        model_tester.test_all()
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
        # print(f"sliced: {result}")
