import json
import os
import subprocess
import threading
import time
from typing import Any

import torch
from pathlib import Path

from torch import Tensor

from src.runners import runner_utils
from src.runners.runner_utils import RunnerUtils
from src.utils.model_utils import ModelUtils

env = os.environ


class EzklRunner:
    def __init__(self, model_directory: str):
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

        self.model_directory = model_directory
        self.base_path = os.path.join(model_directory, "ezkl")
        self.src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    def generate_witness(self, mode: str = None, input_file: str = None, model_path: str = None) -> dict:
        if mode == "sliced":
            return self.generate_witness_sliced(input_file)
        else:
            return self.generate_witness_whole(input_file, model_path)


    def generate_witness_sliced(self, input_file: str = None) -> dict:
        parent_dir = self.src_dir
        input_path = input_file or os.path.join(parent_dir, self.model_directory, "input.json")
        metadata_dir = os.path.join(parent_dir, self.model_directory, "slices")

        slices_dir = os.path.join(self.base_path, "slices")

        # Ensure metadata is loaded
        try:
            metadata = ModelUtils.load_metadata(metadata_dir)
            num_segments = len(metadata['segments'])

            if num_segments == 0:
                print("No segments found to process!")
                return {"error": "No segments found"}

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {"error": str(e)}

        # Start memory tracker
        parent_pid = os.getpid()
        monitor_results = {'peak_subprocess_rss': 0}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=runner_utils.monitor_subprocess_memory,
            args=(parent_pid, "ezkl", monitor_results, stop_event),
            daemon=True
        )
        monitor_thread.start()
        time.sleep(0.1)  # Give thread a moment to start up

        # Start time
        start_time = time.time()
        segment_times = {}

        # Load initial input
        try:
            with open(Path(parent_dir, input_path), 'r') as f:
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
            segment_model_path = os.path.join(parent_dir, slices_dir, segment_name, segment_model_name)

            # Create segment-specific input file
            segment_input_path = os.path.join(parent_dir, slices_dir, segment_name, f"{segment_name}_input.json")
            try:
                with open(segment_input_path, 'w') as f:
                    json.dump(current_input, f)

            except Exception as e:
                print(f"Error creating input file: {e}")
                continue

            # Create segment-specific witness output path
            segment_witness_path = os.path.join(parent_dir, slices_dir, segment_name, f"{segment_name}_witness.json")

            # Run EZKL witness generation command
            cmd = [
                "ezkl",
                "gen-witness",
                "--compiled-circuit", segment_model_path,
                "--data", segment_input_path,
                "--output", segment_witness_path,
            ]

            try:
                process = subprocess.run(
                    cmd,
                    # capture_output=True,
                    cwd=str(parent_dir),
                    text=True,
                    check=True
                )

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

        # Get peak memory
        stop_event.set()
        monitor_thread.join(timeout=2.0)  # Wait briefly for the thread to finish
        if monitor_thread.is_alive():
            print("Warning: Child process memory monitor thread did not terminate cleanly.")
        subprocess_ram = monitor_results.get('peak_subprocess_mem', 0) / 1024
        subprocess_swap = monitor_results.get('peak_subprocess_swap', 0) / 1024
        subprocess_total = monitor_results.get('peak_subprocess_total', 0) / 1024

        memory = {'ram': subprocess_ram, 'swap': subprocess_swap, 'total': subprocess_total}
        
        final_logits = self.process_sliced_witness_output({"witness_paths": witness_paths})

        # Return the results
        results = {
            "memory": memory['total'],
            "total_time": total_time,
            "segment_times": segment_times,
            "result": final_logits,
            # "witness_paths": witness_paths
        }

        return results

    def generate_witness_whole(self, input_file: str = None, model_path: str = None) -> dict:
        input_file = input_file or os.path.join(self.model_directory, "input.json")
        model_path = model_path or os.path.join(self.base_path, "model", "model.compiled")
        witness_path = os.path.join(self.base_path, "model", "witness.json")
        vk_path = os.path.join(self.base_path, "model", "vk.key")
        parent_dir = self.src_dir

        # Start memory tracker
        parent_pid = os.getpid()
        monitor_results = {'peak_subprocess_rss': 0}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=runner_utils.monitor_subprocess_memory,
            args=(parent_pid, "ezkl", monitor_results, stop_event),
            daemon=True
        )
        monitor_thread.start()
        time.sleep(0.1)  # Give thread a moment to start up

        witness_time_start = time.time()
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
            cwd=str(parent_dir),
            check=True
        )

        witness_time_end = time.time()
        witness_time = witness_time_end - witness_time_start

        # Get peak memory
        stop_event.set()
        monitor_thread.join(timeout=2.0)  # Wait briefly for the thread to finish
        if monitor_thread.is_alive():
            print("Warning: Child process memory monitor thread did not terminate cleanly.")
        subprocess_ram = monitor_results.get('peak_subprocess_mem', 0) / 1024
        subprocess_swap = monitor_results.get('peak_subprocess_swap', 0) / 1024
        subprocess_total = monitor_results.get('peak_subprocess_total', 0) / 1024

        memory = {'ram': subprocess_ram, 'swap': subprocess_swap, 'total': subprocess_total}

        # print(f"Witness time: {witness_time:.2f} seconds")

        # return the processed outputs
        with open(Path(parent_dir, witness_path), "r") as f:
            witness_data = json.load(f)
            output = self.process_witness_output(witness_data)

        outputs = {'memory': memory,'total_time': witness_time, 'result': output}
        return outputs

    def prove(self, mode: str = None):
        if mode == "sliced":
            return self.prove_sliced()
        else:
            return self.prove_whole()

    def prove_sliced(self):
        parent_dir = self.src_dir
        slices_directory = os.path.join(parent_dir, self.model_directory, "ezkl", "slices")
        metadata_directory = os.path.join(parent_dir, self.model_directory, "slices")

        segments = RunnerUtils.get_segments(metadata_directory)
        num_segments = len(segments)

        # Start memory tracker
        parent_pid = os.getpid()
        monitor_results = {'peak_subprocess_rss': 0}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=runner_utils.monitor_subprocess_memory,
            args=(parent_pid, "ezkl", monitor_results, stop_event),
            daemon=True
        )
        monitor_thread.start()
        time.sleep(0.1)  # Give thread a moment to start up

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
                process = subprocess.run(
                    cmd,
                    # capture_output=True,
                    cwd=str(parent_dir),
                    text=True,
                    check=True
                )

                # Store proof path and timing
                proof_paths[segment_idx] = segment_proof_path
                segment_time = time.time() - segment_start_time
                segment_times[segment_idx] = segment_time
                # print(f"✓ Segment {segment_idx} proof generated in {segment_time:.2f}s")

            except subprocess.CalledProcessError as e:
                print(f"Error generating proof for segment {segment_idx}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Error code: {e.returncode}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                continue

        # Calculate total time
        total_time = time.time() - start_time

        # Get peak memory
        stop_event.set()
        monitor_thread.join(timeout=2.0)  # Wait briefly for the thread to finish
        if monitor_thread.is_alive():
            print("Warning: Child process memory monitor thread did not terminate cleanly.")
        subprocess_ram = monitor_results.get('peak_subprocess_mem', 0) / 1024
        subprocess_swap = monitor_results.get('peak_subprocess_swap', 0) / 1024
        subprocess_total = monitor_results.get('peak_subprocess_total', 0) / 1024

        memory = {'ram': subprocess_ram, 'swap': subprocess_swap, 'total': subprocess_total}

        # Return the results
        results = {
            "memory": memory,
            "total_time": total_time,
            "segment_times": segment_times,
            "proof_paths": proof_paths,
            "num_segments_processed": len(segment_times)
        }

        return results

    def prove_whole(self):
        parent_dir = self.src_dir
        witness_path = os.path.join(self.base_path, "model", "witness.json")
        model_path = os.path.join(self.base_path, "model", "model.compiled")
        pk_path = os.path.join(self.base_path, "model", "pk.key")
        proof_path = os.path.join(self.base_path, "model", "proof.json")

        # Start memory tracker
        parent_pid = os.getpid()
        monitor_results = {'peak_subprocess_rss': 0}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=runner_utils.monitor_subprocess_memory,
            args=(parent_pid, "ezkl", monitor_results, stop_event),
            daemon=True
        )
        monitor_thread.start()
        time.sleep(0.1)  # Give thread a moment to start up

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
            cwd=str(parent_dir),
            check=True
        )

        proof_time_end = time.time()
        proof_time = proof_time_end - proof_time_start\

        # Get peak memory
        stop_event.set()
        monitor_thread.join(timeout=2.0)  # Wait briefly for the thread to finish
        if monitor_thread.is_alive():
            print("Warning: Child process memory monitor thread did not terminate cleanly.")
        subprocess_ram = monitor_results.get('peak_subprocess_mem', 0) / 1024
        subprocess_swap = monitor_results.get('peak_subprocess_swap', 0) / 1024
        subprocess_total = monitor_results.get('peak_subprocess_total', 0) / 1024

        memory = {'ram': subprocess_ram, 'swap': subprocess_swap, 'total': subprocess_total}

        results = {"memory": memory, "total_time": proof_time}

        return results

    def verify(self, mode: str = None) -> dict:
        if mode == "sliced":
            return self.verify_slices()
        else:
            return self.verify_whole()

    def verify_slices(self) -> dict:
        parent_dir = self.src_dir
        slices_directory = os.path.join(parent_dir, self.model_directory, "ezkl", "slices")
        metadata_directory = os.path.join(parent_dir, self.model_directory, "slices")

        segments = RunnerUtils.get_segments(metadata_directory)
        num_segments = len(segments)

        # Start memory tracker
        parent_pid = os.getpid()
        monitor_results = {'peak_subprocess_rss': 0}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=runner_utils.monitor_subprocess_memory,
            args=(parent_pid, "ezkl", monitor_results, stop_event),
            daemon=True
        )
        monitor_thread.start()
        time.sleep(0.1)  # Give thread a moment to start up

        # Start timing
        start_time = time.time()
        segment_times = {}
        verify_paths = {}

        for segment_idx in range(num_segments):
            segment_start_time = time.time()

            segment_name = f"segment_{segment_idx}"
            segment_settings_path = os.path.join(slices_directory, segment_name, f"{segment_name}_settings.json")
            segment_proof_path = os.path.join(slices_directory, segment_name, f"{segment_name}_proof.json")
            segment_vk_path = os.path.join(slices_directory, segment_name, f"{segment_name}_vk.key")


            # Run EZKL proof generation command
            cmd = [
                "ezkl",
                "verify",
                "--proof-path", segment_proof_path,
                "--settings-path", segment_settings_path,
                "--vk-path", segment_vk_path
            ]

            try:
                process = subprocess.run(
                    cmd,
                    # capture_output=True,
                    cwd=str(parent_dir),
                    text=True,
                    check=True
                )

                # Store proof path and timing
                verify_paths[segment_idx] = segment_proof_path
                segment_time = time.time() - segment_start_time
                segment_times[segment_idx] = segment_time
                # print(f"✓ Segment {segment_idx} proof generated in {segment_time:.2f}s")

            except subprocess.CalledProcessError as e:
                print(f"Error generating proof for segment {segment_idx}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Error code: {e.returncode}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                continue

        # Calculate total time
        total_time = time.time() - start_time

        # Get peak memory
        stop_event.set()
        monitor_thread.join(timeout=2.0)  # Wait briefly for the thread to finish
        if monitor_thread.is_alive():
            print("Warning: Child process memory monitor thread did not terminate cleanly.")
        subprocess_ram = monitor_results.get('peak_subprocess_mem', 0) / 1024
        subprocess_swap = monitor_results.get('peak_subprocess_swap', 0) / 1024
        subprocess_total = monitor_results.get('peak_subprocess_total', 0) / 1024

        memory = {'ram': subprocess_ram, 'swap': subprocess_swap, 'total': subprocess_total}

        # Return the results
        results = {
            "memory": memory,
            "total_time": total_time,
            "segment_times": segment_times,
            "verify_paths": verify_paths,
            "num_segments_processed": len(segment_times)
        }

        return results

    def verify_whole(self) -> dict:
        parent_dir = self.src_dir
        settings_path = os.path.join(self.base_path, "model", "settings.json")
        vk_path = os.path.join(self.base_path, "model", "vk.key")
        proof_path = os.path.join(self.base_path, "model", "proof.json")

        # Start memory tracker
        parent_pid = os.getpid()
        monitor_results = {'peak_subprocess_rss': 0}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=runner_utils.monitor_subprocess_memory,
            args=(parent_pid, "ezkl", monitor_results, stop_event),
            daemon=True
        )
        monitor_thread.start()
        time.sleep(0.1)  # Give thread a moment to start up

        verify_time_start = time.time()

        subprocess.run(
            [
                "ezkl",
                "verify",
                "--proof-path", proof_path,
                "--settings-path", settings_path,
                "--vk-path", vk_path
            ],
            env=env,
            cwd=str(parent_dir),
            check=True
        )

        verify_time_end = time.time()
        verify_time = verify_time_end - verify_time_start

        # Get peak memory
        stop_event.set()
        monitor_thread.join(timeout=2.0)  # Wait briefly for the thread to finish
        if monitor_thread.is_alive():
            print("Warning: Child process memory monitor thread did not terminate cleanly.")
        subprocess_ram = monitor_results.get('peak_subprocess_mem', 0) / 1024
        subprocess_swap = monitor_results.get('peak_subprocess_swap', 0) / 1024
        subprocess_total = monitor_results.get('peak_subprocess_total', 0) / 1024

        memory = {'ram': subprocess_ram, 'swap': subprocess_swap, 'total': subprocess_total}

        results = {"memory": memory, "total_time": verify_time}

        return results

    def circuitize_model(self, model_path, output_path):
        pass

    def circuitize_model_for_slice(self, model_path, layer_name, output_path):
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

    @staticmethod
    def process_sliced_witness_output(sliced_witness_result):
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

        return EzklRunner.process_witness_output(witness_data)

if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net"
    }

    model_dir = base_paths[model_choice]
    runner = EzklRunner(model_dir)

    if model_choice == 1:
        # circuitize model
        # TODO Integration the circuitize model functions for ezkl
        # run inference
        result = runner.generate_witness()
        result = runner.generate_witness(mode="sliced")
        # run proof
        result = runner.prove()
        result = runner.prove(mode="sliced")
        # run verification (mode = slice)
        # TODO: incorporate verification for ezkl
        print(result)
    elif model_choice == 2:
        # run inference
        # result = runner.generate_witness()
        # result = runner.generate_witness(mode="sliced")
        # run proof
        # result = runner.prove()
        # result = runner.prove(mode="sliced")
        result = runner.verify(mode="sliced")
        print(result)