import os
import subprocess
import json
from pathlib import Path

from src.utils.model_utils import ModelUtils


class JSTProveRunner:
    def __init__(self, model_directory):
        self.jstprove_project_path = os.environ.get('JSTPROVE_PATH')
        if not self.jstprove_project_path:
            raise ValueError("JSTPROVE_PATH environment variable is not set")
        
        # check if jstprove cli python file exists, it should be jstprove project path/cli.py

        cli_path = os.path.join(self.jstprove_project_path, 'cli.py')
        if not os.path.isfile(cli_path):
            raise FileNotFoundError(f"JSTProve CLI not found at {cli_path}")

        # extract model info that is needed for running jstprove
        self.model_directory = model_directory
        self.base_path = os.path.join(model_directory, "jstprove")
        self.src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def generate_witness(self, mode: str = None, input_file: str = None) -> dict:
        if mode == "sliced":
            return self.generate_witness_sliced(input_file=input_file)
        else:
            return self.generate_witness_whole(input_file=input_file)
        
    def prove(self, mode: str = None):
        if mode == "sliced":
            return self.prove_slices()
        else:
            return self.prove_whole()
        
    def verify(self, mode: str = None, input_file: str = None) -> dict:
        if mode == "sliced":
            return self.verify_slices(input_path=input_file)
        else:
            return self.verify_whole(input_path=input_file)

    def circuitize(self, mode: str = None) -> dict:
        if mode == "sliced":
            return self.circuitize_slices()
        else:
            return self.circuitize_whole()


    def generate_witness_whole(self,  input_file: str = None, model_path: str = None) -> dict:
        # set base params
        parent_dir = self.src_dir
        model_folder = model_path or os.path.join(parent_dir, self.base_path, "model")
        input_path = input_file or os.path.join(model_folder, "input.json")
        output_path = os.path.join(model_folder, "output.json")
        witness_path = os.path.join(model_folder, "witness.compiled")
        circuit_path = os.path.join(model_folder, "model.compiled")

        # Get subprocess output
        result = subprocess.run(
            [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", "NetModel",
                "--gen_witness",
                "--circuit_path", circuit_path,
                "--input", input_path,
                "--output", output_path,
                "--witness", witness_path
            ],
            capture_output=True,
            text=True,
            env=os.environ,
            cwd=self.jstprove_project_path,
        )

        # Parse output to get memory and time metrics
        output = result.stdout

        # Read and parse output JSON file
        with open(output_path, 'r') as f:
            output_data = json.load(f)

        results = output_data.get('rescaled_output', None)[0]
        return results


    def generate_witness_sliced(self, input_file: str = None, model_path: str = None) -> dict:
        parent_dir = self.src_dir
        slices_folder = model_path or os.path.join(parent_dir, self.base_path, "slices")
        input_path = input_file or os.path.join(slices_folder, "input.json")
        metadata_dir = os.path.join(parent_dir, self.model_directory, "slices")
        slices_dir = slices_folder

        metadata = ModelUtils.load_metadata(metadata_dir)
        num_segments = len(metadata['segments'])

        # Load initial input
        current_input = None
        try:
            with open(Path(input_path), 'r') as f:
                current_input = json.load(f)
        except Exception as e:
            print(f"Error loading input: {e}")
            return {"error": str(e)}

        final_segment_path = None

        class_names = {0: 'NetConv1Model', 1: 'NetConv2Model', 2: 'NetFC1Model', 3: 'NetFC2Model', 4: 'NetFC3Model'}

        for segment_idx in range(num_segments):
            # Get segment model paths
            segment_name = f"segment_{segment_idx}"
            segment_model_name = f"{segment_name}_circuit.compiled"

            # Create segment-specific files
            segment_model_path = os.path.join(slices_dir, segment_name, segment_model_name)
            segment_input_path = os.path.join(slices_dir, segment_name, f"{segment_name}_input.json")
            segment_output_path = os.path.join(slices_dir, segment_name, f"{segment_name}_output.json")
            segment_witness_path = os.path.join(slices_dir, segment_name, f"{segment_name}_witness.compiled")

            try:
                with open(segment_input_path, 'w') as f:
                    json.dump(current_input, f)

            except Exception as e:
                print(f"Error creating input file: {e}")
                continue

            # read and print the first line in input.json
            with open(segment_input_path, 'r') as f:
                input_data = json.load(f)

            # Run EZKL witness generation command
            cmd = [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", class_names[segment_idx],
                "--gen_witness",
                "--circuit_path", segment_model_path,
                "--input", segment_input_path,
                "--output", segment_output_path,
                "--witness", segment_witness_path
            ]

            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    cwd=self.jstprove_project_path,
                    text=True
                )

                final_segment_path = segment_output_path

                # If successful, prepare for next segment
                if segment_idx < num_segments:
                    try:
                        with open(segment_output_path, 'r') as f:
                            output_data = json.load(f)
                            # print(f"{segment_name} output_data: {output_data['rescaled_output']}")
                    except Exception as e:
                        raise ValueError(f"Error loading witness file: {e}")

                    # Extract the output data to use as input for next segment
                    if 'output' in output_data:
                        # Format the outputs as input for the next segment
                        output_data = output_data['output']
                        current_input = {"input": output_data}
                        # print(f"output_data: {output_data[0]}")
                    else:
                        raise ValueError(f"Witness file for segment {segment_idx} does not contain rescaled_output")


            except subprocess.CalledProcessError as e:
                error_msg = f"Error: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
                raise RuntimeError(error_msg)

        # Read and parse output JSON file
        with open(final_segment_path, 'r') as f:
            output_data = json.load(f)

        output = output_data.get('rescaled_output', None)[0]
        return output

    def prove_slices(self) -> bool:
        parent_dir = self.src_dir
        slices_folder = os.path.join(parent_dir, self.base_path, "slices")
        metadata_dir = os.path.join(parent_dir, self.model_directory, "slices")

        slices_dir = slices_folder

        metadata = ModelUtils.load_metadata(metadata_dir)
        num_segments = len(metadata['segments'])

        class_names = {0: 'NetConv1Model', 1: 'NetConv2Model', 2: 'NetFC1Model', 3: 'NetFC2Model', 4: 'NetFC3Model'}

        for segment_idx in range(num_segments):
            # Get segment model paths
            segment_name = f"segment_{segment_idx}"
            segment_model_name = f"{segment_name}_circuit.compiled"

            # Create segment-specific files
            segment_model_path = os.path.join(slices_dir, segment_name, segment_model_name)
            segment_proof_path = os.path.join(slices_dir, segment_name, f"{segment_name}_proof.bin")
            segment_witness_path = os.path.join(slices_dir, segment_name, f"{segment_name}_witness.compiled")

            # Run EZKL witness generation command
            cmd = [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", class_names[segment_idx],
                "--prove",
                "--circuit_path", segment_model_path,
                "--proof", segment_proof_path,
                "--witness", segment_witness_path
            ]

            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    cwd=self.jstprove_project_path,
                    text=True
                )

                # TODO: parse output to get true of false on success/failure

            except subprocess.CalledProcessError as e:
                error_msg = f"Error: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
                print(error_msg)
                return False

        print("All segments verified successfully.")
        return True

    def prove_whole(self) -> bool:
        # set base params
        parent_dir = self.src_dir
        model_folder = os.path.join(parent_dir, self.base_path, "model")
        proof_path = os.path.join(model_folder, "proof.bin")
        witness_path = os.path.join(model_folder, "witness.compiled")
        circuit_path = os.path.join(model_folder, "model.compiled")

        # Get subprocess output
        result = subprocess.run(
            [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", "NetModel",
                "--prove",
                "--circuit_path", circuit_path,
                "--proof", proof_path,
                "--witness", witness_path
            ],
            capture_output=True,
            text=True,
            env=os.environ,
            cwd=self.jstprove_project_path,
        )

        # TODO: parse output to get true of false on success/failure
        return result.returncode == 0

    def verify_slices(self, input_path: str = None) -> dict:
        parent_dir = self.src_dir
        slices_folder = os.path.join(parent_dir, self.base_path, "slices")
        metadata_dir = os.path.join(parent_dir, self.model_directory, "slices")

        slices_dir = slices_folder

        metadata = ModelUtils.load_metadata(metadata_dir)
        num_segments = len(metadata['segments'])

        class_names = {0: 'NetConv1Model', 1: 'NetConv2Model', 2: 'NetFC1Model', 3: 'NetFC2Model', 4: 'NetFC3Model'}

        # if input_path is not None, copy the file into segment_0_input.json
        if input_path is not None:
            segment_0_input_path = os.path.join(slices_dir, "segment_0", "segment_0_input.json")
            try:
                with open(input_path, 'r') as src, open(segment_0_input_path, 'w') as dst:
                    json.dump(json.load(src), dst)
            except Exception as e:
                print(f"Error copying input file: {e}")
        
        for segment_idx in range(num_segments):
            # Get segment model paths
            segment_name = f"segment_{segment_idx}"
            segment_model_name = f"{segment_name}_circuit.compiled"

            # Create segment-specific files
            segment_model_path = os.path.join(slices_dir, segment_name, segment_model_name)
            segment_proof_path = os.path.join(slices_dir, segment_name, f"{segment_name}_proof.bin")
            segment_witness_path = os.path.join(slices_dir, segment_name, f"{segment_name}_witness.compiled")
            segment_output_path = os.path.join(slices_dir, segment_name, f"{segment_name}_output.json")
            segment_input_path = os.path.join(slices_dir, segment_name, f"{segment_name}_input.json")

            # Run EZKL witness generation command
            cmd = [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", class_names[segment_idx],
                "--verify",
                "--circuit_path", segment_model_path,
                "--proof", segment_proof_path,
                "--witness", segment_witness_path,
                "--output", segment_output_path,
                "--input", segment_input_path,
            ]

            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    cwd=self.jstprove_project_path,
                    text=True
                )
            # TODO: parse output to get true of false on success/failure
            except subprocess.CalledProcessError as e:
                error_msg = f"Error: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
                raise RuntimeError(error_msg)

        return results

    def verify_whole(self, input_path: str = None) -> dict:
        # set base params
        parent_dir = self.src_dir
        model_folder = os.path.join(parent_dir, self.base_path, "model")
        proof_path = os.path.join(model_folder, "proof.bin")
        output_path = os.path.join(model_folder, "output.json")
        witness_path = os.path.join(model_folder, "witness.compiled")
        circuit_path = os.path.join(model_folder, "model.compiled")
        input_path = input_path or os.path.join(model_folder, "input.json")

        # Get subprocess output
        result = subprocess.run(
            [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", "NetModel",
                "--verify",
                "--circuit_path", circuit_path,
                "--proof", proof_path,
                "--witness", witness_path,
                "--output", output_path,
                "--input", input_path,
            ],
            capture_output=True,
            text=True,
            env=os.environ,
            cwd=self.jstprove_project_path,
        )

        # TODO: parse output to get true of false on success/failure
        return results

    def circuitize_whole(self):
        # set base params
        parent_dir = self.src_dir
        model_folder = os.path.join(parent_dir, self.base_path, "model")
        circuit_path = os.path.join(model_folder, "model.compiled")

        # Get subprocess output
        result = subprocess.run(
            [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", "NetModel",
                "--compile",
                "--circuit_path", circuit_path
            ],
            capture_output=True,
            text=True,
            env=os.environ,
            cwd=self.jstprove_project_path,
        )

        # Parse output to get memory and time metrics
        output = result.stdout
        memory = float(next(line.split(':')[1].strip() for line in output.split('\n')
                            if 'Peak Memory used Overall' in line).split()[0])
        time = float(next(line.split(':')[1].strip() for line in output.split('\n')
                          if 'Time elapsed' in line).split()[0])

        return {
            'memory': memory,
            'total_time': time,
        }

    def circuitize_slices(self):
        parent_dir = self.src_dir
        slices_folder = os.path.join(parent_dir, self.base_path, "slices")
        metadata_dir = os.path.join(parent_dir, self.model_directory, "slices")

        slices_dir = slices_folder

        metadata = ModelUtils.load_metadata(metadata_dir)
        num_segments = len(metadata['segments'])

        # Process each segment
        segment_results = {}

        class_names = {0: 'NetConv1Model', 1: 'NetConv2Model', 2: 'NetFC1Model', 3: 'NetFC2Model', 4: 'NetFC3Model'}

        for segment_idx in range(num_segments):

            # Get segment model paths
            segment_name = f"segment_{segment_idx}"
            segment_model_name = f"{segment_name}_circuit.compiled"

            # Create segment-specific files
            segment_model_path = os.path.join(slices_dir, segment_name, segment_model_name)

            # Run EZKL witness generation command
            cmd = [
                "python",
                "cli.py",
                "--circuit", "net_model",
                "--class", class_names[segment_idx],
                "--compile",
                "--circuit_path", segment_model_path
            ]

            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    cwd=self.jstprove_project_path,
                    text=True
                )

                output = process.stdout
                memory = float(next(line.split(':')[1].strip() for line in output.split('\n')
                                    if 'Peak Memory used Overall' in line).split()[0])
                time = float(next(line.split(':')[1].strip() for line in output.split('\n')
                                  if 'Time elapsed' in line).split()[0])

                # Store witness path and timing
                segment_results[segment_idx] = {'memory': memory, 'time': time}

            except subprocess.CalledProcessError as e:
                error_msg = f"Error: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
                raise RuntimeError(error_msg)

        max_memory = max(segment_results[segment_idx]['memory'] for segment_idx in segment_results)
        total_time = sum(segment_results[segment_idx]['time'] for segment_idx in segment_results)
        segment_times = {segment_idx: segment_results[segment_idx]['time'] for segment_idx in segment_results}

        # Return the results
        results = {
            "memory": max_memory,
            "total_time": total_time,
            "segment_times": segment_times
        }

        return results


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net"
    }

    model_dir = base_paths[model_choice]
    runner = JSTProveRunner(model_dir)

    # Test
    results = runner.verify(mode="sliced") # change function and mode when needed
    print(results)

