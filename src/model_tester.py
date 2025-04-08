import os
import numpy as np
import json
from src.model_runner import ModelRunner
from src.utils.model_utils import ModelUtils


class ModelTester:
    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.model_runner = ModelRunner(model_directory)


    def test_model_accuracy(self, num_runs=10, input_path: str=None):
        """Run normal/ezkl inference on the sliced and whole and compare output from both."""
        # Check if input file path exists
        input_path = input_path if input_path else os.path.join(self.model_directory, "input.json")

        #  dict of {int run, final output from [original whole model (0-7), original sliced model(0-7), circuitized_whole_model(0-7), circuitized_sliced model(0-7)]}
        results = {}

        # get original input from path, find out its shape so we can generate more inputs
        generated_inputs_directory = os.path.join(self.model_directory, "generated_inputs")
        os.makedirs(generated_inputs_directory, exist_ok=True)

        # for num_runs
        for i in range(num_runs):
        
            # run inference on original whole model --> get output (softmax)
            original_output = self.model_runner.predict(input_path=input_path)
            original_output = original_output['logits']

            # run inference on original sliced model --> get output (softmax)
            sliced_output = self.model_runner.predict(input_path=input_path, mode="sliced")
            sliced_output = sliced_output['logits']

            # run generate_witness on whole model --> get the output from witness.json
            witness_output = self.model_runner.generate_witness(input_file=input_path)
            circuitized_output = self.model_runner.process_witness_output(witness_output)
            circuitized_output = circuitized_output['logits']

            # run generate_witness_sliced --> get the output from last witness.json
                # use helper method to fetch final result and run through a softmax
            sliced_witness_output = self.model_runner.generate_witness_sliced(input_path=input_path)
            sliced_witness_output = self.model_runner.process_sliced_witness_output(sliced_witness_output)
            sliced_circuitized_output = sliced_witness_output['logits']

            # add to results
            results[i] = {
                "original_output": original_output,
                "sliced_output": sliced_output,
                "circuitized_output": circuitized_output,
                "sliced_circuitized_output": sliced_circuitized_output,
            }

            # mutate, or random generate new input for next round
            input_path = self._generate_random_input_file(input_path, generated_inputs_directory)

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
            sliced_circuitized_output = result["sliced_circuitized_output"].detach().cpu().numpy()

            max_error = 1.0

            # Calculate accuracy as (1 - normalized_error) and convert to percentage
            accuracies["original_vs_sliced"].append(
                float(100 * (1 - np.mean(np.abs(original_output - sliced_output)) / max_error))
            )
            accuracies["original_vs_circuitized"].append(
                float(100 * (1 - np.mean(np.abs(original_output - circuitized_output)) / max_error))
            )
            accuracies["original_vs_sliced_circuitized"].append(
                float(100 * (1 - np.mean(np.abs(original_output - sliced_circuitized_output)) / max_error))
            )
            accuracies["circuitized_vs_sliced_circuitized"].append(
                float(100 * (1 - np.mean(np.abs(circuitized_output - sliced_circuitized_output)) / max_error))
            )

        # Calculate the average accuracies over all runs
        average_accuracies = {key: max(0.0, min(100.0, float(np.mean(value)))) for key, value in accuracies.items()}

        # Return the results
        return {
            "results": results,
            "accuracies": average_accuracies
        }


    def _generate_random_input_file(self, input_file, save_path):
        """Helper method to generate random values in the input file"""

        print("Save Path ", save_path)
        print("Input File ", input_file)
        # Generate random data with the specified shape
        if "doom" in self.model_directory.lower():
            dummy_input_shape = (1, 4, 28, 28)
        elif "net" in self.model_directory.lower():
            dummy_input_shape = (1, 3, 32, 32)
        else:
            raise ValueError("Unknown input file shape. Please specify the shape manually in the code. ")

        random_data = np.random.rand(*dummy_input_shape)

        # Normalize to the range [0, 1] (similar to the example data)
        random_data = random_data.astype(np.float32)

        # Flatten the data to match the expected format
        flattened_data = random_data.flatten().tolist()

        # Create the JSON structure
        input_json = {"input_data": [flattened_data]}

        # Save to the specified path
        output_path = os.path.join(save_path, os.path.basename(input_file))
        with open(output_path, 'w') as f:
            json.dump(input_json, f)
        
        return output_path
        

    def test_ezkl_performance(self, input_file_path=None):
        """run a test on sliced, whole, and circuitized models to see how long it takes to run and the memory required"""
        # Check if input file path exists

        # create output results, nested dict
        #  dict of results, {model: {inference time, inference memory, inference result, prove time, prove memory}, sliced_model: {...}}

        # get original input from path, find out its shape so we can generate more inputs

        # start timer for whole model
        # start memory tracker
        

        # run inference on whole model

        # stop timer
        # stop memory tracker

        # repeat for sliced

        # repeat for circuitized model

        # repeat for sliced circuitized model

        # print table showing results on time and memory that each one took
        pass

    def test_expander_performance(self):
        # use gravy to run expander compiler collection

        # this outputs to a folder gravy.helper_f.compile_circuit(params, path param)

        # helper.generatewitness(params, path param)

        pass

    def test_deep_prove_performance(self):
        # use gravy to run expander compiler collection

        # this outputs to a folder gravy.helper_f.compile_circuit(params, path param)

        # helper.generatewitness(params, path param)

        pass


# Example usage
if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net"
    }

    model_dir = base_paths[model_choice]
    model_tester = ModelTester(model_dir)

    if model_choice == 1:
        accuracy = model_tester.test_model_accuracy(num_runs=1000)
        print(f"Model accuracy: {accuracy}")
        # model_tester.test_ezkl_performance()
        # model_tester.test_expander_performance()
        # model_tester.test_deep_prove_performance()

    elif model_choice == 2:
        accuracy = model_tester.test_model_accuracy(num_runs=10000)
        print(f"Model accuracy: {accuracy}")
        # model_tester.test_ezkl_performance()
        # model_tester.test_expander_performance()
        # model_tester.test_deep_prove_performance()
