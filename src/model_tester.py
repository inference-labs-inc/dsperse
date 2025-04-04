import os
import numpy as np
import json
from src.model_runner import ModelRunner
from src.utils.model_utils import ModelUtils


class ModelTester:
    def __init__(self, model_path=None, sliced_model_path=None, circuitized_model_path=None,
                 sliced_circuitized_model_path=None):
        self.model_runner = ModelRunner()
        self.model_path = model_path
        self.sliced_model_path = sliced_model_path
        self.circuitized_model_path = circuitized_model_path
        self.sliced_circuitized_model_path = sliced_circuitized_model_path

    def test_model_accuracy(self, num_runs=10, model_path=None, sliced_model_path=None, circuitized_model_path=None,
                            sliced_circuitized_model_path=None, input_file_path=None
    ):
        """Run normal/ezkl inference on the sliced and whole and compare output from both."""
        # Check if input file path exists
        if not input_file_path or not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file path '{input_file_path}' does not exist. Please provide a valid path.")

        # create output results
        #  dict of {int run, final output from [original whole model (0-7), original sliced model(0-7), circuitized_whole_model(0-7), circuitized_sliced model(0-7)]}
        results = {}

        # get original input from path, find out its shape so we can generate more inputs
        input_file = input_file_path
        save_path = os.path.join(os.path.dirname(input_file_path), "generated")
        os.makedirs(save_path, exist_ok=True)
        input_tensor = ModelUtils.preprocess_input(input_file)

        # for num_runs
        for i in range(num_runs):
        
            # run inference on original whole model --> get output (softmax)
            original_output = self.model_runner.predict(model_path=model_path, input_tensor=input_tensor)
            original_output = original_output['logits']

            # run inference on original sliced model --> get output (softmax)
            sliced_output = self.model_runner.predict(model_path=sliced_model_path, input_tensor=input_tensor)
            sliced_output = sliced_output['logits']

            # run generate_witness on whole model --> get the output from witness.json
            witness_output = self.model_runner.generate_witness(input_file=input_file, base_path="models/doom/circuit/")
            circuitized_output = self.model_runner.process_witness_output(witness_output)
            circuitized_output = circuitized_output['logits']

            # run generate_witness_sliced --> get the output from last witness.json
                # use helper method to fetch final result and run through a softmax
            sliced_witness_output = self.model_runner.generate_witness_sliced(input_path=input_file)
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
            input_file = self._generate_random_input_file(input_file, save_path)
            input_tensor = ModelUtils.preprocess_input(input_file)

        # For the results
        accuracies = {
            "original_vs_sliced": [],
            "original_vs_circuitized": [],
            "original_vs_sliced_circuitized": [],
            "circuitized_vs_sliced_circuitized": [],
        }

        for i, result in results.items():
            original_output = np.array(result["original_output"], dtype=float)
            sliced_output = np.array(result["sliced_output"], dtype=float)
            circuitized_output = np.array(result["circuitized_output"], dtype=float)
            sliced_circuitized_output = np.array(result["sliced_circuitized_output"], dtype=float)

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

    @staticmethod
    def _generate_random_input_file(input_file, save_path):
        """Helper method to generate random values in the input file
    
        Args:
            input_file: Path where the input file should be saved
        """

        # Generate random data with the specified shape
        dummy_input_shape = (1, 4, 28, 28)
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
        

    def test_ezkl_performance(self, model_path=None, sliced_model_path=None, circuitized_model_path=None,
                            sliced_circuitized_model_path=None, input_file_path=None):
        """run a test on sliced, whole, and circuitized models to see how long it takes to run and the memory required"""
        # Check if input file path exists
        if not input_file_path or not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file path '{input_file_path}' does not exist. Please provide a valid path.")

        # create output results, nested dict
        #  dict of results, {model: {inference time, inference memory, inference result, prove time, prove memory}, sliced_model: {...}}
        results = {}

        # get original input from path, find out its shape so we can generate more inputs
        input_file = input_file_path
        input_tensor = ModelUtils.preprocess_input(input_file)

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


# Example usage
if __name__ == "__main__":
    # original model
    model_path_full = "models/doom/model.pth"
    model_path_sliced = "models/doom/output/"
    #
    # #circuitized model
    circuitized_model_path_full = "models/doom/circuit/"
    circuitized_model_path_sliced = "models/doom/output/circuitized_slices/"
    #
    # # input for inference
    input_path = "models/doom/input.json"
    model_tester = ModelTester()

    print(model_tester.test_model_accuracy(input_file_path=input_path, num_runs=10, model_path=model_path_full, sliced_model_path=model_path_sliced, circuitized_model_path=circuitized_model_path_full, sliced_circuitized_model_path=circuitized_model_path_sliced))

