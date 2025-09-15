class DeepProveRunner:
    def __init__(self, model_path):
        # check env var for deepprove project path
        # check if deepprove cli exists

        #if not installed, throw error

        # extract model info that is needed for running
        pass

    def run_inference(self, model_path, input_path, output_path):
        pass

    def run_inference_for_slice(self, model_path, layer_name, input_path, output_path):
        pass

    def run_proof(self, model_path, proof_path):
        pass

    def run_proof_for_slice(self, model_path, layer_name, proof_path):
        pass

    def run_verification(self, model_path, proof_path, input_path, output_path):
        pass

    def run_verification_for_slice(self, model_path, layer_name, proof_path, input_path, output_path):
        pass

    def circuitize_model(self, model_path, output_path):
        pass

    def circuitize_model_for_slice(self, model_path, layer_name, output_path):
        pass

if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net"
    }

    model_dir = base_paths[model_choice]
    runner = DeepProveRunner(model_dir)

    if model_choice == 1:
        # run inference (mode = slice)
        # run proof(mode = slice)
        # run verification (mode = slice)
        # circuitize model(mode = slice)
        pass
    elif model_choice == 2:
        # run inference (mode = slice)
        # run proof(mode = slice)
        # run verification (mode = slice)
        # circuitize model(mode = slice)
        pass