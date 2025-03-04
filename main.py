# main_cli.py
import argparse
import time

from src.create_nanogpt import create_nanogpt, train_nanogpt
from src.singular_workflow_proofs import convert_to_onnx, run_ezkl_proof
from src.composite_workflow_proofs import split_torch_model, convert_layer_to_onnx, run_ezkl_proof_for_layer, \
    compose_layer_proofs


def main():
    """
    CLI interface to run either the singular proof workflow (end-to-end proof) or
    the composite proof workflow (layer-wise proof composition).

    Usage:
        --workflow singular    -> Runs the singular (end-to-end) proof workflow.
        --workflow composite   -> Runs the composite (layer-wise) proof composition workflow.
    """
    parser = argparse.ArgumentParser(description="Zero-Knowledge Proofs for NanoGPT Models using ezkl")
    parser.add_argument("--workflow", choices=["singular", "composite"], required=True,
                        help="Select the workflow: 'singular' for end-to-end proof, 'composite' for layer-wise proof composition.")
    args = parser.parse_args()

    # Common parameters for NanoGPT model
    model_params = {
        "vocab_size": 50257,  # Example vocab size
        "n_embd": 64,  # Embedding dimension
        "n_layer": 2,  # Number of transformer layers
        "n_head": 4  # Number of attention heads
    }
    input_shape = (1, 10)  # Adjust as needed for your model input

    # Create and train a NanoGPT model instance
    torch_model = create_nanogpt(model_params)
    train_nanogpt(torch_model)

    if args.workflow == "singular":
        print("Running singular (end-to-end) workflow...")
        start_time = time.time()
        onnx_model_path = "full_model.onnx"
        proof_output_path = "full_model.proof"
        convert_to_onnx(torch_model, input_shape, onnx_model_path)
        proof = run_ezkl_proof(onnx_model_path, proof_output_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Final Proof:", proof)
        print(f"Singular workflow execution time: {elapsed_time:.2f} seconds")

    elif args.workflow == "composite":
        print("Running composite (layer-wise) workflow...")
        start_time = time.time()
        # Split the model into individual torch layers
        layer_models = split_torch_model(torch_model)
        layer_proofs = []

        for layer_name, layer_model in layer_models:
            layer_onnx_path = f"{layer_name}.onnx"
            layer_proof_path = f"{layer_name}.proof"
            convert_layer_to_onnx(layer_model, input_shape, layer_onnx_path)
            layer_proof = run_ezkl_proof_for_layer(layer_onnx_path, layer_proof_path)
            layer_proofs.append(layer_proof)

        aggregated_proof = compose_layer_proofs(layer_proofs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Aggregated Proof:", aggregated_proof)
        print(f"Composite workflow execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
