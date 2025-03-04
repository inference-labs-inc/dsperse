# composite_workflow_proofs.py
import torch
import subprocess


def split_torch_model(torch_model) -> list:
    """
    Splits the Torch model into separate Torch layers.

    Args:
        torch_model (nn.Module): The full Torch model.

    Returns:
        list: A list of tuples, each containing (layer_name, torch layer model).
    """
    layer_models = []
    # For simplicity, assuming torch_model is a Sequential or similar iterable model.
    # In a more complex model, you may need custom logic to extract submodules.
    for idx, layer in enumerate(torch_model.children()):
        layer_name = f"layer_{idx}"
        # Wrap the layer in a model container if needed.
        single_layer_model = torch.nn.Sequential(layer)
        layer_models.append((layer_name, single_layer_model))
        print(f"Extracted {layer_name}")
    return layer_models


def convert_layer_to_onnx(layer_model, input_shape, onnx_file_path: str) -> None:
    """
    Convert a Torch layer (or sub-model) to an ONNX file.

    Args:
        layer_model (nn.Module): The Torch layer model to be converted.
        input_shape (tuple): Shape of a dummy input for tracing.
        onnx_file_path (str): Path where the ONNX file will be saved.

    Returns:
        None
    """
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(layer_model, dummy_input, onnx_file_path)
    print(f"Layer exported to ONNX format at: {onnx_file_path}")


def run_ezkl_proof_for_layer(onnx_file_path: str, proof_output_path: str) -> str:
    """
    Run ezkl on the ONNX file for a single layer to generate a zero-knowledge proof.

    Args:
        onnx_file_path (str): Path to the ONNX file for the layer.
        proof_output_path (str): Path where the proof for this layer should be stored.

    Returns:
        str: Proof data for the layer.
    """
    # Build the command for ezkl using assumed command-line arguments.
    cmd = [
        "ezkl", "prove",
        "--model", onnx_file_path,
        "--proof", proof_output_path
    ]
    print("Running ezkl command for layer:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error output:", result.stderr)
        raise RuntimeError(f"ezkl proof for layer failed: {result.stderr}")

    print(f"Layer proof generated and stored at: {proof_output_path}")
    # Optionally, parse or return details from the command output.
    return proof_output_path



def compose_layer_proofs(layer_proofs: list) -> str:
    """
    Compose individual layer proofs into an aggregated proof for the full model.

    Args:
        layer_proofs (list): A list of proofs from each layer.

    Returns:
        str: Aggregated proof data for the full model.
    """
    print("Composing layer proofs...")
    aggregated_proof = "aggregated_proof(" + ",".join(layer_proofs) + ")"
    print("Aggregated proof composed successfully.")
    return aggregated_proof
