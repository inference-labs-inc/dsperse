# singular_workflow_proofs.py
import torch
import time
import subprocess



def convert_to_onnx(torch_model, input_shape, onnx_file_path: str) -> None:
    """
    Convert a Torch model to an ONNX file.

    Args:
        torch_model (nn.Module): The Torch model to be converted.
        input_shape (tuple): Shape of a dummy input (e.g., (1, input_dim)).
        onnx_file_path (str): File path where the ONNX model will be saved.

    Returns:
        None
    """
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(torch_model, dummy_input, onnx_file_path)
    print(f"Model exported to ONNX format at: {onnx_file_path}")


def run_ezkl_proof(onnx_file_path: str, proof_output_path: str) -> str:
    """
    Generate a zero-knowledge proof using EZKL CLI commands following the official documentation.
    https://docs.ezkl.xyz/getting-started/setup/
    """
    try:
        # Step 1: Create settings file first
        settings = {
            "input_scale": 1,
            "param_scale": 1,
            "num_threads": 8
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)

        # Step 2: Generate the SRS
        cmd_gen_srs = ["ezkl", "gen-srs", "--power", "20"]
        print("Running gen-srs command:", " ".join(cmd_gen_srs))
        result = subprocess.run(cmd_gen_srs, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"EZKL gen-srs failed: {result.stderr}")
        print("SRS generation completed")

        # Step 3: Create input data
        input_data = {"input_0": [[1, 2, 3, 4, 5]]}  # Match your test input
        with open("input.json", "w") as f:
            json.dump(input_data, f)

        # Step 4: Generate witness data
        cmd_witness = [
            "ezkl",
            "gen-witness",
            "--input", "input.json",
            "--model", onnx_file_path,
            "--settings", "settings.json",
            "-o", "witness.json"
        ]
        print("Running gen-witness command:", " ".join(cmd_witness))
        result = subprocess.run(cmd_witness, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"EZKL gen-witness failed: {result.stderr}")
        print("Witness generation completed")

        # Step 5: Setup
        cmd_setup = [
            "ezkl",
            "setup",
            "--witness", "witness.json",
            "--model", onnx_file_path,
            "--settings", "settings.json"
        ]
        print("Running setup command:", " ".join(cmd_setup))
        result = subprocess.run(cmd_setup, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"EZKL setup failed: {result.stderr}")
        print("Setup completed")

        # Step 6: Generate proof
        cmd_prove = [
            "ezkl",
            "prove",
            "--witness", "witness.json",
            "--model", onnx_file_path,
            "--settings", "settings.json",
            "-o", proof_output_path
        ]
        print("Running prove command:", " ".join(cmd_prove))
        result = subprocess.run(cmd_prove, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"EZKL prove failed: {result.stderr}")

        print(f"Proof generated and stored at: {proof_output_path}")
        return proof_output_path

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"EZKL operation timed out: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"EZKL proof generation failed: {str(e)}")
