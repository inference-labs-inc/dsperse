import os
import subprocess
import pytest
from pathlib import Path
from colorama import Fore, Style

# Initialize colorama
import colorama
colorama.init()

# Test data paths
TEST_MODELS_DIR = Path("~/Downloads/kubz/src/models").expanduser()
TEST_OUTPUT_DIR = Path("~/Downloads/kubz/src/models").expanduser()
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

def run_command(cmd):
    """Helper function to run CLI commands"""
    print("\nExecuting command:", Fore.CYAN + cmd + Style.RESET_ALL)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Command output:", result.stdout)
    print("Command errors:", result.stderr)
    return result

def test_full_workflow():
    """Test the complete workflow: slice -> circuitize -> run-> prove -> verify"""
    model_dir = TEST_MODELS_DIR / "doom"
    output_dir = TEST_OUTPUT_DIR / "output"

    # Slice the model
    cmd = f"kubz slice --model-dir {model_dir} --output-dir {output_dir}"
    result = run_command(cmd)
    assert result.returncode == 0
    assert os.path.exists(output_dir / "slices")
    assert os.path.exists(output_dir / "metadata.json")

    # Circuitize the slices
    cmd = f"kubz circuitize --model-path {output_dir}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Run the circuit
    cmd = f"kubz run --model-dir {output_dir} --input-file {TEST_MODELS_DIR/'doom'/'input.json'} --metadata-path {output_dir/'metadata.json'} --output-file {output_dir/'output.json'}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Prove the circuits
    run_dir = sorted(list(Path(output_dir/"run").glob("run_*")))[-1]
    print(f"RUN DIR: {run_dir}")
    prove_output = output_dir / "prove_output.json"
    cmd = f"kubz prove --run-dir {run_dir} --output-file {prove_output}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Verify the proofs
    verify_output = output_dir / "verify_output.json"
    cmd = f"kubz verify --run-dir {run_dir} --output-file {verify_output}"
    result = run_command(cmd)
    assert result.returncode == 0