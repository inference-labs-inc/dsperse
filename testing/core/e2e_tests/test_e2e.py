import os
from pyexpat import model
import subprocess
import pytest
from pathlib import Path
from colorama import Fore, Style

# Initialize colorama
import colorama
from torch import mode
colorama.init()

# Test data paths
TEST_MODELS_DIR = Path("./src/models")

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
    output_dir = model_dir / "slices"

    # Slice the model
    cmd = f"dsperse slice --model-dir {model_dir}"
    result = run_command(cmd)
    assert result.returncode == 0
    assert os.path.exists(output_dir / "slices")
    assert os.path.exists(output_dir / "metadata.json")

    # Circuitize the slices
    cmd = f"dsperse circuitize --slices-path {output_dir}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Run the circuit
    cmd = f"dsperse run --slices-dir {output_dir} --input-file {TEST_MODELS_DIR/'input.json'} --output-file {output_dir/'output.json'}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Prove the circuits
    run_dir = sorted(list(Path(output_dir/"run").glob("run_*")))[-1]
    print(f"RUN DIR: {run_dir}")
    prove_output = output_dir / "prove_output.json"
    cmd = f"dsperse prove --run-dir {run_dir} --output-file {prove_output}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Verify the proofs
    verify_output = output_dir / "verify_output.json"
    cmd = f"dsperse verify --run-dir {run_dir} --output-file {verify_output}"
    result = run_command(cmd)
    assert result.returncode == 0

def test_layer_flag():
    """Test the complete workflow with layer selection flag: slice -> circuitize -> run-> prove -> verify"""
    model_dir = TEST_MODELS_DIR / "doom"
    output_dir = model_dir / "slices"

    layers = "0,2,4" #First, middle, last layer
    
    # Slice the model
    cmd = f"dsperse slice --model-dir {model_dir}"
    result = run_command(cmd)
    assert result.returncode == 0
    assert os.path.exists(output_dir / "slices")
    assert os.path.exists(output_dir / "metadata.json")

    # Circuitize the slices
    cmd = f"dsperse circuitize --slices-path {output_dir} --layers {layers}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Run the circuit
    cmd = f"dsperse run --slices-dir {output_dir} --input-file {TEST_MODELS_DIR/'input.json'} --output-file {output_dir/'output.json'}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Prove the circuits
    run_dir = sorted(list(Path(output_dir/"run").glob("run_*")))[-1]
    print(f"RUN DIR: {run_dir}")
    prove_output = output_dir / "prove_output.json"
    cmd = f"dsperse prove --run-dir {run_dir} --output-file {prove_output}"
    result = run_command(cmd)
    assert result.returncode == 0

    # Verify the proofs
    verify_output = output_dir / "verify_output.json"
    cmd = f"dsperse verify --run-dir {run_dir} --output-file {verify_output}"
    result = run_command(cmd)
    assert result.returncode == 0