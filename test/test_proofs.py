# test_proofs.py
import sys
import pytest
import torch
from main import main
from src.create_nanogpt import create_nanogpt
from src.singular_workflow_proofs import run_ezkl_proof


@pytest.mark.timeout(300)
def test_nanogpt_proof():
    """
    Test NanoGPT model proof generation using EZKL, following their example approach
    """
    # Model configuration - using smaller values for testing
    model_params = {
        "vocab_size": 100,  # Reduced vocabulary size
        "n_embd": 32,  # Smaller embedding dimension
        "n_layer": 1,  # Single layer for testing
        "n_head": 2  # Reduced number of heads
    }

    # Create model
    model = create_nanogpt(model_params)
    model.eval()  # Set to evaluation mode

    # Create deterministic test input
    # Using smaller sequence length and controlled input values
    input_seq = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # (batch_size=1, seq_len=5)

    # Export to ONNX with specific input
    onnx_path = "test_nanogpt.onnx"
    torch.onnx.export(
        model,
        input_seq,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=14,  # Updated to version 13
        do_constant_folding=True,
        export_params=True
    )

    # Generate proof
    proof_path = "test_proof.pf"

    try:
        # Run the proof generation
        result = run_ezkl_proof(onnx_path, proof_path)

        # Verify proof was generated
        assert result == proof_path, f"Expected proof at {proof_path}, but got {result}"

    except Exception as e:
        pytest.fail(f"Proof generation failed: {str(e)}")



@pytest.mark.timeout(300)
def test_singular_workflow(monkeypatch, capsys):
    """
    Test the singular workflow by setting the command-line arguments,
    invoking the CLI handler, and capturing the output.
    """

    # Create a dummy input tensor with integer indices
    def mock_get_dummy_input(*args, **kwargs):
        return torch.randint(0, 50257, (1, 10), dtype=torch.int64)  # Use long/int64 tensor

    # Patch torch.randn to return integer tensor
    monkeypatch.setattr(torch, "randn", mock_get_dummy_input)

    # Set the CLI arguments
    monkeypatch.setattr(sys, "argv", ["main_cli.py", "--workflow", "singular"])

    # Call the main function directly
    main()

    # Capture the output
    captured = capsys.readouterr().out
    print("Singular workflow output:\n", captured)

    # Verify expected output pieces
    assert "Final Proof:" in captured, "Expected 'Final Proof:' message not found."
    assert "Singular workflow execution time:" in captured, "Timing info for singular workflow is missing."


@pytest.mark.timeout(300)
def test_composite_workflow(monkeypatch, capsys):
    """
    Test the composite workflow by setting the command-line arguments,
    invoking the CLI handler, and capturing the output.
    """

    # Create a dummy input tensor with integer indices
    def mock_get_dummy_input(*args, **kwargs):
        return torch.randint(0, 50257, (1, 10), dtype=torch.int64)  # Use long/int64 tensor

    # Patch torch.randn to return integer tensor
    monkeypatch.setattr(torch, "randn", mock_get_dummy_input)

    # Set the CLI arguments
    monkeypatch.setattr(sys, "argv", ["main_cli.py", "--workflow", "composite"])

    # Call the main function directly
    main()

    # Capture the output
    captured = capsys.readouterr().out
    print("Composite workflow output:\n", captured)

    # Verify expected output pieces
    assert "Aggregated Proof:" in captured, "Expected 'Aggregated Proof:' message not found."
    assert "Composite workflow execution time:" in captured, "Timing info for composite workflow is missing."
