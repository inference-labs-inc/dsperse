#!/usr/bin/env python3
"""
Test script for verifying slices 41 and 52 with JStprove backend.
Demonstrates circuit execution, proof generation, and verification.
"""

import json
import onnxruntime as ort
from pathlib import Path
import numpy as np
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.utils.utils import Utils


def load_slice_metadata(slice_dir: Path) -> dict:
    """Load metadata for a slice."""
    metadata_path = slice_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)


def prepare_input(slice_dir: Path, input_data: list) -> dict:
    """Prepare input JSON for slice execution."""
    # Get slice input shape from metadata
    metadata = load_slice_metadata(slice_dir)
    slice_info = metadata.get("slices", [{}])[0]
    
    # Create input JSON matching slice requirements
    input_json = {
        "input_data": input_data
    }
    return input_json


def test_slice_onnx(slice_dir: Path, input_data: np.ndarray) -> tuple:
    """Test slice execution with ONNX Runtime."""
    slice_onnx = slice_dir / "payload" / f"{slice_dir.name}.onnx"
    
    if not slice_onnx.exists():
        raise FileNotFoundError(f"Slice ONNX not found: {slice_onnx}")
    
    session = ort.InferenceSession(str(slice_onnx))
    input_name = session.get_inputs()[0].name
    
    # Reshape input to match expected shape
    input_shape = session.get_inputs()[0].shape
    expected_size = np.prod([s for s in input_shape if isinstance(s, int)])
    
    if input_data.size >= expected_size:
        input_reshaped = input_data[:expected_size].reshape(input_shape)
    else:
        input_reshaped = input_data.reshape(input_shape)
    
    outputs = session.run(None, {input_name: input_reshaped.astype(np.float32)})
    
    return outputs[0], input_name


def test_slice_jstprove(slice_dir: Path, input_data: np.ndarray, output_dir: Path):
    """Test slice execution with JStprove circuit."""
    slice_id = slice_dir.name
    metadata = load_slice_metadata(slice_dir)
    slice_info = metadata.get("slices", [{}])[0]
    
    # Find JStprove circuit files
    jstprove_dir = slice_dir / "payload" / "jstprove"
    if not jstprove_dir.exists():
        jstprove_dir = slice_dir / "payload" / "jstprove_circuitization"
    
    circuit_path = None
    for pattern in ["*_circuit.txt", "*circuit.txt"]:
        circuits = list(jstprove_dir.glob(pattern))
        if circuits:
            circuit_path = circuits[0]
            break
    
    if not circuit_path or not circuit_path.exists():
        raise FileNotFoundError(f"Circuit file not found in {jstprove_dir}")
    
    # Prepare input JSON
    input_json_path = output_dir / f"{slice_id}_input.json"
    input_json = prepare_input(slice_dir, input_data.flatten().tolist())
    with open(input_json_path, 'w') as f:
        json.dump(input_json, f, indent=2)
    
    # Initialize JStprove backend
    jstprove = JSTprove()
    
    # Generate witness
    output_json_path = output_dir / f"{slice_id}_output.json"
    witness_path = output_dir / f"{slice_id}_witness.bin"
    
    print(f"  Generating witness for {slice_id}...")
    success, witness_data = jstprove.generate_witness(
        input_file=str(input_json_path),
        model_path=str(circuit_path),
        output_file=str(output_json_path)
    )
    
    if not success:
        raise RuntimeError(f"Witness generation failed for {slice_id}")
    
    # Generate proof
    proof_path = output_dir / f"{slice_id}_proof.json"
    print(f"  Generating proof for {slice_id}...")
    success, proof_file = jstprove.prove(
        witness_path=str(witness_path),
        circuit_path=str(circuit_path),
        proof_path=str(proof_path)
    )
    
    if not success:
        raise RuntimeError(f"Proof generation failed for {slice_id}")
    
    # Verify proof
    print(f"  Verifying proof for {slice_id}...")
    success = jstprove.verify(
        proof_path=str(proof_path),
        circuit_path=str(circuit_path),
        input_path=str(input_json_path),
        output_path=str(output_json_path),
        witness_path=str(witness_path)
    )
    
    if not success:
        raise RuntimeError(f"Proof verification failed for {slice_id}")
    
    print(f"  ✓ {slice_id} verified successfully!")
    
    return {
        "slice_id": slice_id,
        "circuit_path": str(circuit_path),
        "input_path": str(input_json_path),
        "output_path": str(output_json_path),
        "witness_path": str(witness_path),
        "proof_path": str(proof_path),
        "verified": success
    }


def main():
    """Main test function."""
    print("=" * 70)
    print("Testing Slices 41 and 52 with JStprove")
    print("=" * 70)
    
    # Setup paths
    slices_dir = Path(__file__).parent
    slice_41_dir = slices_dir / "slice_41"
    slice_52_dir = slices_dir / "slice_52"
    output_dir = slices_dir / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Create dummy input data (64 channels, 80x80 for slice 41, 40x40 for slice 52)
    # In practice, this would come from previous slice outputs
    input_41 = np.random.randn(1, 64, 80, 80).astype(np.float32)
    input_52 = np.random.randn(1, 64, 40, 40).astype(np.float32)
    
    results = {}
    
    # Test slice 41
    print("\n[Testing Slice 41]")
    print("-" * 70)
    try:
        # First test with ONNX
        print("  Testing ONNX execution...")
        onnx_output, input_name = test_slice_onnx(slice_41_dir, input_41)
        print(f"  ✓ ONNX execution successful (output shape: {onnx_output.shape})")
        
        # Then test with JStprove
        print("  Testing JStprove circuit execution...")
        result = test_slice_jstprove(slice_41_dir, input_41, output_dir)
        results["slice_41"] = result
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["slice_41"] = {"error": str(e)}
    
    # Test slice 52
    print("\n[Testing Slice 52]")
    print("-" * 70)
    try:
        # First test with ONNX
        print("  Testing ONNX execution...")
        onnx_output, input_name = test_slice_onnx(slice_52_dir, input_52)
        print(f"  ✓ ONNX execution successful (output shape: {onnx_output.shape})")
        
        # Then test with JStprove
        print("  Testing JStprove circuit execution...")
        result = test_slice_jstprove(slice_52_dir, input_52, output_dir)
        results["slice_52"] = result
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["slice_52"] = {"error": str(e)}
    
    # Save results
    results_path = output_dir / "verification_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for slice_id, result in results.items():
        if "error" in result:
            print(f"  {slice_id}: ✗ Failed - {result['error']}")
        else:
            print(f"  {slice_id}: ✓ Verified")
            print(f"    Circuit: {result['circuit_path']}")
            print(f"    Proof: {result['proof_path']}")
    
    print(f"\nResults saved to: {results_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

