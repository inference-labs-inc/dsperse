#!/usr/bin/env python3
"""
Compare EZKL and JSTprove backend execution on the Doom model.

This script runs inference with both backends and compares their performance,
predictions, and execution methods.

Usage:
    python3 compare_backends.py

Requirements:
    - dsperse environment activated
    - Doom model slices available
    - Both EZKL and JSTprove backends configured
"""

import json
import subprocess
import time
import os
import sys
from pathlib import Path

MODEL_DIR = Path("dsperse/models/doom")
SLICES_DIR = MODEL_DIR / "slices"
INPUT_FILE = MODEL_DIR / "input.json"

def run_backend(backend_name, output_file):
    """Run inference with specified backend and return timing + results."""
    # Clean previous run
    run_dir = MODEL_DIR / "run"
    if run_dir.exists():
        subprocess.run(["rm", "-rf", str(run_dir)], check=True)
    
    # Run inference
    cmd = [
        "dsperse", "run",
        "--slices-dir", str(SLICES_DIR),
        "--input-file", str(INPUT_FILE),
        "--output-file", str(output_file)
    ]
    
    env = {"DSPERSE_BACKEND": backend_name}
    
    print(f"\n{'='*60}")
    print(f"Running {backend_name.upper()} backend...")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Parse results
    with open(output_file) as f:
        data = json.load(f)
    
    return {
        "backend": backend_name,
        "elapsed_time": elapsed,
        "prediction": data.get("prediction"),
        "probabilities": data.get("probabilities"),
        "segment_methods": {k: v.get("method") for k, v in data.get("slice_results", {}).items()},
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def main():
    """Main function to run backend comparison."""
    # Check prerequisites
    if not SLICES_DIR.exists():
        print(f"Error: Slices directory not found: {SLICES_DIR}")
        print("Please ensure the Doom model is properly sliced.")
        sys.exit(1)

    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Please ensure the Doom model input.json exists.")
        sys.exit(1)

    try:
        # Run both backends
        print("Running EZKL backend...")
        ezkl_results = run_backend("ezkl", "/tmp/ezkl_comparison.json")

        print("\nRunning JSTprove backend...")
        jstprove_results = run_backend("jstprove", "/tmp/jstprove_comparison.json")

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")

        print(f"\nTiming:")
        print(f"  EZKL:     {ezkl_results['elapsed_time']:.2f}s")
        print(f"  JSTprove: {jstprove_results['elapsed_time']:.2f}s")

        print(f"\nPredictions:")
        print(f"  EZKL:     {ezkl_results['prediction']}")
        print(f"  JSTprove: {jstprove_results['prediction']}")
        print(f"  Match:    {ezkl_results['prediction'] == jstprove_results['prediction']}")

        print(f"\nSegment Methods:")
        print(f"  EZKL:")
        for seg, method in sorted(ezkl_results['segment_methods'].items()):
            print(f"    {seg}: {method}")

        print(f"  JSTprove:")
        for seg, method in sorted(jstprove_results['segment_methods'].items()):
            print(f"    {seg}: {method}")

    except Exception as e:
        print(f"Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

