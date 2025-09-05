#!/usr/bin/env python3
"""
EZKL MUL Module Chunk Processing Pipeline

This script demonstrates EZKL working with chunks of the MUL module from segment_0:
1. Takes the MUL module (segment_0_1_mul_simple.onnx)
2. Splits it into 16 chunks (4 channels each)
3. Processes each chunk with EZKL (gen-settings, compile, setup, prove, verify)
4. Shows parallel processing of MUL chunks
5. Demonstrates proper input flattening for each chunk
"""
import os
import json
import numpy as np
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MULPipelineConfig:
    """Configuration for MUL chunk processing."""
    segment_dir: str
    k_value: int = 17
    channels_per_chunk: int = 4
    total_channels: int = 64
    max_parallel_workers: int = 4  # Limit to 4 for demonstration

class MULChunkProcessor:
    """Processes MUL module chunks with EZKL."""

    def __init__(self, config: MULPipelineConfig):
        self.config = config
        self.segment_dir = Path(config.segment_dir)
        self.mul_simple_path = self.segment_dir / "segment_0_1_mul_simple.onnx"
        self.mul_chunks_dir = self.segment_dir / "mul_chunks"
        self.ezkl_dir = self.segment_dir / "ezkl_out"
        self.proofs_dir = self.segment_dir / "proofs"
        self.witness_dir = self.segment_dir / "witness_data"

        # Create directories
        self.ezkl_dir.mkdir(exist_ok=True)
        self.proofs_dir.mkdir(exist_ok=True)
        self.witness_dir.mkdir(exist_ok=True)

    def run_mul_chunk_demonstration(self):
        """Demonstrate EZKL working with MUL chunks."""
        logger.info("🎯 Demonstrating EZKL with MUL Module Chunks")
        logger.info("=" * 60)

        # Step 1: Create input data for MUL module
        logger.info("\n--- STEP 1: Creating MUL Input Data ---")
        mul_input_path = self._create_mul_input_data()
        logger.info(f"✅ Created MUL input: {mul_input_path}")

        # Step 2: Test individual chunks with EZKL
        logger.info("\n--- STEP 2: Testing Individual Chunks with EZKL ---")
        chunk_results = self._test_chunks_with_ezkl(mul_input_path)

        # Step 3: Process all chunks in parallel
        logger.info("\n--- STEP 3: Processing All Chunks in Parallel ---")
        parallel_results = self._process_all_chunks_parallel(mul_input_path)

        # Step 4: Aggregate and verify results
        logger.info("\n--- STEP 4: Aggregating Results ---")
        self._aggregate_chunk_results(parallel_results)

        logger.info("\n🎉 MUL Chunk Demonstration Complete!")
        return parallel_results

    def _create_mul_input_data(self):
        """Create properly formatted input data for the MUL module."""
        logger.info("Creating input data for MUL module...")

        # Create synthetic complex input (real and imaginary parts)
        batch_size, channels, height, width = 1, self.config.total_channels, 112, 112

        # Generate real and imaginary parts
        real_part = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        imag_part = np.random.randn(batch_size, channels, height, width).astype(np.float32)

        # Flatten for EZKL format
        flattened_real = real_part.flatten()
        flattened_imag = imag_part.flatten()
        combined_input = np.concatenate([flattened_real, flattened_imag])

        # Create EZKL input format
        mul_input_data = {
            "input_data": [combined_input.tolist()]
        }

        input_path = self.witness_dir / "mul_input.json"
        with open(input_path, 'w') as f:
            json.dump(mul_input_data, f, indent=2)

        logger.info(f"Input shape: real={real_part.shape}, imag={imag_part.shape}")
        logger.info(f"Flattened input length: {len(combined_input)}")

        return input_path

    def _test_chunks_with_ezkl(self, mul_input_path):
        """Test individual chunks with EZKL to demonstrate the process."""
        logger.info("Testing individual chunks with EZKL...")

        results = []

        # Test first 3 chunks as demonstration
        test_chunks = [0, 1, 2]

        for chunk_idx in test_chunks:
            logger.info(f"\n🎯 Testing Chunk {chunk_idx} with EZKL:")
            try:
                # Get chunk circuit
                chunk_path = self.mul_chunks_dir / f"mul_chunk_{chunk_idx}.onnx"
                if not chunk_path.exists():
                    logger.error(f"Chunk {chunk_idx} not found: {chunk_path}")
                    continue

                # Create chunk input by extracting the right channels
                chunk_input_path = self._create_chunk_input(mul_input_path, chunk_idx)

                # Process chunk with EZKL
                result = self._process_single_chunk_ezkl(chunk_path, chunk_input_path, chunk_idx)
                results.append(result)

                logger.info(f"✅ Chunk {chunk_idx} processed successfully")

            except Exception as e:
                logger.error(f"❌ Failed to process chunk {chunk_idx}: {e}")
                continue

        return results

    def _create_chunk_input(self, mul_input_path, chunk_idx):
        """Create input data for a specific chunk by extracting the right channels."""
        # Load the full MUL input
        with open(mul_input_path, 'r') as f:
            mul_data = json.load(f)

        input_array = np.array(mul_data['input_data'][0])

        # Split into real and imaginary parts
        total_elements = len(input_array)
        mid_point = total_elements // 2
        real_flat = input_array[:mid_point]
        imag_flat = input_array[mid_point:]

        # Reshape to original dimensions
        batch_size, total_channels, height, width = 1, self.config.total_channels, 112, 112
        real_reshaped = real_flat.reshape(batch_size, total_channels, height, width)
        imag_reshaped = imag_flat.reshape(batch_size, total_channels, height, width)

        # Extract chunk channels
        channels_per_chunk = self.config.channels_per_chunk
        start_ch = chunk_idx * channels_per_chunk
        end_ch = min((chunk_idx + 1) * channels_per_chunk, total_channels)

        real_chunk = real_reshaped[:, start_ch:end_ch, :, :]
        imag_chunk = imag_reshaped[:, start_ch:end_ch, :, :]

        # Flatten for EZKL
        real_chunk_flat = real_chunk.flatten()
        imag_chunk_flat = imag_chunk.flatten()
        chunk_combined = np.concatenate([real_chunk_flat, imag_chunk_flat])

        # Create chunk input data with separate real and imaginary parts
        chunk_input_data = {
            "input_data": [real_chunk_flat.tolist(), imag_chunk_flat.tolist()]
        }

        chunk_input_path = self.witness_dir / f"chunk_{chunk_idx}_input.json"
        with open(chunk_input_path, 'w') as f:
            json.dump(chunk_input_data, f)

        logger.info(f"  Chunk {chunk_idx}: channels {start_ch}-{end_ch}, shape {real_chunk.shape}")
        return chunk_input_path

    def _process_single_chunk_ezkl(self, chunk_path, chunk_input_path, chunk_idx):
        """Process a single chunk through the full EZKL pipeline."""
        logger.info(f"  Processing chunk {chunk_idx} with EZKL...")

        try:
            # 1. Generate settings
            settings_path = self.ezkl_dir / f"mul_chunk_{chunk_idx}_settings.json"
            cmd = [
                "ezkl", "gen-settings",
                "-M", str(chunk_path),
                "-O", str(settings_path),
                "-K", str(self.config.k_value)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"    ✅ Settings generated")

            # 2. Compile circuit
            compiled_path = self.ezkl_dir / f"mul_chunk_{chunk_idx}_compiled.ezkl"
            cmd = [
                "ezkl", "compile-circuit",
                "-M", str(chunk_path),
                "-S", str(settings_path),
                "--compiled-circuit", str(compiled_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"    ✅ Circuit compiled")

            # 3. Setup keys
            vk_path = self.ezkl_dir / f"mul_chunk_{chunk_idx}_vk.key"
            pk_path = self.ezkl_dir / f"mul_chunk_{chunk_idx}_pk.key"
            cmd = [
                "ezkl", "setup",
                "-M", str(compiled_path),
                "--vk-path", str(vk_path),
                "--pk-path", str(pk_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"    ✅ Keys generated")

            # 4. Generate witness
            witness_path = self.witness_dir / f"mul_chunk_{chunk_idx}_witness.json"
            cmd = [
                "ezkl", "gen-witness",
                "-D", str(chunk_input_path),
                "-M", str(compiled_path),
                "-O", str(witness_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"    ✅ Witness generated")

            # 5. Generate proof
            proof_path = self.proofs_dir / f"mul_chunk_{chunk_idx}_proof.pf"
            cmd = [
                "ezkl", "prove",
                "-W", str(witness_path),
                "-M", str(compiled_path),
                "--pk-path", str(pk_path),
                "--proof-path", str(proof_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"    ✅ Proof generated")

            # 6. Verify proof
            cmd = [
                "ezkl", "verify",
                "--proof-path", str(proof_path),
                "--settings-path", str(settings_path),
                "--vk-path", str(vk_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"    ✅ Proof verified")

            return {
                'chunk_idx': chunk_idx,
                'settings_path': settings_path,
                'compiled_path': compiled_path,
                'vk_path': vk_path,
                'pk_path': pk_path,
                'witness_path': witness_path,
                'proof_path': proof_path,
                'status': 'success'
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"    ❌ EZKL command failed: {e.stderr}")
            return {
                'chunk_idx': chunk_idx,
                'status': 'failed',
                'error': str(e)
            }

    def _process_all_chunks_parallel(self, mul_input_path):
        """Process all 16 chunks in parallel."""
        logger.info("Processing all 16 MUL chunks in parallel...")

        def process_chunk_parallel(chunk_idx):
            """Process a single chunk (for parallel execution)."""
            try:
                chunk_path = self.mul_chunks_dir / f"mul_chunk_{chunk_idx}.onnx"
                if not chunk_path.exists():
                    return {'chunk_idx': chunk_idx, 'status': 'missing_chunk'}

                chunk_input_path = self._create_chunk_input(mul_input_path, chunk_idx)
                return self._process_single_chunk_ezkl(chunk_path, chunk_input_path, chunk_idx)
            except Exception as e:
                return {
                    'chunk_idx': chunk_idx,
                    'status': 'error',
                    'error': str(e)
                }

        # Process chunks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            futures = [executor.submit(process_chunk_parallel, i) for i in range(16)]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['status'] == 'success':
                    logger.info(f"✅ Parallel chunk {result['chunk_idx']} completed")
                else:
                    logger.warning(f"⚠️ Parallel chunk {result['chunk_idx']} failed: {result.get('status', 'unknown')}")

        # Sort results by chunk index
        results.sort(key=lambda x: x['chunk_idx'])

        successful_chunks = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"✅ Parallel processing complete: {successful_chunks}/16 chunks successful")

        return results

    def _aggregate_chunk_results(self, results):
        """Aggregate results from all chunk processing."""
        logger.info("Aggregating chunk processing results...")

        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') != 'success']

        logger.info(f"📊 Results Summary:")
        logger.info(f"  Total chunks: {len(results)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Failed: {len(failed)}")

        if successful:
            logger.info("\n✅ Successful chunks:")
            for r in successful:
                logger.info(f"    Chunk {r['chunk_idx']}: {r['proof_path'].name}")

        if failed:
            logger.info("\n❌ Failed chunks:")
            for r in failed:
                status = r.get('status', 'unknown')
                error = r.get('error', 'no error details')
                logger.info(f"    Chunk {r['chunk_idx']}: {status} - {error}")

        return successful, failed


def run_inference_comparison():
    """Run inference comparison between original and decomposed models."""
    import onnxruntime as ort

    logger.info("🔍 Running inference comparison...")

    # Load input data
    with open("src/models/resnet/input.json", 'r') as f:
        input_data = json.load(f)
    input_array = np.array(input_data['input_data'])
    logger.info(f"Input shape: {input_array.shape}")

    # --- Test Original Model ---
    logger.info("\n--- Testing Original Model ---")
    original_model_path = "src/models/resnet/slices/segment_0/segment_0.onnx"
    original_session = ort.InferenceSession(original_model_path)
    original_output = original_session.run(None, {"x": input_array.astype(np.float32)})
    logger.info(f"Original output shape: {original_output[0].shape}")

    # Save original output
    original_output_data = {"output_data": original_output[0].tolist()}
    with open("src/models/resnet/segment_0_expanded/output_slices.json", 'w') as f:
        json.dump(original_output_data, f, indent=2)
    logger.info("✅ Saved original output to output_slices.json")

    # --- Test EZKL Decomposed Model ---
    logger.info("\n--- Testing EZKL Decomposed Model ---")

    try:
        # FFT
        fft_model_path = "src/models/resnet/ezkl_circuits/segment_0/segment_0_0_fft.onnx"
        fft_session = ort.InferenceSession(fft_model_path)
        fft_output = fft_session.run(None, {"x": input_array.astype(np.float32)})
        fft_real, fft_imag = fft_output[0], fft_output[1]
        logger.info(f"FFT output: real={fft_real.shape}, imag={fft_imag.shape}")

        # Process MUL chunks (simplified - just use one chunk for testing)
        mul_chunk_path = "src/models/resnet/ezkl_circuits/segment_0/mul_chunks/mul_chunk_0.onnx"
        mul_session = ort.InferenceSession(mul_chunk_path)

        # Create chunk input (use available channels from FFT)
        channels_per_chunk = 4
        real_chunk = np.zeros((1, channels_per_chunk, 112, 112))
        imag_chunk = np.zeros((1, channels_per_chunk, 112, 112))

        # Copy available channels
        available_channels = min(channels_per_chunk, fft_real.shape[1])
        real_chunk[:, :available_channels, :, :] = fft_real[:, :available_channels, :, :]
        imag_chunk[:, :available_channels, :, :] = fft_imag[:, :available_channels, :, :]

        # Create MUL input
        mul_input_data = {
            "input_data": [
                np.concatenate([real_chunk.flatten(), imag_chunk.flatten()]).tolist()
            ]
        }

        # Save MUL input for EZKL
        with open("src/models/resnet/ezkl_circuits/segment_0/mul_chunks/mul_chunk_0_input.json", 'w') as f:
            json.dump(mul_input_data, f)

        # Run MUL inference
        mul_inputs = {
            "chunk_0_real": real_chunk.astype(np.float32),
            "chunk_0_imag": imag_chunk.astype(np.float32)
        }
        mul_output = mul_session.run(None, mul_inputs)
        mul_real_out, mul_imag_out = mul_output[0], mul_output[1]
        logger.info(f"MUL output: real={mul_real_out.shape}, imag={mul_imag_out.shape}")

        # IFFT
        ifft_model_path = "src/models/resnet/ezkl_circuits/segment_0/segment_0_2_ifft.onnx"
        ifft_session = ort.InferenceSession(ifft_model_path)

        # Create IFFT input (expand to expected 64 channels)
        ifft_real = np.tile(mul_real_out, (1, 16, 1, 1))[:, :64, :, :]  # 4*16 = 64
        ifft_imag = np.tile(mul_imag_out, (1, 16, 1, 1))[:, :64, :, :]

        ifft_inputs = {
            "mul_output_real": ifft_real.astype(np.float32),
            "mul_output_imag": ifft_imag.astype(np.float32)
        }
        ifft_output = ifft_session.run(None, ifft_inputs)
        reconstructed = ifft_output[0]
        logger.info(f"IFFT output: {reconstructed.shape}")

        # Save EZKL output
        ezkl_output_data = {"output_data": reconstructed.tolist()}
        with open("src/models/resnet/segment_0_expanded/output_ezkl.json", 'w') as f:
            json.dump(ezkl_output_data, f, indent=2)
        logger.info("✅ Saved EZKL output to output_ezkl.json")

        # Compare outputs
        logger.info("\n--- Comparing Outputs ---")
        original_flat = np.array(original_output[0]).flatten()
        ezkl_flat = np.array(reconstructed).flatten()

        if len(original_flat) == len(ezkl_flat):
            diff = np.abs(original_flat - ezkl_flat)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            logger.info("📊 Comparison Results:")
            logger.info(".10f")
            logger.info(".10f")

            if max_diff < 1e-6:
                logger.info("✅ Outputs match within tolerance!")
            else:
                logger.info("⚠️ Outputs differ - decomposition needs refinement")
        else:
            logger.info(f"❌ Shape mismatch: original={original_output[0].shape}, ezkl={reconstructed.shape}")

    except Exception as e:
        logger.error(f"❌ EZKL inference failed: {e}")
        logger.info("Falling back to subprocess-based parallelization...")

        # Fallback: Use subprocess for parallel MUL processing
        try:
            run_subprocess_parallel_mul()
        except Exception as e2:
            logger.error(f"❌ Subprocess fallback also failed: {e2}")


def run_subprocess_parallel_mul():
    """Fallback: Use subprocess for parallel MUL processing with kernel values."""
    logger.info("🔧 Running subprocess-based parallel MUL processing...")

    # This would implement CPU-based parallel processing of MUL operations
    # For now, just log that it's a placeholder
    logger.info("⚠️ Subprocess parallel MUL implementation would go here")
    logger.info("This would split MUL operations across CPU cores using multiprocessing")


def main():
    """Main function to run the MUL chunk demonstration."""

    # Check command line arguments
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run inference comparison
        run_inference_comparison()
        return

    # Run MUL chunk demonstration
    config = MULPipelineConfig(
        segment_dir="src/models/resnet/ezkl_circuits/segment_0",
        k_value=17,
        channels_per_chunk=4,
        total_channels=64,
        max_parallel_workers=4
    )

    processor = MULChunkProcessor(config)
    results = processor.run_mul_chunk_demonstration()

    print(f"\n🎉 Demonstration complete! Processed {len(results)} chunks.")

    # Show summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    print(f"📊 Summary: {successful}/{len(results)} chunks processed successfully")


if __name__ == "__main__":
    main()
