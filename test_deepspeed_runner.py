import os
import time
import torch
from src.runners.ezkl_runner import EzklRunner

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_CPU"] = "1"

def test_deepspeed_runner():
    # Initialize the runner with the net model
    model_dir = "models/net"
    runner = EzklRunner(model_directory=model_dir)
    
    print("Testing DeepSpeed-enabled EZKL runner...")
    print(f"Using device: {torch.device('cpu')}")
    
    # Custom DeepSpeed config for CPU
    deepspeed_config = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": False  # Disable FP16 for CPU
        },
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "device": "cpu"
    }
    
    # Test witness generation with DeepSpeed
    print("\n1. Testing witness generation with DeepSpeed...")
    start_time = time.time()
    try:
        witness_result = runner.generate_witness_with_deepspeed(
            mode="sliced",
            input_file=os.path.join(model_dir, "input.json"),
            deepspeed_config=deepspeed_config
        )
        print(f"Witness generation completed in {time.time() - start_time:.2f} seconds")
        print(f"Memory usage: {witness_result['memory']:.2f} MB")
        print(f"Result: {witness_result['result']}")
    except Exception as e:
        print(f"Error in witness generation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test proof generation with DeepSpeed
    print("\n2. Testing proof generation with DeepSpeed...")
    start_time = time.time()
    try:
        proof_result = runner.prove_with_deepspeed(
            mode="sliced",
            deepspeed_config=deepspeed_config
        )
        print(f"Proof generation completed in {time.time() - start_time:.2f} seconds")
        print(f"Memory usage: {proof_result['memory']['total']:.2f} MB")
        print(f"Number of segments processed: {proof_result['num_segments_processed']}")
    except Exception as e:
        print(f"Error in proof generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deepspeed_runner() 