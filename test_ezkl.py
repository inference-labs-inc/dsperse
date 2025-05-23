from src.runners.ezkl_runner import EzklRunner

def main():
    # Initialize the EZKL runner with the net model
    model_dir = "models/net"
    runner = EzklRunner(model_dir)
    
    # Run witness generation
    print("Generating witness...")
    result = runner.generate_witness()
    print("Witness generation result:", result)
    
    # Run proof generation
    print("\nGenerating proof...")
    result = runner.prove()
    print("Proof generation result:", result)
    
    # Run verification
    print("\nVerifying proof...")
    result = runner.verify()
    print("Verification result:", result)

if __name__ == "__main__":
    main() 