#!/usr/bin/env python3
"""
Kubz CLI - A command-line interface for the Kubz neural network model slicing and analysis toolkit.

This CLI allows you to slice models and run verified inference (both whole and sliced) using
different backends (ezkl, jstprove, plain).
"""

import argparse
import os
import random
import sys
import time
from typing import Optional, List, Dict, Any

import colorama
from colorama import Fore, Style


# Custom ArgumentParser that shows header and easter egg with help
class KubzArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        print_header()
        if random.random() < 0.2:  # 20% chance to show an easter egg
            print_easter_egg()
        super().print_help(file)

    def add_subparsers(self, **kwargs):
        # Ensure that subparsers are also KubzArgumentParser instances
        kwargs.setdefault("parser_class", KubzArgumentParser)
        return super().add_subparsers(**kwargs)


# Import Kubz modules
from kubz.model_slicer import ModelSlicer
from kubz.onnx_slicer import OnnxSlicer
from kubz.runners.model_runner import ModelRunner
from kubz.runners.onnx_runner import OnnxRunner
from kubz.runners.ezkl_runner import EzklRunner
from kubz.runners.jstprove_runner import JSTProveRunner

# Initialize colorama
colorama.init()

# Easter eggs
EASTER_EGGS = [
    "Did you know? Neural networks are just spicy linear algebra!",
    "Fun fact: The first neural network was created in 1943 by Warren McCulloch and Walter Pitts.",
    "Pro tip: Always normalize your inputs!",
    "Kubz fact: Slicing models helps with interpretability and verification.",
    "ZK fact: Zero-knowledge proofs allow you to prove you know something without revealing what it is.",
    "Kubz was named after the idea of 'cubes' of computation in neural networks.",
    "The answer to life, the universe, and everything is... 42 (but you need a neural network to understand why).",
    "Neural networks don't actually think. They just do math really fast.",
    "If you're reading this, you're awesome! Keep up the great work!",
    "Kubz: Making neural networks more transparent, one slice at a time.",
]


def print_header():
    """Print the Kubz CLI header with ASCII art."""
    header = f"""
{Fore.CYAN}
 ██ ▄█▀ █    ██  ▄▄▄▄   ▒███████▒
 ██▄█▒  ██  ▓██▒▓█████▄ ▒ ▒ ▒ ▄▀░
▓███▄░ ▓██  ▒██░▒██▒ ▄██░ ▒ ▄▀▒░
▓██ █▄ ▓▓█  ░██░▒██░█▀  ░ ▄▀▒   ░
▒██▒ █▄▒▒█████▓ ░▓█  ▀█▓▒███████▒
▒ ▒▒ ▓▒░▒▓▒ ▒ ▒ ░▒▓███▀▒░▒▒ ▓░▒░▒
░ ░▒ ▒░░░▒░ ░ ░ ▒░▒   ░ ░░▒ ▒ ░ ▒
░ ░░ ░  ░░░ ░ ░  ░    ░ ░ ░ ░ ░ ░
░  ░      ░      ░      ░ ░    ░
                       ░
{Style.RESET_ALL}
{Fore.YELLOW}Distributed zkML Toolkit{Style.RESET_ALL}
"""
    print(header)


def print_easter_egg():
    """Print a random easter egg."""
    print(f"\n{Fore.GREEN}🥚 {random.choice(EASTER_EGGS)}{Style.RESET_ALL}\n")


def slice_model(args):
    """Slice a model based on the provided arguments."""
    print(f"{Fore.CYAN}Slicing model...{Style.RESET_ALL}")

    if not os.path.exists(args.model_dir):
        print(
            f"{Fore.RED}Error: Model directory '{args.model_dir}' does not exist.{Style.RESET_ALL}"
        )
        return

    # Determine if it's a PyTorch or ONNX model based on model_type or auto-detect
    is_onnx = False
    if args.model_type:
        is_onnx = args.model_type.lower() == "onnx"
    else:
        # Auto-detect model type
        if os.path.exists(os.path.join(args.model_dir, "model.onnx")):
            is_onnx = True
            print(f"{Fore.YELLOW}Auto-detected ONNX model.{Style.RESET_ALL}")
        elif os.path.exists(os.path.join(args.model_dir, "model.pth")):
            print(f"{Fore.YELLOW}Auto-detected PyTorch model.{Style.RESET_ALL}")
        else:
            print(
                f"{Fore.RED}Error: No model.pth or model.onnx found in '{args.model_dir}'.{Style.RESET_ALL}"
            )
            return

    # Create output directory if specified
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        if is_onnx:
            # Slice ONNX model
            onnx_path = os.path.join(args.model_dir, "model.onnx")
            if not os.path.exists(onnx_path):
                print(
                    f"{Fore.RED}Error: ONNX model file not found at '{onnx_path}'.{Style.RESET_ALL}"
                )
                return
            slicer = OnnxSlicer(onnx_path)
            slicer.slice_model(mode="single_layer")
            print(f"{Fore.GREEN}✓ ONNX model sliced successfully!{Style.RESET_ALL}")
        else:
            # Slice PyTorch model
            pth_path = os.path.join(args.model_dir, "model.pth")
            if not os.path.exists(pth_path):
                print(
                    f"{Fore.RED}Error: PyTorch model file not found at '{pth_path}'.{Style.RESET_ALL}"
                )
                return
            slicer = ModelSlicer(model_directory=args.model_dir)
            slicer.slice_model(
                output_dir=output_dir,
                strategy="single_layer",  # Default to single_layer strategy
                input_file=args.input_file,
            )
            print(f"{Fore.GREEN}✓ PyTorch model sliced successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error slicing model: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()


def run_inference(args):
    """Run inference on a model based on the provided arguments."""
    print(f"{Fore.CYAN}Running inference...{Style.RESET_ALL}")

    if not os.path.exists(args.model_dir):
        print(
            f"{Fore.RED}Error: Model directory '{args.model_dir}' does not exist.{Style.RESET_ALL}"
        )
        return

    # Determine if it's a PyTorch or ONNX model
    is_onnx = False
    if os.path.exists(os.path.join(args.model_dir, "model.onnx")):
        is_onnx = True
    elif not os.path.exists(os.path.join(args.model_dir, "model.pth")):
        print(
            f"{Fore.RED}Error: No model.pth or model.onnx found in '{args.model_dir}'.{Style.RESET_ALL}"
        )
        return

    # Determine the mode (sliced or whole)
    mode = "sliced" if args.sliced else None

    try:
        # Run inference with the appropriate backend
        if args.ezkl:
            # Run inference with ezkl
            runner = EzklRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.generate_witness(mode=mode, input_file=args.input_file)
            elapsed_time = time.time() - start_time
            print(
                f"{Fore.GREEN}✓ EZKL inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}"
            )
        elif args.jstprove:
            # Run inference with jstProve
            runner = JSTProveRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.generate_witness(mode=mode, input_file=args.input_file)
            elapsed_time = time.time() - start_time
            print(
                f"{Fore.GREEN}✓ jstProve inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}"
            )
        else:
            # Run plain inference
            if is_onnx:
                runner = OnnxRunner(model_directory=args.model_dir)
            else:
                runner = ModelRunner(model_directory=args.model_dir)

            start_time = time.time()
            result = runner.infer(mode=mode, input_path=args.input_file)
            elapsed_time = time.time() - start_time
            print(
                f"{Fore.GREEN}✓ Inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}"
            )

        # Save the result if output file is specified
        if args.output_file:
            import json

            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"{Fore.GREEN}✓ Results saved to {args.output_file}{Style.RESET_ALL}")

        # Print the result
        print(f"\n{Fore.YELLOW}Results:{Style.RESET_ALL}")
        print(result)

    except Exception as e:
        print(f"{Fore.RED}Error during inference: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()


def run_proof(args):
    """Generate a proof for a model based on the provided arguments."""
    print(f"{Fore.CYAN}Generating proof...{Style.RESET_ALL}")

    if not os.path.exists(args.model_dir):
        print(
            f"{Fore.RED}Error: Model directory '{args.model_dir}' does not exist.{Style.RESET_ALL}"
        )
        return

    # Determine the mode (sliced or whole)
    mode = "sliced" if args.sliced else None

    try:
        # Generate proof with the appropriate backend
        if args.ezkl:
            # Generate proof with ezkl
            runner = EzklRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.prove(mode=mode)
            elapsed_time = time.time() - start_time
            print(
                f"{Fore.GREEN}✓ EZKL proof generated in {elapsed_time:.2f} seconds!{Style.RESET_ALL}"
            )
        elif args.jstprove:
            # Generate proof with jstProve
            runner = JSTProveRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.prove(mode=mode)
            elapsed_time = time.time() - start_time
            print(
                f"{Fore.GREEN}✓ jstProve proof generated in {elapsed_time:.2f} seconds!{Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.RED}Error: Please specify a proof backend (--ezkl or --jstprove).{Style.RESET_ALL}"
            )
            return

        # Save the result if output file is specified
        if args.output_file:
            import json

            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"{Fore.GREEN}✓ Results saved to {args.output_file}{Style.RESET_ALL}")

        # Print the result
        print(f"\n{Fore.YELLOW}Results:{Style.RESET_ALL}")
        print(result)

    except Exception as e:
        print(f"{Fore.RED}Error generating proof: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()


def verify_proof(args):
    """Verify a proof for a model based on the provided arguments."""
    print(f"{Fore.CYAN}Verifying proof...{Style.RESET_ALL}")

    if not os.path.exists(args.model_dir):
        print(
            f"{Fore.RED}Error: Model directory '{args.model_dir}' does not exist.{Style.RESET_ALL}"
        )
        return

    # Determine the mode (sliced or whole)
    mode = "sliced" if args.sliced else None

    try:
        # Verify proof with the appropriate backend
        if args.ezkl:
            # Verify proof with ezkl
            runner = EzklRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.verify(mode=mode)
            elapsed_time = time.time() - start_time
            print(
                f"{Fore.GREEN}✓ EZKL proof verified in {elapsed_time:.2f} seconds!{Style.RESET_ALL}"
            )
        elif args.jstprove:
            # Verify proof with jstProve
            runner = JSTProveRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.verify(mode=mode, input_file=args.input_file)
            elapsed_time = time.time() - start_time
            print(
                f"{Fore.GREEN}✓ jstProve proof verified in {elapsed_time:.2f} seconds!{Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.RED}Error: Please specify a proof backend (--ezkl or --jstprove).{Style.RESET_ALL}"
            )
            return

        # Save the result if output file is specified
        if args.output_file:
            import json

            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"{Fore.GREEN}✓ Results saved to {args.output_file}{Style.RESET_ALL}")

        # Print the result
        print(f"\n{Fore.YELLOW}Results:{Style.RESET_ALL}")
        print(result)

    except Exception as e:
        print(f"{Fore.RED}Error verifying proof: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()


def main():
    """Main entry point for the Kubz CLI."""
    # Create the main parser
    parser = KubzArgumentParser(
        description="Kubz - Distributed zkML Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Made with {Fore.RED}❤️{Style.RESET_ALL}  by the Inference Labs team",
    )

    # Add version argument
    parser.add_argument("--version", action="version", version="Kubz CLI v1.0.0")

    # Add easter egg argument
    parser.add_argument("--easter-egg", action="store_true", help=argparse.SUPPRESS)

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Slice command
    slice_parser = subparsers.add_parser("slice", help="Slice a model into segments")
    slice_parser.add_argument(
        "--model-dir", required=True, help="Directory containing the model"
    )
    slice_parser.add_argument(
        "--output-dir",
        help="Directory to save the sliced model (default: model_dir/slices)",
    )
    slice_parser.add_argument(
        "--model-type",
        choices=["onnx", "pth"],
        help="Type of model to slice (auto-detected if not specified)",
    )
    slice_parser.add_argument(
        "--input-file",
        help="Path to input file for analysis (default: model_dir/input.json)",
    )

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference on a model")
    infer_parser.add_argument(
        "--model-dir", required=True, help="Directory containing the model"
    )
    infer_parser.add_argument(
        "--input-file", help="Path to input file (default: model_dir/input.json)"
    )
    infer_parser.add_argument("--output-file", help="Path to save output results")
    infer_parser.add_argument(
        "--sliced", action="store_true", help="Run inference on sliced model"
    )

    # Add backend group for inference
    infer_backend_group = infer_parser.add_mutually_exclusive_group()
    infer_backend_group.add_argument(
        "--ezkl", action="store_true", help="Use EZKL backend for inference"
    )
    infer_backend_group.add_argument(
        "--jstprove", action="store_true", help="Use jstProve backend for inference"
    )
    infer_backend_group.add_argument(
        "--plain", action="store_true", help="Use plain inference (default)"
    )

    # Prove command
    prove_parser = subparsers.add_parser("prove", help="Generate a proof for a model")
    prove_parser.add_argument(
        "--model-dir", required=True, help="Directory containing the model"
    )
    prove_parser.add_argument("--output-file", help="Path to save output results")
    prove_parser.add_argument(
        "--sliced", action="store_true", help="Generate proof for sliced model"
    )

    # Add backend group for proving
    prove_backend_group = prove_parser.add_mutually_exclusive_group(required=True)
    prove_backend_group.add_argument(
        "--ezkl", action="store_true", help="Use EZKL backend for proving"
    )
    prove_backend_group.add_argument(
        "--jstprove", action="store_true", help="Use jstProve backend for proving"
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a proof for a model")
    verify_parser.add_argument(
        "--model-dir", required=True, help="Directory containing the model"
    )
    verify_parser.add_argument("--input-file", help="Path to input file (for jstProve)")
    verify_parser.add_argument("--output-file", help="Path to save output results")
    verify_parser.add_argument(
        "--sliced", action="store_true", help="Verify proof for sliced model"
    )

    # Add backend group for verifying
    verify_backend_group = verify_parser.add_mutually_exclusive_group(required=True)
    verify_backend_group.add_argument(
        "--ezkl", action="store_true", help="Use EZKL backend for verification"
    )
    verify_backend_group.add_argument(
        "--jstprove", action="store_true", help="Use jstProve backend for verification"
    )

    # Parse arguments
    args = parser.parse_args()

    # Print header
    print_header()

    # Handle easter egg
    if args.easter_egg:
        print_easter_egg()
        return

    # Handle commands
    if args.command == "slice":
        slice_model(args)
    elif args.command == "infer":
        # Set plain as default if no backend specified
        if not (args.ezkl or args.jstprove):
            args.plain = True
        run_inference(args)
    elif args.command == "prove":
        run_proof(args)
    elif args.command == "verify":
        verify_proof(args)
    else:
        # If no command is provided, show help
        parser.print_help()
        # Show an easter egg 20% of the time
        if random.random() < 0.2:
            print_easter_egg()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
