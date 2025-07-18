"""
CLI module for running inference on models.
"""

import os
import time
import traceback
from colorama import Fore, Style

from src.runners.model_runner import ModelRunner
from src.runners.onnx_runner import OnnxRunner
from src.runners.ezkl_runner import EzklRunner
from src.cli.base import check_model_dir, detect_model_type, save_result, prompt_for_value

def setup_parser(subparsers):
    """
    Set up the argument parser for the run command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    run_parser = subparsers.add_parser('run', help='Run inference on a model')
    run_parser.add_argument('--model-dir', help='Directory containing the model')
    run_parser.add_argument('--input-file', help='Path to input file (default: model_dir/input.json)')
    run_parser.add_argument('--output-file', help='Path to save output results')
    run_parser.add_argument('--sliced', action='store_true', help='Run inference on sliced model')

    # Add backend group for inference
    run_backend_group = run_parser.add_mutually_exclusive_group()
    run_backend_group.add_argument('--ezkl', action='store_true', help='Use EZKL backend for inference')
    run_backend_group.add_argument('--plain', action='store_true', help='Use plain inference (default)')

    return run_parser

def run_inference(args):
    """
    Run inference on a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Running inference...{Style.RESET_ALL}")

    # Prompt for model directory if not provided
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = prompt_for_value('model-dir', 'Enter the model directory')

    if not check_model_dir(args.model_dir):
        return

    # Determine if it's a PyTorch or ONNX model
    is_onnx, error_message = detect_model_type(args.model_dir)
    if error_message:
        print(error_message)
        return

    # Determine the mode (sliced or whole)
    mode = "sliced" if args.sliced else None

    # Prompt for input file if not provided
    if not hasattr(args, 'input_file') or not args.input_file:
        default_input_file = os.path.join(args.model_dir, "input.json")
        args.input_file = prompt_for_value('input-file', 'Enter the input file path', default=default_input_file, required=False)

    # Check if input file exists
    if args.input_file and not os.path.exists(args.input_file):
        print(f"{Fore.YELLOW}Warning: Input file '{args.input_file}' does not exist.{Style.RESET_ALL}")
        create_new = prompt_for_value('create-new', 'Create a new input file?', default='n', required=False).lower()
        if create_new.startswith('y'):
            try:
                with open(args.input_file, 'w') as f:
                    f.write('{"input_data": []}')
                print(f"{Fore.GREEN}Created empty input file: {args.input_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error creating input file: {e}{Style.RESET_ALL}")
                return
        else:
            args.input_file = None

    try:
        # Run inference with the appropriate backend
        if args.ezkl:
            # Run inference with ezkl
            runner = EzklRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.generate_witness(mode=mode, input_file=args.input_file)
            elapsed_time = time.time() - start_time
            print(f"{Fore.GREEN}✓ EZKL inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        else:
            # Run plain inference
            if is_onnx:
                runner = OnnxRunner(model_directory=args.model_dir)
            else:
                runner = ModelRunner(model_directory=args.model_dir)

            start_time = time.time()
            result = runner.infer(mode=mode, input_path=args.input_file)
            elapsed_time = time.time() - start_time
            print(f"{Fore.GREEN}✓ Inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")

        # Prompt for output file if not provided
        if not hasattr(args, 'output_file') or not args.output_file:
            save_output = prompt_for_value('save-output', 'Save output to file?', default='y', required=False).lower()
            if save_output.startswith('y'):
                default_output_file = os.path.join(args.model_dir, "output.json")
                args.output_file = prompt_for_value('output-file', 'Enter the output file path', default=default_output_file, required=False)

        # Save the result if output file is specified
        if args.output_file:
            try:
                save_result(result, args.output_file)
            except Exception as e:
                print(f"{Fore.RED}Error saving output file: {e}{Style.RESET_ALL}")

        # Print the result
        print(f"\n{Fore.YELLOW}Results:{Style.RESET_ALL}")
        print(result)

    except Exception as e:
        print(f"{Fore.RED}Error during inference: {e}{Style.RESET_ALL}")
        traceback.print_exc()
