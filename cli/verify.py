"""
CLI module for verifying proofs for models.
"""

import os
import time
import traceback
from colorama import Fore, Style

from src.runners.ezkl_runner import EzklRunner
from cli.base import check_model_dir, save_result, prompt_for_value

def setup_parser(subparsers):
    """
    Set up the argument parser for the verify command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    verify_parser = subparsers.add_parser('verify', help='Verify a proof for a model')
    verify_parser.add_argument('--model-dir', help='Directory containing the model')
    verify_parser.add_argument('--input-file', help='Path to input file')
    verify_parser.add_argument('--output-file', help='Path to save output results')
    verify_parser.add_argument('--sliced', action='store_true', help='Verify proof for sliced model')

    # Add backend group for verifying
    verify_backend_group = verify_parser.add_mutually_exclusive_group(required=True)
    verify_backend_group.add_argument('--ezkl', action='store_true', help='Use EZKL backend for verification')

    return verify_parser

def verify_proof(args):
    """
    Verify a proof for a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Verifying proof...{Style.RESET_ALL}")

    # Prompt for model directory if not provided
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = prompt_for_value('model-dir', 'Enter the model directory')

    if not check_model_dir(args.model_dir):
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
        # Verify proof with the appropriate backend
        if args.ezkl:
            # Verify proof with ezkl
            runner = EzklRunner(model_directory=args.model_dir)
            start_time = time.time()
            result = runner.verify(mode=mode)
            elapsed_time = time.time() - start_time
            print(f"{Fore.GREEN}✓ EZKL proof verified in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Error: Please specify a proof backend (--ezkl).{Style.RESET_ALL}")
            return

        # Prompt for output file if not provided
        if not hasattr(args, 'output_file') or not args.output_file:
            save_output = prompt_for_value('save-output', 'Save verification results to file?', default='y', required=False).lower()
            if save_output.startswith('y'):
                default_output_file = os.path.join(args.model_dir, "verification_results.json")
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
        print(f"{Fore.RED}Error verifying proof: {e}{Style.RESET_ALL}")
        traceback.print_exc()
