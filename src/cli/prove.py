"""
CLI module for generating proofs for models.
"""

import os
import time
import traceback
from colorama import Fore, Style

from src.runners.ezkl_runner import EzklRunner
from src.cli.base import check_model_dir, save_result, prompt_for_value

def setup_parser(subparsers):
    """
    Set up the argument parser for the prove command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    prove_parser = subparsers.add_parser('prove', help='Generate a proof for a model')
    prove_parser.add_argument('--model-dir', help='Directory containing the model')
    prove_parser.add_argument('--output-file', help='Path to save output results')
    prove_parser.add_argument('--sliced', action='store_true', help='Generate proof for sliced model')

    # Add backend group for proving
    prove_backend_group = prove_parser.add_mutually_exclusive_group(required=True)
    prove_backend_group.add_argument('--ezkl', action='store_true', help='Use EZKL backend for proving')

    return prove_parser

def run_proof(args):
    """
    Generate a proof for a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Generating proof...{Style.RESET_ALL}")

    # Prompt for model directory if not provided
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = prompt_for_value('model-dir', 'Enter the model directory')

    if not check_model_dir(args.model_dir):
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
            print(f"{Fore.GREEN}✓ EZKL proof generated in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Error: Please specify a proof backend (--ezkl).{Style.RESET_ALL}")
            return

        # Prompt for output file if not provided
        if not hasattr(args, 'output_file') or not args.output_file:
            save_output = prompt_for_value('save-output', 'Save proof results to file?', default='y', required=False).lower()
            if save_output.startswith('y'):
                default_output_file = os.path.join(args.model_dir, "proof_results.json")
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
        print(f"{Fore.RED}Error generating proof: {e}{Style.RESET_ALL}")
        traceback.print_exc()
