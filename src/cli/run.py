"""
CLI module for running inference on models.
"""

import os
import time
import traceback

from colorama import Fore, Style

from src.cli.base import check_model_dir, save_result, prompt_for_value, logger
from src.runners.runner import Runner


def setup_parser(subparsers):
    """
    Set up the argument parser for the run command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    run_parser = subparsers.add_parser('run', help='Run inference on a model')
    run_parser.add_argument('--model-dir', help='Directory containing the model (can also be a slices directory)')
    run_parser.add_argument('--slices-dir', help='Directory containing the slices (default: model_dir/slices)')
    run_parser.add_argument('--metadata-path', help='Path to slices metadata.json (default: slices_dir/metadata.json)')
    run_parser.add_argument('--run-metadata-path', help='Path to run metadata.json (auto-generated if not provided)')
    run_parser.add_argument('--input-file', help='Path to input file (default: model_dir/input.json)')
    run_parser.add_argument('--output-file', help='Path to save output results')

    return run_parser

def run_inference(args):
    """
    Run inference on a model based on the provided arguments.
    
    The function accepts either a model directory or a slices directory as input.
    If a slices directory is provided, the function will attempt to determine the
    parent model directory automatically.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Running inference...{Style.RESET_ALL}")
    logger.info("Starting model inference")

    # Prompt for model or slices directory if not provided
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = prompt_for_value('model-dir', 'Enter the model or slices directory')

    if not check_model_dir(args.model_dir):
        return

    # Determine if the provided path is a model directory or a slices directory
    # Check if the path ends with 'slices' or if it contains a metadata.json file
    is_slices_dir = args.model_dir.rstrip('/').endswith('slices') or os.path.exists(os.path.join(args.model_dir, 'metadata.json'))
    
    # Set model_dir and slices_dir based on what was provided
    if is_slices_dir:
        # User provided a slices directory
        slices_dir = args.model_dir
        # Try to determine the model directory (parent of slices)
        parent_dir = os.path.dirname(args.model_dir.rstrip('/'))
        if parent_dir:
            args.model_dir = parent_dir
        # If we couldn't determine a parent, keep the original as both model and slices
    else:
        # User provided a model directory
        slices_dir = args.slices_dir if hasattr(args, 'slices_dir') and args.slices_dir else os.path.join(args.model_dir, 'slices')
    
    # Get metadata path if provided, otherwise use default
    metadata_path = args.metadata_path if hasattr(args, 'metadata_path') and args.metadata_path else None
    
    # Get run metadata path if provided, otherwise use default
    run_metadata_path = args.run_metadata_path if hasattr(args, 'run_metadata_path') and args.run_metadata_path else None

    # Prompt for input file if not provided
    if not hasattr(args, 'input_file') or not args.input_file:
        # Set default input file path based on model_dir
        # If the user provided a slices directory, we've already set model_dir to its parent
        default_input_file = os.path.join(args.model_dir, "input.json")
        args.input_file = prompt_for_value('input-file', 'Enter the input file path', default=default_input_file, required=False)

    # Check if input file exists
    if args.input_file and not os.path.exists(args.input_file):
        print(f"{Fore.YELLOW}Warning: Input file '{args.input_file}' does not exist.{Style.RESET_ALL}")
        retry_option = prompt_for_value('retry-option', 'Enter a different file path or "q" to quit', required=False).lower()
        if retry_option == 'q':
            print(f"{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
            logger.info("Operation cancelled by user")
            return
        elif retry_option:
            if os.path.exists(retry_option):
                args.input_file = retry_option
                print(f"{Fore.GREEN}Using input file: {args.input_file}{Style.RESET_ALL}")
                logger.info(f"Using input file: {args.input_file}")
            else:
                print(f"{Fore.RED}Error: File '{retry_option}' does not exist. Aborting.{Style.RESET_ALL}")
                logger.error(f"File '{retry_option}' does not exist")
                return
        else:
            args.input_file = None

    try:
        # Use the Runner class for inference
        logger.info("Using Runner class for model inference")
        logger.info(f"Model path: {args.model_dir}, Slices path: {slices_dir}")
        
        start_time = time.time()
        runner = Runner(
            model_path=args.model_dir,
            slices_path=slices_dir,
            metadata_path=metadata_path,
            run_metadata_path=run_metadata_path
        )
        result = runner.run(args.input_file)
        elapsed_time = time.time() - start_time
        
        print(f"{Fore.GREEN}✓ Inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        logger.info(f"Inference completed in {elapsed_time:.2f} seconds")

        # Prompt for output file if not provided
        if not hasattr(args, 'output_file') or not args.output_file:
            save_output = prompt_for_value('save-output', 'Save output to file?', default='y', required=False).lower()
            if save_output.startswith('y'):
                default_output_file = os.path.join(args.model_dir, "output.json")
                args.output_file = prompt_for_value('output-file', 'Enter the output file path', default=default_output_file, required=False)

        # Save the result if an output file is specified
        if args.output_file:
            try:
                save_result(result, args.output_file)
                logger.info(f"Results saved to {args.output_file}")
            except Exception as e:
                error_msg = f"Error saving output file: {e}"
                print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                logger.error(error_msg)

        # Print the result
        print(f"\n{Fore.YELLOW}Results:{Style.RESET_ALL}")
        print(result)

    except Exception as e:
        error_msg = f"Error during inference: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
