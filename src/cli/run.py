"""
CLI module for running inference on models.
"""

import os
import time
import traceback

from colorama import Fore, Style

from src.cli.base import check_model_dir, save_result, prompt_for_value, logger
from src.runner import Runner


def setup_parser(subparsers):
    """
    Set up the argument parser for the run command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    run_parser = subparsers.add_parser('run', help='Run inference on a model')
    run_parser.add_argument('--slices-dir', help='Directory containing the slices')
    run_parser.add_argument('--run-metadata-path', help='Path to run metadata.json (auto-generated if not provided)')
    run_parser.add_argument('--input-file', help='Path to input file (default: parent_of_slices/input.json)')
    run_parser.add_argument('--output-file', help='Path to save output results (default: parent_of_slices/output.json)')

    return run_parser

def run_inference(args):
    """
    Run inference on a model based on the provided arguments.
    
    This command requires a slices directory. The parent directory of the slices
    is treated as the model directory, which is used for defaults like input/output paths.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Running inference...{Style.RESET_ALL}")
    logger.info("Starting model inference")

    # Require slices directory
    if not hasattr(args, 'slices_dir') or not args.slices_dir:
        args.slices_dir = prompt_for_value('slices-dir', 'Enter the slices directory')

    if not check_model_dir(args.slices_dir):
        return

    # Validate slices directory has metadata and normalize to the actual slices directory
    meta_in_dir = os.path.exists(os.path.join(args.slices_dir, 'metadata.json'))
    meta_in_sub = os.path.exists(os.path.join(args.slices_dir, 'slices', 'metadata.json'))

    if not (meta_in_dir or meta_in_sub):
        print(f"{Fore.YELLOW}Warning: No slices metadata found at the provided path. Please slice the model first.{Style.RESET_ALL}")
        logger.error("Run requires a valid slices directory with metadata.json")
        return

    if meta_in_dir:
        slices_dir_effective = args.slices_dir
        model_dir = os.path.dirname(args.slices_dir.rstrip('/')) or '.'
    else:
        # metadata inside a 'slices' subfolder; treat provided path as model_dir
        slices_dir_effective = os.path.join(args.slices_dir, 'slices')
        model_dir = args.slices_dir

    # Get run metadata path if provided, otherwise None (Runner will auto-generate)
    run_metadata_path = args.run_metadata_path if hasattr(args, 'run_metadata_path') and args.run_metadata_path else None

    # Prompt for input file if not provided
    if not hasattr(args, 'input_file') or not args.input_file:
        # Set default input file path based on model_dir (parent of slices)
        default_input_file = os.path.join(model_dir, "input.json")
        args.input_file = prompt_for_value('input-file', 'Enter the input file path', default=default_input_file, required=True)

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

    # Enforce input file requirement
    if not args.input_file:
        print(f"{Fore.RED}Error: input-file is required and must exist. Aborting.{Style.RESET_ALL}")
        logger.error("Input file missing; aborting run.")
        return

    try:
        # Use the Runner class for inference
        logger.info("Using Runner class for model inference")
        logger.info(f"Model path: {model_dir}, Slices path: {slices_dir_effective}")
        
        start_time = time.time()
        runner = Runner(
            model_path=model_dir,
            slices_path=slices_dir_effective,
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
                default_output_file = os.path.join(model_dir, "output.json")
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
        print(f"Prediction: {result.get('prediction', 'N/A')}")
        print(f"Probabilities: {result.get('probabilities', 'N/A')}")

    except Exception as e:
        error_msg = f"Error during inference: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
