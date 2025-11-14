"""
CLI module for running inference on models.
"""

import os
import time
import traceback

from colorama import Fore, Style

from dsperse.src.cli.base import check_model_dir, save_result, prompt_for_value, logger, normalize_path
from dsperse.src.run.runner import Runner


def setup_parser(subparsers):
    """
    Set up the argument parser for the run command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    run_parser = subparsers.add_parser('run', aliases=['r'], help='Run inference on a model')
    # Ensure canonical command name even when alias is used
    run_parser.set_defaults(command='run')

    # Arguments with aliases/shorthands
    run_parser.add_argument('--path', '-p', '--slices-dir', '--slices-directory', '--slices', '--sd', '-s', dest='path',
                            help='Path to the slices directory or a .dsperse/.dslice file')
    run_parser.add_argument('--run-metadata-path', help='Path to run metadata.json (auto-generated if not provided)')
    run_parser.add_argument('--input-file', '--input', '--if', '-i', dest='input_file',
                            help='Path to input file (default: parent_dir/input.json)')
    run_parser.add_argument('--output-file', '-o', dest='output_file',
                            help='Path to save output results (default: parent_dir/output.json)')

    return run_parser

def run_inference(args):
    """
    Run inference on a model based on the provided arguments.
    
    Accepts a slices directory or a .dsperse/.dslice file. The parent directory
    of the slices (or the file) is treated as the model directory for default IO paths.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Running inference...{Style.RESET_ALL}")
    logger.info("Starting model inference")

    # Resolve target path (slices dir or .dsperse/.dslice file)
    target_path = getattr(args, 'path', None) or getattr(args, 'slices_dir', None)
    if not target_path:
        target_path = prompt_for_value('path', 'Enter the path to the slices directory or .dsperse file')
    target_path = normalize_path(target_path)

    if not check_model_dir(target_path):
        return

    # Derive model_dir and slices hint for Runner
    slices_dir_effective = None
    if os.path.isfile(target_path):
        # File input (e.g., slices.dsperse or slice_0.dslice)
        model_dir = os.path.dirname(target_path) or '.'
        slices_dir_effective = target_path
    else:
        # Directory input
        meta_in_dir = os.path.exists(os.path.join(target_path, 'metadata.json'))
        meta_in_sub = os.path.exists(os.path.join(target_path, 'slices', 'metadata.json'))
        if meta_in_dir:
            slices_dir_effective = target_path
            model_dir = os.path.dirname(target_path.rstrip('/')) or '.'
        elif meta_in_sub:
            slices_dir_effective = os.path.join(target_path, 'slices')
            model_dir = target_path
        else:
            # No immediate metadata found; Runner will attempt to detect dsperse inside the directory
            slices_dir_effective = None
            model_dir = target_path

    # Normalize derived paths
    if slices_dir_effective:
        slices_dir_effective = normalize_path(slices_dir_effective)
    model_dir = normalize_path(model_dir)

    # Get run metadata path if provided, otherwise None (Runner will auto-generate)
    run_metadata_path = args.run_metadata_path if hasattr(args, 'run_metadata_path') and args.run_metadata_path else None
    if run_metadata_path:
        run_metadata_path = normalize_path(run_metadata_path)

    # Prompt for input file if not provided
    if not hasattr(args, 'input_file') or not args.input_file:
        # Set default input file path based on model_dir (parent of slices)
        default_input_file = os.path.join(model_dir, "input.json")
        args.input_file = prompt_for_value('input-file', 'Enter the input file path', default=default_input_file, required=True)

    # Check if input file exists
    if args.input_file:
        args.input_file = normalize_path(args.input_file)
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
        # Runner expects a path to slices (dirs), a .dslice, a .dsperse, or a model dir with slices
        slice_path = slices_dir_effective or model_dir
        runner = Runner(
            slice_path=slice_path,
            run_metadata_path=run_metadata_path
        )
        result = runner.run(args.input_file)
        elapsed_time = time.time() - start_time
        
        print(f"{Fore.GREEN}✓ Inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        logger.info(f"Inference completed in {elapsed_time:.2f} seconds")

        # Prompt for output file if not provided
        if not hasattr(args, 'output_file') or not args.output_file:
            save_output = prompt_for_value('save-output', 'Save output to file?', default='n', required=False).lower()
            if save_output.startswith('y'):
                default_output_file = os.path.join(model_dir, "output.json")
                args.output_file = prompt_for_value('output-file', 'Enter the output file path', default=default_output_file, required=False)

        # Save the result if an output file is specified
        if args.output_file:
            try:
                args.output_file = normalize_path(args.output_file)
                save_result(result, args.output_file)
                # Explicitly inform user where the inference results were saved (in addition to save_result's checkmark)
                print(f"{Fore.GREEN}Results saved to {args.output_file}{Style.RESET_ALL}")
                logger.info(f"Results saved to {args.output_file}")
            except Exception as e:
                error_msg = f"Error saving output file: {e}"
                print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                logger.error(error_msg)

        # Print the result
        print(f"\n{Fore.YELLOW}Results:{Style.RESET_ALL}")
        print(f"Prediction: {result.get('prediction', 'N/A')}")
        print(f"Final Probabilities: {result.get('probabilities')[0] if result.get('probabilities') else 'N/A'}")

        # Print method information for each slice
        slice_results = result.get('slice_results', {})
        if slice_results:
            print("\nSlice Methods:")
            for slice_name, slice_info in slice_results.items():
                print(f"{slice_name}: {slice_info.get('method', 'N/A')}")


    except Exception as e:
        error_msg = f"Error during inference: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
