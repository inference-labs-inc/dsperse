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
    run_parser.add_argument('--input-file', '--input', '--if', '-i', dest='input_file',
                            help='Path to input file (default: parent_dir/input.json)')
    run_parser.add_argument('--output-file', '-o', dest='output_file',
                            help='Path to save output results (default: parent_dir/output.json)')

    # Optional backend selector (applies only when a slice has multiple circuit backends compiled)
    run_parser.add_argument('--backends', '--backend', '-b', dest='force_backend', choices=['jstprove', 'ezkl', 'onnx'],
                            help='Backend to use at runtime when a slice was compiled with multiple circuit backends. '
                                 'If a slice has only one circuit backend compiled, this flag is ignored for that slice (unless you choose "onnx" to skip circuits).')

    run_parser.add_argument('--parallel', '--threads', type=int, default=1, dest='threads',
                            help='Number of parallel processes to use for execution (default: 1)')

    run_parser.add_argument('--batch', action='store_true', default=False,
                            help='Use JSTprove batch witness generation for tiled slices (loads circuit once for all tiles)')

    run_parser.add_argument('--workers', type=int, default=1, dest='workers',
                            help='Number of parallel slice workers for outer parallelization (default: 1). '
                                 'Slices whose inputs are satisfied run concurrently.')

    run_parser.add_argument('--run-dir', '--resume', dest='run_dir',
                            help='Directory for run outputs (default: alongside slices). If it contains metadata.json, or is a parent of run_ subdirs, it will resume.')

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

    # Prompt for input file if not provided
    if not getattr(args, 'input_file', None):
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
            retry_option = normalize_path(retry_option)
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
        threads = getattr(args, 'threads', 1) or 1
        workers = getattr(args, 'workers', 1) or 1
        run_dir = getattr(args, 'run_dir', None)

        runner = Runner(
            run_dir=run_dir,
            threads=threads,
            workers=workers,
            batch=getattr(args, 'batch', False),
        )
        # Pass optional backend selection directly to Runner.run()
        result = runner.run(
            args.input_file,
            slice_path=slices_dir_effective or model_dir,
            output_path=run_dir,
            backend=getattr(args, 'force_backend', None)
        )
        elapsed_time = time.time() - start_time

        print(f"{Fore.GREEN}✓ Inference completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        logger.info(f"Inference completed in {elapsed_time:.2f} seconds")

        # Do not prompt to save output; only save if an explicit output file path is provided via CLI
        # Print the run directory that was just created
        if getattr(runner, 'last_run_dir', None):
            run_dir_path = str(runner.last_run_dir)
            print(f"Run data saved to {run_dir_path}")
            logger.info(f"Run data saved to {run_dir_path}")

        # Save the result only if an output file was explicitly specified
        if getattr(args, 'output_file', None):
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
        tensor_shape = result.get('tensor_shape')
        if tensor_shape:
            print(f"Output shape: {tensor_shape}")

        # Print method information for each slice
        slice_results = result.get('slice_results', {})
        if slice_results:
            print("\nSlice Methods:")
            for slice_name, slice_info in slice_results.items():
                method = slice_info.method if hasattr(slice_info, 'method') else slice_info.get('method', 'N/A')
                print(f"{slice_name}: {method}")

        # Print execution summary if present
        try:
            from dsperse.src.utils.utils import Utils
            rr = Utils.load_run_results(getattr(runner, 'last_run_dir', None)) if getattr(runner, 'last_run_dir', None) else {}
            ec = (rr or {}).get('execution_chain', {})
            if ec:
                print("\nExecution summary:")
                print(f"  JSTprove witness slices: {int(ec.get('jstprove_witness_slices', 0))}")
                print(f"  EZKL witness slices: {int(ec.get('ezkl_witness_slices', 0))}")
                print(f"  Overall security: {ec.get('overall_security', 'N/A')}")
        except Exception:
            pass


    except Exception as e:
        error_msg = f"Error during inference: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
