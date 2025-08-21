"""
CLI module for slicing models.
"""

import os
import traceback

from colorama import Fore, Style

from src.cli.base import check_model_dir, prompt_for_value, logger
from src.slicer import Slicer


def setup_parser(subparsers):
    """
    Set up the argument parser for the slice command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    slice_parser = subparsers.add_parser('slice', help='Slice a model into segments')
    slice_parser.add_argument('--model-dir', help='Path to the model file or directory containing the model')
    slice_parser.add_argument('--output-dir', help='Directory to save the sliced model (default: model_dir/slices)')
    slice_parser.add_argument('--save-file', nargs='?', const='default', help='(Optional) Save path of the model analysis (default: model_dir/analysis/model_metadata.json)')

    return slice_parser

def slice_model(args):
    """
    Slice a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Slicing model...{Style.RESET_ALL}")
    logger.info("Starting model slicing")

    # Prompt for model path if not provided
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = prompt_for_value('model-dir', 'Enter the path to the model file or directory')

    if not check_model_dir(args.model_dir):
        return

    # Check if the provided path is a file or directory
    model_dir = args.model_dir
    model_file = None

    # If the path is a file, extract the directory and filename
    if os.path.isfile(model_dir):
        model_file = model_dir
        model_dir = os.path.dirname(model_dir)
        if not model_dir:  # If the directory is empty (e.g., just "model.onnx")
            model_dir = "."
        print(f"{Fore.YELLOW}Using model file: {model_file}{Style.RESET_ALL}")
        logger.info(f"Using model file: {model_file}")

    # Prompt for output directory if not provided
    if not hasattr(args, 'output_dir') or not args.output_dir:
        default_output_dir = os.path.join(model_dir, "slices")
        args.output_dir = prompt_for_value('output-dir', 'Enter the output directory', default=default_output_dir, required=False)

    # Create output directory if specified
    output_dir = args.output_dir
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            success_msg = f"Output directory created: {output_dir}"
            print(f"{Fore.GREEN}{success_msg}{Style.RESET_ALL}")
            logger.info(success_msg)
        except Exception as e:
            error_msg = f"Error creating output directory: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logger.error(error_msg)
            return

    if args.save_file == 'default':
        # Flag included, no value provided
        save_path = os.path.join(model_dir, "analysis", "model_metadata.json")
    else:
        # Use the provided value or None (if no flag was provided)
        save_path = args.save_file

    try:
            # Slice ONNX model
        if model_file and model_file.lower().endswith('.onnx'):
            onnx_path = model_file
        else:
            onnx_path = os.path.join(model_dir, "model.onnx")

        if not os.path.exists(onnx_path):
            error_msg = f"ONNX model file not found at the specified path '{onnx_path}'."
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            logger.error(error_msg)
            return

        logger.info(f"Creating slicer for model: {onnx_path}")
        slicer = Slicer.create(onnx_path, save_path)
        logger.info(f"Slicing ONNX model to output path: {output_dir}")
        slicer.slice_model(output_path=output_dir)
        success_msg = "ONNX model sliced successfully!"
        print(f"{Fore.GREEN}✓ {success_msg}{Style.RESET_ALL}")
        logger.info(success_msg)

    except Exception as e:
        error_msg = f"Error slicing model: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
