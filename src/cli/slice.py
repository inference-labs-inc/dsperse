"""
CLI module for slicing models.
"""

import os
import traceback
import logging
from colorama import Fore, Style

from src.model_slicer import ModelSlicer
from src.onnx_slicer import OnnxSlicer
from src.cli.base import check_model_dir, prompt_for_value, logger
from src.utils.onnx_analyzer import OnnxAnalyzer


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
    slice_parser.add_argument('--model-type', choices=['onnx', 'pth'], 
                             help='Type of model to slice (auto-detected if not specified)')
    slice_parser.add_argument('--input-file', help='Path to input file for analysis (default: model_dir/input.json)')
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

    # Determine if it's a PyTorch or ONNX model based on model_type or auto-detect
    is_onnx = False
    if args.model_type:
        is_onnx = args.model_type.lower() == 'onnx'
        logger.debug(f"Model type specified: {args.model_type}")
    else:
        # Auto-detect model type
        if model_file and model_file.lower().endswith('.onnx'):
            is_onnx = True
            print(f"{Fore.YELLOW}Auto-detected ONNX model from filename.{Style.RESET_ALL}")
            logger.info("Auto-detected ONNX model from filename.")
        elif os.path.exists(os.path.join(model_dir, "model.onnx")):
            is_onnx = True
            print(f"{Fore.YELLOW}Auto-detected ONNX model in directory.{Style.RESET_ALL}")
            logger.info("Auto-detected ONNX model in directory.")
        else:
            error_msg = f"No ONNX model found at the specified path '{args.model_dir}'."
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            logger.error(error_msg)
            return

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
        if is_onnx:
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

            logger.info(f"Creating ONNX slicer for model: {onnx_path}")
            slicer = OnnxSlicer(onnx_path, save_path)
            logger.info(f"Slicing ONNX model to output path: {output_dir}")
            slicer.slice_model(output_path=output_dir)
            success_msg = "ONNX model sliced successfully!"
            print(f"{Fore.GREEN}✓ {success_msg}{Style.RESET_ALL}")
            logger.info(success_msg)
        else:
            # Slice PyTorch model
            if model_file and model_file.lower().endswith('.pth'):
                pth_path = model_file
            else:
                pth_path = os.path.join(model_dir, "model.pth")

            if not os.path.exists(pth_path):
                error_msg = f"PyTorch model file not found at the specified path '{pth_path}'."
                print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
                logger.error(error_msg)
                return

            # Prompt for input file if not provided
            if not hasattr(args, 'input_file') or not args.input_file:
                default_input_file = os.path.join(model_dir, "input.json")
                args.input_file = prompt_for_value('input-file', 'Enter the input file path', default=default_input_file, required=False)

            # Check if input file exists
            if args.input_file and not os.path.exists(args.input_file):
                warning_msg = f"Input file '{args.input_file}' does not exist."
                print(f"{Fore.YELLOW}Warning: {warning_msg}{Style.RESET_ALL}")
                logger.warning(warning_msg)
                create_new = prompt_for_value('create-new', 'Create a new input file?', default='n', required=False).lower()
                if create_new.startswith('y'):
                    try:
                        with open(args.input_file, 'w') as f:
                            f.write('{"input_data": []}')
                        success_msg = f"Created empty input file: {args.input_file}"
                        print(f"{Fore.GREEN}{success_msg}{Style.RESET_ALL}")
                        logger.info(success_msg)
                    except Exception as e:
                        error_msg = f"Error creating input file: {e}"
                        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                        logger.error(error_msg)
                        return
                else:
                    logger.debug("User chose not to create a new input file")
                    args.input_file = None

            logger.info(f"Creating PyTorch model slicer for directory: {model_dir}")
            slicer = ModelSlicer(model_directory=model_dir)
            logger.info(f"Slicing PyTorch model to output path: {output_dir}")
            slicer.slice_model(
                output_dir=output_dir,
                input_file=args.input_file
            )
            success_msg = "PyTorch model sliced successfully!"
            print(f"{Fore.GREEN}✓ {success_msg}{Style.RESET_ALL}")
            logger.info(success_msg)
    except Exception as e:
        error_msg = f"Error slicing model: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
