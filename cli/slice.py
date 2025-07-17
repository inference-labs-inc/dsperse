"""
CLI module for slicing models.
"""

import os
import traceback
from colorama import Fore, Style

from src.model_slicer import ModelSlicer
from src.onnx_slicer import OnnxSlicer
from cli.base import check_model_dir, prompt_for_value

def setup_parser(subparsers):
    """
    Set up the argument parser for the slice command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    slice_parser = subparsers.add_parser('slice', help='Slice a model into segments')
    slice_parser.add_argument('--model-dir', help='Directory containing the model')
    slice_parser.add_argument('--output-dir', help='Directory to save the sliced model (default: model_dir/slices)')
    slice_parser.add_argument('--model-type', choices=['onnx', 'pth'], 
                             help='Type of model to slice (auto-detected if not specified)')
    slice_parser.add_argument('--input-file', help='Path to input file for analysis (default: model_dir/input.json)')

    return slice_parser

def slice_model(args):
    """
    Slice a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Slicing model...{Style.RESET_ALL}")

    # Prompt for model directory if not provided
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = prompt_for_value('model-dir', 'Enter the model directory')

    if not check_model_dir(args.model_dir):
        return

    # Determine if it's a PyTorch or ONNX model based on model_type or auto-detect
    is_onnx = False
    if args.model_type:
        is_onnx = args.model_type.lower() == 'onnx'
    else:
        # Auto-detect model type
        if os.path.exists(os.path.join(args.model_dir, "model.onnx")):
            is_onnx = True
            print(f"{Fore.YELLOW}Auto-detected ONNX model.{Style.RESET_ALL}")
        elif os.path.exists(os.path.join(args.model_dir, "model.pth")):
            print(f"{Fore.YELLOW}Auto-detected PyTorch model.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Error: No model.pth or model.onnx found in '{args.model_dir}'.{Style.RESET_ALL}")
            return

    # Prompt for output directory if not provided
    if not hasattr(args, 'output_dir') or not args.output_dir:
        default_output_dir = os.path.join(args.model_dir, "slices")
        args.output_dir = prompt_for_value('output-dir', 'Enter the output directory', default=default_output_dir, required=False)

    # Create output directory if specified
    output_dir = args.output_dir
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"{Fore.GREEN}Output directory created: {output_dir}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error creating output directory: {e}{Style.RESET_ALL}")
            return

    try:
        if is_onnx:
            # Slice ONNX model
            onnx_path = os.path.join(args.model_dir, "model.onnx")
            if not os.path.exists(onnx_path):
                print(f"{Fore.RED}Error: ONNX model file not found at '{onnx_path}'.{Style.RESET_ALL}")
                return
            slicer = OnnxSlicer(onnx_path)
            slicer.slice_model()
            print(f"{Fore.GREEN}✓ ONNX model sliced successfully!{Style.RESET_ALL}")
        else:
            # Slice PyTorch model
            pth_path = os.path.join(args.model_dir, "model.pth")
            if not os.path.exists(pth_path):
                print(f"{Fore.RED}Error: PyTorch model file not found at '{pth_path}'.{Style.RESET_ALL}")
                return
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

            slicer = ModelSlicer(model_directory=args.model_dir)
            slicer.slice_model(
                output_dir=output_dir,
                input_file=args.input_file
            )
            print(f"{Fore.GREEN}✓ PyTorch model sliced successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error slicing model: {e}{Style.RESET_ALL}")
        traceback.print_exc()
