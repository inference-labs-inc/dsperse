"""
CLI module for slicing models.
"""

import os
import traceback
from pathlib import Path

from colorama import Fore, Style

from dsperse.src.cli.base import check_model_dir, prompt_for_value, logger, normalize_path
from dsperse.src.slice.slicer import Slicer


def setup_parser(subparsers):
    """
    Set up the argument parser for the slice command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    slice_parser = subparsers.add_parser('slice', aliases=['s'], help='Slice a model into slices')
    # Ensure canonical command name even when alias is used
    slice_parser.set_defaults(command='slice')

    # Arguments with aliases/shorthands
    slice_parser.add_argument('--model-dir', '--model-path', '--mp', '-m', dest='model_dir',
                              help='Path to the model file or directory containing the model')
    slice_parser.add_argument('--output-dir', '-o',
                              help='Directory to save the sliced model (default: model_dir/slices)')
    slice_parser.add_argument('--save-file', '--save', '-S', nargs='?', const='default',
                              help='(Optional) Save path of the model analysis (default: model_dir/analysis/model_metadata.json)')
    # Output format for slicing
    slice_parser.add_argument('--output-type', '--ot', '-ot', dest='output_type',
                              choices=['dirs', 'dslice', 'dsperse'], default='dirs',
                              help='Output format of the slicing result: dirs (default), dslice, or dsperse')

    # Sub-commands under slice
    sub = slice_parser.add_subparsers(dest='slice_subcommand', help='Slice sub-commands')
    convert_parser = sub.add_parser('convert', aliases=['c'], help='Convert between .dsperse/.dslice and directory layouts')
    convert_parser.set_defaults(slice_subcommand='convert')
    convert_parser.add_argument('--input', '-i', dest='input_path', help='Input path (.dsperse/.dslice or directory)')
    convert_parser.add_argument('--to', '--output-type', '--type', '-t', choices=['dirs', 'dslice', 'dsperse'], dest='to_type',
                                help='Desired output type')
    convert_parser.add_argument('--output', '-o', dest='output_path', help='Output path (directory or archive)')
    convert_parser.add_argument('--expand-slices', action='store_true',
                                help='When converting .dsperse -> dirs, also extract embedded .dslice files')
    convert_parser.add_argument('--cleanup', dest='cleanup', action='store_true', default=True,
                                help='Remove source artifact after successful conversion (default: True)')
    convert_parser.add_argument('--no-cleanup', dest='cleanup', action='store_false', help='Keep source artifact')

    return slice_parser



def slice_convert(args):
    """Convert between dsperse/dslice and directories (sub-command under slice)."""
    # Lazy import to keep Converter usage confined to explicit conversion command
    from dsperse.src.slice.utils.converter import Converter
    # Prompt if not provided
    if not getattr(args, 'input_path', None):
        args.input_path = prompt_for_value('input', 'Enter input path (.dsperse/.dslice or directory)')
    else:
        args.input_path = normalize_path(args.input_path)

    input_path = args.input_path
    output_path = normalize_path(args.output_path) if getattr(args, 'output_path', None) else None

    p = Path(input_path)
    if not p.exists():
        print(f"{Fore.RED}Input path does not exist: {p}{Style.RESET_ALL}")
        logger.error(f"Input path does not exist: {p}")
        return

    # Prompt for output type if missing
    if not getattr(args, 'to_type', None):
        valid = {'dirs', 'dslice', 'dsperse'}
        from pathlib import Path as _P
        while True:
            chosen_raw = prompt_for_value('to', 'Enter desired output type (dirs, dslice, dsperse)')
            # Sanitize in case prompt_for_value normalized it like a path
            chosen = _P(str(chosen_raw)).name.strip().lower()
            if chosen in valid:
                args.to_type = chosen
                break
            print(f"{Fore.YELLOW}Invalid choice: '{chosen}'. Please enter one of: dirs, dslice, dsperse{Style.RESET_ALL}")

    try:
        # Delegate all conversions to the central Converter, which handles
        # the dsperse -> dirs default of fully expanding embedded .dslice files.
        result = Converter.convert(
            input_path,
            output_type=getattr(args, 'to_type', None) or getattr(args, 'output_type', None),
            output_path=output_path,
            cleanup=bool(getattr(args, 'cleanup', True)),
        )
        print(f"{Fore.GREEN}✓ Converted to: {result}{Style.RESET_ALL}")
        logger.info(f"Converted to: {result}")
    except Exception as e:
        print(f"{Fore.RED}Error converting: {e}{Style.RESET_ALL}")
        logger.error(f"Error converting: {e}")


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
    else:
        args.model_dir = normalize_path(args.model_dir)

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
    else:
        args.output_dir = normalize_path(args.output_dir)

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
        save_path = normalize_path(save_path)
    else:
        # Use the provided value or None (if no flag was provided)
        save_path = normalize_path(args.save_file) if args.save_file else None

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
        # Determine desired output type (default to 'dirs')
        output_type = getattr(args, 'output_type', 'dirs') or 'dirs'
        logger.info(f"Slicing ONNX model to output path: {output_dir} with output_type={output_type}")
        # Delegate to Slicer which will convert if needed
        slicer.slice_model(
            output_path=output_dir,
            output_type=output_type,
        )
        success_msg = "ONNX model sliced successfully!"
        print(f"{Fore.GREEN}✓ {success_msg}{Style.RESET_ALL}")
        logger.info(success_msg)
        # If a save path for model analysis/metadata was provided, inform the user where it was saved
        if 'save_path' in locals() and save_path:
            print(f"{Fore.GREEN}Model analysis saved to {normalize_path(save_path)}{Style.RESET_ALL}")
            logger.info(f"Model analysis saved to {normalize_path(save_path)}")

    except Exception as e:
        error_msg = f"Error slicing model: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
