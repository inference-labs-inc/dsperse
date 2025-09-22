"""
CLI module for compiling models using EZKL.
"""

import traceback
import os
import json
import logging

from colorama import Fore, Style

from dsperse.src.compiler import Compiler
from dsperse.src.cli.base import check_model_dir, prompt_for_value, logger, normalize_path


def _check_layers(slices_path, layers_str):
    """
    Check if the layers provided exist in the metadata.json file within the slices directory.
    
    Args:
        slices_path (str): Path to the slices directory
        layers_str (str): String specifying which layers to compile (e.g., "3, 20-22")
        
    Returns:
        str: Validated layers string with only existing layers
    """
    if not layers_str:
        return None
        
    # Parse the layers string into a list of indices
    layer_indices = []
    parts = [p.strip() for p in layers_str.split(',')]
    
    for part in parts:
        if '-' in part:
            # Handle range (e.g., "20-22")
            try:
                start, end = map(int, part.split('-'))
                layer_indices.extend(range(start, end + 1))
            except ValueError:
                logger.warning(f"Invalid layer range: {part}. Skipping.")
                print(f"{Fore.YELLOW}Warning: Invalid layer range: {part}. Skipping.{Style.RESET_ALL}")
        else:
            # Handle single number
            try:
                layer_indices.append(int(part))
            except ValueError:
                logger.warning(f"Invalid layer index: {part}. Skipping.")
                print(f"{Fore.YELLOW}Warning: Invalid layer index: {part}. Skipping.{Style.RESET_ALL}")
    
    # Remove duplicates and sort
    layer_indices = sorted(set(layer_indices))
    
    # Find metadata.json file
    metadata_path = os.path.join(slices_path, "metadata.json")
    if not os.path.exists(metadata_path):
        # Check for metadata.json in slices subdirectory
        metadata_path = os.path.join(slices_path, "slices", "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"metadata.json not found in {slices_path} or {os.path.join(slices_path, 'slices')}")
            print(f"{Fore.YELLOW}Warning: metadata.json not found. Cannot validate layers.{Style.RESET_ALL}")
            return layers_str
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load metadata.json: {e}")
        print(f"{Fore.YELLOW}Warning: Failed to load metadata.json. Cannot validate layers.{Style.RESET_ALL}")
        return layers_str
    
    # Get available segments
    segments = metadata.get('segments', [])
    available_indices = [segment.get('index') for segment in segments]
    
    # Check if each layer exists
    valid_indices = []
    for idx in layer_indices:
        if idx in available_indices:
            valid_indices.append(idx)
        else:
            logger.warning(f"Layer {idx} not found in metadata.json, skipping compilation of it")
            print(f"{Fore.YELLOW}Warning: Layer {idx} not found, skipping compilation of it{Style.RESET_ALL}")
    
    # If no valid indices, return None
    if not valid_indices:
        logger.warning("No valid layers found")
        print(f"{Fore.YELLOW}Warning: No valid layers found. Will compile all layers.{Style.RESET_ALL}")
        return None
    
    # Convert valid indices back to a string
    # For simplicity, we'll just use comma-separated values
    return ','.join(map(str, valid_indices))


def setup_parser(subparsers):
    """
    Set up the argument parser for the compile command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    compile_parser = subparsers.add_parser('compile', aliases=['c'], help='Compile slices using EZKL')
    # Ensure canonical command even when alias is used
    compile_parser.set_defaults(command='compile')

    # Arguments with aliases/shorthands
    compile_parser.add_argument('--slices-path', '--slices-dir', '--slices-directory', '--slices', '--sd', '-s', dest='slices_path',
                                help='Path to the slices directory')
    compile_parser.add_argument('--input-file', '--input', '--if', '-i', dest='input_file',
                                help='Path to input file for calibration (optional)')
    compile_parser.add_argument('--layers', '-l', help='Specify which layers to compile (e.g., "3, 20-22"). If not provided, all layers will be compiled.')
    
    return compile_parser


def compile_model(args):
    """
    Compile a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Compiling slices with EZKL...{Style.RESET_ALL}")
    logger.info("Starting slices compilation")

    # Prompt for slices path if not provided
    if not hasattr(args, 'slices_path') or not args.slices_path:
        args.slices_path = prompt_for_value('slices-path', 'Enter the path to the slices directory')
    else:
        args.slices_path = normalize_path(args.slices_path)

    if not check_model_dir(args.slices_path):
        return

    # Normalize input file if provided via flag
    if hasattr(args, 'input_file') and args.input_file:
        args.input_file = normalize_path(args.input_file)

    # Ensure the provided path looks like a slices directory; otherwise guide user to slice first.
    try:
        if not os.path.isdir(args.slices_path):
            msg = (
                "Please provide a slices directory, not a model file. "
                "If you have a model, slice it first before compiling.\n"
                f"Try: dsperse slice --model-dir {args.slices_path}"
            )
            print(f"{Fore.YELLOW}Warning: {msg}{Style.RESET_ALL}")
            logger.error("Compile requires a slices directory (with metadata.json).")
            return
        has_metadata = (
            os.path.exists(os.path.join(args.slices_path, "metadata.json")) or
            os.path.exists(os.path.join(args.slices_path, "slices", "metadata.json"))
        )
        if not has_metadata:
            msg = (
                "No slices metadata found at the provided path. "
                "Please slice the model first before compiling slices.\n"
                f"Try: dsperse slice --model-dir {args.slices_path}"
            )
            print(f"{Fore.YELLOW}Warning: {msg}{Style.RESET_ALL}")
            logger.error("Compile requires slices metadata. Prompted user to run slice first.")
            return
        # Initialize the Compiler
        compiler = Compiler.create(args.slices_path)
        logger.info(f"Compiler initialized successfully")
    except RuntimeError as e:
        error_msg = f"Failed to initialize Compiler: {e}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        return

    # Check if the layers exist in the metadata
    if hasattr(args, 'layers') and args.layers:
        validated_layers = _check_layers(args.slices_path, args.layers)
    else:
        validated_layers = None
    
    # Run the compilation
    ezkl_logger = logging.getLogger('src.backends.ezkl')
    prev_ezkl_level = ezkl_logger.level
    try:
        # Suppress verbose EZKL INFO logs during compilation
        ezkl_logger.setLevel(logging.WARNING)

        output_path = compiler.compile(
            model_path=args.slices_path,
            input_file=args.input_file,
            layers=validated_layers
        )
        success_msg = f"Slices compiled successfully! Output saved to {os.path.dirname(output_path)}"
        print(f"{Fore.GREEN}✓ {success_msg}{Style.RESET_ALL}")
        logger.info(success_msg)
    except Exception as e:
        error_msg = f"Error compiling slices: {e}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()
    finally:
        # Restore previous EZKL logger level
        ezkl_logger.setLevel(prev_ezkl_level)
