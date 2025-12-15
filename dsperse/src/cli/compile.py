"""
CLI module for compiling models using EZKL.
"""

import traceback
import os
import json
import logging

from colorama import Fore, Style

from dsperse.src.compile.compiler import Compiler
from dsperse.src.cli.base import check_model_dir, prompt_for_value, logger, normalize_path
from pathlib import Path


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
    
    # Get available slices
    slices = metadata.get('slices', [])
    available_indices = [slice_item.get('index') for slice_item in slices]
    
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
    compile_parser.add_argument('--path', '-p', '--slices-path', '--slices-dir', '--slices-directory', '--slices', '--sd', '-s', dest='path',
                                help='Path to the model or slices directory (or a .dsperse/.dslice file)')
    compile_parser.add_argument('--input-file', '--input', '--if', '-i', dest='input_file',
                                help='Path to input file for calibration (optional)')
    compile_parser.add_argument('--layers', '-l', help='Layer selection or per-layer backend mapping. Examples: "3,20-22" (select layers), or "0,2:jstprove;3-4:ezkl" (per-layer backends). If not provided, all layers will be compiled with default fallback (jstprove→ezkl→onnx).')
    compile_parser.add_argument('--backend', '-b', default=None,
                                help='Backend specification for all selected layers: "jstprove" | "ezkl" | "onnx". Alternatively, provide per-layer mapping via --layers, e.g., "0,2:jstprove;3-4:ezkl". Default: try both jstprove and ezkl, fallback to onnx.')
    
    return compile_parser


def compile_model(args):
    """
    Compile a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    backend = getattr(args, 'backend', None)
    layers = getattr(args, 'layers', None)
    
    if not layers:
        print(f"{Fore.CYAN}No layers specified. Will compile all layers with default fallback (jstprove -> ezkl -> onnx)...{Style.RESET_ALL}")
        logger.info("No layers specified - compiling all layers with default fallback")
    elif backend:
        if ':' in backend:
            print(f"{Fore.CYAN}Compiling specified layers with mixed backends...{Style.RESET_ALL}")
        else:
            backend_name = 'JSTprove' if backend == 'jstprove' else 'EZKL'
            print(f"{Fore.CYAN}Compiling specified layers with {backend_name}...{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}Compiling specified layers (trying jstprove & ezkl, fallback to onnx)...{Style.RESET_ALL}")
    logger.info(f"Starting slices compilation")

    # Resolve path (slices dir or .dsperse/.dslice file)
    target_path = getattr(args, 'path', None) or getattr(args, 'slices_path', None)
    if not target_path:
        target_path = prompt_for_value('path', 'Enter the path to the slices directory or .dsperse file')
    target_path = normalize_path(target_path)

    # Do not auto-unpack archives here; let the Compiler handle .dslice/.dsperse directly
    target_path_obj = Path(target_path)

    # Heuristic: prefer the provided directory if it already contains metadata.json
    tp_obj = Path(target_path)
    if tp_obj.is_dir():
        if (tp_obj / 'metadata.json').exists():
            # Accept this directory as the compile target; do not override to parent/slices
            logger.info(f"compile: using provided directory with metadata.json: {tp_obj}")
        elif tp_obj.name == 'output':
            # Only try to infer when the provided output dir itself lacks metadata.json
            parent = tp_obj.parent
            inferred = None
            if (parent / 'slices' / 'metadata.json').exists():
                inferred = parent / 'slices'
            elif (parent / 'metadata.json').exists():
                inferred = parent
            if inferred:
                print(f"{Fore.YELLOW}Hint: 'output' directory detected without its own metadata.json. Using '{inferred}' as the compile target.{Style.RESET_ALL}")
                logger.info(f"compile: adjusted target from {tp_obj} to {inferred}")
                target_path = str(inferred)
                tp_obj = inferred
            else:
                print(f"{Fore.YELLOW}Warning: 'output' directory does not contain metadata.json and no adjacent slices/metadata were found. Provide a model or slices directory instead.{Style.RESET_ALL}")
                logger.warning("compile: 'output' dir provided without adjacent metadata/slices")

    # Only verify directory structure when a directory is provided; files are handled by Compiler
    if target_path_obj.is_dir():
        if not check_model_dir(target_path):
            return

    # Normalize input file if provided via flag
    if hasattr(args, 'input_file') and args.input_file:
        args.input_file = normalize_path(args.input_file)

    # If --layers is a per-layer backend mapping (contains ':'), prefer it as compiler backend spec
    if layers and ':' in str(layers):
        backend = layers

    # Initialize the Compiler (it supports dirs or model.onnx)
    try:
        compiler = Compiler(backend=backend)
        logger.info(f"Compiler initialized successfully")
    except RuntimeError as e:
        error_msg = f"Failed to initialize Compiler: {e}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        return

    # Check if the layers exist in the metadata when a directory is provided
    if hasattr(args, 'layers') and args.layers:
        if os.path.isdir(target_path):
            validated_layers = _check_layers(target_path, args.layers)
        else:
            validated_layers = args.layers  # pass through for non-dirs; Compiler will parse
    else:
        validated_layers = None
    
    # Run the compilation
    ezkl_logger = logging.getLogger('src.backends.ezkl')
    prev_ezkl_level = ezkl_logger.level
    try:
        # Suppress verbose EZKL INFO logs during compilation
        ezkl_logger.setLevel(logging.WARNING)

        output_path = compiler.compile(
            model_path=target_path,
            input_file=args.input_file,
            layers=validated_layers
        )
        # Tailored success message based on returned path
        if os.path.isfile(output_path):
            success_msg = f"Slices compiled successfully! Output packaged at {output_path}"
        elif os.path.isdir(output_path):
            success_msg = f"Slices compiled successfully! Output saved under {output_path}"
        else:
            success_msg = "Slices compiled successfully!"
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
        # Note: We don't cleanup unpacked directories - they're unpacked to the same location as the archive
        # This ensures paths remain consistent and files are accessible
