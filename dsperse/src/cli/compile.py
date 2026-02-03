"""
CLI module for compiling models using EZKL.
"""

import traceback
import os
import json
import logging

from colorama import Fore, Style

from dsperse.src.compile.utils.compiler_utils import CompilerUtils
from dsperse.src.compile.compiler import Compiler
from dsperse.src.cli.base import check_model_dir, prompt_for_value, logger, normalize_path
from pathlib import Path


def _check_layers(slices_path, layers_str):
    """Simple validation of layers string against metadata."""
    if not layers_str or ':' in layers_str or ';' in layers_str:
        return layers_str

    indices = CompilerUtils.parse_layers(layers_str)
    if not indices: return None

    # Load metadata to verify existence
    meta_path = Path(slices_path) / "metadata.json"
    if not meta_path.exists(): meta_path = Path(slices_path) / "slices" / "metadata.json"
    
    if not meta_path.exists():
        logger.warning(f"metadata.json not found. Cannot validate layers.")
        return layers_str

    try:
        with open(meta_path, 'r') as f: meta = json.load(f)
        available = {s.get('index') for s in meta.get('slices', [])}
        valid = [i for i in indices if i in available]
        
        for i in indices:
            if i not in available:
                print(f"{Fore.YELLOW}Warning: Layer {i} not found, skipping it{Style.RESET_ALL}")
        
        return ','.join(map(str, valid)) if valid else None
    except Exception:
        return layers_str


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
    compile_parser.add_argument('--parallel', type=int, default=1, dest='parallel',
                                help='Number of parallel processes for compilation (default: 1)')
    compile_parser.add_argument('--resume', '-r', action='store_true', default=False,
                                help='Resume compilation, skipping slices that already have compiled circuits')

    return compile_parser


def compile_model(args):
    """Compile a model based on the provided arguments."""
    backend = getattr(args, 'backend', None)
    layers = getattr(args, 'layers', None)

    # --- Setup and Path Resolution ---
    target_path = getattr(args, 'path', None) or getattr(args, 'slices_path', None)
    if not target_path:
        target_path = prompt_for_value('path', 'Enter the path to the slices directory or .dsperse file')
    target_path = normalize_path(target_path)

    # --- Layer Validation ---
    validated_layers = _check_layers(target_path, args.layers) if (getattr(args, 'layers', None) and os.path.isdir(target_path)) else getattr(args, 'layers', None)

    # Combine layers and backend into a single spec for Compiler.compile
    effective_layers = validated_layers
    if backend:
        if not effective_layers:
            effective_layers = backend
        elif ':' not in str(effective_layers) and effective_layers not in ['ezkl', 'jstprove', 'onnx']:
            effective_layers = f"{effective_layers}:{backend}"

    # --- Initialization ---
    print(f"Initializing compiler...")
    try:
        compiler = Compiler(
            parallel=getattr(args, 'parallel', 1), 
            resume=getattr(args, 'resume', False)
        )
    except RuntimeError as e:
        print(f"{Fore.RED}Error initializing Compiler: {e}{Style.RESET_ALL}")
        return

    # --- Execution ---
    print(f"Starting compilation...")
    ezkl_logger = logging.getLogger('dsperse.src.backends.ezkl')
    prev_level = ezkl_logger.level
    try:
        ezkl_logger.setLevel(logging.WARNING) # Suppress verbose logs
        output_path = compiler.compile(
            model_path=target_path,
            input_file=args.input_file,
            layers=effective_layers
        )
        print(f"{Fore.GREEN}✓ Slices compiled successfully! Output: {output_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error compiling slices: {e}{Style.RESET_ALL}")
        logger.debug("Stack trace:", exc_info=True)
    finally:
        ezkl_logger.setLevel(prev_level)
