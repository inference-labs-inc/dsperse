"""
CLI module for converting between DSperse packaging formats.

Supported conversions (auto-detected by input path):
- .dsperse  -> directory (with optional expansion of embedded .dslice files)
- directory -> .dsperse (expects dsperse-style metadata.json and slices/)
- .dslice   -> directory
- directory (slice dir with metadata.json + payload/) -> .dslice
"""

import json
from pathlib import Path
from colorama import Fore, Style

from dsperse.src.cli.base import logger, normalize_path, prompt_for_value
from dsperse.src.slice.onnx_slicer import OnnxSlicer


def setup_parser(subparsers):
    """
    Set up the argument parser for the convert command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    parser = subparsers.add_parser("convert", aliases=["cv"], help="Convert between .dsperse/.dslice and directory layouts")
    parser.set_defaults(command="convert")

    parser.add_argument("--input", "-i", dest="input_path", help="Input path: a .dsperse/.dslice file or a directory")
    parser.add_argument("--output", "-o", dest="output_path", help="Output path: a directory or target .dsperse/.dslice file (optional)")
    parser.add_argument("--expand-slices", action="store_true", help="When converting .dsperse -> directory, also extract embedded .dslice files into subfolders")

    return parser


def _is_slice_dir(path: Path) -> bool:
    """Heuristic: a slice directory contains metadata.json and a payload/ dir with model.onnx inside."""
    if not path.is_dir():
        return False
    meta = path / "metadata.json"
    payload = path / "payload"
    model = payload / "model.onnx"
    return meta.exists() and payload.exists() and model.exists()


def _is_dsperse_dir(path: Path) -> bool:
    """Heuristic: a dsperse directory contains metadata.json with schema dsperse/1.0 and a slices/ dir."""
    if not path.is_dir():
        return False
    meta = path / "metadata.json"
    slices_dir = path / "slices"
    if not meta.exists() or not slices_dir.exists():
        return False
    try:
        with open(meta, "r") as f:
            data = json.load(f)
        return (data.get("schema") == "dsperse/1.0")
    except Exception:
        return False


def convert(args):
    """
    Execute conversion based on input and output.
    """
    # Prompt if not provided
    if not getattr(args, "input_path", None):
        args.input_path = prompt_for_value("input", "Enter input path (.dsperse/.dslice or directory)")
    else:
        args.input_path = normalize_path(args.input_path)

    input_path = Path(args.input_path)
    output_path = Path(normalize_path(args.output_path)) if getattr(args, "output_path", None) else None

    if not input_path.exists():
        print(f"{Fore.RED}Input path does not exist: {input_path}{Style.RESET_ALL}")
        logger.error(f"Input path does not exist: {input_path}")
        return

    # Case 1: .dsperse -> directory
    if input_path.is_file() and input_path.suffix == ".dsperse":
        if output_path is None:
            # Default to a folder next to the input with the same stem
            output_path = input_path.parent / input_path.stem
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting dsperse to {output_path} (expand_slices={args.expand_slices})")
        res = OnnxSlicer.unzip_dsperse(str(input_path), str(output_path), expand_slices=bool(getattr(args, "expand_slices", False)))
        print(f"{Fore.GREEN}✓ Extracted dsperse to: {res}{Style.RESET_ALL}")
        return

    # Case 2: .dslice -> directory
    if input_path.is_file() and input_path.suffix == ".dslice":
        if output_path is None:
            output_path = input_path.parent / input_path.stem
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting dslice to {output_path}")
        res = OnnxSlicer.unzip_dslice(str(input_path), str(output_path))
        print(f"{Fore.GREEN}✓ Extracted dslice to: {res}{Style.RESET_ALL}")
        return

    # Case 3: directory -> .dsperse or .dslice
    if input_path.is_dir():
        # If directory looks like a dsperse directory, zip to .dsperse
        if _is_dsperse_dir(input_path):
            if output_path is None:
                output_path = input_path / "model.dsperse"
            # Ensure extension
            if output_path.suffix != ".dsperse":
                output_path = output_path.with_suffix(".dsperse")
            logger.info(f"Creating dsperse archive: {output_path}")
            res = OnnxSlicer.zip_slices_dir_to_dsperse(str(input_path), str(output_path))
            print(f"{Fore.GREEN}✓ Created dsperse: {res}{Style.RESET_ALL}")
            return
        # If it's a single slice directory, zip to .dslice
        if _is_slice_dir(input_path):
            if output_path is None:
                output_path = input_path.with_suffix(".dslice")
            if output_path.suffix != ".dslice":
                output_path = output_path.with_suffix(".dslice")
            logger.info(f"Creating dslice archive: {output_path}")
            res = OnnxSlicer.zip_slice_dir_to_dslice(str(input_path), str(output_path))
            print(f"{Fore.GREEN}✓ Created dslice: {res}{Style.RESET_ALL}")
            return
        # Fallback hint
        print(f"{Fore.RED}Input directory is neither a dsperse directory nor a slice directory.\nExpected either: metadata.json + slices/ (for dsperse) or metadata.json + payload/ (for a slice).{Style.RESET_ALL}")
        logger.error("Unsupported input directory structure for conversion")
        return

    print(f"{Fore.RED}Unsupported input. Please provide a .dsperse/.dslice file or a compatible directory.{Style.RESET_ALL}")
    logger.error("Unsupported input for convert command")
