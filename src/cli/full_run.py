"""
CLI module for running the full pipeline: slice -> compile -> run -> prove -> verify
This is a meta-command that orchestrates existing commands without changing their logic.
"""
import os
from argparse import Namespace
from colorama import Fore, Style

from src.cli.base import prompt_for_value, normalize_path, logger
from src.cli.slice import slice_model
from src.cli.compile import compile_model
from src.cli.run import run_inference
from src.cli.prove import run_proof, get_latest_run
from src.cli.verify import verify_proof


def setup_parser(subparsers):
    """
    Set up the argument parser for the full-run command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    full_run_parser = subparsers.add_parser('full-run', aliases=['fr'], help='Run the full pipeline (slice, compile, run, prove, verify)')
    # Ensure canonical command even when alias is used
    full_run_parser.set_defaults(command='full-run')

    # Arguments with aliases/shorthands
    full_run_parser.add_argument('--model-dir', '--model-path', '--mp', '-m', dest='model_dir',
                                 help='Path to the model file (.onnx) or directory containing the model')
    full_run_parser.add_argument('--input-file', '--input', '--if', '-i', dest='input_file',
                                 help='Path to input file for inference and compilation calibration (e.g., input.json)')
    full_run_parser.add_argument('--slices-dir', '--slices-directory', '--slices-directroy', '--sd', '-s', dest='slices_dir',
                                 help='Optional: Pre-existing slices directory to reuse (skips slicing step)')
    full_run_parser.add_argument('--layers', '-l', help='Optional: Layers to compile (e.g., "3, 20-22") passed through to compile')
    # Optional: allow non-interactive mode later if desired; kept interactive by default
    return full_run_parser


def _determine_model_dir(model_path: str) -> str:
    """Return the canonical model directory given a file or a directory path."""
    model_path = normalize_path(model_path)
    if os.path.isfile(model_path):
        # If a file is provided (e.g., model.onnx), the model directory is its parent
        parent = os.path.dirname(model_path)
        return parent if parent else '.'
    return model_path


def full_run(args):
    """
    Run the full pipeline by invoking the existing CLI handlers.
    Preserves interactivity: if an argument is missing, we'll prompt for it here
    (and the underlying commands will prompt for any of their own missing args).
    """
    print(f"{Fore.CYAN}Starting full pipeline: slice → compile → run → prove → verify{Style.RESET_ALL}")

    # 1) Resolve inputs interactively
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = prompt_for_value('model-dir', 'Enter the path to the model file (.onnx) or directory')
    else:
        args.model_dir = normalize_path(args.model_dir)

    # Determine canonical model directory for downstream steps
    canonical_model_dir = _determine_model_dir(args.model_dir)

    if not hasattr(args, 'input_file') or not args.input_file:
        # Suggest default input.json in the model directory
        default_input = os.path.join(canonical_model_dir, 'input.json')
        args.input_file = prompt_for_value('input-file', 'Enter the input file path', default=default_input, required=True)
    else:
        args.input_file = normalize_path(args.input_file)

    # If user provided an existing slices directory, skip slicing step
    slices_dir = None
    if hasattr(args, 'slices_dir') and args.slices_dir:
        slices_dir = normalize_path(args.slices_dir)

    # 2) Slice (unless slices-dir provided)
    if not slices_dir:
        default_slices_dir = os.path.join(canonical_model_dir, 'slices')
        # Call existing slice command; keep its logic and interactivity
        slice_args = Namespace(model_dir=args.model_dir, output_dir=default_slices_dir, save_file=None)
        print(f"{Fore.CYAN}Step 1/5: Slicing model...{Style.RESET_ALL}")
        slice_model(slice_args)
        slices_dir = default_slices_dir
    else:
        print(f"{Fore.YELLOW}Skipping slicing step, using existing slices at: {slices_dir}{Style.RESET_ALL}")

    # 3) Compile (circuitize) with calibration input
    compile_args = Namespace(slices_path=slices_dir, input_file=args.input_file, layers=getattr(args, 'layers', None))
    print(f"{Fore.CYAN}Step 2/5: Compiling slices (EZKL circuitization)...{Style.RESET_ALL}")
    compile_model(compile_args)

    # 4) Run inference
    run_args = Namespace(slices_dir=slices_dir, run_metadata_path=None, input_file=args.input_file, output_file=None)
    print(f"{Fore.CYAN}Step 3/5: Running inference over slices...{Style.RESET_ALL}")
    run_inference(run_args)

    # Determine latest run directory for proving/verifying
    run_root_dir = os.path.join(canonical_model_dir, 'run')
    latest_run_dir = get_latest_run(run_root_dir)
    if not latest_run_dir:
        # If no run_* found, fall back to run_root_dir; the prove/verify commands will guide the user
        latest_run_dir = run_root_dir

    # 5) Generate proof
    prove_args = Namespace(run_dir=latest_run_dir, output_file=None)
    print(f"{Fore.CYAN}Step 4/5: Generating proof...{Style.RESET_ALL}")
    run_proof(prove_args)

    # 6) Verify proof
    verify_args = Namespace(run_dir=latest_run_dir, output_file=None)
    print(f"{Fore.CYAN}Step 5/5: Verifying proof...{Style.RESET_ALL}")
    verify_proof(verify_args)

    print(f"{Fore.GREEN}✓ Full pipeline completed!{Style.RESET_ALL}")
    logger.info("Full pipeline completed")
