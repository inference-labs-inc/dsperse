"""
CLI module for running the full pipeline: slice -> compile -> run -> prove -> verify
This is a meta-command that orchestrates existing commands without changing their logic.
"""
import os
from argparse import Namespace
from colorama import Fore, Style

from dsperse.src.cli.base import prompt_for_value, normalize_path, logger
from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model
from dsperse.src.cli.run import run_inference
from dsperse.src.cli.prove import run_proof, get_latest_run
from dsperse.src.cli.verify import verify_proof


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

    Enhancement: If invoked without flags, allow selecting a built-in model
    (doom, net, resnet). Built-in models are sourced from src/models/{name},
    but all outputs are written to ~/dsperse/{name}.
    """
    print(f"{Fore.CYAN}Starting full pipeline: slice → compile → run → prove → verify{Style.RESET_ALL}")

    # Built-in models mapping
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    builtin_roots = {
        'doom': os.path.join(project_root, 'models', 'doom'),
        'net': os.path.join(project_root, 'models', 'net'),
        # 'resnet': os.path.join(project_root, 'models', 'resnet'),
    }

    using_builtin = False
    builtin_name = None

    # 1) Resolve inputs interactively
    if (not hasattr(args, 'model_dir') or not args.model_dir) and (not hasattr(args, 'input_file') or not args.input_file):
        # Special prompt that accepts either a filesystem location or a built-in token
        choice = prompt_for_value(
            'selection',
            'Enter model location OR choose built-in: doom, net',
            required=True
        )
        # Do not normalize in prompt; handle here explicitly
        choice_str = (choice or '').strip().strip('\"\'').lower()
        if choice_str in builtin_roots:
            using_builtin = True
            builtin_name = choice_str
            source_root = builtin_roots[builtin_name]
            model_onnx = os.path.join(source_root, 'model.onnx')
            input_json = os.path.join(source_root, 'input.json')
            # Validate built-in assets exist
            if not os.path.exists(model_onnx) or not os.path.exists(input_json):
                print(f"{Fore.RED}Error: Built-in assets missing for '{builtin_name}'. Expected model.onnx and input.json in {source_root}{Style.RESET_ALL}")
                return
            # Set args to point to sources
            args.model_dir = model_onnx
            args.input_file = input_json
            # Output root under user's home
            output_root = os.path.expanduser(os.path.join('~', 'dsperse', builtin_name))
            os.makedirs(output_root, exist_ok=True)
            # For downstream steps, canonical model dir should be the output root
            canonical_model_dir = normalize_path(output_root)
            print(f"{Fore.CYAN}Using built-in model '{builtin_name}'. Outputs will be saved under {canonical_model_dir}{Style.RESET_ALL}")
        else:
            # Treat as a user-provided file or directory path
            args.model_dir = normalize_path(choice)
            canonical_model_dir = _determine_model_dir(args.model_dir)
    else:
        # Normalize provided values
        if hasattr(args, 'model_dir') and args.model_dir:
            args.model_dir = normalize_path(args.model_dir)
        # Determine canonical model directory for downstream steps
        canonical_model_dir = _determine_model_dir(args.model_dir)

    # Input file resolution
    if hasattr(args, 'input_file') and args.input_file:
        args.input_file = normalize_path(args.input_file)
    else:
        # Suggest default input.json in the (canonical) model directory unless using built-in (already set)
        default_input = os.path.join(canonical_model_dir, 'input.json') if not using_builtin else args.input_file
        args.input_file = prompt_for_value('input-file', 'Enter the input file', default=default_input, required=True)
        args.input_file = normalize_path(args.input_file) if args.input_file else args.input_file

    # If user provided an existing slices directory, skip slicing step
    slices_dir = None
    if hasattr(args, 'slices_dir') and args.slices_dir:
        slices_dir = normalize_path(args.slices_dir)

    # 2) Slice (unless slices-dir provided)
    if not slices_dir:
        # Default slices dir depends on whether we're using a built-in selection
        default_slices_dir = os.path.join(canonical_model_dir, 'slices')
        analysis_dir = os.path.join(canonical_model_dir, 'analysis')
        try:
            os.makedirs(default_slices_dir, exist_ok=True)
            os.makedirs(analysis_dir, exist_ok=True)
        except Exception:
            pass
        # Call existing slice command; keep its logic and interactivity.
        # For built-ins, we point the slicer to the built-in model file but output to ~/dsperse/{name}/slices
        model_metadata_path = os.path.join(analysis_dir, 'model_metadata.json')
        slice_args = Namespace(model_dir=args.model_dir, output_dir=default_slices_dir, save_file=model_metadata_path)
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
    run_root_dir = os.path.join(canonical_model_dir, 'run')
    try:
        os.makedirs(run_root_dir, exist_ok=True)
    except Exception:
        pass
    inference_output_path = os.path.join(run_root_dir, 'inference_results.json')
    run_args = Namespace(slices_dir=slices_dir, run_metadata_path=None, input_file=args.input_file, output_file=inference_output_path)
    print(f"{Fore.CYAN}Step 3/5: Running inference over slices...{Style.RESET_ALL}")
    run_inference(run_args)

    # Determine latest run directory for proving/verifying
    latest_run_dir = get_latest_run(run_root_dir)
    if not latest_run_dir:
        # If no run_* found, fall back to run_root_dir; the prove/verify commands will guide the user
        latest_run_dir = run_root_dir

    # 5) Generate proof
    proof_output_path = os.path.join(latest_run_dir, 'proof_results.json')
    prove_args = Namespace(run_dir=latest_run_dir, output_file=proof_output_path)
    print(f"{Fore.CYAN}Step 4/5: Generating proof...{Style.RESET_ALL}")
    run_proof(prove_args)

    # 6) Verify proof
    verification_output_path = os.path.join(run_root_dir, 'verification_results.json')
    verify_args = Namespace(run_dir=latest_run_dir, output_file=verification_output_path)
    print(f"{Fore.CYAN}Step 5/5: Verifying proof...{Style.RESET_ALL}")
    verify_proof(verify_args)

    # Final notice about saved files when using built-ins
    if using_builtin:
        print(f"{Fore.GREEN}Saved all artifacts (slices, runs, metadata) under {canonical_model_dir}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}✓ Full pipeline completed!{Style.RESET_ALL}")
    logger.info("Full pipeline completed")
