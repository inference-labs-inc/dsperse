"""
CLI module for running the full pipeline: slice -> compile -> run -> prove -> verify
This is a meta-command that orchestrates existing commands without changing their logic.
"""
import logging
import os
from argparse import Namespace

from colorama import Fore, Style

from dsperse.src.cli.base import prompt_for_value, normalize_path, get_latest_run

logger = logging.getLogger(__name__)
from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model
from dsperse.src.cli.run import run_inference
from dsperse.src.cli.prove import run_proof
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
    full_run_parser.add_argument('--weights-as-inputs', '--wai', action='store_true', default=False, dest='weights_as_inputs',
                                 help='Treat model weights as witness inputs instead of compile-time constants (JSTprove only)')
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
    if not getattr(args, 'model_dir', None) and not getattr(args, 'input_file', None):
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
        if getattr(args, 'model_dir', None):
            args.model_dir = normalize_path(args.model_dir)
        else:
            args.model_dir = prompt_for_value('model-dir', 'Enter the path to the model file or directory', required=True)
            if not args.model_dir:
                print(f"{Fore.RED}Error: model path is required.{Style.RESET_ALL}")
                return
            args.model_dir = normalize_path(args.model_dir)
        canonical_model_dir = _determine_model_dir(args.model_dir)

    # Input file resolution
    if getattr(args, 'input_file', None):
        args.input_file = normalize_path(args.input_file)
    else:
        # Suggest default input.json in the (canonical) model directory unless using built-in (already set)
        default_input = os.path.join(canonical_model_dir, 'input.json') if not using_builtin else args.input_file
        args.input_file = prompt_for_value('input-file', 'Enter the input file', default=default_input, required=True)
        args.input_file = normalize_path(args.input_file) if args.input_file else args.input_file

    # If user provided an existing slices directory, skip slicing step
    slices_dir = None
    if getattr(args, 'slices_dir', None):
        slices_dir = normalize_path(args.slices_dir)
        if not os.path.isdir(slices_dir):
            print(f"{Fore.RED}Error: slices directory '{slices_dir}' does not exist.{Style.RESET_ALL}")
            logger.error(f"Slices directory does not exist: {slices_dir}")
            return

    # 2) Slice (unless slices-dir provided)
    if not slices_dir:
        # Default slices dir depends on whether we're using a built-in selection
        default_slices_dir = os.path.join(canonical_model_dir, 'slices')
        analysis_dir = os.path.join(canonical_model_dir, 'analysis')
        os.makedirs(default_slices_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        # Call existing slice command; keep its logic and interactivity.
        # For built-ins, we point the slicer to the built-in model file but output to ~/dsperse/{name}/slices
        model_metadata_path = os.path.join(analysis_dir, 'model_metadata.json')
        slice_args = Namespace(model_dir=args.model_dir, output_dir=default_slices_dir, save_file=model_metadata_path, output_type='dirs')
        print(f"{Fore.CYAN}Step 1/5: Slicing model...{Style.RESET_ALL}")
        slice_model(slice_args)
        slices_dir = default_slices_dir
    else:
        print(f"{Fore.YELLOW}Skipping slicing step, using existing slices at: {slices_dir}{Style.RESET_ALL}")

    # 3) Compile (circuitize) with calibration input
    # User requested: compilation should just be jstprove.
    compile_args = Namespace(path=slices_dir, input_file=args.input_file, layers=getattr(args, 'layers', None), backend='jstprove', weights_as_inputs=getattr(args, 'weights_as_inputs', False))
    print(f"{Fore.CYAN}Step 2/5: Compiling slices (JSTprove circuitization)...{Style.RESET_ALL}")
    compile_model(compile_args)

    # 4) Run inference
    run_root_dir = os.path.join(canonical_model_dir, 'run')
    os.makedirs(run_root_dir, exist_ok=True)
    inference_output_path = os.path.join(run_root_dir, 'inference_results.json')
    # run_inference expects 'path' for slices, and doesn't use output_file directly for run_results.json anymore (it saves to run_results.json in the run dir)
    run_args = Namespace(path=slices_dir, input_file=args.input_file, output_file=inference_output_path, force_backend=None)
    print(f"{Fore.CYAN}Step 3/5: Running inference over slices...{Style.RESET_ALL}")
    run_inference(run_args)

    # Determine latest run directory for proving/verifying
    latest_run_dir = get_latest_run(run_root_dir)
    if not latest_run_dir:
        # If no run_* found, fall back to run_root_dir; the prove/verify commands will guide the user
        latest_run_dir = run_root_dir

    # 5) Generate proof
    # run_proof expects run_dir and slices_path
    prove_args = Namespace(run_dir=latest_run_dir, slices_path=slices_dir, backend=None)
    print(f"{Fore.CYAN}Step 4/5: Generating proof...{Style.RESET_ALL}")
    run_proof(prove_args)

    # 6) Verify proof
    # verify_proof expects run_dir and slices_path
    verify_args = Namespace(run_dir=latest_run_dir, slices_path=slices_dir, backend=None)
    print(f"{Fore.CYAN}Step 5/5: Verifying proof...{Style.RESET_ALL}")
    verify_proof(verify_args)

    # Final notice about saved files when using built-ins
    if using_builtin:
        print(f"{Fore.GREEN}Saved all artifacts (slices, runs, metadata) under {canonical_model_dir}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}✓ Full pipeline completed!{Style.RESET_ALL}")
    logger.info("Full pipeline completed")
