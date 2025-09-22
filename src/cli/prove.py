"""
CLI module for generating proofs for models.
"""

import os
import time
import traceback
import glob
from colorama import Fore, Style

from src.prover import Prover
from src.cli.base import save_result, prompt_for_value, normalize_path

def setup_parser(subparsers):
    """
    Set up the argument parser for the prove command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    prove_parser = subparsers.add_parser('prove', aliases=['p'], help='Generate a proof for a run')
    # Ensure canonical command even when alias is used
    prove_parser.set_defaults(command='prove')

    prove_parser.add_argument('--run-dir', '--rd', dest='run_dir', help='Specific run directory to prove')
    prove_parser.add_argument('--output-file', '-o', dest='output_file', help='Path to save output results')

    return prove_parser

def get_all_runs(run_root_dir):
    """
    Get all run directories in the provided runs root directory.
    
    Args:
        run_root_dir (str): Path to the runs root directory (contains metadata.json and run_* subdirs)
        
    Returns:
        list: List of run directories, sorted by name (latest last)
    """
    if not os.path.exists(run_root_dir):
        return []
    
    # Get all run directories sorted by name (which includes timestamp)
    run_dirs = sorted(glob.glob(os.path.join(run_root_dir, "run_*")))
    
    return run_dirs

def get_latest_run(run_root_dir):
    """
    Get the latest run directory in the provided runs root directory.
    
    Args:
        run_root_dir (str): Path to the runs root directory
        
    Returns:
        str: Path to the latest run directory, or None if no runs found
    """
    run_dirs = get_all_runs(run_root_dir)
    
    if not run_dirs:
        return None
    
    # Return the latest run directory
    return run_dirs[-1]

def run_proof(args):
    """
    Generate a proof based on a provided runs root directory or a specific run directory.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Generating proof...{Style.RESET_ALL}")

    run_root_dir = None
    run_dir = None

    # Helper predicates
    def is_run_id_dir(p):
        return os.path.exists(os.path.join(p, "run_result.json"))

    def is_run_root_dir(p):
        return os.path.exists(os.path.join(p, "metadata.json"))

    # Determine input
    default_model_path = None  # Initialize at function scope
    if hasattr(args, 'run_dir') and args.run_dir:
        candidate = normalize_path(args.run_dir)
    else:
        # Try to suggest latest run as default
        default_run = None

        # First, check if there's a run/ directory in current directory
        current_run_dir = os.path.join(os.getcwd(), "run")
        if os.path.exists(current_run_dir):
            latest_run = get_latest_run(current_run_dir)
            if latest_run:
                default_run = os.path.basename(latest_run)
                default_model_path = os.getcwd()

        # If no local run directory, find latest run across all model directories
        if not default_run:
            models_dir = os.path.join(os.getcwd(), "src", "models")
            if os.path.exists(models_dir):
                all_runs = []

                # Find all run_* directories in all model directories
                for model_name in os.listdir(models_dir):
                    model_path = os.path.join(models_dir, model_name)
                    if os.path.isdir(model_path):
                        model_run_dir = os.path.join(model_path, "run")
                        if os.path.exists(model_run_dir):
                            for item in os.listdir(model_run_dir):
                                if item.startswith('run_') and os.path.isdir(os.path.join(model_run_dir, item)):
                                    run_path = os.path.join(model_run_dir, item)
                                    all_runs.append(run_path)

                # Sort by timestamp descending (run_YYYYMMDD_HHMMSS format)
                if all_runs:
                    all_runs.sort(key=lambda x: os.path.basename(x), reverse=True)
                    latest_run_path = all_runs[0]
                    default_run = os.path.basename(latest_run_path)
                    default_model_path = os.path.dirname(os.path.dirname(latest_run_path))

        # Prompt with default if found
        if default_run:
            candidate = prompt_for_value('run-or-run-id-dir', 'Enter run directory (runs root or a run_* directory)', default=default_run)
        else:
            candidate = prompt_for_value('run-or-run-id-dir', 'Enter run directory (runs root or a run_* directory)')

    # Handle run names (starts with "run_") - prepend run/ directory BEFORE normalization
    if candidate and candidate.startswith('run_') and not candidate.startswith('/') and not candidate.startswith('./') and not candidate.startswith('../'):
        # Always try current directory's run/ first (for when running from model directory)
        current_run_dir = os.path.join(os.getcwd(), "run")
        if os.path.exists(current_run_dir):
            candidate = os.path.join(current_run_dir, candidate)
        elif 'default_model_path' in locals() and default_model_path and default_model_path != os.getcwd():
            # Use stored default model path if different from current directory
            model_run_dir = os.path.join(default_model_path, "run")
            candidate = os.path.join(model_run_dir, candidate)
        else:
            # Look for the run in model directories
            models_dir = os.path.join(os.getcwd(), "src", "models")
            if os.path.exists(models_dir):
                for model_name in os.listdir(models_dir):
                    model_path = os.path.join(models_dir, model_name)
                    if os.path.isdir(model_path):
                        model_run_dir = os.path.join(model_path, "run")
                        if os.path.exists(model_run_dir) and os.path.exists(os.path.join(model_run_dir, candidate)):
                            candidate = os.path.join(model_run_dir, candidate)
                            break
    # Handle already-normalized run names (absolute paths ending with run_*)
    elif candidate and candidate.startswith('/') and os.path.basename(candidate).startswith('run_'):
        # Check if this is a run name that was normalized to the wrong directory
        basename = os.path.basename(candidate)
        dirname = os.path.dirname(candidate)

        # If the directory doesn't exist but we have model directories, look there
        if not os.path.exists(candidate):
            models_dir = os.path.join(os.getcwd(), "src", "models")
            if os.path.exists(models_dir):
                for model_name in os.listdir(models_dir):
                    model_path = os.path.join(models_dir, model_name)
                    if os.path.isdir(model_path):
                        model_run_dir = os.path.join(model_path, "run")
                        potential_path = os.path.join(model_run_dir, basename)
                        if os.path.exists(potential_path):
                            candidate = potential_path
                            break

    # Ensure candidate is normalized in case prompt returned a path-like
    candidate = normalize_path(candidate)

    if not os.path.exists(candidate):
        print(f"{Fore.RED}Error: Path {candidate} does not exist{Style.RESET_ALL}")
        return

    if is_run_id_dir(candidate):
        # Specific run directory selected
        run_dir = candidate
        run_root_dir = os.path.dirname(candidate)
    elif is_run_root_dir(candidate):
        # Runs root provided; let user choose run
        run_root_dir = candidate
        all_runs = get_all_runs(run_root_dir)
        if not all_runs:
            print(f"{Fore.RED}Error: No runs found in {run_root_dir}{Style.RESET_ALL}")
            return
        run_names = [os.path.basename(p) for p in all_runs]
        default_run = run_names[-1]
        run_list = ", ".join(run_names)
        print(f"We found {len(all_runs)} runs, {run_list}, enter which run you would like to prove (default {default_run}):")
        user_input = input().strip()
        if not user_input:
            run_dir = all_runs[-1]
        else:
            try:
                idx = int(user_input) - 1
                if 0 <= idx < len(all_runs):
                    run_dir = all_runs[idx]
                else:
                    print(f"{Fore.RED}Error: Invalid run index{Style.RESET_ALL}")
                    return
            except ValueError:
                candidate_run = normalize_path(os.path.join(run_root_dir, user_input))
                if os.path.exists(candidate_run) and is_run_id_dir(candidate_run):
                    run_dir = candidate_run
                else:
                    print(f"{Fore.RED}Error: Run directory {candidate_run} does not exist or is invalid{Style.RESET_ALL}")
                    return
    else:
        # Not a valid runs root or run directory
        print(f"{Fore.RED}Error: Provided path is neither a runs root (metadata.json) nor a run directory (run_result.json){Style.RESET_ALL}")
        return

    # Validate resolved paths
    run_dir = normalize_path(run_dir)
    run_root_dir = normalize_path(run_root_dir)
    run_result_path = os.path.join(run_dir, "run_result.json")
    if not os.path.exists(run_result_path):
        print(f"{Fore.RED}Error: run_result.json not found in {run_dir}{Style.RESET_ALL}")
        return

    metadata_path = os.path.join(run_root_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"{Fore.RED}Error: metadata.json not found in {run_root_dir}{Style.RESET_ALL}")
        return

    # Print proving message
    print("proving...")

    try:
        prover = Prover()
        start_time = time.time()
        result = prover.prove_run(run_result_path, metadata_path)
        elapsed_time = time.time() - start_time

        print(f"{Fore.GREEN}✓ Proof generation completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        print("\nDone!")

        # Prompt for output file if not provided
        if not hasattr(args, 'output_file') or not args.output_file:
            save_output = prompt_for_value('save-output', 'Save proof results to separate file?', default='n', required=False).lower()
            if save_output.startswith('y'):
                default_output_file = os.path.join(run_root_dir, "proof_results.json")
                args.output_file = prompt_for_value('output-file', 'Enter the output file path', default=default_output_file, required=False)

        # Save the result if output file is specified
        if args.output_file:
            try:
                args.output_file = normalize_path(args.output_file)
                save_result(result, args.output_file)
                print(f"{Fore.GREEN}Results saved to {args.output_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error saving output file: {e}{Style.RESET_ALL}")

        # Print the proof generation summary
        if "execution_chain" in result:
            execution_chain = result["execution_chain"]
            print(f"\n{Fore.YELLOW}Proof Generation Summary:{Style.RESET_ALL}")
            print(f"Proved segments: {execution_chain.get('ezkl_proved_slices', 0)} of {execution_chain.get('ezkl_witness_slices', 0)}")
            if execution_chain.get('ezkl_witness_slices', 0) > 0:
                proof_percentage = (execution_chain.get('ezkl_proved_slices', 0) / execution_chain.get('ezkl_witness_slices', 0)) * 100
                print(f"Proof generation percentage: {proof_percentage:.1f}%")
        else:
            print(f"\n{Fore.YELLOW}No proof generation results found{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error proving run: {e}{Style.RESET_ALL}")
        traceback.print_exc()
