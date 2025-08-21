"""
CLI module for verifying proofs for models.
"""

import os
import time
import traceback
import glob
from colorama import Fore, Style

from src.verifier import Verifier
from src.cli.base import save_result, prompt_for_value, normalize_path

def setup_parser(subparsers):
    """
    Set up the argument parser for the verify command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    verify_parser = subparsers.add_parser('verify', help='Verify a proof for a run')
    verify_parser.add_argument('--run-dir', help='Specific run directory to verify')
    verify_parser.add_argument('--output-file', help='Path to save output results')

    return verify_parser

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

def verify_proof(args):
    """
    Verify a proof based on a provided runs root directory or a specific run directory.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Verifying proof...{Style.RESET_ALL}")

    run_root_dir = None
    run_dir = None

    def is_run_id_dir(p):
        return os.path.exists(os.path.join(p, "run_result.json"))

    def is_run_root_dir(p):
        return os.path.exists(os.path.join(p, "metadata.json"))

    # Determine input
    if hasattr(args, 'run_dir') and args.run_dir:
        candidate = normalize_path(args.run_dir)
    else:
        candidate = prompt_for_value('run-or-run-id-dir', 'Enter run directory (runs root or a run_* directory)')

    # Normalize candidate from prompt
    candidate = normalize_path(candidate)

    if not os.path.exists(candidate):
        print(f"{Fore.RED}Error: Path {candidate} does not exist{Style.RESET_ALL}")
        return

    if is_run_id_dir(candidate):
        run_dir = candidate
        run_root_dir = os.path.dirname(candidate)
    elif is_run_root_dir(candidate):
        run_root_dir = candidate
        all_runs = get_all_runs(run_root_dir)
        if not all_runs:
            print(f"{Fore.RED}Error: No runs found in {run_root_dir}{Style.RESET_ALL}")
            return
        run_names = [os.path.basename(p) for p in all_runs]
        default_run = run_names[-1]
        run_list = ", ".join(run_names)
        print(f"We found {len(all_runs)} runs, {run_list}, enter which run you would like to verify (default {default_run})")
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
        print(f"{Fore.RED}Error: Provided path is neither a runs root (metadata.json) nor a run directory (run_result.json){Style.RESET_ALL}")
        return

    # Validate
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

    print("verifying...")

    try:
        verifier = Verifier()
        start_time = time.time()
        result = verifier.verify_run(run_result_path, metadata_path)
        elapsed_time = time.time() - start_time

        print(f"{Fore.GREEN}✓ Verification completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        print("\nDone!")

        if not hasattr(args, 'output_file') or not args.output_file:
            save_output = prompt_for_value('save-output', 'Save verification results to separate file?', default='n', required=False).lower()
            if save_output.startswith('y'):
                default_output_file = os.path.join(run_root_dir, "verification_results.json")
                args.output_file = prompt_for_value('output-file', 'Enter the output file path', default=default_output_file, required=False)

        if args.output_file:
            try:
                args.output_file = normalize_path(args.output_file)
                save_result(result, args.output_file)
                print(f"{Fore.GREEN}Results saved to {args.output_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error saving output file: {e}{Style.RESET_ALL}")

        if "execution_chain" in result:
            execution_chain = result["execution_chain"]
            print(f"\n{Fore.YELLOW}Verification Summary:{Style.RESET_ALL}")
            print(f"Verified segments: {execution_chain.get('ezkl_verified_slices', 0)} of {execution_chain.get('ezkl_proved_slices', 0)}")
            denom = execution_chain.get('ezkl_proved_slices', 0) or 1
            print(f"Verification percentage: {(execution_chain.get('ezkl_verified_slices', 0) / denom * 100):.1f}%")
        else:
            print(f"\n{Fore.YELLOW}No verification results found{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error verifying run: {e}{Style.RESET_ALL}")
        traceback.print_exc()