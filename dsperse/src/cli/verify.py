"""
CLI module for verifying proofs for models.
"""

import os
import time
import traceback
import glob
from pathlib import Path
from colorama import Fore, Style

from dsperse.src.verifier import Verifier
from dsperse.src.cli.base import normalize_path, logger, prompt_for_value
from dsperse.src.utils.utils import Utils

def setup_parser(subparsers):
    """
    Set up the argument parser for the verify command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    verify_parser = subparsers.add_parser('verify', aliases=['v'], help='Verify proofs for a run using EZKL')
    # Ensure canonical command even when alias is used
    verify_parser.set_defaults(command='verify')

    # Flags-only interface
    verify_parser.add_argument('--run-dir', '--rd', dest='run_dir', help='The run directory generated when you run the model')
    verify_parser.add_argument('--slices', '--sd', '-s', dest='slices_path', help='The path to the dslice file, the slice directory, or the dsperse file')

    return verify_parser

def get_all_runs(run_root_dir):
    """
    Get all run directories in the provided runs root directory.
    
    Args:
        run_root_dir (str): Path to the runs root directory (contains metadata.json and run_* subdirs)
        
    Returns:
        list: List of run directories (absolute paths), sorted by name (latest last)
    """
    if not os.path.exists(run_root_dir):
        return []
    
    # Normalize the run root directory to ensure absolute paths
    run_root_dir = normalize_path(run_root_dir)
    
    # Get all run directories sorted by name (which includes timestamp)
    run_dirs = sorted(glob.glob(os.path.join(run_root_dir, "run_*")))
    
    # Ensure all paths are normalized/absolute
    run_dirs = [normalize_path(d) for d in run_dirs]
    
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
    Verify proofs for a run.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Verifying proof...{Style.RESET_ALL}")

    # Flags-only behavior with prompts when missing: require --run-dir and --slices
    run_dir = getattr(args, 'run_dir', None)
    slices_path = getattr(args, 'slices_path', None)

    # If flags missing, prompt interactively for them
    if not run_dir:
        run_dir = prompt_for_value('run-dir', 'Enter the run directory (run/run_<timestamp>)')
    if not slices_path:
        slices_path = prompt_for_value('slices', 'Enter the slices path (dslice file, slices directory, or dsperse file)')

    run_dir = normalize_path(run_dir)
    slices_path = normalize_path(slices_path)

    if not os.path.exists(run_dir):
        print(f"{Fore.RED}Error: Run directory not found: {run_dir}{Style.RESET_ALL}")
        return
    # Validate run_dir by presence of either run-root files or per-slice files
    rd = Path(run_dir)
    is_run_root = (rd / 'metadata.json').exists()
    is_slice_run = (rd / 'input.json').exists() and (rd / 'output.json').exists()
    if not (is_run_root or is_slice_run):
        print(f"{Fore.RED}Error: run-dir must contain either run-root files (metadata.json) or per-slice files (input.json + output.json): {run_dir}{Style.RESET_ALL}")
        return

    print("verifying...")

    try:
        verifier = Verifier()
        start_time = time.time()
        result = verifier.verify(run_dir, slices_path)
        elapsed_time = time.time() - start_time

        print(f"{Fore.GREEN}✓ Verification completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        print(f"Verification saved to run_results.json within the run directory {run_dir}")
        print("\nDone!")

        # Print the verification summary
        if isinstance(result, dict) and "execution_chain" in result:
            execution_chain = result["execution_chain"]
            print(f"\n{Fore.YELLOW}Verification Summary:{Style.RESET_ALL}")
            j_verified = int(execution_chain.get('jstprove_verified_slices', 0) or 0)
            e_verified = int(execution_chain.get('ezkl_verified_slices', 0) or 0)
            j_proved = int(execution_chain.get('jstprove_proved_slices', 0) or 0)
            e_proved = int(execution_chain.get('ezkl_proved_slices', 0) or 0)
            total_verified = j_verified + e_verified
            total_proved = j_proved + e_proved
            pct = (total_verified / total_proved * 100.0) if total_proved > 0 else 0.0
            print(f"Verified slices: {total_verified} of {total_proved}")
            print(f"Verification percentage: {pct:.1f}%")
        else:
            print(f"\n{Fore.YELLOW}No verification results found{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error verifying run: {e}{Style.RESET_ALL}")
        traceback.print_exc()
    finally:
        pass