"""
CLI module for verifying proofs for models.
"""

import os
import time
import traceback
from colorama import Fore, Style

from dsperse.src.verify.verifier import Verifier
from dsperse.src.utils.pipeline_utils import parse_tiles_range
from dsperse.src.cli.base import normalize_path, logger, prompt_for_value, validate_run_dir


def setup_parser(subparsers):
    """
    Set up the argument parser for the verify command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    verify_parser = subparsers.add_parser('verify', aliases=['v'], help='Verify proofs for a run')
    # Ensure canonical command even when alias is used
    verify_parser.set_defaults(command='verify')

    # Flags-only interface
    verify_parser.add_argument('--run-dir', '--rd', dest='run_dir',
                               help='The run directory generated when you run the model')
    verify_parser.add_argument('--slices', '--sd', '-s', dest='slices_path',
                               help='The path to the dslice file, the slice directory, or the dsperse file')
    verify_parser.add_argument('--backend', '-b', choices=['jstprove', 'ezkl'],
                               help='Backend to use. In single-slice mode this is required. In run-root mode, only verify slices whose witness backend matches this choice.')
    verify_parser.add_argument('--parallel', type=int, default=1, dest='parallel',
                               help='Number of parallel processes for verification (default: 1)')
    verify_parser.add_argument('--tiles', '-t', dest='tiles',
                               help='Range of tiles to verify (e.g., "0-2" or "0,1,5"). Only applicable in single-slice mode.')

    return verify_parser
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
        slices_path = prompt_for_value('slices',
                                       'Enter the slices path (dslice file, slices directory, or dsperse file)')

    run_dir = normalize_path(run_dir)
    slices_path = normalize_path(slices_path)

    if not os.path.exists(run_dir):
        print(f"{Fore.RED}Error: Run directory not found: {run_dir}{Style.RESET_ALL}")
        return
    if not validate_run_dir(run_dir):
        print(
            f"{Fore.RED}Error: run-dir must contain either run-root files (metadata.json/run_results.json) or per-slice files (input.json + output.json): {run_dir}{Style.RESET_ALL}")
        return

    print("verifying...")

    try:
        parallel = getattr(args, 'parallel', 1)
        verifier = Verifier(parallel=parallel)
        start_time = time.time()

        # Parse the tile range from CLI args
        tiles_range = parse_tiles_range(getattr(args, 'tiles', None))

        result = verifier.verify(
            run_dir,
            slices_path,
            backend=getattr(args, 'backend', None),
            tiles_range=tiles_range
        )
        elapsed_time = time.time() - start_time

        print(f"{Fore.GREEN}✓ Verification completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        print(f"Verification results saved to {run_dir}")
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