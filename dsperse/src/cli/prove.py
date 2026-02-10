"""
CLI module for generating proofs for models.
"""

import os
import time
import traceback
from colorama import Fore, Style

from dsperse.src.prove.prover import Prover
from dsperse.src.utils.pipeline_utils import parse_tiles_range
from dsperse.src.cli.base import normalize_path, logger, prompt_for_value, validate_run_dir

def setup_parser(subparsers):
    """
    Set up the argument parser for the prove command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    prove_parser = subparsers.add_parser('prove', aliases=['p'], help='Generate proofs for a run')
    # Ensure canonical command even when alias is used
    prove_parser.set_defaults(command='prove')

    prove_parser.add_argument('--run-dir', '--rd', dest='run_dir', help='The run directory generated when you run the model')
    prove_parser.add_argument('--slices', '--sd', '-s', dest='slices_path', help='The path to the dslice file, the slice directory, or the dsperse file')
    prove_parser.add_argument('--backend', '-b', choices=['jstprove', 'ezkl'],
                             help='Backend to use. In single-slice mode this is required. In run-root mode, only prove slices whose witness backend matches this choice.')
    prove_parser.add_argument('--parallel', type=int, default=1, dest='parallel',
                             help='Number of parallel processes for proof generation (default: 1)')
    prove_parser.add_argument('--tiles', '-t', dest='tiles',
                             help='Range of tiles to prove (e.g., "0-2" or "0,1,5"). Only applicable in single-slice mode.')

    return prove_parser


def run_proof(args):
    """
    Generate a proof based on a provided runs root directory or a specific run directory.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Generating proof...{Style.RESET_ALL}")

    # New flags-only behavior: require --run-dir and --slices
    run_dir = getattr(args, 'run_dir', None)
    slices_path = getattr(args, 'slices_path', None)

    if not run_dir:
        run_dir = prompt_for_value('run-dir', 'Enter the run directory (run/run_<timestamp>)')
    if not slices_path:
        slices_path = prompt_for_value('slices', 'Enter the slices path (dslice file, slices directory, or dsperse file)')

    run_dir = normalize_path(run_dir)
    slices_path = normalize_path(slices_path)

    if not os.path.exists(run_dir):
        print(f"{Fore.RED}Error: Run directory not found: {run_dir}{Style.RESET_ALL}")
        return
    if not validate_run_dir(run_dir):
        print(
            f"{Fore.RED}Error: run-dir does not contain recognized run artifacts "
            f"(metadata.json, run_results.json, input.json + output.json, "
            f"split/, tile_*, or slice_* directories): {run_dir}{Style.RESET_ALL}")
        return

    print("proving...")

    try:
        parallel = getattr(args, 'parallel', 1)
        prover = Prover(parallel=parallel)
        start_time = time.time()

        # Parse the tile range from CLI args
        tiles_range = parse_tiles_range(getattr(args, 'tiles', None))

        result = prover.prove(
            run_dir,
            slices_path,
            None,
            backend=getattr(args, 'backend', None),
            tiles_range=tiles_range
        )
        elapsed_time = time.time() - start_time
        print(f"{Fore.GREEN}✓ Proof generation completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        print(f"Proof saved to run_results.json within the run directory {run_dir}{Style.RESET_ALL}")
        print("\nDone!")

        # Print the proof generation summary
        if isinstance(result, dict) and "execution_chain" in result:
            execution_chain = result["execution_chain"]
            print(f"\n{Fore.YELLOW}Proof Generation Summary:{Style.RESET_ALL}")
            j_proved = int(execution_chain.get('jstprove_proved_slices', 0) or 0)
            e_proved = int(execution_chain.get('ezkl_proved_slices', 0) or 0)
            j_witness = int(execution_chain.get('jstprove_witness_slices', 0) or 0)
            e_witness = int(execution_chain.get('ezkl_witness_slices', 0) or 0)
            total_proved = j_proved + e_proved
            total_witness = j_witness + e_witness
            pct = (total_proved / total_witness * 100.0) if total_witness > 0 else 0.0
            print(f"Proved slices: {total_proved} of {total_witness}")
            print(f"Proof generation percentage: {pct:.1f}%")
        else:
            print(f"\n{Fore.YELLOW}No proof generation results found{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error proving run: {e}{Style.RESET_ALL}")
        traceback.print_exc()