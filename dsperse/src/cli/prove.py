"""
CLI module for generating proofs for models.
"""

import time
import traceback

from colorama import Fore, Style

from dsperse.src.prove.prover import Prover
from dsperse.src.utils.pipeline_utils import parse_tiles_range
from dsperse.src.cli.base import add_stage_arguments, resolve_stage_paths

def setup_parser(subparsers):
    prove_parser = subparsers.add_parser('prove', aliases=['p'], help='Generate proofs for a run')
    prove_parser.set_defaults(command='prove')
    add_stage_arguments(prove_parser, "proof generation")
    return prove_parser


def run_proof(args):
    print(f"{Fore.CYAN}Generating proof...{Style.RESET_ALL}")

    resolved = resolve_stage_paths(args)
    if resolved is None:
        return
    run_dir, slices_path = resolved

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