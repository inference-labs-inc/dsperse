"""
CLI module for verifying proofs for models.
"""

import time
import traceback

from colorama import Fore, Style

from dsperse.src.verify.verifier import Verifier
from dsperse.src.utils.pipeline_utils import parse_tiles_range
from dsperse.src.cli.base import add_stage_arguments, resolve_stage_paths


def setup_parser(subparsers):
    verify_parser = subparsers.add_parser('verify', aliases=['v'], help='Verify proofs for a run')
    verify_parser.set_defaults(command='verify')
    add_stage_arguments(verify_parser, "verification")
    return verify_parser


def verify_proof(args):
    print(f"{Fore.CYAN}Verifying proof...{Style.RESET_ALL}")

    resolved = resolve_stage_paths(args)
    if resolved is None:
        return
    run_dir, slices_path = resolved

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