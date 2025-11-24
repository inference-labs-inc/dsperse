"""
CLI module for generating proofs for models.
"""

import os
import time
import traceback
import glob
from pathlib import Path
from colorama import Fore, Style

from dsperse.src.prover import Prover
from dsperse.src.cli.base import save_result, prompt_for_value, normalize_path, logger
from dsperse.src.utils.utils import Utils

def setup_parser(subparsers):
    """
    Set up the argument parser for the prove command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    prove_parser = subparsers.add_parser('prove', aliases=['p'], help='Generate proofs for a run using EZKL')
    # Ensure canonical command even when alias is used
    prove_parser.set_defaults(command='prove')

    # New preferred positional arguments
    prove_parser.add_argument('run_path', nargs='?', help='Path to run_<timestamp> directory (must contain metadata.json)')
    prove_parser.add_argument('data_path', nargs='?', help='Path to slices root, a single slice_* dir, a .dslice, or a .dsperse')

    # Optional: where to write proofs (default is under run_path/slice_#/proof.json)
    prove_parser.add_argument('--proof-output', '-po', dest='proof_output', help='Directory to write proofs (overrides default under run_path)')

    # Results JSON save path (optional convenience)
    prove_parser.add_argument('--output-file', '-o', dest='output_file', help='Path to save the run_results.json copy')

    # Deprecated/legacy flags (kept for backwards compatibility with older workflows)
    prove_parser.add_argument('--run-dir', '--rd', dest='run_dir', help='[Deprecated] Run directory; prefer positional run_path')
    prove_parser.add_argument('--from', '--dsperse-file', '--dsperse', dest='dsperse_file', help='[Deprecated] Data archive path; prefer positional data_path')

    return prove_parser

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

def run_proof(args):
    """
    Generate a proof based on a provided runs root directory or a specific run directory.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Generating proof...{Style.RESET_ALL}")

    # Fast path: new streamlined interface using positional args
    run_path_arg = getattr(args, 'run_path', None)
    data_path_arg = getattr(args, 'data_path', None)
    if run_path_arg:
        run_path = normalize_path(run_path_arg)
        data_path = normalize_path(data_path_arg) if data_path_arg else None
        try:
            meta = Utils.load_run_metadata(Path(run_path))
        except Exception as e:
            print(f"{Fore.RED}Error loading run metadata: {e}{Style.RESET_ALL}")
            return
        if not data_path:
            # Prefer the original source_path captured in metadata; fallback to model_path/slices
            source_path = meta.get('source_path')
            if source_path:
                data_path = normalize_path(source_path)
            else:
                model_path = meta.get('model_path')
                data_path = normalize_path(str(Path(model_path) / 'slices')) if model_path else None
        if not data_path:
            print(f"{Fore.RED}Error: Could not determine data_path. Provide it explicitly (slices root, a slice_* dir, .dslice, or .dsperse).{Style.RESET_ALL}")
            return

        try:
            prover = Prover()
            start_time = time.time()
            result = prover.prove(run_path, data_path, getattr(args, 'proof_output', None))
            elapsed_time = time.time() - start_time
            print(f"{Fore.GREEN}✓ Proof generation completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error proving run: {e}{Style.RESET_ALL}")
            traceback.print_exc()
            return

        # Optional: save a copy of results
        if getattr(args, 'output_file', None):
            try:
                outp = normalize_path(args.output_file)
                save_result(result, outp)
                print(f"{Fore.GREEN}Results saved to {outp}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error saving output file: {e}{Style.RESET_ALL}")

        # Print summary
        if isinstance(result, dict) and 'execution_chain' in result:
            ec = result.get('execution_chain', {})
            proved = ec.get('ezkl_proved_slices', 0)
            witnessed = ec.get('ezkl_witness_slices', 0)
            print(f"\n{Fore.YELLOW}Proof Generation Summary:{Style.RESET_ALL}")
            print(f"Proved slices: {proved} of {witnessed}")
        return

    run_root_dir = None
    run_dir = None

    # Helper predicates
    def is_run_id_dir(p: str) -> bool:
        # Only treat as a per-run directory if it's named like run_* and has run_results.json
        base = os.path.basename(os.path.abspath(p))
        return base.startswith("run_") and os.path.exists(os.path.join(p, "run_results.json"))

    def is_run_root_dir(p):
        # A runs root contains subdirectories named run_*
        try:
            return any(d.startswith("run_") and os.path.isdir(os.path.join(p, d)) for d in os.listdir(p))
        except Exception:
            return False

    # Determine input
    default_model_path = None  # Initialize at function scope
    specified_run_dir = None  # Track if user specified a specific run
    
    # Determine run candidate (legacy flag support)
    if hasattr(args, 'run_dir') and args.run_dir:
        candidate = normalize_path(args.run_dir)
        # Treat as a specific run dir only if it looks like run_* and has run_results.json
        base = os.path.basename(os.path.abspath(candidate))
        if base.startswith("run_") and os.path.exists(os.path.join(candidate, "run_results.json")):
            specified_run_dir = candidate
    else:
        # No flags provided - automatically use latest run from current directory
        current_run_dir = os.path.join(os.getcwd(), "run")
        if os.path.exists(current_run_dir):
            latest_run = get_latest_run(current_run_dir)
            if latest_run and os.path.exists(os.path.join(latest_run, "run_results.json")):
                # Use latest run automatically
                candidate = normalize_path(latest_run)
                logger.info(f"Using latest run automatically: {candidate}")
            else:
                # No valid runs, prompt user
                candidate = prompt_for_value('run-or-run-id-dir', 'Enter run directory (runs root or a run_* directory)')
        else:
            # No run directory found, prompt user
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
        # Specific run directory selected (either user-specified or auto-selected latest)
        run_dir = candidate
        run_root_dir = os.path.dirname(candidate)
    elif is_run_root_dir(candidate):
        # Runs root provided
        run_root_dir = candidate
        all_runs = get_all_runs(run_root_dir)
        if not all_runs:
            print(f"{Fore.RED}Error: No runs found in {run_root_dir}{Style.RESET_ALL}")
            return
        
        # Prompt user to choose run
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
        print(f"{Fore.RED}Error: Provided path is neither a runs root (contains run_*/ subdirs) nor a run directory (named run_* with run_results.json){Style.RESET_ALL}")
        return

    # Validate resolved paths
    run_dir = normalize_path(run_dir)
    run_root_dir = normalize_path(run_root_dir)
    run_results_path = os.path.join(run_dir, "run_results.json")
    if not os.path.exists(run_results_path):
        print(f"{Fore.YELLOW}Warning: run_results.json not found in {run_dir}; a new one will be created/updated.{Style.RESET_ALL}")

    # Determine data_path: prefer explicit --from archive, else metadata.source_path, else model_path/slices
    try:
        meta = Utils.load_run_metadata(Path(run_dir))
    except Exception as e:
        print(f"{Fore.RED}Error: metadata.json not found in run directory {run_dir}: {e}{Style.RESET_ALL}")
        return
    data_path = None
    if hasattr(args, 'dsperse_file') and args.dsperse_file:
        data_path = normalize_path(args.dsperse_file)
    else:
        data_path = meta.get('source_path') or (str(Path(meta.get('model_path', '')) / 'slices') if meta.get('model_path') else None)
        if data_path:
            data_path = normalize_path(data_path)
    if not data_path:
        print(f"{Fore.RED}Error: Could not determine data_path for proving. Provide it explicitly or ensure run metadata contains source_path/model_path.{Style.RESET_ALL}")
        return

    # Print proving message
    print("proving...")

    try:
        prover = Prover()
        start_time = time.time()
        result = prover.prove(run_dir, data_path, getattr(args, 'proof_output', None))
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
            print(f"Proved slices: {execution_chain.get('ezkl_proved_slices', 0)} of {execution_chain.get('ezkl_witness_slices', 0)}")
            if execution_chain.get('ezkl_witness_slices', 0) > 0:
                proof_percentage = (execution_chain.get('ezkl_proved_slices', 0) / execution_chain.get('ezkl_witness_slices', 0)) * 100
                print(f"Proof generation percentage: {proof_percentage:.1f}%")
        else:
            print(f"\n{Fore.YELLOW}No proof generation results found{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error proving run: {e}{Style.RESET_ALL}")
        traceback.print_exc()
    finally:
        pass
