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
from dsperse.src.cli.base import save_result, prompt_for_value, normalize_path, logger
from dsperse.src.slice.utils.converter import Converter
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

    # New preferred positional arguments
    verify_parser.add_argument('run_path', nargs='?', help='Path to run_<timestamp> directory (must contain metadata.json)')
    verify_parser.add_argument('data_path', nargs='?', help='Path to slices root, a single slice_* dir, a .dslice, or a .dsperse')

    # Optional: save a copy of results to a separate file
    verify_parser.add_argument('--output-file', '-o', dest='output_file', help='Path to save a copy of updated run_results.json')

    # Deprecated/legacy flags (kept for backwards compatibility)
    verify_parser.add_argument('--run-dir', '--rd', dest='run_dir', help='[Deprecated] Run directory; prefer positional run_path')
    verify_parser.add_argument('--from', '--dsperse-file', '--dsperse', dest='dsperse_file', help='[Deprecated] Data archive path; prefer positional data_path')

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
            verifier = Verifier()
            start_time = time.time()
            result = verifier.verify(run_path, data_path)
            elapsed_time = time.time() - start_time
            print(f"{Fore.GREEN}✓ Verification completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error verifying run: {e}{Style.RESET_ALL}")
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
            verified = ec.get('ezkl_verified_slices', 0)
            proved = ec.get('ezkl_proved_slices', 0)
            print(f"\n{Fore.YELLOW}Verification Summary:{Style.RESET_ALL}")
            print(f"Verified slices: {verified} of {proved}")
        return

    run_root_dir = None
    run_dir = None

    def is_run_id_dir(p):
        # A run-id directory contains per-slice IO and run_results.json; metadata.json is also expected
        return os.path.exists(os.path.join(p, "run_results.json")) or os.path.exists(os.path.join(p, "metadata.json"))

    def is_run_root_dir(p):
        # A runs root contains subdirectories named run_*
        try:
            return any(d.startswith("run_") and os.path.isdir(os.path.join(p, d)) for d in os.listdir(p))
        except Exception:
            return False

    # Determine input
    specified_run_dir = None  # Track if user specified a specific run
    
    # Check if --from/--dsperse-file is provided
    dsperse_file = None
    if hasattr(args, 'dsperse_file') and args.dsperse_file:
        dsperse_file = normalize_path(args.dsperse_file)
        if not os.path.exists(dsperse_file):
            print(f"{Fore.RED}Error: Dsperse file not found: {dsperse_file}{Style.RESET_ALL}")
            return
        # If dsperse file is provided, use it as candidate (will be unpacked)
        candidate = dsperse_file
    elif hasattr(args, 'run_dir') and args.run_dir:
        candidate = normalize_path(args.run_dir)
        # Check if this is a specific run directory (contains run_results.json)
        if os.path.exists(os.path.join(candidate, "run_results.json")) or os.path.exists(os.path.join(candidate, "metadata.json")):
            specified_run_dir = candidate
    else:
        # No flags provided - automatically use latest run from current directory
        current_run_dir = os.path.join(os.getcwd(), "run")
        if os.path.exists(current_run_dir) and os.path.exists(os.path.join(current_run_dir, "metadata.json")):
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

    # Normalize candidate from prompt
    candidate = normalize_path(candidate)

    # Check if candidate is a .dsperse or .dslice file - unpack if needed
    # Also check if dsperse_file was explicitly provided via --from flag
    unpacked_dir = None
    cleanup_unpacked = False
    if dsperse_file or (os.path.isfile(candidate) and Path(candidate).suffix in ['.dsperse', '.dslice']):
        p = Path(candidate)
        if p.suffix in ['.dsperse', '.dslice']:
            print(f"{Fore.CYAN}Detected archive file: {candidate}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Unpacking to same directory...{Style.RESET_ALL}")
            try:
                # Import Converter here to avoid circular imports
                from dsperse.src.slice.utils.converter import Converter
                
                # Unpack to same directory as archive (e.g., slices.dsperse -> slices/)
                if p.suffix == '.dsperse':
                    unpacked_dir = Converter._dsperse_to_dirs(p, p.parent / p.stem, expand_slices=True)
                else:
                    # For dslice, unpack to same directory
                    unpacked_dir = Converter.convert(str(candidate), output_type='dirs', output_path=str(p.parent / p.stem), cleanup=False)
                
                unpacked_path = Path(unpacked_dir)
                
                # Look for run directory - prioritize parent directory (where dsperse file is)
                # The run directory is typically in the parent model directory (e.g., net/run/)
                parent_run = unpacked_path.parent / "run"
                
                # Also check inside unpacked directory
                unpacked_run = unpacked_path / "run"
                
                # Search recursively for run directories in unpacked structure
                run_candidates = []
                if unpacked_run.exists():
                    run_candidates.append(unpacked_run)
                if parent_run.exists():
                    run_candidates.append(parent_run)
                
                # Walk unpacked directory to find any run directories
                for root, dirs, files in os.walk(unpacked_path):
                    if 'run' in dirs:
                        run_candidates.append(Path(root) / "run")
                
                # Find the run directory - prefer parent run directory (most common case)
                # dsperse files don't include run folders, so run is always in parent directory
                if parent_run.exists() and (parent_run / "metadata.json").exists():
                    # Parent has run directory with metadata.json - this is the run root
                    # If user didn't specify a specific run, use latest run that has run_results.json
                    if not specified_run_dir:
                        all_runs = get_all_runs(str(parent_run))
                        if all_runs:
                            # Filter to only runs that have run_results.json (completed runs)
                            valid_runs = [r for r in all_runs if os.path.exists(os.path.join(r, "run_results.json"))]
                            if valid_runs:
                                # Use latest valid run automatically - ensure it's normalized
                                candidate = normalize_path(valid_runs[-1])
                                logger.info(f"Unpacked archive to {unpacked_dir}, using latest valid run: {candidate}")
                            else:
                                # No valid runs yet, use run root and let user choose
                                candidate = normalize_path(str(parent_run))
                                logger.info(f"Unpacked archive to {unpacked_dir}, using run root (no valid runs yet): {candidate}")
                        else:
                            candidate = normalize_path(str(parent_run))
                            logger.info(f"Unpacked archive to {unpacked_dir}, using run root: {candidate}")
                    else:
                        # User specified a run - normalize it
                        candidate = normalize_path(specified_run_dir)
                        logger.info(f"Unpacked archive to {unpacked_dir}, using specified run: {candidate}")
                elif unpacked_run.exists() and (unpacked_run / "metadata.json").exists():
                    # Unpacked directory has run directory (unlikely but possible)
                    if not specified_run_dir:
                        all_runs = get_all_runs(str(unpacked_run))
                        if all_runs:
                            # Filter to valid runs and use latest
                            valid_runs = [r for r in all_runs if os.path.exists(os.path.join(r, "run_results.json"))]
                            if valid_runs:
                                candidate = normalize_path(valid_runs[-1])
                            else:
                                candidate = normalize_path(all_runs[-1])
                        else:
                            candidate = normalize_path(str(unpacked_run))
                    else:
                        candidate = normalize_path(specified_run_dir)
                elif run_candidates:
                    # Use first valid run candidate
                    candidate = normalize_path(str(run_candidates[0]))
                else:
                    # No run directory found - try parent run directory
                    if parent_run.exists():
                        if not specified_run_dir:
                            all_runs = get_all_runs(str(parent_run))
                            if all_runs:
                                # Filter to only runs that have run_results.json (completed runs)
                                valid_runs = [r for r in all_runs if os.path.exists(os.path.join(r, "run_results.json"))]
                                if valid_runs:
                                    candidate = normalize_path(valid_runs[-1])  # Already normalized
                                else:
                                    candidate = normalize_path(str(parent_run))
                            else:
                                candidate = normalize_path(str(parent_run))
                        else:
                            candidate = normalize_path(specified_run_dir)
                    else:
                        candidate = normalize_path(unpacked_dir)
                
                cleanup_unpacked = False  # Don't cleanup - keep unpacked files in same dir
                logger.info(f"Unpacked archive to {unpacked_dir}, using run directory: {candidate}")
            except Exception as e:
                print(f"{Fore.RED}Error unpacking archive: {e}{Style.RESET_ALL}")
                logger.error(f"Error unpacking archive: {e}")
                return

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
        
        # If we unpacked a dsperse and didn't specify a run, use latest automatically
        if unpacked_dir and not specified_run_dir:
            run_dir = all_runs[-1]
            logger.info(f"Using latest run from unpacked dsperse: {run_dir}")
        else:
            # Prompt user to choose run
            run_names = [os.path.basename(p) for p in all_runs]
            default_run = run_names[-1]
            run_list = ", ".join(run_names)
            print(f"We found {len(all_runs)} runs, {run_list}, enter which run you would like to verify (default {default_run}):")
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
        print(f"{Fore.RED}Error: Provided path is neither a runs root (contains run_*/ subdirs) nor a run directory (contains metadata.json/run_results.json){Style.RESET_ALL}")
        return

    # Validate and derive inputs
    run_dir = normalize_path(run_dir)
    run_root_dir = normalize_path(run_root_dir)
    run_results_path = os.path.join(run_dir, "run_results.json")
    if not os.path.exists(run_results_path):
        print(f"{Fore.YELLOW}Warning: run_results.json not found in {run_dir}; proceeding — it will be created/updated.{Style.RESET_ALL}")

    # Determine data_path
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
        print(f"{Fore.RED}Error: Could not determine data_path for verification. Provide it explicitly or ensure run metadata contains source_path/model_path.{Style.RESET_ALL}")
        return

    print("verifying...")

    try:
        verifier = Verifier()
        start_time = time.time()
        result = verifier.verify(run_dir, data_path)
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
            print(f"Verified slices: {execution_chain.get('ezkl_verified_slices', 0)} of {execution_chain.get('ezkl_proved_slices', 0)}")
            denom = execution_chain.get('ezkl_proved_slices', 0) or 1
            print(f"Verification percentage: {(execution_chain.get('ezkl_verified_slices', 0) / denom * 100):.1f}%")
        else:
            print(f"\n{Fore.YELLOW}No verification results found{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error verifying run: {e}{Style.RESET_ALL}")
        traceback.print_exc()
    finally:
        # Note: We don't cleanup unpacked directories - they're unpacked to the same location as the archive
        # This ensures paths remain consistent and files are accessible
        pass