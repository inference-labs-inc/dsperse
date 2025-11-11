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
from dsperse.src.slice.utils.converter import Converter

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

    prove_parser.add_argument('--run-dir', '--rd', dest='run_dir', help='Specific run directory to prove (defaults to latest run)')
    prove_parser.add_argument('--from', '--dsperse-file', '--dsperse', dest='dsperse_file', help='Path to .dsperse file (will unpack and use latest run automatically)')
    prove_parser.add_argument('--output-file', '-o', dest='output_file', help='Path to save output results')

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

    run_root_dir = None
    run_dir = None

    # Helper predicates
    def is_run_id_dir(p):
        return os.path.exists(os.path.join(p, "run_result.json"))

    def is_run_root_dir(p):
        return os.path.exists(os.path.join(p, "metadata.json"))

    # Determine input
    default_model_path = None  # Initialize at function scope
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
        # Check if this is a specific run directory (contains run_result.json)
        if os.path.exists(os.path.join(candidate, "run_result.json")):
            specified_run_dir = candidate
    else:
        # No flags provided - automatically use latest run from current directory
        current_run_dir = os.path.join(os.getcwd(), "run")
        if os.path.exists(current_run_dir) and os.path.exists(os.path.join(current_run_dir, "metadata.json")):
            latest_run = get_latest_run(current_run_dir)
            if latest_run and os.path.exists(os.path.join(latest_run, "run_result.json")):
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
                    # If user didn't specify a specific run, use latest run that has run_result.json
                    if not specified_run_dir:
                        all_runs = get_all_runs(str(parent_run))
                        if all_runs:
                            # Filter to only runs that have run_result.json (completed runs)
                            valid_runs = [r for r in all_runs if os.path.exists(os.path.join(r, "run_result.json"))]
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
                            valid_runs = [r for r in all_runs if os.path.exists(os.path.join(r, "run_result.json"))]
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
                                candidate = all_runs[-1]  # Already normalized
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
        result = prover.prove(run_result_path, metadata_path)
        elapsed_time = time.time() - start_time

        print(f"{Fore.GREEN}✓ Proof generation completed in {elapsed_time:.2f} seconds!{Style.RESET_ALL}")
        
        # If we unpacked a dsperse file, repack it with proofs (exclude pk for verification)
        if unpacked_dir:
            try:
                unpacked_path = Path(unpacked_dir)
                # Determine the proved dsperse filename (add _proved suffix)
                original_dsperse = None
                # Check if dsperse_file was provided via --from flag
                if dsperse_file:
                    original_dsperse = Path(dsperse_file)
                elif hasattr(args, 'run_dir') and args.run_dir:
                    original_path = Path(args.run_dir)
                    if original_path.suffix in ['.dsperse', '.dslice']:
                        original_dsperse = original_path
                
                if original_dsperse:
                    proved_dsperse = original_dsperse.parent / f"{original_dsperse.stem}_proved.dsperse"
                else:
                    # Use unpacked directory name
                    proved_dsperse = unpacked_path.parent / f"{unpacked_path.name}_proved.dsperse"
                
                print(f"{Fore.CYAN}Repacking with proofs (excluding pk files for verification)...{Style.RESET_ALL}")
                logger.info(f"Repacking proved slices to {proved_dsperse} (excluding pk files)")
                
                # Repack with pk exclusion (for verification)
                from dsperse.src.slice.utils.converter import Converter
                Converter.convert(str(unpacked_path), output_type='dsperse', output_path=str(proved_dsperse), cleanup=False, exclude_patterns=['*pk*'])
                
                print(f"{Fore.GREEN}✓ Repacked to {proved_dsperse} (ready for verification){Style.RESET_ALL}")
                logger.info(f"Repacked proved slices to {proved_dsperse}")
            except Exception as e:
                logger.warning(f"Could not repack proved slices: {e}")
                print(f"{Fore.YELLOW}Warning: Could not repack proved slices: {e}{Style.RESET_ALL}")
        
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
        # Note: We don't cleanup unpacked directories - they're unpacked to the same location as the archive
        # This ensures paths remain consistent and files are accessible
        pass
