"""
CLI module for generating proofs for models.
"""

import os
import time
import traceback
import glob
from colorama import Fore, Style

from src.prover import Prover
from src.cli.base import check_model_dir, save_result, prompt_for_value

def setup_parser(subparsers):
    """
    Set up the argument parser for the prove command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    prove_parser = subparsers.add_parser('prove', help='Generate a proof for a model')
    prove_parser.add_argument('--run-dir', help='Specific run directory to prove')
    prove_parser.add_argument('--output-file', help='Path to save output results')

    return prove_parser

def get_all_runs(model_dir):
    """
    Get all run directories in the model's run subdirectory.
    
    Args:
        model_dir (str): Path to the model directory
        
    Returns:
        list: List of run directories, sorted by name (latest last)
    """
    run_dir = os.path.join(model_dir, "run")
    if not os.path.exists(run_dir):
        return []
    
    # Get all run directories sorted by name (which includes timestamp)
    run_dirs = sorted(glob.glob(os.path.join(run_dir, "run_*")))
    
    return run_dirs

def get_latest_run(model_dir):
    """
    Get the latest run directory in the model's run subdirectory.
    
    Args:
        model_dir (str): Path to the model directory
        
    Returns:
        str: Path to the latest run directory, or None if no runs found
    """
    run_dirs = get_all_runs(model_dir)
    
    if not run_dirs:
        return None
    
    # Return the latest run directory
    return run_dirs[-1]

def run_proof(args):
    """
    Generate a proof for a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Proving model execution...{Style.RESET_ALL}")

    # Determine if we're starting with a model directory or a run directory
    model_dir = None
    run_dir = None
    
    # If run_dir is provided, check if it's a valid run directory
    if hasattr(args, 'run_dir') and args.run_dir:
        run_dir = args.run_dir
        
        # Check if run directory exists
        if not os.path.exists(run_dir):
            print(f"{Fore.RED}Error: Run directory {run_dir} does not exist{Style.RESET_ALL}")
            return
        
        # Check for run_result.json in the run directory
        run_result_path = os.path.join(run_dir, "run_result.json")
        if not os.path.exists(run_result_path):
            print(f"{Fore.RED}Error: run_result.json not found in {run_dir}{Style.RESET_ALL}")
            return
        
        # Try to determine the model directory from the run directory
        # Assuming run directory structure is model_dir/run/run_YYYYMMDD_HHMMSS
        potential_model_dir = os.path.dirname(os.path.dirname(run_dir))
        if os.path.exists(os.path.join(potential_model_dir, "run", "metadata.json")):
            model_dir = potential_model_dir
        else:
            print(f"{Fore.RED}Error: Could not locate run metadata.json in {os.path.join(potential_model_dir, 'run')} for the provided run directory.{Style.RESET_ALL}")
            print("Please provide a run directory under a valid model directory (model_dir/run/run_*/).")
            return
    else:
        # Prompt for run directory or model directory
        dir_input = prompt_for_value('model-or-run-dir', 'Enter run directory (or model directory to choose a run)')
        
        # Check if the input is a run directory
        if os.path.exists(os.path.join(dir_input, "run_result.json")):
            run_dir = dir_input
            # Try to determine the model directory from the run directory
            potential_model_dir = os.path.dirname(os.path.dirname(run_dir))
            if os.path.exists(os.path.join(potential_model_dir, "run", "metadata.json")):
                model_dir = potential_model_dir
            else:
                print(f"{Fore.RED}Error: Could not locate run metadata.json in {os.path.join(potential_model_dir, 'run')} for the provided run directory.{Style.RESET_ALL}")
                print("Please provide a run directory under a valid model directory (model_dir/run/run_*/).")
                return
        else:
            # Assume it's a model directory
            model_dir = dir_input
            if not check_model_dir(model_dir):
                return
            
            # Get all runs in the model directory
            all_runs = get_all_runs(model_dir)
            
            if not all_runs:
                print(f"{Fore.RED}Error: No runs found in {os.path.join(model_dir, 'run')}{Style.RESET_ALL}")
                return
            
            # Display all runs and let the user select one
            run_names = [os.path.basename(run_path) for run_path in all_runs]
            default_run = run_names[-1]  # Latest run
            
            # Format the message to match the example in the issue description
            run_list = ", ".join(run_names)
            print(f"We found {len(all_runs)} runs, {run_list}, enter which run you would like to prove (default {default_run}):")
            user_input = input().strip()  # Hit enter to signify the default
            
            if not user_input:
                # User hit enter, use the default (latest) run
                run_dir = all_runs[-1]
            else:
                try:
                    # Check if the input is a number (index)
                    index = int(user_input) - 1
                    if 0 <= index < len(all_runs):
                        run_dir = all_runs[index]
                    else:
                        print(f"{Fore.RED}Error: Invalid run index{Style.RESET_ALL}")
                        return
                except ValueError:
                    # Assume the input is a run name
                    run_path = os.path.join(model_dir, "run", user_input)
                    if os.path.exists(run_path):
                        run_dir = run_path
                    else:
                        print(f"{Fore.RED}Error: Run directory {run_path} does not exist{Style.RESET_ALL}")
                        return
    
    # At this point, we should have both model_dir and run_dir
    # Check for run_result.json in the run directory
    run_result_path = os.path.join(run_dir, "run_result.json")
    if not os.path.exists(run_result_path):
        print(f"{Fore.RED}Error: run_result.json not found in {run_dir}{Style.RESET_ALL}")
        return
    
    # Check for metadata.json in the model's run directory
    metadata_path = os.path.join(model_dir, "run", "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"{Fore.RED}Error: metadata.json not found in {os.path.join(model_dir, 'run')}{Style.RESET_ALL}")
        return
        
    # Store the model_dir and run_dir in args for later use
    args.model_dir = model_dir
    args.run_dir = run_dir

    # Print proving message to match the example in the issue description
    print("proving...")

    try:
        # Create prover and prove the run
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
                default_output_file = os.path.join(args.model_dir, "proof_results.json")
                args.output_file = prompt_for_value('output-file', 'Enter the output file path', default=default_output_file, required=False)

        # Save the result if output file is specified
        if args.output_file:
            try:
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
