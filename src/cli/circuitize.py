"""
CLI module for circuitizing models using EZKL.
"""

import os
import traceback
import logging
from colorama import Fore, Style

from src.cicuitizers.circuitizer import Circuitizer
from src.cli.base import check_model_dir, prompt_for_value, logger


def setup_parser(subparsers):
    """
    Set up the argument parser for the circuitize command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    circuitize_parser = subparsers.add_parser('circuitize', help='Circuitize a model or slices using EZKL')
    circuitize_parser.add_argument('--model-path', help='Path to the model file or directory containing slices')
    circuitize_parser.add_argument('--input-file', help='Path to input file for calibration (optional)')
    
    return circuitize_parser


def circuitize_model(args):
    """
    Circuitize a model based on the provided arguments.

    Args:
        args: The parsed command-line arguments
    """
    print(f"{Fore.CYAN}Circuitizing model with EZKL...{Style.RESET_ALL}")
    logger.info("Starting model circuitization")

    # Prompt for model path if not provided
    if not hasattr(args, 'model_path') or not args.model_path:
        args.model_path = prompt_for_value('model-path', 'Enter the path to the model file or directory containing slices')

    if not check_model_dir(args.model_path):
        return

    # Initialize the Circuitizer
    try:
        circuitizer = Circuitizer.create(args.model_path)
        logger.info(f"Circuitizer initialized successfully")
    except RuntimeError as e:
        error_msg = f"Failed to initialize Circuitizer: {e}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        return

    # Run the circuitization
    try:
        output_path = circuitizer.circuitize(
            model_path=args.model_path,
            input_file=args.input_file
        )
        success_msg = f"Model circuitized successfully! Output saved to {output_path}"
        print(f"{Fore.GREEN}✓ {success_msg}{Style.RESET_ALL}")
        logger.info(success_msg)
    except Exception as e:
        error_msg = f"Error circuitizing model: {e}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        logger.debug("Stack trace:", exc_info=True)
        traceback.print_exc()