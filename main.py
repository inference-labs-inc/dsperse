#!/usr/bin/env python3
"""
Kubz CLI - A command-line interface for the Kubz neural network model slicing and analysis toolkit.

This CLI allows you to slice models and run verified inference (both whole and sliced) using
different backends (ezkl, plain).
"""

import sys
import random
import logging
from colorama import Fore, Style

# Import CLI modules
from src.cli import (
    KubzArgumentParser, print_header, print_easter_egg, configure_logging, logger,
    setup_slice_parser, slice_model,
    setup_run_parser, run_inference,
    setup_prove_parser, run_proof,
    setup_verify_parser, verify_proof,
    setup_circuitize_parser, circuitize_model
)

def main():
    """Main entry point for the Kubz CLI."""
    # Create the main parser
    parser = KubzArgumentParser(
        description="Kubz - Distributed zkML Toolkit",
        formatter_class=sys.modules['argparse'].RawDescriptionHelpFormatter,
        epilog=f"Made with {Fore.RED}❤️{Style.RESET_ALL}  by the Inference Labs team"
    )

    # Add version argument
    parser.add_argument('--version', action='version', version='Kubz CLI v1.0.0')

    # Add logging level argument
    parser.add_argument('--log-level', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='WARNING',
                      help='Set the logging level (default: WARNING)')

    # Add easter egg argument
    parser.add_argument('--easter-egg', action='store_true', help=sys.modules['argparse'].SUPPRESS)

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Set up parsers for each command
    setup_slice_parser(subparsers)
    setup_run_parser(subparsers)
    setup_prove_parser(subparsers)
    setup_verify_parser(subparsers)
    setup_circuitize_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_level)
    logger.debug(f"Logging configured with level: {args.log_level}")

    # Print header
    print_header()

    # Handle easter egg
    if args.easter_egg:
        print_easter_egg()
        return

    # Handle commands
    if args.command == 'slice':
        slice_model(args)
    elif args.command == 'run':
        # Set plain as default if no backend specified
        if not args.ezkl:
            args.plain = True
        run_inference(args)
    elif args.command == 'prove':
        run_proof(args)
    elif args.command == 'verify':
        verify_proof(args)
    elif args.command == 'circuitize':
        circuitize_model(args)
    else:
        # If no command is provided, show help
        parser.print_help()
        # Show an easter egg 20% of the time
        if random.random() < 0.2:
            print_easter_egg()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
