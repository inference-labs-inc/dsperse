#!/usr/bin/env python3
"""
Dsperse CLI - A command-line interface for the Dsperse neural network model slicing and analysis toolkit.

This CLI allows you to slice models and run verified inference on sliced models, with the runner
automatically determining the appropriate backend (EZKL or ONNX) for each slice.
"""

import sys
import random
import logging
import importlib.metadata
from colorama import Fore, Style

# Check dependencies before importing main modules
from dsperse.deps_checker import ensure_dependencies

# Import CLI modules
from dsperse.src.cli import (
    DsperseArgumentParser,
    print_header,
    print_easter_egg,
    configure_logging,
    logger,
    setup_slice_parser,
    slice_model,
    setup_run_parser,
    run_inference,
    setup_prove_parser,
    run_proof,
    setup_verify_parser,
    verify_proof,
    setup_compile_parser,
    compile_model,
    setup_full_run_parser,
    full_run,
)


def main():
    """Main entry point for the Dsperse CLI."""
    # Check and install dependencies if needed (except for help/version)
    if len(sys.argv) > 1 and sys.argv[1] not in ["--help", "-h", "--version"]:
        if not ensure_dependencies():
            print(f"{Fore.RED}Failed to ensure dependencies. Some features may not work.{Style.RESET_ALL}", file=sys.stderr)
            print(f"{Fore.YELLOW}Proceeding anyway...{Style.RESET_ALL}", file=sys.stderr)

    # Create the main parser
    parser = DsperseArgumentParser(
        description="Dsperse - Distributed zkML Toolkit",
        formatter_class=sys.modules["argparse"].RawDescriptionHelpFormatter,
        epilog=f"Made with {Fore.RED}❤️{Style.RESET_ALL}  by the Inference Labs team",
    )

    try:
        version = importlib.metadata.version("dsperse")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    parser.add_argument(
        "--version", action="version", version=f"Dsperse CLI v{version}"
    )

    # Add logging level argument
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )

    # Add easter egg argument
    parser.add_argument(
        "--easter-egg", action="store_true", help=sys.modules["argparse"].SUPPRESS
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Set up parsers for each command
    setup_slice_parser(subparsers)
    setup_run_parser(subparsers)
    setup_prove_parser(subparsers)
    setup_verify_parser(subparsers)
    setup_compile_parser(subparsers)
    setup_full_run_parser(subparsers)

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
    if args.command == "slice":
        # Dispatch slice sub-commands
        if getattr(args, "slice_subcommand", None) == "convert":
            from dsperse.src.cli.slice import slice_convert
            slice_convert(args)
        else:
            slice_model(args)
    elif args.command == "run":
        run_inference(args)
    elif args.command == "prove":
        run_proof(args)
    elif args.command == "verify":
        verify_proof(args)
    elif args.command == "compile":
        compile_model(args)
    elif args.command == "full-run":
        full_run(args)
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
