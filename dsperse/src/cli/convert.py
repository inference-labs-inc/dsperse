"""
CLI module for converting between DSperse packaging formats.

Supported conversions (auto-detected by input path):
- .dsperse  -> directory (with optional expansion of embedded .dslice files)
- directory -> .dsperse (expects dsperse-style metadata.json and slices/)
- .dslice   -> directory
- directory (slice dir with metadata.json + payload/) -> .dslice
"""

from pathlib import Path
from colorama import Fore, Style

from dsperse.src.cli.base import logger, normalize_path, prompt_for_value
from dsperse.src.slice.utils.converter import Converter


def setup_parser(subparsers):
    """
    Set up the argument parser for the convert command.

    Args:
        subparsers: The subparsers object from argparse

    Returns:
        The created parser
    """
    parser = subparsers.add_parser("convert", aliases=["cv"], help="Convert between .dsperse/.dslice and directory layouts")
    parser.set_defaults(command="convert")

    parser.add_argument("--input", "-i", dest="input_path", help="Input path: a .dsperse/.dslice file or a directory")
    parser.add_argument("--to", "--output-type", choices=["dirs", "dslice", "dsperse"], dest="to_type",
                        help="Desired output type")
    parser.add_argument("--output", "-o", dest="output_path", help="Output path: a directory or target .dsperse/.dslice file (optional)")
    parser.add_argument("--expand-slices", action="store_true", help="When converting .dsperse -> directory, also extract embedded .dslice files into subfolders")
    parser.add_argument("--cleanup", dest="cleanup", action="store_true", default=True, help="Remove source artifact after successful conversion (default: True)")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false", help="Keep source artifact")

    return parser




def convert(args):
    """
    Execute conversion based on input and output using the centralized Converter.
    """
    # Prompt if not provided
    if not getattr(args, 'input_path', None):
        args.input_path = prompt_for_value('input', 'Enter input path (dsperse file or slices directory)')
    else:
        args.input_path = normalize_path(args.input_path)

    input_path = args.input_path
    output_path = normalize_path(args.output_path) if getattr(args, 'output_path', None) else None

    p = Path(input_path)
    if not p.exists():
        print(f"{Fore.RED}Input path does not exist: {p}{Style.RESET_ALL}")
        logger.error(f"Input path does not exist: {p}")
        return

    # Prompt for output type if missing
    if not getattr(args, 'to_type', None):
        valid = {'dirs', 'dslice', 'dsperse'}
        while True:
            chosen_raw = prompt_for_value('to', 'Enter desired output type (dirs, dslice, dsperse)')
            chosen = Path(str(chosen_raw)).name.strip().lower()
            if chosen in valid:
                args.to_type = chosen
                break
            print(f"{Fore.YELLOW}Invalid choice: '{chosen}'. Please enter one of: dirs, dslice, dsperse{Style.RESET_ALL}")

    try:
        # Special-case: dsperse -> dirs with optional expand toggle
        if p.is_file() and p.suffix == '.dsperse' and getattr(args, 'to_type', None) == 'dirs':
            result = Converter._dsperse_to_dirs(p, output_path, expand_slices=bool(getattr(args, 'expand_slices', False)))
            if bool(getattr(args, 'cleanup', True)):
                try:
                    p.unlink()
                except Exception:
                    pass
        else:
            # Extract parameters for conversion
            output_type = getattr(args, 'to_type', None)
            cleanup = bool(getattr(args, 'cleanup', True))
            result = Converter.convert(
                input_path,
                output_type=output_type,
                output_path=output_path,
                cleanup=cleanup
            )

        print(f"{Fore.GREEN}✓ Converted to: {result}{Style.RESET_ALL}")
        logger.info(f"Converted to: {result}")
    except Exception as e:
        print(f"{Fore.RED}Error converting: {e}{Style.RESET_ALL}")
        logger.error(f"Error converting: {e}")
