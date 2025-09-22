"""
Dsperse CLI package.
Contains modules for the Dsperse command-line interface.
"""

from src.cli.base import DsperseArgumentParser, print_header, print_easter_egg, configure_logging, logger
from src.cli.slice import setup_parser as setup_slice_parser, slice_model
from src.cli.run import setup_parser as setup_run_parser, run_inference
from src.cli.prove import setup_parser as setup_prove_parser, run_proof
from src.cli.verify import setup_parser as setup_verify_parser, verify_proof
from src.cli.compile import setup_parser as setup_compile_parser, compile_model
from src.cli.full_run import setup_parser as setup_full_run_parser, full_run

__all__ = [
    'DsperseArgumentParser',
    'print_header',
    'print_easter_egg',
    'configure_logging',
    'logger',
    'setup_slice_parser',
    'slice_model',
    'setup_run_parser',
    'run_inference',
    'setup_prove_parser',
    'run_proof',
    'setup_verify_parser',
    'verify_proof',
    'setup_compile_parser',
    'compile_model',
    'setup_full_run_parser',
    'full_run'
]
