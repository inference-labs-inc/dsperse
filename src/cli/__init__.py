"""
Kubz CLI package.
Contains modules for the Kubz command-line interface.
"""

from src.cli.base import KubzArgumentParser, print_header, print_easter_egg, configure_logging, logger
from src.cli.slice import setup_parser as setup_slice_parser, slice_model
from src.cli.run import setup_parser as setup_run_parser, run_inference
from src.cli.prove import setup_parser as setup_prove_parser, run_proof
from src.cli.verify import setup_parser as setup_verify_parser, verify_proof
from src.cli.circuitize import setup_parser as setup_circuitize_parser, circuitize_model

__all__ = [
    'KubzArgumentParser',
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
    'setup_circuitize_parser',
    'circuitize_model'
]
