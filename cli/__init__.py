"""
Kubz CLI package.
Contains modules for the Kubz command-line interface.
"""

from cli.base import KubzArgumentParser, print_header, print_easter_egg
from cli.slice import setup_parser as setup_slice_parser, slice_model
from cli.run import setup_parser as setup_run_parser, run_inference
from cli.prove import setup_parser as setup_prove_parser, run_proof
from cli.verify import setup_parser as setup_verify_parser, verify_proof

__all__ = [
    'KubzArgumentParser',
    'print_header',
    'print_easter_egg',
    'setup_slice_parser',
    'slice_model',
    'setup_run_parser',
    'run_inference',
    'setup_prove_parser',
    'run_proof',
    'setup_verify_parser',
    'verify_proof'
]
