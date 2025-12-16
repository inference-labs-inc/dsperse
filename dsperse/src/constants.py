"""
Constants used across the Dsperse project
"""
from pathlib import Path

# EZKL configuration
MIN_EZKL_VERSION = "22.0.0"
EZKL_PATH = Path.home() / ".ezkl" / "ezkl"
SRS_DIR = Path.home() / ".ezkl" / "srs"

SRS_LOGROWS_MIN = 2
SRS_LOGROWS_MAX = 24
SRS_LOGROWS_RANGE = range(SRS_LOGROWS_MIN, SRS_LOGROWS_MAX + 1)
SRS_FILES = [f"kzg{n}.srs" for n in SRS_LOGROWS_RANGE]

# JSTprove CLI command
JSTPROVE_COMMAND = "jst"