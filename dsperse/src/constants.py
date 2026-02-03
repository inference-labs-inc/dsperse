"""
Constants used across the Dsperse project
"""
from pathlib import Path

import shutil
import os

# EZKL configuration
MIN_EZKL_VERSION = "22.0.0"

def _find_ezkl():
    # 1. Check if it's already in PATH
    path_ezkl = shutil.which("ezkl")
    if path_ezkl:
        return Path(path_ezkl)
    
    # 2. Check common installation paths
    home = Path.home()
    potential_paths = [
        home / ".ezkl" / "ezkl",
        home / ".ezkl" / "bin" / "ezkl",
        home / ".config" / ".ezkl" / "ezkl",
        home / ".config" / ".ezkl" / "bin" / "ezkl",
        home / ".config" / ".ezkl", # Some versions might name it this
        home / ".local" / "bin" / "ezkl",
        home / ".cargo" / "bin" / "ezkl",
    ]
    
    for p in potential_paths:
        if p.exists() and os.access(p, os.X_OK):
            return p
            
    # Default fallback
    return home / ".ezkl" / "ezkl"

EZKL_PATH = _find_ezkl()
SRS_DIR = Path.home() / ".ezkl" / "srs"

SRS_LOGROWS_MIN = 2
SRS_LOGROWS_MAX = 24
SRS_LOGROWS_RANGE = range(SRS_LOGROWS_MIN, SRS_LOGROWS_MAX + 1)
SRS_FILES = [f"kzg{n}.srs" for n in SRS_LOGROWS_RANGE]
