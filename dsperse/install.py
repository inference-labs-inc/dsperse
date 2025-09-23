"""
Dsperse dependency installer - installs EZKL and related dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request
import tempfile
import asyncio
import logging
import ezkl
from packaging import version
from dsperse.src.constants import EZKL_PATH

logger = logging.getLogger(__name__)

MIN_EZKL_VERSION = "22.0.0"


def run_command(cmd, shell=False, capture_output=True, check=True):
    """Run a shell command and return result"""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=capture_output, text=True, check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return e


def check_ezkl():
    """Check if EZKL is installed and get version"""
    ezkl_path = shutil.which("ezkl")
    if not ezkl_path:
        if EZKL_PATH.exists():
            ezkl_path = str(EZKL_PATH)
        else:
            return None, None

    try:
        result = run_command([ezkl_path, "--version"], check=False)
        version_output = result.stdout or result.stderr or ""

        parts = version_output.strip().split()
        if parts:
            return ezkl_path, parts[-1]

        return ezkl_path, version_output.strip()
    except Exception:
        pass

    return ezkl_path, None


def install_ezkl_official():
    """Install EZKL using the official installer script"""
    logger.info("Installing EZKL from the official source...")

    result = run_command(
        "curl -fsSL https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash",
        shell=True,
        check=False,
    )

    if result.returncode == 0:
        logger.info("✓ EZKL installed via official script")
        if EZKL_PATH.parent.exists():
            os.environ["PATH"] = f"{EZKL_PATH.parent}:{os.environ.get('PATH', '')}"
        return True
    else:
        logger.error("Failed to install EZKL via the official script")
        return False


version_ge = lambda v1, v2: version.parse(v1) >= version.parse(v2)


def install_deps(skip_pip=True, interactive=False, force=False):
    """Main installation function that can be called programmatically"""

    ezkl_path, ezkl_version = check_ezkl()

    if ezkl_path:
        logger.info(f"EZKL already installed: {ezkl_path}")
        if ezkl_version:
            logger.info(f"EZKL version: {ezkl_version}")
            if not version_ge(ezkl_version, MIN_EZKL_VERSION):
                logger.warning(
                    f"EZKL version {ezkl_version} is older than recommended {MIN_EZKL_VERSION}"
                )
                if interactive:
                    response = input("Upgrade EZKL? [y/N]: ").lower()
                    if response == "y":
                        install_ezkl_official()
    else:
        if not install_ezkl_official():
            logger.error("Failed to install EZKL")
            logger.info("Please install EZKL manually from:")
            logger.info("  https://github.com/zkonduit/ezkl#installation")
            return False

    ezkl_path, ezkl_version = check_ezkl()

    if ezkl_path:
        logger.info("✓ EZKL is ready")
        return True
    else:
        logger.error("EZKL not found after installation")
        return False
