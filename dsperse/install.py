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
import ezkl
from packaging import version

MIN_EZKL_VERSION = "22.0.0"
MIN_LOGROWS = 2
MAX_LOGROWS = 24


def info(msg):
    print(f"[INFO] {msg}", file=sys.stderr)


def warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)


def error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


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
        ezkl_bin = Path.home() / ".ezkl" / "ezkl"
        if ezkl_bin.exists():
            ezkl_path = str(ezkl_bin)
        else:
            return None, None

    try:
        result = run_command([ezkl_path, "--version"], check=False)
        version_output = result.stdout or result.stderr or ""
        # Extract version number (e.g., "22.0.1")
        import re

        version_match = re.search(r"\d+\.\d+\.\d+", version_output)
        if version_match:
            return ezkl_path, version_match.group()
    except Exception:
        pass

    return ezkl_path, None


def install_ezkl_official():
    """Install EZKL using the official installer script"""
    info("Installing EZKL from the official source...")

    result = run_command(
        "curl -fsSL https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash",
        shell=True,
        check=False,
    )

    if result.returncode == 0:
        info("✓ EZKL installed via official script")
        ezkl_bin_dir = Path.home() / ".ezkl"
        if ezkl_bin_dir.exists():
            os.environ["PATH"] = f"{ezkl_bin_dir}:{os.environ.get('PATH', '')}"
        return True
    else:
        error("Failed to install EZKL via the official script")
        return False


def check_srs_files():
    """Check which SRS files are present"""
    srs_dir = Path.home() / ".ezkl" / "srs"
    if not srs_dir.exists():
        return [], list(range(MIN_LOGROWS, MAX_LOGROWS + 1))

    present = []
    missing = []

    for logrows in range(MIN_LOGROWS, MAX_LOGROWS + 1):
        srs_file = srs_dir / f"kzg{logrows}.srs"
        if srs_file.exists():
            present.append(logrows)
        else:
            missing.append(logrows)

    return present, missing


async def download_srs_async(logrows):
    """Download SRS file using Python API"""
    await ezkl.get_srs(logrows=logrows, commitment=ezkl.PyCommitments.KZG)


def download_srs(logrows, interactive=False):
    """Download SRS file for given logrows"""
    info(f"Downloading SRS for logrows={logrows} (kzg)...")

    try:
        asyncio.run(download_srs_async(logrows))
        info(f"✓ Downloaded kzg{logrows}.srs")
        return True
    except Exception as e:
        warn(f"Failed to download SRS for logrows={logrows}: {e}")
        return False


def ensure_srs_files(interactive=False, critical_only=False):
    """Ensure SRS files are present"""
    present, missing = check_srs_files()

    if not missing:
        info(f"✓ All required SRS files already present ({len(present)} files)")
        return True

    if critical_only:
        critical_sizes = [16, 18, 20, 22]
        to_download = [s for s in critical_sizes if s in missing]
        info(f"Downloading critical SRS files: {to_download}")
    else:
        to_download = missing
        info(f"SRS files missing for logrows: {missing}")

    if not interactive:
        warn(
            "Non-interactive mode: downloading only critical SRS files to avoid long install times"
        )
        warn(
            f"You can download more later with: ezkl get-srs --logrows <N> --commitment kzg"
        )

    success_count = 0
    fail_count = 0

    for logrows in to_download:
        if download_srs(logrows, interactive):
            success_count += 1
        else:
            fail_count += 1

    info(f"SRS download summary: success={success_count}, failed={fail_count}")

    if fail_count > 0:
        warn("Some SRS downloads failed. You can retry manually:")
        warn("ezkl get-srs --logrows <N> --commitment kzg")

    return fail_count == 0


version_ge = lambda v1, v2: version.parse(v1) >= version.parse(v2)


def install_deps(skip_pip=True, interactive=False, force=False):
    """Main installation function that can be called programmatically"""

    ezkl_path, ezkl_version = check_ezkl()

    if ezkl_path:
        info(f"EZKL already installed: {ezkl_path}")
        if ezkl_version:
            info(f"EZKL version: {ezkl_version}")
            if not version_ge(ezkl_version, MIN_EZKL_VERSION):
                warn(
                    f"EZKL version {ezkl_version} is older than recommended {MIN_EZKL_VERSION}"
                )
                if interactive:
                    response = input("Upgrade EZKL? [y/N]: ").lower()
                    if response == "y":
                        install_ezkl_official()
    else:
        if not install_ezkl_official():
            error("Failed to install EZKL")
            info("Please install EZKL manually from:")
            info("  https://github.com/zkonduit/ezkl#installation")
            return False

    ensure_srs_files(interactive=interactive, critical_only=not interactive)

    ezkl_path, ezkl_version = check_ezkl()

    if ezkl_path:
        info("✓ EZKL is ready")
        return True
    else:
        error("EZKL not found after installation")
        return False
