"""SRS manager utilities"""

import os
import json
import asyncio
from pathlib import Path
import ezkl
import logging
from dsperse.src.constants import SRS_DIR

logger = logging.getLogger(__name__)


def check_srs(logrows: int) -> bool:
    """Check if SRS file exists for given logrows"""
    srs_file = SRS_DIR / f"kzg{logrows}.srs"
    return srs_file.exists()


async def download_srs_async(logrows: int):
    """Download SRS file using Python API"""
    await ezkl.get_srs(logrows=logrows, commitment=ezkl.PyCommitments.KZG)


def ensure_srs(logrows: int) -> bool:
    """Ensure SRS file exists, download if missing"""
    if check_srs(logrows):
        return True

    logger.info(f"SRS file missing for logrows={logrows}, downloading...")

    try:
        asyncio.run(download_srs_async(logrows))
        logger.info(f"✓ Downloaded kzg{logrows}.srs")
        return True
    except Exception as e:
        logger.error(f"Failed to download SRS for logrows={logrows}: {e}")
        return False


def get_logrows_from_settings(settings_path: str) -> int:
    """Extract logrows from EZKL settings.json file"""
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)

        logrows = settings.get('run_args', {}).get('logrows')
        if logrows is not None:
            return int(logrows)

        for key in ['logrows', 'circuit_settings', 'params']:
            if key in settings and isinstance(settings[key], dict):
                logrows = settings[key].get('logrows')
                if logrows is not None:
                    return int(logrows)

        logger.warning(f"Could not find logrows in settings file: {settings_path}")
        return None

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error reading settings file {settings_path}: {e}")
        return None