import logging
from dsperse.install import install_deps, check_ezkl

logger = logging.getLogger(__name__)


def ensure_dependencies():
    """Check and install dependencies"""
    ezkl_path, ezkl_version = check_ezkl()

    if ezkl_path:
        return True

    logger.info("EZKL not found. Installing dependencies...")

    if install_deps(skip_pip=True, interactive=False):
        logger.info("Dependencies installed successfully!")
        return True
    else:
        logger.error("Failed to install dependencies automatically.")
        logger.info("Please install EZKL manually from:")
        logger.info("https://github.com/zkonduit/ezkl#installation")
        return False
