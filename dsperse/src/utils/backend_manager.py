"""
Backend manager for dsperse.

This module provides a factory for creating backend instances based on user
configuration (CLI args or environment variables), with support for multiple
proof system backends (EZKL, JSTprove, etc.).
"""

import importlib
import logging
import os
from typing import Optional

from dsperse.src.backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class BackendManager:
    """
    Factory class for managing proof system backends in dsperse.

    Handles backend selection through:
    1. Explicit CLI argument (--backend)
    2. Environment variable (DSPERSE_BACKEND)
    3. Default fallback to EZKL with warning
    """

    SUPPORTED_BACKENDS = {
        'ezkl': ('dsperse.src.backends.ezkl', 'EZKL'),
        'jstprove': ('dsperse.src.backends.JSTprove', 'JSTprove'),
    }

    @staticmethod
    def get_backend(
        backend_name: Optional[str] = None,
        model_directory: Optional[str] = None
    ) -> BaseBackend:
        """
        Get a backend instance based on configuration hierarchy.

        Priority order:
        1. backend_name parameter (from CLI --backend)
        2. DSPERSE_BACKEND environment variable
        3. Default to 'ezkl' with warning

        Args:
            backend_name: Explicit backend name ('ezkl' or 'jstprove')
            model_directory: Optional model directory for backend initialization

        Returns:
            Configured backend instance

        Raises:
            ValueError: If backend_name is not supported
            ImportError: If backend module/class cannot be imported
        """
        # Determine backend name
        if backend_name is None:
            backend_name = os.environ.get('DSPERSE_BACKEND', 'ezkl')
            if backend_name == 'ezkl':
                logger.warning(
                    "No backend specified. Defaulting to EZKL. "
                    "Set --backend flag or DSPERSE_BACKEND environment variable to choose a backend."
                )

        # Validate backend name
        if backend_name not in BackendManager.SUPPORTED_BACKENDS:
            supported = ', '.join(BackendManager.SUPPORTED_BACKENDS.keys())
            raise ValueError(
                f"Unsupported backend '{backend_name}'. "
                f"Supported backends: {supported}"
            )

        # Get module and class info
        module_name, class_name = BackendManager.SUPPORTED_BACKENDS[backend_name]

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Get the class
            backend_class = getattr(module, class_name)

            # Instantiate and return
            logger.debug(f"Initializing {backend_name} backend")
            return backend_class(model_directory=model_directory)

        except ImportError as e:
            raise ImportError(
                f"Could not import backend '{backend_name}' from '{module_name}': {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Could not find class '{class_name}' in module '{module_name}': {e}"
            ) from e

    @staticmethod
    def list_supported_backends() -> list[str]:
        """
        Get list of supported backend names.

        Returns:
            List of supported backend names
        """
        return list(BackendManager.SUPPORTED_BACKENDS.keys())
