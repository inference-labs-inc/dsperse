import json
import logging
import os
import subprocess
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from dsperse.src.run.utils.runner_utils import RunnerUtils

logger = logging.getLogger(__name__)

class EZKLUtils:
    """Utility class for EZKL backend operations."""

    @staticmethod
    def detect_srs_error(stderr_output: str) -> Optional[str]:
        """Detect SRS-related errors in EZKL stderr output."""
        if not stderr_output:
            return None

        stderr_lower = stderr_output.lower()
        srs_error_patterns = [
            "srs", "structured reference string", "trusted setup",
            "ceremony", "powers of tau", "no srs file",
            "srs not found", "missing srs", "srs table", "srs path",
        ]

        for pattern in srs_error_patterns:
            if pattern in stderr_lower:
                return f"SRS Error Detected: {stderr_output.strip()}"

        return None

    @staticmethod
    def run_ezkl_command_with_srs_check(
        cmd_list: list, env: dict = None, check: bool = True, capture_output: bool = True, text: bool = True, **kwargs
    ) -> subprocess.CompletedProcess:
        """Wrapper for subprocess.run that detects SRS errors and bubbles them up."""
        try:
            process = subprocess.run(
                cmd_list, env=env, check=check, capture_output=capture_output, text=text, **kwargs
            )

            if process.stderr:
                srs_error = EZKLUtils.detect_srs_error(process.stderr)
                if srs_error:
                    logger.error(f"EZKL SRS Error in command '{' '.join(cmd_list)}': {srs_error}")
                    raise RuntimeError(f"DSperse detected SRS error: {srs_error}")

            return process

        except subprocess.CalledProcessError as e:
            if e.stderr:
                srs_error = EZKLUtils.detect_srs_error(e.stderr)
                if srs_error:
                    logger.error(f"EZKL SRS Error in command '{' '.join(cmd_list)}': {srs_error}")
                    raise RuntimeError(f"DSperse detected SRS error: {srs_error}")
            raise

    @staticmethod
    def initialize_compilation_artifacts(model_path: Path, output_path: Path, input_file_path: Optional[str]) -> Dict[str, Any]:
        """Initialize paths and metadata for EZKL compilation."""
        settings_path = output_path / "settings.json"
        compiled_path = output_path / "model.compiled"
        vk_path = output_path / "vk.key"
        pk_path = output_path / "pk.key"
        calibration_dest = output_path / "calibration.json" if input_file_path else None

        return {
            "paths": {
                "settings": settings_path,
                "compiled": compiled_path,
                "vk": vk_path,
                "pk": pk_path,
                "calibration": calibration_dest,
            },
            "data": {
                "settings": str(settings_path),
                "compiled": str(compiled_path),
                "vk_key": str(vk_path),
                "pk_key": str(pk_path),
                "calibration": str(calibration_dest) if calibration_dest else None,
            }
        }

    @staticmethod
    def process_witness_output(witness_data: dict) -> Optional[dict]:
        """Process the witness.json data to get prediction results."""
        try:
            rescaled_outputs = witness_data["pretty_elements"]["rescaled_outputs"][0]
            float_values = [float(val) for val in rescaled_outputs]
            tensor_output = torch.tensor([float_values], dtype=torch.float32)
            return RunnerUtils.process_final_output(tensor_output)
        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Could not process witness data: {e}")
            return None
