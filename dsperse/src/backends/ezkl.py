import json
import os
import subprocess
import torch
import logging
import traceback
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, List
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.backends.utils.ezkl_utils import EZKLUtils
from dsperse.src.constants import EZKL_PATH
from dsperse.src.utils.srs_manager import ensure_srs, get_logrows_from_settings

logger = logging.getLogger(__name__)


class EZKL:
    """EZKL backend for zero-knowledge proof generation."""
    def __init__(self, model_directory=None):
        """
        Initialize the EZKL backend.

        Args:
            model_directory (str, optional): Path to the model directory.

        Raises:
            RuntimeError: If EZKL is not installed
        """
        self.env = os.environ
        self.model_directory = model_directory

        if model_directory:
            self.base_path = os.path.join(model_directory, "ezkl")

        # Check if ezkl is installed via cli
        try:
            result = subprocess.run(
                [str(EZKL_PATH), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError("EZKL CLI not found. Please install EZKL first.")
        except FileNotFoundError:
            raise RuntimeError("EZKL CLI not found. Please install EZKL first.")

    @staticmethod
    def get_version():
        """
        Get the EZKL version.

        Returns:
            str: EZKL version string, or None if version cannot be determined
        """
        try:
            result = subprocess.run(
                [str(EZKL_PATH), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse version from output (e.g., "ezkl 1.2.3" -> "ezkl 1.2.3")
                version_output = result.stdout.strip() or result.stderr.strip()
                return version_output
        except Exception as e:
            logger.debug(f"Could not get EZKL version: {e}")

        return None

    def generate_witness(
        self, input_file: Union[str, Path], model_path: Union[str, Path], output_file: Union[str, Path], 
        vk_path: Union[str, Path], settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Any]:
        """Generate a witness for the given model and input."""
        # --- Normalization & Validation ---
        input_file, model_path, output_file, vk_path = str(input_file), str(model_path), str(output_file), str(vk_path)
        if not os.path.exists(input_file): raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vk_path): raise FileNotFoundError(f"Verification key file not found: {vk_path}")

        # --- SRS Setup ---
        if settings_path and os.path.exists(settings_path):
            logrows = get_logrows_from_settings(str(settings_path))
            if logrows and not ensure_srs(logrows): return False, f"Failed to ensure SRS for logrows={logrows}"

        # --- Execution ---
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            cmd = [
                str(EZKL_PATH), "gen-witness", "--data", input_file,
                "--compiled-circuit", model_path, "--output", output_file, "--vk-path", vk_path,
            ]
            process = subprocess.run(cmd, env=self.env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Witness generation failed: {e.stderr}")
            return False, e.stderr

        # --- Result Processing ---
        try:
            with open(output_file, "r") as f:
                witness_data = json.load(f)
                output = self.process_witness_output(witness_data)
            return True, output
        except Exception as e:
            logger.error(f"Failed to process witness output: {e}")
            return False, str(e)

    def prove(
        self, witness_path: Union[str, Path], model_path: Union[str, Path], proof_path: Union[str, Path],
        pk_path: Union[str, Path], check_mode: str = "unsafe", settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Union[str, Path]]:
        """Generate a proof for the given witness and model."""
        # --- Normalization & Validation ---
        witness_path, model_path, proof_path, pk_path = str(witness_path), str(model_path), str(proof_path), str(pk_path)
        if not os.path.exists(witness_path): raise FileNotFoundError(f"Witness file not found: {witness_path}")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(pk_path): raise FileNotFoundError(f"PK key file not found: {pk_path}")

        # --- SRS Setup ---
        if settings_path and os.path.exists(settings_path):
            logrows = get_logrows_from_settings(str(settings_path))
            if logrows and not ensure_srs(logrows): return False, f"Failed to ensure SRS for logrows={logrows}"

        # --- Execution ---
        os.makedirs(os.path.dirname(proof_path), exist_ok=True)
        try:
            cmd = [
                str(EZKL_PATH), "prove", "--check-mode", check_mode, "--witness", witness_path,
                "--compiled-circuit", model_path, "--proof-path", proof_path, "--pk-path", pk_path,
            ]
            EZKLUtils.run_ezkl_command_with_srs_check(cmd, env=self.env, check=True, capture_output=True, text=True)
            return True, proof_path
        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            return False, str(e)

    def verify(self, proof_path: Union[str, Path], settings_path: Union[str, Path], vk_path: Union[str, Path]) -> bool:
        """Verify a proof using EZKL."""
        # --- Normalization & Validation ---
        proof_path, settings_path, vk_path = str(proof_path), str(settings_path), str(vk_path)
        if not os.path.exists(proof_path): raise FileNotFoundError(f"Proof file not found: {proof_path}")
        if not os.path.exists(settings_path): raise FileNotFoundError(f"Settings file not found: {settings_path}")
        if not os.path.exists(vk_path): raise FileNotFoundError(f"Verification key file not found: {vk_path}")

        # --- SRS Setup ---
        logrows = get_logrows_from_settings(settings_path)
        if logrows and not ensure_srs(logrows): return False

        # --- Execution ---
        try:
            cmd = [
                str(EZKL_PATH), "verify", "--proof-path", proof_path,
                "--settings-path", settings_path, "--vk-path", vk_path,
            ]
            subprocess.run(cmd, env=self.env, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Proof verification failed: {e.stderr}")
            return False

    def gen_settings(
        self,
        model_path: str,
        settings_path: str,
        param_visibility: str = "fixed",
        input_visibility: str = "public",
    ):
        """
        Generate EZKL settings.
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        os.makedirs(os.path.dirname(settings_path) or ".", exist_ok=True)
        try:
            cmd = [
                str(EZKL_PATH),
                "gen-settings",
                "--param-visibility",
                param_visibility,
                "--input-visibility",
                input_visibility,
                "--model",
                model_path,
                "--settings-path",
                settings_path,
            ]
            process = subprocess.run(
                cmd,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                if process.stderr:
                    logger.error(process.stderr)
                return False, process.stderr or "gen-settings failed"
            return True, None
        except subprocess.CalledProcessError as e:
            if getattr(e, "stderr", None):
                logger.error(e.stderr)
            return False, getattr(e, "stderr", str(e))

    def calibrate_settings(
        self, model_path: str, settings_path: str, data_path: str, target: str = None
    ):
        """
        Calibrate EZKL settings using provided data.
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Calibration data file not found: {data_path}")
        cmd = [
            str(EZKL_PATH),
            "calibrate-settings",
            "--model",
            model_path,
            "--settings-path",
            settings_path,
            "--data",
            data_path,
        ]
        if target:
            cmd += ["--target", target]
        # Run without raising so we can capture and surface EZKL's own error text
        try:
            process = subprocess.run(
                cmd,
                env=self.env,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to invoke ezkl calibrate-settings: {e}")
            return False, str(e)

        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            stdout = (process.stdout or "").strip()
            combined = stderr if stderr else stdout
            if stderr and stdout:
                combined = f"{stderr}\n{stdout}"
            logging.getLogger(__name__).error(f"EZKL calibrate-settings failed:\n{combined}")
            return False, combined or "calibrate-settings failed"

        return True, None

    def compile_circuit(self, model_path: str, settings_path: str, compiled_path: str):
        """
        Compile EZKL circuit.
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        os.makedirs(os.path.dirname(compiled_path) or ".", exist_ok=True)
        try:
            cmd = [
                str(EZKL_PATH),
                "compile-circuit",
                "--model",
                model_path,
                "--settings-path",
                settings_path,
                "--compiled-circuit",
                compiled_path,
            ]
            process = subprocess.run(
                cmd,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                if process.stderr:
                    logger.error(process.stderr)
                return False, process.stderr or "compile-circuit failed"
            return True, None
        except subprocess.CalledProcessError as e:
            if getattr(e, "stderr", None):
                logger.error(e.stderr)
            return False, getattr(e, "stderr", str(e))

    def setup(self, compiled_path: str, vk_path: str, pk_path: str, settings_path: str = None):
        """
        Generate proving and verification keys (setup).
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(compiled_path):
            raise FileNotFoundError(f"Compiled circuit file not found: {compiled_path}")
        os.makedirs(os.path.dirname(vk_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(pk_path) or ".", exist_ok=True)

        if settings_path and os.path.exists(settings_path):
            logrows = get_logrows_from_settings(settings_path)
            if logrows:
                if not ensure_srs(logrows):
                    return False, f"Failed to ensure SRS for logrows={logrows}"

        try:
            cmd = [
                str(EZKL_PATH),
                "setup",
                "--compiled-circuit",
                compiled_path,
                "--vk-path",
                vk_path,
                "--pk-path",
                pk_path,
            ]
            process = EZKLUtils.run_ezkl_command_with_srs_check(
                cmd,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                if process.stderr:
                    logger.error(process.stderr)
                return False, process.stderr or "setup failed"
            return True, None
        except subprocess.CalledProcessError as e:
            if getattr(e, "stderr", None):
                logger.error(e.stderr)
            return False, getattr(e, "stderr", str(e))
        except RuntimeError as e:
            # SRS error detected and bubbled up
            return False, str(e)

    def compilation_pipeline(self, model_path: Union[str, Path], output_path: Union[str, Path], input_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the full EZKL compilation pipeline: settings -> calibration -> compile -> setup."""
        # --- Validation & Initialization ---
        model_path, output_path = Path(model_path), Path(output_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Artifact Preparation ---
        artifacts = EZKLUtils.initialize_compilation_artifacts(model_path, output_path, input_file_path)
        paths, compilation_data = artifacts["paths"], artifacts["data"]

        try:
            # --- Calibration Setup ---
            if input_file_path and os.path.exists(input_file_path):
                try:
                    shutil.copyfile(os.path.abspath(input_file_path), os.path.abspath(paths["calibration"]))
                except Exception as copy_err:
                    logger.warning(f"Could not write calibration.json: {copy_err}")

            # --- Settings Generation ---
            logger.info(f"Generating settings for {model_path.stem}")
            ok, err = self.gen_settings(model_path=str(model_path), settings_path=str(paths["settings"]))
            if not ok:
                logger.warning(f"Failed to generate settings: {err}")
                compilation_data["gen-settings_error"] = err
                return compilation_data

            # --- Settings Calibration ---
            if input_file_path and os.path.exists(input_file_path):
                logger.info(f"Calibrating settings using {input_file_path}")
                ok, err = self.calibrate_settings(
                    model_path=str(model_path), settings_path=str(paths["settings"]),
                    data_path=input_file_path, target="accuracy",
                )
                if not ok:
                    logger.warning(f"Failed to calibrate settings: {err}")
                    compilation_data["calibrate-settings_error"] = err
            else:
                logger.info("No input file provided, skipping calibration step")

            # --- Circuit Compilation ---
            logger.info(f"Compiling circuit for {model_path}")
            ok, err = self.compile_circuit(
                model_path=str(model_path), settings_path=str(paths["settings"]), compiled_path=str(paths["compiled"]),
            )
            if not ok:
                logger.warning(f"Failed to compile circuit: {err}")
                compilation_data["compile-circuit_error"] = err
                return compilation_data

            # --- Key Generation (Setup) ---
            logger.info("Setting up verification and proving keys")
            ok, err = self.setup(
                compiled_path=str(paths["compiled"]), vk_path=str(paths["vk"]),
                pk_path=str(paths["pk"]), settings_path=str(paths["settings"]),
            )
            if not ok:
                logger.warning(f"Failed to setup (generate keys): {err}")
                compilation_data["setup_error"] = err
                return compilation_data

            logger.info(f"Circuitization pipeline completed for {model_path}")
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error during circuitization: {str(e)}"
            logger.error(error_msg)
            compilation_data["error"] = error_msg

        return compilation_data


    @staticmethod
    def process_witness_output(witness_data: dict) -> Optional[dict]:
        """Process the witness.json data to get prediction results."""
        return EZKLUtils.process_witness_output(witness_data)


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/yolov3",
    }
    abs_path = os.path.abspath(base_paths[model_choice])
    model_dir = abs_path
    slices_dir = os.path.join(abs_path, "slices")

    # Circuitize
    model_path = os.path.abspath(model_dir)
    EZKL().compile(model_path=abs_path)

    # # Generate witness
    # input_file = os.path.join(model_dir, "input.json")
    # model_path = os.path.join(model_dir, "model.compiled")
    # vk_path = os.path.join(model_dir, "vk.json")
    # output_file = os.path.join(model_dir, "witness.json")
    # result = ezkl.generate_witness(input_file=input_file, model_path=model_path, output_file=output_file, vk_path=vk_path)
    # print(result)
