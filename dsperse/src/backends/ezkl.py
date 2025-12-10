import json
import os
import subprocess
import torch
import logging
import traceback
from pathlib import Path
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.constants import EZKL_PATH
from dsperse.src.utils.srs_manager import ensure_srs, get_logrows_from_settings

# Configure logger
logger = logging.getLogger(__name__)


def _detect_srs_error(stderr_output):
    """
    Detect SRS-related errors in EZKL stderr output.

    Args:
        stderr_output (str): The stderr output from EZKL command

    Returns:
        str: SRS error message if detected, None otherwise
    """
    if not stderr_output:
        return None

    stderr_lower = stderr_output.lower()

    # Common SRS error patterns
    srs_error_patterns = [
        "srs",  # General SRS reference
        "structured reference string",  # Full name
        "trusted setup",  # Alternative name
        "ceremony",  # Related to trusted setup ceremonies
        "powers of tau",  # Technical SRS component
        "no srs file",  # Specific error message
        "srs not found",  # Specific error message
        "missing srs",  # Specific error message
        "srs table",  # SRS table reference
        "srs path",  # SRS path reference
    ]

    for pattern in srs_error_patterns:
        if pattern in stderr_lower:
            return f"SRS Error Detected: {stderr_output.strip()}"

    return None


def _run_ezkl_command_with_srs_check(
    cmd_list, env=None, check=True, capture_output=True, text=True, **kwargs
):
    """
    Wrapper for subprocess.run that detects SRS errors and bubbles them up.

    Args:
        cmd_list (list): Command list to run
        env (dict): Environment variables
        check (bool): Whether to check return code
        capture_output (bool): Whether to capture output
        text (bool): Whether to return text
        **kwargs: Additional subprocess.run arguments

    Returns:
        subprocess.CompletedProcess: The completed process

    Raises:
        RuntimeError: If SRS error is detected
    """
    try:
        process = subprocess.run(
            cmd_list,
            env=env,
            check=check,
            capture_output=capture_output,
            text=text,
            **kwargs,
        )

        # Check for SRS errors even if command succeeded
        if process.stderr:
            srs_error = _detect_srs_error(process.stderr)
            if srs_error:
                logger.error(
                    f"EZKL SRS Error in command '{' '.join(cmd_list)}': {srs_error}"
                )
                raise RuntimeError(f"DSperse detected SRS error: {srs_error}")

        return process

    except subprocess.CalledProcessError as e:
        # Check for SRS errors in stderr
        if e.stderr:
            srs_error = _detect_srs_error(e.stderr)
            if srs_error:
                logger.error(
                    f"EZKL SRS Error in command '{' '.join(cmd_list)}': {srs_error}"
                )
                raise RuntimeError(f"DSperse detected SRS error: {srs_error}")

        # Re-raise the original exception if not SRS-related
        raise


class EZKL:
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
        self, input_file: str, model_path: str, output_file: str, vk_path: str, settings_path: str = None
    ):
        """
        Generate a witness for the given model and input.

        Args:
            input_file (str): Path to the input file
            model_path (str): Path to the compiled model
            output_file (str): Path where to save the output
            vk_path (str): Path to the verification key

        Returns:
            tuple: (success, output) where success is a boolean and output is the processed witness output
        """
        # Normalize possible Path-like arguments to strings for subprocess and logging clarity
        input_file = str(input_file)
        model_path = str(model_path)
        output_file = str(output_file)
        vk_path = str(vk_path)

        # Validate required files exist
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vk_path):
            raise FileNotFoundError(f"Verification key file not found: {vk_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if settings_path and os.path.exists(settings_path):
            logrows = get_logrows_from_settings(settings_path)
            if logrows:
                if not ensure_srs(logrows):
                    return False, f"Failed to ensure SRS for logrows={logrows}"

        try:
            cmd = [
                str(EZKL_PATH),
                "gen-witness",
                "--data",
                input_file,
                "--compiled-circuit",
                model_path,
                "--output",
                output_file,
                "--vk-path",
                vk_path,
            ]
            process = subprocess.run(
                cmd, env=self.env, check=True, capture_output=True, text=True
            )

            if process.returncode != 0:
                # Print the full Python stack and the process stderr
                traceback.print_stack()
                if process.stderr:
                    print(process.stderr)
                error_msg = (
                    f"Witness generation failed with return code {process.returncode}"
                )
                if process.stderr:
                    error_msg += f"\nError: {process.stderr}"
                return False, error_msg

        except subprocess.CalledProcessError as e:
            # Print the full stack trace from the exception and the process stderr
            traceback.print_exc()
            if getattr(e, "stderr", None):
                print(e.stderr)
            error_msg = f"Witness generation failed: {e}"
            if e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            return False, error_msg

        # return the processed outputs
        with open(output_file, "r") as f:
            witness_data = json.load(f)
            output = self.process_witness_output(witness_data)

        return True, output

    def prove(
        self,
        witness_path: str,
        model_path: str,
        proof_path: str,
        pk_path: str,
        check_mode: str = "unsafe",
        settings_path: str = None
    ):
        """
        Generate a proof for the given witness and model.

        Args:
            witness_path (str): Path to the witness file
            model_path (str): Path to the compiled model
            proof_path (str): Path where to save the proof
            pk_path (str): Path to the proving key
            check_mode (str, optional): Check mode for the prover. Defaults to "unsafe".

        Returns:
            tuple: (success, results) where success is a boolean and results is the path to the proof
        """
        # Normalize path-like args
        witness_path = str(witness_path)
        model_path = str(model_path)
        proof_path = str(proof_path)
        pk_path = str(pk_path)

        # Validate required files exist
        if not os.path.exists(witness_path):
            raise FileNotFoundError(f"Witness file not found: {witness_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(pk_path):
            raise FileNotFoundError(f"PK key file not found: {pk_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(proof_path), exist_ok=True)

        if settings_path and os.path.exists(settings_path):
            logrows = get_logrows_from_settings(settings_path)
            if logrows:
                if not ensure_srs(logrows):
                    return False, f"Failed to ensure SRS for logrows={logrows}"

        try:
            cmd = [
                str(EZKL_PATH),
                "prove",
                "--check-mode",
                check_mode,
                "--witness",
                witness_path,
                "--compiled-circuit",
                model_path,
                "--proof-path",
                proof_path,
                "--pk-path",
                pk_path,
            ]
            process = _run_ezkl_command_with_srs_check(
                cmd, env=self.env, check=True, capture_output=True, text=True
            )

            if process.returncode != 0:
                # Print the full Python stack and the process stderr
                traceback.print_stack()
                if process.stderr:
                    print(process.stderr)
                error_msg = (
                    f"Proof generation failed with return code {process.returncode}"
                )
                if process.stderr:
                    error_msg += f"\nError: {process.stderr}"
                return False, error_msg

        except subprocess.CalledProcessError as e:
            print(f"Error during proof generation: {e}")
            traceback.print_exc()
            if getattr(e, "stderr", None):
                print(e.stderr)
            return False, e.stderr
        except RuntimeError as e:
            # SRS error detected and bubbled up
            traceback.print_exc()
            return False, str(e)

        results = proof_path
        return True, results

    def verify(self, proof_path: str, settings_path: str, vk_path: str) -> bool:
        """
        Verify a proof.

        Args:
            proof_path (str): Path to the proof file
            settings_path (str): Path to the settings file
            vk_path (str): Path to the verification key

        Returns:
            bool: True if verification succeeded, False otherwise
        """
        # Normalize path-like args
        proof_path = str(proof_path)
        settings_path = str(settings_path)
        vk_path = str(vk_path)

        # Validate required files exist
        if not os.path.exists(proof_path):
            raise FileNotFoundError(f"Proof file not found: {proof_path}")
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        if not os.path.exists(vk_path):
            raise FileNotFoundError(f"Verification key file not found: {vk_path}")

        logrows = get_logrows_from_settings(settings_path)
        if logrows:
            if not ensure_srs(logrows):
                return False

        try:
            cmd = [
                str(EZKL_PATH),
                "verify",
                "--proof-path",
                proof_path,
                "--settings-path",
                settings_path,
                "--vk-path",
                vk_path,
            ]
            process = subprocess.run(
                cmd, env=self.env, check=True, capture_output=True, text=True
            )

            if process.returncode != 0:
                # Print the full Python stack and the process stderr
                traceback.print_stack()
                if process.stderr:
                    print(process.stderr)
                error_msg = f"Verification generation failed with return code {process.returncode}"
                if process.stderr:
                    error_msg += f"\nError: {process.stderr}"
                return False
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error verifying proof: {e}")
            traceback.print_exc()
            if getattr(e, "stderr", None):
                print(e.stderr)
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
                # Print the error stack from the process itself
                if process.stderr:
                    print(process.stderr)
                return False, process.stderr or "gen-settings failed"
            return True, None
        except subprocess.CalledProcessError as e:
            # Print the error stack from the exception
            if getattr(e, "stderr", None):
                print(e.stderr)
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
        try:
            process = subprocess.run(
                cmd,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                # Print the full Python stack and the process stderr
                traceback.print_stack()
                if process.stderr:
                    print(process.stderr)
                return False, process.stderr or "calibrate-settings failed"
            return True, None
        except subprocess.CalledProcessError as e:
            # Print the full stack trace from the exception and the process stderr
            traceback.print_exc()
            if getattr(e, "stderr", None):
                print(e.stderr)
            return False, getattr(e, "stderr", str(e))

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
                # Print the error stack from the process itself
                if process.stderr:
                    print(process.stderr)
                return False, process.stderr or "compile-circuit failed"
            return True, None
        except subprocess.CalledProcessError as e:
            # Print the error stack from the exception
            if getattr(e, "stderr", None):
                print(e.stderr)
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
            process = _run_ezkl_command_with_srs_check(
                cmd,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                # Print the error stack from the process itself
                if process.stderr:
                    print(process.stderr)
                return False, process.stderr or "setup failed"
            return True, None
        except subprocess.CalledProcessError as e:
            # Print the error stack from the exception
            if getattr(e, "stderr", None):
                print(e.stderr)
            return False, getattr(e, "stderr", str(e))
        except RuntimeError as e:
            # SRS error detected and bubbled up
            return False, str(e)

    def compilation_pipeline(self, model_path, output_path, input_file_path=None):
        """
        Run the full EZKL circuitization pipeline: gen-settings, calibrate-settings, compile-circuit, setup.

        Args:
            model_path (str): Path to the ONNX model file.
            output_path (str): Base path for output files (without extension).
            input_file_path (str, optional): Path to input data file for calibration.
            slice_details (dict, optional): Details about the segment being processed.

        Returns:
            dict: Dictionary containing paths to generated files and any error information.
        """
        # Ensure model_path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        model_name = Path(model_path).stem

        # Define file paths
        settings_path = os.path.join(output_path, f"settings.json")
        compiled_path = os.path.join(output_path, f"model.compiled")
        vk_path = os.path.join(output_path, f"vk.key")
        pk_path = os.path.join(output_path, f"pk.key")

        # Initialize circuitization data dictionary
        compilation_data = {
            "settings": settings_path,
            "compiled": compiled_path,
            "vk_key": vk_path,
            "pk_key": pk_path,
            "calibration": input_file_path,
        }

        try:
            # Step 1: Generate settings
            logger.info(f"Generating settings for {model_name}")
            ok, err = self.gen_settings(
                model_path=model_path, settings_path=settings_path
            )
            if not ok:
                logger.warning("Failed to generate settings")
                compilation_data["gen-settings_error"] = err

            # Step 2/3: Calibrate settings
            if input_file_path and os.path.exists(input_file_path):
                logger.info(f"Calibrating settings using {input_file_path}")
                ok, err = self.calibrate_settings(
                    model_path=model_path,
                    settings_path=settings_path,
                    data_path=input_file_path,
                    target="accuracy",
                )
                compilation_data["calibration"] = input_file_path
                if not ok:
                    logger.warning("Failed to calibrate settings")
                    compilation_data["calibrate-settings_error"] = err
            else:
                # If no input file, log and skip calibration
                logger.info("No input file provided, skipping calibration step")

            # Step 4: Compile circuit
            logger.info(f"Compiling circuit for {model_path}")
            ok, err = self.compile_circuit(
                model_path=model_path,
                settings_path=settings_path,
                compiled_path=compiled_path,
            )
            if not ok:
                logger.warning("Failed to compile circuit")
                compilation_data["compile-circuit_error"] = err

            # Step 5: Setup (generate verification and proving keys)
            logger.info("Setting up verification and proving keys")
            ok, err = self.setup(
                compiled_path=compiled_path, vk_path=vk_path, pk_path=pk_path, settings_path=settings_path
            )
            if not ok:
                logger.warning("Failed to setup (generate keys)")
                compilation_data["setup_error"] = err

            logger.info(f"Circuitization pipeline completed for {model_path}")

        except Exception as e:
            # Print the full stack trace for any unexpected pipeline error
            traceback.print_exc()
            error_msg = f"Error during circuitization: {str(e)}"
            logger.error(error_msg)
            compilation_data["error"] = error_msg

        return compilation_data


    @staticmethod
    def process_witness_output(witness_data):
        """
        Process the witness.json data to get prediction results.
        """
        try:
            rescaled_outputs = witness_data["pretty_elements"]["rescaled_outputs"][0]
        except KeyError:
            print("Error: Could not find rescaled_outputs in witness data")
            return None

        # Convert string values to float and create a tensor
        float_values = [float(val) for val in rescaled_outputs]

        # Create a tensor with shape [1, num_classes] to match batch_size, num_classes format
        tensor_output = torch.tensor([float_values], dtype=torch.float32)

        # Process the tensor through _process_final_output (simulating one segment)
        output = RunnerUtils.process_final_output(tensor_output)
        return output


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
