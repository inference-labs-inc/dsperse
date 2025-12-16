"""
JSTprove backend for zero-knowledge proof generation.
This module provides a backend for generating ZK proofs using the JSTprove CLI.
"""
import json
import os
import subprocess
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

from dsperse.src.constants import JSTPROVE_COMMAND

# Configure logger
logger = logging.getLogger(__name__)


class JSTprove:
    """JSTprove backend for zero-knowledge proof generation using the JSTprove CLI."""
    
    # Class constants
    COMMAND = JSTPROVE_COMMAND
    DEFAULT_FLAGS = ["--no-banner"]

    def __init__(self, model_directory: Optional[str] = None) -> None:
        """
        Initialize the JSTprove backend.

        Args:
            model_directory: Optional path to the model directory for organizing artifacts.

        Raises:
            RuntimeError: If JSTprove CLI is not available
        """
        self.env = os.environ.copy()
        self.model_directory = Path(model_directory) if model_directory else None
        self._witness_format = "jstprove"  # Track witness output format

        # Check if JSTprove CLI is available
        try:
            result = subprocess.run(
                [self.COMMAND, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("JSTprove CLI not found. Please install JSTprove first.")
        except FileNotFoundError:
            raise RuntimeError("JSTprove CLI not found. Please install JSTprove: uv tool install jstprove")

    def _run_command(
        self,
        subcommand: str,
        args: List[str],
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Execute a JSTprove CLI command.

        Args:
            subcommand: The jst subcommand (compile, witness, prove, verify)
            args: Additional arguments for the subcommand
            check: Whether to check return code
            capture_output: Whether to capture output

        Returns:
            subprocess.CompletedProcess: The completed process

        Raises:
            RuntimeError: If command fails
        """
        cmd = [self.COMMAND] + self.DEFAULT_FLAGS + [subcommand] + args
        try:
            logger.debug(f"Running JSTprove command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                env=self.env,
                check=check,
                capture_output=capture_output,
                text=True,
            )
            return process
        except subprocess.CalledProcessError as e:
            error_msg = f"JSTprove command failed: {' '.join(cmd)}"
            if e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    #
    # High-level methods that dispatch to specific implementations
    #

    def generate_witness(
        self,
        input_file: Union[str, Path],
        model_path: Union[str, Path],  # This is the circuit path in JSTprove context
        output_file: Union[str, Path],
        vk_path: Optional[Union[str, Path]] = None,  # Kept for backward compatibility but not used
        settings_path: Optional[Union[str, Path]] = None  # Kept for backward compatibility but not used
    ) -> Tuple[bool, Any]:
        """
        Generate a witness for the given circuit and input using JSTprove.

        Args:
            input_file: Path to the input JSON file
            model_path: Path to the compiled circuit file (called model_path for interface compatibility)
            output_file: Path where to save the model outputs JSON
            vk_path: Ignored (kept for backward compatibility)
            settings_path: Ignored (kept for backward compatibility)

        Returns:
            Tuple of (success: bool, output: Any) where output is the processed witness data
        """
        # Normalize paths
        input_file = Path(input_file)
        circuit_path = Path(model_path)  # model_path is actually the circuit path
        output_file = Path(output_file)
        witness_path = output_file.parent / f"{output_file.stem}_witness.bin"

        # Validate required files exist
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Check if we have an ONNX model that needs compilation, or an existing circuit
        onnx_model_path = None
        if circuit_path.exists() and circuit_path.suffix == '.onnx':
            onnx_model_path = circuit_path
            circuit_path = circuit_path.parent / f"{circuit_path.stem}_jstprove_circuit.txt"

        # If we have an ONNX model, compile it first only if circuit doesn't exist
        if onnx_model_path and not circuit_path.exists():
            logger.info(f"JSTprove: Compiling ONNX model {onnx_model_path} to circuit {circuit_path}")
            ok, err = self.compile_circuit(onnx_model_path, circuit_path)
            if not ok:
                raise RuntimeError(f"Circuit compilation failed: {err}")
        elif onnx_model_path and circuit_path.exists():
            logger.info(f"Using existing circuit: {circuit_path}")
        elif not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

        # Create output directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        witness_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_command("witness", [
                "-c", str(circuit_path),
                "-i", str(input_file),
                "-o", str(output_file),
                "-w", str(witness_path),
            ])
        except RuntimeError as e:
            error_msg = f"Witness generation failed: {e}"
            logger.error(error_msg)
            return False, error_msg

        # Process the outputs
        try:
            with open(output_file, "r") as f:
                output_data = json.load(f)
                processed_output = self.process_witness_output(output_data)
            return True, processed_output
        except (json.JSONDecodeError, FileNotFoundError) as e:
            error_msg = f"Failed to process witness output: {e}"
            logger.error(error_msg)
            return False, error_msg

    def prove(
        self,
        witness_path: Union[str, Path],
        circuit_path: Union[str, Path],
        proof_path: Union[str, Path],
        pk_path: Optional[Union[str, Path]] = None,  # Kept for backward compatibility but not used
        check_mode: str = "unsafe",  # Kept for backward compatibility but not used
        settings_path: Optional[Union[str, Path]] = None  # Kept for backward compatibility but not used
    ) -> Tuple[bool, Union[str, Path]]:
        """
        Generate a proof for the given witness and circuit using JSTprove.

        Args:
            witness_path: Path to the witness file
            circuit_path: Path to the compiled circuit
            proof_path: Path where to save the proof
            pk_path: Ignored (kept for backward compatibility)
            check_mode: Ignored (kept for backward compatibility)
            settings_path: Ignored (kept for backward compatibility)

        Returns:
            Tuple of (success: bool, results: Union[str, Path]) where results is the proof path
        """
        # Normalize paths
        witness_path = Path(witness_path)
        circuit_path = Path(circuit_path)
        proof_path = Path(proof_path)

        # Validate required files exist
        if not witness_path.exists():
            raise FileNotFoundError(f"Witness file not found: {witness_path}")
        if not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

        # Create output directory if it doesn't exist
        proof_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_command("prove", [
                "-c", str(circuit_path),
                "-w", str(witness_path),
                "-p", str(proof_path),
            ])
        except RuntimeError as e:
            error_msg = f"Proof generation failed: {e}"
            logger.error(error_msg)
            return False, error_msg

        return True, proof_path

    def verify(
        self,
        proof_path: Union[str, Path],
        circuit_path: Union[str, Path],
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        witness_path: Union[str, Path],
        settings_path: Optional[Union[str, Path]] = None,  # Kept for backward compatibility but not used
        vk_path: Optional[Union[str, Path]] = None  # Kept for backward compatibility but not used
    ) -> bool:
        """
        Verify a proof using JSTprove.

        Args:
            proof_path: Path to the proof file
            circuit_path: Path to the compiled circuit
            input_path: Path to the input JSON used for the proof
            output_path: Path to the expected outputs JSON
            witness_path: Path to the witness file
            settings_path: Ignored (kept for backward compatibility)
            vk_path: Ignored (kept for backward compatibility)

        Returns:
            True if verification succeeded, False otherwise
        """
        # Normalize paths
        proof_path = Path(proof_path)
        circuit_path = Path(circuit_path)
        input_path = Path(input_path)
        output_path = Path(output_path)
        witness_path = Path(witness_path)

        # Validate required files exist
        required_files = [proof_path, circuit_path, input_path, output_path, witness_path]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        try:
            self._run_command("verify", [
                "-c", str(circuit_path),
                "-i", str(input_path),
                "-o", str(output_path),
                "-w", str(witness_path),
                "-p", str(proof_path),
            ])
            return True
        except RuntimeError as e:
            logger.error(f"Proof verification failed: {e}")
            return False

    def compile_circuit(
        self,
        model_path: Union[str, Path],
        circuit_path: Union[str, Path],
        settings_path: Optional[Union[str, Path]] = None  # Kept for backward compatibility but not used
    ) -> Tuple[bool, Optional[str]]:
        """
        Compile a circuit from an ONNX model using JSTprove.

        Args:
            model_path: Path to the original ONNX model
            circuit_path: Path where to save the compiled circuit
            settings_path: Ignored (kept for backward compatibility)

        Returns:
            Tuple of (success: bool, error: Optional[str])
        """
        # Normalize paths
        model_path = Path(model_path)
        circuit_path = Path(circuit_path)

        # Validate required files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create output directory if it doesn't exist
        circuit_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_command("compile", [
                "-m", str(model_path),
                "-c", str(circuit_path),
            ])
            return True, None
        except Exception as e:
            error_msg = f"Circuit compilation failed: {e}"
            logger.error(error_msg)
            return False, error_msg

    def circuitization_pipeline(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        input_file_path: Optional[Union[str, Path]] = None,
        segment_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the JSTprove circuitization pipeline.

        In JSTprove, circuitization is a single step that compiles the model into a circuit.
        The compile command handles all the necessary setup internally.

        Args:
            model_path: Path to the ONNX model file.
            output_path: Base path for output files.
            input_file_path: Ignored (kept for backward compatibility).
            segment_details: Ignored (kept for backward compatibility).

        Returns:
            Dictionary containing paths to generated files and any error information.
        """
        # Normalize paths
        model_path = Path(model_path)
        output_path = Path(output_path)

        # Ensure model_path exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        model_name = model_path.stem

        # Define file paths (JSTprove outputs circuit and quantized model)
        circuit_path = output_path / f"{model_name}_circuit.txt"
        quantized_model_path = output_path / f"{model_name}_circuit_quantized_model.onnx"
        witness_solver_path = output_path / f"{model_name}_circuit_witness_solver.txt"

        # Create dummy settings file for compatibility with runner analyzer
        settings_path = output_path / f"{model_name}_settings.json"

        # Initialize circuitization data dictionary (match EZKL structure for compatibility)
        circuitization_data: Dict[str, Any] = {
            "compiled": str(circuit_path),  # This is what runner_analyzer looks for
            "circuit": str(circuit_path),
            "quantized_model": str(quantized_model_path),
            "witness_solver": str(witness_solver_path),
            "calibration": input_file_path,
            # Create dummy settings file for runner analyzer compatibility
            "settings": str(settings_path),
            # JSTprove doesn't use vk, pk in the same way as EZKL
            "vk_key": None,
            "pk_key": None,
        }

        try:
            logger.info(f"Compiling circuit for {model_name}")
            # JSTprove compile command handles everything in one step
            ok, err = self.compile_circuit(
                model_path=model_path,
                circuit_path=circuit_path,
            )
            if not ok:
                logger.warning("Failed to compile circuit")
                circuitization_data["compile_error"] = err
            else:
                # Create dummy settings file for runner analyzer compatibility
                dummy_settings = {
                    "backend": "jstprove",
                    "model_path": str(model_path),
                    "circuit_path": str(circuit_path),
                    "compiled_at": str(output_path),
                    "note": "This is a dummy settings file for dsperse compatibility. JSTprove handles settings internally."
                }
                with open(settings_path, 'w') as f:
                    json.dump(dummy_settings, f, indent=2)
                logger.info(f"Circuitization pipeline completed for {model_path}")
        except Exception as e:
            error_msg = f"Error during circuitization: {str(e)}"
            logger.exception(error_msg)
            circuitization_data["error"] = error_msg

        return circuitization_data

    # Alias for backward compatibility with EZKL interface
    compilation_pipeline = circuitization_pipeline

    def process_witness_output(self, witness_data: Any) -> Optional[Dict[str, Any]]:
        """
        Process the witness output data to get prediction results.

        This method handles JSTprove witness output format. JSTprove outputs
        a raw array of floats representing the final logits.

        Args:
            witness_data: The parsed JSON data from witness output.

        Returns:
            Dictionary containing processed predictions, or None if processing fails.
        """
        def _to_logits(data) -> torch.Tensor:
            """Helper to convert data to logits tensor with batch dimension."""
            logits = torch.tensor(data)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            return logits

        try:
            # JSTprove dict format with 'rescaled_output' key
            if isinstance(witness_data, dict) and "rescaled_output" in witness_data:
                self._witness_format = "jstprove_dict"
                # NOTE: Rescaled outputs are in output.json (from -o flag), not in the witness binary file (-w flag).
                # The witness binary contains only the raw quantized values needed for proof generation.
                logger.warning(
                    "Using rescaled outputs from output.json (not witness binary). "
                    "These are the model's floating-point outputs after de-quantization."
                )
                return {"logits": _to_logits(witness_data["rescaled_output"])}
            # Raw array format
            elif isinstance(witness_data, list):
                self._witness_format = "jstprove_list"
                return {"logits": _to_logits(witness_data)}
            # EZKL-like format fallback
            else:
                self._witness_format = "ezkl_compat"
                rescaled = witness_data["pretty_elements"]["rescaled_outputs"][0]
                return {"logits": _to_logits(rescaled)}
        except (KeyError, TypeError) as e:
            logger.error(f"Could not process witness data: {e}")
            return None

    @classmethod
    def get_version(cls) -> Optional[str]:
        """
        Get the JSTprove version.

        Returns:
            str: JSTprove version string, or None if version cannot be determined
        """
        try:
            result = subprocess.run(
                [cls.COMMAND, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse version from output
                version_output = result.stdout.strip() or result.stderr.strip()
                return version_output
        except Exception as e:
            logger.debug(f"Could not get JSTprove version: {e}")
        return None


if __name__ == "__main__":
    # Example usage with JSTprove
    print("JSTprove backend example:")
    print("backend = JSTprove()")
    print("backend.compile_circuit('model.onnx', 'circuit.txt')")
    print("backend.generate_witness('input.json', 'circuit.txt', 'output.json')")
    print("backend.prove('witness.bin', 'circuit.txt', 'proof.bin')")
    print("backend.verify('proof.bin', 'circuit.txt', 'input.json', 'output.json', 'witness.bin')")

