import json
import os
import subprocess
import torch
import logging
import traceback
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from argparse import Namespace
import onnx
from dsperse.src.utils.utils import Utils
from dsperse.src.utils.runner_utils.runner_utils import RunnerUtils
from dsperse.src.backends.base_backend import BaseBackend

# Configure logger
logger = logging.getLogger(__name__)

# Try to import JSTprove functions directly
try:
    # Import JSTprove CLI functions directly to avoid external CLI dependency
    import sys
    # Get the absolute path to JSTprove directory (parent of python package)
    # Use a reliable method: find the project root by going up until we find JSTprove
    current_file = Path(__file__)
    # Start from the file and go up until we find a directory containing JSTprove
    candidate_root = current_file.parent
    while candidate_root.parent != candidate_root:  # Not at filesystem root
        jstprove_candidate = candidate_root / "JSTprove"
        if jstprove_candidate.exists() and (jstprove_candidate / "python").exists():
            jstprove_root = str(jstprove_candidate)
            break
        candidate_root = candidate_root.parent
    else:
        # Fallback: assume we're in dsperse and JSTprove is at ../JSTprove
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
        jstprove_root = os.path.join(project_root, "JSTprove")

    if os.path.exists(jstprove_root) and jstprove_root not in sys.path:
        sys.path.insert(0, jstprove_root)

    from python.frontend.cli import _run_compile, _run_witness, _run_prove, _run_verify
    JSTPROVE_AVAILABLE = True
    logger.info("JSTprove library imported successfully")
except ImportError as e:
    logger.warning(f"Could not import JSTprove library directly: {e}. Will try external CLI.")
    JSTPROVE_AVAILABLE = False

# Fallback to external CLI command
JST_COMMAND = "jst"


def _run_jst_command(
    cmd_list: list[str],
    env: Optional[dict] = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Wrapper for subprocess.run that executes JSTprove CLI commands.

    Args:
        cmd_list: Command list to run
        env: Environment variables
        check: Whether to check return code
        capture_output: Whether to capture output
        text: Whether to return text
        **kwargs: Additional subprocess.run arguments

    Returns:
        subprocess.CompletedProcess: The completed process

    Raises:
        RuntimeError: If command fails
    """
    try:
        logger.debug(f"Running JSTprove command: {' '.join(cmd_list)}")
        process = subprocess.run(
            cmd_list,
            env=env,
            check=check,
            capture_output=capture_output,
            text=text,
            **kwargs,
        )
        return process

    except subprocess.CalledProcessError as e:
        error_msg = f"JSTprove command failed: {' '.join(cmd_list)}"
        if e.stderr:
            error_msg += f"\nError output: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


class JSTprove(BaseBackend):
    """JSTprove backend for zero-knowledge proof generation using the JSTprove CLI."""

    def __init__(self, model_directory: Optional[str] = None) -> None:
        """
        Initialize the JSTprove backend.

        Args:
            model_directory: Optional path to the model directory for organizing artifacts.

        Raises:
            RuntimeError: If neither direct import nor CLI is available
        """
        self.env = os.environ.copy()
        self.model_directory = Path(model_directory) if model_directory else None

        # Check if we can use JSTprove (either direct import or CLI)
        if not JSTPROVE_AVAILABLE:
            # Try external CLI as fallback
            try:
                result = subprocess.run(
                    [JST_COMMAND, "--help"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError("JSTprove CLI not found. Please install JSTprove first.")
            except FileNotFoundError:
                raise RuntimeError("JSTprove is not available. Please ensure JSTprove is properly installed.")
        else:
            logger.info("Using JSTprove direct library integration")

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
        witness_path = output_file.parent / f"{output_file.stem}_witness.bin"  # Generate witness path

        # JSTprove expects to be run from its own directory
        jstprove_dir = Path(__file__).parent.parent.parent.parent.parent / "JSTprove"
        original_cwd = os.getcwd()

        # Validate required files exist
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Check if we have an ONNX model that needs compilation, or an existing circuit
        onnx_model_path = None
        if circuit_path.exists() and circuit_path.suffix == '.onnx':
            # This is actually an ONNX model that needs compilation
            onnx_model_path = circuit_path
            circuit_path = circuit_path.parent / f"{circuit_path.stem}_jstprove_circuit.txt"
            logger.info(f"JSTprove: Detected ONNX model {onnx_model_path}, will compile to {circuit_path}")
        elif not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

        # If we have an ONNX model, compile it first
        if onnx_model_path:
            logger.info(f"JSTprove: Compiling ONNX model {onnx_model_path} to circuit {circuit_path}")
            try:
                os.chdir(str(jstprove_dir))
                compile_args = Namespace(
                    model_path=str(onnx_model_path),
                    circuit_path=str(circuit_path),
                    cmd="compile"
                )
                _run_compile(compile_args)
                logger.info("JSTprove: Circuit compilation completed successfully")
            except Exception as e:
                logger.error(f"JSTprove: Circuit compilation failed: {e}")
                raise
            finally:
                os.chdir(original_cwd)

        # Create output directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        witness_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if JSTPROVE_AVAILABLE:
                # Use direct library call
                logger.debug(f"Generating witness using JSTprove library: {circuit_path} + {input_file}")

                try:
                    os.chdir(str(jstprove_dir))
                    logger.info(f"JSTprove: Changed to directory {jstprove_dir}")
                    args = Namespace(
                        circuit_path=str(circuit_path),
                        input_path=str(input_file),
                        output_path=str(output_file),
                        witness_path=str(witness_path),
                        cmd="witness"
                    )
                    logger.info(f"JSTprove: Calling _run_witness with circuit={circuit_path}, input={input_file}")
                    _run_witness(args)
                    logger.info("JSTprove: Witness generation completed successfully")
                except Exception as e:
                    logger.error(f"JSTprove: Witness generation failed: {e}")
                    logger.error(f"JSTprove: Circuit path: {circuit_path}")
                    logger.error(f"JSTprove: Input file: {input_file}")
                    logger.error(f"JSTprove: Output file: {output_file}")
                    logger.error(f"JSTprove: Witness file: {witness_path}")
                    raise
                finally:
                    os.chdir(original_cwd)
            else:
                # Use external CLI as fallback
                cmd = [
                    JST_COMMAND,
                    "--no-banner",
                    "witness",
                    "-c", str(circuit_path),
                    "-i", str(input_file),
                    "-o", str(output_file),
                    "-w", str(witness_path),
                ]
                _run_jst_command(cmd, env=self.env)

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
            if JSTPROVE_AVAILABLE:
                # Use direct library call
                logger.debug(f"Generating proof using JSTprove library: {circuit_path} + {witness_path}")

                # JSTprove expects to be run from its own directory
                jstprove_dir = Path(__file__).parent.parent.parent.parent.parent / "JSTprove"
                original_cwd = os.getcwd()

                try:
                    os.chdir(str(jstprove_dir))
                    args = Namespace(
                        circuit_path=str(circuit_path),
                        witness_path=str(witness_path),
                        proof_path=str(proof_path),
                        cmd="prove"
                    )
                    _run_prove(args)
                finally:
                    os.chdir(original_cwd)
            else:
                # Use external CLI as fallback
                cmd = [
                    JST_COMMAND,
                    "--no-banner",
                    "prove",
                    "-c", str(circuit_path),
                    "-w", str(witness_path),
                    "-p", str(proof_path),
                ]
                _run_jst_command(cmd, env=self.env)

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
            if JSTPROVE_AVAILABLE:
                # Use direct library call
                logger.debug(f"Verifying proof using JSTprove library: {proof_path}")

                # JSTprove expects to be run from its own directory
                jstprove_dir = Path(__file__).parent.parent.parent.parent.parent / "JSTprove"
                original_cwd = os.getcwd()

                try:
                    os.chdir(str(jstprove_dir))
                    args = Namespace(
                        circuit_path=str(circuit_path),
                        input_path=str(input_path),
                        output_path=str(output_path),
                        witness_path=str(witness_path),
                        proof_path=str(proof_path),
                        cmd="verify"
                    )
                    _run_verify(args)
                finally:
                    os.chdir(original_cwd)
            else:
                # Use external CLI as fallback
                cmd = [
                    JST_COMMAND,
                    "--no-banner",
                    "verify",
                    "-c", str(circuit_path),
                    "-i", str(input_path),
                    "-o", str(output_path),
                    "-w", str(witness_path),
                    "-p", str(proof_path),
                ]
                _run_jst_command(cmd, env=self.env)
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
            if JSTPROVE_AVAILABLE:
                # Use direct library call
                logger.debug(f"Compiling circuit using JSTprove library: {model_path} -> {circuit_path}")

                # JSTprove expects to be run from its own directory
                jstprove_dir = Path(__file__).parent.parent.parent.parent.parent / "JSTprove"
                original_cwd = os.getcwd()

                try:
                    os.chdir(str(jstprove_dir))
                    args = Namespace(
                        model_path=str(model_path),
                        circuit_path=str(circuit_path),
                        cmd="compile"
                    )
                    _run_compile(args)
                finally:
                    os.chdir(original_cwd)
            else:
                # Use external CLI as fallback
                cmd = [
                    JST_COMMAND,
                    "--no-banner",
                    "compile",
                    "-m", str(model_path),
                    "-c", str(circuit_path),
                ]
                _run_jst_command(cmd, env=self.env)
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
            # Print the full stack trace for any unexpected pipeline error
            traceback.print_exc()
            error_msg = f"Error during circuitization: {str(e)}"
            logger.error(error_msg)
            circuitization_data["error"] = error_msg

        return circuitization_data

    def compilation_pipeline(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        input_file_path: Optional[Union[str, Path]] = None,
        segment_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Alias for circuitization_pipeline.

        Args:
            model_path: Path to the ONNX model file.
            output_path: Base path for output files.
            input_file_path: Ignored (kept for backward compatibility).
            segment_details: Ignored (kept for backward compatibility).

        Returns:
            Dictionary containing paths to generated files and any error information.
        """
        return self.circuitization_pipeline(
            model_path,
            output_path,
            input_file_path=input_file_path,
            segment_details=segment_details,
        )

    @staticmethod
    def process_witness_output(witness_data: Any) -> Optional[Dict[str, Any]]:
        """
        Process the witness output data to get prediction results.

        This method handles JSTprove witness output format. JSTprove outputs
        a raw array of floats representing the final logits.

        Args:
            witness_data: The parsed JSON data from witness output.

        Returns:
            Dictionary containing processed predictions, or None if processing fails.
        """
        try:
            import torch

            # JSTprove outputs a dict with 'output' and 'rescaled_output' keys
            if isinstance(witness_data, dict) and "rescaled_output" in witness_data:
                # JSTprove format - use rescaled_output as logits
                logits = torch.tensor(witness_data["rescaled_output"])
                # Reshape to expected format [batch_size, ...]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)  # Add batch dimension
                return {"logits": logits}
            elif isinstance(witness_data, list):
                # Raw array format - return as logits
                logits = torch.tensor(witness_data)
                # Reshape to expected format [batch_size, ...]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)  # Add batch dimension
                return {"logits": logits}
            else:
                # Try EZKL-like format as fallback
                rescaled_outputs = witness_data["pretty_elements"]["rescaled_outputs"][0]
                logits = torch.tensor(rescaled_outputs)
                # Reshape to expected format [batch_size, ...]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)  # Add batch dimension
                return {"logits": logits}
        except (KeyError, TypeError) as e:
            logger.error(f"Could not process witness data: {e}")
            return None

        # Convert string values to float and create a tensor
        float_values = [float(val) for val in rescaled_outputs]

        # Create a tensor with shape [1, num_classes] to match batch_size, num_classes format
        tensor_output = torch.tensor([float_values], dtype=torch.float32)

        # Process the tensor through _process_final_output (simulating one segment)
        output = RunnerUtils.process_final_output(tensor_output)
        return output


# Backward compatibility alias with deprecation warning
class EZKL(JSTprove):
    """Deprecated: Use JSTprove instead of EZKL.

    This class is kept for backward compatibility but will be removed in a future version.
    JSTprove uses a different CLI interface than EZKL.
    """

    def __init__(self, model_directory=None):
        warnings.warn(
            "EZKL class is deprecated. Use JSTprove instead. "
            "Note: JSTprove has a different API than EZKL.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(model_directory)


if __name__ == "__main__":
    # Example usage with JSTprove
    print("JSTprove backend example:")
    print("backend = JSTprove()")
    print("backend.compile_circuit('model.onnx', 'circuit.txt')")
    print("backend.generate_witness('input.json', 'circuit.txt', 'output.json', 'witness.bin')")
    print("backend.prove('witness.bin', 'circuit.txt', 'proof.bin')")
    print("backend.verify('proof.bin', 'circuit.txt', 'input.json', 'output.json', 'witness.bin')")
