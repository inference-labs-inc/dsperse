"""
JSTprove backend for zero-knowledge proof generation.
This module provides a backend for generating ZK proofs using the JSTprove CLI.
"""
import json
import os
import subprocess
import tempfile
import torch
import logging
import onnx
import numpy as np
from onnx import numpy_helper
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

from dsperse.src.constants import JSTPROVE_COMMAND
from dsperse.src.backends.utils.jstprove_utils import JSTproveUtils, JSTPROVE_SUPPORTED_OPS

logger = logging.getLogger(__name__)


class JSTprove:
    """JSTprove backend for zero-knowledge proof generation using the JSTprove CLI."""

    # Class constants
    COMMAND = JSTPROVE_COMMAND
    DEFAULT_FLAGS = ["--no-banner"]
    SUPPORTED_OPS = JSTPROVE_SUPPORTED_OPS

    @staticmethod
    def is_compatible(model_path: Union[str, Path]) -> Tuple[bool, set]:
        """Check if an ONNX model contains only JSTprove-supported operations."""
        return JSTproveUtils.is_compatible(model_path)

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
        vk_path: Optional[Union[str, Path]] = None,
        settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Any]:
        """Generate a witness for the given circuit and input using JSTprove."""
        # --- Normalization & Validation ---
        input_file, output_file = Path(input_file), Path(output_file)
        circuit_path = Path(model_path)
        witness_path = output_file.parent / f"{output_file.stem}_witness.bin"

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # --- Circuit Compilation (JIT) ---
        onnx_model_path = None
        if circuit_path.exists() and circuit_path.suffix == '.onnx':
            onnx_model_path = circuit_path
            circuit_path = circuit_path.parent / f"{circuit_path.stem}_jstprove_circuit.txt"

        if onnx_model_path and not circuit_path.exists():
            logger.info(f"JSTprove: Compiling ONNX model {onnx_model_path} to circuit {circuit_path}")
            ok, err = self.compile_circuit(onnx_model_path, circuit_path)
            if not ok: raise RuntimeError(f"Circuit compilation failed: {err}")
        elif not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

        # --- Execution ---
        output_file.parent.mkdir(parents=True, exist_ok=True)
        witness_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_command("witness", [
                "-c", str(circuit_path), "-i", str(input_file),
                "-o", str(output_file), "-w", str(witness_path),
            ])
        except RuntimeError as e:
            logger.error(f"Witness generation failed: {e}")
            return False, str(e)

        # --- Result Processing ---
        try:
            with open(output_file, "r") as f:
                output_data = json.load(f)
                processed_output = self.process_witness_output(output_data)
            return True, processed_output
        except Exception as e:
            logger.error(f"Failed to process witness output: {e}")
            return False, str(e)

    def generate_witness_batch(
        self,
        circuit_path: Union[str, Path],
        jobs: List[Dict[str, str]],
        manifest_dir: Optional[Union[str, Path]] = None,
    ) -> List[Tuple[bool, Any]]:
        """Generate witnesses for multiple inputs in a single batch call.

        All jobs share the same circuit. The circuit is loaded once by the
        Rust binary, reducing disk IO from O(n) to O(1).

        Args:
            circuit_path: Path to the compiled circuit (or ONNX model for JIT compilation)
            jobs: List of dicts each with 'input', 'output', 'witness' string paths
            manifest_dir: Directory to write the manifest file (defaults to circuit parent)

        Returns:
            List of (success, result) tuples, one per job
        """
        circuit_path = Path(circuit_path)

        onnx_model_path = None
        if circuit_path.exists() and circuit_path.suffix == '.onnx':
            onnx_model_path = circuit_path
            circuit_path = circuit_path.parent / f"{circuit_path.stem}_jstprove_circuit.txt"

        if onnx_model_path and not circuit_path.exists():
            ok, err = self.compile_circuit(onnx_model_path, circuit_path)
            if not ok:
                raise RuntimeError(f"Circuit compilation failed: {err}")
        elif not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

        for job in jobs:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["witness"]).parent.mkdir(parents=True, exist_ok=True)

        manifest_parent = Path(manifest_dir) if manifest_dir else circuit_path.parent
        manifest_parent.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_parent / "batch_witness_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({"jobs": jobs}, f)

        try:
            self._run_command("batch", [
                "witness",
                "-c", str(circuit_path),
                "-f", str(manifest_path),
            ])
        except RuntimeError as e:
            logger.error(f"Batch witness generation failed: {e}")
            return [(False, str(e)) for _ in jobs]

        results = []
        for job in jobs:
            output_path = Path(job["output"])
            try:
                with open(output_path, "r") as f:
                    output_data = json.load(f)
                processed = self.process_witness_output(output_data)
                results.append((True, processed))
            except Exception as e:
                results.append((False, str(e)))

        return results

    def generate_witness_batch_from_tensors(
        self,
        circuit_path: Union[str, Path],
        jobs: List[Dict[str, Any]],
        manifest_dir: Optional[Union[str, Path]] = None,
        workers: int = 1,
    ) -> List[Tuple[bool, Any]]:
        from python.frontend.commands.batch import batch_witness_from_tensors

        circuit_path = Path(circuit_path)

        onnx_model_path = None
        if circuit_path.exists() and circuit_path.suffix == '.onnx':
            onnx_model_path = circuit_path
            circuit_path = circuit_path.parent / f"{circuit_path.stem}_jstprove_circuit.txt"

        if onnx_model_path and not circuit_path.exists():
            ok, err = self.compile_circuit(onnx_model_path, circuit_path)
            if not ok:
                raise RuntimeError(f"Circuit compilation failed: {err}")
        elif not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

        for job in jobs:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["witness"]).parent.mkdir(parents=True, exist_ok=True)

        manifest_parent = Path(manifest_dir) if manifest_dir else circuit_path.parent
        manifest_parent.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_parent / "batch_witness_manifest.json"

        try:
            raw_outputs = batch_witness_from_tensors(
                circuit_path=str(circuit_path),
                jobs=jobs,
                manifest_path=str(manifest_path),
                workers=workers,
            )
        except Exception as e:
            logger.error(f"Batch witness generation failed: {e}")
            return [(False, str(e)) for _ in jobs]

        results = []
        for output_data in raw_outputs:
            try:
                processed = self.process_witness_output(output_data)
                results.append((True, processed))
            except Exception as e:
                results.append((False, str(e)))

        return results

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
        settings_path: Optional[Union[str, Path]] = None,
        vk_path: Optional[Union[str, Path]] = None
    ) -> bool:
        proof_path = Path(proof_path)
        circuit_path = Path(circuit_path)
        input_path = Path(input_path)
        output_path = Path(output_path)
        witness_path = Path(witness_path)

        required_files = [proof_path, circuit_path, input_path, output_path, witness_path]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        veri_output_path = self._prepare_verification_output(circuit_path, output_path)

        try:
            self._run_command("verify", [
                "-c", str(circuit_path),
                "-i", str(input_path),
                "-o", str(veri_output_path),
                "-w", str(witness_path),
                "-p", str(proof_path),
            ])
            return True
        except RuntimeError as e:
            logger.error(f"Proof verification failed: {e}")
            return False

    @staticmethod
    def _prepare_verification_output(circuit_path: Path, output_path: Path) -> Path:
        metadata_path = circuit_path.with_name(circuit_path.stem + "_metadata.json")
        if not metadata_path.exists():
            return output_path

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        scale_base = metadata.get("scale_base")
        scale_exponent = metadata.get("scale_exponent")
        if scale_base is None or scale_exponent is None:
            return output_path

        with open(output_path, "r") as f:
            output_data = json.load(f)

        raw_output = output_data.get("output")
        if raw_output is None:
            return output_path

        flat = torch.tensor(raw_output).flatten()

        if flat.is_floating_point():
            scale = scale_base ** scale_exponent
            flat = torch.round(flat * scale).long()

        scaled = flat.long().tolist()

        veri_path = output_path.parent / "output_veri.json"
        with open(veri_path, "w") as f:
            json.dump({"output": scaled}, f)

        return veri_path

    def compile_circuit(
        self,
        model_path: Union[str, Path],
        circuit_path: Union[str, Path],
        settings_path: Optional[Union[str, Path]] = None
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
        model_path = Path(model_path)
        circuit_path = Path(circuit_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        circuit_path.parent.mkdir(parents=True, exist_ok=True)

        model = onnx.load(str(model_path))
        model = JSTproveUtils.add_zero_bias_to_conv(model)

        fd, preprocessed_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            onnx.save(model, preprocessed_path)
            self._run_command("compile", [
                "-m", preprocessed_path,
                "-c", str(circuit_path),
            ])
            return True, None
        except Exception as e:
            error_msg = f"Circuit compilation failed: {e}"
            logger.error(error_msg)
            return False, error_msg
        finally:
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)

    def circuitization_pipeline(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        input_file_path: Optional[Union[str, Path]] = None,
        segment_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the JSTprove circuitization pipeline."""
        # --- Validation & Normalization ---
        model_path, output_path = Path(model_path), Path(output_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Artifact Preparation ---
        artifacts = JSTproveUtils.initialize_circuitization_artifacts(model_path, output_path, str(input_file_path) if input_file_path else None)
        circuit_path = artifacts["paths"]["circuit"]
        settings_path = artifacts["paths"]["settings"]
        circuitization_data = artifacts["data"]

        # --- Circuit Compilation ---
        try:
            logger.info(f"Compiling circuit for {model_path.stem}")
            ok, err = self.compile_circuit(model_path=model_path, circuit_path=circuit_path)
            
            if not ok:
                logger.warning(f"Failed to compile circuit: {err}")
                circuitization_data["compile_error"] = err
                return circuitization_data

            # --- Settings Generation (Dummy) ---
            dummy_settings = JSTproveUtils.create_dummy_settings(model_path, circuit_path, output_path)
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
        """Process the witness output data to get prediction results."""
        try:
            # --- JSTprove Dict Format ---
            if isinstance(witness_data, dict) and "rescaled_output" in witness_data:
                self._witness_format = "jstprove_dict"
                logger.debug("Using rescaled outputs from output.json (not witness binary).")
                return {"logits": JSTproveUtils.convert_to_logits(witness_data["rescaled_output"])}
            
            # --- Raw Array Format ---
            elif isinstance(witness_data, list):
                self._witness_format = "jstprove_list"
                return {"logits": JSTproveUtils.convert_to_logits(witness_data)}
            
            # --- EZKL Fallback Format ---
            else:
                self._witness_format = "ezkl_compat"
                rescaled = witness_data["pretty_elements"]["rescaled_outputs"][0]
                return {"logits": JSTproveUtils.convert_to_logits(rescaled)}
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

    def __repr__(self) -> str:
        return f"JSTprove(version={self.get_version()})"
