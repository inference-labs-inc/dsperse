"""
Base backend interface for dsperse proof systems.

This module defines the abstract base class that all backends (EZKL, JSTprove, etc.)
must implement to ensure consistent API across the dsperse system.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


class BaseBackend(ABC):
    """
    Abstract base class for all proof system backends in dsperse.

    This interface standardizes the methods that all backends must implement,
    allowing the orchestrators (Compiler, Runner, Prover, Verifier) to work
    with any backend without knowing the specific implementation details.
    """

    @abstractmethod
    def compile_circuit(
        self,
        model_path: Union[str, Path],
        circuit_path: Union[str, Path],
        settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Compile a circuit from an ONNX model.

        Args:
            model_path: Path to the ONNX model file
            circuit_path: Path where to save the compiled circuit
            settings_path: Optional path to settings file for compilation

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        pass

    @abstractmethod
    def generate_witness(
        self,
        input_file: Union[str, Path],
        model_path: Union[str, Path],
        output_file: Union[str, Path],
        vk_path: Union[str, Path],
        settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Any]:
        """
        Generate a witness for inference on the given model.

        Args:
            input_file: Path to input JSON file
            model_path: Path to compiled model/circuit
            output_file: Path where to save model outputs
            vk_path: Path to verification key
            settings_path: Optional path to settings file

        Returns:
            Tuple of (success: bool, witness_data: Any)
        """
        pass

    @abstractmethod
    def prove(
        self,
        witness_path: Union[str, Path],
        model_path: Union[str, Path],
        proof_path: Union[str, Path],
        pk_path: Union[str, Path],
        check_mode: str = "unsafe",
        settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Union[str, Path]]:
        """
        Generate a proof from a witness.

        Args:
            witness_path: Path to witness file
            model_path: Path to compiled model/circuit
            proof_path: Path where to save the proof
            pk_path: Path to proving key
            check_mode: Verification mode (e.g., "unsafe", "safe")
            settings_path: Optional path to settings file

        Returns:
            Tuple of (success: bool, proof_path: Union[str, Path])
        """
        pass

    @abstractmethod
    def verify(
        self,
        proof_path: Union[str, Path],
        settings_path: Union[str, Path],
        vk_path: Union[str, Path]
    ) -> bool:
        """
        Verify a proof.

        Args:
            proof_path: Path to proof file
            settings_path: Path to settings file
            vk_path: Path to verification key

        Returns:
            True if verification succeeds, False otherwise
        """
        pass

    @abstractmethod
    def circuitization_pipeline(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        input_file_path: Optional[Union[str, Path]] = None,
        segment_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the full circuitization pipeline for a model.

        Args:
            model_path: Path to ONNX model file
            output_path: Base directory for output artifacts
            input_file_path: Optional path to input data for calibration
            segment_details: Optional metadata about the segment being processed

        Returns:
            Dictionary containing paths and status of generated artifacts
        """
        pass
