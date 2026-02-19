"""
JSTprove backend for zero-knowledge proof generation.
"""
import importlib.metadata
import json
import os
import tempfile
import torch
import logging
import onnx
from onnx import numpy_helper
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

from python.core.circuit_models.generic_onnx import GenericModelONNX
from python.core.utils.helper_functions import (
    CircuitExecutionConfig,
    RunType,
    read_from_json,
)
from python.core.utils.witness_utils import (
    compare_field_values,
    extract_io_from_witness,
    load_witness,
    scale_to_field,
    ZKProofSystems,
)
from python.frontend.commands.base import BaseCommand
from python.frontend.commands.batch import _run_witness_chunk_piped

from dsperse.src.backends.utils.jstprove_utils import JSTproveUtils, JSTPROVE_SUPPORTED_OPS

logger = logging.getLogger(__name__)


class JSTprove:
    """JSTprove backend for zero-knowledge proof generation."""

    SUPPORTED_OPS = JSTPROVE_SUPPORTED_OPS

    @staticmethod
    def is_compatible(model_path: Union[str, Path]) -> Tuple[bool, set]:
        """Check if an ONNX model contains only JSTprove-supported operations."""
        return JSTproveUtils.is_compatible(model_path)

    def __init__(self, model_directory: Optional[str] = None) -> None:
        self.env = os.environ.copy()
        self.model_directory = Path(model_directory) if model_directory else None
        self._witness_format = "jstprove"
        self._circuit_cache: Dict[str, GenericModelONNX] = {}

    def _build_circuit(self, model_path):
        circuit = BaseCommand._build_circuit(Path(model_path).stem)
        circuit.model_file_name = str(model_path)
        circuit.onnx_path = str(model_path)
        circuit.model_path = str(model_path)
        return circuit

    def _get_circuit(self, circuit_path: Path) -> GenericModelONNX:
        key = str(circuit_path)
        if key in self._circuit_cache:
            return self._circuit_cache[key]
        circuit = GenericModelONNX("_")
        quantized_path = circuit_path.parent / f"{circuit_path.stem}_quantized_model.onnx"
        if not quantized_path.exists():
            raise FileNotFoundError(f"Quantized model not found: {quantized_path}")
        circuit.load_quantized_model(str(quantized_path))
        self._circuit_cache[key] = circuit
        return circuit

    def _resolve_circuit_path(self, model_path: Path) -> Path:
        circuit_path = model_path
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
        return circuit_path

    @staticmethod
    def _populate_wai_inputs(circuit: GenericModelONNX, inputs: dict) -> Tuple[dict, set]:
        if not hasattr(circuit, 'quantized_model') or circuit.quantized_model is None:
            return inputs, set()
        if not hasattr(circuit, 'input_shape') or not isinstance(circuit.input_shape, dict):
            return inputs, set()
        missing = set(circuit.input_shape.keys()) - set(inputs.keys())
        if not missing:
            return inputs, set()
        init_map = {init.name: init for init in circuit.quantized_model.graph.initializer}
        inputs = dict(inputs)
        added = set()
        for name in missing:
            if name in init_map:
                inputs[name] = numpy_helper.to_array(init_map[name]).tolist()
                added.add(name)
        return inputs, added

    def _process_inputs(self, circuit: GenericModelONNX, inputs: dict) -> Tuple[dict, dict]:
        inputs, wai_keys = self._populate_wai_inputs(circuit, inputs)
        scaled = circuit.scale_inputs_only(inputs)
        inference_inputs = circuit.reshape_inputs_for_inference(scaled)
        circuit_inputs = circuit.reshape_inputs_for_circuit(scaled)
        outputs = circuit.get_outputs(inference_inputs)
        formatted = circuit.format_outputs(outputs)
        return circuit_inputs, formatted

    @staticmethod
    def _wai_safe_metadata(circuit_path: Path) -> str:
        meta_path = circuit_path.parent / f"{circuit_path.stem}_metadata.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if not meta.get("weights_as_inputs"):
            return str(meta_path)
        meta["weights_as_inputs"] = False
        safe_path = circuit_path.parent / f"{circuit_path.stem}_metadata_nowai.json"
        with open(safe_path, "w") as f:
            json.dump(meta, f)
        return str(safe_path)

    def generate_witness(
        self,
        input_file: Union[str, Path],
        model_path: Union[str, Path],
        output_file: Union[str, Path],
        vk_path: Optional[Union[str, Path]] = None,
        settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Any]:
        input_file, output_file = Path(input_file), Path(output_file)
        circuit_path = self._resolve_circuit_path(Path(model_path))
        witness_path = output_file.parent / f"{output_file.stem}_witness.bin"

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        witness_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            circuit = self._get_circuit(circuit_path)
            inputs = read_from_json(str(input_file))
            circuit_inputs, formatted = self._process_inputs(circuit, inputs)

            metadata_path = self._wai_safe_metadata(circuit_path)
            result = _run_witness_chunk_piped(
                binary_name=circuit.name,
                circuit_path=str(circuit_path),
                metadata_path=metadata_path,
                chunk_jobs=[{
                    "_circuit_inputs": circuit_inputs,
                    "_circuit_outputs": formatted,
                    "witness": str(witness_path),
                }],
            )

            if result.get("failed", 0) > 0:
                errors = result.get("errors", [])
                raise RuntimeError(f"Witness generation failed: {errors}")

            if not witness_path.exists():
                raise FileNotFoundError(f"Witness file not created: {witness_path}")

            with open(output_file, "w") as f:
                json.dump(formatted, f)

            return True, self.process_witness_output(formatted)
        except Exception as e:
            logger.error(f"Witness generation failed: {e}")
            return False, str(e)

    def generate_witness_batch(
        self,
        circuit_path: Union[str, Path],
        jobs: List[Dict[str, str]],
        manifest_dir: Optional[Union[str, Path]] = None,
    ) -> List[Tuple[bool, Any]]:
        circuit_path = self._resolve_circuit_path(Path(circuit_path))

        for job in jobs:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["witness"]).parent.mkdir(parents=True, exist_ok=True)

        try:
            circuit = self._get_circuit(circuit_path)
            metadata_path = self._wai_safe_metadata(circuit_path)

            piped_jobs = []
            per_job_formatted = []
            for job in jobs:
                inputs = read_from_json(job["input"])
                circuit_inputs, formatted = self._process_inputs(circuit, inputs)
                piped_jobs.append({
                    "_circuit_inputs": circuit_inputs,
                    "_circuit_outputs": formatted,
                    "witness": job["witness"],
                })
                per_job_formatted.append(formatted)

            result = _run_witness_chunk_piped(
                binary_name=circuit.name,
                circuit_path=str(circuit_path),
                metadata_path=metadata_path,
                chunk_jobs=piped_jobs,
            )
            if result.get("failed", 0) > 0:
                raise RuntimeError(f"Batch witness failed: {result.get('errors', [])}")

            results = []
            for job, formatted in zip(jobs, per_job_formatted):
                with open(job["output"], "w") as f:
                    json.dump(formatted, f)
                results.append((True, self.process_witness_output(formatted)))

            return results
        except Exception as e:
            logger.error(f"Batch witness generation failed: {e}")
            return [(False, str(e)) for _ in jobs]

    def generate_witness_batch_from_tensors(
        self,
        circuit_path: Union[str, Path],
        jobs: List[Dict[str, Any]],
        manifest_dir: Optional[Union[str, Path]] = None,
        workers: int = 1,
    ) -> List[Tuple[bool, Any]]:
        circuit_path = self._resolve_circuit_path(Path(circuit_path))

        for job in jobs:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["witness"]).parent.mkdir(parents=True, exist_ok=True)

        try:
            circuit = self._get_circuit(circuit_path)
            metadata_path = self._wai_safe_metadata(circuit_path)

            piped_jobs = []
            per_job_formatted = []
            for job in jobs:
                inputs = job.get("_tensor_inputs") or job.get("input")
                if isinstance(inputs, str):
                    inputs = read_from_json(inputs)
                circuit_inputs, formatted = self._process_inputs(circuit, inputs)
                piped_jobs.append({
                    "_circuit_inputs": circuit_inputs,
                    "_circuit_outputs": formatted,
                    "witness": job["witness"],
                })
                per_job_formatted.append(formatted)

            chunk_size = max(1, len(piped_jobs) // max(1, workers)) if workers > 1 else 0
            if chunk_size <= 0:
                chunks = [piped_jobs]
            else:
                chunks = [piped_jobs[i:i + chunk_size] for i in range(0, len(piped_jobs), chunk_size)]

            total_failed = 0
            all_errors = []
            for chunk in chunks:
                result = _run_witness_chunk_piped(
                    binary_name=circuit.name,
                    circuit_path=str(circuit_path),
                    metadata_path=metadata_path,
                    chunk_jobs=chunk,
                )
                total_failed += result.get("failed", 0)
                all_errors.extend(result.get("errors", []))
            if total_failed > 0:
                raise RuntimeError(f"Batch witness failed for {total_failed} jobs: {all_errors}")

            results = []
            for job, formatted in zip(jobs, per_job_formatted):
                with open(job["output"], "w") as f:
                    json.dump(formatted, f)
                results.append((True, self.process_witness_output(formatted)))

            return results
        except Exception as e:
            logger.error(f"Batch witness generation failed: {e}")
            return [(False, str(e)) for _ in jobs]

    def prove(
        self,
        witness_path: Union[str, Path],
        circuit_path: Union[str, Path],
        proof_path: Union[str, Path],
        pk_path: Optional[Union[str, Path]] = None,
        check_mode: str = "unsafe",
        settings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Union[str, Path]]:
        witness_path = Path(witness_path)
        circuit_path = Path(circuit_path)
        proof_path = Path(proof_path)

        if not witness_path.exists():
            raise FileNotFoundError(f"Witness file not found: {witness_path}")
        if not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

        proof_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            circuit = self._build_circuit("cli")
            circuit.base_testing(CircuitExecutionConfig(
                run_type=RunType.PROVE_WITNESS,
                circuit_path=str(circuit_path),
                witness_file=str(witness_path),
                proof_file=str(proof_path),
            ))
        except Exception as e:
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
            circuit = self._build_circuit("cli")
            circuit.base_testing(CircuitExecutionConfig(
                run_type=RunType.GEN_VERIFY,
                circuit_path=str(circuit_path),
                input_file=str(input_path),
                output_file=str(veri_output_path),
                witness_file=str(witness_path),
                proof_file=str(proof_path),
            ))
            return True
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False

    def verify_with_io_extraction(
        self,
        circuit_path: Union[str, Path],
        witness_bytes: bytes,
        proof_bytes: bytes,
        num_inputs: int,
        expected_inputs: Optional[List] = None,
        num_outputs: Optional[int] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify a proof and extract cryptographically-bound inputs/outputs from the witness.

        This method performs trustless verification by:
        1. Loading the witness and extracting public inputs
        2. Parsing inputs, outputs, and scaling parameters from the witness
        3. Optionally comparing extracted inputs against expected inputs
        4. Verifying the proof directly against the witness

        Args:
            circuit_path: Path to the circuit file
            witness_bytes: Raw witness bytes
            proof_bytes: Raw proof bytes
            num_inputs: Number of input values in the witness public inputs
            expected_inputs: Optional flat list of expected input values for comparison
            num_outputs: Number of output values in the witness public inputs.
                When None, inferred from the witness layout (valid for non-WAI witnesses).

        Returns:
            Tuple of (success, extracted_io) where extracted_io contains:
                - "inputs": raw input values from witness
                - "outputs": raw output values from witness
                - "rescaled_outputs": outputs converted back to original scale
                - "scale_base": scaling base from witness
                - "scale_exponent": scaling exponent from witness
                - "inputs_match": whether inputs matched expected (if provided)
        """
        circuit_path = Path(circuit_path)
        if not circuit_path.exists():
            logger.error(f"Circuit file not found: {circuit_path}")
            return False, None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            witness_path = tmp_path / "witness.bin"
            proof_path = tmp_path / "proof.bin"

            with open(witness_path, "wb") as f:
                f.write(witness_bytes)
            with open(proof_path, "wb") as f:
                f.write(proof_bytes)

            try:
                witness_data = load_witness(str(witness_path), ZKProofSystems.Expander)
            except Exception as e:
                logger.error(f"Failed to load witness: {e}")
                return False, None

            if num_outputs is None:
                public_inputs = (
                    witness_data.get("witnesses", [{}])[0].get("public_inputs", [])
                )
                num_outputs = len(public_inputs) - num_inputs - 2

            extracted = extract_io_from_witness(witness_data, num_inputs, num_outputs)
            if extracted is None:
                logger.error("Invalid witness structure")
                return False, None

            inputs_match = None
            if expected_inputs is not None:
                if len(expected_inputs) != len(extracted["inputs"]):
                    logger.error(
                        f"Input length mismatch: expected {len(expected_inputs)}, "
                        f"witness has {len(extracted['inputs'])}"
                    )
                    return False, None

                scaled_expected = scale_to_field(
                    expected_inputs,
                    extracted["scale_base"],
                    extracted["scale_exponent"],
                    extracted["modulus"],
                )
                inputs_match = compare_field_values(
                    scaled_expected, extracted["inputs"], extracted["modulus"]
                )
                if not inputs_match:
                    logger.error("Input verification failed: witness inputs don't match expected")
                    return False, None

            input_json = tmp_path / "input_veri.json"
            output_json = tmp_path / "output_veri.json"
            with open(input_json, "w") as f:
                json.dump({"input_data": extracted["inputs"]}, f)
            with open(output_json, "w") as f:
                json.dump({"output": extracted["outputs"]}, f)

            try:
                circuit = self._build_circuit("cli")
                circuit.base_testing(CircuitExecutionConfig(
                    run_type=RunType.GEN_VERIFY,
                    circuit_path=str(circuit_path),
                    input_file=str(input_json),
                    output_file=str(output_json),
                    witness_file=str(witness_path),
                    proof_file=str(proof_path),
                ))
            except Exception as e:
                logger.error(f"Proof verification failed: {e}")
                return False, None

            extracted_io = {
                "inputs": extracted["inputs"],
                "outputs": extracted["outputs"],
                "rescaled_outputs": extracted["rescaled_outputs"],
                "scale_base": extracted["scale_base"],
                "scale_exponent": extracted["scale_exponent"],
                "inputs_match": inputs_match,
            }
            return True, extracted_io

    @staticmethod
    def _prepare_verification_output(circuit_path: Path, output_path: Path) -> Path:
        with open(output_path, "r") as f:
            output_data = json.load(f)

        raw_output = output_data.get("raw_output")
        if raw_output is not None:
            veri_path = output_path.parent / "output_veri.json"
            with open(veri_path, "w") as f:
                json.dump({"output": raw_output}, f)
            return veri_path

        metadata_path = circuit_path.with_name(circuit_path.stem + "_metadata.json")
        if not metadata_path.exists():
            return output_path

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        scale_base = metadata.get("scale_base")
        scale_exponent = metadata.get("scale_exponent")
        if scale_base is None or scale_exponent is None:
            return output_path

        output_values = output_data.get("output")
        if output_values is None:
            return output_path

        flat = torch.tensor(output_values).flatten()

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
        _settings_path: Optional[Union[str, Path]] = None,
        weights_as_inputs: bool = False,
        input_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[bool, Optional[str]]:
        model_path = Path(model_path)
        circuit_path = Path(circuit_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        circuit_path.parent.mkdir(parents=True, exist_ok=True)

        model = onnx.load(str(model_path))
        model = JSTproveUtils.add_zero_bias_to_conv(model)
        if weights_as_inputs:
            model = JSTproveUtils.promote_initializers_to_inputs(model)

        fd, preprocessed_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            onnx.save(model, preprocessed_path)
            circuit = self._build_circuit(preprocessed_path)
            circuit.weights_as_inputs = weights_as_inputs
            circuit.input_bounds = input_bounds
            circuit.base_testing(CircuitExecutionConfig(
                run_type=RunType.COMPILE_CIRCUIT,
                circuit_path=str(circuit_path),
            ))
            if not circuit_path.exists():
                return False, f"Rust binary exited successfully but did not produce circuit file at {circuit_path}"
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
        _segment_details: Optional[Dict[str, Any]] = None,
        weights_as_inputs: bool = False
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
            ok, err = self.compile_circuit(model_path=model_path, circuit_path=circuit_path, weights_as_inputs=weights_as_inputs)
            
            if not ok:
                logger.warning(f"Failed to compile circuit: {err}")
                circuitization_data["compile_error"] = err
                return circuitization_data

            quantized_model_path = artifacts["paths"]["quantized_model"]
            if not Path(quantized_model_path).exists():
                alt_path = circuit_path.parent / f"{circuit_path.stem}_quantized_model.onnx"
                if alt_path.exists():
                    circuitization_data["quantized_model"] = str(alt_path)
                else:
                    logger.warning(f"Quantized model not found after compilation: {quantized_model_path}")
                    circuitization_data["compile_error"] = f"Missing quantized model at {quantized_model_path}"
                    return circuitization_data
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
        try:
            return importlib.metadata.version("jstprove")
        except Exception as e:
            logger.debug(f"Could not get JSTprove version: {e}")
        return None

    def __repr__(self) -> str:
        return f"JSTprove(version={self.get_version()})"
