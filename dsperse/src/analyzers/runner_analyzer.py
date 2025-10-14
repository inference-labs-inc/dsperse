"""
Generates execution chain metadata for EzKL circuit and ONNX slice inference
with proper fallback mapping and security calculation.
"""
import logging
import json
import os
from pathlib import Path

from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class RunnerAnalyzer:
    def __init__(self, model_directory, backend_name=None):
        """
        Args:
            model_directory: Path to the model directory.
            backend_name: Name of the backend (ezkl, jstprove, etc.)
        """
        self.model_directory = model_directory
        self.backend_name = backend_name or 'ezkl'
        self.slices_dir = Path(os.path.join(model_directory, "slices")).resolve()
        self.slices_metadata_path = self.slices_dir / "metadata.json"

        self.size_limit = 1000 * 1024 * 1024  # 1000MB

        if not self.slices_dir.exists():
            raise FileNotFoundError(f"Slice output directory not found: {self.slices_dir}")

    def generate_metadata(self, save_path=None):
        """
        Generate runner metadata and save to run_metadata.json.
        Returns:
            Path to generated metadata for running the slices and model
        """

        logger.info(f"Generating runner metadata...")
        slices_metadata = self._load_slices_metadata()

        runner_metadata = self._generate_metadata(slices_metadata)

        save_path = Path(save_path) if save_path else Path(self.model_directory) / "run" / "metadata.json"
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        runner_metadata["run_directory"] = str(save_path.parent)

        print(f"Saving runner metadata to {save_path}")
        Utils.save_metadata_file(runner_metadata, save_path)

        logger.info(f"Runner metadata saved to {save_path}")

        return save_path

    def _load_slices_metadata(self):
        try:
            with open(self.slices_metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load slices metadata: {e}")
            raise

    def _generate_metadata(self, slices_metadata):
        segments = slices_metadata.get('segments', [])
        slices = self._process_slices(segments)
        execution_chain = self._build_execution_chain(segments)
        circuit_slices = self._build_circuit_slices(slices)
        overall_security = self._calculate_security(slices)
        return {
            "model_path": str(self.model_directory),
            "overall_security": overall_security,
            "slices": slices,
            "execution_chain": execution_chain,
            "circuit_slices": circuit_slices,
        }

    def _process_slices(self, segments):
        """
        Build the slices dictionary with metadata for each segment.
        """

        slices = {}
        for segment in segments:
            segment_idx = segment['index']
            segment_key = f"segment_{segment_idx}"

            # Get backend circuitization paths from metadata
            backend_circuitization = segment.get(f'{self.backend_name}_circuitization', {})

            # Use paths from metadata or set to None if not present
            compiled_circuit_path = backend_circuitization.get('compiled', None)
            settings_path = backend_circuitization.get('settings', None)
            pk_path = backend_circuitization.get('pk_key', None)
            vk_path = backend_circuitization.get('vk_key', None)

            # Check if this is a JSTprove circuit (different requirements)
            is_jstprove = False
            if settings_path and os.path.exists(settings_path):
                try:
                    with open(settings_path, 'r') as f:
                        settings_data = json.load(f)
                        is_jstprove = settings_data.get('backend') == 'jstprove'
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

            # Set circuit_exists and keys_exist flags based on actual file existence
            # For JSTprove, we use the ONNX model path since JSTprove can compile it directly
            if is_jstprove:
                # Use the segment ONNX path for JSTprove (it will compile it)
                compiled_circuit_path = segment.get('path')  # This is the ONNX file
                circuit_exists = bool(compiled_circuit_path) and os.path.exists(compiled_circuit_path)
            else:
                # For EZKL, use the compiled circuit path
                circuit_exists = bool(compiled_circuit_path) and os.path.exists(compiled_circuit_path) \
                                 and bool(settings_path) and os.path.exists(settings_path)

            # For JSTprove, keys are not required; for EZKL, keys are required
            keys_exist = is_jstprove or (bool(pk_path) and os.path.exists(pk_path) and bool(vk_path) and os.path.exists(vk_path))

            # Determine circuit size and use_circuit flag
            circuit_size = 0
            if circuit_exists:
                try:
                    circuit_size = Path(compiled_circuit_path).stat().st_size
                except Exception:
                    # If there's any error, just use 0 as the size
                    circuit_size = 0

            # Treat any recorded circuitization error as not ready
            backend_errors = any(k.endswith("_error") for k in backend_circuitization.keys()) or ("error" in backend_circuitization)

            use_circuit = circuit_exists and keys_exist and (not backend_errors) and circuit_size <= self.size_limit

            onnx_slice_path = segment.get('path', '')
            if not onnx_slice_path:
                logger.warning(f"No ONNX slice path for segment {segment_idx}")

            # Build slice metadata
            slice_metadata = {
                "path": onnx_slice_path,
                "input_shape": segment.get('shape', {}).get('tensor_shape', {}).get('input', ["batch_size", "unknown"]),
                "output_shape": segment.get('shape', {}).get('tensor_shape', {}).get('output',
                                                                                     ["batch_size", "unknown"]),
                "ezkl_compatible": True,
                "ezkl": use_circuit,
                "circuit_size": circuit_size,
                "dependencies": segment.get('dependencies', {}),
                "parameters": segment.get('parameters', 0)
            }

            # Add circuit paths to metadata if they exist in the segment metadata
            if circuit_exists:
                # Get the parent directory for the circuit to derive proof and witness paths
                circuit_dir = str(Path(compiled_circuit_path).parent)

                # Use standard naming convention for proof and witness paths
                # proof_path = f"{circuit_dir}/segment_{segment_idx}_proof.json"
                # witness_path = f"{circuit_dir}/segment_{segment_idx}_witness.json"

                slice_metadata.update({
                    "circuit_path": compiled_circuit_path,
                    # "proof_path": proof_path,
                    # "witness_path": witness_path,
                    "settings_path": settings_path
                })

                if keys_exist:
                    slice_metadata.update({
                        "vk_path": vk_path,
                        "pk_path": pk_path
                    })

            slices[segment_key] = slice_metadata

        return slices

    def _build_execution_chain(self, segments):
        """
        Build the execution chain with proper node connections and fallback mapping.
        """
        execution_chain = {
            "head": "segment_0" if segments else None,
            "nodes": {},
            "fallback_map": {}
        }

        for segment in segments:
            segment_idx = segment['index']
            segment_key = f"segment_{segment_idx}"

            # Get backend circuitization paths from metadata
            backend_circuitization = segment.get(f'{self.backend_name}_circuitization', {})
            compiled_circuit_path = backend_circuitization.get('compiled', None)

            # Get ONNX path
            onnx_slice_path = segment.get('path', '')

            # Determine if circuit is usable based on actual files and errors
            settings_path = backend_circuitization.get('settings')
            pk_path = backend_circuitization.get('pk_key')
            vk_path = backend_circuitization.get('vk_key')

            # Check if this is a JSTprove circuit
            is_jstprove = False
            if settings_path and os.path.exists(settings_path):
                try:
                    with open(settings_path, 'r') as f:
                        settings_data = json.load(f)
                        is_jstprove = settings_data.get('backend') == 'jstprove'
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

            circuit_exists = bool(compiled_circuit_path) and os.path.exists(compiled_circuit_path) \
                              and bool(settings_path) and os.path.exists(settings_path)
            # For JSTprove, keys are not required; for EZKL, keys are required
            keys_exist = is_jstprove or (bool(pk_path) and os.path.exists(pk_path) and bool(vk_path) and os.path.exists(vk_path))
            backend_errors = any(k.endswith("_error") for k in backend_circuitization.keys()) or ("error" in backend_circuitization)
            use_circuit = circuit_exists and keys_exist and (not backend_errors)

            # Set up the execution chain node
            next_slice = f"segment_{segment_idx + 1}" if segment_idx < len(segments) - 1 else None
            execution_chain["nodes"][segment_key] = {
                "segment_id": segment_key,
                "primary": compiled_circuit_path if use_circuit else onnx_slice_path,
                "fallback": onnx_slice_path,
                "use_circuit": use_circuit,
                "next": next_slice,
                "circuit_path": compiled_circuit_path if circuit_exists else None,
                "onnx_path": onnx_slice_path
            }

            # Set up the fallback map
            if circuit_exists and onnx_slice_path:
                execution_chain["fallback_map"][compiled_circuit_path] = onnx_slice_path
            elif onnx_slice_path:
                execution_chain["fallback_map"][segment_key] = onnx_slice_path

        return execution_chain

    @staticmethod
    def _build_circuit_slices(slices):
        """
        Build dictionary tracking which slices use circuits.
        """
        circuit_slices = {}
        for slice_key, slice_data in slices.items():
            # Check if the slice has circuit_path and pk_path set from the metadata
            has_circuit = slice_data.get("circuit_path") is not None
            has_keys = slice_data.get("pk_path") is not None

            # A slice is considered to use a circuit if it has both circuit_path and pk_path
            circuit_slices[slice_key] = has_circuit and has_keys

        return circuit_slices

    @staticmethod
    def _calculate_security(slices):
        if not slices:
            return 0.0
        total_slices = len(slices)
        circuit_slices = sum(1 for slice_data in slices.values() if slice_data.get("ezkl", False))
        return round((circuit_slices / total_slices) * 100, 1)

if __name__ == "__main__":

    model_choice = 1

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet"
    }

    model_dir = base_paths[model_choice] #+ "/model.onnx"
    model_path = Path(model_dir).resolve()
    print(f"Model path: {model_path}")
    metadata = RunnerAnalyzer(model_dir)
    metadata.generate_metadata()
