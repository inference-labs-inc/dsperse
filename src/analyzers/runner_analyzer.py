"""
Generates execution chain metadata for EzKL circuit and ONNX slice inference
with proper fallback mapping and security calculation.
"""
import logging
import json
import os
from pathlib import Path

from src.utils.utils import Utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RunnerAnalyzer:
    def __init__(self, model_directory):
        """
        Args:
            model_directory: Path to the model directory.
        """
        # Expand and resolve the model directory path
        self.model_directory = os.path.expanduser(model_directory)
        # Create absolute paths for slices directory and metadata
        self.slices_dir = Path(os.path.join(model_directory, "slices")).resolve()
        self.slices_metadata_path = self.slices_dir / "metadata.json"

        self.size_limit = 100 * 1024 * 1024  # 100MB

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

            # Get EZKL paths directly from metadata
            ezkl_circuitization = segment.get('ezkl_circuitization', {})

            # Use paths from metadata or set to None if not present
            compiled_circuit_path = ezkl_circuitization.get('compiled', None)
            settings_path = ezkl_circuitization.get('settings', None)
            pk_path = ezkl_circuitization.get('pk_key', None)
            vk_path = ezkl_circuitization.get('vk_key', None)

            # Set circuit_exists and keys_exist flags based on whether the paths exist in metadata
            circuit_exists = compiled_circuit_path is not None and settings_path is not None
            keys_exist = pk_path is not None and vk_path is not None

            # Determine circuit size and use_circuit flag
            circuit_size = 0
            if circuit_exists:
                try:
                    circuit_size = Path(compiled_circuit_path).stat().st_size
                except:
                    # If there's any error, just use 0 as the size
                    circuit_size = 0

            use_circuit = circuit_exists and keys_exist and circuit_size <= self.size_limit

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

    @staticmethod
    def _build_execution_chain(segments):
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

            # Get EZKL paths from metadata
            ezkl_circuitization = segment.get('ezkl_circuitization', {})
            compiled_circuit_path = ezkl_circuitization.get('compiled', None)

            # Get ONNX path
            onnx_slice_path = segment.get('path', '')

            # Determine if circuit is usable based purely on metadata presence
            circuit_exists = compiled_circuit_path is not None
            use_circuit = circuit_exists and ezkl_circuitization.get('pk_key') is not None

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
