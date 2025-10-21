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
    def __init__(self, model_directory):
        """
        Args:
            model_directory: Path to the model directory.
        """
        self.model_directory = model_directory
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
        execution_chain = self._build_execution_chain(slices)
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
        Build the slices dictionary with metadata for each slice.
        Reads per-slice metadata from slice_#/metadata.json and adapts to new layout.
        """

        slices = {}
        for segment in segments:
            segment_idx = segment.get('index')
            slice_key = f"slice_{segment_idx}"

            onnx_slice_path = segment.get('path', '')
            if not onnx_slice_path:
                logger.warning(f"No ONNX slice path for slice {segment_idx}")

            # Resolve per-slice metadata path
            slice_meta_path = segment.get('slice_metadata')
            if not slice_meta_path and onnx_slice_path:
                try:
                    p = Path(onnx_slice_path)
                    # Expecting .../slice_#/payload/slice_#.onnx -> slice dir is parent of payload
                    slice_dir = p.parent.parent if p.parent.name == 'payload' else p.parent
                    candidate = slice_dir / 'metadata.json'
                    if candidate.exists():
                        slice_meta_path = str(candidate.resolve())
                except Exception:
                    slice_meta_path = None

            # Load slice-level metadata if available
            slice_level_meta = {}
            if slice_meta_path and os.path.exists(slice_meta_path):
                try:
                    with open(slice_meta_path, 'r') as sf:
                        slice_level_meta = json.load(sf)
                except Exception as e:
                    logger.warning(f"Failed to read slice metadata at {slice_meta_path}: {e}")

            # Extract IO shapes and deps
            io_meta = slice_level_meta.get('io', {}) if isinstance(slice_level_meta, dict) else {}
            deps_meta = slice_level_meta.get('deps', {}) if isinstance(slice_level_meta, dict) else {}
            input_shape = io_meta.get('input_shape') or segment.get('shape', {}).get('tensor_shape', {}).get('input', ["batch_size", "unknown"])
            output_shape = io_meta.get('output_shape') or segment.get('shape', {}).get('tensor_shape', {}).get('output', ["batch_size", "unknown"])

            # Extract EZKL compilation paths
            comp = slice_level_meta.get('compilation', {}).get('ezkl', {}) if isinstance(slice_level_meta, dict) else {}
            files = comp.get('files', {}) if isinstance(comp, dict) else {}
            compiled_flag = comp.get('compiled', False)
            compiled_circuit_path = files.get('compiled_circuit')
            settings_path = files.get('settings')
            pk_path = files.get('pk_key')
            vk_path = files.get('vk_key')

            # Check existence
            circuit_exists = bool(compiled_circuit_path) and os.path.exists(compiled_circuit_path) and bool(settings_path) and os.path.exists(settings_path)
            keys_exist = bool(pk_path) and os.path.exists(pk_path) and bool(vk_path) and os.path.exists(vk_path)

            # Determine circuit size
            circuit_size = 0
            if circuit_exists:
                try:
                    circuit_size = Path(compiled_circuit_path).stat().st_size
                except Exception:
                    circuit_size = 0

            use_circuit = compiled_flag and circuit_exists and keys_exist and circuit_size <= self.size_limit

            # Build slice metadata
            slice_metadata = {
                "path": onnx_slice_path,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "ezkl_compatible": True,
                "ezkl": use_circuit,
                "circuit_size": circuit_size,
                "dependencies": segment.get('dependencies') or deps_meta,
                "parameters": segment.get('parameters', 0)
            }

            # Add circuit paths
            if circuit_exists:
                slice_metadata.update({
                    "circuit_path": compiled_circuit_path,
                    "settings_path": settings_path
                })
                if keys_exist:
                    slice_metadata.update({
                        "vk_path": vk_path,
                        "pk_path": pk_path
                    })

            # For diagnostics, include slice metadata path
            if slice_meta_path:
                slice_metadata["slice_metadata_path"] = slice_meta_path

            slices[slice_key] = slice_metadata

        return slices

    @staticmethod
    def _build_execution_chain(slices: dict):
        """
        Build the execution chain with proper node connections and fallback mapping,
        using new slice_* ids and per-slice metadata.
        """
        # Order slices by numeric index extracted from key 'slice_#'
        ordered_keys = sorted(slices.keys(), key=lambda k: int(str(k).split('_')[-1])) if slices else []

        execution_chain = {
            "head": ordered_keys[0] if ordered_keys else None,
            "nodes": {},
            "fallback_map": {}
        }

        for i, slice_key in enumerate(ordered_keys):
            meta = slices.get(slice_key, {})
            circuit_path = meta.get('circuit_path')
            onnx_path = meta.get('path')
            has_circuit = bool(circuit_path) and os.path.exists(circuit_path)
            has_keys = bool(meta.get('pk_path')) and os.path.exists(meta.get('pk_path')) and bool(meta.get('vk_path')) and os.path.exists(meta.get('vk_path'))
            use_circuit = bool(meta.get('ezkl')) and has_circuit and has_keys

            next_slice = ordered_keys[i + 1] if i < len(ordered_keys) - 1 else None
            execution_chain["nodes"][slice_key] = {
                "segment_id": slice_key,
                "primary": circuit_path if use_circuit else onnx_path,
                "fallback": onnx_path,
                "use_circuit": use_circuit,
                "next": next_slice,
                "circuit_path": circuit_path if has_circuit else None,
                "onnx_path": onnx_path
            }

            if has_circuit and onnx_path:
                execution_chain["fallback_map"][circuit_path] = onnx_path
            elif onnx_path:
                execution_chain["fallback_map"][slice_key] = onnx_path

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
