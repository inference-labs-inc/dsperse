"""
Generates execution chain metadata for EzKL circuit and ONNX slice inference
with proper fallback mapping and security calculation.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import onnx

from src.utils.onnx_analyzer import OnnxAnalyzer
from src.slicers.onnx_slicer import OnnxSlicer
from src.utils.onnx_utils import OnnxUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RunnerMetadata:
    def __init__(self, slice_output_dir: str, onnx_model_path: str):
        """
        Args:
            slice_output_dir: Path to the directory containing ONNX slices and metadata.json
            onnx_model_path: Path to the ONNX model file
        """
        self.slice_output_dir = Path(slice_output_dir).resolve()
        self.onnx_path = Path(onnx_model_path).resolve()
        self.model_name = self.onnx_path.parent.name
        self.size_limit = 100 * 1024 * 1024  # 100MB
        self.slices_metadata_path = self.slice_output_dir / "metadata.json"
        self.run_metadata_path = self.slice_output_dir / "run_metadata.json"
        # EzKL circuits are in the same segment directories as ONNX slices
        self.ezkl_slices_dir = self.slice_output_dir

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.onnx_path}")
        if not self.slice_output_dir.exists():
            raise FileNotFoundError(f"Slice output directory not found: {self.slice_output_dir}")
        try:
            self.onnx_model = onnx.load(str(self.onnx_path))
            logger.info(f"Loaded ONNX model: {self.onnx_path}")
        except Exception as e:
            raise ValueError(f"Invalid ONNX model: {e}")

    def _ensure_analyzer_metadata(self) -> None:
        metadata_path = self.onnx_path.parent / "analysis" / "model_metadata.json"
        if not metadata_path.exists():
            logger.warning("ONNX analyzer metadata missing - generating...")
            analyzer = OnnxAnalyzer(str(self.onnx_path))
            analyzer.analyze()
            logger.info("ONNX analyzer completed successfully")

    def _ensure_slices_metadata(self) -> None:
        if not self.slices_metadata_path.exists():
            logger.warning("ONNX slices missing - generating...")
            slicer = OnnxSlicer(str(self.onnx_path))
            slicer.slice_model()
            logger.info("ONNX slicing completed successfully")

    def generate(self) -> str:
        """
        Generate runner metadata and save to run_metadata.json.
        Returns:
            Path to generated run_metadata.json
        """
        self._ensure_analyzer_metadata()
        self._ensure_slices_metadata()
        logger.info(f"Generating metadata for {self.model_name}...")
        onnx_metadata = self._load_onnx_metadata()
        slices_metadata = self._load_slices_metadata()
        runner_metadata = self._generate_metadata(onnx_metadata, slices_metadata)
        self._validate_metadata(runner_metadata)
        OnnxUtils.save_metadata_file(runner_metadata, str(self.run_metadata_path))
        logger.info(f"Metadata generated successfully: {self.run_metadata_path}")
        logger.info(f"Security level: {runner_metadata.get('overall_security', 0):.1f}%")
        return str(self.run_metadata_path)

    def _load_onnx_metadata(self) -> Dict[str, Any]:
        metadata_path = self.onnx_path.parent / "analysis" / "model_metadata.json"
        if not metadata_path.exists():
            metadata_path = self.slice_output_dir / "model_metadata.json"
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ONNX metadata: {e}")
            logger.warning("Using fallback metadata")
            # Return minimal fallback metadata
            return {
                "input_shape": [["batch_size", 3, 32, 32]],
                "output_shapes": [["batch_size", 10]],
                "node_count": 12,
                "nodes": {},
                "total_nodes": 12
            }

    def _load_slices_metadata(self) -> Dict[str, Any]:
        try:
            with open(self.slices_metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load slices metadata: {e}")
            raise

    def _generate_metadata(self, onnx_metadata: Dict[str, Any], slices_metadata: Dict[str, Any]) -> Dict[str, Any]:
        slices, execution_chain, circuit_slices = self._process_slices(slices_metadata)
        overall_security = self._calculate_security(slices)
        analyzer_available = bool(onnx_metadata and onnx_metadata.get("node_count", 0) > 0 and "nodes" in onnx_metadata)
        reconstruct_available = bool(slices_metadata and slices_metadata.get('segments'))
        total_parameters = sum(node_info.get("parameters", 0) for node_info in onnx_metadata.get("nodes", {}).values())
        return {
            "model_name": self.model_name,
            "model_type": "ONNX",
            "original_model_path": str(self.onnx_path),
            "Analyzer": analyzer_available,
            "Reconstruct": reconstruct_available,
            "match": self._check_slices_match(slices_metadata),
            "precircuit": self._get_precircuit_status(),
            "num_slices": len(slices_metadata.get('segments', [])),
            "ezkl_compatible": "Yes" if any(s.get("ezkl", False) for s in slices.values()) else "No",
            "overall_security": overall_security,
            "size_limit": self.size_limit,
            "total_parameters": total_parameters,
            "input_shape": onnx_metadata.get("input_shape", onnx_metadata.get("input_shapes", [])),
            "output_shape": onnx_metadata.get("output_shapes", onnx_metadata.get("output_shape", [])),
            "node_count": onnx_metadata.get("node_count", onnx_metadata.get("total_nodes", 0)),
            "io_paths": {
                "input_path": str(self.onnx_path.parent / "input.json"),
                "output_path": str(self.onnx_path.parent / f"{self.model_name}_inference_output.json"),
                "model_path": str(self.onnx_path)
            },
            "slices": slices,
            "execution_chain": execution_chain,
            "circuit_slices": circuit_slices,
            "workflow_ready": True,
            "analysis_metadata_available": True,
            "slice_metadata_available": True
        }

    def _process_slices(self, slices_metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, bool]]:
        segments = slices_metadata.get('segments', [])
        if not segments:
            logger.warning("No segments found in slices metadata")
        slices = {}
        circuit_slices = {}
        execution_chain = {
            "head": "slice_0" if segments else None,
            "nodes": {},
            "fallback_map": {}
        }
        for segment in segments:
            segment_idx = segment['index']
            slice_key = f"slice_{segment_idx}"
            ezkl_segment_dir = self.slice_output_dir / f"segment_{segment_idx}"
            
            compiled_circuit_path = ezkl_segment_dir / f"segment_{segment_idx}_model.compiled"
            settings_path = ezkl_segment_dir / f"segment_{segment_idx}_settings.json"
            pk_path = ezkl_segment_dir / f"segment_{segment_idx}_pk.key"
            vk_path = ezkl_segment_dir / f"segment_{segment_idx}_vk.key"
            circuit_exists = compiled_circuit_path.exists() and settings_path.exists()
            keys_exist = pk_path.exists() and vk_path.exists()
            
            circuit_size = compiled_circuit_path.stat().st_size if circuit_exists else 0
            use_circuit = circuit_exists and keys_exist and circuit_size <= self.size_limit
            onnx_slice_path = segment.get('path', '')
            if not onnx_slice_path:
                logger.warning(f"No ONNX slice path for segment {segment_idx}")
            slice_metadata = {
                "path": onnx_slice_path,
                "input_shape": segment.get('shape', {}).get('input', ["batch_size", "unknown"]),
                "output_shape": segment.get('shape', {}).get('output', ["batch_size", "unknown"]),
                "ezkl_compatible": True,
                "ezkl": use_circuit,
                "circuit_size": circuit_size,
                "dependencies": segment.get('dependencies', {}),
                "parameters": segment.get('parameters', 0)
            }
            if circuit_exists:
                slice_metadata.update({
                    "circuit_path": str(compiled_circuit_path),
                    "proof_path": str(ezkl_segment_dir / f"segment_{segment_idx}_proof.json"),
                    "witness_path": str(ezkl_segment_dir / f"segment_{segment_idx}_witness.json"),
                    "settings_path": str(settings_path)
                })
                if keys_exist:
                    slice_metadata.update({
                        "vk_path": str(vk_path),
                        "pk_path": str(pk_path)
                    })
            slices[slice_key] = slice_metadata
            circuit_slices[slice_key] = use_circuit
            next_slice = f"slice_{segment_idx + 1}" if segment_idx < len(segments) - 1 else None
            execution_chain["nodes"][slice_key] = {
                "slice_id": slice_key,
                "primary": str(compiled_circuit_path) if circuit_exists else onnx_slice_path,
                "fallback": onnx_slice_path,
                "use_circuit": use_circuit,
                "next": next_slice,
                "circuit_path": str(compiled_circuit_path) if circuit_exists else None,
                "onnx_path": onnx_slice_path
            }
            if circuit_exists and onnx_slice_path:
                execution_chain["fallback_map"][str(compiled_circuit_path)] = onnx_slice_path
            elif onnx_slice_path:
                execution_chain["fallback_map"][slice_key] = onnx_slice_path
        return slices, execution_chain, circuit_slices

    def _calculate_security(self, slices: Dict[str, Any]) -> float:
        if not slices:
            return 0.0
        total_slices = len(slices)
        circuit_slices = sum(1 for slice_data in slices.values() if slice_data.get("ezkl", False))
        return round((circuit_slices / total_slices) * 100, 1)

    def _check_slices_match(self, slices_metadata: Dict[str, Any]) -> bool:
        segments = slices_metadata.get('segments', [])
        for segment in segments:
            segment_idx = segment['index']
            ezkl_segment_dir = self.slice_output_dir / f"segment_{segment_idx}"
            if not ezkl_segment_dir.exists():
                return False
        return True

    def _get_precircuit_status(self) -> str:
        if not self.slice_output_dir.exists():
            return "precircuit needs building - run onnx_slicer.py first"
        
        # Check if any EzKL circuits exist
        has_circuits = False
        for segment_dir in self.slice_output_dir.glob("segment_*"):
            if (segment_dir / f"{segment_dir.name}_model.compiled").exists():
                has_circuits = True
                break
        
        if has_circuits:
            return "circuits compiled and ready"
        else:
            return "precircuit built - ready for EzKL"

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        required_fields = [
            "model_name", "execution_chain", "slices", "circuit_slices",
            "overall_security", "size_limit"
        ]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field: {field}")
        execution_chain = metadata["execution_chain"]
        if not isinstance(execution_chain, dict):
            raise ValueError("execution_chain must be a dictionary")
        required_chain_fields = ["head", "nodes", "fallback_map"]
        for field in required_chain_fields:
            if field not in execution_chain:
                raise ValueError(f"Missing execution chain field: {field}")
        security = metadata["overall_security"]
        if not isinstance(security, (int, float)) or security < 0 or security > 100:
            raise ValueError(f"Invalid security percentage: {security}")
        logger.debug("Metadata validation passed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate runner metadata for ONNX slices.")
    parser.add_argument("slice_output_dir", help="Path to the output directory containing ONNX slices and metadata.json")
    parser.add_argument("onnx_model_path", help="Path to the ONNX model file")
    args = parser.parse_args()

    try:
        metadata_path = RunnerMetadata(args.slice_output_dir, args.onnx_model_path).generate()
        print(f"Success: {metadata_path}")
    except Exception as e:
        print(f" Error: {e}")
        logger.exception("Detailed error information:") 