import json
import logging
import os
import onnx
import numpy as np
import torch
from onnx import numpy_helper
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

JSTPROVE_SUPPORTED_OPS = {
    "Add", "Clip", "BatchNormalization", "Div", "Sub",
    "Mul", "Constant", "Flatten", "Gemm",
    "MaxPool", "Max", "Min", "Relu", "Reshape",
    "Conv",
}

class JSTproveUtils:
    """Utility class for JSTprove backend operations."""

    @staticmethod
    def add_zero_bias_to_conv(model: onnx.ModelProto) -> onnx.ModelProto:
        """Add zero bias to Conv nodes that don't have one (JSTprove requires 3 inputs)."""
        for node in model.graph.node:
            if node.op_type == "Conv" and len(node.input) == 2:
                weight_name = node.input[1]
                weight_init = next((i for i in model.graph.initializer if i.name == weight_name), None)
                if weight_init is None:
                    continue

                weight_arr = numpy_helper.to_array(weight_init)
                out_channels = weight_arr.shape[0]

                node_id = node.name or node.output[0] if node.output else weight_name
                bias_name = f"{node_id}_zero_bias"
                zero_bias = np.zeros(out_channels, dtype=weight_arr.dtype)
                bias_init = numpy_helper.from_array(zero_bias, name=bias_name)
                model.graph.initializer.append(bias_init)
                node.input.append(bias_name)

        return model

    @staticmethod
    def promote_initializers_to_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
        """Promote ONNX initializers to graph inputs so the Rust backend can read their shapes for WAI."""
        existing_input_names = {inp.name for inp in model.graph.input}
        for init in model.graph.initializer:
            if init.name in existing_input_names:
                continue
            value_info = onnx.helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
            model.graph.input.append(value_info)
        return model

    @staticmethod
    def is_compatible(model_path: Union[str, Path]) -> Tuple[bool, set]:
        """Check if an ONNX model contains only JSTprove-supported operations."""
        model_path = Path(model_path)
        if not model_path.exists():
            return False, {"FILE_NOT_FOUND"}

        try:
            model = onnx.load(str(model_path))
            ops = {node.op_type for node in model.graph.node}
            unsupported = ops - JSTPROVE_SUPPORTED_OPS
            if unsupported:
                return False, unsupported
            return True, set()
        except Exception as e:
            logger.warning(f"Failed to check JSTprove compatibility: {e}")
            return False, {"LOAD_ERROR"}

    @staticmethod
    def initialize_circuitization_artifacts(model_path: Path, output_path: Path, input_file_path: Optional[str]) -> Dict[str, Any]:
        """Initialize paths and metadata for JSTprove circuitization."""
        model_name = model_path.stem
        circuit_path = output_path / f"{model_name}_circuit.txt"
        quantized_model_path = output_path / f"{model_name}_circuit_quantized_model.onnx"
        witness_solver_path = output_path / f"{model_name}_circuit_witness_solver.txt"
        settings_path = output_path / f"{model_name}_settings.json"

        return {
            "paths": {
                "circuit": circuit_path,
                "quantized_model": quantized_model_path,
                "witness_solver": witness_solver_path,
                "settings": settings_path,
            },
            "data": {
                "compiled": str(circuit_path),
                "circuit": str(circuit_path),
                "quantized_model": str(quantized_model_path),
                "witness_solver": str(witness_solver_path),
                "calibration": input_file_path,
                "settings": str(settings_path),
                "vk_key": None,
                "pk_key": None,
            }
        }

    @staticmethod
    def create_dummy_settings(model_path: Path, circuit_path: Path, output_path: Path) -> dict:
        """Create a dummy settings file for dsperse compatibility."""
        return {
            "backend": "jstprove",
            "model_path": str(model_path),
            "circuit_path": str(circuit_path),
            "compiled_at": str(output_path),
            "note": "This is a dummy settings file for dsperse compatibility. JSTprove handles settings internally."
        }

    @staticmethod
    def convert_to_logits(data: Any) -> torch.Tensor:
        """Helper to convert data to logits tensor with batch dimension."""
        logits = torch.tensor(data)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits
