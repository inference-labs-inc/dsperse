"""
Slicer module — slices ONNX models and optionally converts the output format.
"""

import logging
from typing import Optional

from dsperse.src.slice.onnx_slicer import OnnxSlicer
from dsperse.src.slice.utils.converter import Converter


logger = logging.getLogger(__name__)


class Slicer:
    def __init__(self, model_path: str, save_path: Optional[str] = None):
        logger.info(f"Creating ONNX slicer for model: {model_path}")
        self._impl = OnnxSlicer(model_path, save_path)

    def slice_model(self, output_path: str, output_type: str = "dirs", tile_size: Optional[int] = None):
        if not output_path:
            raise ValueError("output_path must be provided for slicing")

        logger.info(f"Slicing model to output path: {output_path}")
        result = self._impl.slice_model(output_path=output_path, tile_size=tile_size)

        if output_type != "dirs":
            Converter.convert(output_path, output_type)

        return result
