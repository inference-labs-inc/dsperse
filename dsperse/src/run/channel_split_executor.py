import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from dsperse.src.analyzers.schema import (
    Backend, ChannelGroupInfo, ChannelSplitInfo, ExecutionMethod, RunSliceMetadata
)
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class ChannelSplitExecutor:
    """Handles channel-split slice execution."""

    def __init__(self, slices_path: Path, tensor_cache: dict, jstprove_runner=None, ezkl_runner=None):
        self.slices_path = slices_path
        self.tensor_cache = tensor_cache
        self.jstprove_runner = jstprove_runner
        self.ezkl_runner = ezkl_runner

    def prepare_config(self, meta: RunSliceMetadata) -> tuple:
        channel_split = meta.channel_split
        if not channel_split:
            return None, None, None

        input_name = channel_split.input_name or (
            meta.dependencies.filtered_inputs[0] if meta.dependencies.filtered_inputs else "input"
        )
        input_tensor = self.tensor_cache.get(input_name)
        if input_tensor is None:
            return None, None, None

        target_shape = meta.get_target_shape()
        output_shape = (
            tuple(target_shape) if target_shape
            else (1, channel_split.c_out, channel_split.h, channel_split.w)
        )
        return input_tensor, output_shape, input_name

    def execute_group(
        self, group: ChannelGroupInfo, group_input: torch.Tensor, run_dir: Path,
        output_shape: tuple, backend: str = None
    ) -> tuple[torch.Tensor, str]:
        forced = backend

        has_jst = bool(group.jstprove_circuit_path) and self.jstprove_runner is not None
        has_ezkl = bool(group.ezkl_circuit_path) and (group.vk_path or group.ezkl_vk_path)

        if isinstance(group_input, torch.Tensor):
            input_arr = group_input.detach().cpu().numpy().astype(np.float32)
        else:
            input_arr = np.asarray(group_input, dtype=np.float32)

        group_dir = run_dir / f"channel_group_{group.group_idx}"
        group_dir.mkdir(parents=True, exist_ok=True)
        in_file = group_dir / "input.json"
        out_file = group_dir / "output.json"
        Utils.write_input(torch.from_numpy(input_arr), in_file)

        def _extract_and_reshape(result):
            tensor = self._extract_output_tensor(result)
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.dim() < 4:
                expected = np.prod(output_shape)
                if tensor.numel() == expected:
                    return tensor.reshape(output_shape)
            return tensor

        if forced == Backend.ONNX or (not has_jst and not has_ezkl):
            group_onnx_path = self._resolve_path(group.path)
            if not group_onnx_path:
                raise ValueError(f"Cannot resolve ONNX path for channel group {group.group_idx}: {group.path}")
            session = ort.InferenceSession(str(group_onnx_path))
            outputs = session.run(None, {session.get_inputs()[0].name: input_arr})
            return torch.from_numpy(outputs[0]), ExecutionMethod.ONNX_ONLY

        if (forced == Backend.JSTPROVE and has_jst) or (has_jst and forced != Backend.EZKL):
            circuit_path = self._resolve_path(group.jstprove_circuit_path)
            success, result = self.jstprove_runner.generate_witness(str(in_file), circuit_path, str(out_file))
            if success:
                tensor = _extract_and_reshape(result)
                if tensor is not None:
                    return tensor, ExecutionMethod.JSTPROVE_GEN_WITNESS
            logger.warning(f"JSTprove failed for group {group.group_idx}, falling back")

        if (forced == Backend.EZKL and has_ezkl) or (has_ezkl and forced != Backend.JSTPROVE):
            circuit_path = self._resolve_path(group.ezkl_circuit_path)
            vk_path = self._resolve_path(group.ezkl_vk_path or group.vk_path)
            settings_path = self._resolve_path(group.ezkl_settings_path or group.settings_path)
            ezkl_in = self._flatten_input_for_ezkl(in_file)
            success, result = self.ezkl_runner.generate_witness(str(ezkl_in), circuit_path, str(out_file), vk_path, settings_path)
            if success:
                tensor = _extract_and_reshape(result)
                if tensor is not None:
                    return tensor, ExecutionMethod.EZKL_GEN_WITNESS
            logger.warning(f"EZKL failed for group {group.group_idx}, falling back")

        group_onnx_path = self._resolve_path(group.path)
        if not group_onnx_path:
            raise ValueError(f"Cannot resolve ONNX path for channel group {group.group_idx}: {group.path}")
        session = ort.InferenceSession(str(group_onnx_path))
        outputs = session.run(None, {session.get_inputs()[0].name: input_arr})
        return torch.from_numpy(outputs[0]), ExecutionMethod.ONNX_ONLY

    def apply_bias(self, summed: torch.Tensor, channel_split: ChannelSplitInfo) -> torch.Tensor:
        if channel_split.bias_path:
            bias_path = self._resolve_path(channel_split.bias_path)
            if Path(bias_path).exists():
                bias_tensor = torch.from_numpy(np.load(bias_path)).reshape(1, -1, 1, 1)
                summed = summed + bias_tensor
        return summed

    def _resolve_path(self, p: str) -> str | None:
        from dsperse.src.run.utils.runner_utils import RunnerUtils
        return RunnerUtils.resolve_relative_path(p, self.slices_path)

    def _flatten_input_for_ezkl(self, in_file: Path) -> Path:
        from dsperse.src.run.utils.runner_utils import RunnerUtils
        return RunnerUtils.flatten_input_for_ezkl(in_file)

    @staticmethod
    def _extract_output_tensor(result):
        if result is None:
            return None
        if isinstance(result, dict):
            for key in ['output', 'logits', 'output_tensor']:
                if key in result and result[key] is not None:
                    return result[key]
            return None
        return result
