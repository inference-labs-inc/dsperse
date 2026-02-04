import pytest
import torch
from pathlib import Path
from dsperse.src.analyzers.schema import (
    Dependencies, RunSliceMetadata, TilingInfo, TileInfo, ChannelSplitInfo, ChannelGroupInfo
)
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.run.tile_executor import TileExecutor
from dsperse.src.run.channel_split_executor import ChannelSplitExecutor


class TestPrepareSliceInput:
    def _make_meta(self, filtered_inputs):
        return RunSliceMetadata(
            path="slice_0/payload/slice_0.onnx",
            dependencies=Dependencies(filtered_inputs=filtered_inputs),
        )

    def test_valid_lookup(self, tmp_path):
        meta = self._make_meta(["activation_0"])
        tensor = torch.randn(1, 3, 4, 4)
        cache = {"activation_0": tensor}
        result = RunnerUtils.prepare_slice_input(meta, cache, torch.zeros(1), tmp_path / "in.json", skip_write=True)
        assert torch.equal(result, tensor)

    def test_missing_tensor_raises(self, tmp_path):
        meta = self._make_meta(["activation_0"])
        cache = {"wrong_name": torch.randn(1)}
        with pytest.raises(ValueError, match="activation_0.*not found"):
            RunnerUtils.prepare_slice_input(meta, cache, torch.zeros(1), tmp_path / "in.json", skip_write=True)

    def test_error_message_shows_available_keys(self, tmp_path):
        meta = self._make_meta(["missing_tensor"])
        cache = {"tensor_a": torch.randn(1), "tensor_b": torch.randn(1)}
        with pytest.raises(ValueError, match="tensor_a"):
            RunnerUtils.prepare_slice_input(meta, cache, torch.zeros(1), tmp_path / "in.json", skip_write=True)

    def test_first_slice_with_seeded_cache(self, tmp_path):
        meta = self._make_meta(["model_input"])
        model_input = torch.randn(1, 3, 32, 32)
        cache = {"model_input": model_input}
        result = RunnerUtils.prepare_slice_input(meta, cache, model_input, tmp_path / "in.json", skip_write=True)
        assert torch.equal(result, model_input)


class TestTileExecutorValidation:
    def test_missing_input_tensor_raises(self):
        cache = {}
        executor = TileExecutor(Path("/tmp"), cache)
        tiling = TilingInfo(
            slice_idx=0, tile_size=4, num_tiles=4, tiles_y=2, tiles_x=2,
            halo=(1, 1), out_tile=(2, 2), stride=(1, 1), c_in=3, c_out=8,
            input_name="conv_input", output_name="conv_output",
            tile=TileInfo(path="tile.onnx", conv_out=(2, 2)),
        )
        meta = RunSliceMetadata(path="slice_0.onnx")
        with pytest.raises(ValueError, match="conv_input"):
            executor.get_input_tensor("slice_0", tiling, meta)

    def test_missing_tile_output_raises(self):
        cache = {}
        executor = TileExecutor(Path("/tmp"), cache)
        tiling = TilingInfo(
            slice_idx=0, tile_size=4, num_tiles=4, tiles_y=2, tiles_x=2,
            halo=(1, 1), out_tile=(2, 2), stride=(1, 1), c_in=3, c_out=8,
            input_name="in", output_name="out",
            tile=TileInfo(path="tile.onnx", conv_out=(2, 2)),
        )
        with pytest.raises(ValueError, match="tile_0_0_out"):
            executor.reconstruct_from_tiles("slice_0", tiling)


class TestChannelSplitExecutorValidation:
    def test_missing_input_tensor_raises(self):
        cache = {"other_tensor": torch.randn(1, 3, 4, 4)}
        executor = ChannelSplitExecutor(Path("/tmp"), cache)
        meta = RunSliceMetadata(
            path="slice_0.onnx",
            dependencies=Dependencies(filtered_inputs=["missing_input"]),
            channel_split=ChannelSplitInfo(
                num_groups=2, c_in=6, c_out=8, h=4, w=4,
                input_name="missing_input", output_name="out",
                groups=[
                    ChannelGroupInfo(group_idx=0, c_start=0, c_end=3, path="g0.onnx"),
                    ChannelGroupInfo(group_idx=1, c_start=3, c_end=6, path="g1.onnx"),
                ],
            ),
        )
        with pytest.raises(ValueError, match="missing_input.*not found"):
            executor.prepare_config(meta)

    def test_error_shows_available_keys(self):
        cache = {"tensor_x": torch.randn(1), "tensor_y": torch.randn(1)}
        executor = ChannelSplitExecutor(Path("/tmp"), cache)
        meta = RunSliceMetadata(
            path="slice_0.onnx",
            dependencies=Dependencies(filtered_inputs=["bad_name"]),
            channel_split=ChannelSplitInfo(
                num_groups=1, c_in=3, c_out=8, h=4, w=4,
                input_name="bad_name", output_name="out",
                groups=[ChannelGroupInfo(group_idx=0, c_start=0, c_end=3, path="g0.onnx")],
            ),
        )
        with pytest.raises(ValueError, match="tensor_x"):
            executor.prepare_config(meta)
