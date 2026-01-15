import shutil
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from pathlib import Path
from onnx import helper, TensorProto, numpy_helper

from dsperse.src.slice.autotiler import (
    compute_halo,
    compute_min_tile_size,
    find_tile_size,
    is_tileable,
    autotile_slice,
    ELEMENTWISE_OPS,
)


class TestComputeHalo:
    def test_3x3_kernel_no_dilation(self):
        assert compute_halo([3, 3], [1, 1]) == (1, 1)

    def test_5x5_kernel_no_dilation(self):
        assert compute_halo([5, 5], [1, 1]) == (2, 2)

    def test_7x7_kernel_no_dilation(self):
        assert compute_halo([7, 7], [1, 1]) == (3, 3)

    def test_3x3_kernel_dilation_2(self):
        # effective_k = (3-1)*2 + 1 = 5, halo = 2
        assert compute_halo([3, 3], [2, 2]) == (2, 2)

    def test_3x3_kernel_dilation_3(self):
        # effective_k = (3-1)*3 + 1 = 7, halo = 3
        assert compute_halo([3, 3], [3, 3]) == (3, 3)

    def test_asymmetric_kernel(self):
        # 3x5 kernel, dilation 1
        assert compute_halo([3, 5], [1, 1]) == (1, 2)

    def test_asymmetric_dilation(self):
        # 3x3 kernel, dilation [1, 2]
        # effective_kh = 3, effective_kw = 5
        assert compute_halo([3, 3], [1, 2]) == (1, 2)


class TestComputeMinTileSize:
    def test_3x3_kernel_no_dilation(self):
        # effective_k = 3, min_tile = 4
        assert compute_min_tile_size([3, 3], [1, 1]) == 4

    def test_5x5_kernel_no_dilation(self):
        # effective_k = 5, min_tile = 6
        assert compute_min_tile_size([5, 5], [1, 1]) == 6

    def test_7x7_kernel_no_dilation(self):
        # effective_k = 7, min_tile = 8
        assert compute_min_tile_size([7, 7], [1, 1]) == 8

    def test_3x3_kernel_dilation_2(self):
        # effective_k = 5, min_tile = 6
        assert compute_min_tile_size([3, 3], [2, 2]) == 6

    def test_asymmetric_takes_max(self):
        # 3x7 kernel -> effective 3x7, min = max(3,7)+1 = 8
        assert compute_min_tile_size([3, 7], [1, 1]) == 8


class TestFindTileSize:
    def test_exact_divisor(self):
        # 64 divides 128 evenly
        assert find_tile_size(128, 64, min_tile=4) == 64

    def test_finds_largest_divisor(self):
        # 100 doesn't divide 128, but 64 does
        assert find_tile_size(128, 100, min_tile=4) == 64

    def test_respects_min_tile(self):
        # 8 divides 64, but min_tile=10 means we need 16 or 32
        result = find_tile_size(64, 8, min_tile=10)
        assert result is None or result >= 10

    def test_returns_none_when_no_valid_divisor(self):
        # Prime number spatial dim with high min_tile
        assert find_tile_size(17, 16, min_tile=10) is None

    def test_target_smaller_than_min_tile(self):
        assert find_tile_size(128, 4, min_tile=8) is None

    def test_spatial_dim_smaller_than_target(self):
        assert find_tile_size(32, 64, min_tile=4) is None


class TestIsTileable:
    def _make_conv_model(self, extra_ops=None):
        """Create a simple Conv model, optionally with extra ops."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 30, 30])

        W = helper.make_tensor("W", TensorProto.FLOAT, [8, 3, 3, 3],
                               np.random.randn(8, 3, 3, 3).astype(np.float32).flatten().tolist())

        nodes = [helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3])]

        if extra_ops:
            prev_out = "conv_out"
            for i, op in enumerate(extra_ops):
                out_name = f"op_{i}_out" if i < len(extra_ops) - 1 else "Y"
                nodes.append(helper.make_node(op, [prev_out], [out_name]))
                prev_out = out_name
        else:
            nodes[0].output[0] = "Y"

        graph = helper.make_graph(nodes, "test", [X], [Y], [W])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    def test_conv_only_is_tileable(self):
        model = self._make_conv_model()
        assert is_tileable(model)

    def test_conv_with_relu_is_tileable(self):
        model = self._make_conv_model(extra_ops=["Relu"])
        assert is_tileable(model)

    def test_conv_with_sigmoid_is_tileable(self):
        model = self._make_conv_model(extra_ops=["Sigmoid"])
        assert is_tileable(model)

    def test_conv_with_matmul_not_tileable(self):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 30, 30])
        W = helper.make_tensor("W", TensorProto.FLOAT, [8, 3, 3, 3],
                               np.random.randn(8, 3, 3, 3).astype(np.float32).flatten().tolist())
        nodes = [
            helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3]),
            helper.make_node("Flatten", ["conv_out"], ["flat"], axis=1),
        ]
        graph = helper.make_graph(nodes, "test", [X],
                                  [helper.make_tensor_value_info("flat", TensorProto.FLOAT, None)], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        assert not is_tileable(model)

    def test_no_conv_not_tileable(self):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 32, 32])
        nodes = [helper.make_node("Relu", ["X"], ["Y"])]
        graph = helper.make_graph(nodes, "test", [X], [Y], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        assert not is_tileable(model)

    def test_multiple_inputs_not_tileable(self):
        X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 3, 32, 32])
        X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 32, 32])
        nodes = [helper.make_node("Add", ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(nodes, "test", [X1, X2], [Y], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        assert not is_tileable(model)


class TestTiledVsNonTiledParity:
    """Test that tiled convolution produces identical output to non-tiled."""

    def _create_conv_model(self, c_in, c_out, spatial, kernel, stride, padding, dilation):
        """Create a Conv + Relu model."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])

        np.random.seed(42)
        w_data = np.random.randn(c_out, c_in, kernel, kernel).astype(np.float32)
        b_data = np.random.randn(c_out).astype(np.float32)

        W = numpy_helper.from_array(w_data, "W")
        B = numpy_helper.from_array(b_data, "B")

        conv = helper.make_node(
            "Conv", ["X", "W", "B"], ["conv_out"],
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[padding, padding, padding, padding],
            dilations=[dilation, dilation]
        )
        relu = helper.make_node("Relu", ["conv_out"], ["Y"])

        effective_k = (kernel - 1) * dilation + 1
        out_spatial = (spatial + 2 * padding - effective_k) // stride + 1
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, out_spatial, out_spatial])

        graph = helper.make_graph([conv, relu], "conv_relu", [X], [Y], [W, B])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model

    def _run_onnx(self, model_path, input_data):
        """Run ONNX model and return output."""
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        return sess.run(None, {input_name: input_data})[0]

    def _run_tiled(self, tiled_info, input_data, tiled_dir):
        """Run the split -> single tile N times -> concat pipeline manually."""
        split_path = tiled_info["split"]["path"]
        split_sess = ort.InferenceSession(split_path, providers=["CPUExecutionProvider"])
        split_input_name = split_sess.get_inputs()[0].name
        tile_inputs = split_sess.run(None, {split_input_name: input_data})

        tile_info = tiled_info["tile"]
        tile_sess = ort.InferenceSession(tile_info["path"], providers=["CPUExecutionProvider"])
        tile_input_name = tile_sess.get_inputs()[0].name

        tile_outputs = []
        for i in range(tiled_info["num_tiles"]):
            tile_out = tile_sess.run(None, {tile_input_name: tile_inputs[i]})[0]
            tile_outputs.append(tile_out)

        concat_path = tiled_info["concat"]["path"]
        concat_sess = ort.InferenceSession(concat_path, providers=["CPUExecutionProvider"])
        concat_inputs = {f"tile_{tiled_info['slice_idx']}_{i}_out": tile_outputs[i]
                         for i in range(len(tile_outputs))}
        return concat_sess.run(None, concat_inputs)[0]

    @pytest.mark.parametrize("kernel,stride,padding,dilation", [
        (3, 1, 1, 1),  # Standard 3x3 with same padding
        (3, 2, 1, 1),  # 3x3 with stride 2
        (5, 1, 2, 1),  # 5x5 kernel
        (3, 1, 2, 2),  # 3x3 with dilation 2
    ])
    def test_parity(self, kernel, stride, padding, dilation, tmp_path):
        c_in, c_out, spatial = 3, 8, 64
        tile_size = 16

        model = self._create_conv_model(c_in, c_out, spatial, kernel, stride, padding, dilation)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        np.random.seed(123)
        input_data = np.random.randn(1, c_in, spatial, spatial).astype(np.float32)

        expected = self._run_onnx(model_path, input_data)

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()
        tiled_info = autotile_slice(0, model_path, tile_size, tiled_dir)

        if tiled_info is None:
            pytest.skip("Model not tileable with given parameters")

        actual = self._run_tiled(tiled_info, input_data, tiled_dir)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5,
                                   err_msg=f"Parity failed for kernel={kernel}, stride={stride}, "
                                           f"padding={padding}, dilation={dilation}")

    def test_parity_larger_spatial(self, tmp_path):
        """Test with larger spatial dimensions and more tiles."""
        c_in, c_out, spatial = 3, 16, 128
        kernel, stride, padding, dilation = 3, 2, 1, 1
        tile_size = 32

        model = self._create_conv_model(c_in, c_out, spatial, kernel, stride, padding, dilation)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        np.random.seed(456)
        input_data = np.random.randn(1, c_in, spatial, spatial).astype(np.float32)

        expected = self._run_onnx(model_path, input_data)

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()
        tiled_info = autotile_slice(0, model_path, tile_size, tiled_dir)

        assert tiled_info is not None, "Model should be tileable"
        assert tiled_info["num_tiles"] == 16, f"Expected 16 tiles (4x4), got {tiled_info['num_tiles']}"

        actual = self._run_tiled(tiled_info, input_data, tiled_dir)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class TestInvalidTileSize:
    """Test that invalid tile sizes are rejected gracefully."""

    def _create_conv_model(self, spatial, kernel=3, stride=1, padding=1, dilation=1):
        c_in, c_out = 3, 8
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])

        np.random.seed(42)
        w_data = np.random.randn(c_out, c_in, kernel, kernel).astype(np.float32)
        W = numpy_helper.from_array(w_data, "W")

        conv = helper.make_node(
            "Conv", ["X", "W"], ["Y"],
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[padding, padding, padding, padding],
            dilations=[dilation, dilation]
        )

        effective_k = (kernel - 1) * dilation + 1
        out_spatial = (spatial + 2 * padding - effective_k) // stride + 1
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, out_spatial, out_spatial])

        graph = helper.make_graph([conv], "conv", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model

    def test_tile_size_larger_than_spatial(self, tmp_path):
        """Tile size > spatial dimension should return None."""
        model = self._create_conv_model(spatial=32)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        result = autotile_slice(0, model_path, tile_size=64, output_dir=tiled_dir)
        assert result is None

    def test_tile_size_equal_to_spatial(self, tmp_path):
        """Tile size == spatial dimension should return None (no benefit to tiling)."""
        model = self._create_conv_model(spatial=32)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        result = autotile_slice(0, model_path, tile_size=32, output_dir=tiled_dir)
        assert result is None

    def test_tile_size_no_valid_divisor(self, tmp_path):
        """Tile size that can't find a valid divisor should return None."""
        model = self._create_conv_model(spatial=37)  # prime
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        result = autotile_slice(0, model_path, tile_size=16, output_dir=tiled_dir)
        assert result is None

    def test_tile_size_smaller_than_kernel(self, tmp_path):
        """Tile size smaller than kernel effective size should return None."""
        model = self._create_conv_model(spatial=64, kernel=7, padding=3)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        # 7x7 kernel needs min_tile of 8, tile_size=4 is too small
        result = autotile_slice(0, model_path, tile_size=4, output_dir=tiled_dir)
        assert result is None

    def test_tile_size_smaller_than_dilated_kernel(self, tmp_path):
        """Tile size smaller than dilated kernel should return None."""
        model = self._create_conv_model(spatial=64, kernel=3, dilation=3, padding=3)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        # 3x3 kernel with dilation=3 -> effective 7x7, needs min_tile of 8
        result = autotile_slice(0, model_path, tile_size=4, output_dir=tiled_dir)
        assert result is None

    def test_non_square_spatial_rejected(self, tmp_path):
        """Non-square spatial dimensions should return None."""
        c_in, c_out = 3, 8
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, 64, 32])

        np.random.seed(42)
        w_data = np.random.randn(c_out, c_in, 3, 3).astype(np.float32)
        W = numpy_helper.from_array(w_data, "W")

        conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, 64, 32])

        graph = helper.make_graph([conv], "conv", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        result = autotile_slice(0, model_path, tile_size=16, output_dir=tiled_dir)
        assert result is None

    def test_non_4d_input_rejected(self, tmp_path):
        """Non-4D input tensor should return None."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64, 64])  # 3D

        np.random.seed(42)
        w_data = np.random.randn(8, 64, 3).astype(np.float32)
        W = numpy_helper.from_array(w_data, "W")

        conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 62])

        graph = helper.make_graph([conv], "conv1d", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        result = autotile_slice(0, model_path, tile_size=16, output_dir=tiled_dir)
        assert result is None
