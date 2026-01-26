import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from pathlib import Path
from onnx import helper, TensorProto, numpy_helper

from dsperse.src.slice.autotiler import Autotiler, ELEMENTWISE_OPS


class TestComputeHaloSize:
    def test_3x3_kernel_no_dilation(self):
        assert Autotiler.compute_halo_size([3, 3], [1, 1]) == (1, 1)

    def test_5x5_kernel_no_dilation(self):
        assert Autotiler.compute_halo_size([5, 5], [1, 1]) == (2, 2)

    def test_7x7_kernel_no_dilation(self):
        assert Autotiler.compute_halo_size([7, 7], [1, 1]) == (3, 3)

    def test_3x3_kernel_dilation_2(self):
        assert Autotiler.compute_halo_size([3, 3], [2, 2]) == (2, 2)

    def test_3x3_kernel_dilation_3(self):
        assert Autotiler.compute_halo_size([3, 3], [3, 3]) == (3, 3)

    def test_asymmetric_kernel(self):
        assert Autotiler.compute_halo_size([3, 5], [1, 1]) == (1, 2)

    def test_asymmetric_dilation(self):
        assert Autotiler.compute_halo_size([3, 3], [1, 2]) == (1, 2)


class TestComputeMinSpatialTile:
    def test_3x3_kernel_no_dilation(self):
        assert Autotiler.compute_min_spatial_tile([3, 3], [1, 1]) == 4

    def test_5x5_kernel_no_dilation(self):
        assert Autotiler.compute_min_spatial_tile([5, 5], [1, 1]) == 6

    def test_7x7_kernel_no_dilation(self):
        assert Autotiler.compute_min_spatial_tile([7, 7], [1, 1]) == 8

    def test_3x3_kernel_dilation_2(self):
        assert Autotiler.compute_min_spatial_tile([3, 3], [2, 2]) == 6

    def test_asymmetric_takes_max(self):
        assert Autotiler.compute_min_spatial_tile([3, 7], [1, 1]) == 8


class TestFindOptimalTileSize:
    def test_exact_divisor(self):
        assert Autotiler.find_optimal_tile_size(128, 64, min_tile=4) == 64

    def test_finds_largest_divisor(self):
        assert Autotiler.find_optimal_tile_size(128, 100, min_tile=4) == 64

    def test_respects_min_tile(self):
        result = Autotiler.find_optimal_tile_size(64, 8, min_tile=10)
        assert result is None or result >= 10

    def test_returns_none_when_no_valid_divisor(self):
        assert Autotiler.find_optimal_tile_size(17, 16, min_tile=10) is None

    def test_target_smaller_than_min_tile(self):
        assert Autotiler.find_optimal_tile_size(128, 4, min_tile=8) is None

    def test_spatial_dim_smaller_than_target(self):
        assert Autotiler.find_optimal_tile_size(32, 64, min_tile=4) is None


class TestIsTileable:
    def _make_conv_model(self, extra_ops=None):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 32, 32])

        W = helper.make_tensor("W", TensorProto.FLOAT, [8, 3, 3, 3],
                               np.random.randn(8, 3, 3, 3).astype(np.float32).flatten().tolist())

        nodes = [helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])]

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
        assert Autotiler.is_tileable(model)

    def test_conv_with_relu_is_tileable(self):
        model = self._make_conv_model(extra_ops=["Relu"])
        assert Autotiler.is_tileable(model)

    def test_conv_with_sigmoid_is_tileable(self):
        model = self._make_conv_model(extra_ops=["Sigmoid"])
        assert Autotiler.is_tileable(model)

    def test_conv_with_flatten_not_tileable(self):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
        W = helper.make_tensor("W", TensorProto.FLOAT, [8, 3, 3, 3],
                               np.random.randn(8, 3, 3, 3).astype(np.float32).flatten().tolist())
        nodes = [
            helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3]),
            helper.make_node("Flatten", ["conv_out"], ["flat"], axis=1),
        ]
        graph = helper.make_graph(nodes, "test", [X],
                                  [helper.make_tensor_value_info("flat", TensorProto.FLOAT, None)], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        assert not Autotiler.is_tileable(model)

    def test_no_conv_not_tileable(self):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 32, 32])
        nodes = [helper.make_node("Relu", ["X"], ["Y"])]
        graph = helper.make_graph(nodes, "test", [X], [Y], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        assert not Autotiler.is_tileable(model)

    def test_multiple_inputs_not_tileable(self):
        X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 3, 32, 32])
        X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 32, 32])
        nodes = [helper.make_node("Add", ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(nodes, "test", [X1, X2], [Y], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        assert not Autotiler.is_tileable(model)


class TestDetectTilingNeeds:
    def _create_conv_model(self, c_in, c_out, spatial, kernel=3, stride=1, padding=1, dilation=1):
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

    def test_returns_params_for_tileable_model(self, tmp_path):
        model = self._create_conv_model(3, 8, 64)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        params = Autotiler.detect_tiling_needs(model_path, tile_size=3*16*16)
        assert params is not None
        assert params["tile_size"] == 16
        assert params["tiles_y"] == 4
        assert params["tiles_x"] == 4
        assert params["num_tiles"] == 16
        assert params["halo"] == [1, 1]

    def test_returns_none_for_tile_larger_than_spatial(self, tmp_path):
        model = self._create_conv_model(3, 8, 32)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        # This will not return None anymore just because of the value, 
        # but because 3*64*64 is larger than the total elements in a 32x32 image (3*32*32)
        # Wait, if tile_size (max elements) > total elements, it returns None, "already_fits" in calculate_spatial_tile_config
        # which results in detect_tiling_needs returning None.
        params = Autotiler.detect_tiling_needs(model_path, tile_size=3*64*64)
        assert params is None

    def test_returns_none_for_non_square_spatial(self, tmp_path):
        c_in, c_out = 3, 8
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, 64, 32])
        np.random.seed(42)
        W = numpy_helper.from_array(np.random.randn(c_out, c_in, 3, 3).astype(np.float32), "W")
        conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, 64, 32])
        graph = helper.make_graph([conv], "conv", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        params = Autotiler.detect_tiling_needs(model_path, tile_size=3*16*16)
        assert params is None

    def test_halo_for_large_kernel(self, tmp_path):
        model = self._create_conv_model(3, 8, 64, kernel=7, padding=3)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        params = Autotiler.detect_tiling_needs(model_path, tile_size=3*16*16)
        assert params is not None
        assert params["halo"] == [3, 3]

    def test_halo_for_dilated_kernel(self, tmp_path):
        model = self._create_conv_model(3, 8, 64, kernel=3, dilation=2, padding=2)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        params = Autotiler.detect_tiling_needs(model_path, tile_size=3*16*16)
        assert params is not None
        assert params["halo"] == [2, 2]

    def test_tile_size_calculates_tile(self, tmp_path):
        model = self._create_conv_model(3, 8, 64)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        params = Autotiler.detect_tiling_needs(model_path, tile_size=3 * 32 * 32)
        assert params is not None
        assert params["tile_size"] <= 32


class TestCreateTileSlice:
    def _create_conv_model(self, c_in, c_out, spatial, kernel=3, stride=1, padding=1):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])
        np.random.seed(42)
        w_data = np.random.randn(c_out, c_in, kernel, kernel).astype(np.float32)
        W = numpy_helper.from_array(w_data, "W")
        conv = helper.make_node("Conv", ["X", "W"], ["Y"],
                                kernel_shape=[kernel, kernel],
                                strides=[stride, stride],
                                pads=[padding, padding, padding, padding])
        out_spatial = (spatial + 2 * padding - kernel) // stride + 1
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, out_spatial, out_spatial])
        graph = helper.make_graph([conv], "conv", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model

    def test_creates_tile_model(self, tmp_path):
        model = self._create_conv_model(3, 8, 64)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        params = Autotiler.detect_tiling_needs(model_path, tile_size=3*16*16)
        assert params is not None

        output_dir = tmp_path / "tiles"
        output_dir.mkdir()

        tile_info = Autotiler.create_tile_slice(model_path, params["tile_size"], 0, output_dir)
        assert tile_info is not None
        assert "path" in tile_info
        assert Path(tile_info["path"]).exists()

    def test_tile_model_has_correct_input_shape(self, tmp_path):
        c_in, c_out, spatial = 3, 8, 64
        tile_size_max_elements = 3*16*16
        halo = 1

        model = self._create_conv_model(c_in, c_out, spatial)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        params = Autotiler.detect_tiling_needs(model_path, tile_size=tile_size_max_elements)
        output_dir = tmp_path / "tiles"
        output_dir.mkdir()

        tile_info = Autotiler.create_tile_slice(model_path, params["tile_size"], 0, output_dir)
        tile_model = onnx.load(tile_info["path"])

        inp = tile_model.graph.input[0]
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        expected_tile_input = 16 + 2 * halo
        assert dims == [1, c_in, expected_tile_input, expected_tile_input]


class TestTiledParity:
    """Test that Python split -> tile inference -> Python concat matches original."""

    def _create_conv_model(self, c_in, c_out, spatial, kernel=3, stride=1, padding=1):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])
        np.random.seed(42)
        w_data = np.random.randn(c_out, c_in, kernel, kernel).astype(np.float32)
        W = numpy_helper.from_array(w_data, "W")
        conv = helper.make_node("Conv", ["X", "W"], ["Y"],
                                kernel_shape=[kernel, kernel],
                                strides=[stride, stride],
                                pads=[padding, padding, padding, padding])
        out_spatial = (spatial + 2 * padding - kernel) // stride + 1
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, out_spatial, out_spatial])
        graph = helper.make_graph([conv], "conv", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model

    def _run_original(self, model_path, input_data):
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return sess.run(None, {sess.get_inputs()[0].name: input_data})[0]

    def _run_tiled_python(self, tile_path, tiling_params, input_tensor):
        """Run tiling with pure Python split/concat like the runner does."""
        tile_size = tiling_params["tile_size"]
        halo = tiling_params["halo"]
        tiles_y = tiling_params["tiles_y"]
        tiles_x = tiling_params["tiles_x"]
        out_tile = tiling_params["out_tile"]
        stride = tiling_params["stride"]

        halo_h, halo_w = halo
        out_tile_h, out_tile_w = out_tile
        sh, sw = stride

        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor)

        _, c, h, w = input_tensor.shape
        pad_h = halo_h
        pad_w = halo_w
        padded = torch.nn.functional.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        tile_sess = ort.InferenceSession(tile_path, providers=["CPUExecutionProvider"])
        tile_input_name = tile_sess.get_inputs()[0].name

        tile_outputs = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                y_start = ty * tile_size
                x_start = tx * tile_size
                tile_h = tile_size + 2 * halo_h
                tile_w = tile_size + 2 * halo_w
                tile = padded[:, :, y_start:y_start + tile_h, x_start:x_start + tile_w]
                tile_np = tile.numpy().astype(np.float32)
                tile_out = tile_sess.run(None, {tile_input_name: tile_np})[0]
                tile_outputs.append(torch.from_numpy(tile_out))

        c_out = tile_outputs[0].shape[1]
        out_h = tiles_y * out_tile_h
        out_w = tiles_x * out_tile_w
        output = torch.zeros(1, c_out, out_h, out_w)

        idx = 0
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                y_start = ty * out_tile_h
                x_start = tx * out_tile_w
                output[:, :, y_start:y_start + out_tile_h, x_start:x_start + out_tile_w] = tile_outputs[idx]
                idx += 1

        return output.numpy()

    @pytest.mark.parametrize("kernel,stride,padding", [
        (3, 1, 1),
        (3, 2, 1),
        (5, 1, 2),
    ])
    def test_parity(self, kernel, stride, padding, tmp_path):
        c_in, c_out, spatial = 3, 8, 64
        tile_size_max_elements = 3*16*16

        model = self._create_conv_model(c_in, c_out, spatial, kernel, stride, padding)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        np.random.seed(123)
        input_data = np.random.randn(1, c_in, spatial, spatial).astype(np.float32)

        expected = self._run_original(model_path, input_data)

        params = Autotiler.detect_tiling_needs(model_path, tile_size=tile_size_max_elements)
        if params is None:
            pytest.skip("Model not tileable with given parameters")

        output_dir = tmp_path / "tiles"
        output_dir.mkdir()
        tile_info = Autotiler.create_tile_slice(model_path, params["tile_size"], 0, output_dir)

        actual = self._run_tiled_python(tile_info["path"], params, input_data)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5,
                                   err_msg=f"Parity failed for kernel={kernel}, stride={stride}, padding={padding}")

    def test_boundary_pixels_exact(self, tmp_path):
        c_in, c_out, spatial = 1, 1, 32
        tile_size_max_elements = 1*16*16

        model = self._create_conv_model(c_in, c_out, spatial)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        np.random.seed(888)
        input_data = np.random.randn(1, c_in, spatial, spatial).astype(np.float32)

        expected = self._run_original(model_path, input_data)

        params = Autotiler.detect_tiling_needs(model_path, tile_size=tile_size_max_elements)
        output_dir = tmp_path / "tiles"
        output_dir.mkdir()
        tile_info = Autotiler.create_tile_slice(model_path, params["tile_size"], 0, output_dir)

        actual = self._run_tiled_python(tile_info["path"], params, input_data)

        boundary_h = 16
        np.testing.assert_array_equal(
            actual[0, :, boundary_h-1:boundary_h+1, :],
            expected[0, :, boundary_h-1:boundary_h+1, :],
            err_msg="Horizontal tile boundary pixels differ"
        )


class TestInvalidInputs:
    def _create_conv_model(self, spatial, kernel=3, stride=1, padding=1):
        c_in, c_out = 3, 8
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])
        np.random.seed(42)
        W = numpy_helper.from_array(np.random.randn(c_out, c_in, kernel, kernel).astype(np.float32), "W")
        conv = helper.make_node("Conv", ["X", "W"], ["Y"],
                                kernel_shape=[kernel, kernel],
                                strides=[stride, stride],
                                pads=[padding, padding, padding, padding])
        out_spatial = (spatial + 2 * padding - kernel) // stride + 1
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, out_spatial, out_spatial])
        graph = helper.make_graph([conv], "conv", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model

    def test_tile_size_larger_than_spatial(self, tmp_path):
        model = self._create_conv_model(spatial=32)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        # 3 channels * 32 * 32 = 3072. 3*64*64 = 12288.
        # So 12288 elements fits a 3072 image without tiling.
        # calculate_spatial_tile_config returns None, "already_fits"
        result = Autotiler.detect_tiling_needs(model_path, tile_size=3*64*64)
        assert result is None

    def test_prime_spatial_no_valid_divisor(self, tmp_path):
        model = self._create_conv_model(spatial=37)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        # c_in=3. sqrt((3*16*16)/3) = 16. 
        # find_optimal_tile_size(37, 16, min_tile=7) will look for divisors of 37 <= 16.
        # 37 is prime. No divisors. returns None.
        result = Autotiler.detect_tiling_needs(model_path, tile_size=3*16*16)
        assert result is None

    def test_tile_smaller_than_kernel(self, tmp_path):
        model = self._create_conv_model(spatial=64, kernel=7, padding=3)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        # min_tile = 8 for kernel 7.
        # sqrt((3*4*4)/3) = 4. 4 < 8. returns None.
        result = Autotiler.detect_tiling_needs(model_path, tile_size=3*4*4)
        assert result is None

    def test_multiple_conv_rejected(self, tmp_path):
        c_in, c_mid, c_out = 3, 8, 16
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, 64, 64])
        np.random.seed(42)
        W1 = numpy_helper.from_array(np.random.randn(c_mid, c_in, 3, 3).astype(np.float32), "W1")
        W2 = numpy_helper.from_array(np.random.randn(c_out, c_mid, 3, 3).astype(np.float32), "W2")
        nodes = [
            helper.make_node("Conv", ["X", "W1"], ["conv1_out"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["conv1_out"], ["relu_out"]),
            helper.make_node("Conv", ["relu_out", "W2"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        ]
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, 64, 64])
        graph = helper.make_graph(nodes, "two_conv", [X], [Y], [W1, W2])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        result = Autotiler.detect_tiling_needs(model_path, tile_size=3*16*16)
        assert result is None
