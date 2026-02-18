import pytest
from dsperse.src.analyzers.schema import BackendCompilation, Backend
from dsperse.src.compile.utils.compiler_utils import CompilerUtils, CompilationTask


class TestBuildCompilationBlock:

    def test_jstprove_includes_wai_true(self):
        block = CompilerUtils.build_compilation_block(
            backend=Backend.JSTPROVE, version="2.4.0", success=True,
            file_paths={"compiled": "c.txt"}, slice_dir="/tmp/slice_0",
            tiling_info=None, weights_as_inputs=True,
        )
        assert block["weights_as_inputs"] is True

    def test_jstprove_includes_wai_false(self):
        block = CompilerUtils.build_compilation_block(
            backend=Backend.JSTPROVE, version="2.4.0", success=True,
            file_paths={"compiled": "c.txt"}, slice_dir="/tmp/slice_0",
            tiling_info=None, weights_as_inputs=False,
        )
        assert block["weights_as_inputs"] is False

    def test_ezkl_omits_wai(self):
        block = CompilerUtils.build_compilation_block(
            backend=Backend.EZKL, version="22.0", success=True,
            file_paths={"compiled": "m.compiled"}, slice_dir="/tmp/slice_0",
            tiling_info=None, weights_as_inputs=False,
        )
        assert "weights_as_inputs" not in block


class TestBuildChannelSplitBlock:

    def test_jstprove_includes_wai(self):
        groups = [{"group_idx": 0, "success": True, "files": {"compiled": "g0.txt"}}]
        block = CompilerUtils.build_channel_split_block(
            backend=Backend.JSTPROVE, num_groups=1, group_results=groups,
            slice_dir="/tmp/slice_0", weights_as_inputs=True,
        )
        assert block["weights_as_inputs"] is True

    def test_ezkl_omits_wai(self):
        groups = [{"group_idx": 0, "success": True, "files": {"compiled": "m.compiled"}}]
        block = CompilerUtils.build_channel_split_block(
            backend=Backend.EZKL, num_groups=1, group_results=groups,
            slice_dir="/tmp/slice_0", weights_as_inputs=False,
        )
        assert "weights_as_inputs" not in block


class TestBackendCompilationSchema:

    def test_from_dict_with_wai_true(self):
        bc = BackendCompilation.from_dict({"compiled": True, "weights_as_inputs": True})
        assert bc.weights_as_inputs is True

    def test_from_dict_with_wai_false(self):
        bc = BackendCompilation.from_dict({"compiled": True, "weights_as_inputs": False})
        assert bc.weights_as_inputs is False

    def test_from_dict_missing_wai_defaults_false(self):
        bc = BackendCompilation.from_dict({"compiled": True})
        assert bc.weights_as_inputs is False

    def test_from_dict_none_defaults_false(self):
        bc = BackendCompilation.from_dict(None)
        assert bc.weights_as_inputs is False


class TestPrepareCompilationTasksWAI:

    def test_propagates_weights_as_inputs(self):
        slice_data = {"path": "/fake/model.onnx", "relative_path": "model.onnx"}
        tasks, _, _ = CompilerUtils.prepare_compilation_tasks(
            slices_data=[slice_data], base_path="/fake", layer_indices=None,
            resume=False, layer_backends={}, default_layer_indices=set(),
            default_backend=Backend.JSTPROVE, use_fallback=False,
            weights_as_inputs=True,
        )
        assert len(tasks) == 1
        assert tasks[0].weights_as_inputs is True

    def test_defaults_to_false(self):
        slice_data = {"path": "/fake/model.onnx", "relative_path": "model.onnx"}
        tasks, _, _ = CompilerUtils.prepare_compilation_tasks(
            slices_data=[slice_data], base_path="/fake", layer_indices=None,
            resume=False, layer_backends={}, default_layer_indices=set(),
            default_backend=Backend.JSTPROVE, use_fallback=False,
        )
        assert len(tasks) == 1
        assert tasks[0].weights_as_inputs is False
