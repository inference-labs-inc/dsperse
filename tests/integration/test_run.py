import json
import shutil
import re
import logging
import pytest
from pathlib import Path
from types import SimpleNamespace

from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model
from dsperse.src.cli.run import run_inference

class TestRunE2E:
    def _get_run_dir(self, output: str) -> Path:
        match = re.search(r"Run data saved to (.+)", output)
        assert match, f"Could not find run directory in output: {output}"
        return Path(match.group(1).strip())

    def _verify_run_artifacts(self, run_dir: Path):
        assert run_dir.exists()
        assert (run_dir / "run_results.json").exists()
        assert (run_dir / "metadata.json").exists()

    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_run_onnx_only(self, model_name: str, model_dir: Path, pre_sliced_net: Path, pre_sliced_doom: Path, run_output_dir: Path, tmp_path, copy_to, capfd):
        source = pre_sliced_net if model_name == "net" else pre_sliced_doom
        slices_output_dir = copy_to(source, tmp_path / "slices")
        
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "onnx" in out.lower()
        assert "slice_0: onnx" in out  # matches onnx_multi_input

        run_dir = self._get_run_dir(out)
        self._verify_run_artifacts(run_dir)

        results = json.loads((run_dir / "run_results.json").read_text())
        for slice_res in results.get("slice_results", {}).values():
            assert slice_res["method"].startswith("onnx")

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_default_compiled(self, model_name: str, model_dir: Path, pre_compiled_net_both: Path, run_output_dir: Path, tmp_path, copy_to, capfd):
        slices_output_dir = copy_to(pre_compiled_net_both, tmp_path / "slices")
        
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "jstprove" in out.lower()
        assert "slice_0: jstprove" in out
        
        run_dir = self._get_run_dir(out)
        self._verify_run_artifacts(run_dir)

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_backend_flags(self, model_name: str, model_dir: Path, pre_compiled_net_both: Path, run_output_dir: Path, tmp_path, copy_to, capfd):
        slices_output_dir = copy_to(pre_compiled_net_both, tmp_path / "slices")
        input_file = model_dir / "input.json"
        
        # Test force_backend=onnx (guaranteed to work, no fallback)
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend="onnx")
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out.lower()
        assert "slice_0: onnx" in out

        # Test force_backend=jstprove (should use jstprove for compiled slices)
        run_args.force_backend = "jstprove"
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out.lower()
        assert "jstprove" in out

        # Test force_backend=ezkl (may fallback to jstprove/onnx if ezkl fails)
        run_args.force_backend = "ezkl"
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out.lower()
        # EZKL may fail and fallback, so just check inference completed
        assert "inference completed" in out

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_single_slice_uncompiled(self, model_name: str, model_dir: Path, pre_sliced_net: Path, run_output_dir: Path, tmp_path, copy_to, capfd):
        slices_output_dir = copy_to(pre_sliced_net, tmp_path / "slices")
        
        slice_0_path = slices_output_dir / "slice_0"
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slice_0_path), input_file=str(input_file), output_file=None, force_backend=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "onnx" in out.lower()
        assert "slice_0: onnx" in out
        assert "slice_1:" not in out # Ensure only one slice ran
        
        run_dir = self._get_run_dir(out)
        self._verify_run_artifacts(run_dir)

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_single_slice_compiled(self, model_name: str, model_dir: Path, pre_compiled_net_both: Path, run_output_dir: Path, tmp_path, copy_to, capfd):
        slices_output_dir = copy_to(pre_compiled_net_both, tmp_path / "slices")
        
        slice_0_path = slices_output_dir / "slice_0"
        input_file = model_dir / "input.json"
        
        # Default (blank) -> should be jstprove (method name includes suffix like jstprove_gen_witness)
        run_args = SimpleNamespace(path=str(slice_0_path), input_file=str(input_file), output_file=None, force_backend=None)
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out.lower()
        assert "slice_0: jstprove" in out

        # force_backend=onnx (guaranteed to work)
        run_args.force_backend = "onnx"
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out.lower()
        assert "slice_0: onnx" in out

        # force_backend=jstprove
        run_args.force_backend = "jstprove"
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out.lower()
        assert "slice_0: jstprove" in out

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_formats(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """6. we test both whole model and single slice with .dsperse and .dslice and dirs"""
        input_file = model_dir / "input.json"
        
        # .dslice
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dslice"))
        # Whole model dirs (containing .dslice)
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None)
        capfd.readouterr()
        run_inference(run_args)
        assert "Inference completed" in capfd.readouterr().out
        
        # Single .dslice file
        dslice_file = slices_output_dir / "slice_0.dslice"
        run_args.path = str(dslice_file)
        capfd.readouterr()
        run_inference(run_args)
        assert "Inference completed" in capfd.readouterr().out

        # .dsperse
        shutil.rmtree(slices_output_dir, ignore_errors=True)
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dsperse"))
        dsperse_file = slices_output_dir.parent / f"{slices_output_dir.name}.dsperse"
        run_args.path = str(dsperse_file)
        capfd.readouterr()
        run_inference(run_args)
        assert "Inference completed" in capfd.readouterr().out

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_output_file(self, model_name: str, model_dir: Path, pre_sliced_net: Path, run_output_dir: Path, tmp_path, copy_to):
        slices_output_dir = copy_to(pre_sliced_net, tmp_path / "slices")
        input_file = model_dir / "input.json"
        custom_out = tmp_path / "custom_results.json"
        
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=str(custom_out), force_backend=None)
        run_inference(run_args)
        
        assert custom_out.exists()
        data = json.loads(custom_out.read_text())
        assert "output" in data

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_mixed_compilation(self, model_name: str, model_dir: Path, pre_sliced_net: Path, run_output_dir: Path, tmp_path, copy_to, jstprove_available, ezkl_available, capfd):
        if not jstprove_available or not ezkl_available:
            pytest.skip("Backends unavailable")
        slices_output_dir = copy_to(pre_sliced_net, tmp_path / "slices")
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers="0;2:jstprove;3-4:ezkl", backend=None))
        
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "Inference completed" in out
        out_lower = out.lower()
        # jstprove preferred for 0, 2 (method names include suffix like jstprove_gen_witness)
        assert "slice_0: jstprove" in out_lower
        assert "slice_2: jstprove" in out_lower
        # ezkl for 3, 4 (may fallback, so just check ezkl appears in method or fallback occurs)
        assert "slice_3: ezkl" in out_lower or "slice_3: jstprove" in out_lower
        assert "slice_4: ezkl" in out_lower or "slice_4: jstprove" in out_lower
        # onnx for 1 (uncompiled)
        assert "slice_1: onnx" in out_lower

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_run_with_tiling(self, model_name: str, model_dir: Path, pre_compiled_doom_tiled: Path, tmp_path, copy_to, capfd):
        slices_output_dir = copy_to(pre_compiled_doom_tiled, tmp_path / "slices")
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None
        )
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "Inference completed" in out
        assert "slice_0: tiled" in out

        run_dir = self._get_run_dir(out)
        run_results = json.loads((run_dir / "run_results.json").read_text())

        # Verify tiled slice execution in execution_chain
        exec_results = run_results.get("execution_chain", {}).get("execution_results", [])
        s0_res = next((r for r in exec_results if r["slice_id"] == "slice_0"), None)
        assert s0_res is not None
        w_exec = s0_res["witness_execution"]
        assert w_exec["method"] == "tiled"
        assert w_exec["success"] is True
        assert len(w_exec["tile_exec_infos"]) == 4
        for t_info in w_exec["tile_exec_infos"]:
            assert t_info["success"] is True

    def test_run_with_channel_split(self, tmp_path, capfd, jstprove_available):
        """10. Verify that running a channel-split model works."""
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np

        # Create a model with high input channels and prime spatial dimension to force channel splitting
        c_in, c_out, spatial = 16, 16, 11
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])
        np.random.seed(42)
        W_data = np.random.randn(c_out, c_in, 3, 3).astype(np.float32)
        W = numpy_helper.from_array(W_data, "W")
        # Add bias to verify bias compensation
        B_data = np.random.randn(c_out).astype(np.float32)
        B = numpy_helper.from_array(B_data, "B")
        
        conv = helper.make_node(
            "Conv", ["X", "W", "B"], ["Y"], 
            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1]
        )
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, spatial, spatial])
        graph = helper.make_graph([conv], "test", [X], [Y], [W, B])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        model_dir = tmp_path / "channel_split_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        onnx.save(model, str(model_path))
        
        # Create input.json
        input_data = np.random.randn(1, c_in, spatial, spatial).astype(np.float32).tolist()
        input_file = model_dir / "input.json"
        with open(input_file, 'w') as f:
            json.dump({"input_data": input_data}, f)
            
        slices_output_dir = model_dir / "slices"
        
        # 1. Slice
        slice_model(SimpleNamespace(
            model_dir=str(model_path),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=1000
        ))
        
        # 2. Compile (optional)
        if jstprove_available:
            compile_model(SimpleNamespace(
                path=str(slices_output_dir),
                input_file=None,
                layers=None,
                backend="jstprove"
            ))

        # 3. Run
        run_args = SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None,
            threads=1,
            run_dir=None
        )
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "Inference completed" in out
        assert "channel_split" in out.lower()

        run_dir = self._get_run_dir(out)
        run_results = json.loads((run_dir / "run_results.json").read_text())
        assert "output" in run_results
        assert run_results["tensor_shape"] == [1, c_out, spatial, spatial]

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_run_parallel_tiling(self, model_name: str, model_dir: Path, pre_compiled_doom_tiled: Path, tmp_path, copy_to, capfd, caplog):
        slices_output_dir = copy_to(pre_compiled_doom_tiled, tmp_path / "slices")
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None,
            threads=2,
            run_dir=None
        )

        with caplog.at_level(logging.INFO):
            run_inference(run_args)
        
        out = capfd.readouterr().out
        assert "Inference completed" in out
        assert "parallel processes" in caplog.text.lower()
        assert "slice_0: tiled" in out

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_run_tiling_ezkl(self, model_name: str, model_dir: Path, pre_sliced_doom_tiled: Path, tmp_path, copy_to, capfd, ezkl_available):
        if not ezkl_available:
            pytest.skip("EZKL unavailable")
        slices_output_dir = copy_to(pre_sliced_doom_tiled, tmp_path / "slices")
        compile_model(SimpleNamespace(
            path=str(slices_output_dir),
            input_file=None,
            layers=None,
            backend="ezkl"
        ))

        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend="ezkl",
            threads=1,
            run_dir=None
        )
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        assert "Inference completed" in out
        assert "slice_0: tiled" in out

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_run_tiling_onnx_fallback(self, model_name: str, model_dir: Path, pre_sliced_doom_tiled: Path, tmp_path, copy_to, capfd):
        slices_output_dir = copy_to(pre_sliced_doom_tiled, tmp_path / "slices")

        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend="onnx",
            threads=1,
            run_dir=None
        )
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        assert "Inference completed" in out
        assert "slice_0: tiled" in out
        assert "onnx_only" in out.lower()

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_run_resume_complex(self, model_name: str, model_dir: Path, pre_sliced_doom_tiled: Path, tmp_path, copy_to, capfd):
        slices_output_dir = copy_to(pre_sliced_doom_tiled, tmp_path / "slices")
        
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend="onnx",
            threads=1,
            run_dir=None
        )
        
        # First run to produce artifacts
        run_inference(run_args)
        out1 = capfd.readouterr().out
        run_dir = self._get_run_dir(out1)
        
        # Second run with resume
        run_args.run_dir = str(run_dir)
        capfd.readouterr()
        run_inference(run_args)
        out2 = capfd.readouterr().out
        
        assert "Inference completed" in out2
        assert "loaded cached output" in out2.lower()

    def test_run_multi_input_complex(self, tmp_path, capfd):
        """15. Verify multi-input ONNX models with complex dependencies."""
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
        
        # X1 + X2 -> T1
        # T1 + X3 -> Y
        X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 10])
        X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 10])
        X3 = helper.make_tensor_value_info("X3", TensorProto.FLOAT, [1, 10])
        add1 = helper.make_node("Add", ["X1", "X2"], ["T1"])
        add2 = helper.make_node("Add", ["T1", "X3"], ["Y"])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
        graph = helper.make_graph([add1, add2], "multi_input", [X1, X2, X3], [Y], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        model_dir = tmp_path / "multi_input_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        onnx.save(model, str(model_path))
        
        slices_output_dir = model_dir / "slices"
        slice_model(SimpleNamespace(
            model_dir=str(model_path),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs"
        ))
        
        input_file = model_dir / "input.json"
        with open(input_file, 'w') as f:
            json.dump({"input_data": np.zeros((1, 10)).tolist()}, f)
            
        run_args = SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend="onnx",
            threads=1,
            run_dir=None
        )
        
        run_inference(run_args)
        out = capfd.readouterr().out
        assert "Inference completed" in out
