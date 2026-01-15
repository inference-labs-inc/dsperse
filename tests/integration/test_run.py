import json
import shutil
import re
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
    def test_run_onnx_only(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """1. test with slicing (no compilation), then we run and see the output that it was onnx only"""
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "onnx" in out.lower()
        assert "slice_0: onnx" in out
        
        run_dir = self._get_run_dir(out)
        self._verify_run_artifacts(run_dir)
        
        results = json.loads((run_dir / "run_results.json").read_text())
        for slice_res in results.get("slice_results", {}).values():
            assert slice_res["method"] == "onnx"

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_default_compiled(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, jstprove_available, ezkl_available, capfd):
        """2. then compile (default compile, both ezkl and jstprove) and run, we should see that by default jstprove is chosen"""
        if not jstprove_available or not ezkl_available:
            pytest.skip("Backends unavailable")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=None, backend=None))
        
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "jstprove" in out.lower()
        assert "slice_0: jstprove" in out
        
        run_dir = self._get_run_dir(out)
        self._verify_run_artifacts(run_dir)

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_backend_flags(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, jstprove_available, ezkl_available, capfd):
        """3. we add a backend flag = ezkl, backendflag = jstprove, backendflag = onnx and the output should show that"""
        if not jstprove_available or not ezkl_available:
            pytest.skip("Backends unavailable")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=None, backend=None))
        input_file = model_dir / "input.json"
        
        for backend in ["ezkl", "jstprove", "onnx"]:
            run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=backend, run_metadata_path=None)
            capfd.readouterr()
            run_inference(run_args)
            out = capfd.readouterr().out
            assert f"slice_0: {backend}" in out.lower()

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_single_slice_uncompiled(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """4. we run with a single slice (slice = path/to/slice_0) and verify that it runs only that one slice (no compilation)"""
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        
        slice_0_path = slices_output_dir / "slice_0"
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slice_0_path), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "onnx" in out.lower()
        assert "slice_0: onnx" in out
        assert "slice_1:" not in out # Ensure only one slice ran
        
        run_dir = self._get_run_dir(out)
        self._verify_run_artifacts(run_dir)

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_single_slice_compiled(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """5. we run single slice with compilation (default) and verify with default behavior then with ezkl then with jstprove then with onnx"""
        try:
            from dsperse.src.backends.jstprove import JSTprove
            from dsperse.src.backends.ezkl import EZKL
            _j, _e = JSTprove(), EZKL()
        except Exception as e:
            pytest.skip(f"Backends unavailable: {e}")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers="0", backend=None))
        
        slice_0_path = slices_output_dir / "slice_0"
        input_file = model_dir / "input.json"
        
        # Default (blank) -> should be jstprove
        run_args = SimpleNamespace(path=str(slice_0_path), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None)
        capfd.readouterr()
        run_inference(run_args)
        assert "slice_0: jstprove" in capfd.readouterr().out.lower()

        # Explicit flags
        for backend in ["ezkl", "jstprove", "onnx"]:
            run_args.force_backend = backend
            capfd.readouterr()
            run_inference(run_args)
            assert f"slice_0: {backend}" in capfd.readouterr().out.lower()

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_formats(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """6. we test both whole model and single slice with .dsperse and .dslice and dirs"""
        input_file = model_dir / "input.json"
        
        # .dslice
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dslice"))
        # Whole model dirs (containing .dslice)
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None)
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
    def test_run_output_file(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, tmp_path):
        """7. we test the output_file flag"""
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        input_file = model_dir / "input.json"
        custom_out = tmp_path / "custom_results.json"
        
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=str(custom_out), force_backend=None, run_metadata_path=None)
        run_inference(run_args)
        
        assert custom_out.exists()
        data = json.loads(custom_out.read_text())
        assert "prediction" in data

    @pytest.mark.parametrize("model_name", ["net"])
    def test_run_mixed_compilation(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """8. we test with a crazy compile `0;2:jstprove;3-4:ezkl` and verify that it runs all slices by default settings."""
        try:
            from dsperse.src.backends.jstprove import JSTprove
            from dsperse.src.backends.ezkl import EZKL
            _j, _e = JSTprove(), EZKL()
        except Exception as e:
            pytest.skip(f"Backends unavailable: {e}")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers="0;2:jstprove;3-4:ezkl", backend=None))
        
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None)
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "Inference completed" in out
        # jstprove preferred for 0, 2
        assert "slice_0: jstprove" in out
        assert "slice_2: jstprove" in out
        # ezkl for 3, 4
        assert "slice_3: ezkl" in out
        assert "slice_4: ezkl" in out
        # onnx for 1
        assert "slice_1: onnx" in out

    def test_run_with_tiling(self, project_root: Path, capfd, jstprove_available):
        """9. Verify that running a tiled model works and produces correct run_results.json."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        model_dir = project_root / "tests" / "models" / "doom"
        output_dir = project_root / "tests" / "models" / "doom_tiling_run"
        
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # 1. Slice with tiling
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=14
        ))

        # 2. Compile
        compile_model(SimpleNamespace(
            path=str(output_dir),
            input_file=None,
            layers=None,
            backend="jstprove"
        ))

        # 3. Run
        input_file = model_dir / "input.json"
        run_args = SimpleNamespace(
            path=str(output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None,
            run_metadata_path=None
        )
        
        capfd.readouterr()
        run_inference(run_args)
        out = capfd.readouterr().out
        
        assert "Inference completed" in out
        assert "slice_0: tiled_parallel" in out
        
        run_dir = self._get_run_dir(out)
        run_results = json.loads((run_dir / "run_results.json").read_text())
        
        # Verify witness_execution for tiled slice
        s0_res = next(r for r in run_results["execution_chain"]["execution_results"] if r["slice_id"] == "slice_0")
        w_exec = s0_res["witness_execution"]
        
        assert w_exec["method"] == "tiled_parallel"
        assert w_exec["input_file"] != "unknown"
        assert w_exec["output_file"] != "unknown"
        assert "slice_0/split/input.json" in w_exec["input_file"]
        assert "slice_0/concat/output.json" in w_exec["output_file"]
        assert len(w_exec["tile_exec_infos"]) == 4
        
        for t_info in w_exec["tile_exec_infos"]:
            assert t_info["success"] is True
            assert "jstprove" in t_info["method"]
        
        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)
