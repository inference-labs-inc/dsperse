import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model
from dsperse.src.cli.run import run_inference
from dsperse.src.cli.prove import run_proof
from dsperse.src.cli.verify import verify_proof

class TestVerifyE2E:
    def _get_run_dir(self, output: str) -> Path:
        match = re.search(r"Run data saved to (.+)", output)
        if not match:
             match = re.search(r"within the run directory (.+)", output)
        assert match, f"Could not find run directory in output: {output}"
        path_str = match.group(1).strip()
        path_str = re.sub(r'\x1b\[[0-9;]*m', '', path_str).split()[0]
        return Path(path_str)

    def _verify_verify_artifacts(self, run_dir: Path):
        assert run_dir.exists()
        run_results_path = run_dir / "run_results.json"
        assert run_results_path.exists()
        
        results = json.loads(run_results_path.read_text())
        assert "execution_chain" in results
        exec_chain = results["execution_chain"]
        assert "execution_results" in exec_chain
        
        return results

    @pytest.mark.parametrize("model_name", ["net"])
    def test_verify_happy_path(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, jstprove_available, ezkl_available, capfd):
        """1) The happy path test. We can slice, compile, run, prove then verify all the slices."""
        if not jstprove_available or not ezkl_available:
            pytest.skip("Backends unavailable")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=None, backend=None))
        
        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend=None))
        capfd.readouterr()

        verify_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend=None))
        out = capfd.readouterr().out
        
        assert "Verification completed" in out
        results = self._verify_verify_artifacts(run_dir)
        exec_chain = results["execution_chain"]
        
        for res in exec_chain["execution_results"]:
            assert "verification_execution" in res
            assert res["verification_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_verify_single_slice(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, jstprove_available, capfd):
        """2) we can pass in a single slice (slices/slice_0 directory) and single run (run/run_id/slice_0) and the backend flag (jstprove) and it would verify that single slice only."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=None, backend="jstprove"))
        
        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend="jstprove", run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend="jstprove"))
        capfd.readouterr()

        slice_0_dir = slices_output_dir / "slice_0"
        run_slice_0_dir = run_dir / "slice_0"
        
        verify_proof(SimpleNamespace(run_dir=str(run_slice_0_dir), slices_path=str(slice_0_dir), backend="jstprove"))
        out = capfd.readouterr().out
        
        assert "Verification completed" in out
        results = self._verify_verify_artifacts(run_slice_0_dir)
        exec_chain = results["execution_chain"]
        assert exec_chain["execution_results"][0]["verification_execution"]["success"] is True
        
        assert "Verification completed" in out
        
        # In single slice mode, it should update the run_results.json in the per-slice run directory.
        results = self._verify_verify_artifacts(run_slice_0_dir)
        exec_chain = results["execution_chain"]
        
        # Find slice_0 result
        slice_0_res = next((res for res in exec_chain["execution_results"] if res["slice_id"] == "slice_0"), None)
        assert slice_0_res is not None
        assert "verification_execution" in slice_0_res
        assert slice_0_res["verification_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_verify_mixed_backends(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """3) we slice, compile (with mixed backends  0:jstprove) run (default) prove and verify (default, no backend flag provided)"""
        try:
            from dsperse.src.backends.jstprove import JSTprove
            from dsperse.src.backends.ezkl import EZKL
            _j, _e = JSTprove(), EZKL()
        except Exception as e:
            pytest.skip(f"Backends unavailable: {e}")

        # Slice
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        
        # Mixed compile
        layers = "0:jstprove" 
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=layers, backend=None))
        
        # Run
        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        # Prove
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend=None))
        capfd.readouterr()

        # Verify
        verify_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend=None))
        out = capfd.readouterr().out
        
        assert "Verification completed" in out
        results = self._verify_verify_artifacts(run_dir)
        assert (results["execution_chain"].get("jstprove_verified_slices", 0) + 
                results["execution_chain"].get("ezkl_verified_slices", 0)) > 0

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_verify_backend_filter_doom(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """4) with the same compile (mixed backends) we provide a backend flag (ezkl) and only ezkl slices are verified)"""
        try:
            from dsperse.src.backends.jstprove import JSTprove
            from dsperse.src.backends.ezkl import EZKL
            _j, _e = JSTprove(), EZKL()
        except Exception as e:
            pytest.skip(f"Backends unavailable: {e}")

        # Slice
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        
        # Mixed compile: slice 0:jstprove, slice 1:ezkl (doom has at least 2 slices)
        layers = "0:jstprove;1:ezkl"
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=layers, backend=None))
        
        # Run
        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        # Prove all
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend=None))
        capfd.readouterr()

        # Verify with backend=ezkl
        verify_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend="ezkl"))
        out = capfd.readouterr().out
        
        results = self._verify_verify_artifacts(run_dir)
        exec_chain = results["execution_chain"]
        
        # Should only have ezkl verifications
        assert exec_chain.get("jstprove_verified_slices", 0) == 0
        assert exec_chain.get("ezkl_verified_slices", 0) > 0
        
        # Specifically slice_1 should have verification, slice_0 should NOT
        found_slice_0 = False
        found_slice_1 = False
        for res in exec_chain["execution_results"]:
            if res["slice_id"] == "slice_0":
                found_slice_0 = True
                assert "verification_execution" not in res
            if res["slice_id"] == "slice_1":
                found_slice_1 = True
                assert "verification_execution" in res
                assert res["verification_execution"]["success"] is True

        assert found_slice_0, "slice_0 not found in execution_results"
        assert found_slice_1, "slice_1 not found in execution_results"

    def test_verify_with_tiling(self, project_root: Path, capfd, jstprove_available):
        """Verify that verification correctly handles tiled slices."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        model_dir = project_root / "tests" / "models" / "doom"
        output_dir = project_root / "tests" / "models" / "doom_tiling_verify"
        
        import shutil
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
        capfd.readouterr()
        run_inference(SimpleNamespace(
            path=str(output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None,
            run_metadata_path=None
        ))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        # 4. Prove
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(output_dir), backend="jstprove"))
        
        # 5. Verify
        capfd.readouterr()
        verify_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(output_dir), backend="jstprove"))
        
        # 6. Verify run_results.json
        run_results = json.loads((run_dir / "run_results.json").read_text())
        exec_results = run_results["execution_chain"]["execution_results"]
        
        # slice_0 is tiled
        s0_res = next(r for r in exec_results if r["slice_id"] == "slice_0")
        v_exec = s0_res["verification_execution"]
        
        assert v_exec["success"] is True
        assert v_exec["verified"] is True
        assert "tile_verifs_info" in v_exec
        assert len(v_exec["tile_verifs_info"]) == 4
        
        for t_verif in v_exec["tile_verifs_info"]:
            assert t_verif["success"] is True

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)
