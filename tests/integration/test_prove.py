import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model
from dsperse.src.cli.run import run_inference
from dsperse.src.cli.prove import run_proof

class TestProveE2E:
    def _get_run_dir(self, output: str) -> Path:
        match = re.search(r"Run data saved to (.+)", output)
        if not match:
             match = re.search(r"within the run directory (.+)", output)
        assert match, f"Could not find run directory in output: {output}"
        path_str = match.group(1).strip()
        path_str = re.sub(r'\x1b\[[0-9;]*m', '', path_str).split()[0]
        return Path(path_str)

    def _verify_prove_artifacts(self, run_dir: Path):
        assert run_dir.exists()
        run_results_path = run_dir / "run_results.json"
        assert run_results_path.exists()
        
        results = json.loads(run_results_path.read_text())
        assert "execution_chain" in results
        exec_chain = results["execution_chain"]
        assert "execution_results" in exec_chain
        
        return results

    @pytest.mark.parametrize("model_name", ["net"])
    def test_prove_happy_path(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, jstprove_available, ezkl_available, capfd):
        """1) The happy path test. We can slice, compile, run (all default settings) then prove all the slices."""
        if not jstprove_available or not ezkl_available:
            pytest.skip("Backends unavailable")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=None, backend=None))
        
        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend=None))
        out = capfd.readouterr().out
        
        assert "Proof generation completed" in out
        results = self._verify_prove_artifacts(run_dir)
        exec_chain = results["execution_chain"]
        
        for res in exec_chain["execution_results"]:
            assert "proof_execution" in res
            assert res["proof_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_prove_single_slice(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, jstprove_available, capfd):
        """2) we can pass in a single slice (slices/slice_0 directory) and single run (run/run_id/slice_0) and the backend flag (jstprove) and it would prove that single slice only."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=None, layers=None, backend="jstprove"))
        
        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend="jstprove", run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        slice_0_dir = slices_output_dir / "slice_0"
        run_slice_0_dir = run_dir / "slice_0"
        
        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_slice_0_dir), slices_path=str(slice_0_dir), backend="jstprove"))
        out = capfd.readouterr().out
        
        assert "Proof generation completed" in out
        
        # In single slice mode, it should update the run_results.json in the run_dir
        # Actually it updates it in the per-slice run directory.
        results = self._verify_prove_artifacts(run_slice_0_dir)
        exec_chain = results["execution_chain"]
        
        # Find slice_0 result
        slice_0_res = next((res for res in exec_chain["execution_results"] if res["slice_id"] == "slice_0"), None)
        assert slice_0_res is not None
        assert "proof_execution" in slice_0_res
        assert slice_0_res["proof_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_prove_mixed_backends(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """3) we slice, compile (with mixed backends  0,2:jstprove;3-4:ezkl) run (default) and prove (default, no backend flag provided)"""
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
        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend=None))
        out = capfd.readouterr().out
        
        assert "Proof generation completed" in out
        results = self._verify_prove_artifacts(run_dir)
        assert results["execution_chain"]["jstprove_proved_slices"] > 0

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_prove_backend_filter_doom(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd):
        """4) with the same compile (mixed backends) we provide a backend flag (ezkl) and only slices 3,4 are proven)"""
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
        
        # Prove with backend=ezkl
        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend="ezkl"))
        out = capfd.readouterr().out
        
        results = self._verify_prove_artifacts(run_dir)
        exec_chain = results["execution_chain"]
        
        # Should only have ezkl proofs
        assert exec_chain.get("jstprove_proved_slices", 0) == 0
        assert exec_chain.get("ezkl_proved_slices", 0) > 0
        
        # Specifically slice_1 should have proof, slice_0 should NOT
        found_slice_0 = False
        found_slice_1 = False
        for res in exec_chain["execution_results"]:
            if res["slice_id"] == "slice_0":
                found_slice_0 = True
                assert "proof_execution" not in res
            if res["slice_id"] == "slice_1":
                found_slice_1 = True
                assert "proof_execution" in res
                assert res["proof_execution"]["success"] is True

        assert found_slice_0, "slice_0 not found in execution_results"
        assert found_slice_1, "slice_1 not found in execution_results"

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_prove_with_tiling(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd, jstprove_available):
        """Verify that proving correctly handles tiled slices."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        # 1. Slice with tiling
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=14
        ))

        # 2. Compile
        compile_model(SimpleNamespace(
            path=str(slices_output_dir),
            input_file=None,
            layers=None,
            backend="jstprove"
        ))

        # 3. Run
        input_file = model_dir / "input.json"
        capfd.readouterr()
        run_inference(SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None,
            run_metadata_path=None
        ))
        out = capfd.readouterr().out
        run_dir = self._get_run_dir(out)
        
        # 4. Prove
        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(slices_output_dir), backend="jstprove"))
        
        # 5. Verify run_results.json
        run_results = json.loads((run_dir / "run_results.json").read_text())
        exec_results = run_results["execution_chain"]["execution_results"]
        
        # slice_0 is tiled
        s0_res = next(r for r in exec_results if r["slice_id"] == "slice_0")
        p_exec = s0_res["proof_execution"]
        
        assert p_exec["success"] is True
        assert p_exec["proof_file"] is None # Tiled slices don't have a single proof file
        assert "tile_proofs_info" in p_exec
        assert len(p_exec["tile_proofs_info"]) == 4
        
        for t_proof in p_exec["tile_proofs_info"]:
            assert t_proof["success"] is True
            assert "proof.json" in t_proof["proof_path"]
            assert Path(t_proof["proof_path"]).exists()
