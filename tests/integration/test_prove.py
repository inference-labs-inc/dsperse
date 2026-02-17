import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model
from dsperse.src.cli.run import run_inference
from dsperse.src.cli.prove import run_proof

class TestProveE2E:
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
    def test_prove_happy_path(self, model_name: str, model_dir: Path, pre_compiled_net_both: Path, tmp_path: Path, copy_to, capfd, parse_run_dir):
        work_dir = copy_to(pre_compiled_net_both, tmp_path / "slices")

        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(work_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = parse_run_dir(out)

        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(work_dir), backend=None, parallel=1, tiles=None))
        out = capfd.readouterr().out

        assert "Proof generation completed" in out
        results = self._verify_prove_artifacts(run_dir)
        exec_chain = results["execution_chain"]

        for res in exec_chain["execution_results"]:
            assert "proof_execution" in res
            assert res["proof_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_prove_single_slice(self, model_name: str, model_dir: Path, pre_compiled_net: Path, tmp_path: Path, copy_to, capfd, parse_run_dir):
        work_dir = copy_to(pre_compiled_net, tmp_path / "slices")

        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(work_dir), input_file=str(input_file), output_file=None, force_backend="jstprove", run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = parse_run_dir(out)

        slice_0_dir = work_dir / "slice_0"
        run_slice_0_dir = run_dir / "slice_0"

        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_slice_0_dir), slices_path=str(slice_0_dir), backend="jstprove", parallel=1, tiles=None))
        out = capfd.readouterr().out

        assert "Proof generation completed" in out

        results = self._verify_prove_artifacts(run_slice_0_dir)
        exec_chain = results["execution_chain"]

        slice_0_res = next((res for res in exec_chain["execution_results"] if res["slice_id"] == "slice_0"), None)
        assert slice_0_res is not None
        assert "proof_execution" in slice_0_res
        assert slice_0_res["proof_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_prove_mixed_backends(self, model_name: str, model_dir: Path, pre_sliced_net: Path, tmp_path: Path, copy_to, capfd, parse_run_dir):
        try:
            from dsperse.src.backends.jstprove import JSTprove
            from dsperse.src.backends.ezkl import EZKL
            _j, _e = JSTprove(), EZKL()
        except Exception as e:
            pytest.skip(f"Backends unavailable: {e}")

        work_dir = copy_to(pre_sliced_net, tmp_path / "slices")

        layers = "0:jstprove"
        compile_model(SimpleNamespace(path=str(work_dir), input_file=None, layers=layers, backend=None))

        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(work_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = parse_run_dir(out)

        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(work_dir), backend=None, parallel=1, tiles=None))
        out = capfd.readouterr().out

        assert "Proof generation completed" in out
        results = self._verify_prove_artifacts(run_dir)
        assert results["execution_chain"]["jstprove_proved_slices"] > 0

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_prove_backend_filter_doom(self, model_name: str, model_dir: Path, pre_sliced_doom: Path, tmp_path: Path, copy_to, capfd, parse_run_dir):
        try:
            from dsperse.src.backends.jstprove import JSTprove
            from dsperse.src.backends.ezkl import EZKL
            _j, _e = JSTprove(), EZKL()
        except Exception as e:
            pytest.skip(f"Backends unavailable: {e}")

        work_dir = copy_to(pre_sliced_doom, tmp_path / "slices")

        layers = "0:jstprove;1:ezkl"
        compile_model(SimpleNamespace(path=str(work_dir), input_file=None, layers=layers, backend=None))

        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(work_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        run_dir = parse_run_dir(out)

        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(work_dir), backend="ezkl", parallel=1, tiles=None))
        out = capfd.readouterr().out

        results = self._verify_prove_artifacts(run_dir)
        exec_chain = results["execution_chain"]

        assert exec_chain.get("jstprove_proved_slices", 0) == 0
        assert exec_chain.get("ezkl_proved_slices", 0) > 0

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
    def test_prove_with_tiling(self, model_name: str, model_dir: Path, pre_compiled_doom_tiled_14: Path, tmp_path: Path, copy_to, capfd, parse_run_dir):
        work_dir = copy_to(pre_compiled_doom_tiled_14, tmp_path / "slices")

        input_file = model_dir / "input.json"
        capfd.readouterr()
        run_inference(SimpleNamespace(
            path=str(work_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None,
            run_metadata_path=None
        ))
        out = capfd.readouterr().out
        run_dir = parse_run_dir(out)

        capfd.readouterr()
        run_proof(SimpleNamespace(run_dir=str(run_dir), slices_path=str(work_dir), backend="jstprove", parallel=1, tiles=None))

        run_results = json.loads((run_dir / "run_results.json").read_text())
        exec_results = run_results["execution_chain"]["execution_results"]

        s0_res = next(r for r in exec_results if r["slice_id"] == "slice_0")
        p_exec = s0_res["proof_execution"]

        assert p_exec["success"] is True
        assert p_exec["proof_file"] is None
        assert "tile_proofs_info" in p_exec
        assert len(p_exec["tile_proofs_info"]) == 4

        for t_proof in p_exec["tile_proofs_info"]:
            assert t_proof["success"] is True
            assert "proof.json" in t_proof["proof_path"]
            assert (run_dir / t_proof["proof_path"]).exists()

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", ["doom"])
    def test_prove_tiled_dslice_loop(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path,
                                     jstprove_available, capfd, parse_run_dir):
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(hardcoded_output_dir),
            save_file=None,
            output_type="dslice",
            tile_size=14
        ))

        compile_model(SimpleNamespace(path=str(hardcoded_output_dir), input_file=None, layers=None, backend="jstprove"))

        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(hardcoded_output_dir), input_file=str(input_file), output_file=None,
                                      force_backend="jstprove", run_metadata_path=None))
        run_dir = parse_run_dir(capfd.readouterr().out)

        compiled_slices = [(0, True), (2, True), (4, False)]
        for i, is_tiled in compiled_slices:
            slice_id = f"slice_{i}"
            dslice_path = hardcoded_output_dir / f"{slice_id}.dslice"
            run_slice_dir = run_dir / slice_id

            run_proof(SimpleNamespace(run_dir=str(run_slice_dir), slices_path=str(dslice_path), backend="jstprove", parallel=1, tiles=None))

            results = self._verify_prove_artifacts(run_slice_dir)
            s_res = next(r for r in results["execution_chain"]["execution_results"] if r["slice_id"] == slice_id)
            p_exec = s_res["proof_execution"]

            assert p_exec["success"] is True
            if is_tiled:
                assert "tile_proofs_info" in p_exec
                assert len(p_exec["tile_proofs_info"]) == 4
            else:
                assert p_exec["proof_file"] is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", ["doom"])
    def test_prove_partial_tile_ranges_cli(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path,
                                           jstprove_available, capfd, parse_run_dir):
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(hardcoded_output_dir), save_file=None,
                                    output_type="dirs", tile_size=14))
        compile_model(SimpleNamespace(path=str(hardcoded_output_dir), input_file=None, layers=None, backend="jstprove"))

        input_file = model_dir / "input.json"
        run_inference(SimpleNamespace(path=str(hardcoded_output_dir), input_file=str(input_file), output_file=None,
                                      force_backend="jstprove", run_metadata_path=None))
        run_dir = parse_run_dir(capfd.readouterr().out)

        s0_dir = hardcoded_output_dir / "slice_0"
        s0_run = run_dir / "slice_0"

        run_proof(SimpleNamespace(run_dir=str(s0_run), slices_path=str(s0_dir), backend="jstprove", parallel=1, tiles="0-1"))
        res = json.loads((s0_run / "run_results.json").read_text())
        assert len(res["execution_chain"]["execution_results"][0]["proof_execution"]["tile_proofs_info"]) == 2

        run_proof(SimpleNamespace(run_dir=str(s0_run), slices_path=str(s0_dir), backend="jstprove", parallel=1, tiles="2-3"))
        res = json.loads((s0_run / "run_results.json").read_text())
        assert len(res["execution_chain"]["execution_results"][0]["proof_execution"]["tile_proofs_info"]) == 2

        s2_dir = hardcoded_output_dir / "slice_2"
        s2_run = run_dir / "slice_2"

        run_proof(SimpleNamespace(run_dir=str(s2_run), slices_path=str(s2_dir), backend="jstprove", parallel=1, tiles="0,1"))
        res = json.loads((s2_run / "run_results.json").read_text())
        assert len(res["execution_chain"]["execution_results"][0]["proof_execution"]["tile_proofs_info"]) == 2

        run_proof(SimpleNamespace(run_dir=str(s2_run), slices_path=str(s2_dir), backend="jstprove", parallel=1, tiles="2"))
        res = json.loads((s2_run / "run_results.json").read_text())
        tile_info = res["execution_chain"]["execution_results"][0]["proof_execution"]["tile_proofs_info"]
        assert tile_info[0]["tile_idx"] == 2

        run_proof(SimpleNamespace(run_dir=str(s2_run), slices_path=str(s2_dir), backend="jstprove", parallel=1, tiles="3"))
        res = json.loads((s2_run / "run_results.json").read_text())
        tile_info = res["execution_chain"]["execution_results"][0]["proof_execution"]["tile_proofs_info"]
        assert tile_info[0]["tile_idx"] == 3
