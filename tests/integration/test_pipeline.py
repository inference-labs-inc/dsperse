"""
Real integration tests that verify DSperse pipeline correctness.

These tests verify actual functionality:
- Slicing preserves model semantics (sliced output matches original)
- Tiling preserves computation (tiled output matches untiled)
- ONNX inference produces valid outputs
- JSTprove witness generation works
- Proofs are generated and verify successfully
"""
import json
import numpy as np
import onnxruntime as ort
import pytest
from pathlib import Path
from types import SimpleNamespace


class TestSlicingCorrectness:
    """Verify that slicing preserves model semantics."""

    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_sliced_output_matches_original(self, model_name: str, model_dir: Path, slices_output_dir: Path):
        """Sliced model chain should produce same output as original model."""
        from dsperse.src.cli.slice import slice_model
        from dsperse.src.run.runner import Runner

        original_model = model_dir / "model.onnx"
        input_file = model_dir / "input.json"

        with open(input_file) as f:
            input_data = json.load(f)
        input_tensor = np.array(input_data["input_data"], dtype=np.float32)

        session = ort.InferenceSession(str(original_model))
        original_output = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]

        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs"
        ))

        runner = Runner()
        results = runner.run(str(input_file), str(slices_output_dir))
        sliced_output = np.array(results["output"])

        assert sliced_output.shape == original_output.shape, \
            f"Shape mismatch: {sliced_output.shape} vs {original_output.shape}"
        np.testing.assert_allclose(sliced_output, original_output, rtol=1e-5, atol=1e-5,
            err_msg="Sliced output does not match original model output")


class TestTilingCorrectness:
    """Verify that tiling preserves computation."""

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_tiled_output_matches_untiled(self, model_name: str, model_dir: Path, tmp_path: Path):
        """Tiled execution should produce same output as untiled."""
        from dsperse.src.cli.slice import slice_model
        from dsperse.src.run.runner import Runner

        input_file = model_dir / "input.json"

        untiled_dir = tmp_path / "untiled"
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(untiled_dir),
            save_file=None,
            output_type="dirs"
        ))
        runner = Runner()
        untiled_results = runner.run(str(input_file), str(untiled_dir))
        untiled_output = np.array(untiled_results["output"])

        tiled_dir = tmp_path / "tiled"
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(tiled_dir),
            save_file=None,
            output_type="dirs",
            tile_size=14
        ))
        runner2 = Runner()
        tiled_results = runner2.run(str(input_file), str(tiled_dir))
        tiled_output = np.array(tiled_results["output"])

        assert tiled_output.shape == untiled_output.shape, \
            f"Shape mismatch: {tiled_output.shape} vs {untiled_output.shape}"
        np.testing.assert_allclose(tiled_output, untiled_output, rtol=1e-4, atol=1e-4,
            err_msg="Tiled output does not match untiled output")


class TestOnnxInference:
    """Verify ONNX inference produces valid results."""

    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_onnx_inference_succeeds(self, model_name: str, model_dir: Path, pre_sliced_net, pre_sliced_doom, tmp_path, copy_to):
        """ONNX backend should produce valid numeric output."""
        from dsperse.src.run.runner import Runner

        input_file = model_dir / "input.json"
        source = pre_sliced_net if model_name == "net" else pre_sliced_doom
        work_dir = copy_to(source, tmp_path / "slices")

        runner = Runner()
        results = runner.run(str(input_file), str(work_dir))

        assert "output" in results
        assert "slice_results" in results
        output = np.array(results["output"])
        assert not np.isnan(output).any(), "Output contains NaN"
        assert not np.isinf(output).any(), "Output contains Inf"
        assert all(getattr(r, 'success', False) for r in results["slice_results"].values()), \
            "Not all slices succeeded"


class TestWitnessGeneration:
    """Verify JSTprove witness generation works."""

    @pytest.mark.parametrize("model_name", ["net"])
    def test_jstprove_witness_generation(self, model_name: str, model_dir: Path, pre_compiled_net, tmp_path, copy_to, jstprove_available):
        """JSTprove should successfully generate witness for compiled slices."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        from dsperse.src.run.runner import Runner

        input_file = model_dir / "input.json"
        work_dir = copy_to(pre_compiled_net, tmp_path / "slices")

        runner = Runner()
        results = runner.run(str(input_file), str(work_dir))

        jstprove_slices = [
            (sid, r) for sid, r in results["slice_results"].items()
            if "jstprove" in (getattr(r, 'method', ''))
        ]
        assert len(jstprove_slices) >= 1, "No slices used JSTprove"
        for sid, r in jstprove_slices:
            success = getattr(r, 'success', False)
            assert success, f"JSTprove witness failed for {sid}"


class TestProofGeneration:
    """Verify proof generation works."""

    @pytest.mark.parametrize("model_name", ["net"])
    def test_proof_generation(self, model_name: str, model_dir: Path, pre_compiled_net, tmp_path, copy_to, jstprove_available, capfd):
        """Prover should successfully generate proofs for witnessed slices."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        from dsperse.src.cli.run import run_inference
        from dsperse.src.prove.prover import Prover
        import re

        input_file = model_dir / "input.json"
        work_dir = copy_to(pre_compiled_net, tmp_path / "slices")

        capfd.readouterr()
        run_inference(SimpleNamespace(
            path=str(work_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None
        ))
        out = capfd.readouterr().out
        match = re.search(r"Run data saved to (.+)", out)
        assert match, "Could not find run directory"
        run_dir = Path(match.group(1).strip())

        prover = Prover()
        prover.prove(str(run_dir), str(work_dir))

        run_results = json.loads((run_dir / "run_results.json").read_text())
        exec_results = run_results.get("execution_chain", {}).get("execution_results", [])

        proved_count = sum(
            1 for e in exec_results
            if e.get("proof_execution", {}).get("success")
        )
        assert proved_count >= 1, "No proofs were successfully generated"


class TestVerification:
    """Verify proof verification works."""

    @pytest.mark.parametrize("model_name", ["net"])
    def test_proof_verification(self, model_name: str, model_dir: Path, pre_compiled_net, tmp_path, copy_to, jstprove_available, capfd):
        """Verifier should successfully verify generated proofs."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        from dsperse.src.cli.run import run_inference
        from dsperse.src.prove.prover import Prover
        from dsperse.src.verify.verifier import Verifier
        import re

        input_file = model_dir / "input.json"
        work_dir = copy_to(pre_compiled_net, tmp_path / "slices")

        capfd.readouterr()
        run_inference(SimpleNamespace(
            path=str(work_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None
        ))
        out = capfd.readouterr().out
        match = re.search(r"Run data saved to (.+)", out)
        assert match
        run_dir = Path(match.group(1).strip())

        prover = Prover()
        prover.prove(str(run_dir), str(work_dir))

        verifier = Verifier()
        verifier.verify(str(run_dir), str(work_dir))

        run_results = json.loads((run_dir / "run_results.json").read_text())
        exec_results = run_results.get("execution_chain", {}).get("execution_results", [])

        verified_count = sum(
            1 for e in exec_results
            if e.get("verification_execution", {}).get("success")
        )
        assert verified_count >= 1, "No proofs were successfully verified"


class TestEndToEnd:
    """Full pipeline end-to-end test."""

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", ["net"])
    def test_full_pipeline_correctness(self, model_name: str, model_dir: Path, slices_output_dir: Path, jstprove_available, capfd):
        """
        Full pipeline: slice -> compile -> run -> prove -> verify.
        Verifies output matches original AND proofs verify.
        """
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        from dsperse.src.cli.slice import slice_model
        from dsperse.src.cli.compile import compile_model
        from dsperse.src.cli.run import run_inference
        from dsperse.src.prove.prover import Prover
        from dsperse.src.verify.verifier import Verifier
        import re

        original_model = model_dir / "model.onnx"
        input_file = model_dir / "input.json"

        with open(input_file) as f:
            input_data = json.load(f)
        input_tensor = np.array(input_data["input_data"], dtype=np.float32)
        session = ort.InferenceSession(str(original_model))
        original_output = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]

        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs"
        ))

        compile_model(SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            layers=None,
            backend="jstprove"
        ))

        capfd.readouterr()
        run_inference(SimpleNamespace(
            path=str(slices_output_dir),
            input_file=str(input_file),
            output_file=None,
            force_backend=None
        ))
        out = capfd.readouterr().out
        match = re.search(r"Run data saved to (.+)", out)
        assert match
        run_dir = Path(match.group(1).strip())

        run_results = json.loads((run_dir / "run_results.json").read_text())
        sliced_output = np.array(run_results["output"])

        np.testing.assert_allclose(sliced_output, original_output, rtol=1e-4, atol=1e-4,
            err_msg="Sliced output does not match original")

        prover = Prover()
        prover.prove(str(run_dir), str(slices_output_dir))

        verifier = Verifier()
        verifier.verify(str(run_dir), str(slices_output_dir))

        final_results = json.loads((run_dir / "run_results.json").read_text())
        exec_results = final_results.get("execution_chain", {}).get("execution_results", [])

        verified_count = sum(
            1 for e in exec_results
            if e.get("verification_execution", {}).get("success")
        )
        assert verified_count >= 1, "Pipeline completed but no proofs verified"
