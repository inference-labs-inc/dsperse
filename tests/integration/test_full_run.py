import json
import pytest
from pathlib import Path
from types import SimpleNamespace
import shutil

from dsperse.src.cli.full_run import full_run
from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.run import run_inference
from dsperse.src.cli.prove import run_proof
from dsperse.src.cli.verify import verify_proof

class TestFullRunE2E:
    def test_full_run_loop_slices(self, models_root, pre_compiled_net, tmp_path, copy_to, jstprove_available, capfd):
        """
        Test a workflow with looped single-slice proving and verification:
        - Run (using pre-compiled slices)
        - Loop through each slice and prove it as a single slice
        - Loop through each slice and verify it as a single slice
        """
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        input_file = models_root / "net" / "input.json"
        slices_output_dir = copy_to(pre_compiled_net, tmp_path / "slices")

        import re
        capfd.readouterr()
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None))
        out = capfd.readouterr().out

        match = re.search(r"Run data saved to (.+)", out)
        assert match
        run_dir = Path(match.group(1).strip())

        slice_ids = sorted([d.name for d in slices_output_dir.iterdir() if d.is_dir() and d.name.startswith("slice_")])
        assert slice_ids

        # 4. Prove (loop through slices that have jstprove compilation)
        proved_slices = []
        for slice_id in slice_ids:
            slice_path = slices_output_dir / slice_id
            run_slice_path = run_dir / slice_id
            jst_dir = slice_path / "payload" / "jstprove"

            # Only prove slices that were actually compiled with jstprove
            if not jst_dir.exists() or not any(jst_dir.iterdir()):
                continue

            run_proof(SimpleNamespace(run_dir=str(run_slice_path), slices_path=str(slice_path), backend="jstprove"))
            proved_slices.append(slice_id)

            assert (run_slice_path / "proof.json").exists()
            assert (run_slice_path / "run_results.json").exists()

            res = json.loads((run_slice_path / "run_results.json").read_text())
            assert "proof_execution" in res["execution_chain"]["execution_results"][0]

        assert len(proved_slices) >= 1, "At least one slice should have been proved"

        # 5. Verify (loop through proved slices)
        for slice_id in proved_slices:
            slice_path = slices_output_dir / slice_id
            run_slice_path = run_dir / slice_id

            verify_proof(SimpleNamespace(run_dir=str(run_slice_path), slices_path=str(slice_path), backend="jstprove"))

            res = json.loads((run_slice_path / "run_results.json").read_text())
            assert "verification_execution" in res["execution_chain"]["execution_results"][0]
            assert res["execution_chain"]["execution_results"][0]["verification_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_full_run_happy_path(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, analysis_output_dir: Path, jstprove_available, capfd):
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        input_file = model_dir / "input.json"
        actual_slices_dir = model_dir / "slices"
        actual_run_dir = model_dir / "run"
        if actual_slices_dir.exists():
            shutil.rmtree(actual_slices_dir)
        if actual_run_dir.exists():
            shutil.rmtree(actual_run_dir)

        args = SimpleNamespace(
            model_dir=str(model_dir),
            input_file=str(input_file),
            slices_dir=None,
            layers=None
        )

        try:
            capfd.readouterr()
            full_run(args)
            out = capfd.readouterr().out

            assert "Step 1/5: Slicing model" in out
            assert "Step 2/5: Compiling slices (JSTprove circuitization)" in out
            assert "Step 3/5: Running inference over slices" in out
            assert "Step 4/5: Generating proof" in out
            assert "Step 5/5: Verifying proof" in out
            assert "Full pipeline completed" in out

            assert actual_slices_dir.exists()
            assert (actual_slices_dir / "metadata.json").exists()

            slice_ids = sorted([d.name for d in actual_slices_dir.iterdir() if d.is_dir() and d.name.startswith("slice_")])
            assert slice_ids

            compiled_count = 0
            for slice_id in slice_ids:
                jst_dir = actual_slices_dir / slice_id / "payload" / "jstprove"
                if jst_dir.exists() and any(jst_dir.iterdir()):
                    compiled_count += 1
            assert compiled_count >= 1, "At least one slice should have jstprove artifacts"

            assert actual_run_dir.exists()
            run_dirs = sorted(list(actual_run_dir.glob("run_*")))
            assert len(run_dirs) >= 1
            latest_run = run_dirs[-1]

            run_results_path = latest_run / "run_results.json"
            assert run_results_path.exists()

            run_results = json.loads(run_results_path.read_text())
            exec_chain = run_results["execution_chain"]

            proved_count = 0
            for res in exec_chain["execution_results"]:
                proof_exec = res.get("proof_execution", {})
                verif_exec = res.get("verification_execution", {})
                if proof_exec.get("success") and verif_exec.get("success"):
                    proved_count += 1
            assert proved_count >= 1, "At least one slice should have successful proof and verification"
        finally:
            if actual_slices_dir.exists():
                shutil.rmtree(actual_slices_dir, ignore_errors=True)
            if actual_run_dir.exists():
                shutil.rmtree(actual_run_dir, ignore_errors=True)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", ["doom"])
    def test_tiled_full_run(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, capfd, jstprove_available):
        """Verify end-to-end full cycle with tiling: slice -> compile -> run -> prove -> verify."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        # 1. Slice with tiling
        args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=1000  # doom input is 28x28, so 1000 triggers spatial tiling
        )
        slice_model(args)
        
        # 2. Compile (should use JSTprove by default or EZKL if JSTprove fails)
        # We'll force JSTprove for this test to be predictable
        from dsperse.src.compile.compiler import Compiler
        compiler = Compiler()
        compiler.compile(str(slices_output_dir), layers="0-4:jstprove; 5:ezkl")
        
        # 3. Run
        input_file = model_dir / "input.json"
        from dsperse.src.run.runner import Runner
        runner = Runner()
        run_results = runner.run(str(input_file), str(slices_output_dir))
        
        run_dir = Path(runner.last_run_dir)
        assert run_dir.exists()

        # Check that tiled slices were executed (method should be "tiled")
        slice_results = run_results.get("slice_results", {})
        tiled_slices = [s for s in slice_results.values() if (s.method if hasattr(s, 'method') else s.get("method")) == "tiled"]
        assert len(tiled_slices) >= 1, "At least one slice should be tiled"

        # 4. Prove
        from dsperse.src.prove.prover import Prover
        prover = Prover()
        prover.prove(str(run_dir), str(slices_output_dir))

        # 5. Verify
        from dsperse.src.verify.verifier import Verifier
        verifier = Verifier()
        verifier.verify(str(run_dir), str(slices_output_dir))

        # Assertions - output key instead of prediction
        assert run_results.get("output") is not None

        # Verify results in run_results.json
        run_results_path = run_dir / "run_results.json"
        final_results = json.loads(run_results_path.read_text())
        exec_results = final_results.get("execution_chain", {}).get("execution_results", [])

        # Check at least one tiled slice was proved and verified
        tiled_proved = 0
        for entry in exec_results:
            w_exec = entry.get("witness_execution", {})
            proof_exec = entry.get("proof_execution", {})
            verif_exec = entry.get("verification_execution", {})

            if w_exec.get("method") == "tiled":
                if proof_exec.get("success") and verif_exec.get("success"):
                    tiled_proved += 1
                    # Verify tile execution info exists
                    assert len(w_exec.get("tile_exec_infos", [])) >= 2

        assert tiled_proved >= 1, "At least one tiled slice should have successful proof and verification"
