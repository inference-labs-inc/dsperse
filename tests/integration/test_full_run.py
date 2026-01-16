import json
import pytest
from pathlib import Path
from types import SimpleNamespace
import os
import shutil

from dsperse.src.cli.full_run import full_run
from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model
from dsperse.src.cli.run import run_inference
from dsperse.src.cli.prove import run_proof
from dsperse.src.cli.verify import verify_proof

class TestFullRunE2E:
    @pytest.mark.parametrize("model_name", ["net"])
    def test_full_run_loop_slices(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, analysis_output_dir: Path, jstprove_available, capfd):
        """
        Test a workflow with looped single-slice proving and verification:
        - Slice -> Compile (default) -> Run (default)
        - Loop through each slice and prove it as a single slice
        - Loop through each slice and verify it as a single slice
        """
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        input_file = model_dir / "input.json"

        # 1. Slice
        slice_model(SimpleNamespace(model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"))

        # 2. Compile (default)
        compile_model(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), layers=None, backend=None))

        # 3. Run (default)
        capfd.readouterr() # clear buffers
        run_inference(SimpleNamespace(path=str(slices_output_dir), input_file=str(input_file), output_file=None, force_backend=None, run_metadata_path=None))
        out = capfd.readouterr().out
        
        # Extract run directory
        import re
        match = re.search(r"Run data saved to (.+)", out)
        assert match
        run_dir = Path(match.group(1).strip())

        # Determine slice IDs
        slice_ids = sorted([d.name for d in slices_output_dir.iterdir() if d.is_dir() and d.name.startswith("slice_")])
        assert slice_ids

        # 4. Prove (loop through each slice as single slice)
        for slice_id in slice_ids:
            slice_path = slices_output_dir / slice_id
            run_slice_path = run_dir / slice_id
            
            run_proof(SimpleNamespace(run_dir=str(run_slice_path), slices_path=str(slice_path), backend="jstprove"))
            
            assert (run_slice_path / "proof.json").exists()
            assert (run_slice_path / "run_results.json").exists()
            
            res = json.loads((run_slice_path / "run_results.json").read_text())
            assert "proof_execution" in res["execution_chain"]["execution_results"][0]

        # 5. Verify (loop through each slice as single slice)
        for slice_id in slice_ids:
            slice_path = slices_output_dir / slice_id
            run_slice_path = run_dir / slice_id
            
            verify_proof(SimpleNamespace(run_dir=str(run_slice_path), slices_path=str(slice_path), backend="jstprove"))
            
            res = json.loads((run_slice_path / "run_results.json").read_text())
            assert "verification_execution" in res["execution_chain"]["execution_results"][0]
            assert res["execution_chain"]["execution_results"][0]["verification_execution"]["success"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_full_run_happy_path(self, model_name: str, model_dir: Path, slices_output_dir: Path, run_output_dir: Path, analysis_output_dir: Path, jstprove_available, capfd):
        """
        Test the full-run command happy path:
        - Slices are created
        - Compilation uses jstprove (as requested by user)
        - Inference run
        - Proof generated
        - Verification completed
        """
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        input_file = model_dir / "input.json"
        
        args = SimpleNamespace(
            model_dir=str(model_dir),
            input_file=str(input_file),
            slices_dir=None,
            layers=None
        )

        capfd.readouterr() # clear buffers
        full_run(args)
        out = capfd.readouterr().out
        
        # Verify output
        assert "Step 1/5: Slicing model" in out
        assert "Step 2/5: Compiling slices (JSTprove circuitization)" in out
        assert "Step 3/5: Running inference over slices" in out
        assert "Step 4/5: Generating proof" in out
        assert "Step 5/5: Verifying proof" in out
        assert "Full pipeline completed" in out

        # Verify artifacts
        assert slices_output_dir.exists()
        assert (slices_output_dir / "metadata.json").exists()
        
        slice_ids = sorted([d.name for d in slices_output_dir.iterdir() if d.is_dir() and d.name.startswith("slice_")])
        assert slice_ids

        for slice_id in slice_ids:
            jst_dir = slices_output_dir / slice_id / "jstprove"
            assert jst_dir.exists()
            assert any(jst_dir.iterdir())

        assert run_output_dir.exists()
        run_dirs = sorted(list(run_output_dir.glob("run_*")))
        assert len(run_dirs) >= 1
        latest_run = run_dirs[-1]
        
        run_results_path = latest_run / "run_results.json"
        assert run_results_path.exists()
        
        run_results = json.loads(run_results_path.read_text())
        exec_chain = run_results["execution_chain"]
        
        for res in exec_chain["execution_results"]:
            assert "proof_execution" in res
            assert res["proof_execution"]["success"] is True
            assert "verification_execution" in res
            assert res["verification_execution"]["success"] is True

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
            tile_size=14  # doom input is 28x28, so 14 triggers tiling for 2 slices
        )
        slice_model(args)
        
        # 2. Compile (should use JSTprove by default or EZKL if JSTprove fails)
        # We'll force JSTprove for this test to be predictable
        from dsperse.src.compile.compiler import Compiler
        compiler = Compiler()
        compiler.compile(str(slices_output_dir), layers="jstprove")
        
        # 3. Run
        input_file = model_dir / "input.json"
        from dsperse.src.run.runner import Runner
        runner = Runner()
        run_results = runner.run(str(input_file), str(slices_output_dir))
        
        run_dir = Path(runner.last_run_dir)
        assert run_dir.exists()
        
        # Check if tiled slices were executed
        tiled_slice_0_dir = run_dir / "slice_0" / "tile_0"
        assert tiled_slice_0_dir.exists(), "Tiled execution directory missing"
        
        # 4. Prove
        from dsperse.src.prover import Prover
        prover = Prover()
        prove_results = prover.prove(str(run_dir), str(slices_output_dir))
        
        # 5. Verify
        from dsperse.src.verifier import Verifier
        verifier = Verifier()
        verify_results = verifier.verify(str(run_dir), str(slices_output_dir))
        
        # Assertions
        assert run_results["prediction"] is not None
        
        # Verify that proofs were generated for tiled slices
        exec_results = run_results.get("execution_chain", {}).get("execution_results", [])
        for entry in exec_results:
            slice_id = entry.get("slice_id")
            w_exec = entry.get("witness_execution", {})
            
            if w_exec.get("method") == "tiled_parallel":
                # Check that input_file and output_file are populated for tiled slice
                assert w_exec.get("input_file") != "unknown"
                assert w_exec.get("output_file") != "unknown"
                assert "input.json" in w_exec.get("input_file")
                assert "output.json" in w_exec.get("output_file")

        # Verify results in run_results.json
        run_results_path = run_dir / "run_results.json"
        final_results = json.loads(run_results_path.read_text())
        exec_results = final_results.get("execution_chain", {}).get("execution_results", [])
        
        for entry in exec_results:
            slice_id = entry.get("slice_id")
            proof_exec = entry.get("proof_execution", {})
            verif_exec = entry.get("verification_execution", {})
            w_exec = entry.get("witness_execution", {})
            
            if w_exec.get("method") == "tiled_parallel":
                 assert proof_exec.get("success"), f"Proof failed for tiled slice {slice_id}: {proof_exec.get('error')}"
                 assert "tile_proofs_info" in proof_exec
                 assert len(proof_exec["tile_proofs_info"]) == 4
                 assert proof_exec["proof_file"] is None
                 
                 assert verif_exec.get("success"), f"Verification failed for tiled slice {slice_id}: {verif_exec.get('error')}"
                 assert "tile_verifs_info" in verif_exec
                 assert len(verif_exec["tile_verifs_info"]) == 4
