"""
Orchestration for various provers.
"""
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


def _prove_slice_worker(args: tuple) -> dict:
    """
    Worker function for parallel proof generation.
    Must be module-level for pickling by ProcessPoolExecutor.
    """
    (slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path, tiling_info, run_path, slice_dir) = args

    result = {
        'slice_id': slice_id,
        'success': False,
        'method': None,
        'proof_path': str(proof_path) if not tiling_info else None,
        'time_sec': 0,
        'error': None,
        'attempted_jstprove': preferred == "jstprove",
        'attempted_ezkl': preferred == "ezkl",
        'tile_proofs_info': None
    }

    start = time.time()

    if tiling_info:
        num_tiles = tiling_info["num_tiles"]
        tile_results = []
        method = None

        for tile_idx in range(num_tiles):
            tile_name = f"tile_{tile_idx}"
            tile_run_dir = Path(run_path) / slice_id / tile_name

            if preferred == "jstprove":
                tile_witness_path = tile_run_dir / "output_witness.bin"
            else:
                tile_witness_path = tile_run_dir / "output.json"

            tile_proof_path = Path(run_path) / slice_id / tile_name / "proof.json"
            os.makedirs(tile_proof_path.parent, exist_ok=True)

            if not tile_witness_path.exists():
                tile_results.append({"tile_idx": tile_idx, "success": False, "error": "witness_missing"})
                continue

            if not circuit_path or not os.path.exists(circuit_path):
                tile_results.append({"tile_idx": tile_idx, "success": False, "error": "circuit_missing"})
                continue

            try:
                if preferred == "jstprove":
                    from dsperse.src.backends.jstprove import JSTprove
                    backend = JSTprove()
                    ok, res = backend.prove(
                        witness_path=str(tile_witness_path),
                        circuit_path=str(circuit_path),
                        proof_path=str(tile_proof_path),
                    )
                    method = "jstprove_prove"
                else:
                    from dsperse.src.backends.ezkl import EZKL
                    backend = EZKL()
                    ok, res = backend.prove(
                        witness_path=str(tile_witness_path),
                        model_path=str(circuit_path),
                        proof_path=str(tile_proof_path),
                        pk_path=str(pk_path) if pk_path else None,
                        settings_path=settings_path,
                    )
                    method = "ezkl_prove"

                tile_results.append({
                    "tile_idx": tile_idx,
                    "success": ok,
                    "proof_path": str(tile_proof_path),
                    "error": None if ok else str(res)
                })
            except Exception as e:
                tile_results.append({"tile_idx": tile_idx, "success": False, "error": str(e)})

        result['success'] = all(r["success"] for r in tile_results)
        result['method'] = method
        result['tile_proofs_info'] = tile_results
        result['error'] = None if result['success'] else "One or more tiles failed to prove"
    else:
        os.makedirs(Path(proof_path).parent, exist_ok=True)
        try:
            if preferred == "jstprove":
                from dsperse.src.backends.jstprove import JSTprove
                backend = JSTprove()
                ok, res = backend.prove(
                    witness_path=str(witness_path),
                    circuit_path=str(circuit_path),
                    proof_path=str(proof_path),
                )
                result['success'] = ok
                result['method'] = "jstprove_prove"
                result['error'] = None if ok else str(res)
            else:
                if not pk_path or not os.path.exists(pk_path):
                    result['error'] = f"Proving key not found at {pk_path}"
                else:
                    from dsperse.src.backends.ezkl import EZKL
                    backend = EZKL()
                    ok, res = backend.prove(
                        witness_path=str(witness_path),
                        model_path=str(circuit_path),
                        proof_path=str(proof_path),
                        pk_path=str(pk_path),
                        settings_path=settings_path,
                    )
                    result['success'] = ok
                    result['method'] = "ezkl_prove"
                    result['error'] = None if ok else str(res)
        except Exception as e:
            result['error'] = str(e)
            result['method'] = f"{preferred}_prove"

    result['time_sec'] = time.time() - start
    return result

class Prover:
    """
    Orchestrator for proving model execution slices.
    """

    def __init__(self, parallel: int = 1):
        """
        Initialize the prover.

        Args:
            parallel: Number of parallel processes for proof generation (default: 1)
        """
        self.parallel = max(1, parallel)
        try:
            self.ezkl_runner = EZKL()
        except RuntimeError:
            self.ezkl_runner = None
            logger.warning("EZKL CLI not available. EZKL backend will be disabled.")

        try:
            self.jstprove_runner = JSTprove()
        except Exception:
            self.jstprove_runner = None
            logger.warning("JSTprove CLI not available. JSTprove backend will be disabled.")

    # ------------------------ Small helpers (no behavior change) ------------------------
    @staticmethod
    def _resolve_slice_artifacts(
        slice_dir: Path,
        meta: dict,
        backend: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """Resolve circuit, pk, and settings paths under a given slice directory.
        Selection is backend-aware:
        - For 'ezkl': prefer `ezkl_*` paths, fallback to generic fields
        - For 'jstprove': prefer `jstprove_*` paths, fallback to generic fields
        - Otherwise: use generic fields only
        Returns (circuit_path, pk_path, settings_path), any may be None.
        """
        b = (backend or "").lower()
        if b == "ezkl":
            circuit_rel = meta.get("ezkl_circuit_path") or meta.get("circuit_path") or meta.get("compiled")
            pk_rel = meta.get("ezkl_pk_path") or meta.get("pk_path")
            settings_rel = meta.get("ezkl_settings_path") or meta.get("settings_path")
        elif b == "jstprove":
            circuit_rel = meta.get("jstprove_circuit_path") or meta.get("circuit_path") or meta.get("compiled")
            # JSTprove does not use pk, but keep any provided value for API symmetry
            pk_rel = meta.get("pk_path")
            settings_rel = meta.get("jstprove_settings_path") or meta.get("settings_path")
        else:
            circuit_rel = meta.get("circuit_path") or meta.get("compiled")
            pk_rel = meta.get("pk_path")
            settings_rel = meta.get("settings_path")

        circuit_path = Utils.resolve_under_slice(slice_dir, circuit_rel)
        pk_path = Utils.resolve_under_slice(slice_dir, pk_rel)
        settings_path = Utils.resolve_under_slice(slice_dir, settings_rel)

        if settings_path and not os.path.exists(settings_path):
            logger.warning(f"Settings file not found at {settings_path}; proceeding without it.")
            settings_path = None

        return circuit_path, pk_path, settings_path

    @staticmethod
    def _get_witness_backend_from_run(run_path: Path, slice_id: str) -> str | None:
        """Inspect run_results.json to see which backend produced the witness for a slice.
        Returns 'jstprove', 'ezkl', or None if not found.
        """
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        exec_chain = (rr or {}).get("execution_chain") or {}
        exec_results = exec_chain.get("execution_results") or []
        for entry in exec_results:
            if entry.get("slice_id") == slice_id:
                w = entry.get("witness_execution") or {}
                method = (w.get("method") or "").lower()
                if method == "tiled_parallel":
                    # Tiled execution: check first tile to see backend
                    tile_infos = w.get("tile_exec_infos", [])
                    if tile_infos:
                        method = (tile_infos[0].get("method", "") or "").lower()

                if method.startswith("jstprove"):
                    return "jstprove"
                if method.startswith("ezkl"):
                    return "ezkl"
        return None

    def _select_proving_backend(self, run_path: Path, slice_id: str, meta: dict) -> str:
        """Choose proving backend with this priority: witness backend → meta backend → jstprove.
        Returns 'jstprove' or 'ezkl'.
        """
        from_run = self._get_witness_backend_from_run(run_path, slice_id)
        if from_run in ("jstprove", "ezkl"):
            return from_run
        meta_backend = (meta.get("backend") or "").lower()
        if meta_backend in ("jstprove", "ezkl"):
            return meta_backend
        return "jstprove"

    @staticmethod
    def _get_witness_file_from_run(run_path: Path, slice_id: str) -> str | None:
        """Return concrete witness file path recorded by runner for a slice, if any."""
        try:
            rr = Utils.load_run_results(run_path)
        except Exception:
            return None
        exec_chain = (rr or {}).get("execution_chain") or {}
        exec_results = exec_chain.get("execution_results") or []
        for entry in exec_results:
            if entry.get("slice_id") == slice_id:
                w = entry.get("witness_execution") or {}
                wf = w.get("witness_file") or w.get("witness_path")
                return wf
        return None

    def _prove_with_backend(
        self,
        backend: str,
        witness_path: Path,
        circuit_path: str,
        proof_path: Path,
        pk_path: str | None,
        settings_path: str | None,
    ) -> tuple[bool, str, str | Path | None]:
        """Dispatch to the requested backend. Returns (success, method, result_or_error)."""
        if backend == "jstprove":
            if self.jstprove_runner is None:
                return False, "jstprove_prove", "JSTprove CLI not available"
            try:
                ok, res = self.jstprove_runner.prove(
                    witness_path=str(witness_path),
                    circuit_path=str(circuit_path),
                    proof_path=str(proof_path),
                )
                return ok, "jstprove_prove", res
            except Exception as e:
                return False, "jstprove_prove", str(e)

        # EZKL requires pk
        if not pk_path or not os.path.exists(pk_path):
            return False, "ezkl_prove", f"Proving key not found at {pk_path}"
        try:
            ok, res = self.ezkl_runner.prove(
                witness_path=str(witness_path),
                model_path=str(circuit_path),
                proof_path=str(proof_path),
                pk_path=str(pk_path),
                settings_path=settings_path,
            )
            return ok, "ezkl_prove", res
        except Exception as e:
            return False, "ezkl_prove", str(e)


    def prove_dirs(self, run_path: str | Path, dirs_path: str | Path, output_path: str | Path | None = None, backend: str | None = None) -> dict:
        """Prove all circuit-capable slices (JSTprove preferred, fallback EZKL) given a slices directory layout."""
        run_path = Path(run_path)
        dirs_path = Utils.dirs_root_from(Path(dirs_path))
        metadata = Utils.load_run_metadata(run_path)

        proofs: dict[str, dict] = {}
        proved_jst = 0
        proved_ezkl = 0

        nodes = ((metadata or {}).get("execution_chain") or {}).get("nodes", {})
        all_slices = (metadata or {}).get("slices", {})
        slices_iter = [(sid, all_slices.get(sid, {})) for sid, node in nodes.items() if node.get("use_circuit")]

        if not slices_iter:
            try:
                slices_iter = list(Utils.iter_circuit_slices(metadata))
            except Exception:
                slices_iter = []

        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to prove under run {run_path}. Nothing to do.")
            return Utils.load_run_results(run_path)

        if backend in ("jstprove", "ezkl"):
            filtered = []
            for slice_id, meta in slices_iter:
                wb = self._get_witness_backend_from_run(Path(run_path), slice_id)
                if wb == backend:
                    filtered.append((slice_id, meta))
            slices_iter = filtered
            if not slices_iter:
                logger.info(f"No slices found with witness backend '{backend}' to prove under run {run_path}.")
                return Utils.load_run_results(run_path)

        work_items = []
        for slice_id, meta in slices_iter:
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)
            preferred = self._select_proving_backend(Path(run_path), slice_id, meta)
            circuit_path, pk_path, settings_path = self._resolve_slice_artifacts(slice_dir, meta, preferred)

            tiling = meta.get("tiling")
            if tiling:
                witness_path = None
                proof_path = None
            else:
                if preferred == "jstprove":
                    wf = self._get_witness_file_from_run(Path(run_path), slice_id)
                    witness_path = Path(wf) if wf else (Path(run_path) / slice_id / "output_witness.bin")
                else:
                    witness_path = Utils.witness_path_for(run_path, slice_id)

                if not witness_path.exists():
                    logger.warning(f"Skipping {slice_id}: witness not found at {witness_path}")
                    continue

                proof_path = Utils.proof_output_path(run_path, slice_id, output_path)

                if not circuit_path or not os.path.exists(circuit_path):
                    logger.warning(f"Skipping {slice_id}: compiled circuit not found ({circuit_path})")
                    continue

            work_items.append((slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path, tiling, str(run_path), str(slice_dir)))

        total = len(work_items)

        if self.parallel > 1 and len(work_items) > 1:
            logger.info(f"Proving {len(work_items)} slices with {self.parallel} parallel processes...")
            results = []
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(_prove_slice_worker, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    slice_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Slice {slice_id} proving failed: {e}")
                        results.append({'slice_id': slice_id, 'success': False, 'method': None, 'error': str(e)})
        else:
            results = [_prove_slice_worker(item) for item in work_items]

        for result in results:
            slice_id = result['slice_id']
            proofs[slice_id] = result

            if result['success']:
                method = result.get('method')
                if method == "jstprove_prove":
                    proved_jst += 1
                elif method == "ezkl_prove":
                    proved_ezkl += 1
            else:
                logger.error(f"Proof failed for {slice_id}: {result.get('error')}")

        run_results = Utils.load_run_results(run_path)
        run_results, _ = Utils.merge_execution_into_run_results(run_results, proofs, "proof")
        exec_chain = run_results.setdefault("execution_chain", {})
        exec_chain["jstprove_proved_slices"] = int(proved_jst)
        exec_chain["ezkl_proved_slices"] = int(proved_ezkl)
        exec_chain["ezkl_verified_slices"] = exec_chain.get("ezkl_verified_slices", 0) if proved_ezkl == 0 else 0

        Utils.save_run_results(run_path, run_results)
        Utils.update_metadata_after_execution(run_path, total, proved_jst + proved_ezkl, "proof")
        return run_results

    def prove_dslice(self, run_path: str | Path, dslice_path: str | Path, output_path: str | Path | None = None, backend: str | None = None) -> dict:
        """Convert dslice -> dirs, delegate to `prove_dirs`, then convert back to dslice."""
        temp_dirs = Converter.convert(dslice_path, output_type="dirs", cleanup=False)

        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        summary = self.prove_dirs(run_path, dirs_root, output_path, backend=backend)

        # Convert back to dslice packaging (non-destructive)
        Converter.convert(str(dirs_root), output_type="dslice", cleanup=False)

        return summary

    def prove_dsperse(self, run_path: str | Path, dsperse_path: str | Path, output_path: str | Path | None = None, backend: str | None = None) -> dict:
        """Convert dsperse -> dirs, delegate to `prove_dirs`, then convert back to dsperse."""
        temp_dirs = Converter.convert(dsperse_path, output_type="dirs", cleanup=False)

        dirs_root = Utils.dirs_root_from(Path(temp_dirs))
        summary = self.prove_dirs(run_path, dirs_root, output_path, backend=backend)

        Converter.convert(str(dirs_root), output_type="dsperse", cleanup=False)

        return summary

    def prove(self, run_path: str | Path, model_dir: str | Path, output_path: str | Path | None = None, backend: str | None = None) -> dict:
        """Route to the appropriate prove path based on `model_dir` packaging.
        Supports two modes for run_path:
        - Run-root mode: run_path contains metadata.json (prove all circuit-capable slices)
        - Single-slice mode: run_path contains input.json and output.json (prove just this slice)
        """
        run_path = Path(run_path)

        # Determine run mode
        is_run_root = (run_path / "metadata.json").exists()
        is_slice_run = (run_path / "input.json").exists() and (run_path / "output.json").exists()

        detected = Converter.detect_type(model_dir)

        if is_run_root:
            # Existing behavior
            Utils.load_run_metadata(run_path)
            if detected == "dslice":
                return self.prove_dslice(run_path, model_dir, output_path, backend=backend)
            if detected == "dsperse":
                return self.prove_dsperse(run_path, model_dir, output_path, backend=backend)
            if detected == "dirs":
                return self.prove_dirs(run_path, model_dir, output_path, backend=backend)
            raise ValueError(f"Unsupported data type for proving: {detected}")

        # Single-slice mode: no run metadata, but we must have per-slice files
        if is_slice_run:
            # Single-slice mode requires explicit backend
            if backend not in ("jstprove", "ezkl"):
                raise ValueError("Single-slice proving requires explicit backend: 'jstprove' or 'ezkl'.")
            # Ensure we operate on a directory layout for the provided slices/model path
            dirs_model_path = model_dir
            if detected != "dirs":
                dirs_model_path = Converter.convert(str(model_dir), output_type="dirs", cleanup=False)

            # Prove using the directory layout, while remembering original packaging for reconversion
            result = self._prove_single_slice(run_path, dirs_model_path, detected, backend)

            # Convert back to the original packaging if needed
            if detected != "dirs":
                Converter.convert(str(Utils.dirs_root_from(Path(dirs_model_path))), output_type=detected, cleanup=True)

            return result

        # Otherwise invalid run_dir
        raise FileNotFoundError(f"Run path invalid; expected run-root (metadata.json) or per-slice (input.json + output.json) at {run_path}")

    def _prove_single_slice(self, run_path: Path, model_dir: str | Path, detected: str, backend: str) -> dict:
        """Internal: prove exactly one slice using a per-slice run directory.
        Expects `<run_path>/input.json` and `<run_path>/output.json` to exist and `model_dir` to
        resolve to a single-slice metadata source (slice dir or .dslice).
        Writes `proof.json` and updates/creates `run_results.json` in `run_path`.
        """
        # Normalize/expand model_dir into run-like metadata so we can get artifact paths
        sdir = Path(model_dir)

        # Generate run-like metadata strictly from provided slices path
        run_meta = RunnerAnalyzer.generate_run_metadata(Path(sdir if sdir.is_dir() else Path(sdir)), save_path=None, original_format=detected)
        model_slices = (run_meta or {}).get("slices", {})

        # Expect exactly one slice in provided metadata for single-slice proving
        if len(model_slices) != 1:
            raise ValueError(f"Slices path must represent exactly one slice for single-slice proving; found {len(model_slices)} in {model_dir}")

        # Extract the only slice id and meta
        (slice_id, meta), = model_slices.items()

        # Use explicitly provided backend (single-slice mode enforces this)
        preferred = (backend or "").lower()

        # Resolve artifacts relative to the provided slices path root, backend-aware
        dirs_root = Utils.dirs_root_from(Path(model_dir))
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)
        model_path_res, pk_path_res, settings_path_res = self._resolve_slice_artifacts(slice_dir, meta, preferred)

        # Validate required inputs (circuit and witness); pk may be absent for JSTprove
        if not model_path_res or not os.path.exists(model_path_res):
            raise FileNotFoundError(f"Compiled circuit not found for {slice_id} at {model_path_res}")

        proof_path = Path(run_path) / "proof.json"
        proof_path.parent.mkdir(parents=True, exist_ok=True)

        # Select witness path strictly per backend
        if preferred == "jstprove":
            wf = self._get_witness_file_from_run(run_path, slice_id)
            witness_path = Path(wf) if wf else (Path(run_path) / "output_witness.bin")
        elif preferred == "ezkl":
            witness_path = Path(run_path) / "output.json"
        else:
            raise ValueError("Unsupported backend for single-slice proving. Use 'jstprove' or 'ezkl'.")
        if not witness_path.exists():
            raise FileNotFoundError(f"Witness not found at {witness_path}")

        start = time.time()
        success, method, result = self._prove_with_backend(
            preferred, witness_path, str(model_path_res), proof_path, pk_path_res, settings_path_res
        )
        elapsed = time.time() - start

        # Merge into run_results.json located in the per-slice dir
        proofs = {
            slice_id: {
                "success": bool(success),
                "proof_path": str(proof_path),
                "time_sec": elapsed,
                "method": method or "unknown",
                "attempted_jstprove": preferred == "jstprove",
                "attempted_ezkl": preferred == "ezkl",
                "error": None if success else str(result),
            }
        }
        run_results = Utils.load_run_results(Path(run_path))
        run_results, _ = Utils.merge_execution_into_run_results(run_results, proofs, "proof")
        exec_chain = run_results.setdefault("execution_chain", {})
        if method == "jstprove_prove":
            exec_chain["jstprove_proved_slices"] = int(exec_chain.get("jstprove_proved_slices", 0)) + 1
        elif method == "ezkl_prove":
            exec_chain["ezkl_proved_slices"] = int(exec_chain.get("ezkl_proved_slices", 0)) + 1

        existing_witness = int(exec_chain.get("ezkl_witness_slices", 0) or 0)
        if existing_witness == 0:
            exec_chain["ezkl_witness_slices"] = len(proofs)
        # Save run_results.json next to the per-slice files
        Utils.save_run_results(Path(run_path), run_results)

        return run_results


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/age",
        5: "../models/version"
    }

    model_dir = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(model_dir, "slices") # slices dir, or single slice, or dsperse file
    # slices_dir = os.path.join(slices_dir, "slice_0")  # give a single slice to test

    # Get run directory - use the latest run in the model's run directory
    run_dir = os.path.join(model_dir, "run")
    if not os.path.exists(run_dir):
        print(f"Run directory not found at {run_dir}, assuming input file provided.")

    # Find the latest run
    run_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith("run_")])
    if not run_dirs:
        print(f"Error: No runs found in {run_dir}")
        exit(1)

    latest_run = run_dirs[-1]
    run_path = os.path.join(run_dir, latest_run)

    # Initialize prover
    prover = Prover()

    # Run proving
    print(f"Proving run {latest_run} for model {base_paths[model_choice]}...")
    results = prover.prove(run_path, slices_dir, backend="jstprove")

    # Display results
    print(f"\nProving completed!")

    # Print details for each slice
    print("\nSlice details:")
    for slice_result in results["execution_chain"]["execution_results"]:
        # print(f"\n{slice_result}:")
        slice_id = slice_result["slice_id"]
        if "proof_execution" in slice_result:
            success = slice_result["proof_execution"]["success"]
            status = "Success" if success else "Failed"
            time_taken = slice_result["proof_execution"]["time_sec"]
            print(f"  {slice_id}: {status} (Time: {time_taken:.2f}s)")
