"""
Orchestration for verifying proofs.
"""
import logging
import os
from pathlib import Path

from dsperse.src.analyzers.schema import TileResult, SliceResult, Backend, ExecutionMethod
from dsperse.src.backends.dispatch import WITNESS_FILENAME, resolve_circuit_path
from dsperse.src.pipeline_stage import PipelineStage
from dsperse.src.verify.utils.verifier_utils import VerifierUtils
from dsperse.src.utils.pipeline_utils import select_backend, get_witness_file, initialize_stage_metadata, filter_circuit_slices
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class Verifier(PipelineStage):

    def verify(self, run_path, model_path, backend=None, tiles_range=None):
        return self._dispatch(run_path, model_path, backend=backend, tiles_range=tiles_range)

    def _execute_dirs(self, run_path, dirs_path, _output_path, backend):
        run_path, dirs_path = Path(run_path), Utils.dirs_root_from(Path(dirs_path))
        metadata = initialize_stage_metadata(run_path, dirs_path)
        run_results = Utils.load_run_results(run_path)

        slices_iter = filter_circuit_slices(metadata, backend, run_path)
        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to verify under run {run_path}. Nothing to do.")
            return run_results

        proof_paths = VerifierUtils.get_proof_paths(run_results, run_path)
        work_items = self._prepare_work_items(slices_iter, dirs_path, run_path, proof_paths)
        if not work_items:
            return run_results

        results = self._execute_verification(work_items)

        verifs, jst_verified, ezkl_verified = self._process_results(results)
        return VerifierUtils.finalize_verify_results(run_path, verifs, jst_verified, ezkl_verified, len(work_items))

    def _execute_single_slice(self, run_path, model_path, backend, tiles_range):
        model_path = Path(model_path)
        run_path, dirs_root = Path(run_path), Utils.dirs_root_from(model_path)
        run_meta = initialize_stage_metadata(run_path, dirs_root)

        target_name = model_path.name
        if target_name.startswith("slice_") and target_name in run_meta.slices and len(run_meta.slices) > 1:
            run_meta.slices = {target_name: run_meta.slices[target_name]}

        if len(run_meta.slices) != 1:
            raise ValueError("Slices path must represent exactly one slice.")

        (slice_id, meta), = run_meta.slices.items()
        preferred = (backend or "").lower()
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)

        if meta.tiling:
            circuit_path = Utils.resolve_under_slice(slice_dir, resolve_circuit_path(meta, preferred))
            settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
            vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)
            success, tile_verifs = VerifierUtils.verify_tile_batch(slice_id, run_path, meta.tiling.num_tiles, preferred, circuit_path=circuit_path, settings_path=settings_path, vk_path=vk_path, tiles_range=tiles_range)
            method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY
            verifs = {
                slice_id: SliceResult(
                    slice_id=slice_id,
                    success=bool(success),
                    method=method,
                    tiles=[TileResult.from_dict(t) if isinstance(t, dict) else t for t in tile_verifs],
                    error=None if success else "One or more tiles failed verification",
                ).to_dict()
            }
        else:
            proof_path = run_path / "proof.json"
            if not proof_path.exists():
                raise FileNotFoundError(f"Proof file not found at {proof_path}")

            input_path, output_path = run_path / "input.json", run_path / "output.json"
            wf = get_witness_file(run_path, slice_id)
            witness_path = Path(wf) if wf else (run_path / WITNESS_FILENAME[preferred])

            circuit_path = Utils.resolve_under_slice(slice_dir, resolve_circuit_path(meta, preferred))
            settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
            vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)

            ok = VerifierUtils.verify_with_backend(preferred, str(proof_path), str(circuit_path), str(input_path), str(output_path), str(witness_path), settings_path, vk_path)
            method = ExecutionMethod.JSTPROVE_VERIFY if preferred == Backend.JSTPROVE else ExecutionMethod.EZKL_VERIFY
            verifs = {
                slice_id: SliceResult(
                    slice_id=slice_id,
                    success=bool(ok),
                    method=method,
                    error=None if ok else "verification_failed",
                ).to_dict()
            }

        jst_v = 1 if method == ExecutionMethod.JSTPROVE_VERIFY and verifs[slice_id]['success'] else 0
        ezkl_v = 1 if method == ExecutionMethod.EZKL_VERIFY and verifs[slice_id]['success'] else 0
        return VerifierUtils.finalize_verify_results(run_path, verifs, jst_v, ezkl_v, 1)

    @staticmethod
    def _prepare_work_items(slices_iter, dirs_path, run_path, proof_paths):
        work_items = []
        for slice_id, meta in slices_iter:
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)
            preferred = select_backend(run_path, slice_id, meta)
            tiling = meta.tiling

            circuit_path = Utils.resolve_under_slice(slice_dir, resolve_circuit_path(meta, preferred))
            settings_path = Utils.resolve_under_slice(slice_dir, meta.settings_path)
            vk_path = Utils.resolve_under_slice(slice_dir, meta.vk_path)

            if tiling:
                proof_path, input_path, output_path, witness_path = None, None, None, None
            else:
                proof_path = Path(proof_paths.get(slice_id, run_path / slice_id / "proof.json"))
                if not proof_path.exists():
                    logger.warning(f"Skipping {slice_id}: proof not found at {proof_path}")
                    continue

                input_path, output_path = run_path / slice_id / "input.json", run_path / slice_id / "output.json"
                wf = get_witness_file(run_path, slice_id)
                witness_path = Path(wf) if wf else (run_path / slice_id / WITNESS_FILENAME[preferred])

                if preferred == Backend.EZKL:
                    if not settings_path or not os.path.exists(settings_path):
                        logger.warning(f"Skipping {slice_id}: settings file not found ({settings_path})")
                        continue
                    if not vk_path or not os.path.exists(vk_path):
                        logger.warning(f"Skipping {slice_id}: verification key not found ({vk_path})")
                        continue

            work_items.append((
                slice_id, preferred, proof_path, circuit_path, settings_path, vk_path,
                input_path, output_path, witness_path, tiling.to_dict() if tiling else None,
                str(run_path), str(slice_dir)
            ))
        return work_items

    def _execute_verification(self, work_items):
        return self._run_work_items(
            work_items,
            VerifierUtils.verify_slice_logic,
            "Verifying",
            lambda sid, e: SliceResult(slice_id=sid, success=False, error=str(e)).to_dict(),
        )

    @staticmethod
    def _process_results(results):
        verifs = {}
        jst_verified, ezkl_verified = 0, 0
        for result in results:
            slice_id = result['slice_id']
            verifs[slice_id] = result
            if result['success']:
                method = result.get('method')
                if method == ExecutionMethod.JSTPROVE_VERIFY:
                    jst_verified += 1
                elif method == ExecutionMethod.EZKL_VERIFY:
                    ezkl_verified += 1
        return verifs, jst_verified, ezkl_verified
