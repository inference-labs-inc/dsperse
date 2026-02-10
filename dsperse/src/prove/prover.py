"""
Orchestration for various provers.
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dsperse.src.analyzers.schema import Backend, ExecutionMethod
from dsperse.src.backends.dispatch import WITNESS_FILENAME
from dsperse.src.pipeline_stage import PipelineStage
from dsperse.src.prove.utils.prover_utils import ProverUtils
from dsperse.src.utils.pipeline_utils import select_backend, get_witness_file, initialize_stage_metadata, filter_circuit_slices
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class Prover(PipelineStage):

    def prove(self, run_path, model_dir, output_path=None, backend=None, tiles_range=None):
        return self._dispatch(run_path, model_dir, backend=backend, output_path=output_path, tiles_range=tiles_range)

    def _execute_dirs(self, run_path, dirs_path, output_path, backend):
        run_path, dirs_path = Path(run_path), Utils.dirs_root_from(Path(dirs_path))
        metadata = initialize_stage_metadata(run_path, dirs_path)

        slices_iter = filter_circuit_slices(metadata, backend, run_path)
        if not slices_iter:
            logger.warning(f"No circuit-capable slices found to prove under run {run_path}.")
            return Utils.load_run_results(run_path)

        work_items = self._prepare_work_items(slices_iter, dirs_path, run_path, output_path)
        if not work_items:
            return Utils.load_run_results(run_path)

        results = self._execute_proving(work_items)

        proofs, proved_jst, proved_ezkl = self._process_results(results)
        return ProverUtils.finalize_prove_results(run_path, proofs, proved_jst, proved_ezkl, len(work_items))

    def _execute_single_slice(self, run_path, model_dir, detected, backend, tiles_range):
        model_dir_path = Path(model_dir)
        run_path, dirs_root = Path(run_path), Utils.dirs_root_from(model_dir_path)
        run_meta = initialize_stage_metadata(run_path, dirs_root)

        target_name = model_dir_path.name
        if target_name.startswith("slice_") and target_name in run_meta.slices and len(run_meta.slices) > 1:
            run_meta.slices = {target_name: run_meta.slices[target_name]}

        if len(run_meta.slices) != 1:
            raise ValueError(f"Slices path must represent exactly one slice; found {len(run_meta.slices)}")

        (slice_id, meta), = run_meta.slices.items()
        preferred = (backend or "").lower()
        slice_dir = Utils.slice_dirs_path(dirs_root, slice_id)
        circuit_path, pk_path, settings_path = ProverUtils.resolve_prove_artifacts(slice_dir, meta, preferred)

        if meta.tiling:
            witness_path, proof_path = None, None
        else:
            if preferred == Backend.JSTPROVE:
                wf = get_witness_file(run_path, slice_id)
                witness_path = Path(wf) if wf else (run_path / WITNESS_FILENAME[Backend.JSTPROVE])
            else:
                witness_path = run_path / WITNESS_FILENAME[Backend.EZKL]
            proof_path = run_path / "proof.json"

        result_dict = ProverUtils.execute_slice_proving(
            slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path,
            meta.tiling.to_dict() if meta.tiling else None, str(run_path), tiles_range=tiles_range
        )

        method = result_dict.get('method')
        proved_jst = 1 if method == ExecutionMethod.JSTPROVE_PROVE else 0
        proved_ezkl = 1 if method == ExecutionMethod.EZKL_PROVE else 0
        return ProverUtils.finalize_prove_results(run_path, {slice_id: result_dict}, proved_jst, proved_ezkl, 1)

    @staticmethod
    def _prepare_work_items(slices_iter, dirs_path, run_path, output_path):
        work_items = []
        for slice_id, meta in slices_iter:
            slice_dir = Utils.slice_dirs_path(dirs_path, slice_id)
            preferred = select_backend(run_path, slice_id, meta)
            circuit_path, pk_path, settings_path = ProverUtils.resolve_prove_artifacts(slice_dir, meta, preferred)

            tiling = meta.tiling
            if tiling:
                witness_path = None
                proof_path = None
            else:
                if preferred == Backend.JSTPROVE:
                    wf = get_witness_file(run_path, slice_id)
                    witness_path = Path(wf) if wf else (run_path / slice_id / WITNESS_FILENAME[Backend.JSTPROVE])
                else:
                    witness_path = Utils.witness_path_for(run_path, slice_id)

                if not witness_path.exists():
                    logger.warning(f"Skipping {slice_id}: witness not found at {witness_path}")
                    continue

                proof_path = Utils.proof_output_path(run_path, slice_id, output_path)

                if not circuit_path or not os.path.exists(circuit_path):
                    logger.warning(f"Skipping {slice_id}: compiled circuit not found ({circuit_path})")
                    continue

            work_items.append((
                slice_id, preferred, witness_path, circuit_path, proof_path, pk_path, settings_path,
                tiling.to_dict() if tiling else None, str(run_path), str(slice_dir)
            ))
        return work_items

    def _execute_proving(self, work_items):
        if self.parallel > 1 and len(work_items) > 1:
            logger.info(f"Proving {len(work_items)} slices with {self.parallel} parallel processes...")
            results = []
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(ProverUtils.prove_slice_logic, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    slice_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Slice {slice_id} proving failed: {e}")
                        results.append({'slice_id': slice_id, 'success': False, 'method': None, 'error': str(e)})
            return results
        else:
            results = []
            for item in work_items:
                slice_id = item[0]
                try:
                    results.append(ProverUtils.prove_slice_logic(item))
                except Exception as e:
                    logger.error(f"Slice {slice_id} proving failed: {e}")
                    results.append({'slice_id': slice_id, 'success': False, 'method': None, 'error': str(e)})
            return results

    @staticmethod
    def _process_results(results):
        proofs = {}
        proved_jst, proved_ezkl = 0, 0
        for result in results:
            slice_id = result['slice_id']
            proofs[slice_id] = result
            if result['success']:
                method = result.get('method')
                if method == ExecutionMethod.JSTPROVE_PROVE:
                    proved_jst += 1
                elif method == ExecutionMethod.EZKL_PROVE:
                    proved_ezkl += 1
            else:
                logger.error(f"Proof failed for {slice_id}: {result.get('error')}")
        return proofs, proved_jst, proved_ezkl
