"""
Runner for EzKL Circuit and ONNX Inference
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from dsperse.src.analyzers.runner_analyzer import RunnerAnalyzer
from dsperse.src.analyzers.schema import RunSliceMetadata, TilingInfo, ExecutionInfo, TileResult, Backend, \
    ExecutionMethod
from dsperse.src.backends.ezkl import EZKL
from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.backends.onnx_models import OnnxModels
from dsperse.src.run.channel_split_executor import ChannelSplitExecutor
from dsperse.src.run.tile_executor import TileExecutor
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

SliceCompleteCallback = Callable[[str, ExecutionInfo, Path], None]


class Runner:

    def __init__(
        self,
        run_dir: str = None,
        threads: int = 1,
        on_slice_complete: Optional[SliceCompleteCallback] = None,
        batch: bool = False,
    ):
        """Initialize the Runner.

        Args:
            run_dir: Directory for run outputs (default: alongside slices). If it contains metadata.json, it's a resume.
            threads: Number of parallel tile execution processes
            on_slice_complete: Optional callback called after each slice completes.
                               Signature: (slice_id: str, exec_info: ExecutionInfo, run_dir: Path) -> None
            batch: Use JSTprove batch witness generation for tiled slices (loads circuit once for all tiles)
        """
        self.run_dir = Path(run_dir) if run_dir else None
        self.threads = max(1, threads)
        self.resume = bool(self.run_dir)
        self.on_slice_complete = on_slice_complete
        self.batch = batch

        self.run_metadata = None
        self.slices_path: Path | None = None
        self.last_run_dir: Path | None = None

        try:
            self.ezkl_runner = EZKL()
        except RuntimeError:
            self.ezkl_runner = None
            logger.warning("EZKL CLI not available. EZKL backend will be disabled.")

        try:
            self.jstprove_runner = JSTprove()
        except RuntimeError:
            self.jstprove_runner = None
            logger.warning("JSTprove CLI not available. JSTprove backend will be disabled.")


    def dispatch_slice(self, slice_id, info, node, tensor_cache, run_dir, slices_path, in_file, out_file, backend, current_tensor):
        """Route execution to the appropriate backend based on slice type."""
        # --- Channel Split Execution ---
        if info.channel_split:
            logger.info(f"Running channel-split slice {slice_id} with {info.channel_split.num_groups} groups")
            exec_info = self.run_channel_split_inference(slice_id, info, tensor_cache, run_dir, slices_path, backend=backend)
            output_name = info.channel_split.output_name or (info.dependencies.output[0] if info.dependencies.output else None)
            if not output_name:
                raise ValueError(f"Channel-split slice {slice_id} has no output name in metadata")
            output_tensor = tensor_cache.get(output_name)
            if output_tensor is None:
                raise ValueError(f"Channel-split slice {slice_id}: output '{output_name}' not in tensor cache after execution")
            return exec_info.success, {"output": output_tensor}, exec_info

        # --- Tiled Execution ---
        if info.tiling:
            logger.info(f"Running tiled slice {slice_id} with parallel tiles")
            exec_info = self.run_tiled_inference(slice_id, info, tensor_cache, run_dir, backend=backend)
            output_name = info.tiling.output_name
            if not output_name:
                raise ValueError(f"Tiled slice {slice_id} has no output_name in tiling metadata")
            output_tensor = tensor_cache.get(output_name)
            if output_tensor is None:
                raise ValueError(f"Tiled slice {slice_id}: output '{output_name}' not in tensor cache after execution")
            return exec_info.success, {"output": output_tensor}, exec_info

        # --- Standard Circuit Execution ---
        use_circuit = node.use_circuit and backend != Backend.ONNX
        if use_circuit:
            return self.run_circuit_with_fallback(info, in_file, out_file, slices_path, backend=backend)
        
        # --- ONNX Fallback / Multi-input Execution ---
        filtered_inputs = [n for n in info.dependencies.filtered_inputs if n]
        if len(filtered_inputs) > 1:
            missing = [n for n in filtered_inputs if n not in tensor_cache]
            if missing:
                raise ValueError(f"Missing input tensors for {slice_id}: {missing}")
            extra_tensors = {name: tensor_cache[name] for name in filtered_inputs}
        else:
            extra_tensors = {filtered_inputs[0] if filtered_inputs else "input": current_tensor}
        
        return self.run_onnx_multi(info, out_file, slices_path, extra_tensors)

    def run_circuit_with_fallback(self, meta: RunSliceMetadata, in_file: Path, out_file: Path, slice_dir: Path, backend: str = None):
        """Execute a slice using best available backend: jstprove -> ezkl -> onnx."""
        # --- Setup and Capability Check ---
        slice_id = slice_dir.name if slice_dir else "unknown"
        forced = backend

        has_jst = bool(meta.jstprove_circuit_path) and self.jstprove_runner is not None
        has_ezkl = bool(meta.ezkl_circuit_path) and (bool(meta.vk_path) or bool(meta.ezkl_vk_path))

        # --- Forced Backend Handling ---
        if forced == Backend.ONNX:
            logger.info(f"[{slice_id}] Running with ONNX (forced)")
            return self.run_onnx_single(meta, in_file, out_file, slice_dir)

        if forced == Backend.JSTPROVE and has_jst:
            logger.info(f"[{slice_id}] Running with JSTprove (forced)")
            j_meta = RunnerUtils.prepare_jstprove_meta(meta)
            return self.run_jstprove_inference(j_meta, in_file, out_file, slice_dir)

        if forced == Backend.EZKL and has_ezkl:
            logger.info(f"[{slice_id}] Running with EZKL (forced)")
            e_meta = RunnerUtils.prepare_ezkl_meta(meta)
            ezkl_in = RunnerUtils.flatten_input_for_ezkl(in_file)
            return self.run_ezkl_inference(e_meta, ezkl_in, out_file, slice_dir)

        # --- Best Available (Automatic) Backend Selection ---
        if has_jst and forced != Backend.EZKL:
            logger.info(f"[{slice_id}] Running with JSTprove (best available)")
            j_meta = RunnerUtils.prepare_jstprove_meta(meta)
            ok, tensor, j_info = self.run_jstprove_inference(j_meta, in_file, out_file, slice_dir)
            if ok:
                return ok, tensor, j_info
            logger.warning(f"[{slice_id}] JSTprove failed, trying fallback...")

            if has_ezkl:
                logger.info(f"[{slice_id}] Falling back to EZKL")
                e_meta = RunnerUtils.prepare_ezkl_meta(meta)
                ezkl_in = RunnerUtils.flatten_input_for_ezkl(in_file)
                ok, tensor, e_info = self.run_ezkl_inference(e_meta, ezkl_in, out_file, slice_dir)
                if ok:
                    return ok, tensor, e_info
                logger.warning(f"[{slice_id}] EZKL failed, falling back to ONNX")

            logger.info(f"[{slice_id}] Falling back to ONNX")
            ok, tensor, o_info = self.run_onnx_single(meta, in_file, out_file, slice_dir)
            o_info.method = ExecutionMethod.JSTPROVE_FALLBACK_ONNX
            return ok, tensor, o_info

        if has_ezkl and forced != Backend.JSTPROVE:
            logger.info(f"[{slice_id}] Running with EZKL (best available)")
            e_meta = RunnerUtils.prepare_ezkl_meta(meta)
            ezkl_in = RunnerUtils.flatten_input_for_ezkl(in_file)
            ok, tensor, e_info = self.run_ezkl_inference(e_meta, ezkl_in, out_file, slice_dir)
            if ok:
                return ok, tensor, e_info
            logger.warning(f"[{slice_id}] EZKL failed, falling back to ONNX")
            ok, tensor, o_info = self.run_onnx_single(meta, in_file, out_file, slice_dir)
            o_info.method = ExecutionMethod.EZKL_FALLBACK_ONNX
            return ok, tensor, o_info

        # --- Final ONNX Fallback ---
        logger.info(f"[{slice_id}] Running with ONNX (no ZK circuits available)")
        return self.run_onnx_single(meta, in_file, out_file, slice_dir)

    def run_ezkl_inference(self, meta: RunSliceMetadata, input_tensor_path: Path, output_witness_path: Path, slice_dir: Path = None):
        """Run EZKL inference for a slice."""
        # --- Pre-execution Checks ---
        if self.ezkl_runner is None:
            return False, "EZKL CLI not available", ExecutionInfo(method=ExecutionMethod.EZKL_GEN_WITNESS, success=False, error='EZKL CLI not available')

        # --- Path Resolution ---
        paths = RunnerUtils.resolve_inference_paths(meta, slice_dir)

        # --- Witness Generation ---
        try:
            success, output_tensor = self.ezkl_runner.generate_witness(
                input_file=str(input_tensor_path),
                model_path=paths['circuit'],
                output_file=str(output_witness_path),
                vk_path=paths['vk'],
                settings_path=paths['settings']
            )
        except Exception as e:
            success, output_tensor = False, str(e)

        # --- Result Packaging ---
        exec_info = ExecutionInfo(
            method=ExecutionMethod.EZKL_GEN_WITNESS,
            success=success,
            error=None if success else (output_tensor if isinstance(output_tensor, str) else "Unknown EZKL error"),
            witness_file=str(output_witness_path),
        )
        return success, output_tensor, exec_info

    def run_jstprove_inference(self, meta: RunSliceMetadata, input_tensor_path: Path, output_witness_path: Path, slice_dir: Path = None):
        """Run JSTprove inference for a slice."""
        # --- Pre-execution Checks ---
        if self.jstprove_runner is None:
            return False, "JSTprove CLI not available", ExecutionInfo(method=ExecutionMethod.JSTPROVE_GEN_WITNESS, success=False, error='JSTprove CLI not available')

        # --- Path Resolution ---
        paths = RunnerUtils.resolve_inference_paths(meta, slice_dir)

        # --- Witness Generation ---
        try:
            witness_file_path = Path(output_witness_path).with_name("output_witness.bin")
            success, output_tensor = self.jstprove_runner.generate_witness(
                input_file=str(input_tensor_path),
                model_path=paths['circuit'],
                output_file=str(output_witness_path),
            )
        except Exception as e:
            success, output_tensor = False, str(e)
            witness_file_path = Path(output_witness_path).with_name("output_witness.bin")

        # --- Result Packaging ---
        exec_info = ExecutionInfo(
            method=ExecutionMethod.JSTPROVE_GEN_WITNESS,
            success=success,
            error=None if success else (output_tensor if isinstance(output_tensor, str) else "Unknown JSTprove error"),
            witness_file=str(witness_file_path),
        )
        return success, output_tensor, exec_info

    def run_tiled_inference(self, slice_id: str, meta: RunSliceMetadata, tensor_cache: dict, run_dir: Path, backend: str = None) -> ExecutionInfo:
        """Run a tiled slice by splitting into overlapping tiles and executing them."""
        tiling = meta.tiling
        if not tiling:
            return ExecutionInfo(method=ExecutionMethod.TILED, success=False, error='missing_tiling_info')

        tile_executor = TileExecutor(self.slices_path, tensor_cache)
        input_tensor = tile_executor.get_input_tensor(slice_id, tiling, meta)
        tile_executor.split_into_tiles(slice_id, tiling, input_tensor)

        tile_results = self.run_tiles(slice_id, tiling, meta, run_dir, tensor_cache, backend=backend)

        tile_executor.reconstruct_from_tiles(slice_id, tiling)

        return ExecutionInfo(method=ExecutionMethod.TILED, success=True, tiles=tile_results)

    def run_tiles(self, slice_id: str, tiling: TilingInfo, meta: RunSliceMetadata, run_dir: Path, tensor_cache: dict, backend: str = None) -> list[TileResult]:
        """Execute individual tiles for a tiled slice in parallel or sequentially."""
        tile_executor = TileExecutor(self.slices_path, tensor_cache)
        config = tile_executor.get_execution_config(tiling, meta, backend=backend, has_jst_runner=(self.jstprove_runner is not None))

        if config['tile_onnx_path'] is None or not Path(config['tile_onnx_path']).exists():
            raise ValueError(f"Tile ONNX path not found for {slice_id}: {config['tile_onnx_path']}")

        tile_exec_infos = []
        parallel_count = min(self.threads, tiling.num_tiles)

        if self.batch and config['has_jst']:
            tile_args_list = tile_executor.prepare_tasks_in_memory(tiling.num_tiles, tiling.slice_idx, tiling, meta, run_dir, config)
            return self._run_tiles_batch_jst(slice_id, tiling, meta, tile_args_list, tile_executor, run_dir)

        if self.batch and config['tile_onnx_path']:
            tile_args_list = tile_executor.prepare_tasks_in_memory(tiling.num_tiles, tiling.slice_idx, tiling, meta, run_dir, config)
            return self._run_tiles_batch_onnx(slice_id, tiling, tile_args_list, tile_executor, config)

        tile_args_list = tile_executor.prepare_tasks(tiling.num_tiles, tiling.slice_idx, tiling, meta, run_dir, config)

        if parallel_count > 1:
            logger.info(f"Running {tiling.num_tiles} tiles with {config['backend_name']} using {parallel_count} parallel processes")
            with ProcessPoolExecutor(max_workers=parallel_count) as executor:
                futures = {executor.submit(TileExecutor.execute_worker, args): args['tile_idx'] for args in tile_args_list}
                results_map = {}
                for future in as_completed(futures):
                    tile_idx = futures[future]
                    try:
                        results_map[tile_idx] = future.result()
                    except Exception as e:
                        results_map[tile_idx] = {'success': False, 'error': str(e), 'tile_idx': tile_idx}

            tile_exec_infos = tile_executor.collect_outputs(results_map, tiling.num_tiles, tiling.slice_idx, tiling)
        else:
            if config['has_jst'] or config['has_ezkl']:
                logger.info(f"Running {tiling.num_tiles} tiles with {config['backend_name']} circuits (sequential)")
                for args in tile_args_list:
                    tile_meta = RunSliceMetadata(
                        path=args['tile_onnx_path'],
                        jstprove_circuit_path=args['jstprove_circuit_path'],
                        ezkl_circuit_path=args['ezkl_circuit_path'],
                        settings_path=args['settings_path'],
                        vk_path=args['vk_path'],
                        jstprove_settings_path=args.get('jstprove_settings_path'),
                        ezkl_settings_path=args.get('ezkl_settings_path'),
                        ezkl_vk_path=args.get('ezkl_vk_path'),
                        ezkl_pk_path=args.get('ezkl_pk_path'),
                        dependencies=meta.dependencies,
                    )
                    ok, result, exec_info = self.run_circuit_with_fallback(
                        tile_meta, Path(args['tile_in']), Path(args['tile_out']),
                        Path(args['slice_specific_dir']), backend=config['effective_backend']
                    )
                    if not ok:
                        raise RuntimeError(f"Tile {args['tile_idx']} failed: {exec_info.error}")
                    tile_executor.process_result(result, tiling, tiling.slice_idx, args['tile_idx'])
                    tile_exec_infos.append(TileResult(tile_idx=args['tile_idx'], success=True, method=exec_info.method))
            else:
                logger.info(f"Running {tiling.num_tiles} tiles with ONNX only")
                for args in tile_args_list:
                    tile_meta = RunSliceMetadata(path=args['tile_onnx_path'], dependencies=meta.dependencies)
                    ok, result, o_info = self.run_onnx_single(tile_meta, Path(args['tile_in']), Path(args['tile_out']), self.slices_path)
                    if not ok:
                        raise RuntimeError(f"Tile {args['tile_idx']} failed: {o_info.error}")
                    tile_executor.process_result(result, tiling, tiling.slice_idx, args['tile_idx'])
                    tile_exec_infos.append(TileResult(tile_idx=args['tile_idx'], success=True, method=ExecutionMethod.ONNX_ONLY))

        return tile_exec_infos

    def _run_tiles_batch_jst(
        self, slice_id: str, tiling: TilingInfo, meta: RunSliceMetadata,
        tile_args_list: list[dict], tile_executor: TileExecutor, run_dir: Path,
    ) -> list[TileResult]:
        circuit_path = RunnerUtils.resolve_relative_path(meta.jstprove_circuit_path, self.slices_path)

        jobs = []
        has_tensors = "tile_tensor" in tile_args_list[0]
        for args in tile_args_list:
            tile_out = Path(args['tile_out'])
            witness_path = tile_out.parent / f"{tile_out.stem}_witness.bin"
            job = {
                "output": args['tile_out'],
                "witness": str(witness_path),
            }
            if has_tensors:
                job["_tensor_inputs"] = {"input_data": args['tile_tensor'].tolist()}
            else:
                job["input"] = args['tile_in']
            jobs.append(job)

        manifest_dir = run_dir / f"slice_{tiling.slice_idx}"
        cpus = os.cpu_count() or 1
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        cgroup_limit = Path("/sys/fs/cgroup/memory.max")
        if not cgroup_limit.exists():
            cgroup_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if cgroup_limit.exists():
            raw = cgroup_limit.read_text().strip()
            if raw != "max" and raw.isdigit():
                total_bytes = min(total_bytes, int(raw))
        mem_workers = max(1, int(total_bytes / (1024 ** 3) * 0.5))
        witness_workers = max(1, min(cpus, mem_workers, len(jobs) // 16))
        logger.info(f"Running {tiling.num_tiles} tiles with JSTprove batch witness for {slice_id} ({witness_workers} workers)")

        if has_tensors:
            results = self.jstprove_runner.generate_witness_batch_from_tensors(
                circuit_path=circuit_path,
                jobs=jobs,
                manifest_dir=str(manifest_dir),
                workers=witness_workers,
            )
        else:
            results = self.jstprove_runner.generate_witness_batch(
                circuit_path=circuit_path,
                jobs=jobs,
                manifest_dir=str(manifest_dir),
            )

        tile_exec_infos = []
        for args, (success, output) in zip(tile_args_list, results):
            if not success:
                raise RuntimeError(f"Batch witness failed for tile {args['tile_idx']}: {output}")
            tile_executor.process_result(output, tiling, tiling.slice_idx, args['tile_idx'])
            tile_exec_infos.append(TileResult(
                tile_idx=args['tile_idx'],
                success=True,
                method=ExecutionMethod.JSTPROVE_GEN_WITNESS,
            ))

        return tile_exec_infos

    def _run_tiles_batch_onnx(
        self, slice_id: str, tiling: TilingInfo,
        tile_args_list: list[dict], tile_executor: TileExecutor, config: dict,
    ) -> list[TileResult]:
        import numpy as np
        import torch

        model_path = config['tile_onnx_path']
        logger.info(f"Running {tiling.num_tiles} tiles with ONNX batch for {slice_id}")

        session = OnnxModels._create_session(model_path)
        model_inputs = session.get_inputs()
        input_name = model_inputs[0].name
        input_shape = [int(d) if isinstance(d, int) else 1 for d in model_inputs[0].shape]
        expected_dtype = OnnxModels._parse_onnx_type(model_inputs[0].type)

        tile_exec_infos = []
        for args in tile_args_list:
            tile_tensor = args.get('tile_tensor')
            if tile_tensor is not None:
                if isinstance(tile_tensor, torch.Tensor):
                    arr = tile_tensor.numpy()
                else:
                    arr = np.asarray(tile_tensor)
            else:
                input_tensor = RunnerUtils.preprocess_input(args['tile_in'])
                arr = input_tensor.numpy() if isinstance(input_tensor, torch.Tensor) else np.asarray(input_tensor)
            arr = arr.reshape(input_shape).astype(expected_dtype)

            raw_output = session.run(None, {input_name: arr})
            output_tensor = torch.from_numpy(raw_output[0]).float()
            result = RunnerUtils.process_final_output(output_tensor)

            tile_executor.process_result(result, tiling, tiling.slice_idx, args['tile_idx'])
            tile_exec_infos.append(TileResult(
                tile_idx=args['tile_idx'],
                success=True,
                method=ExecutionMethod.ONNX_ONLY,
            ))

        return tile_exec_infos

    def run_channel_split_inference(self, slice_id: str, meta: RunSliceMetadata, tensor_cache: dict, run_dir: Path, slices_path: Path, backend: str = None) -> ExecutionInfo:
        """Coordinate the execution of multiple channel groups and sum their outputs."""
        cs_executor = ChannelSplitExecutor(
            slices_path, tensor_cache,
            jstprove_runner=self.jstprove_runner,
            ezkl_runner=self.ezkl_runner
        )

        input_tensor, output_shape, input_name = cs_executor.prepare_config(meta)
        if input_tensor is None:
            err = "missing_channel_split_info" if not meta.channel_split else f"Missing input tensor for channel-split slice {slice_id}"
            return ExecutionInfo(method="channel_split", success=False, error=err)

        partial_outputs, methods_used = [], []
        for group in meta.channel_split.groups:
            group_input = input_tensor[:, group.c_start:group.c_end, :, :]
            output, method = cs_executor.execute_group(group, group_input, run_dir / slice_id, output_shape, backend=backend)
            partial_outputs.append(output)
            methods_used.append(method)

        summed = partial_outputs[0]
        for po in partial_outputs[1:]:
            summed = summed + po

        summed = cs_executor.apply_bias(summed, meta.channel_split)

        output_name = meta.channel_split.output_name or (meta.dependencies.output[0] if meta.dependencies.output else "output")
        tensor_cache[output_name] = summed
        primary_method = methods_used[0] if len(set(methods_used)) == 1 else "channel_split_mixed"
        logger.info(f"Channel split {slice_id}: {meta.channel_split.num_groups} groups via {primary_method}, output {list(summed.shape)}")
        return ExecutionInfo(method=f"channel_split:{primary_method}", success=True)

    @staticmethod
    def run_onnx_multi(meta: RunSliceMetadata, output_file: Path, slice_dir: Path, extra_tensors: dict):
        """Run ONNX inference for a multi-input slice."""
        # --- ONNX Path Resolution ---
        onnx_path = RunnerUtils.resolve_relative_path(meta.path, slice_dir)
        if not onnx_path or not Path(onnx_path).exists():
            return False, f"ONNX file not found: {onnx_path}", ExecutionInfo(method=ExecutionMethod.ONNX_MULTI_INPUT, success=False, error='file_not_found')

        # --- Inference Execution ---
        try:
            success, result = OnnxModels.run_inference_multi(model_path=onnx_path, extra_tensors=extra_tensors, output_file=str(output_file))
        except Exception as e:
            success, result = False, str(e)

        # --- Result Packaging ---
        exec_info = ExecutionInfo(
            method=ExecutionMethod.ONNX_MULTI_INPUT, success=success,
            error=None if success else (result if isinstance(result, str) else 'unknown'),
        )
        return success, result, exec_info

    @staticmethod
    def run_onnx_single(meta: RunSliceMetadata, input_tensor_path: Path, output_tensor_path: Path, slice_dir: Path = None):
        """Run ONNX inference for a single-input slice."""
        # --- ONNX Path Resolution ---
        onnx_path = RunnerUtils.resolve_relative_path(meta.path, slice_dir)
        if not onnx_path or not Path(onnx_path).exists():
            return False, f"ONNX file not found: {onnx_path}", ExecutionInfo(method=ExecutionMethod.ONNX_ONLY, success=False, error='file_not_found')

        # --- Inference Execution ---
        success, result = OnnxModels.run_inference(model_path=onnx_path, input_file=str(input_tensor_path), output_file=str(output_tensor_path))
        
        # --- Result Packaging ---
        exec_info = ExecutionInfo(
            method=ExecutionMethod.ONNX_ONLY, success=success,
            error=None if success else (result if isinstance(result, str) else 'inference_failed'),
        )
        return success, result, exec_info

    def _run(self, input_json_path=None, backend: str = None):
        """Internal execution loop for model inference."""
        # --- Initialization ---
        nodes = self.run_metadata.execution_chain.nodes
        current_slice_id = self.run_metadata.execution_chain.head
        run_dir = self.last_run_dir

        input_tensor, tensor_cache = RunnerUtils.initialize_run_state(input_json_path, self.run_metadata,
                                                                      current_slice_id)
        slice_results, final_tensor = {}, None

        # --- Execution Loop ---
        while current_slice_id:
            info = self.run_metadata.get_slice(current_slice_id)
            node = nodes[current_slice_id]
            in_file, out_file = RunnerUtils.prepare_slice_io(run_dir, current_slice_id)

            # Resume Logic
            if self.resume:
                resumed, cached_tensor, cached_info = RunnerUtils.try_resume_slice(run_dir, current_slice_id)
                if resumed:
                    for oname in info.dependencies.output: tensor_cache[oname] = cached_tensor
                    final_tensor, slice_results[current_slice_id] = cached_tensor, cached_info
                    current_slice_id = node.next
                    continue

            # Execution Step
            skip_write = bool(info.tiling or info.channel_split)
            current_tensor = RunnerUtils.prepare_slice_input(info, tensor_cache, input_tensor, in_file, skip_write=skip_write)
            ok, result, exec_info = self.dispatch_slice(
                current_slice_id, info, node, tensor_cache, run_dir, self.slices_path, in_file, out_file, backend,
                current_tensor
            )

            # Result Processing
            final_tensor = RunnerUtils.process_inference_result(current_slice_id, info, ok, result, exec_info,
                                                                tensor_cache)
            slice_results[current_slice_id] = exec_info

            if not ok:
                err = exec_info.error if isinstance(exec_info, ExecutionInfo) else exec_info.get('error', 'unknown')
                raise Exception(f"Inference failed for {current_slice_id}: {err}")

            if final_tensor is not None:
                RunnerUtils.save_intermediate_output(out_file, final_tensor, exec_info)

            if self.on_slice_complete:
                self.on_slice_complete(current_slice_id, exec_info, run_dir)

            current_slice_id = nodes[current_slice_id].next

        # --- Finalization ---
        return RunnerUtils.finalize_run_results(self.run_metadata, input_tensor, final_tensor, slice_results, run_dir)

    def run(self, input_json_path, slice_path: str, output_path: str = None, backend: str | None = None) -> dict:
        """Run inference through the chain using run/metadata.json.

        slice_path can be provided here (preferred) or at construction time for backward compatibility.

        Args:
            input_json_path: Path to the input JSON tensor file
            slice_path: Path to the slices directory or packaged slices (.dsperse/.dslice)
            output_path: Optional path where run data/results should be saved
            backend: Optional backend selector ('jstprove' | 'ezkl' | 'onnx').
                     - When provided, applies only at run-time and only affects slices that
                       have multiple circuit backends compiled. If 'onnx', skips circuit backends.
        """
        if slice_path is None or not Path(slice_path).exists():
            raise Exception("A valid path must be provided for slices")
        self.slices_path = Path(slice_path)

        format = Converter.detect_type(self.slices_path)
        if format != "dirs":
            slices_path = Converter.convert(str(self.slices_path), output_type="dirs")
            self.slices_path = Path(slices_path)

        # Generate run metadata if needed
        self.last_run_dir, self.resume, self.run_metadata = RunnerAnalyzer.initialize_run_metadata(
            self.slices_path, run_dir=self.run_dir, output_path=output_path, format=format
        )

        # run inference
        results = self._run(input_json_path=input_json_path, backend=backend)

        if format != "dirs":
            self.slices_path = Converter.convert(str(self.slices_path), output_type=format, cleanup=True)

        return results

if __name__ == "__main__":
    # Choose which model to test
    model_choice = 2  # Change this to test different models

    # Model configurations
    base_paths = {
        1: "../../models/doom",
        2: "../../models/net",
        3: "../../models/resnet",
        4: "../../models/yolov3",
        5: "../../models/age",
    }

    # Get model directory
    abs_path = os.path.abspath(base_paths[model_choice])
    slices_dir = os.path.join(abs_path, "slices")
    # slices_dir = os.path.join(slices_dir, "slice_0")
    input_json = os.path.join(abs_path, "input.json")

    # Initialize runner (auto-generates run metadata if needed). Slices dir is now passed to run(...).
    runner = Runner()

    # Run inference
    print(f"Running inference on model {base_paths[model_choice]}...")
    results = runner.run(input_json, slice_path=slices_dir)#, backend="ezkl")

    # Display results
    print(f"\nOutput shape: {results['tensor_shape']}")
    print("Execution summary:")
    for slice_id, info in results["slice_results"].items():
        method = info.method if isinstance(info, ExecutionInfo) else info.get('method', 'unknown')
        print(f"  {slice_id}: {method}")
