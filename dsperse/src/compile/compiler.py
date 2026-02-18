import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import torch

from dsperse.src.analyzers.schema import Backend
from dsperse.src.backends.onnx_models import OnnxModels
from dsperse.src.compile.utils.compiler_utils import CompilerUtils, CompilationTask, BackendParseResult
from dsperse.src.run.utils.runner_utils import RunnerUtils
from dsperse.src.slice.utils.converter import Converter
from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)


class Compiler:
    """
    Orchestrator class for compiling models of different types.
    
    This class provides a unified interface for compiling models by delegating
    to the appropriate compiler implementation based on the model type.
    """
    def __init__(self, parallel: int = 1, resume: bool = False, weights_as_inputs: bool = False):
        """Initialize the Compiler with a specific configuration."""
        self.parallel = max(1, parallel)
        self.resume = resume
        self.weights_as_inputs = weights_as_inputs
        self.layer_backends = {}
        self.default_layer_indices = set()
        self.default_backend = None
        self.use_fallback = True

    @staticmethod
    def _compile_slice_worker(task: CompilationTask) -> dict:
        """Worker function for parallel slice compilation."""
        results = {
            'idx': task.idx,
            'successful_backends': [],
            'compilation_blocks': {},
            'errors': [],
            'channel_group_circuits': {}
        }

        if task.channel_split_info:
            Compiler.compile_channel_split_groups(task.idx, task.slice_dir, task.base_path, task.backends_to_build, task.channel_split_info, results, weights_as_inputs=task.weights_as_inputs)
        else:
            Compiler.compile_slice(task.idx, task.base_path, task.slice_dir, task.backends_to_build, task.tiling_info, task.compilation_slice_data, results, weights_as_inputs=task.weights_as_inputs)

        return results
    
    def _needs_calibration_chain(self) -> bool:
        """Check if EZKL calibration chain is needed. Only EZKL requires calibration data."""
        if self.default_backend == Backend.JSTPROVE and not self.use_fallback:
            return False
        if self.default_backend is None or self.default_backend == Backend.EZKL:
            return True
        if any(b == Backend.EZKL for b in self.layer_backends.values()):
            return True
        return self.use_fallback

    def _compile_slices(self, dir_path: str, input_file_path: Optional[str] = None, layer_indices=None):
        """Coordinates the compilation of multiple slices."""
        # --- Metadata and Calibration ---
        metadata, metadata_path, base_path, slices_data = CompilerUtils.initialize_compilation_metadata(dir_path)
        if input_file_path and self._needs_calibration_chain():
            self.run_calibration_inference(slices_data, base_path, input_file_path)

        # --- Task Preparation ---
        work_items, slice_info_map, stats = CompilerUtils.prepare_compilation_tasks(
            slices_data, base_path, layer_indices, self.resume,
            self.layer_backends, self.default_layer_indices, self.default_backend, self.use_fallback
        )
        for task in work_items:
            task.weights_as_inputs = self.weights_as_inputs

        # --- Compilation Execution ---
        if self.parallel > 1 and len(work_items) > 1:
            results = self._execute_parallel_compilation(work_items)
        else:
            results = CompilerUtils.run_sequential_compilation(work_items, Compiler._compile_slice_worker)

        # --- Result Processing ---
        for result in results:
            idx = result['idx']
            info = slice_info_map[idx]
            CompilerUtils.process_compilation_result(result, info, stats)

        # --- Finalization ---
        Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))
        CompilerUtils.log_compilation_summary(stats['backend_stats'], stats['compiled_count'], stats['skipped_count'])

    def _execute_parallel_compilation(self, work_items: list) -> list[dict]:
        """Executes compilation tasks in parallel using ProcessPoolExecutor."""
        logger.info(f"Compiling {len(work_items)} slices with {self.parallel} parallel processes...")
        results = []
        with ProcessPoolExecutor(max_workers=self.parallel) as executor:
            futures = {executor.submit(Compiler._compile_slice_worker, task): task.idx for task in work_items}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Slice {idx} compilation failed: {e}")
                    results.append({'idx': idx, 'successful_backends': [], 'compilation_blocks': {}, 'errors': [str(e)]})
        return results

    @staticmethod
    def run_calibration_inference(slices_data: list, base_path: str, input_file_path: Optional[str] = None):
        """Phase 1: Run ONNX inference chain to generate calibration files."""
        # --- Validation and Setup ---
        if not input_file_path or not os.path.exists(input_file_path):
            logger.warning("No input file provided, skipping ONNX inference chain")
            return

        logger.info("Running ONNX inference chain to generate calibration files")
        initial_tensor = RunnerUtils.preprocess_input(input_file_path)
        
        if not slices_data:
            logger.warning("No slices data, skipping ONNX inference chain")
            return

        # --- Initial State ---
        model_input_name = CompilerUtils.get_model_input_name(slices_data[0])
        tensor_cache = {model_input_name: initial_tensor}

        # --- Inference Loop ---
        for idx, slice_data in enumerate(slices_data):
            # Resolve Slice Path
            slice_path = CompilerUtils.get_slice_onnx_path(slice_data, base_path, idx)
            if not slice_path: continue

            # Setup Calibration Artifacts
            calibration_path = CompilerUtils.prepare_calibration_artifact(slice_path)
            
            # Prepare Input and Run Inference
            ok, result = Compiler._execute_calibration_inference(idx, slice_path, slice_data, tensor_cache, initial_tensor, model_input_name, calibration_path)
            if not ok: break

            # Process and Cache Results
            CompilerUtils.cache_calibration_outputs(idx, slice_data, result, tensor_cache)
            logger.info(f"Slice {idx}: Saved calibration to {calibration_path}")

    @staticmethod
    def _execute_calibration_inference(idx: int, slice_path: str, slice_data: dict, tensor_cache: dict, initial_tensor: torch.Tensor, model_input_name: str, calibration_path: str):
        """Prepares input and runs inference for a single calibration step."""
        deps = slice_data.get('dependencies', {})
        filtered_inputs = [n for n in deps.get('filtered_inputs', deps.get('input', [])) if n]
        
        try:
            if len(filtered_inputs) <= 1:
                input_name = filtered_inputs[0] if filtered_inputs else model_input_name
                input_tensor = tensor_cache.get(input_name, initial_tensor)
                
                RunnerUtils.save_to_file_flattened(input_tensor, calibration_path)
                success, result = OnnxModels.run_inference_tensor(input_tensor, slice_path)
            else:
                missing = [n for n in filtered_inputs if n not in tensor_cache]
                if missing:
                    logger.warning(f"Slice {idx}: Missing inputs {missing}, skipping multi-input inference")
                    return False, None
                
                extra_tensors = {name: tensor_cache[name] for name in filtered_inputs}
                RunnerUtils.save_to_file_flattened(tensor_cache[filtered_inputs[0]], calibration_path)
                success, result = OnnxModels.run_inference_multi(slice_path, extra_tensors)
            
            if not success:
                logger.error(f"ONNX inference failed for slice {idx}: {result}")
                return False, None
            return True, result
        except Exception as e:
            logger.error(f"Calibration error at slice {idx}: {e}")
            return False, None

    @staticmethod
    def compile_slice(idx: int, base_path: str, slice_dir: str, backends: list[str], tiling_info: Optional[dict], compilation_slice_data: dict, results: dict, weights_as_inputs: bool = False):
        """Compiles circuits for a standard or tiled slice."""
        for be in backends:
            try:
                # --- Setup ---
                output_dir = CompilerUtils.setup_compilation_dir(slice_dir, be, tiling_info)
                calibration_input = CompilerUtils.get_calibration_input_path(output_dir)
                slice_path = CompilerUtils.resolve_slice_source_path(compilation_slice_data, base_path)

                slice_name = f"slice_{idx}"
                logger.info(f"[{be}] {slice_name}: compiling...")

                # --- Compilation ---
                comp_data, version = Compiler.run_backend_pipeline(be, idx, slice_path, output_dir, calibration_input, weights_as_inputs=weights_as_inputs)
                if not comp_data: continue

                # --- Result Packaging ---
                success = CompilerUtils.is_compilation_successful(comp_data)
                file_paths = CompilerUtils.get_relative_paths(comp_data, calibration_input, slice_dir)

                status = "OK" if success else "FAILED"
                logger.info(f"[{be}] {slice_name}: {status}")

                results['compilation_blocks'][be] = CompilerUtils.build_compilation_block(
                    be, version, success, file_paths, slice_dir, tiling_info, weights_as_inputs=(weights_as_inputs and be == Backend.JSTPROVE)
                )
                if success:
                    results['successful_backends'].append(be)

            except Exception as e:
                results['errors'].append(f"{be}: {str(e)}")

    @staticmethod
    def compile_channel_split_groups(idx: int, slice_dir: str, base_path: str, backends: list[str], channel_split_info: dict, results: dict, weights_as_inputs: bool = False):
        """Compiles circuits for all groups in a channel-split slice."""
        groups = channel_split_info.get('groups', [])
        for be in backends:
            group_results, all_success = [], True

            for group in groups:
                # --- Group Preparation ---
                group_idx = group.get('group_idx', 0)
                group_onnx = CompilerUtils.resolve_group_onnx_path(group, base_path)
                if not group_onnx:
                    all_success = False; continue

                output_dir = CompilerUtils.setup_group_compilation_dir(slice_dir, be, group_idx)

                # --- Group Compilation ---
                logger.info(f"[{be}] slice_{idx} group_{group_idx}: compiling...")
                comp_data, _ = Compiler.run_backend_pipeline(be, idx, group_onnx, output_dir, None, weights_as_inputs=weights_as_inputs)
                
                # --- Result Collection ---
                success = CompilerUtils.is_compilation_successful(comp_data) if comp_data else False
                if success:
                    rel_paths = CompilerUtils.get_relative_paths(comp_data, None, slice_dir)
                    group_results.append({'group_idx': group_idx, 'success': True, 'files': rel_paths})
                else:
                    all_success = False
                    group_results.append({'group_idx': group_idx, 'success': False})

            # --- Backend Aggregation ---
            results['channel_group_circuits'][be] = group_results
            if all_success and group_results:
                results['successful_backends'].append(be)
                results['compilation_blocks'][be] = CompilerUtils.build_channel_split_block(be, len(groups), group_results, slice_dir, weights_as_inputs=(weights_as_inputs and be == Backend.JSTPROVE))

    @staticmethod
    def run_backend_pipeline(backend: str, idx: int, slice_path: str, output_dir: str, calibration_input: str | None, weights_as_inputs: bool = False):
        """Executes the specific backend compilation pipeline with appropriate imports."""
        if backend == Backend.JSTPROVE:
            from dsperse.src.backends.jstprove import JSTprove
            compatible, unsupported_ops = JSTprove.is_compatible(slice_path)
            if not compatible:
                logger.info(f"[jstprove] slice_{idx}: SKIP - unsupported ops {unsupported_ops}")
                return None, None
            be_instance = JSTprove()
            data = be_instance.compilation_pipeline(slice_path, output_dir, input_file_path=calibration_input, weights_as_inputs=weights_as_inputs)
            return data, be_instance.get_version()

        elif backend == Backend.EZKL:
            from dsperse.src.backends.ezkl import EZKL
            be_instance = EZKL()
            data = be_instance.compilation_pipeline(slice_path, output_dir, input_file_path=calibration_input)
            return data, be_instance.get_version()
            
        return None, None


    def compile(self, model_path: str, input_file: Optional[str] = None, layers: Optional[str] = None):
        """Compile the model, deciding between whole-model or sliced-model compilation."""
        if not os.path.exists(model_path): raise FileNotFoundError(f"Path does not exist: {model_path}")

        # --- Configuration and Layer Parsing ---
        parse_result = CompilerUtils.parse_backend_and_layers(layers)
        if parse_result.default_backend is not None:
            self.default_backend, self.use_fallback = parse_result.default_backend, parse_result.use_fallback

        layer_indices = parse_result.layer_indices
        if parse_result.needs_complex_parse:
            self.layer_backends, self.default_layer_indices = CompilerUtils.parse_complex_layer_backends(layers)
            layer_indices = sorted(set(self.layer_backends.keys()) | self.default_layer_indices)

        CompilerUtils.log_compilation_start(model_path, layer_indices, self.default_backend, self.use_fallback)

        # --- Format Detection and Dispatch ---
        is_sliced, slice_path, format = CompilerUtils.is_sliced_model(model_path)
        if not is_sliced: raise ValueError(f"Invalid model path: {model_path}. Must be a sliced model.")

        if format != "dirs":
            slice_path = Converter.convert(model_path, output_type="dirs", cleanup=True)

        self._compile_slices(slice_path, input_file_path=input_file, layer_indices=layer_indices)

        if format != "dirs":
            slice_path = Converter.convert(slice_path, output_type=format, cleanup=True)

        return slice_path
