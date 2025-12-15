import json
import logging
import os
import time
from pathlib import Path
import torch

# Configure logger
logger = logging.getLogger(__name__)


class Utils:
    """
    Utility functions for working with ONNX models.
    """

    @staticmethod
    def save_metadata_file(metadata, output_path, filename="metadata.json"):
        """
        Save metadata to a JSON file.

        Args:
            metadata: Dictionary containing metadata
            output_path: Directory where the metadata will be saved
            filename: Name of the metadata file (default: "metadata.json")
        """
        output = Path(output_path)

        # Check if the provided path is a directory
        if output.is_dir():
            # Combine the directory with the default or given filename
            file_path = output / filename
        else:
            # Use the path as-is, assuming it includes the filename
            file_path = output

        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write metadata to the file
        with file_path.open('w') as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def find_metadata_path(dir_path: str) -> str:
        """
        Find the metadata.json file in the given directory or its slices subdirectory.

        Args:
            dir_path: Directory path to search for metadata.json

        Returns:
            str: Path to the metadata.json file

        Raises:
            FileNotFoundError: If metadata.json is not found in the directory or its slices subdirectory
        """
        metadata_path = os.path.join(dir_path, "metadata.json")
        if not os.path.exists(metadata_path):
            alt = os.path.join(dir_path, "slices", "metadata.json")
            if os.path.exists(alt):
                metadata_path = alt
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {dir_path} or its slices subdirectory")
        return metadata_path


    @staticmethod
    def filter_inputs(slice_inputs, graph):
        # Filter input names from slice details
        slice_filtered_inputs = []
        for input_info in slice_inputs:
            # Only include actual inputs that are not weights or biases
            # Typically, weights and biases have names containing "weight" or "bias"
            if (not any(pattern in input_info.name.lower() for pattern in ["weight", "bias"]) and
                    input_info.name in [inp.name for inp in graph.input]):
                slice_filtered_inputs.append(input_info.name)
            # Also include intermediate tensors from previous layers
            elif input_info.name.startswith('/'):  # Intermediate tensors often start with '/'
                slice_filtered_inputs.append(input_info.name)
        # If there are no inputs after filtering, include the first non-weight/bias input
        if not slice_filtered_inputs:
            for input_info in slice_inputs:
                if not any(pattern in input_info.name.lower() for pattern in ["weight", "bias"]):
                    slice_filtered_inputs.append(input_info.name)
                    break

            # If still no inputs, use the first input as a fallback
            if not slice_filtered_inputs and slice_inputs:
                slice_filtered_inputs.append(slice_inputs[0].name)
        return slice_filtered_inputs

    @staticmethod
    def _get_original_model_shapes(model_metadata: dict):
        """
        Extract shape information from model metadata.

        Args:
            model_metadata: Dictionary containing model metadata with shape information

        Returns:
            dict: Dictionary mapping tensor names to their shapes
        """
        shapes = {}

        # Extract shapes from input_shape
        input_shape = model_metadata.get("input_shape", [])
        if input_shape and len(input_shape) > 0:
            shapes["input"] = input_shape[0]

        # Extract shapes from output_shapes
        output_shapes = model_metadata.get("output_shapes", [])
        if output_shapes and len(output_shapes) > 0:
            shapes["output"] = output_shapes[0]

        # Extract shapes from nodes if available
        nodes = model_metadata.get("nodes", {})
        for node_name, node_info in nodes.items():
            if "parameter_details" in node_info:
                for param_name, param_info in node_info["parameter_details"].items():
                    if "shape" in param_info:
                        shapes[param_name] = param_info["shape"]

        return shapes

    @staticmethod
    def write_input(tensor: torch.Tensor, file_path):
        """Write tensor to input.json format."""
        data = {"input_data": tensor.tolist()}
        with open(file_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def read_input(file_path) -> torch.Tensor:
        """Read tensor from a flexible input.json format.
        Supported keys: "input_data" (preferred), "input", "data", "inputs".
        If a batch (dim0 > 1) is provided, only the first item is used with a warning.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Try a set of candidate keys to locate the input tensor data
        candidate_keys = ["input_data", "input", "data", "inputs"]
        found_key = None
        array_like = None

        # Direct key lookup
        for k in candidate_keys:
            if k in data:
                found_key = k
                array_like = data[k]
                break

        # If not found, try common nested pattern {"inputs": [{"data": [...]}]}
        if array_like is None and isinstance(data, dict) and "inputs" in data and isinstance(data["inputs"], list) and data["inputs"]:
            first = data["inputs"][0]
            if isinstance(first, dict) and "data" in first:
                found_key = "inputs[0].data"
                array_like = first["data"]

        if array_like is None:
            raise KeyError("Could not find input tensor data in JSON. Expected one of keys: " + ", ".join(candidate_keys))

        tensor = torch.tensor(array_like)

        # If batch dimension is present and > 1, take only the first item
        if tensor.dim() >= 1 and tensor.size(0) > 1:
            logger.warning(f"Input JSON appears to contain a batch of size {tensor.size(0)}; using only the first item. To run batches, call the runner per item.")
            tensor = tensor[0]

        return tensor

    @staticmethod
    def load_run_metadata(run_path: str | Path) -> dict:
        run_path = Path(run_path)
        meta_file = run_path / "metadata.json"
        if not run_path.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Run path invalid; expected {meta_file}")
        with open(meta_file, 'r') as f:
            return json.load(f)

    @staticmethod
    def dirs_root_from(path: Path) -> Path:
        """Return the slices root directory for proving.
        Accepts either a root containing `slice_*` subdirs or a single `slice_*` dir.
        """
        path = Path(path)
        if path.is_dir() and path.name.startswith("slice_"):
            return path.parent
        return path

    @staticmethod
    def iter_circuit_slices(metadata: dict):
        """Yield (slice_id, slice_meta) for slices that have circuit and pk present.
        Prefer `use_circuit` flag; otherwise check presence of compiled circuit + keys.
        """
        slices = (metadata or {}).get("slices", {})
        for sid, meta in slices.items():
            use_circuit = bool(meta.get("use_circuit"))
            circuit_path = meta.get("circuit_path") or meta.get("compiled")
            pk_path = meta.get("pk_path")
            if use_circuit or (circuit_path and pk_path):
                yield sid, meta

    @staticmethod
    def resolve_under_slice(slice_dir: Path, rel_or_abs: str | None) -> str | None:
        if not rel_or_abs:
            return None
        p = str(rel_or_abs)
        if os.path.isabs(p):
            return p
        # strip leading `slice_#` if present
        sd_name = os.path.basename(os.path.abspath(str(slice_dir)))
        parts = p.split(os.sep)
        if parts and parts[0] == sd_name:
            parts = parts[1:]
            p = os.path.join(*parts) if parts else ''
        return os.path.abspath(os.path.join(str(slice_dir), p))

    @staticmethod
    def slice_dirs_path(dirs_root: Path, slice_id: str) -> Path:
        return Path(dirs_root) / slice_id

    @staticmethod
    def witness_path_for(run_path: Path, slice_id: str) -> Path:
        return Path(run_path) / slice_id / "output.json"

    @staticmethod
    def proof_output_path(run_path: Path, slice_id: str, output_root: str | Path | None) -> Path:
        if output_root:
            root = Path(output_root)
            # Treat provided output as a root directory
            return root / slice_id / "proof.json"
        return Path(run_path) / slice_id / "proof.json"

    @staticmethod
    def run_results_path(run_path: Path) -> Path:
        return Path(run_path) / "run_results.json"

    @staticmethod
    def load_run_results(run_path: Path) -> dict:
        rp = Utils.run_results_path(run_path)
        if not rp.exists():
            # Minimal skeleton if inference wasn't run (unlikely)
            return {"execution_chain": {"execution_results": []}}
        with open(rp, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_run_results(run_path: Path, data: dict) -> None:
        rp = Utils.run_results_path(run_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def merge_execution_into_run_results(run_results: dict, execution_data: dict, execution_type: str) -> tuple[
        dict, int]:
        """Attach per-slice execution data (proof or verification) and compute success count.

        Args:
            run_results: The run results dictionary to update
            execution_data: Mapping slice_id -> {success, time_sec, error, ...}
            execution_type: Either 'proof' or 'verification'

        Returns: (updated_results, success_count)
        """
        exec_chain = (run_results or {}).setdefault("execution_chain", {})
        results = exec_chain.setdefault("execution_results", [])
        success_count = 0

        # Build quick index by slice_id/segment_id
        by_id = {}
        for entry in results:
            sid = entry.get("slice_id") or entry.get("segment_id")
            if sid:
                by_id[sid] = entry

        for sid, info in execution_data.items():
            # Build execution entry based on type
            if execution_type == "proof":
                exec_entry = {
                    "proof_file": info.get("proof_path"),
                    "success": bool(info.get("success")),
                    # Standardized timing key
                    "time_sec": float(info.get("time_sec", 0.0)),
                }
            elif execution_type == "verification":
                exec_entry = {
                    # Keep a simple boolean while also storing success
                    "verified": bool(info.get("success")),
                    "success": bool(info.get("success")),
                    # Standardized timing key
                    "time_sec": float(info.get("time_sec", 0.0)),
                }
            else:
                raise ValueError(f"Invalid execution_type: {execution_type}. Must be 'proof' or 'verification'")

            # Add error if present and not successful
            if not info.get("success") and info.get("error"):
                exec_entry["error"] = info.get("error")

            # Determine the key name for this execution type
            exec_key = f"{execution_type}_execution"

            # Update or append entry
            if sid in by_id:
                by_id[sid][exec_key] = exec_entry
            else:
                # If run_results lacks this slice, append a minimal entry
                results.append({"slice_id": sid, exec_key: exec_entry})

            if info.get("success"):
                success_count += 1

        return run_results, success_count

    @staticmethod
    def update_metadata_after_execution(run_path: Path, total: int, success_count: int, execution_type: str) -> None:
        """Update metadata with execution summary (proof or verification).

        Args:
            run_path: Path to the run directory
            total: Total number of slices processed
            success_count: Number of successful executions
            execution_type: Either 'proof' or 'verification'
        """
        meta_path = Path(run_path) / "metadata.json"
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            logger.warning(f"Could not read metadata at {meta_path} to update {execution_type} summary.")
            return

        # Determine summary key and field names based on execution type
        if execution_type == "proof":
            summary_key = "proof_summary"
            fields = {
                "total_circuit_slices": int(total),
                "proved_slices": int(success_count),
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
            }
        elif execution_type == "verification":
            summary_key = "verification_summary"
            fields = {
                "total_proof_candidates": int(total),
                "verified_slices": int(success_count),
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
            }
        else:
            raise ValueError(f"Invalid execution_type: {execution_type}. Must be 'proof' or 'verification'")

        # Non-breaking addition: add/refresh a top-level summary
        meta.setdefault(summary_key, {})
        meta[summary_key].update(fields)

        try:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write updated metadata to {meta_path}: {e}")
