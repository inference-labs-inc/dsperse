import json
import os
import shutil
import time
from pathlib import Path

import onnx
import logging

from dsperse.src.utils.utils import Utils

logger = logging.getLogger(__name__)

class OnnxUtils:
    def __init__(self):
        pass

    @staticmethod
    def _get_dsperse_version() -> str:
        """
        Read the dsperse project version from the nearest pyproject.toml.
        Returns a string like "1.0.1" or "unknown" on failure.
        """
        try:
            here = Path(__file__).resolve()
            for parent in [here.parent, *here.parents]:
                pyproject = parent / "pyproject.toml"
                if pyproject.exists():
                    try:
                        txt = pyproject.read_text(encoding="utf-8", errors="ignore")
                        # naive parse: look for first version assignment under [project]
                        in_project = False
                        for line in txt.splitlines():
                            s = line.strip()
                            if s.startswith("[project]"):
                                in_project = True
                                continue
                            if in_project and s.startswith("[") and s.endswith("]"):
                                # left [project] section
                                break
                            if in_project and s.startswith("version") and "=" in s:
                                # version = "x.y.z"
                                try:
                                    val = s.split("=", 1)[1].strip().strip('"').strip("'")
                                    if val:
                                        return val
                                except Exception:
                                    pass
                        # fallback: find any version line
                        for line in txt.splitlines():
                            s = line.strip()
                            if s.startswith("version") and "=" in s:
                                try:
                                    val = s.split("=", 1)[1].strip().strip('"').strip("'")
                                    if val:
                                        return val
                                except Exception:
                                    pass
                    except Exception:
                        continue
        except Exception:
            pass
        return "unknown"


    @staticmethod
    def write_slice_dirs_metadata(slices_root: str):
        """
        Ensure each per-slice directory (slice_#) contains a dslice-style metadata.json
        alongside payload/model.onnx so the folder can be zipped to become a valid .dslice.
        Also updates the global slices metadata slice 'path' to payload/model.onnx if needed.
        """
        root = Path(slices_root)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            alt = root / "slices" / "metadata.json"
            if alt.exists():
                metadata_path = alt
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found near {root}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        # Pull model-level info
        original_model = meta.get("original_model")
        model_type = meta.get("model_type", "ONNX")
        dsperse_ver = OnnxUtils._get_dsperse_version()

        # Best-effort opset
        opset_version = None
        try:
            if original_model and os.path.exists(original_model):
                mdl = onnx.load(original_model)
                for ops in mdl.opset_import:
                    if ops.domain in ("", "ai.onnx"):
                        opset_version = int(ops.version)
                        break
                if opset_version is None and mdl.opset_import:
                    opset_version = int(mdl.opset_import[0].version)
        except Exception:
            opset_version = None

        slices = meta.get("slices", []) or []
        for idx, seg in enumerate(slices):
            seg_path_val = seg.get("path")
            if not seg_path_val:
                continue
            seg_path = Path(seg_path_val)

            # Determine current slice and payload dirs
            if seg_path.suffix == ".onnx":
                payload_dir = seg_path.parent
                slice_dir = payload_dir.parent
            else:
                slice_dir = seg_path if seg_path.is_dir() else seg_path.parent
                payload_dir = slice_dir / "payload"

            # Rename legacy slice_# directory to slice_# if needed
            expected_dir_name = f"slice_{idx}"
            if slice_dir.name != expected_dir_name:
                try:
                    target_dir = slice_dir.parent / expected_dir_name
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    if not target_dir.exists():
                        shutil.move(str(slice_dir), str(target_dir))
                    slice_dir = target_dir
                    payload_dir = slice_dir / "payload"
                except Exception as e:
                    logger.warning(f"Failed to rename slice directory for idx {idx}: {e}")

            # Ensure payload dir exists
            payload_dir.mkdir(parents=True, exist_ok=True)

            # Normalize ONNX filename to slice_{idx}.onnx
            expected_filename = f"slice_{idx}.onnx"
            desired_path = payload_dir / expected_filename

            # Identify existing onnx path candidates
            current_file = None
            if (payload_dir / expected_filename).exists():
                current_file = payload_dir / expected_filename
            elif (payload_dir / "model.onnx").exists():
                current_file = payload_dir / "model.onnx"
            elif seg_path.is_file():
                current_file = seg_path

            if current_file and current_file != desired_path:
                try:
                    shutil.move(str(current_file), str(desired_path))
                except Exception as e:
                    logger.warning(f"Failed to move ONNX for idx {idx} to expected name: {e}")
                    # If move fails but file exists at current_file, set desired_path to it
                    desired_path = current_file
            elif not current_file:
                logger.warning(f"ONNX payload not found for index {idx} under {payload_dir}")
                continue

            tensor_shape = (seg.get("shape", {}) or {}).get("tensor_shape", {}) if isinstance(seg, dict) else {}
            input_shapes = tensor_shape.get("input") or seg.get("input_shape") or seg.get("input_shapes") or []
            output_shapes = tensor_shape.get("output") or seg.get("output_shape") or seg.get("output_shapes") or []

            single_slice_entry = dict(seg)
            single_slice_entry["index"] = idx
            single_slice_entry["filename"] = expected_filename
            single_slice_entry["path"] = str(desired_path)
            single_slice_entry["relative_path"] = str(desired_path.relative_to(root).relative_to("slice_" + str(idx)))
            single_slice_entry["dsperse_version"] = dsperse_ver
            single_slice_entry["opset_version"] = opset_version


            if "shape" not in single_slice_entry or not single_slice_entry["shape"]:
                single_slice_entry["shape"] = {
                    "weight_shape": {
                        "input": [],
                        "output": []
                    },
                    "tensor_shape": {
                        "input": input_shapes,
                        "output": output_shapes,
                    },
                }
            else:
                single_slice_entry["shape"].setdefault("tensor_shape", {
                    "input": input_shapes,
                    "output": output_shapes,
                })

            single_meta = {
                "original_model": original_model,
                "model_type": model_type,
                "total_parameters": single_slice_entry.get("parameters", 0),
                "input_shape": meta.get("input_shape", input_shapes),
                "output_shapes": meta.get("output_shapes", output_shapes),
                "slice_points": [single_slice_entry.get("index")] if single_slice_entry.get("index") is not None else [],
                "slices": [single_slice_entry],
            }

            slice_metadata_path = slice_dir / "metadata.json"
            try:
                with open(slice_metadata_path, "w") as mf:
                    json.dump(single_meta, mf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write slice metadata for {slice_dir}: {e}")

            seg["path"] = str(desired_path)
            seg["relative_path"] = str(desired_path.relative_to(root))
            seg["slice_metadata"] = str(slice_metadata_path.resolve())
            seg["slice_metadata_relative_path"] = str(slice_metadata_path.relative_to(root))
            seg["dsperse_version"] = dsperse_ver
            seg["opset_version"] = opset_version
            

        Utils.save_metadata_file(meta, metadata_path.parent, metadata_path.name)
