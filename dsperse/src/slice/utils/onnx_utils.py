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
        Ensure each per-segment directory (segment_#) contains a dslice-style metadata.json
        alongside payload/model.onnx so the folder can be zipped to become a valid .dslice.
        Also updates the global slices metadata segment 'path' to payload/model.onnx if needed.
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

        segments = meta.get("segments", []) or []
        for idx, seg in enumerate(segments):
            seg_path_val = seg.get("path")
            if not seg_path_val:
                continue
            seg_path = Path(seg_path_val)

            # Determine current segment and payload dirs
            if seg_path.suffix == ".onnx":
                payload_dir = seg_path.parent
                segment_dir = payload_dir.parent
            else:
                segment_dir = seg_path if seg_path.is_dir() else seg_path.parent
                payload_dir = segment_dir / "payload"

            # Rename legacy segment_# directory to slice_# if needed
            expected_dir_name = f"slice_{idx}"
            if segment_dir.name != expected_dir_name:
                try:
                    target_dir = segment_dir.parent / expected_dir_name
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    if not target_dir.exists():
                        shutil.move(str(segment_dir), str(target_dir))
                    segment_dir = target_dir
                    payload_dir = segment_dir / "payload"
                except Exception as e:
                    logger.warning(f"Failed to rename segment directory for idx {idx}: {e}")

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

            # Build dslice-style metadata.json with zero-based slice id and correct entry path
            segment_id = f"slice_{idx}"
            deps = seg.get("dependencies", {}) if isinstance(seg, dict) else {}
            tensor_shape = (seg.get("shape", {}) or {}).get("tensor_shape", {}) if isinstance(seg, dict) else {}
            input_shapes = tensor_shape.get("input") or seg.get("input_shape") or seg.get("input_shapes") or []
            output_shapes = tensor_shape.get("output") or seg.get("output_shape") or seg.get("output_shapes") or []
            input_names = deps.get("filtered_inputs") or deps.get("input") or []
            output_names = deps.get("output") or []

            io_meta = {
                "input_shape": input_shapes,
                "output_shape": output_shapes,
                "input_names": input_names,
                "output_names": output_names,
            }
            dslice_meta = {
                "schema": "dslice/1.0",
                "slice_id": segment_id,
                "backend": "onnx",
                "entry": {"model": f"payload/{expected_filename}"},
                "io": io_meta,
                "deps": deps,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dsperse_version": dsperse_ver,
                "model_type": model_type,
                "original_model": original_model,
            }
            if opset_version is not None:
                dslice_meta["opset_version"] = opset_version

            # Write metadata.json inside slice dir
            slice_metadata_path = segment_dir / "metadata.json"
            try:
                with open(slice_metadata_path, "w") as mf:
                    json.dump(dslice_meta, mf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write slice metadata for {segment_dir}: {e}")

            # Update global metadata with normalized payload path and slice metadata path
            seg["path"] = str(desired_path)
            seg["slice_metadata"] = str(slice_metadata_path.resolve())

        # Save updated global metadata (legacy model-level)
        Utils.save_metadata_file(meta, metadata_path.parent, metadata_path.name)
