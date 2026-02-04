import json
import shutil
import hashlib
import pytest
from pathlib import Path
from types import SimpleNamespace

from dsperse.src.cli.slice import slice_model, slice_convert

class TestSliceE2E:
    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_slice_dirs(self, model_name: str, model_dir: Path, slices_output_dir: Path, capfd):
        args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs",
        )

        # Run CLI slicing entrypoint directly
        slice_model(args)

        # Capture console output and verify key messages (ignore color codes)
        out = capfd.readouterr().out
        assert "Slicing model" in out
        assert "Output directory created:" in out
        assert "ONNX model sliced successfully" in out

        # Verify files are created as expected
        root = slices_output_dir
        assert root.exists() and root.is_dir()

        # Check that at least one slice directory exists
        slice_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("slice_")])
        assert len(slice_dirs) >= 1

        # For each slice dir, ensure payload and ONNX file exist
        for d in slice_dirs:
            payload = d / "payload"
            assert payload.exists() and payload.is_dir()
            # derive expected onnx filename from directory suffix
            try:
                idx = d.name.split("_")[-1]
            except Exception:
                idx = "0"
            onnx_path = payload / f"slice_{idx}.onnx"
            assert onnx_path.exists(), f"Missing slice ONNX: {onnx_path}"

        # Check top-level metadata.json exists and validate essential fields
        metadata_path = root / "metadata.json"
        assert metadata_path.exists(), f"Missing metadata.json at {metadata_path}"
        meta = json.loads(metadata_path.read_text())
        assert isinstance(meta.get("original_model"), str)
        assert meta["original_model"].endswith(".onnx")
        assert isinstance(meta.get("model_type"), str) and len(meta["model_type"]) > 0
        assert isinstance(meta.get("slice_points"), list)
        assert isinstance(meta.get("slices"), list) and len(meta["slices"]) >= 1



    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_slice_dirs_custom_output(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path, capfd):
        """Slicing to a non-default custom output directory should succeed and write expected artifacts."""
        # Provide a custom (non-default) output directory explicitly to avoid prompts
        args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(hardcoded_output_dir),
            save_file=None,
            output_type="dirs",
        )

        # Run CLI slicing entrypoint directly
        slice_model(args)

        # Capture console output and verify key messages
        out = capfd.readouterr().out
        assert "Slicing model" in out
        assert "Output directory created:" in out
        assert "ONNX model sliced successfully" in out

        # Verify files are created as expected in the custom output directory
        root = hardcoded_output_dir
        assert root.exists() and root.is_dir()

        # Check that at least one slice directory exists
        slice_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("slice_")])
        assert len(slice_dirs) >= 1

        # For each slice dir, ensure payload and ONNX file exist
        for d in slice_dirs:
            payload = d / "payload"
            assert payload.exists() and payload.is_dir()
            # derive expected onnx filename from directory suffix
            try:
                idx = d.name.split("_")[-1]
            except Exception:
                idx = "0"
            onnx_path = payload / f"slice_{idx}.onnx"
            assert onnx_path.exists(), f"Missing slice ONNX: {onnx_path}"

        # Check top-level metadata.json exists and validate essential fields
        metadata_path = root / "metadata.json"
        assert metadata_path.exists(), f"Missing metadata.json at {metadata_path}"
        meta = json.loads(metadata_path.read_text())
        assert isinstance(meta.get("original_model"), str)
        assert meta["original_model"].endswith(".onnx")
        assert isinstance(meta.get("model_type"), str) and len(meta["model_type"]) > 0
        assert isinstance(meta.get("slice_points"), list)
        assert isinstance(meta.get("slices"), list) and len(meta["slices"]) >= 1



    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_slice_dslice_output(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path, capfd):
        """Slicing to dslice should produce .dslice files and remove slice_* directories in place."""
        args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(hardcoded_output_dir),
            save_file=None,
            output_type="dslice",
        )

        slice_model(args)

        out = capfd.readouterr().out
        assert "Slicing model" in out
        assert "Output directory created:" in out
        assert "ONNX model sliced successfully" in out

        root = hardcoded_output_dir
        assert root.exists() and root.is_dir()

        # After dirs -> dslice conversion, slice_* directories should be removed
        slice_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("slice_")]
        assert len(slice_dirs) == 0, f"Expected slice_* dirs to be removed, found: {[d.name for d in slice_dirs]}"

        # And .dslice files should be present
        dslice_files = [f for f in root.iterdir() if f.is_file() and f.suffix == ".dslice"]
        assert len(dslice_files) >= 1, "Expected at least one .dslice file"

        # metadata.json should still exist at root
        metadata_path = root / "metadata.json"
        assert metadata_path.exists(), f"Missing metadata.json at {metadata_path}"
        meta = json.loads(metadata_path.read_text())
        assert isinstance(meta.get("slices"), list) and len(meta["slices"]) >= 1


    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_slice_dsperse_output(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path, capfd):
        """Slicing to dsperse should create a <output>.dsperse archive at the parent directory and keep slice_* dirs."""
        args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(hardcoded_output_dir),
            save_file=None,
            output_type="dsperse",
        )

        slice_model(args)

        out = capfd.readouterr().out
        assert "Slicing model" in out
        assert "Output directory created:" in out
        assert "ONNX model sliced successfully" in out

        root = hardcoded_output_dir
        # .dsperse should exist at the parent with the folder name
        dsperse_path = root.parent / f"{root.name}.dsperse"
        assert dsperse_path.exists(), f"Missing dsperse bundle at {dsperse_path}"
        # Converter cleans up the original output directory after bundling
        assert not root.exists(), "Expected output directory to be removed after dsperse conversion"


    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_slice_dirs_with_save_default(self, model_name: str, model_dir: Path, slices_output_dir: Path, capfd):
        """Using --save default should write analysis/model_metadata.json under the model directory."""
        args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file="default",  # trigger default path: <model_dir>/analysis/model_metadata.json
            output_type="dirs",
        )

        slice_model(args)

        out = capfd.readouterr().out
        assert "Slicing model" in out
        assert "Output directory created:" in out
        assert "ONNX model sliced successfully" in out
        assert "Model analysis saved to" in out

        metadata_file = model_dir / "analysis" / "model_metadata.json"
        assert metadata_file.exists(), f"Expected analysis metadata at {metadata_file}"
        data = json.loads(metadata_file.read_text())
        # Spot-check a few key fields
        assert isinstance(data.get("original_model"), str)
        assert data["original_model"].endswith(".onnx")
        assert isinstance(data.get("model_type"), str) and data["model_type"]

    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_slice_with_tiling(self, model_name: str, model_dir: Path, capfd):
        """Verify end-to-end slicing with tiling enabled."""
        output_dir = model_dir.parent / f"{model_name}_tiling_output"
        
        # Clean up before
        if output_dir.exists():
            shutil.rmtree(output_dir)

        args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=1024
        )

        try:
            slice_model(args)

            # Capture console output
            captured = capfd.readouterr()
            combined_out = captured.out + captured.err
            
            assert "Slicing model" in combined_out
            assert "ONNX model sliced successfully" in combined_out
            assert "Tiled" in combined_out and "Conv slices" in combined_out

            # Verify artifacts
            metadata_path = output_dir / "metadata.json"
            assert metadata_path.exists()
            meta = json.loads(metadata_path.read_text())

            tiled_slices = [s for s in meta["slices"] if "tiling" in s]
            # Note: net model may not have tileable Conv layers, doom should have some
            if model_name == "doom":
                assert len(tiled_slices) > 0, "At least one slice should be tiled for doom model"

            for s in tiled_slices:
                assert s["tiling"]["num_tiles"] >= 2
                payload_dir = output_dir / f"slice_{s['index']}" / "payload"
                tiles_dir = payload_dir / "tiles"
                assert tiles_dir.exists()
                # New architecture: only tile.onnx is created, split/concat done in pure Python
                assert (tiles_dir / "tile.onnx").exists()

                # Verify tiling info has tile path
                assert "tile" in s["tiling"]
                assert s["tiling"]["tile"]["path"].endswith("tile.onnx")

        finally:
            # Clean up after
            if output_dir.exists():
                shutil.rmtree(output_dir)

    def test_slice_with_channel_splitting(self, tmp_path):
        """Verify end-to-end channel splitting when spatial tiling is not possible."""
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        
        # Create a model with high input channels and prime spatial dimension
        c_in, c_out, spatial = 64, 64, 37
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])
        np.random.seed(42)
        W = numpy_helper.from_array(np.random.randn(c_out, c_in, 3, 3).astype(np.float32), "W")
        conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, spatial, spatial])
        graph = helper.make_graph([conv], "test", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        model_dir = tmp_path / "prime_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        onnx.save(model, str(model_path))
        
        output_dir = tmp_path / "prime_output"
        
        args = SimpleNamespace(
            model_dir=str(model_path),
            output_dir=str(output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=10000
        )
        
        slice_model(args)
        
        # Verify metadata
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()
        meta = json.loads(metadata_path.read_text())
        
        # Find the Conv slice (likely index 0)
        conv_slice = next((s for s in meta["slices"] if "channel_split" in s), None)
        assert conv_slice is not None, "Channel splitting should have been applied"
        
        cs = conv_slice["channel_split"]
        assert cs["num_groups"] > 1
        
        # Verify group files exist
        payload_dir = output_dir / f"slice_{conv_slice['index']}" / "payload"
        groups_dir = payload_dir / "channel_groups"
        assert groups_dir.exists()
        assert (groups_dir / "group_0.onnx").exists()

    def test_slice_in_memory_build(self, tmp_path, capfd):
        """Verify slices are built directly from in-memory graph data."""
        import onnx
        from onnx import helper, TensorProto

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
        nodes = [helper.make_node("Relu", ["X"], ["Y"])]
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])
        graph = helper.make_graph(nodes, "test", [X], [Y], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        model_dir = tmp_path / "inmem_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        onnx.save(model, str(model_path))

        output_dir = tmp_path / "inmem_output"
        args = SimpleNamespace(
            model_dir=str(model_path),
            output_dir=str(output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=None
        )

        slice_model(args)

        out = capfd.readouterr().out
        assert "ONNX model sliced successfully" in out
        assert (output_dir / "slice_0" / "payload" / "slice_0.onnx").exists()

    def test_slice_constant_filtering(self, tmp_path):
        """Verify that constant-only slices are merged into functional slices."""
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        
        # Model: Constant -> Relu
        c_val = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "C")
        nodes = [
            helper.make_node("Constant", [], ["C"], value=c_val),
            helper.make_node("Relu", ["C"], ["Y"]),
        ]
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
        graph = helper.make_graph(nodes, "test", [], [Y], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        model_dir = tmp_path / "const_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        onnx.save(model, str(model_path))
        
        from dsperse.src.slice.onnx_slicer import OnnxSlicer
        slicer = OnnxSlicer(str(model_path))
        slicer._ensure_analysis() # Initialize analyzer and shapes
        
        # Manually provide slice points that isolate Constant (index 0)
        points = [1, 2]
        model_metadata = slicer.analysis
        
        from dsperse.src.slice.utils.onnx_utils import OnnxUtils
        optimized_points = OnnxUtils.filter_constant_only_slices(points, model_metadata)
        
        # Point 1 should be removed because it creates a constant-only slice [0, 1)
        assert 1 not in optimized_points
        assert 2 in optimized_points

    @pytest.mark.parametrize("model_name", ["net", "doom"])
    def test_slice_convert_roundtrip(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path, tmp_path: Path, capfd):
        """
        End-to-end roundtrip:
        1) Slice to directory layout (slice_*/payload/*.onnx + metadata.json)
        2) Convert in-place to .dslice files
        3) Convert to a single .dsperse archive (removes the output dir)
        4) Convert .dsperse back to directory layout into a fresh folder
        5) Verify restored directory content matches the original (baseline)
        """

        # 1) Slice to dirs at the shared output directory
        args_dirs = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(hardcoded_output_dir),
            save_file=None,
            output_type="dirs",
        )
        slice_model(args_dirs)

        # Capture console output for basic sanity
        out = capfd.readouterr().out
        assert "Slicing model" in out
        assert "Output directory created:" in out
        assert "ONNX model sliced successfully" in out

        root = hardcoded_output_dir
        assert root.exists() and root.is_dir()

        # Take a baseline snapshot by copying the directory to a temp location
        baseline_dir = tmp_path / "baseline"
        shutil.copytree(root, baseline_dir)

        # Helper to collect canonical payload file hashes and metadata.json hash
        def build_manifest(folder: Path):
            hashes = {}
            # payload ONNX files
            for slice_dir in sorted([d for d in folder.iterdir() if d.is_dir() and d.name.startswith("slice_")]):
                payload = slice_dir / "payload"
                onnx_candidates = list(payload.glob("*.onnx"))
                for f in onnx_candidates:
                    rel = f.relative_to(folder).as_posix()
                    h = hashlib.sha256(f.read_bytes()).hexdigest()
                    hashes[rel] = h
            # metadata.json
            meta = folder / "metadata.json"
            if meta.exists():
                hashes["metadata.json"] = hashlib.sha256(meta.read_bytes()).hexdigest()
            return hashes

        baseline_manifest = build_manifest(baseline_dir)
        assert baseline_manifest, "Baseline manifest should not be empty"

        # 2) Convert dirs -> dslice in place
        args_to_dslice = SimpleNamespace(
            input_path=str(root),
            to_type="dslice",
            output_path=None,
            cleanup=True,
        )
        slice_convert(args_to_dslice)

        # After conversion, slice_* dirs should be removed and .dslice files present
        assert all(not d.is_dir() for d in root.glob("slice_*/")), "slice_* directories should be removed after dslice conversion"
        dslice_files = [f for f in root.iterdir() if f.suffix == ".dslice"]
        assert dslice_files, "Expected .dslice files after conversion"

        # 3) Convert dslice -> dsperse in place (this removes the output directory)
        expected_dsperse = root.parent / f"{root.name}.dsperse"
        # Ensure no leftover from prior runs
        if expected_dsperse.exists():
            expected_dsperse.unlink()
        args_to_dsperse = SimpleNamespace(
            input_path=str(root),
            to_type="dsperse",
            output_path=None,
            cleanup=True,
        )
        slice_convert(args_to_dsperse)
        assert expected_dsperse.exists(), f"Missing dsperse archive at {expected_dsperse}"
        assert not root.exists(), "Output dir should be removed after dsperse conversion"

        # 4) Convert dsperse -> dirs into a fresh restored directory
        restored_dir = tmp_path / "restored"
        args_to_dirs = SimpleNamespace(
            input_path=str(expected_dsperse),
            to_type="dirs",
            output_path=str(restored_dir),
            cleanup=True,
        )
        slice_convert(args_to_dirs)
        assert restored_dir.exists() and restored_dir.is_dir()

        restored_manifest = build_manifest(restored_dir)
        assert restored_manifest, "Restored manifest should not be empty"

        # 5) Compare manifests (only payload .onnx and metadata.json)
        assert baseline_manifest == restored_manifest, (
            "Roundtrip mismatch between original sliced output and restored directory"
        )


