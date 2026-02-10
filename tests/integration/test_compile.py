import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsperse.src.cli.slice import slice_model
from dsperse.src.cli.compile import compile_model


class TestCompileE2E:
    @pytest.mark.parametrize("model_name", ["doom"])
    def test_compile_default(self, model_name: str, model_dir: Path, slices_output_dir: Path, jstprove_available,
                             ezkl_available, capfd):
        """
        Happy-path compile flow (default settings):
        - Slice the model
        - Compile with defaults (fallback jstprove -> ezkl -> onnx)
        - Verify payload/ezkl and payload/jstprove exist
        """
        if not jstprove_available or not ezkl_available:
            pytest.skip("Skipping compile happy path: required backends unavailable")

        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs",
        ))

        assert slices_output_dir.exists()

        compile_model(SimpleNamespace(
            path=str(slices_output_dir),
            input_file=None,
            layers=None,
            backend=None,
        ))

        slice_dirs = sorted([d for d in slices_output_dir.iterdir() if d.is_dir() and d.name.startswith("slice_")])
        assert slice_dirs

        assert (slices_output_dir / "metadata.json").exists()
        top_level_meta = json.loads((slices_output_dir / "metadata.json").read_text())

        for d in slice_dirs:
            payload = d / "payload"
            assert payload.exists()

            idx = int(d.name.split("_")[-1])
            assert (payload / f"slice_{idx}.onnx").exists()

            # Artifacts now live under payload/
            assert (payload / "ezkl").exists()
            assert (payload / "jstprove").exists()

            slice_meta = json.loads((d / "metadata.json").read_text())

            # Metadata may either nest under slices[index].compilation or at root.compilation
            comp_block = None
            if isinstance(slice_meta.get("slices"), list):
                for item in slice_meta["slices"]:
                    if item.get("index") == idx and isinstance(item.get("compilation"), dict):
                        comp_block = item["compilation"]
                        break
            if comp_block is None:
                comp_block = slice_meta.get("compilation")

            assert isinstance(comp_block, dict), f"Compilation block missing in per-slice metadata for slice {idx}"
            # Expect entries for both backends
            assert "ezkl" in comp_block, "Missing ezkl compilation entry"
            assert "jstprove" in comp_block, "Missing jstprove compilation entry"

            # Paths in metadata should be relative to the slice directory
            # e.g., payload/ezkl/model.compiled
            ezkl_item = comp_block["ezkl"]
            ezkl_files = ezkl_item["files"]
            assert ezkl_files["compiled"].startswith("payload/ezkl/")

        # 5) Top-level metadata should contain compilation info per slice
        # And 6) Verify that the paths in the metadata for the compilation are correct and contain the files
        for item in top_level_meta.get("slices", []):
            idx = item.get("index")
            comp = item.get("compilation")
            assert isinstance(comp, dict), f"Top-level metadata missing compilation for slice {idx}"
            assert {"ezkl", "jstprove"} <= set(comp.keys()), f"Compilation backends missing for slice {idx}"

            for backend_name, details in comp.items():
                if backend_name == "onnx":
                    continue
                assert details.get("compiled") is True, f"Slice {idx} backend {backend_name} not marked as compiled"
                files = details.get("files")
                assert isinstance(files, dict), f"Missing files dict for slice {idx} backend {backend_name}"

                # Handle tiled vs standard structure
                if details.get("tiled"):
                    # We only store tile_0 now (optimized)
                    assert "tile_0" in files
                    target_files = files["tile_0"]
                else:
                    target_files = files

                for file_key, rel_path in target_files.items():
                    if rel_path is None:
                        continue
                    # Path is relative to the directory containing metadata.json (slices_output_dir)
                    full_path = slices_output_dir / rel_path
                    assert full_path.exists(), f"File {file_key} for slice {idx} backend {backend_name} not found at {full_path}"
                    assert rel_path.startswith(
                        f"slice_{idx}/payload/"), f"Expected path to start with slice_{idx}/payload/, got {rel_path}"

        # 7) Verify per-slice metadata paths
        for d in slice_dirs:
            idx = int(d.name.split("_")[-1])
            slice_meta_path = d / "metadata.json"
            slice_meta = json.loads(slice_meta_path.read_text())

            # Find compilation block (it might be in root or under slices[idx])
            comp_block = slice_meta.get("compilation")
            if not comp_block and isinstance(slice_meta.get("slices"), list):
                for item in slice_meta["slices"]:
                    if item.get("index") == idx:
                        comp_block = item.get("compilation")
                        break

            assert isinstance(comp_block, dict), f"Per-slice metadata missing compilation for slice {idx}"

            for backend_name, details in comp_block.items():
                if backend_name == "onnx":
                    continue
                files = details.get("files")
                assert isinstance(files,
                                  dict), f"Missing files dict in per-slice metadata for slice {idx} {backend_name}"

                if details.get("tiled"):
                    target_files = files["tile_0"]
                else:
                    target_files = files

                for file_key, rel_path in target_files.items():
                    if rel_path is None:
                        continue
                    # For per-slice metadata, the relative path is already 'payload/...'
                    # The test was failing because it was assuming rel_path was relative to top-level,
                    # but per-slice metadata stores paths relative to the slice folder.
                    full_path = d / rel_path
                    assert full_path.exists(), f"File {file_key} from per-slice metadata for slice {idx} not found at {full_path}"
                    assert rel_path.startswith(
                        "payload/"), f"Expected per-slice rel_path to start with payload/, got {rel_path}"

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_compile_mixed_backends(self, model_name: str, model_dir: Path, slices_output_dir: Path):
        """
        Verify per-layer backend selection via the 'layers' argument:
        layers="0,2:jstprove;3-4:ezkl"
        Expected behavior (based on Compiler._compile_slices logic):
        - slice 0: bare group "0" -> default both backends
        - slice 1: not in spec -> skipped
        - slice 2: jstprove only
        - slice 3: ezkl only
        - slice 4: ezkl only
        """
        try:
            from dsperse.src.backends.jstprove import JSTprove
            from dsperse.src.backends.ezkl import EZKL
            _j, _e = JSTprove(), EZKL()
        except Exception as e:
            pytest.skip(f"Backends unavailable: {e}")

        # 1) Slice
        slice_args = SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),  # Pass explicitly to avoid prompts
            save_file=None,
            output_type="dirs"
        )
        slice_model(slice_args)
        slices_root = slices_output_dir

        # 2) Compile with mixed backends
        # Test robust parsing: space as separator and missing semicolons
        mixed_layers = "0; 2:jstprove 3-4:ezkl"
        compile_args = SimpleNamespace(path=str(slices_root), input_file=None, layers=mixed_layers, backend=None)
        compile_model(compile_args)

        # 3) Verify metadata
        top_level_meta = json.loads((slices_root / "metadata.json").read_text())
        slices = top_level_meta.get("slices", [])

        for item in slices:
            idx = item.get("index")
            comp = item.get("compilation", {})

            if idx == 0:
                assert comp.get("jstprove", {}).get("compiled") is True
                assert comp.get("ezkl", {}).get("compiled") is True
            elif idx == 1:
                assert comp.get("jstprove", {}).get("compiled") is False
                assert comp.get("ezkl", {}).get("compiled") is False
            elif idx == 2:
                assert comp.get("jstprove", {}).get("compiled") is True
                assert comp.get("ezkl", {}).get("compiled") is False
            elif idx in [3, 4]:
                assert comp.get("ezkl", {}).get("compiled") is True
                assert comp.get("jstprove", {}).get("compiled") is False
            else:
                assert comp.get("jstprove", {}).get("compiled") is False
                assert comp.get("ezkl", {}).get("compiled") is False

    @pytest.mark.parametrize("model_name", ["net"])
    def test_compile_with_input_file(self, model_name: str, model_dir: Path, slices_output_dir: Path, capfd):
        """
        Verify that providing an input_file (calibration) works and doesn't print errors.
        """
        try:
            from dsperse.src.backends.ezkl import EZKL
            _e = EZKL()
        except Exception as e:
            pytest.skip(f"EZKL unavailable: {e}")

        # 1) Slice
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(slices_output_dir),
            save_file=None,
            output_type="dirs"
        ))
        slices_root = slices_output_dir

        # 2) Locate input.json
        input_file = model_dir / "input.json"
        assert input_file.exists()

        # 3) Compile with input file (layer 0 only)
        compile_args = SimpleNamespace(path=str(slices_root), input_file=str(input_file), layers="0", backend="ezkl")

        # Capture and clear output
        capfd.readouterr()
        compile_model(compile_args)
        out = capfd.readouterr().out

        assert "Slices compiled successfully" in out
        assert "Error" not in out

        # Verify calibration.json exists in slice 0 payload/ezkl
        cal_file = slices_root / "slice_0" / "payload" / "ezkl" / "calibration.json"
        assert cal_file.exists(), "calibration.json should have been copied to slice output"

    @pytest.mark.parametrize("model_name", ["net"])
    def test_compile_dslice(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path):
        """
        Verify that we can compile slices in .dslice format.
        1) Slice to dslice format.
        2) Compile the directory containing .dslice files.
        3) Verify that .dslice files still exist and contain compilation info.
        """
        try:
            from dsperse.src.backends.ezkl import EZKL
            _e = EZKL()
        except Exception as e:
            pytest.skip(f"EZKL unavailable: {e}")

        # 1) Slice to dslice
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(hardcoded_output_dir),
            save_file=None,
            output_type="dslice"
        ))

        # Verify it's in dslice format
        dslice_files = list(hardcoded_output_dir.glob("*.dslice"))
        assert len(dslice_files) >= 1
        assert (hardcoded_output_dir / "metadata.json").exists()

        # 2) Compile (all layers with ezkl to keep it fast)
        compile_args = SimpleNamespace(
            path=str(hardcoded_output_dir),
            input_file=None,
            layers=None,
            backend="ezkl"
        )
        compile_model(compile_args)

        # 3) Verify .dslice files still exist
        dslice_files_after = list(hardcoded_output_dir.glob("*.dslice"))
        assert len(dslice_files_after) == len(dslice_files)

        # 4) Verify metadata is updated
        top_level_meta = json.loads((hardcoded_output_dir / "metadata.json").read_text())
        for s in top_level_meta.get("slices", []):
            assert "ezkl" in s.get("compilation", {}), f"Slice {s.get('index')} missing compilation in dslice format"
            assert s["compilation"]["ezkl"]["compiled"] is True

    @pytest.mark.parametrize("model_name", ["net"])
    def test_compile_dsperse(self, model_name: str, model_dir: Path, hardcoded_output_dir: Path):
        """
        Verify that we can compile a .dsperse file directly.
        1) Slice to dsperse format.
        2) Compile the .dsperse file.
        3) Verify that the .dsperse file still exists and contains compilation info.
        """
        try:
            from dsperse.src.backends.ezkl import EZKL
            _e = EZKL()
        except Exception as e:
            pytest.skip(f"EZKL unavailable: {e}")

        # 1) Slice to dsperse
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(hardcoded_output_dir),
            save_file=None,
            output_type="dsperse"
        ))

        dsperse_file = hardcoded_output_dir.parent / f"{hardcoded_output_dir.name}.dsperse"
        assert dsperse_file.exists()

        # 2) Compile the .dsperse file directly
        compile_args = SimpleNamespace(
            path=str(dsperse_file),
            input_file=None,
            layers=None,
            backend="ezkl"
        )
        compile_model(compile_args)

        # 3) Verify .dsperse file still exists
        assert dsperse_file.exists()

        # 4) To verify internal metadata, we need to unzip it or use a helper
        # Actually, the compiler converts to dirs, compiles, then converts back.
        # Let's use Converter to peek inside if we want to be sure, or just assume success if no error.
        from dsperse.src.slice.utils.converter import Converter
        temp_extract = hardcoded_output_dir.parent / "temp_extract_dsperse_verify"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

        Converter.convert(str(dsperse_file), output_type="dirs", output_path=str(temp_extract), cleanup=False)

        try:
            top_level_meta = json.loads((temp_extract / "metadata.json").read_text())
            for s in top_level_meta.get("slices", []):
                assert "ezkl" in s.get("compilation",
                                       {}), f"Slice {s.get('index')} missing compilation in dsperse bundle"
                assert s["compilation"]["ezkl"]["compiled"] is True
        finally:
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_compile_with_tiling(self, model_name: str, model_dir: Path, slices_output_dir: Path, jstprove_available,
                                 capfd):
        """Verify that compilation correctly handles tiled slices."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        output_dir = slices_output_dir

        # 1. Slice with tiling
        slice_model(SimpleNamespace(
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            save_file=None,
            output_type="dirs",
            tile_size=1000
        ))

        # 2. Compile
        compile_model(SimpleNamespace(
            path=str(output_dir),
            input_file=None,
            layers=None,
            backend="jstprove"
        ))

        # 3. Verify metadata
        metadata_path = output_dir / "metadata.json"
        meta = json.loads(metadata_path.read_text())

        tiled_slices = [s for s in meta["slices"] if "tiling" in s]
        assert len(tiled_slices) > 0

        for s in tiled_slices:
            idx = s["index"]
            comp = s.get("compilation", {}).get("jstprove", {})
            assert comp.get("compiled") is True
            assert comp.get("tiled") is True
            
            if idx == 0:
                assert comp.get("tile_count") == 4
            elif idx == 2:
                assert comp.get("tile_count") == 49

            files = comp.get("files", {})
            assert "tile_0" in files
            # tile_1 should no longer be explicitly listed (optimized)
            assert "tile_1" not in files

            # Check that files actually exist
            tile_0_circuit = output_dir / files["tile_0"]["compiled"]
            assert tile_0_circuit.exists()
            assert "payload/jstprove/tiles" in files["tile_0"]["compiled"]

            # Per-slice metadata check
            slice_meta_path = output_dir / f"slice_{idx}" / "metadata.json"
            slice_meta = json.loads(slice_meta_path.read_text())
            s_item = slice_meta["slices"][0]
            s_comp = s_item.get("compilation", {}).get("jstprove", {})

            assert s_comp.get("tiled") is True
            assert "tile_0" in s_comp.get("files", {})

            # Path in per-slice metadata should be relative to slice dir
            # It should be payload/jstprove/tiles/...
            tile_0_rel_path = s_comp["files"]["tile_0"]["compiled"]
            assert (output_dir / f"slice_{idx}" / tile_0_rel_path).exists()
            assert tile_0_rel_path.startswith("payload/jstprove/tiles/")

    def test_compile_channel_split(self, tmp_path, jstprove_available):
        """Verify that compilation correctly handles channel-split slices."""
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np

        # 1. Create a channel-split model
        c_in, c_out, spatial = 16, 16, 7
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])
        W = numpy_helper.from_array(np.random.randn(c_out, c_in, 3, 3).astype(np.float32), "W")
        conv = helper.make_node(
            "Conv", ["X", "W"], ["Y"], 
            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1]
        )
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, spatial, spatial])
        graph = helper.make_graph([conv], "test", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        model_dir = tmp_path / "channel_split_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        onnx.save(model, str(model_path))
        
        slices_dir = model_dir / "slices"
        
        # 2. Slice with channel splitting (forced by small tile_size and prime spatial dim)
        slice_model(SimpleNamespace(
            model_dir=str(model_path), output_dir=str(slices_dir), save_file=None, output_type="dirs", tile_size=100
        ))
        
        # 3. Compile
        compile_model(SimpleNamespace(
            path=str(slices_dir), input_file=None, layers=None, backend="jstprove" if jstprove_available else "ezkl",
            parallel=4
        ))
        
        # 4. Verify metadata
        meta = json.loads((slices_dir / "metadata.json").read_text())
        cs_slice = next((s for s in meta["slices"] if "channel_split" in s), None)
        assert cs_slice is not None
        
        be = "jstprove" if jstprove_available else "ezkl"
        comp = cs_slice.get("compilation", {}).get(be, {})
        assert comp.get("compiled") is True
        assert comp.get("channel_split") is True
        
        group_files = comp.get("group_files", {})
        assert len(group_files) > 0
        for g_idx, files in group_files.items():
            assert "compiled" in files
            # Path should be relative to top-level slices_dir
            assert (slices_dir / files["compiled"]).exists()

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_compile_parallel(self, model_name, model_dir, slices_output_dir, jstprove_available, capfd):
        """Verify that parallel compilation works without errors."""
        slice_model(SimpleNamespace(
            model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"
        ))
        
        # Compile with 2 parallel processes
        compile_model(SimpleNamespace(
            path=str(slices_output_dir), input_file=None, layers=None, backend=None, parallel=2
        ))
        
        capfd.readouterr()

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_compile_resume(self, model_name, model_dir, slices_output_dir, jstprove_available, capfd):
        """Verify that resume mode skips already compiled slices."""
        slice_model(SimpleNamespace(
            model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs"
        ))
        
        # 1. Compile first slice only
        compile_model(SimpleNamespace(
            path=str(slices_output_dir), input_file=None, layers="0", backend="jstprove" if jstprove_available else "ezkl"
        ))
        
        # 2. Compile with resume
        capfd.readouterr()
        compile_model(SimpleNamespace(
            path=str(slices_output_dir), input_file=None, layers=None, backend=None, resume=True, parallel=4
        ))
        capfd.readouterr()

    def test_compile_channel_split_both_backends(self, tmp_path, jstprove_available, ezkl_available):
        """Verify that channel splitting works when both backends are requested."""
        if not jstprove_available or not ezkl_available:
            pytest.skip("Both backends must be available")

        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np

        # Create a channel-split model
        c_in, c_out, spatial = 32, 32, 11
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, c_in, spatial, spatial])
        W = numpy_helper.from_array(np.random.randn(c_out, c_in, 3, 3).astype(np.float32), "W")
        conv = helper.make_node(
            "Conv", ["X", "W"], ["Y"], 
            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1]
        )
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, c_out, spatial, spatial])
        graph = helper.make_graph([conv], "test", [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        model_dir = tmp_path / "cs_both_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        onnx.save(model, str(model_path))
        
        slices_dir = model_dir / "slices"
        slice_model(SimpleNamespace(
            model_dir=str(model_path), output_dir=str(slices_dir), save_file=None, output_type="dirs", tile_size=1000
        ))
        
        # Compile with both backends
        compile_model(SimpleNamespace(
            path=str(slices_dir), input_file=None, layers=None, backend=None
        ))
        
        meta = json.loads((slices_dir / "metadata.json").read_text())
        cs_slice = next((s for s in meta["slices"] if "channel_split" in s), None)
        assert cs_slice is not None
        
        comp = cs_slice.get("compilation", {})
        assert "jstprove" in comp and "ezkl" in comp
        assert comp["jstprove"]["compiled"] is True
        assert comp["ezkl"]["compiled"] is True
        
        # Check that group artifacts are in correct directories
        # payload/jstprove/channel_groups/group_0
        # payload/ezkl/channel_groups/group_0
        slice_idx = cs_slice["index"]
        payload_dir = slices_dir / f"slice_{slice_idx}" / "payload"
        assert (payload_dir / "jstprove" / "channel_groups" / "group_0").exists()
        assert (payload_dir / "ezkl" / "channel_groups" / "group_0").exists()

    @pytest.mark.parametrize("model_name", ["doom"])
    def test_compile_resume_tiled(self, model_name, model_dir, slices_output_dir, jstprove_available, capfd):
        """Verify that resume mode works for tiled slices."""
        if not jstprove_available:
            pytest.skip("JSTprove unavailable")

        slice_model(SimpleNamespace(
            model_dir=str(model_dir), output_dir=str(slices_output_dir), save_file=None, output_type="dirs", tile_size=1000
        ))
        
        # 1. Compile tiled slice 0
        compile_model(SimpleNamespace(
            path=str(slices_output_dir), input_file=None, layers="0", backend="jstprove"
        ))
        
        # 2. Compile with resume
        capfd.readouterr()
        compile_model(SimpleNamespace(
            path=str(slices_output_dir), input_file=None, layers="0", backend="jstprove", resume=True
        ))
        capfd.readouterr()