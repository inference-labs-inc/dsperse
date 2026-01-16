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

            ezkl_dir = payload / "ezkl"
            ezkl_expected = [ezkl_dir / "settings.json", ezkl_dir / "model.compiled", ezkl_dir / "vk.key",
                             ezkl_dir / "pk.key"]
            assert any(p.exists() for p in ezkl_expected)

            jst_dir = payload / "jstprove"
            stem = f"slice_{idx}"
            jst_expected = [jst_dir / f"{stem}_circuit.txt", jst_dir / f"{stem}_settings.json"]
            assert any(p.exists() for p in jst_expected) or any(jst_dir.iterdir())

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
                assert "jstprove" in comp and "ezkl" in comp
            elif idx == 1:
                assert "jstprove" not in comp and "ezkl" not in comp
            elif idx == 2:
                assert "jstprove" in comp and "ezkl" not in comp
            elif idx in [3, 4]:
                assert "ezkl" in comp and "jstprove" not in comp
            else:
                assert "jstprove" not in comp and "ezkl" not in comp

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
            tile_size=14
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
            assert comp.get("tile_count") == 4

            files = comp.get("files", {})
            assert "tile_0" in files
            # tile_1 should no longer be explicitly listed (optimized)
            assert "tile_1" not in files

            # Check that files actually exist
            tile_0_circuit = output_dir / files["tile_0"]["compiled"]
            assert tile_0_circuit.exists()
            assert "payload/jstprove/tiled" in files["tile_0"]["compiled"]

            # Per-slice metadata check
            slice_meta_path = output_dir / f"slice_{idx}" / "metadata.json"
            slice_meta = json.loads(slice_meta_path.read_text())
            s_item = slice_meta["slices"][0]
            s_comp = s_item.get("compilation", {}).get("jstprove", {})

            assert s_comp.get("tiled") is True
            assert "tile_0" in s_comp.get("files", {})

            # Path in per-slice metadata should be relative to slice dir
            # It should be payload/jstprove/tiled/...
            tile_0_rel_path = s_comp["files"]["tile_0"]["compiled"]
            assert (output_dir / f"slice_{idx}" / tile_0_rel_path).exists()
            assert tile_0_rel_path.startswith("payload/jstprove/tiled/")