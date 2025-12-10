"""
Utility functions to convert sliced directory outputs into dslice files or a single dsperse bundle.

This module centralizes conversion so higher-level orchestrators (like Slicer) can remain minimal.
"""
from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


class Converter:

    @staticmethod
    def convert(path: str, output_type: str = "dirs", output_path: str = None, cleanup: bool = True):
        """Convert between dslices, dsperse, or directory outputs.

        Args:
            path: Path to the directory containing slices or .dslice files or .dsperse file.
            output_type: Type of output to generate ('dirs', 'dslice', 'dsperse')
            output_path: Optional custom output path. If not provided, converts in place or alongside.
            cleanup: When True, remove the original source artifacts after successful conversion
                (e.g., delete the .dsperse file after extracting to dirs). Defaults to True.

        Returns:
            str: Path to the converted output
        """
        # check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        if output_type not in ["dirs", "dslice", "dsperse"]:
            raise ValueError(f"Invalid output type: {output_type}")

        # Determine current type based on path analysis
        path_obj = Path(path)
        current_type = Converter.detect_type(path_obj)

        # If already in desired format, return
        if current_type == output_type:
            logger.info(f"Already in desired format: {output_type}")
            print(f"Already in desired format: {output_type}")
            return str(path_obj)

        logger.info(f"Converting from {current_type} to {output_type}")

        # Route to appropriate conversion method
        if current_type == "dirs":
            if output_type == "dslice":
                return Converter._dirs_to_dslice(path_obj, output_path)
            elif output_type == "dsperse":
                return Converter._dirs_to_dsperse(path_obj, output_path)

        elif current_type == "dslice":
            if output_type == "dirs":
                return Converter._dslice_to_dirs(path_obj, output_path)
            elif output_type == "dsperse":
                # Convert dslice -> dirs -> dsperse
                temp_dirs = Converter._dslice_to_dirs(path_obj, None)
                result = Converter._dirs_to_dsperse(Path(temp_dirs), output_path)
                # Optionally clean up temp dirs
                return result

        elif current_type == "dsperse":
            if output_type == "dirs":
                # Extract and expand contained .dslice files into slice_* directories
                result = Converter._dsperse_to_dirs(path_obj, output_path, expand_slices=True)
                if cleanup:
                    try:
                        path_obj.unlink()
                        logger.info(f"Removed source dsperse file: {path_obj}")
                    except Exception as e:
                        logger.warning(f"Could not remove source dsperse file {path_obj}: {e}")
                return result
            elif output_type == "dslice":
                # Convert dsperse -> dirs (without expanding slices)
                temp_dirs = Converter._dsperse_to_dirs(path_obj, None, expand_slices=False)
                # The dsperse already contains dslices, so just return the directory
                logger.info(f"Extracted dsperse to directory with dslice files: {temp_dirs}")
                if cleanup:
                    try:
                        path_obj.unlink()
                        logger.info(f"Removed source dsperse file: {path_obj}")
                    except Exception as e:
                        logger.warning(f"Could not remove source dsperse file {path_obj}: {e}")
                return temp_dirs

    @staticmethod
    def detect_type(path: str | Path) -> str:
        """Detect the type of the given path.
        """
        # Convert string to Path if needed
        path = Path(path) if isinstance(path, str) else path

        if path.is_file():
            if path.suffix == '.dsperse':
                return 'dsperse'
            elif path.suffix == '.dslice':
                return 'dslice'
            else:
                raise ValueError(f"Unknown file type: {path.suffix}")

        elif path.is_dir():
            # Check if it contains slice_* directories (dirs format)
            if any(d.is_dir() and d.name.startswith('slice_') for d in path.iterdir()):
                return 'dirs'

            # Check if it's a single slice directory
            if Converter._is_slice_dir(path):
                return 'dirs'

            # Check if it contains dslice files at root
            if any(f.is_file() and f.suffix == '.dslice' for f in path.iterdir()):
                return 'dslice'

            raise ValueError(f"Cannot determine type of directory at {path}")

        raise ValueError(f"Invalid path: {path}")

    @staticmethod
    def _is_slice_dir(path: Path) -> bool:
        """Check if a directory is a single slice directory."""
        if not path.is_dir():
            return False
        metadata = path / "metadata.json"
        payload = path / "payload"
        return metadata.exists() and payload.exists()

    # ===== dirs -> dslice =====
    @staticmethod
    def _dirs_to_dslice(path: Path, output_path: str = None) -> str:
        """Convert directory format to dslice files.

        Zips each slice_* directory (containing payload/ and metadata.json) into a .dslice file.
        Also supports zipping a single slice directory (metadata.json + payload/) into <dir>.dslice.
        """
        # Find all slice_* directories
        slice_dirs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith('slice_')]) if path.is_dir() else []

        # If no slice_* subdirectories, but the provided path itself is a slice directory, zip it directly
        if not slice_dirs:
            if Converter._is_slice_dir(path):
                # Determine destination .dslice path
                if output_path:
                    dslice_out = Path(output_path)
                    if dslice_out.is_dir() or not dslice_out.suffix:
                        dslice_out = dslice_out / f"{path.name}.dslice"
                else:
                    dslice_out = path.parent / f"{path.name}.dslice"
                dslice_out.parent.mkdir(parents=True, exist_ok=True)

                # Zip the slice directory into a single .dslice
                Converter._zip_directory(path, dslice_out)
                logger.info(f"Created {dslice_out}")

                # If converting in place, remove the original slice directory
                if output_path is None or Path(output_path) == path:
                    try:
                        shutil.rmtree(path)
                        logger.info(f"Removed {path}")
                    except Exception as e:
                        logger.warning(f"Could not remove source slice directory {path}: {e}")

                return str(dslice_out)
            else:
                raise ValueError(f"No slice_* directories found in {path}")

        # Normal case: zip each slice_* directory into a .dslice at the root
        output_dir = Path(output_path) if output_path else path
        output_dir.mkdir(parents=True, exist_ok=True)

        dslice_files = []
        for slice_dir in slice_dirs:
            # Zip the entire slice_* directory (contains payload/ and metadata.json)
            dslice_path = output_dir / f"{slice_dir.name}.dslice"
            Converter._zip_directory(slice_dir, dslice_path)
            dslice_files.append(dslice_path)
            logger.info(f"Created {dslice_path}")

        # Clean up slice_* directories if converting in place
        if output_path is None or Path(output_path) == path:
            for slice_dir in slice_dirs:
                shutil.rmtree(slice_dir)
                logger.info(f"Removed {slice_dir}")

        logger.info(f"Converted {len(dslice_files)} slices to .dslice format")
        return str(output_dir)

    # ===== dirs -> dsperse =====
    @staticmethod
    def _dirs_to_dsperse(path: Path, output_path: str = None) -> str:
        """Convert directory format to a single dsperse file.

        Zips each slice_* directory into .dslice files at the root level,
        then zips everything (*.dslice + metadata.json) into a .dsperse file.

        Structure inside .dsperse:
        - slice_0.dslice
        - slice_1.dslice
        - ...
        - metadata.json
        """
        # Find all slice_* directories
        slice_dirs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith('slice_')])

        if not slice_dirs:
            raise ValueError(f"No slice_* directories found in {path}")

        # Zip each slice_* directory directly into the root as slice_X.dslice
        dslice_files = []
        for slice_dir in slice_dirs:
            # Extract the number from slice_X to name the dslice file
            slice_num = slice_dir.name.split('_')[-1]
            dslice_out = path / f"slice_{slice_num}.dslice"
            Converter._zip_directory(slice_dir, dslice_out)
            dslice_files.append(dslice_out)
            logger.info(f"Packaged {slice_dir.name} as {dslice_out.name}")

        # Determine output location for the .dsperse file
        if output_path:
            dsperse_out = Path(output_path)
            # If a directory or a path without suffix is provided, use the source folder name
            if dsperse_out.is_dir() or not dsperse_out.suffix:
                dsperse_out = dsperse_out / f"{path.name}.dsperse"
        else:
            # Put it in the parent directory of the slices folder, using the folder's name
            dsperse_out = path.parent / f"{path.name}.dsperse"

        # Ensure destination directory exists
        dsperse_out.parent.mkdir(parents=True, exist_ok=True)

        # Zip the entire directory: *.dslice + metadata.json -> <slices_name>.dsperse
        # Exclude slice_* directories since we already have them as .dslice files
        Converter._zip_directory(path, dsperse_out, exclude_patterns=['slice_*'])
        logger.info(f"Created dsperse archive: {dsperse_out}")

        # Cleanup: remove .dslice files and slice_* dirs
        for dslice_file in dslice_files:
            if dslice_file.exists():
                dslice_file.unlink()
        for slice_dir in slice_dirs:
            if slice_dir.exists():
                shutil.rmtree(slice_dir)

        # Also remove the model-level metadata.json (it's included inside the dsperse)
        metadata_file = path / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

        try:
            # Check if directory is empty before removing
            if path.exists() and not any(path.iterdir()):
                path.rmdir()
            else:
                shutil.rmtree(path)
        except OSError:
            pass

        return str(dsperse_out)

    # ===== dslice -> dirs =====
    @staticmethod
    def _dslice_to_dirs(path: Path, output_path: str = None) -> str:
        """Convert dslice files to directory format.

        Unzips .dslice files back into slice_* directories.
        """
        if path.is_file():
            # Single dslice file
            if output_path:
                output_dir = Path(output_path)
            else:
                output_dir = path.parent / path.stem

            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            Converter._unzip_file(path, output_dir)
            logger.info(f"Extracted {path.name} to {output_dir}")
            return str(output_dir)

        elif path.is_dir():
            # Directory containing multiple dslice files
            dslice_files = sorted([f for f in path.iterdir() if f.suffix == '.dslice'])

            if not dslice_files:
                raise ValueError(f"No .dslice files found in {path}")

            output_dir = Path(output_path) if output_path else path

            for dslice_file in dslice_files:
                slice_dir = output_dir / dslice_file.stem
                slice_dir.mkdir(parents=True, exist_ok=True)
                Converter._unzip_file(dslice_file, slice_dir)
                logger.info(f"Extracted {dslice_file.name} to {slice_dir}")

                # Remove dslice file if converting in place
                if output_path is None or Path(output_path) == path:
                    dslice_file.unlink()

            logger.info(f"Converted {len(dslice_files)} .dslice files to directories")
            return str(output_dir)

    # ===== dsperse -> dirs =====
    @staticmethod
    def _dsperse_to_dirs(path: Path, output_path: str = None, expand_slices: bool = False) -> str:
        """Convert dsperse file to directory format.

        Unzips .dsperse file (contains *.dslice + metadata.json at root level),
        optionally expanding the .dslice files into slice_* directories.
        """
        if not path.is_file() or path.suffix != '.dsperse':
            raise ValueError(f"Expected .dsperse file, got {path}")

        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = path.parent / path.stem

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the dsperse archive (contains *.dslice + metadata.json at root)
        Converter._unzip_file(path, output_dir)
        logger.info(f"Extracted dsperse to {output_dir}")

        # If expand_slices is True, extract the dslice files into slice_* directories
        if expand_slices:
            dslice_files = sorted([f for f in output_dir.iterdir() if f.suffix == '.dslice'])
            for dslice_file in dslice_files:
                # Extract to slice_* directory at the output_dir level
                slice_dir = output_dir / dslice_file.stem
                slice_dir.mkdir(parents=True, exist_ok=True)
                Converter._unzip_file(dslice_file, slice_dir)
                logger.info(f"Expanded {dslice_file.name} to {slice_dir}")
                dslice_file.unlink()

        return str(output_dir)

    # ===== Helper methods =====
    @staticmethod
    def _zip_directory(source_dir: Path, output_file: Path, exclude_patterns: list = None):
        """Zip a directory to a file, preserving the internal structure."""
        exclude_patterns = exclude_patterns or []

        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                root_path = Path(root)

                # Check if we should skip this directory
                if exclude_patterns:
                    rel_root = root_path.relative_to(source_dir)
                    skip = False
                    for pattern in exclude_patterns:
                        if pattern.startswith('*'):
                            # Match file extension patterns
                            pass  # We'll check files individually
                        elif str(rel_root).startswith(pattern.replace('*', '')):
                            skip = True
                            break
                    if skip:
                        continue

                for file in files:
                    # Skip files matching exclude patterns
                    if exclude_patterns:
                        skip_file = False
                        for pattern in exclude_patterns:
                            if pattern.startswith('*') and file.endswith(pattern[1:]):
                                skip_file = True
                                break
                        if skip_file:
                            continue

                    file_path = root_path / file
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)

    @staticmethod
    def _unzip_file(zip_file: Path, output_dir: Path):
        """Unzip a file to a directory, preserving the internal structure."""
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall(output_dir)

if __name__ == "__main__":
    # Basic logger setup when running this module directly
    logging.basicConfig(level=logging.INFO)

    # Choose which model to test
    model_choice = 1  # Change this to test different models

    # Model configurations (relative to this file, similar to slicer.py)
    base_paths = {
        1: "../../../models/doom",
        2: "../../../models/net",
        3: "../../../models/resnet",
        4: "../../../models/age",
        5: "../../../models/version",
        6: "../../../models/bert",
        7: "../../../models/roberta",
    }

    # Resolve paths
    abs_path = os.path.abspath(base_paths[model_choice])

    # Common paths used for conversions
    slices_dir = os.path.join(abs_path, "slices")
    dsperse_file = os.path.join(abs_path, "slices.dsperse")

    try:
        print("\n=== Converter test harness ===")
        print(f"Selected model root: {abs_path}")

        # DSPERSE -> DIRS
        # out_path = Converter.convert(dsperse_file, output_type="dirs")
        # print(f"Extracted dsperse to dirs: {out_path}")

        # DSPERSE -> DSLICE
        # out_path = Converter.convert(dsperse_file, output_type="dslice")
        # print(f"Extracted dsperse to dslice: {out_path}")

        # DIRS -> DSPERSE
        # out_path = Converter.convert(slices_dir, output_type="dsperse")
        # print(f"Converted dirs to dsperse: {out_path}")

        # DIRS -> DSLICE
        out_path = Converter.convert(slices_dir, output_type="dslice")
        print(f"Converted dirs to dslice: {out_path}")

        # DSLICE -> DIRS
        # out_path = Converter.convert(slices_dir, output_type="dirs")
        # print(f"Converted dslice to dirs: {out_path}")

        # DSLICE -> DSPERSE
        # out_path = Converter.convert(slices_dir, output_type="dsperse")
        # print(f"Converted dslice to dsperse: {out_path}")

    except Exception as e:
        print(f"Error during conversion: {e}")
