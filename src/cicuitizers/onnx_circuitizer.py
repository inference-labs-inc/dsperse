import json
import os
import random
import subprocess
import logging
from pathlib import Path
import onnx
from src.utils.onnx_analyzer import OnnxAnalyzer
from src.utils.onnx_utils import OnnxUtils

# Configure logger
logger = logging.getLogger('kubz.onnx_circuitizer')

class OnnxCircuitizer:
    def __init__(self):
        """
        Initialize the Onnx Circuitizer.

        Args:
            model_directory (str, optional): Directory containing the model or slices.
            input_file_path (str, optional): Path to the input file for calibration.

        Raises:
            RuntimeError: If EZKL is not installed
        """
        self.env = os.environ
        try:
            subprocess.run(['ezkl', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("EZKL CLI is not installed. Please install EZKL before using this circuitizer.")


    def _circuitize_onnx_slices(self, dir_path, input_file_path=None):
        """
        Circuitize ONNX slices found in the provided directory.

        Args:
            dir_path (str): Path to the directory containing ONNX slices.
            input_file_path (str, optional): Path to input data file for calibration.
            
        Returns:
            str: Path to the directory where circuitization results are saved.
        """
        # Ensure model_path is a directory
        if not os.path.isdir(dir_path):
            raise ValueError(f"path must be a directory: {dir_path}")

        # Look for metadata.json in the slices directory
        metadata_path = os.path.join(dir_path, "metadata.json")
        if not os.path.exists(metadata_path):
            # Check for metadata.json in slices subdirectory
            metadata_path = os.path.join(dir_path, "slices", "metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"metadata.json not found in {dir_path} or {os.path.join(dir_path, 'slices')}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {dir_path}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Process each segment
        segments = metadata.get('segments', [])
        for idx, segment in enumerate(segments):
            segment_filename = segment.get('filename')
            if not segment_filename:
                logger.warning(f"No filename found for segment {idx}")
                continue

            segment_path = segment.get('path')
            if not os.path.exists(segment_path):
                logger.warning(f"Segment file not found: {segment_path}")
                continue

            # Create output directory for this segment
            segment_output_path = os.path.dirname(segment_path)

            # Run the circuitization pipeline for this segment and get the circuitization data
            circuitization_data = self._circuitization_pipeline(
                segment_path, 
                segment_output_path,
                input_file_path=input_file_path,
                segment_details=segment
            )
            
            # Add circuitization data to the segment
            segment['circuitization'] = circuitization_data
            logger.info(f"Added circuitization data to segment {idx}")

        # Save the updated metadata back to the file
        from src.utils.onnx_utils import OnnxUtils
        OnnxUtils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))
        logger.info(f"Updated metadata.json with circuitization data")

        logger.info(f"Circuitization of slices completed. Output saved to {os.path.dirname(segment_output_path)}")
        return os.path.dirname(segment_output_path)

    def _circuitize_onnx(self, model_path, input_file_path=None):
        """
        Circuitize a whole ONNX model.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        # Ensure model_path is a file
        if not os.path.isfile(model_path):
            raise ValueError(f"model_path must be a file: {model_path}")

        output_path = os.path.splitext(model_path)[0]

        # Create output directory
        circuit_folder = os.path.join(os.path.dirname(output_path), "model")
        os.makedirs(circuit_folder, exist_ok=True)

        # Run the circuitization pipeline
        self._circuitization_pipeline(model_path, circuit_folder, input_file_path=input_file_path)

        logger.info(f"Circuitization completed. Output saved to {circuit_folder}")
        return circuit_folder

    def _circuitization_pipeline(self, model_path, output_path, input_file_path=None, segment_details=None):
        """
        Run the EZKL circuitization pipeline: gen-settings, calibrate-settings, compile-circuit, setup.

        Args:
            model_path (str): Path to the ONNX model file.
            output_path (str): Base path for output files (without extension).
            input_file_path (str, optional): Path to input data file for calibration.
            segment_details (dict, optional): Details about the segment being processed.

        Returns:
            dict: Dictionary containing paths to generated files and any error information.
        """
        # Ensure model_path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        model_name = Path(model_path).stem

        # Define file paths
        settings_path = os.path.join(output_path, f"{model_name}_settings.json")
        compiled_path = os.path.join(output_path, f"{model_name}_model.compiled")
        vk_path = os.path.join(output_path, f"{model_name}_vk.key")
        pk_path = os.path.join(output_path, f"{model_name}_pk.key")
        
        # Initialize circuitization data dictionary
        circuitization_data = {
            "settings": settings_path,
            "compiled": compiled_path,
            "vk_key": vk_path,
            "pk_key": pk_path,
            "calibration": None
        }

        try:
            # Step 1: Generate settings
            logger.info(f"Generating settings for {model_name}")
            subprocess.run(
                [
                    "ezkl",
                    "gen-settings",
                    "--param-visibility", "fixed",
                    "--input-visibility", "public",
                    "--model", model_path,
                    "--settings-path", settings_path
                ],
                env=self.env,
                check=True
            )

            # Step 2: Create calibration data if input_file_path is provided
            if input_file_path and os.path.exists(input_file_path):
                # Step 3: Calibrate settings
                logger.info(f"Calibrating settings using {input_file_path}")
                subprocess.run(
                    [
                        "ezkl",
                        "calibrate-settings",
                        "--model", model_path,
                        "--settings-path", settings_path,
                        "--data", input_file_path,
                        "--target", "accuracy"
                    ],
                    env=self.env,
                    check=True
                )
                circuitization_data["calibration"] = input_file_path
            else:
                # If no input file, try to analyze the model to create a dummy calibration
                try:
                    logger.info("No input file provided, creating dummy calibration data")
                    # Load the ONNX model
                    onnx_model = onnx.load(model_path)

                    # Create a dummy calibration file
                    calibration_path = os.path.join(output_path, f"{model_name}_calibration.json")
                    self._create_dummy_calibration(onnx_model, calibration_path, segment_details)
                    circuitization_data["calibration"] = calibration_path

                    # Calibrate settings with the dummy data
                    subprocess.run(
                        [
                            "ezkl",
                            "calibrate-settings",
                            "--model", model_path,
                            "--settings-path", settings_path,
                            "--data", calibration_path
                        ],
                        env=self.env,
                        check=True
                    )
                except Exception as e:
                    error_msg = f"Failed to create dummy calibration: {e}"
                    logger.warning(error_msg)
                    logger.warning("Skipping calibration step")
                    circuitization_data["calibration"] = error_msg

            # Step 4: Compile circuit
            logger.info(f"Compiling circuit for {model_path}")
            subprocess.run(
                [
                    "ezkl",
                    "compile-circuit",
                    "--model", model_path,
                    "--settings-path", settings_path,
                    "--compiled-circuit", compiled_path
                ],
                env=self.env,
                check=True
            )

            # Step 5: Setup (generate verification and proving keys)
            logger.info("Setting up verification and proving keys")
            subprocess.run(
                [
                    "ezkl",
                    "setup",
                    "--compiled-circuit", compiled_path,
                    "--vk-path", vk_path,
                    "--pk-path", pk_path
                ],
                env=self.env,
                check=True
            )

            logger.info(f"Circuitization pipeline completed for {model_path}")
        
        except Exception as e:
            error_msg = f"Error during circuitization: {str(e)}"
            logger.error(error_msg)
            # Add error information to the circuitization data
            circuitization_data["error"] = error_msg
        
        return circuitization_data

    def _create_dummy_calibration(self, onnx_model, output_path, segment_details=None):
        """
        Create a dummy calibration file for an ONNX model.

        Args:
            onnx_model: ONNX model
            output_path: Path where to save the calibration file
            segment_details: Details of the segment including shape information
        """
        if segment_details and "shape" in segment_details and "input" in segment_details["shape"]:
            shape = segment_details["shape"]["input"]
            shape = [1 if dim == "batch_size" else dim for dim in shape]
        else:
            shape = []
            for input_info in onnx_model.graph.input:
                dim_shape = []
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        dim_shape.append(1)
                    else:
                        dim_shape.append(dim.dim_value)
                if dim_shape:  # Only add non-empty shapes
                    shape = dim_shape
                    break

            if not shape:
                raise ValueError("Failed to determine input shape from ONNX model or segment details")

        # Create dummy data (all normalized to 0.5)
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        flat_data = [random.random() for _ in range(total_elements)]


        # Create the calibration data JSON structure
        calibration_data = {"input_data": [flat_data]}

        # Write the calibration data to a JSON file
        try:
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f)
            logger.info(f"Created dummy calibration file: {output_path}")
        except Exception as e:
            logger.error(f"Failed to create dummy calibration file: {str(e)}")
            raise


    def circuitize(self, model_path, input_file=None):
        """
        Circuitize an ONNX model or slices.

        Args:
            model_path (str): Path to the ONNX model file or directory containing slices.
            input_file (str): Path to the input file for calibration.

    Raises:
        ValueError: If the model_path is invalid or doesn't contain required files,
        FileNotFoundError: If the model_path doesn't exist
    """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path does not exist: {model_path}")

        # Check if it's a directory with metadata.json
        if os.path.isdir(model_path) and (os.path.exists(os.path.join(model_path, "metadata.json")) or os.path.exists(
                os.path.join(model_path, "slices", "metadata.json"))):
            return self._circuitize_onnx_slices(model_path, input_file)
        # Check if it's an ONNX file
        elif os.path.isfile(model_path) and model_path.endswith('.onnx'):
            return self._circuitize_onnx(model_path, input_file)
        else:
            raise ValueError(
                f"Invalid model path: {model_path}. Must be either a directory containing metadata.json or an .onnx file")



if __name__ == "__main__":
    model_choice = 1

    base_paths = {
        1: "../models/doom",
        2: "../models/net"
    }

    model_dir = base_paths[model_choice] #+ "/model.onnx"
    circuitizer = OnnxCircuitizer()
    model_path = os.path.abspath(model_dir)
    circuitizer.circuitize(model_path=model_path)
