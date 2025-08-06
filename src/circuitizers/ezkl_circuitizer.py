import json
import os
import random
import subprocess
import logging
from pathlib import Path
import onnx
from src.utils.utils import Utils

# Configure logger
logger = logging.getLogger(__name__)

class EZKLCircuitizer:
    def __init__(self):
        """
        Initialize the Ezkl Circuitizer.

        Raises:
            RuntimeError: If EZKL is not installed
        """
        self.env = os.environ
        try:
            subprocess.run(['ezkl', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("EZKL CLI is not installed. Please install EZKL before using this circuitizer.")


    def _circuitize_slices(self, dir_path, input_file_path=None, layer_indices=None):
        """
        Circuitize ONNX slices found in the provided directory.

        Args:
            dir_path (str): Path to the directory containing ONNX slices.
            input_file_path (str, optional): Path to input data file for calibration.
            layer_indices (list, optional): List of layer indices to circuitize. If None, all layers will be circuitized.
            
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
        segment_output_path = None
        circuitized_count = 0
        skipped_count = 0
        
        for idx, segment in enumerate(segments):
            # Skip this segment if it's not in the specified layer indices
            if layer_indices is not None and idx not in layer_indices:
                logger.info(f"Skipping segment {idx} as it's not in the specified layers")
                skipped_count += 1
                continue
                
            segment_filename = segment.get('filename')
            if not segment_filename:
                logger.warning(f"No filename found for segment {idx}")
                continue

            segment_path = segment.get('path')
            if not os.path.exists(segment_path):
                logger.warning(f"Segment file not found: {segment_path}")
                continue

            # Create output directory for this segment
            segment_output_path = os.path.join(os.path.dirname(segment_path), "ezkl_circuitization")

            # Run the circuitization pipeline for this segment and get the circuitization data
            circuitization_data = self._circuitization_pipeline(
                segment_path, 
                segment_output_path,
                input_file_path=input_file_path,
                segment_details=segment
            )
            
            # Add circuitization data to the segment
            segment['ezkl_circuitization'] = circuitization_data
            logger.info(f"Added circuitization data to segment {idx}")
            circuitized_count += 1

            # Save the updated metadata back to the file
            Utils.save_metadata_file(metadata, os.path.dirname(metadata_path), os.path.basename(metadata_path))
            logger.info(f"Updated metadata.json with circuitization data")

        if segment_output_path:
            output_dir = os.path.dirname(segment_output_path)
        else:
            output_dir = os.path.dirname(metadata_path)
            
        logger.info(f"Circuitization of slices completed. Circuitized {circuitized_count} segments, skipped {skipped_count} segments.")
        logger.info(f"Output saved to {output_dir}")
        return output_dir

    def _circuitize_model(self, model_path, input_file_path=None):
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
        os.makedirs(output_path, exist_ok=True)
        
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
            process = subprocess.run(
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
            
            if process.returncode != 0:
                logger.warning("Failed to generate settings")
                circuitization_data["gen-settings_error"] = f"Failed to generate settings with EZKL with message {process.stderr}, {process.stderr}"

            # Step 2: Create calibration data if input_file_path is provided
            if input_file_path and os.path.exists(input_file_path):
                # Step 3: Calibrate settings
                logger.info(f"Calibrating settings using {input_file_path}")
                process = subprocess.run(
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
                if process.returncode != 0:
                    logger.warning("Failed to calibrate settings")
                    circuitization_data[
                        "calibrate-settings_error"] = f"Failed to calibrate settings with EZKL with message {process.stderr}, {process.stderr}"

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
                    process = subprocess.run(
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
                    if process.returncode != 0:
                        logger.warning("Failed to calibrate settings")
                        circuitization_data[
                            "calibrate-settings_error"] = f"Failed to calibrate settings with EZKL with message {process.stderr}, {process.stderr}"

                except Exception as e:
                    error_msg = f"Failed to create dummy calibration: {e}"
                    logger.warning(error_msg)
                    logger.warning("Skipping calibration step")
                    circuitization_data["calibration"] = error_msg

            # Step 4: Compile circuit
            logger.info(f"Compiling circuit for {model_path}")
            process = subprocess.run(
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

            if process.returncode != 0:
                logger.warning("Failed to compile circuit")
                circuitization_data["compile-circuit_error"] = f"Failed to compile circuit with EZKL with message {process.stderr}, {process.stderr}"


            # Step 5: Setup (generate verification and proving keys)
            logger.info("Setting up verification and proving keys")
            process = subprocess.run(
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

            if process.returncode != 0:
                logger.warning("Failed to setup (generate keys)")
                circuitization_data["setup_error"] = f"Failed to generate keys with EZKL with message {process.stderr}, {process.stderr}"


            logger.info(f"Circuitization pipeline completed for {model_path}")
        
        except Exception as e:
            error_msg = f"Error during circuitization: {str(e)}"
            logger.error(error_msg)
            # Add error information to the circuitization data
            circuitization_data["error"] = error_msg
        
        return circuitization_data

    @staticmethod
    def _create_dummy_calibration(onnx_model, output_path, segment_details=None):
        """
        Create a dummy calibration file for an ONNX model, handling multiple inputs if needed.

        Args:
            onnx_model: ONNX model
            output_path: Path where to save the calibration file
            segment_details: Details of the segment including shape information
        """
        # Get input shapes from the ONNX model
        input_shapes = []
        input_names = []

        # First, collect all input shapes from the ONNX model (excluding initializers)
        initializers = {init.name for init in onnx_model.graph.initializer}
        for input_info in onnx_model.graph.input:
            if input_info.name not in initializers:  # Skip weights and biases
                input_name = input_info.name
                input_names.append(input_name)

                dim_shape = []
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        dim_shape.append(1)  # Replace named dimensions with 1
                    else:
                        dim_shape.append(dim.dim_value if dim.dim_value != 0 else 1)  # Replace 0 with 1

                if dim_shape:  # Only add non-empty shapes
                    input_shapes.append((input_name, dim_shape))

        logger.info(f"Found {len(input_shapes)} inputs in ONNX model: {input_shapes}")

        # If we have metadata, use it to enhance our understanding of the shapes
        if segment_details and "shape" in segment_details and "tensor_shape" in segment_details["shape"]:
            tensor_shape = segment_details["shape"]["tensor_shape"]
            if "input" in tensor_shape and len(tensor_shape["input"]) > 0:
                # Try to map each ONNX input to the corresponding metadata shape
                for i, (input_name, shape) in enumerate(input_shapes):
                    for meta_shape in tensor_shape["input"]:
                        # Check if this shape contains string dimensions (likely actual inputs, not weights)
                        if any(isinstance(dim, str) for dim in meta_shape):
                            # Found a shape with named dimensions, use it to enhance our understanding
                            enhanced_shape = [1 if isinstance(dim, str) else dim for dim in meta_shape]

                            # Only update if the rank matches or if we're reasonably sure this is the right shape
                            if len(enhanced_shape) == len(shape) or i == len(input_shapes) - 1:
                                input_shapes[i] = (input_name, enhanced_shape)
                                logger.info(f"Enhanced shape for {input_name}: {enhanced_shape}")
                                break

        # Generate random data for each input and combine into a single flat array
        all_flat_data = []

        for input_name, shape in input_shapes:
            # Calculate total elements for this input
            total_elements = 1
            for dim in shape:
                total_elements *= dim

            # Generate random data (consistent with model_circuitizer.py's approach)
            input_data = [random.random() for _ in range(total_elements)]
            all_flat_data.extend(input_data)

            logger.info(f"Generated {len(input_data)} random values for input {input_name} with shape {shape}")

        # If no inputs were found, create a default dummy input
        if not all_flat_data:
            logger.warning("No inputs found, creating default dummy input")
            all_flat_data = [random.random() for _ in range(10)]

        # Create the calibration data JSON structure that EZKL expects
        calibration_data = {"input_data": [all_flat_data]}

        # Write the calibration data to a JSON file
        try:
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f)
            logger.info(f"Created dummy calibration file at {output_path} with {len(all_flat_data)} total values")
        except Exception as e:
            logger.error(f"Failed to create dummy calibration file: {str(e)}")
            raise

    def _parse_layers(self, layers_str):
        """
        Parse a layers string into a list of layer indices.
        
        Args:
            layers_str (str): String specifying which layers to circuitize (e.g., "3, 20-22")
            
        Returns:
            list: List of layer indices to circuitize
        """
        if not layers_str:
            return None
            
        layer_indices = []
        # Split by comma and process each part
        parts = [p.strip() for p in layers_str.split(',')]
        
        for part in parts:
            if '-' in part:
                # Handle range (e.g., "20-22")
                try:
                    start, end = map(int, part.split('-'))
                    layer_indices.extend(range(start, end + 1))
                except ValueError:
                    logger.warning(f"Invalid layer range: {part}. Skipping.")
            else:
                # Handle single number
                try:
                    layer_indices.append(int(part))
                except ValueError:
                    logger.warning(f"Invalid layer index: {part}. Skipping.")
                    
        return sorted(set(layer_indices))  # Remove duplicates and sort
    
    def circuitize(self, model_path, input_file=None, layers=None):
        """
        Circuitize an ONNX model or slices.

        Args:
            model_path (str): Path to the ONNX model file or directory containing slices.
            input_file (str): Path to the input file for calibration.
            layers (str, optional): String specifying which layers to circuitize (e.g., "3, 20-22").
                                   If not provided, all layers will be circuitized.

        Raises:
            ValueError: If the model_path is invalid or doesn't contain required files,
            FileNotFoundError: If the model_path doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path does not exist: {model_path}")
            
        # Parse layers string if provided
        layer_indices = self._parse_layers(layers)
        if layer_indices:
            logger.info(f"Will circuitize only layers with indices: {layer_indices}")
        else:
            logger.info("No layers specified, will circuitize all layers")

        # Check if it's a directory with metadata.json
        if os.path.isdir(model_path) and (os.path.exists(os.path.join(model_path, "metadata.json")) or os.path.exists(
                os.path.join(model_path, "slices", "metadata.json"))):
            return self._circuitize_slices(model_path, input_file, layer_indices)
        # Check if it's an ONNX file
        elif os.path.isfile(model_path) and model_path.endswith('.onnx'):
            if layer_indices:
                logger.warning("Layer selection is only supported for sliced models, not single ONNX files.")
            return self._circuitize_model(model_path, input_file)
        else:
            raise ValueError(
                f"Invalid model path: {model_path}. Must be either a directory containing metadata.json or an .onnx file")



if __name__ == "__main__":
    model_choice = 2

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet"
    }

    model_dir = base_paths[model_choice] #+ "/model.onnx"
    circuitizer = EZKLCircuitizer()
    model_path = os.path.abspath(model_dir)
    circuitizer.circuitize(model_path=model_path)
