import json
import os
import random
import subprocess
import torch
import logging
from pathlib import Path
import onnx
from src.utils.utils import Utils
from src.utils.runner_utils.runner_utils import RunnerUtils

# Configure logger
logger = logging.getLogger(__name__)

class EZKL:
    def __init__(self, model_directory=None):
        """
        Initialize the EZKL backend.

        Args:
            model_directory (str, optional): Path to the model directory.

        Raises:
            RuntimeError: If EZKL is not installed
        """
        self.env = os.environ
        self.model_directory = model_directory
        
        if model_directory:
            self.base_path = os.path.join(model_directory, "ezkl")
        
        # Check if ezkl is installed via cli
        try:
            result = subprocess.run(['ezkl', '--version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError("EZKL CLI not found. Please install EZKL first.")
        except FileNotFoundError:
            raise RuntimeError("EZKL CLI not found. Please install EZKL first.")

    #
    # High-level methods that dispatch to specific implementations
    #
    
    def generate_witness(self, input_file: str, model_path: str, output_file: str, vk_path: str):
        """
        Generate a witness for the given model and input.
        
        Args:
            input_file (str): Path to the input file
            model_path (str): Path to the compiled model
            output_file (str): Path where to save the output
            vk_path (str): Path to the verification key
            
        Returns:
            tuple: (success, output) where success is a boolean and output is the processed witness output
        """
        # Normalize possible Path-like arguments to strings for subprocess and logging clarity
        input_file = str(input_file)
        model_path = str(model_path)
        output_file = str(output_file)
        vk_path = str(vk_path)

        # Validate required files exist
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vk_path):
            raise FileNotFoundError(f"Verification key file not found: {vk_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "gen-witness",
                    "--data", input_file,
                    "--compiled-circuit", model_path,
                    "--output", output_file,
                    "--vk-path", vk_path
                ],
                env=self.env,
                check=True,
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                error_msg = f"Witness generation failed with return code {process.returncode}"
                if process.stderr:
                    error_msg += f"\nError: {process.stderr}"
                return False, error_msg

        except subprocess.CalledProcessError as e:
            error_msg = f"Witness generation failed: {e}"
            if e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            return False, error_msg

        # return the processed outputs
        with open(output_file, "r") as f:
            witness_data = json.load(f)
            output = self.process_witness_output(witness_data)

        return True, output

    def prove(self, witness_path: str, model_path: str, proof_path: str, pk_path: str, check_mode: str = "unsafe"):
        """
        Generate a proof for the given witness and model.
        
        Args:
            witness_path (str): Path to the witness file
            model_path (str): Path to the compiled model
            proof_path (str): Path where to save the proof
            pk_path (str): Path to the proving key
            check_mode (str, optional): Check mode for the prover. Defaults to "unsafe".
            
        Returns:
            tuple: (success, results) where success is a boolean and results is the path to the proof
        """
        # Normalize path-like args
        witness_path = str(witness_path)
        model_path = str(model_path)
        proof_path = str(proof_path)
        pk_path = str(pk_path)

        # Validate required files exist
        if not os.path.exists(witness_path):
            raise FileNotFoundError(f"Witness file not found: {witness_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(pk_path):
            raise FileNotFoundError(f"PK key file not found: {pk_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(proof_path), exist_ok=True)

        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "prove",
                    "--check-mode", check_mode,
                    "--witness", witness_path,
                    "--compiled-circuit", model_path,
                    "--proof-path", proof_path,
                    "--pk-path", pk_path
                ],
                env=self.env,
                check=True,
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                error_msg = f"Proof generation failed with return code {process.returncode}"
                if process.stderr:
                    error_msg += f"\nError: {process.stderr}"
                return False, error_msg

        except subprocess.CalledProcessError as e:
            print(f"Error during proof generation: {e}")
            return False, e.stderr

        results = proof_path
        return True, results

    def verify(self, proof_path: str, settings_path: str, vk_path: str) -> bool:
        """
        Verify a proof.
        
        Args:
            proof_path (str): Path to the proof file
            settings_path (str): Path to the settings file
            vk_path (str): Path to the verification key
            
        Returns:
            bool: True if verification succeeded, False otherwise
        """
        # Normalize path-like args
        proof_path = str(proof_path)
        settings_path = str(settings_path)
        vk_path = str(vk_path)

        # Validate required files exist
        if not os.path.exists(proof_path):
            raise FileNotFoundError(f"Proof file not found: {proof_path}")
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        if not os.path.exists(vk_path):
            raise FileNotFoundError(f"Verification key file not found: {vk_path}")

        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "verify",
                    "--proof-path", proof_path,
                    "--settings-path", settings_path,
                    "--vk-path", vk_path
                ],
                env=self.env,
                check=True,
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                error_msg = f"Verification generation failed with return code {process.returncode}"
                if process.stderr:
                    error_msg += f"\nError: {process.stderr}"
                return False
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error verifying proof: {e}")
            return False


    def gen_settings(self, model_path: str, settings_path: str, param_visibility: str = "fixed", input_visibility: str = "public"):
        """
        Generate EZKL settings.
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        os.makedirs(os.path.dirname(settings_path) or ".", exist_ok=True)
        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "gen-settings",
                    "--param-visibility", param_visibility,
                    "--input-visibility", input_visibility,
                    "--model", model_path,
                    "--settings-path", settings_path,
                ],
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                return False, process.stderr or "gen-settings failed"
            return True, None
        except subprocess.CalledProcessError as e:
            return False, getattr(e, "stderr", str(e))

    def calibrate_settings(self, model_path: str, settings_path: str, data_path: str, target: str = None):
        """
        Calibrate EZKL settings using provided data.
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Calibration data file not found: {data_path}")
        cmd = [
            "ezkl",
            "calibrate-settings",
            "--model", model_path,
            "--settings-path", settings_path,
            "--data", data_path,
        ]
        if target:
            cmd += ["--target", target]
        try:
            process = subprocess.run(
                cmd,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                return False, process.stderr or "calibrate-settings failed"
            return True, None
        except subprocess.CalledProcessError as e:
            return False, getattr(e, "stderr", str(e))

    def compile_circuit(self, model_path: str, settings_path: str, compiled_path: str):
        """
        Compile EZKL circuit.
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        os.makedirs(os.path.dirname(compiled_path) or ".", exist_ok=True)
        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "compile-circuit",
                    "--model", model_path,
                    "--settings-path", settings_path,
                    "--compiled-circuit", compiled_path,
                ],
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                return False, process.stderr or "compile-circuit failed"
            return True, None
        except subprocess.CalledProcessError as e:
            return False, getattr(e, "stderr", str(e))

    def setup(self, compiled_path: str, vk_path: str, pk_path: str):
        """
        Generate proving and verification keys (setup).
        Returns (success: bool, error: str|None)
        """
        if not os.path.exists(compiled_path):
            raise FileNotFoundError(f"Compiled circuit file not found: {compiled_path}")
        os.makedirs(os.path.dirname(vk_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(pk_path) or ".", exist_ok=True)
        try:
            process = subprocess.run(
                [
                    "ezkl",
                    "setup",
                    "--compiled-circuit", compiled_path,
                    "--vk-path", vk_path,
                    "--pk-path", pk_path,
                ],
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                return False, process.stderr or "setup failed"
            return True, None
        except subprocess.CalledProcessError as e:
            return False, getattr(e, "stderr", str(e))

    def circuitization_pipeline(self, model_path, output_path, input_file_path=None, segment_details=None):
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
            ok, err = self.gen_settings(model_path=model_path, settings_path=settings_path)
            if not ok:
                logger.warning("Failed to generate settings")
                circuitization_data["gen-settings_error"] = err

            # Step 2/3: Calibrate settings
            if input_file_path and os.path.exists(input_file_path):
                logger.info(f"Calibrating settings using {input_file_path}")
                ok, err = self.calibrate_settings(model_path=model_path, settings_path=settings_path, data_path=input_file_path, target="accuracy")
                circuitization_data["calibration"] = input_file_path
                if not ok:
                    logger.warning("Failed to calibrate settings")
                    circuitization_data["calibrate-settings_error"] = err
            else:
                # If no input file, create dummy calibration
                try:
                    logger.info("No input file provided, creating dummy calibration data")
                    onnx_model = onnx.load(model_path)
                    calibration_path = os.path.join(output_path, f"{model_name}_calibration.json")
                    self._create_dummy_calibration(onnx_model, calibration_path, segment_details)
                    circuitization_data["calibration"] = calibration_path
                    ok, err = self.calibrate_settings(model_path=model_path, settings_path=settings_path, data_path=calibration_path)
                    if not ok:
                        logger.warning("Failed to calibrate settings")
                        circuitization_data["calibrate-settings_error"] = err
                except Exception as e:
                    error_msg = f"Failed to create dummy calibration: {e}"
                    logger.warning(error_msg)
                    logger.warning("Skipping calibration step")
                    circuitization_data["calibration"] = error_msg

            # Step 4: Compile circuit
            logger.info(f"Compiling circuit for {model_path}")
            ok, err = self.compile_circuit(model_path=model_path, settings_path=settings_path, compiled_path=compiled_path)
            if not ok:
                logger.warning("Failed to compile circuit")
                circuitization_data["compile-circuit_error"] = err

            # Step 5: Setup (generate verification and proving keys)
            logger.info("Setting up verification and proving keys")
            ok, err = self.setup(compiled_path=compiled_path, vk_path=vk_path, pk_path=pk_path)
            if not ok:
                logger.warning("Failed to setup (generate keys)")
                circuitization_data["setup_error"] = err

            logger.info(f"Circuitization pipeline completed for {model_path}")
        
        except Exception as e:
            error_msg = f"Error during circuitization: {str(e)}"
            logger.error(error_msg)
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


    @staticmethod
    def process_witness_output(witness_data):
        """
        Process the witness.json data to get prediction results.
        """
        try:
            rescaled_outputs = witness_data["pretty_elements"]["rescaled_outputs"][0]
        except KeyError:
            print("Error: Could not find rescaled_outputs in witness data")
            return None

        # Convert string values to float and create a tensor
        float_values = [float(val) for val in rescaled_outputs]

        # Create a tensor with shape [1, num_classes] to match batch_size, num_classes format
        tensor_output = torch.tensor([float_values], dtype=torch.float32)

        # Process the tensor through _process_final_output (simulating one segment)
        output = RunnerUtils.process_final_output(tensor_output)
        return output


if __name__ == "__main__":
    # Choose which model to test
    model_choice = 1 # Change this to test different models

    base_paths = {
        1: "../models/doom",
        2: "../models/net",
        3: "../models/resnet",
        4: "../models/yolov3"
    }
    abs_path = os.path.abspath(base_paths[model_choice])
    model_dir = abs_path
    slices_dir = os.path.join(abs_path, "slices")

    # Circuitize
    model_path = os.path.abspath(model_dir)
    EZKL().circuitize(model_path=abs_path)

    # # Generate witness
    # input_file = os.path.join(model_dir, "input.json")
    # model_path = os.path.join(model_dir, "model.compiled")
    # vk_path = os.path.join(model_dir, "vk.json")
    # output_file = os.path.join(model_dir, "witness.json")
    # result = ezkl.generate_witness(input_file=input_file, model_path=model_path, output_file=output_file, vk_path=vk_path)
    # print(result)