import json
import os
import torch
import torch.nn.functional as F

from src.utils.model_utils import ModelUtils

class RunnerUtils:
    def __init__(self):
        pass

    @staticmethod
    def _get_file_path() -> str:
        """Get the parent directory path of the current file."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def preprocess_input(input_path:str, model_directory: str = None, save_reshape: bool = False) -> torch.Tensor:
        """
        Preprocess input data from JSON.
        """

        if os.path.isfile(input_path):
            with open(input_path, 'r') as f:
                input_data = json.load(f)
        else:
            input_path = os.path.join(RunnerUtils._get_file_path(), input_path)
            print(
                f"Warning: Input file not found. Trying to use relative path: {input_path} instead."
            )
            with open(input_path, 'r') as f:
                input_data = json.load(f)

        if isinstance(input_data, dict):
            if 'input_data' in input_data:
                input_data = input_data['input_data']
            elif 'input' in input_data:
                input_data = input_data['input']

        # Convert to tensor
        if isinstance(input_data, list):
            if isinstance(input_data[0], list):
                # 2D input
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                # 1D input
                input_tensor = torch.tensor([input_data], dtype=torch.float32)
        else:
            raise ValueError("Expected input data to be a list or nested list")

        # reshape input tensor for the model
        # input_tensor = RunnerUtils.reshape(input_tensor, model_directory=model_directory)
        # if save_reshape:
        #     ModelUtils.save_tensor_to_json(input_tensor, "input_data_reshaped.json", model_directory)

        return input_tensor
        
        
    @staticmethod
    def process_final_output(torch_tensor):
        """Process the final output of the model."""
        # Apply softmax to get probabilities if not already applied
        if len(torch_tensor.shape) != 2:  # Ensure raw output is 2D [batch_size, num_classes]
            print(f"Warning: Raw output shape {torch_tensor.shape} is not as expected. Reshaping to [1, -1].")
            torch_tensor = torch_tensor.reshape(1, -1)

        probabilities = F.softmax(torch_tensor, dim=1)
        predicted_action = torch.argmax(probabilities, dim=1).item()

        result = {
            "logits": torch_tensor,
            "probabilities": probabilities,
            "predicted_action": predicted_action
        }

        return result

    @staticmethod
    def needs_reshape(input_tensor, model_directory: str = None) -> bool:
        """Check if tensor needs reshaping based on dimensions and model type"""
        if input_tensor.dim() != 2:
            return False
        if input_tensor.size(1) == 3136 and model_directory and 'doom' in model_directory.lower():
            return True
        if input_tensor.size(1) == 3072:
            return True
        return False

    @staticmethod
    def reshape(input_tensor, model_directory: str = None):
        # TODO: remove hardcoding to doom and net
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.dim() == 2 and input_tensor.size(1) == 3136 and 'doom' in model_directory.lower():
            input_tensor = input_tensor.reshape(1, 4, 28, 28)
        elif input_tensor.dim() == 2 and input_tensor.size(1) == 3072:
            if input_tensor.size(0) == 1:
                # Single sample case
                input_tensor = input_tensor.reshape(1, 3, 32, 32)
            else:
                # Multiple samples case (e.g., batch size 100)
                print(f"Processing only the first sample out of {input_tensor.size(0)}")
                input_tensor = input_tensor[0:1].reshape(1, 3, 32, 32)
        else:
            raise ValueError(f"Input tensor has unsupported dimensions: {input_tensor.shape}")

        return input_tensor

    @staticmethod
    def get_segments(slices_directory):
        metadata = ModelUtils.load_metadata(slices_directory)
        if metadata is None:
            return None

        segments = metadata.get('segments', [])
        if not segments:
            print("No segments found in metadata.json")
            return None

        return segments

    @staticmethod
    def save_to_file_shaped(input_tensor: torch.Tensor, file_path: str):
        # Convert tensor to list
        tensor_data = input_tensor.tolist()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save tensor data as JSON
        data = {
            "input": tensor_data
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def save_to_file_flattened(input_tensor: torch.Tensor, file_path: str):
        # Flatten and convert tensor to list
        tensor_data = input_tensor.flatten().tolist()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save flattened tensor data as JSON
        data = {
            "input_data": [tensor_data]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    print(f"Parent path: {RunnerUtils.get_file_path()}")
