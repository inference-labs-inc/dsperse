import json
import os
import torch


def create_test_model_with_embedded_activations(model_dir="test_models/test_model_embedded"):
    """
    Creates a test model with activation function information embedded inside the .pth file itself.
    Returns the path to the saved model.
    """

    os.makedirs(os.path.join(os.getcwd(), model_dir), exist_ok=True)

    model_path = os.path.join(model_dir, "test_model_embedded.pth")

    # State dictionary for the model weights
    state_dict = {
        'layer1.weight': torch.randn(10, 5),
        'layer2.weight': torch.randn(20, 10),
        'layer3.weight': torch.randn(15, 20),
        'layer4.weight': torch.randn(25, 15),
        'layer5.weight': torch.randn(30, 25),
        'layer6.weight': torch.randn(18, 30),
        'layer7.weight': torch.randn(12, 18),
        'layer8.weight': torch.randn(22, 12),
        'layer9.weight': torch.randn(16, 22),
        'layer10.weight': torch.randn(14, 16),
        'layer11.weight': torch.randn(28, 14),
        'layer12.weight': torch.randn(32, 28),
        'layer13.weight': torch.randn(19, 32),
        'layer14.weight': torch.randn(2, 19)
    }

    # Configuration with activation details
    config = {
        "model_type": "test",
        "layers": {
            "input": {"size": 5, "type": "input"},
            "layer1": {"size": 10, "activation": "ReLU"},
            "layer2": {"size": 20, "activation": "Tanh"},
            "layer3": {"size": 15, "activation": "Sigmoid"},
            "layer4": {"size": 25, "activation": "LeakyReLU"},
            "layer5": {"size": 30, "activation": "ELU"},
            "layer6": {"size": 18, "activation": "PReLU"},
            "layer7": {"size": 12, "activation": "ReLU"},
            "layer8": {"size": 22, "activation": "Tanh"},
            "layer9": {"size": 16, "activation": "Softplus"},
            "layer10": {"size": 14, "activation": "Softsign"},
            "layer11": {"size": 28, "activation": "ReLU"},
            "layer12": {"size": 32, "activation": "Tanh"},
            "layer13": {"size": 19, "activation": "ReLU"},
            "layer14": {"size": 2, "activation": "Softmax"}
        }
    }

    # Combine state_dict and config so everything is in a single file
    combined_dict = {
        'state_dict': state_dict,
        'config': config
    }

    # Save model weights and config together
    print(f"Creating test model (with embedded activations) in {model_dir}/...")
    torch.save(combined_dict, model_path)
    print("Test model file (with activations embedded) created successfully.")

    return model_path


def create_test_model(model_dir="test_models/test_model",):
    """Creates a test model with activation function information"""

    os.makedirs(os.path.join(os.getcwd(), model_dir), exist_ok=True)

    model_path = os.path.join(model_dir, "test_model.pth")
    config_path = os.path.join(model_dir, "test_config.json")

    state_dict = {
        'layer1.weight': torch.randn(10, 5),
        'layer2.weight': torch.randn(20, 10),
        'layer3.weight': torch.randn(15, 20),
        'layer4.weight': torch.randn(25, 15),
        'layer5.weight': torch.randn(30, 25),
        'layer6.weight': torch.randn(18, 30),
        'layer7.weight': torch.randn(12, 18),
        'layer8.weight': torch.randn(22, 12),
        'layer9.weight': torch.randn(16, 22),
        'layer10.weight': torch.randn(14, 16),
        'layer11.weight': torch.randn(28, 14),
        'layer12.weight': torch.randn(32, 28),
        'layer13.weight': torch.randn(19, 32),
        'layer14.weight': torch.randn(2, 19)
    }

    config = {
        "model_type": "test",
        "layers": {
            "input": {"size": 5, "type": "input"},
            "layer1": {"size": 10, "activation": "ReLU"},
            "layer2": {"size": 20, "activation": "Tanh"},
            "layer3": {"size": 15, "activation": "Sigmoid"},
            "layer4": {"size": 25, "activation": "LeakyReLU"},
            "layer5": {"size": 30, "activation": "ELU"},
            "layer6": {"size": 18, "activation": "PReLU"},
            "layer7": {"size": 12, "activation": "ReLU"},
            "layer8": {"size": 22, "activation": "Tanh"},
            "layer9": {"size": 16, "activation": "Softplus"},
            "layer10": {"size": 14, "activation": "Softsign"},
            "layer11": {"size": 28, "activation": "ReLU"},
            "layer12": {"size": 32, "activation": "Tanh"},
            "layer13": {"size": 19, "activation": "ReLU"},
            "layer14": {"size": 2, "activation": "Softmax"}
        }
    }

    print(f"Creating test model in {model_dir}/...")
    torch.save(state_dict, model_path)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Test model and config files created successfully")

    return model_path, config_path