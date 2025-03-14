import json
import os
import torch

def create_test_model(model_dir="models/test_model", ):
    """
    Creates a test model with predefined state dictionaries and configuration settings, storing
    them in the specified directory. The function generates random weights for each layer and
    creates a JSON configuration file detailing the architecture of the model. It ensures the
    specified directory structure exists and writes both the model and configuration to disk.

    :param model_dir: Directory path where the test model and its configuration are saved.
                      Default is "models/test_model".
                      If the directory does not exist, it is created.
    :type model_dir: str
    :return: A tuple containing the file paths of the saved model and its configuration.
    :rtype: tuple
    """

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

def create_test_model_with_embedded_activations(model_dir="models/test_model_embedded"):
    """
    Creates a test model with predefined weights and activations embedded in its configuration.

    This function generates a model file containing:
    1. A state dictionary with random weights for the model layers.
    2. A configuration dictionary that defines the model type, layer structure, and corresponding
       activation functions for each layer.

    The generated file is saved at the specified directory path with the filename `test_model_embedded.pth`.
    The directory path is created if it does not exist.

    :param model_dir: Directory path to save the test model file (string). Defaults to "models/test_model_embedded".
    :return: Full file path to the saved test model file (string).
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

def create_test_model_with_biases(model_dir="models/test_model_with_biases"):
    """
    Creates a test model with specified layer weights, biases, and configuration file. This function is designed
    to generate a dummy model structure saved in `pth` format and a corresponding configuration saved in JSON
    format. The model and configuration files are stored under the specified directory. If the directory does
    not exist, it will be created.

    :param model_dir: Path to the directory where the test model and its configuration will be saved.
                      Default is "models/test_model_with_biases".
    :type model_dir: str
    :return: A tuple containing the paths to the saved model file ("test_model.pth") and configuration file
             ("test_config.json").
    :rtype: Tuple[str, str]
    """

    os.makedirs(os.path.join(os.getcwd(), model_dir), exist_ok=True)

    model_path = os.path.join(model_dir, "test_model.pth")
    config_path = os.path.join(model_dir, "test_config.json")

    # Define layer dimensions for clarity
    layer_dims = [
        (5, 10),  # layer1: 5 input, 10 output
        (10, 20),  # layer2: 10 input, 20 output
        (20, 15),  # layer3: 20 input, 15 output
        (15, 25),  # layer4: 15 input, 25 output
        (25, 30),  # layer5: 25 input, 30 output
        (30, 18),  # layer6: 30 input, 18 output
        (18, 12),  # layer7: 18 input, 12 output
        (12, 22),  # layer8: 12 input, 22 output
        (22, 16),  # layer9: 22 input, 16 output
        (16, 14),  # layer10: 16 input, 14 output
        (14, 28),  # layer11: 14 input, 28 output
        (28, 32),  # layer12: 28 input, 32 output
        (32, 19),  # layer13: 32 input, 19 output
        (19, 2),  # layer14: 19 input, 2 output
    ]

    state_dict = {}

    # Create weights and biases for each layer
    for i, (in_size, out_size) in enumerate(layer_dims, 1):
        # Create weights
        state_dict[f'layer{i}.weight'] = torch.randn(out_size, in_size)
        # Create biases
        state_dict[f'layer{i}.bias'] = torch.randn(out_size)

    config = {
        "model_type": "test",
        "layers": {
            "input": {"size": 5, "type": "input"},
            "layer1": {"size": 10, "activation": "ReLU", "in_features": 5, "out_features": 10},
            "layer2": {"size": 20, "activation": "Tanh", "in_features": 10, "out_features": 20},
            "layer3": {"size": 15, "activation": "Sigmoid", "in_features": 20, "out_features": 15},
            "layer4": {"size": 25, "activation": "LeakyReLU", "in_features": 15, "out_features": 25},
            "layer5": {"size": 30, "activation": "ELU", "in_features": 25, "out_features": 30},
            "layer6": {"size": 18, "activation": "PReLU", "in_features": 30, "out_features": 18},
            "layer7": {"size": 12, "activation": "ReLU", "in_features": 18, "out_features": 12},
            "layer8": {"size": 22, "activation": "Tanh", "in_features": 12, "out_features": 22},
            "layer9": {"size": 16, "activation": "Softplus", "in_features": 22, "out_features": 16},
            "layer10": {"size": 14, "activation": "Softsign", "in_features": 16, "out_features": 14},
            "layer11": {"size": 28, "activation": "ReLU", "in_features": 14, "out_features": 28},
            "layer12": {"size": 32, "activation": "Tanh", "in_features": 28, "out_features": 32},
            "layer13": {"size": 19, "activation": "ReLU", "in_features": 32, "out_features": 19},
            "layer14": {"size": 2, "activation": "Softmax", "in_features": 19, "out_features": 2}
        }
    }

    print(f"Creating test model in {model_dir}/...")
    torch.save(state_dict, model_path)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Test model and config files created successfully")

    return model_path, config_path

def create_test_cnn_model_with_biases(model_dir="models/test_cnn_model_with_biases"):
    """
    Creates a test Convolutional Neural Network (CNN) model along with its weights and
    configuration file. This function generates a synthetic CNN model with predefined
    architecture, saving the model's state dictionary and configuration to the specified
    directory. The CNN consists of convolutional, max-pooling, and fully connected layers.

    :param model_dir: Directory where the CNN model and its configuration will be saved.
                      Defaults to 'models/test_cnn_model_with_biases'.
    :type model_dir: str

    :return: A tuple containing the paths to the saved model file and configuration file.
    :rtype: tuple
    """

    import os
    import json
    import torch

    os.makedirs(os.path.join(os.getcwd(), model_dir), exist_ok=True)

    model_path = os.path.join(model_dir, "test_cnn_model.pth")
    config_path = os.path.join(model_dir, "test_cnn_config.json")

    # Define CNN architecture
    # Format: (layer_type, in_channels/features, out_channels/features, kernel_size, stride, padding)
    cnn_architecture = [
        # Convolutional layers
        ("conv", 3, 16, 3, 1, 1),  # conv1: 3 input channels, 16 output channels
        ("pool", 2, 2, 0),  # max_pool1: kernel_size=2, stride=2
        ("conv", 16, 32, 3, 1, 1),  # conv2: 16 input channels, 32 output channels
        ("pool", 2, 2, 0),  # max_pool2: kernel_size=2, stride=2
        ("conv", 32, 64, 3, 1, 1),  # conv3: 32 input channels, 64 output channels
        ("pool", 2, 2, 0),  # max_pool3: kernel_size=2, stride=2
        ("conv", 64, 128, 3, 1, 1),  # conv4: 64 input channels, 128 output channels
        # Assume input size was 32x32, after 3 pooling layers it's now 4x4
        # Fully connected layers
        ("fc", 128 * 4 * 4, 512),  # fc1: flattened conv output to 512 neurons
        ("fc", 512, 256),  # fc2: 512 to 256 neurons
        ("fc", 256, 128),  # fc3: 256 to 128 neurons
        ("fc", 128, 10),  # fc4: 128 to 10 output classes
    ]

    state_dict = {}
    config = {
        "model_type": "cnn",
        "input_shape": [3, 32, 32],  # Channels, Height, Width
        "layers": {
            "input": {"channels": 3, "height": 32, "width": 32, "type": "input"}
        }
    }

    # Create weights and biases for each layer
    for i, layer_info in enumerate(cnn_architecture, 1):
        if layer_info[0] == "conv":
            _, in_channels, out_channels, kernel_size, stride, padding = layer_info

            # Create weights and biases for conv layers
            state_dict[f'conv{i}.weight'] = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
            state_dict[f'conv{i}.bias'] = torch.randn(out_channels)

            # Add to config
            config["layers"][f"conv{i}"] = {
                "type": "conv2d",
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "activation": "ReLU"
            }

        elif layer_info[0] == "pool":
            _, kernel_size, stride, padding = layer_info

            # No weights for pooling layers
            config["layers"][f"pool{i}"] = {
                "type": "maxpool2d",
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding
            }

        elif layer_info[0] == "fc":
            _, in_features, out_features = layer_info

            # Create weights and biases for fully connected layers
            state_dict[f'fc{i}.weight'] = torch.randn(out_features, in_features)
            state_dict[f'fc{i}.bias'] = torch.randn(out_features)

            # Add to config
            activation = "Softmax" if i == len(cnn_architecture) else "ReLU"  # Last layer gets Softmax
            config["layers"][f"fc{i}"] = {
                "type": "linear",
                "in_features": in_features,
                "out_features": out_features,
                "activation": activation
            }

    print(f"Creating test CNN model in {model_dir}/...")
    torch.save(state_dict, model_path)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Test CNN model and config files created successfully")

    return model_path, config_path

if __name__ == "__main__":
    # generate model
    # create_test_model()
    # create_test_model_with_embedded_activations()
    # create_test_model_with_biases()
    create_test_cnn_model_with_biases()
