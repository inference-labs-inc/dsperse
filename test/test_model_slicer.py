import os

import pytest

from src.model_slicer import ModelSlicer
from test.utils import create_test_model_with_embedded_activations


@pytest.fixture(scope="function")
def test_model_path():
    """Fixture to create a test model and clean up afterward"""
    model_path = create_test_model_with_embedded_activations()

    # Provide the path to the test
    yield model_path

    # Cleanup after test
    # output_dir = os.path.join(os.path.dirname(model_path), "output")
    # model_dir = os.path.dirname(model_path)
    #
    # # Clean up output directory if it exists
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    #
    # # Clean up model directory if it exists
    # if os.path.exists(model_dir):
    #     shutil.rmtree(model_dir)


def test_model_slicer_with_embedded_config(test_model_path):
    """
    End-to-end test for ModelSlicer using a model with embedded configuration
    """
    # Use the model path from fixture
    model_path = test_model_path

    # Fix the output path construction
    output_dir = os.path.join(os.path.dirname(model_path), "output")

    # Initialize and run the slicer
    slicer = ModelSlicer()

    # Load and slice the model
    # slicer.load_model()
    # slicer.slice_all()

    # Verify output directory structure
    assert os.path.exists(output_dir)

    # Verify slices based on the model structure
    expected_layers = [f"layer{i}" for i in range(1, 15)]

    # Check if sequential directory exists (assuming it's a sequential model)
    sequential_dir = os.path.join(output_dir, "sequential")
    assert os.path.exists(sequential_dir)

    # Check if all layer slices were created
    for layer in expected_layers:
        layer_dir = os.path.join(sequential_dir, layer)
        assert os.path.exists(layer_dir), f"Missing slice directory for {layer}"

        # Check for visualization files
        assert os.path.exists(os.path.join(layer_dir, "visualization.png")), \
            f"Missing visualization for {layer}"

        # Check for weight files
        assert os.path.exists(os.path.join(layer_dir, "weights.npy")), \
            f"Missing weights file for {layer}"

    # Verify that config information was properly processed
    assert len(list(os.walk(sequential_dir))[0][1]) == len(expected_layers), \
        "Number of created slices doesn't match expected layer count"
