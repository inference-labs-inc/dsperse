# Kubz: Distributed zkML

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/inferencelabs/kubz)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289DA?style=flat-square&logo=discord)](https://discord.gg/inferencelabs)
[![Telegram](https://img.shields.io/badge/Telegram-Join%20Channel-0088cc?style=flat-square&logo=telegram)](https://t.me/inferencelabs)
[![Twitter](https://img.shields.io/badge/Twitter-Follow%20Us-1DA1F2?style=flat-square&logo=twitter)](https://twitter.com/inferencelabs)
[![Website](https://img.shields.io/badge/Website-Visit%20Us-ff7139?style=flat-square&logo=firefox-browser)](https://inferencelabs.io)
[![Whitepaper](https://img.shields.io/badge/Whitepaper-Read-lightgrey?style=flat-square&logo=read-the-docs)](https://inferencelabs.io/whitepaper)

Kubz is a toolkit for slicing, analyzing, and running neural network models. It supports both PyTorch models and ONNX models, allowing you to break down complex models into smaller segments for detailed analysis, optimization, and verification.

## Features

- **Model Slicing**: Split neural network models into individual layers or custom segments
- **ONNX Support**: Convert and slice ONNX models
- **PyTorch Support**: Work directly with PyTorch models
- **Layered Inference**: Run inference on sliced models, chaining the output of each segment
- **Zero-Knowledge Proofs**: Generate proofs for model execution (via ezkl integration)
- **Visualization**: Analyze model structure and performance

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/inference-labs-inc/kubz.git
   cd kubz
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the CLI:
   ```bash
   pip install -e .
   ```

   This installs the `kubz` command-line tool, which you can use from anywhere on your system.

## Usage Examples

### Slicing a PyTorch Model

```python
from src.model_slicer import ModelSlicer

# Initialize the model slicer with the directory containing your model
model_slicer = ModelSlicer(model_directory="models/net")

# Slice the model (creates slices in the model directory by default)
model_slicer.slice_model()

# You can also specify an output directory and slicing strategy
model_slicer.slice_model(
    output_dir="custom_output_dir",
    strategy="single_layer",  # Slice by individual layers
    input_file="path/to/input.json"  # Sample input for analysis
)
```

### Slicing an ONNX Model

```python
from src.onnx_slicer import OnnxSlicer

# Initialize the ONNX slicer with the path to your ONNX model
onnx_slicer = OnnxSlicer("models/yolov3/model.onnx")

# Slice the model by individual layers
onnx_slicer.slice_model(mode="single_layer")
```

### Running Inference on a Sliced PyTorch Model

```python
from src.runners.model_runner import ModelRunner

# Initialize the model runner with the model directory
model_runner = ModelRunner(model_directory="models/net")

# Run inference on the full model
result = model_runner.infer()
print(result)

# Run inference on the sliced model
result = model_runner.infer(mode="sliced")
print(result)
```

### Running Inference on a Sliced ONNX Model

```python
from src.runners.onnx_runner import OnnxRunner

# Initialize the ONNX runner with the model directory
onnx_runner = OnnxRunner(model_directory="models/yolov3")

# Run inference on the full model
result = onnx_runner.infer()
print(result)

# Run inference on the sliced model
result = onnx_runner.infer(mode="sliced")
print(result)
```

### Zero-Knowledge Proofs for Models

Kubz supports generating zero-knowledge proofs for neural network models using both ezkl and jstProve libraries. You can run proofs on either whole models or sliced models:

#### Using ezkl

```python
from src.runners.ezkl_runner import EzklRunner

# Initialize the ezkl runner with the model directory
ezkl_runner = EzklRunner(model_directory="models/net")

# Generate witness for the whole model
result = ezkl_runner.generate_witness()
print(result)

# Generate witness for the sliced model
result = ezkl_runner.generate_witness(mode="sliced")
print(result)

# Generate proof for the whole model
result = ezkl_runner.prove()
print(result)

# Generate proof for the sliced model
result = ezkl_runner.prove(mode="sliced")
print(result)

# Verify proof for the whole model
result = ezkl_runner.verify()
print(result)

# Verify proof for the sliced model
result = ezkl_runner.verify(mode="sliced")
print(result)
```

#### Using jstProve

```python
from src.runners.jstprove_runner import JSTProveRunner

# Initialize the jstProve runner with the model directory
jstprove_runner = JSTProveRunner(model_directory="models/net")

# Circuitize the whole model
result = jstprove_runner.circuitize()
print(result)

# Circuitize the sliced model
result = jstprove_runner.circuitize(mode="sliced")
print(result)

# Generate witness for the whole model
result = jstprove_runner.generate_witness()
print(result)

# Generate witness for the sliced model
result = jstprove_runner.generate_witness(mode="sliced")
print(result)

# Generate proof for the whole model
result = jstprove_runner.prove()
print(result)

# Generate proof for the sliced model
result = jstprove_runner.prove(mode="sliced")
print(result)

# Verify proof for the whole model
result = jstprove_runner.verify()
print(result)

# Verify proof for the sliced model
result = jstprove_runner.verify(mode="sliced")
print(result)
```

You can also use the CLI interface for various operations:

### Command Line Interface (CLI)

Kubz provides a powerful command-line interface for model slicing, inference, and zero-knowledge proof operations.

#### Basic Usage

```bash
kubz [command] [options]
```

Available commands:
- `slice`: Slice a model into segments
- `infer`: Run inference on a model
- `prove`: Generate a proof for a model
- `verify`: Verify a proof for a model

#### Slicing Models

```bash
# Slice a PyTorch model using the default single_layer strategy
kubz slice --model-dir models/net

# Slice a PyTorch model with a specific output directory and strategy
kubz slice --model-dir models/net --output-dir custom_slices --strategy by_type

# Slice a model with a specific input file
kubz slice --model-dir models/net --input-file custom_input.json
```

#### Running Inference

```bash
# Run inference on a whole model (default)
kubz infer --model-dir models/net

# Run inference on a sliced model
kubz infer --model-dir models/net --sliced

# Run inference with a specific input file and save results
kubz infer --model-dir models/net --input-file input.json --output-file results.json

# Run inference using the EZKL backend
kubz infer --model-dir models/net --ezkl

# Run inference using the jstProve backend
kubz infer --model-dir models/net --jstprove
```

#### Generating Proofs

```bash
# Generate a proof for a whole model using EZKL
kubz prove --model-dir models/net --ezkl

# Generate a proof for a sliced model using EZKL
kubz prove --model-dir models/net --ezkl --sliced

# Generate a proof using jstProve and save results
kubz prove --model-dir models/net --jstprove --output-file proof_results.json
```

#### Verifying Proofs

```bash
# Verify a proof for a whole model using EZKL
kubz verify --model-dir models/net --ezkl

# Verify a proof for a sliced model using jstProve
kubz verify --model-dir models/net --jstprove --sliced

# Verify a proof with a specific input file
kubz verify --model-dir models/net --jstprove --input-file input.json
```

#### Getting Help

```bash
# Show general help
kubz --help

# Show help for a specific command
kubz slice --help
kubz infer --help
kubz prove --help
kubz verify --help
```

#### Easter Eggs

The CLI includes some fun easter eggs! Try running:

```bash
kubz --easter-egg
```

Or just run any command - you might get lucky and see an easter egg 20% of the time!

## Project Structure

- `src/`: Main source code
  - `model_slicer.py`: Slices PyTorch models
  - `onnx_slicer.py`: Slices ONNX models
  - `runners/`: Code for running inference on models
    - `model_runner.py`: Runs inference on PyTorch models
    - `onnx_runner.py`: Runs inference on ONNX models
  - `utils/`: Utility functions
  - `models/`: Example models
- `main.py`: CLI interface for running workflows
- `requirements.txt`: Project dependencies

## Advanced Usage

### Custom Slicing Strategies

You can implement custom slicing strategies by modifying the `get_slice_points` method in `ModelUtils`. The default strategies include:

- `single_layer`: Slice at every layer
- `by_type`: Group layers by type (e.g., all convolutional layers together)
- `by_size`: Group layers to achieve a target size per segment

### Working with Model Metadata

After slicing a model, metadata is saved in a JSON file that contains information about each segment, including:

- Layer types
- Input/output shapes
- Dependencies between segments
- Paths to saved segment files

You can use this metadata to understand the model structure or to implement custom inference pipelines.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
