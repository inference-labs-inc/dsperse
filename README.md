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

## Documentation

For more detailed information about the project, please refer to the following documentation:

- [Overview](docs/overview.md): A high-level overview of the project, its goals, and features
- [Architecture](docs/arc42.md): Detailed architecture documentation following the arc42 template

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

# You can also specify an output directory
model_slicer.slice_model(
    output_dir="custom_output_dir",
    input_file="path/to/input.json"  # Sample input for analysis
)
```

### Slicing an ONNX Model

```python
from src.onnx_slicer import OnnxSlicer

# Initialize the ONNX slicer with the path to your ONNX model
onnx_slicer = OnnxSlicer("models/yolov3/model.onnx")

# Slice the model
onnx_slicer.slice_model()
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

Kubz supports generating zero-knowledge proofs for neural network models using the ezkl library. You can run proofs on either whole models or sliced models:

You can also use the CLI interface for various operations:

### Command Line Interface (CLI)

Kubz provides a powerful command-line interface for model slicing, inference, and zero-knowledge proof operations.

#### Basic Usage

```bash
kubz [command] [options]
```

Available commands:
- `slice`: Slice a model into segments
- `run`: Run inference on a model
- `prove`: Generate a proof for a model
- `verify`: Verify a proof for a model
- `circuitize`: Circuitize a model or slices

#### Slicing Models

```bash
# Slice a PyTorch model
kubz slice --model-dir models/net

# Slice a PyTorch model with a specific output directory
kubz slice --model-dir models/net --output-dir custom_slices

# Slice a model with a specific input file
kubz slice --model-dir models/net --input-file custom_input.json
```

#### Running Inference

```bash
# Run inference on a whole model (default)
kubz run --model-dir models/net

# Run inference on a sliced model
kubz run --model-dir models/net --sliced

# Run inference with a specific input file and save results
kubz run --model-dir models/net --input-file input.json --output-file results.json

# Run inference 
kubz run --model-dir models/net

```

#### Generating Proofs

```bash
# Generate a proof for a whole model 
kubz prove --model-dir models/net

# Generate a proof for a sliced model 
kubz prove --model-dir models/net --sliced

```

#### Verifying Proofs

```bash
# Verify a proof for a whole model
kubz verify --model-dir models/net

```

#### Circuitizing Models

```bash
# Circuitize a single ONNX model
kubz circuitize --model-path models/my_model.onnx

# Circuitize sliced model with custom input file
kubz circuitize --model-path models/my_model/slices --input-file models/my_model/input.json
```

#### Getting Help

```bash
# Show general help
kubz --help

# Show help for a specific command
kubz slice --help
kubz circuitize --help
kubz run --help
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
- `cli/`: Command-line interface modules
  - `base.py`: Common CLI utilities and classes
  - `slice.py`: CLI module for slicing models
  - `circuitize.py`: CLI module for circuitizing models
  - `run.py`: CLI module for running inference
  - `prove.py`: CLI module for generating proofs
  - `verify.py`: CLI module for verifying proofs
- `main.py`: Main entry point for the CLI
- `requirements.txt`: Project dependencies

## Advanced Usage


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

See the LICENSE file for details.
