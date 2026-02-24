# DSperse: Distributed zkML

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/inference-labs-inc/dsperse)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289DA?style=flat-square&logo=discord)](https://discord.gg/GBxBCWJs)
[![Telegram](https://img.shields.io/badge/Telegram-Join%20Channel-0088cc?style=flat-square&logo=telegram)](https://t.me/inference_labs)
[![Twitter](https://img.shields.io/badge/Twitter-Follow%20Us-1DA1F2?style=flat-square&logo=twitter)](https://x.com/inference_labs)
[![Website](https://img.shields.io/badge/Website-Visit%20Us-ff7139?style=flat-square&logo=firefox-browser)](https://inferencelabs.com)
[![Whitepaper](https://img.shields.io/badge/Whitepaper-Read-lightgrey?style=flat-square&logo=read-the-docs)](http://arxiv.org/abs/2508.06972)

DSperse is a Rust-native toolkit for slicing, analyzing, and running neural network models with zero-knowledge proof generation. It supports ONNX models, allowing you to break down complex models into smaller segments for distributed analysis, optimization, and verification.

## Features

- **Model Slicing**: Split neural network models into individual layers or custom segments
- **ONNX Support**: Slice and orchestrate ONNX models
- **Layered Inference**: Run inference on sliced models, chaining the output of each segment
- **Zero-Knowledge Proofs**: Generate and verify proofs for model execution via JSTprove
- **Tiling and Channel Splitting**: Automatically decompose large convolutions for circuit-compatible execution
- **Archive Formats**: Package slices as `.dslice` or `.dsperse` bundles for portable distribution

## Documentation

- [Overview](docs/overview.md): High-level overview of the project, its goals, and features
- [Architecture](docs/arc42.md): Architecture summary (arc42-style)
- [JSTprove Backend](docs/JSTPROVE_BACKEND.md): JSTprove integration and usage

## Installation

### From PyPI (includes CLI)

```bash
pip install dsperse
```

This installs both the `dsperse` CLI command and the Python library bindings. No additional dependencies required — everything is compiled into a single native extension.

### From source (Rust binary)

```bash
cargo install --path crates/dsperse
```

### As a Rust library

```toml
[dependencies]
dsperse = { git = "https://github.com/inference-labs-inc/dsperse.git" }
```

## CLI Usage

DSperse provides seven subcommands that form a complete pipeline:

| Command | Description |
|---------|-------------|
| `slice` | Split an ONNX model into segments |
| `compile` | Compile slices into ZK circuits |
| `run` | Execute chained inference across slices |
| `prove` | Generate ZK proofs for a completed run |
| `verify` | Verify ZK proofs |
| `full-run` | Execute the entire pipeline (slice, compile, run, prove, verify) |
| `convert` | Convert between archive formats (dirs, dslice, dsperse) |

### Quickstart

```bash
dsperse slice --model-dir models/net --format dirs
dsperse compile --model-dir models/net --parallel 4
dsperse run --model-dir models/net --input-file models/net/input.json
dsperse prove --model-dir models/net --run-dir models/net/run/run_*
dsperse verify --model-dir models/net --run-dir models/net/run/run_*
```

Or run the entire pipeline at once:

```bash
dsperse full-run --model-dir models/net --input-file models/net/input.json
```

### Output Formats

The `--format` flag on `slice` controls the output layout:

| Format | Description |
|--------|-------------|
| `dirs` (default) | Raw `slice_N/` directories with `payload/` and per-slice `metadata.json` |
| `dslice` | Per-slice `.dslice` ZIP archives |
| `dsperse` | Single `.dsperse` bundle containing all slices |

Use `dsperse convert` to transform between formats after slicing.

## Python Library Usage

```python
import dsperse

metadata_json = dsperse.slice_model("models/net/model.onnx", output_dir="models/net/slices")
dsperse.compile_slices("models/net/slices", parallel=4)
run_json = dsperse.run_inference("models/net/slices", "models/net/input.json", "models/net/run")
proof_json = dsperse.prove_run("models/net/run", "models/net/slices")
verify_json = dsperse.verify_run("models/net/run", "models/net/slices")
```

All functions return JSON strings that can be parsed with `json.loads()`.

## Project Structure

```
crates/dsperse/
  src/
    cli/          CLI argument parsing and command dispatch
    slicer/       ONNX model analysis, slicing, autotiling, channel splitting
    pipeline/     Compilation, inference, proving, verification orchestration
    backend/      JSTprove backend integration
    schema/       Metadata and execution result types (serde)
    archive/      Format conversion (dirs, dslice, dsperse)
    utils/        I/O helpers and path resolution
  tests/          Unit and integration tests
python/           Thin Python wrapper for PyO3 bindings
```

## Contributing

Contributions are welcome. Please open issues and PRs on GitHub.

## License

See the [LICENSE](LICENSE) file for details.
