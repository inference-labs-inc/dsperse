# DSperse: Distributed zkML

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/inference-labs-inc/dsperse)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289DA?style=flat-square&logo=discord)](https://discord.gg/GBxBCWJs)
[![Telegram](https://img.shields.io/badge/Telegram-Join%20Channel-0088cc?style=flat-square&logo=telegram)](https://t.me/inference_labs)
[![Twitter](https://img.shields.io/badge/Twitter-Follow%20Us-1DA1F2?style=flat-square&logo=twitter)](https://x.com/inference_labs)
[![Website](https://img.shields.io/badge/Website-Visit%20Us-ff7139?style=flat-square&logo=firefox-browser)](https://inferencelabs.com)
[![Whitepaper](https://img.shields.io/badge/Whitepaper-Read-lightgrey?style=flat-square&logo=read-the-docs)](http://arxiv.org/abs/2508.06972)

DSperse is a proving-system-agnostic intelligent slicer for verifiable AI. It decomposes ONNX neural network models into circuit-compatible segments and orchestrates compilation, inference, proving, and verification across pluggable ZK backends.

## OpenClaw (branch: `openclaw`)

This branch extends DSperse with **ZKProxy** — a zero-knowledge verification protocol for AI agent guardrails. Any ONNX guard model compiles to a ZK circuit via jstprove, and every guardrail decision is proven and verified before any action executes.

### ZKProxy quick start

```bash
# 1. Install (Python 3.13 required)
git clone https://github.com/inference-labs-inc/dsperse && cd dsperse
git checkout openclaw
uv pip install -e .

# 2. Build a guard model from a YAML spec
python3 openclaw_demo/models/guard_builder.py benchmarks/ironclaw_guard.yaml
export ZKPROXY_MODEL_PATH=./ironclaw_guard.onnx
export ZKPROXY_CONFIG_PATH=./ironclaw_guard_config.json

# 3. Start the JSON-RPC worker
python3 benchmarks/zkproxy_worker.py
# → {"jsonrpc": "2.0", "method": "startup", "params": {"status": "ready"}}

# 4. Compile the circuit (send on stdin, once per session)
echo '{"jsonrpc":"2.0","id":1,"method":"compile","params":{"model_path":"ironclaw_guard.onnx"}}' | ...

# 5. Guard check (send on stdin for each request)
echo '{"jsonrpc":"2.0","id":2,"method":"guard_check","params":{"model_path":"ironclaw_guard.onnx","features":[0.3,0.1,0.42,0.0,0.0,0.0,0.5,0.2]}}' | ...
```

The decision rule: `allowed = (score < threshold) AND (verified == true)`. A passing score without a verified proof is a block.

### Key files (openclaw branch)

| Path | Purpose |
|------|---------|
| `benchmarks/zkproxy_worker.py` | Persistent JSON-RPC worker subprocess |
| `openclaw_demo/models/guard_builder.py` | YAML → ONNX + config JSON compiler |
| `openclaw_demo/skills/zkproxy/SKILL.md` | Full protocol reference (RPC methods, YAML spec, error codes, audit log format) |
| `benchmarks/ironclaw_guard.yaml` | 8-feature guard spec (Aho-Corasick patterns from IronClaw) |
| `benchmarks/zeroclaw_guard.yaml` | 8-feature guard spec (regex patterns from ZeroClaw) |
| `openclaw_demo/README.md` | Gateway demo with `/filter`, `/guarded_chat`, `/metrics`, `/events` endpoints |

### Agent integration

ZKProxy is agent-agnostic. The worker is a stdin/stdout JSON-RPC subprocess — any language or framework spawns it and sends requests. Rust integrations exist for [ZKironclaw](https://github.com/shirin-shahabi/ZKironclaw) and [verifiedzeroclaw](https://github.com/shirin-shahabi/verifiedzeroclaw), both behind a `zkproxy` feature flag.

## Features

- **Model Slicing**: Split neural network models into individual layers or custom segments
- **ONNX Support**: Slice and orchestrate ONNX models
- **Layered Inference**: Run inference on sliced models, chaining the output of each segment
- **Zero-Knowledge Proofs**: Generate and verify proofs for model execution via JSTprove
- **Tiling and Channel Splitting**: Automatically decompose large convolutions for circuit-compatible execution
- **Proof System Agnostic**: Pluggable backend architecture supporting Expander and Remainder proof systems

## Documentation

- [Overview](docs/overview.md): High-level overview of the project, its goals, and features
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

DSperse provides six subcommands that form a complete pipeline:

| Command | Description |
|---------|-------------|
| `slice` | Split an ONNX model into segments |
| `compile` | Compile slices into ZK circuits |
| `run` | Execute chained inference across slices (`--weights` to inject consumer ONNX) |
| `prove` | Generate ZK proofs for a completed run |
| `verify` | Verify ZK proofs |
| `full-run` | Execute compile, run, prove, verify in sequence (supports `--weights`) |

### Quickstart

```bash
dsperse slice --model-dir models/net
dsperse compile --model-dir models/net --parallel 4
dsperse run --model-dir models/net --input-file models/net/input.json
dsperse prove --model-dir models/net --run-dir models/net/run/run_*
dsperse verify --model-dir models/net --run-dir models/net/run/run_*
```

Or run the entire pipeline at once:

```bash
dsperse full-run --model-dir models/net --input-file models/net/input.json
```

To inject consumer weights from a fine-tuned ONNX model (same architecture, different weights):

```bash
dsperse run --model-dir models/net --input-file models/net/input.json --weights path/to/consumer.onnx
dsperse full-run --model-dir models/net --input-file models/net/input.json --weights path/to/consumer.onnx
```

## Python Library Usage

```python
import dsperse

metadata_json = dsperse.slice_model("models/net/model.onnx", output_dir="models/net/slices")
dsperse.compile_slices("models/net/slices", parallel=4)
run_json = dsperse.run_inference("models/net/slices", "models/net/input.json", "models/net/run")
proof_json = dsperse.prove_run("models/net/run", "models/net/slices")
verify_json = dsperse.verify_run("models/net/run", "models/net/slices")
```

To inject consumer weights at inference time, pass `weights_onnx` (path to a fine-tuned ONNX with the same architecture):

```python
run_json = dsperse.run_inference(
    "models/net/slices", "models/net/input.json", "models/net/run",
    weights_onnx="path/to/consumer.onnx",
)
```

`slice_model`, `run_inference`, `prove_run`, and `verify_run` return JSON strings parseable with `json.loads()`. `compile_slices` returns `None`.

## Project Structure

```text
crates/dsperse/
  src/
    cli/          CLI argument parsing and command dispatch
    slicer/       ONNX model analysis, slicing, autotiling, channel splitting
    pipeline/     Compilation, inference, proving, verification orchestration
    backend/      JSTprove backend integration
    schema/       Metadata and execution result types (serde)
    converter.rs  Prepares JSTprove artifacts from ONNX files
    utils/        I/O helpers and path resolution
  tests/          Unit and integration tests
python/           Thin Python wrapper for PyO3 bindings
```

## Contributing

Contributions are welcome. Please open issues and PRs on GitHub.

## License

See the [LICENSE](LICENSE) file for details.
