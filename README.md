# DSperse: Distributed zkML

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/inference-labs-inc/dsperse)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289DA?style=flat-square&logo=discord)](https://discord.gg/GBxBCWJs)
[![Telegram](https://img.shields.io/badge/Telegram-Join%20Channel-0088cc?style=flat-square&logo=telegram)](https://t.me/inference_labs)
[![Twitter](https://img.shields.io/badge/Twitter-Follow%20Us-1DA1F2?style=flat-square&logo=twitter)](https://x.com/inference_labs)
[![Website](https://img.shields.io/badge/Website-Visit%20Us-ff7139?style=flat-square&logo=firefox-browser)](https://inferencelabs.com)
[![Whitepaper](https://img.shields.io/badge/Whitepaper-Read-lightgrey?style=flat-square&logo=read-the-docs)](http://arxiv.org/abs/2508.06972)

DSperse is a toolkit for slicing, analyzing, and running neural network models. It currently supports ONNX models, allowing you to break down complex models into smaller segments for detailed analysis, optimization, and verification.

## Features

- **Model Slicing**: Split neural network models into individual layers or custom segments
- **ONNX Support**: Slice and orchestrate ONNX models
- **Layered Inference**: Run inference on sliced models, chaining the output of each segment
- **Zero-Knowledge Proofs**: Generate proofs for model execution via multiple backends:
  - **JSTprove**: Fast, efficient ZK proof generation (default, with fallback)
  - **EZKL**: Alternative ZK proof backend
- **Flexible Backend Selection**: Choose backends per-layer or use automatic fallback
- **Visualization**: Analyze model structure and performance

## Documentation

For more detailed information about the project, please refer to the following documentation:

- [Overview](docs/overview.md): A high-level overview of the project, its goals, and features
- [Architecture](docs/arc42.md): Up-to-date architecture summary (arc42-style) reflecting the current CLI and ONNX-only support
- [JSTprove Backend](docs/JSTPROVE_BACKEND.md): Detailed guide on JSTprove integration, backend selection, and usage examples

## CLI shorthands and aliases

Commands (subcommands):
- slice (alias: s)
- compile (alias: c)
- run (alias: r)
- prove (alias: p)
- verify (alias: v)
- full-run (alias: fr)

Common short flags:
- Model path: -m, --mp, --model-dir, --model-path
- Slices path/dir: -s, --sd, --slices, --slices-dir, --slices-directory, --slices-path (compile uses --slices-path; run uses --slices-dir)
- Input file: -i, --if, --input, --input-file
- Output file: -o, --output-file
- Layers: -l, --layers (compile, full-run)
- Run directory: --rd, --run-dir (prove, verify)
- Save analysis (slice only): -S, --save, --save-file

Examples:
```bash
# Slice (short form of command and flags)
dsperse s -m models/net -o models/net/slices -S

# Compile selected layers with calibration input
dsperse c -s models/net/slices -i models/net/input.json -l 0-2

# Run inference over slices with input and output paths
dsperse r -s models/net/slices -i models/net/input.json -o models/net/output.json

# Prove and verify a specific run
dsperse p --rd models/net/run/run_YYYYMMDD_HHMMSS
dsperse v --rd models/net/run/run_YYYYMMDD_HHMMSS

# Full pipeline (alias fr)
dsperse fr -m models/net -i models/net/input.json
```

## Installation

### Install from PyPI

The simplest way to install DSperse is via PyPI:

```bash
# Using pip
pip install dsperse

# Using uv (recommended)
uv tool install dsperse
```

When installed via PyPI, all dependencies are automatically installed when you run a command for the first time, eliminating any manual setup.

### Install from source

Preferred: one-step installer script

- Unix/macOS:
  - Make sure you have Python 3.9+ available (and optionally a virtualenv activated).
  - From the repo root:
```bash
./install.sh
```
  - The script will:
    - Install the DSperse CLI in editable mode so the dsperse command is available
    - Install EZKL (prompting for cargo or pip method if needed)
    - Check EZKL SRS files (~/.ezkl/srs). It will offer to download them interactively (downloads can take a while) because having them locally speeds up circuitization/proving.

Non-interactive/CI-friendly:
```bash
./install.sh -n
```

Manual install

If you prefer to install manually or are on Windows:

1) Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2) Install the DSperse CLI

```bash
pip install -e .
```
This exposes the dsperse command.

3) Install ZK Backend CLIs

**EZKL (optional, for EZKL backend):**
Recommended via cargo (requires Rust):
```bash
cargo install --locked ezkl
# Ensure $HOME/.cargo/bin is on your PATH
```
Alternative via pip:
```bash
pip install -U ezkl
# Note: CLI availability may vary by platform with the pip package. Verify with:
ezkl --version
```

**JSTprove (default backend):**
```bash
uv tool install jstprove
# or: pip install jstprove
```
Also requires Open MPI:
```bash
brew install open-mpi  # macOS
# or apt-get install openmpi-bin libopenmpi-dev  # Linux
```

Verify installation:
```bash
jst --help
```

4) (Optional but recommended) Download EZKL SRS files

SRS files are stored at ~/.ezkl/srs (kzg<N>.srs). They are needed for EZKL circuit setup/proving and downloading them ahead of time speeds things up.

Example manual command to fetch one size:
```bash
ezkl get-srs --logrows 20 --commitment kzg
```
Repeat for other logrows you need (commonly 2..21).

Note: JSTprove does not require SRS files.

Quickstart workflow

Below is an end-to-end walkthrough using the dsperse CLI. You can try it with the example model under models/net.

1) Slice the model
- You can provide either the model.onnx file or the model directory containing it.

Common examples:
```bash
dsperse slice --model-dir models/net
dsperse slice --model-dir models/net/model.onnx
```

Choose output directory (default: models/net/slices):
```bash
dsperse slice --model-dir models/net --output-dir models/net/slices
```

Optionally save analysis metadata to a file (use --save-file; if flag is given without value, default path is model_dir/analysis/model_metadata.json):
```bash
dsperse slice --model-dir models/net --save-file
dsperse slice --model-dir models/net --save-file models/net/analysis/model_metadata.json
```

What happens:
- Slices are written to models/net/slices/slice_<i>/payload/slice_<i>.onnx
- A model-level metadata.json is created at models/net/slices/metadata.json

### Metadata Files Behavior

DSperse creates different types of metadata files for different purposes:

**Operational Metadata** (`metadata.json`):
- **Location**: Always created in the output directory (e.g., `models/net/slices/metadata.json`)
- **Purpose**: Contains segment information, paths, and parameters needed for circuitization and proving
- **Created**: Automatically during slicing
- **Used by**: compile, prove, verify commands

**Analysis Metadata** (`model_metadata.json`):
- **Location**: Created in `model_dir/analysis/` when using `--save-file` flag
- **Purpose**: Detailed node-by-node analysis information for debugging and inspection
- **Created**: Only when `--save-file` flag is used
- **Used by**: Developers and analysts for model inspection

**Important Notes:**
- These are **two different files** serving different purposes
- The operational metadata is essential for dsperse to function
- The analysis metadata is optional and primarily for debugging
- Both files contain similar top-level information but different levels of detail
- **This behavior is intended** but can be confusing due to similar names and purposes

2) Compile with ZK Backends (JSTprove or EZKL)
- Compile either the whole model.onnx or the sliced segments (recommended for incremental proofs):

**Default (Fallback Mode):**
```bash
dsperse compile --slices-path models/net/slices
```
- Tries JSTprove first
- Falls back to EZKL if JSTprove fails
- Falls back to ONNX (skip ZK compilation) if both fail

**Single Backend:**
```bash
dsperse compile --slices-path models/net/slices --backend jstprove
dsperse compile --slices-path models/net/slices --backend ezkl
```

**Per-Layer Backend Assignment:**
```bash
dsperse compile --slices-path models/net/slices --backend "0,2:jstprove;3-4:ezkl"
```
- Layer 0 and 2: Use JSTprove
- Layer 3 and 4: Use EZKL
- Unspecified layers use default backend

Optional calibration input to improve settings:
```bash
dsperse compile --slices-path models/net/slices --input-file models/net/input.json
```

Optional layer selection (sliced models only):
```bash
dsperse compile --slices-path models/net/slices --layers 2,3,4
dsperse compile --slices-path models/net/slices --layers 0-2
```

What happens:
- For each selected segment, the chosen backend compiles the circuit
- JSTprove: Circuit artifacts saved under `models/net/slices/slice_<i>/jstprove/`
- EZKL: Circuit artifacts saved under `models/net/slices/slice_<i>/ezkl/`
- Slices metadata is updated with compilation info per segment

For more details on backend selection and JSTprove integration, see [JSTprove Backend Documentation](docs/JSTPROVE_BACKEND.md).

Note on missing slices:
- If you pass a model directory without slices metadata present, the CLI will prompt you to slice first.

3) Run inference
- Runs a chained execution over the slices using ZK backends where available and falling back to ONNX per-segment on failure.

Common examples:
```bash
dsperse run --slices-dir models/net         # points to model dir (auto-detects slices)
dsperse run --slices-dir models/net/slices  # or point directly to slices
```

You will be prompted for an input file if not provided (default: model_dir/input.json).

To save the final output:
```bash
dsperse run --slices-dir models/net/slices --input-file models/net/input.json --output-file models/net/output.json
```

What happens:
- A run metadata file is auto-generated at models/net/run/metadata.json if missing
- A timestamped run directory is created: models/net/run/run_YYYYMMDD_HHMMSS/
- Segment-by-segment inputs/outputs are saved under that run directory
- A run_results.json is written summarizing the chain execution

4) Generate proofs
- Proves the segments that successfully produced witnesses in the selected run.

Typical usage (new positional-args form):
```bash
dsperse prove models/net/run/run_YYYYMMDD_HHMMSS models/net/slices
# data_path can also be a single slice_* dir, a .dslice, or a .dsperse
# To write proofs under a custom root:
dsperse prove models/net/run/run_YYYYMMDD_HHMMSS models/net/slices --proof-output /tmp/my_proofs
```

Legacy flags are still accepted for backward compatibility (will prompt for selection):
```bash
dsperse prove --run-dir models/net/run
```

Optionally save the updated run results to a separate file:
```bash
dsperse prove models/net/run/run_YYYYMMDD_HHMMSS models/net/slices --output-file models/net/proof_results.json
```

What happens:
- For each segment with a successful witness, the CLI generates proofs
- Proof files are stored under the specific run's segment folder
- The run_results.json is updated with proof_execution details

5) Verify proofs
- Verifies the proofs generated in step 4 against the stored verification keys and settings.

Typical usage (new positional-args form):
```bash
dsperse verify models/net/run/run_YYYYMMDD_HHMMSS models/net/slices
# data_path can also be a single slice_* dir, a .dslice, or a .dsperse
```

Legacy flags are still accepted for backward compatibility (will prompt for selection):
```bash
dsperse verify --run-dir models/net/run
```

Optionally save verification results to a separate file:
```bash
dsperse verify models/net/run/run_YYYYMMDD_HHMMSS models/net/slices --output-file models/net/verification_results.json
```

What happens:
- For each segment with a proof, the CLI verifies the proof
- The run_results.json is updated with verification_execution details
- A summary of verified segments is printed

Tips and troubleshooting

- Backend CLI not found:
  - JSTprove: Ensure `jst` is on your PATH (`uv tool install jstprove` or `pip install jstprove`) and Open MPI is installed
  - EZKL: Ensure `ezkl` is on your PATH. If installed via cargo, add $HOME/.cargo/bin to PATH.
- SRS files missing/slow downloads (EZKL only):
  - You can skip downloads during install and fetch later with `ezkl get-srs --logrows <N> --commitment kzg`
- Compile says "slice first":
  - Run dsperse slice --model-dir <model_dir> to produce slices and metadata.json
- Paths in saved JSON are absolute on your machine; sharing outputs across machines may require path adjustments.

Project structure (updated)

- src/
  - slicer.py: orchestrator for slicing (uses OnnxSlicer)
  - compiler.py: orchestrator for compilation (backend pipeline with fallback)
  - runner.py: chained execution across segments (ZK backends or ONNX fallback)
  - backends/
    - onnx_models.py: ONNX inference utilities
    - jstprove.py: JSTprove CLI bindings and circuitization pipeline
    - ezkl.py: EZKL CLI bindings and circuitization pipeline
  - cli/
    - base.py: shared CLI helpers
    - slice.py: slice command
    - compile.py: compile command
    - run.py: run command
    - prove.py: prove command
    - verify.py: verify command
  - analyzers/: metadata generation for runs/slices
  - utils/: common helpers
- main.py: CLI entry point (dsperse)
- install.sh: installer for CLI, EZKL, and optional SRS

Contributing

Contributions are welcome! Please feel free to open issues and PRs.

License

See the LICENSE file for details.


## End-to-end: full-run

Run the entire pipeline (slice → compile → run → prove → verify) with a single interactive command.

Usage:
```bash
# Kebab-case (preferred)
dsperse full-run --model-dir path/to/model_or_dir --input-file path/to/input.json

# Short alias also works
dsperse fr --model-dir path/to/model_or_dir --input-file path/to/input.json
```

Notes:
- You can pass a model directory that contains model.onnx or a direct path to model.onnx.
- The command is interactive; if an argument is missing, it will prompt you (consistent with other subcommands).
- Slices will be created under <model_dir>/slices unless you provide an existing one.
- Proofs and verification use the latest run under <model_dir>/run by default.

Optional flags:
- --slices-dir: Reuse a pre-existing slices directory to skip the slicing step.
- --layers: Restrict which layers to compile (same format as compile, e.g., "3, 20-22").

Examples:
```bash
# One-shot end-to-end on the sample model
cd src/models/net
# if you have an input.json in this directory
dsperse full-run --model-dir . --input-file ./input.json

# From repo root, specifying paths explicitly
dsperse full-run --model-dir src/models/resnet --input-file src/models/resnet/input.json

# Reuse pre-sliced directory and only compile select layers
dsperse full-run \
  --model-dir src/models/net \
  --slices-dir src/models/net/slices \
  --input-file src/models/net/input.json \
  --layers "1, 3-5"
```


## Slicing outputs and flags

By default, `dsperse slice` produces a single portable bundle named after your slices folder, e.g. `slices.dsperse`. When you pass `--output-dir models/net/slices`, the slicer stages files under `models/net/slices/` and then the converter creates `models/net/slices.dsperse` and cleans up the staging directory.

What the `.dsperse` contains:
- A top-level `metadata.json` describing the model and slices
- All per-slice `.dslice` archives (one per slice)

Choose the output format with `--output-type` (default: `dsperse`):
- `--output-type dsperse` (default): creates `models/net/slices.dsperse` and removes `models/net/slices/`
- `--output-type dslice`: creates `.dslice` files under `models/net/slices/` (and keeps `metadata.json`); removes intermediate `slice_#/` directories
- `--output-type dirs`: keeps raw `slice_#/` directories with `payload/` and per-slice `metadata.json`

Examples:
```bash
# Default: produce only slices.dsperse (staging dir cleaned up)
dsperse slice -m models/net/model.onnx -o models/net/slices

# Produce per-slice .dslice files in a directory (keeps metadata.json)
dsperse slice -m models/net/model.onnx -o models/net/slices --output-type dslice

# Keep unpacked slice_# directories
 dsperse slice -m models/net/model.onnx -o models/net/slices --output-type dirs
```

Notes:
- The `.dsperse` bundle is a ZIP file; you can inspect it with any unzip tool.
- Each `.dslice` inside the bundle is also a ZIP with its own `metadata.json` and `payload/model.onnx`.


## Convert between formats

Use `dsperse slice convert` (or the top-level `dsperse convert`) to go back and forth between single-file bundles and directory layouts. The converter auto-detects the input type; you specify the target with `--to {dirs, dslice, dsperse}`.

Supported conversions:
- slices.dsperse -> directory (contains `*.dslice` + `metadata.json`; add `--expand-slices` to also create `slice_#/` folders)
- directory (with `slice_#/` or `*.dslice` + `metadata.json`) -> slices.dsperse
- slice_X.dslice -> directory (extracts to `slice_X/` with `payload/` and `metadata.json`)
- directory (a single `slice_X/` folder with `payload/` + `metadata.json`) -> slice_X.dslice

Usage examples:
```bash
# 1) Unpack a bundle next to the file (default: keeps .dslice files, does not expand slices)
dsperse slice convert -i models/net/slices.dsperse --to dirs

# 1b) Unpack and also expand each embedded .dslice into slice_# folders
dsperse slice convert -i models/net/slices.dsperse --to dirs --expand-slices

# 1c) Preserve the input bundle instead of deleting it after a successful conversion
 dsperse slice convert -i models/net/slices.dsperse --to dirs --no-cleanup

# 2) Create a .dsperse from a directory
# If the input is a slices/ directory with slice_#/ subfolders, the converter will package and name it `<dirname>.dsperse`.
 dsperse slice convert -i models/net/slices --to dsperse

# 3) Unpack a single .dslice to a directory
 dsperse slice convert -i models/net/slices/slice_1.dslice --to dirs -o models/net/slices/slice_1

# 4) Create a .dslice from a slice directory (must contain metadata.json and payload/)
 dsperse slice convert -i models/net/slices/slice_1 --to dslice -o models/net/slices/slice_1.dslice
```

Notes:
- If you omit --output, sensible defaults are chosen (e.g., extracting next to the input file with the same stem, or naming `<dirname>.dsperse`).
- Use `--cleanup/--no-cleanup` to control whether the source artifact is deleted after a successful conversion (default: cleanup enabled for `.dsperse` sources).
- The converter is the single source of truth for formats; CLI commands delegate to it to ensure consistent behavior and cleanup.
