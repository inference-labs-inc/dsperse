# JSTprove Backend Integration

## Overview

This document describes the integration of JSTprove as an additional ZK proof backend alongside EZKL in the Dsperse compilation pipeline.

## Features

### 1. JSTprove Backend Support
- New backend class `JSTprove` in `dsperse/src/backends/JSTprove.py`
- Uses JSTprove CLI (`jst` command) for circuit compilation, witness generation, proof generation, and verification
- Compatible with existing EZKL interface for seamless integration

### 2. Flexible Backend Selection (compile-time)
The compiler supports three modes:

**Default (Fallback Mode):**
```bash
dsperse compile --path model/slices
```
- Tries JSTprove first
- Falls back to EZKL if JSTprove fails
- Falls back to ONNX (skip ZK compilation) if both fail

**Single Backend:**
```bash
dsperse compile --path model/slices --backend jstprove
dsperse compile --path model/slices --backend ezkl
```

**Per-Layer Backend Assignment:**
```bash
dsperse compile --path model/slices --backend "0,2:jstprove;3-4:ezkl"
```
- Layer 0 and 2: Use JSTprove
- Layer 3 and 4: Use EZKL
- Unspecified layers use default backend

You can also mix default groups (compile both backends) using a bare group without a backend:

```bash
# slice 0 -> default (both backends)
# slice 2 -> jstprove only
# slices 3-4 -> ezkl only
dsperse compile --path model/slices --backend "0;2:jstprove;3-4:ezkl"
```

Notes:
- Bare groups like `"0"` or `"0,5-6"` mean “default behavior (compile both backends)”.
- Unspecified slices are skipped at compile time and will run with ONNX at runtime.

### 3. Runtime Backend Selection (Runner)

When a slice has multiple circuit backends compiled, you can choose which backend to use at runtime:

```bash
dsperse run -p model/slices -i model/input.json -b jstprove   # or -b ezkl | -b onnx
```

Python API:
```text
from dsperse.src.run.runner import Runner
Runner().run(input_json_path="model/input.json", slice_path="model/slices", backend="ezkl")
```

Behavior rules:
- If a slice has both JSTprove and EZKL compiled, the selected backend is used for that slice.
- If a slice has only one circuit backend compiled, the flag is ignored for that slice (unless `onnx` is specified to skip circuits).
- If the selected backend fails (and multiple are available), the runner falls back to the other compiled backend, then to ONNX.

## Installation

1. Install Open MPI (required for JSTprove):
   ```bash
   brew install open-mpi  # macOS
   # or apt-get install openmpi-bin libopenmpi-dev  # Linux
   ```

2. Install JSTprove:
   ```bash
   uv tool install jstprove
   # or: pip install jstprove
   ```

3. Verify installation:
   ```bash
   jst --help
   ```

The `install.sh` script has been updated to automatically install these dependencies.

## File Changes

### New Files
- `dsperse/src/backends/JSTprove.py` - JSTprove backend implementation

### Modified Files
- `dsperse/src/cli/compile.py` - Added `--backend` argument
- `dsperse/src/compile/compiler.py` - Backend selection and fallback logic
- `dsperse/src/compile/utils/compiler_utils.py` - Support for JSTprove compilation success check
- `dsperse/src/constants.py` - Added JSTprove command constant
- `install.sh` - Added Open MPI and JSTprove installation
- `requirements.txt` - Added mpi4py dependency

## Usage Examples

**Compile all layers with default fallback:**
```bash
dsperse compile --path model/slices
```

**Compile specific layers with mixed backends:**
```bash
dsperse compile --path model/slices --layers "0-4" --backend "0,2:jstprove;3-4:ezkl"
```

**Compile with single backend:**
```bash
dsperse compile --path model/slices --backend jstprove
```

**Run with a chosen backend when multiple are available:**
```bash
dsperse run --path model/slices --input-file model/input.json -b ezkl
```

## Backend Comparison

| Feature | JSTprove | EZKL |
|---------|----------|------|
| Circuit Format | `.txt` | `.compiled` |
| Keys | Not required | `vk.key`, `pk.key` |
| Settings | Dummy JSON | Full settings.json |
| CLI Command | `jst` | `ezkl` |

## Notes

- JSTprove uses a CLI interface (no Python package import)
- Fallback logic ensures compilation continues even if preferred backend fails
- Metadata tracks which backend was used for each slice and which backend produced the witness at runtime
- Proving and verifying use the witness backend automatically (JSTprove does not require pk/vk; EZKL requires `pk.key` for proving and `vk.key` for verifying, plus `settings.json` when available)
- All changes maintain backward compatibility with existing EZKL workflows

