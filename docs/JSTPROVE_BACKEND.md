# JSTprove Backend Integration

## Overview

This document describes the integration of JSTprove as an additional ZK proof backend alongside EZKL in the Dsperse compilation pipeline.

## Features

### 1. JSTprove Backend Support
- New backend class `JSTprove` in `dsperse/src/backends/JSTprove.py`
- Uses JSTprove CLI (`jst` command) for circuit compilation, witness generation, proof generation, and verification
- Compatible with existing EZKL interface for seamless integration

### 2. Flexible Backend Selection
The compiler now supports three modes:

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

## Backend Comparison

| Feature | JSTprove | EZKL |
|---------|----------|------|
| Circuit Format | `.txt` | `.compiled` |
| Keys | Not required | `vk.key`, `pk.key` |
| Settings | Dummy JSON | Full settings.json |
| CLI Command | `jst` | `ezkl` |

## Notes

- JSTprove uses CLI-only interface (no Python package import)
- Fallback logic ensures compilation continues even if preferred backend fails
- Metadata tracks which backend was used for each slice
- All changes maintain backward compatibility with existing EZKL workflows

