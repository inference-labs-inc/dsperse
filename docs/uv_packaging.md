# Developer Guide

This document provides a guide for developers who contribute to the project.

## Build System

The project uses [maturin](https://www.maturin.rs/) as its build backend. The native Rust extension is compiled via PyO3 and exposed as `dsperse._native`. There are no Python-level dependencies beyond the compiled extension itself.

The build configuration in `pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["python"]
module-name = "dsperse._native"
python-source = "python"
manifest-path = "crates/dsperse/Cargo.toml"
```

## Local Development

Create a virtual environment and build the extension in development mode:

```sh
uv venv
source .venv/bin/activate
maturin develop --features python
```

This compiles the Rust crate and installs the resulting native extension into the active virtualenv. Re-run `maturin develop` after any Rust code changes.

## Building a Wheel

```sh
maturin build --release --features python
```

The output wheel is self-contained with no additional Python dependencies.
