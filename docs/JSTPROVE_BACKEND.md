# JSTprove Backend Integration

## Overview

DSperse uses [JSTprove](https://github.com/inference-labs-inc/JSTprove) as its ZK proving backend. JSTprove is integrated as a Rust library dependency (`jstprove_circuits` crate) linked at compile time — there is no external CLI or Python process involved.

DSperse is proving-system-agnostic. JSTprove currently provides two proof system backends selectable via the `--proof-system` flag:

| Proof System | Description |
|--------------|-------------|
| `expander` (default) | Expander-based proving system |
| `remainder` | Remainder-based proving system |

## Architecture

The integration lives in `crates/dsperse/src/backend/jstprove.rs`, which wraps the `jstprove_circuits` crate. The `JstproveBackend` struct exposes the following operations that map directly to the pipeline stages:

| Pipeline Stage | JSTprove Function | Description |
|----------------|-------------------|-------------|
| Compile | `compile_bn254` | Compiles an ONNX slice into a BN254 circuit (msgpack bundle) |
| Witness | `witness_bn254` / `witness_bn254_from_f64` | Generates a witness from JSON or raw f64 inputs |
| Prove | `prove_bn254` | Generates a proof from a compiled circuit and witness |
| Verify | `verify_bn254` | Verifies a proof against a circuit and witness |
| Extract | `extract_outputs_bn254` | Extracts model outputs from a witness |

Circuit compilation produces a msgpack bundle containing the circuit, witness solver, and optional metadata (`CircuitParams`). All subsequent operations load this bundle via `read_circuit_msgpack`.

## Proof Pipeline Flow

```text
ONNX slice
    |
    v
compile_bn254 --> compiled circuit bundle (.msgpack)
    |
    v
witness_bn254 --> witness bytes
    |
    v
prove_bn254 --> proof bytes
    |
    v
verify_bn254 --> bool (valid/invalid)
```

## Proof System Selection

The `--proof-system` flag is available on `slice`, `compile`, and `full-run` subcommands. Each proof system defines its own set of supported ONNX operations, queryable via `ProofSystem::supported_ops()`. The `--circuit-ops` flag allows restricting compilation to a subset of supported ops.

```bash
dsperse slice --model-dir models/net --proof-system expander
dsperse compile --model-dir models/net --proof-system remainder
dsperse full-run --model-dir models/net --proof-system expander --circuit-ops "MatMul,Relu"
```

## Dependency

JSTprove is pulled in as a Cargo git dependency via the `jstprove_circuits` crate. No separate installation step is required — it is compiled into the dsperse binary.
