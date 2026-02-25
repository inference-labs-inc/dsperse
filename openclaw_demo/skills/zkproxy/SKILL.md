---
name: zkproxy
version: 0.2.0
description: >
  Zero-knowledge guardrail verification protocol built on DSperse and jstprove.
  Compiles any ONNX guard model into a ZK circuit, then proves and verifies every
  guardrail decision before any action executes. Verify first. Act second.
activation:
  keywords:
    - zkproxy
    - zk proof
    - guard verification
    - proof of inference
    - prove guardrail
    - dsperse guard
    - jstprove
    - zk circuit
    - verified inference
    - guard check
  patterns:
    - "(?i)verify.*before.*act"
    - "(?i)prove.*guard"
    - "(?i)zk.*guardrail"
    - "(?i)dsperse.*proof"
  max_context_tokens: 4000
metadata:
  openclaw:
    requires:
      bins:
        - python3
      env:
        - ZKPROXY_MODEL_PATH
---

# ZKProxy Verification Protocol

Every action MUST pass `guard_check` and receive `verified: true` before it executes.
No proof. No action.

## What you need

Two things. That is all.

**1. DSperse (openclaw branch)** — the entire stack in one install: DSperse pipeline, jstprove 2.4.0, onnxruntime, torch, numpy.

```bash
# Python 3.13 exactly — hard requirement in dsperse pyproject.toml
python3 --version   # must be 3.13.x

git clone https://github.com/inference-labs-inc/dsperse
cd dsperse && git checkout openclaw
uv pip install -e .
```

`jstprove` is not a separate install. It is `jstprove==2.4.0` inside dsperse's dependency tree. `uv pip install -e .` is the only install step for the entire proving stack.

**2. zkproxy_worker.py** — the persistent JSON-RPC subprocess. It imports directly from `dsperse.src.backends.jstprove`. No dsperse = worker will not start.

```bash
# ships in this repo at benchmarks/zkproxy_worker.py
export ZKPROXY_WORKER=benchmarks/zkproxy_worker.py
```

Nothing else. No agent framework. No SDK.

## Build the guard model

The worker needs a compiled ONNX model and a feature config JSON. Build both from a YAML spec:

```bash
python3 openclaw_demo/models/guard_builder.py your_guard.yaml
# outputs: your_guard.onnx  +  your_guard_config.json
```

**Minimal guard YAML:**

```yaml
name: my_guard
threshold: 0.5

features:
  - name: instruction_override
    type: regex_count
    index: 0
    patterns:
      - "(?i)ignore\\s+(previous|all)"
      - "(?i)forget\\s+everything"

  - name: system_injection
    type: string_match
    index: 1
    strings:
      - "system:"
      - "[INST]"

  - name: normalized_length
    type: builtin
    index: 2

model:
  layers:
    - op: Gemm
      out_dim: 16
    - op: Relu
    - op: Gemm
      out_dim: 1
```

**Feature types:**

| Type | Config key | Normalization |
|------|-----------|---------------|
| `regex_count` | `patterns: []` | `min(match_count, 10) / 10.0` |
| `string_match` | `strings: []` | `min(match_count, 10) / 10.0` |
| `builtin` | name only | computed, already 0.0–1.0 |

**Available builtin names:** `normalized_length`, `digit_ratio`, `whitespace_ratio`, `uppercase_ratio`, `special_char_ratio`, `avg_word_length`, `line_count_norm`, `entropy`

**Guard builder supported ops** (what `guard_builder.py` can actually emit):
`Gemm`, `Relu`, `Add`, `Mul`, `Sub`, `Clip`

jstprove accepts a broader set (`Add`, `BatchNormalization`, `Clip`, `Constant`, `Conv`, `Div`, `Flatten`, `Gemm`, `Max`, `MaxPool`, `Min`, `Mul`, `Relu`, `Reshape`, `Sub`) and `validate_guard_spec` checks against that superset, but the builder only implements the six ops above. Hand-crafted ONNX models can use the full jstprove set.

Set env vars to point at the build outputs:

```bash
export ZKPROXY_MODEL_PATH=./my_guard.onnx
export ZKPROXY_CONFIG_PATH=./my_guard_config.json
```

## Start and compile

```bash
python3 benchmarks/zkproxy_worker.py
```

The worker emits exactly one startup line then blocks waiting for JSON-RPC on stdin:

```json
{"jsonrpc": "2.0", "method": "startup", "params": {"status": "ready"}}
```

If you do not see `"status": "ready"`, dsperse is not installed correctly.

**Compile the circuit once on startup** — required before `guard_check` produces real proofs. Without it, the worker returns the score but skips prove/verify and sets `"verified": false`:

```json
{"jsonrpc": "2.0", "id": 1, "method": "compile", "params": {"model_path": "my_guard.onnx"}}
```

Response:
```json
{"result": {"success": true, "circuit_key": "a1b2c3d4", "circuit_path": "/tmp/zkproxy_.../my_guard_circuit.txt", "timing_ms": 1840.2}}
```

The circuit is cached in memory. Call `compile` once per worker session, not per request.

## Guard check protocol

For every user message, extract the feature vector according to the loaded config JSON and call `guard_check`:

**Request:**
```json
{"jsonrpc": "2.0", "id": 1, "method": "guard_check", "params": {
  "model_path": "my_guard.onnx",
  "features": [0.3, 0.1, 0.42]
}}
```

**Response — verified:**
```json
{"result": {
  "success": true,
  "score": 0.18,
  "proof_hash": "e3b0c44298fc1c149afbf4c8996fb924...",
  "verified": true,
  "timings": {"witness_ms": 11.2, "prove_ms": 94.5, "verify_ms": 9.1},
  "timing_ms": 115.3
}}
```

**Response — circuit not compiled:**
```json
{"result": {
  "success": true,
  "score": 0.18,
  "proof_hash": "",
  "verified": false,
  "note": "no compiled circuit available, skipping prove/verify",
  "timing_ms": 12.1
}}
```

**Decision rule:** `allowed = (score < threshold) AND (verified == true)`

A score below the threshold without `verified: true` is a block. The proof is the authorization, not the score.

## Other RPC methods

| Method | Required params | Purpose |
|--------|----------------|---------|
| `health` | — | jstprove version, cached circuit keys, supported ops list |
| `compile` | `model_path` | ONNX → jstprove circuit, cached for session |
| `witness` | `model_path`, `features[]` | Generate witness data only |
| `prove` | `witness_path`, `circuit_path` | Generate ZK proof, returns `proof_hash` |
| `verify` | `proof_path`, `circuit_path`, `input_path`, `output_path`, `witness_path` | Verify an existing proof independently |

`guard_check` is `witness → prove → verify` in one call. Use the individual methods only if you need the intermediate artifacts for external audit.

## Errors

```json
{"error": {"code": -32000, "message": "...", "data": "<full traceback>"}}
```

| Code | Cause |
|------|-------|
| `-32700` | JSON parse error on stdin |
| `-32601` | Unknown method name |
| `-32000` | Runtime exception — traceback in `data` |

## Audit log

Append one JSON line per decision. Never truncate or delete this file.

```json
{
  "timestamp": "2026-02-24T12:00:00Z",
  "content_hash": "<sha256 of raw input>",
  "features": [0.3, 0.1, 0.42],
  "score": 0.18,
  "allowed": true,
  "proof_hash": "e3b0c44298fc...",
  "proof_verified": true,
  "timings": {"witness_ms": 11.2, "prove_ms": 94.5, "verify_ms": 9.1}
}
```

The `proof_hash` is SHA-256 of the raw proof binary. Any party with the proof and circuit artifacts can re-run `verify` to confirm the decision without access to model weights.

## File map

| File | Purpose |
|------|---------|
| `benchmarks/zkproxy_worker.py` | Persistent JSON-RPC worker |
| `openclaw_demo/models/guard_builder.py` | YAML → ONNX + config JSON compiler |
| `benchmarks/ironclaw_guard.yaml` | 8-feature guard spec (Aho-Corasick pattern set from IronClaw) |
| `benchmarks/zeroclaw_guard.yaml` | 8-feature guard spec (regex pattern set from ZeroClaw) |
| `dsperse/src/backends/jstprove.py` | DSperse JSTprove backend |

All paths relative to the dsperse openclaw branch root.
