# ZK Proxy Benchmark Report: IronClaw vs ZeroClaw

**Date**: 2026-02-24
**Platform**: macOS Darwin 23.5.0, Apple Silicon (aarch64)
**Rust**: 1.92.0 (IronClaw), 1.90.0 (ZeroClaw)
**Profile**: `bench` (optimized, release)

## Executive Summary

This report compares the prompt injection defense systems of two Rust agent runtimes — **ZKironclaw (IronClaw)** and **verifiedzeroclaw (ZeroClaw)** — across latency and detection accuracy. Both projects now share an identical `zkproxy` module that bridges to DSperse/jstprove zero-knowledge proof circuits for cryptographically attested guard decisions.

The Rust-native pattern matching (regex/Aho-Corasick) and ZK feature extraction are benchmarked independently. The ZK proof pipeline (witness, prove, verify) requires a running Python subprocess and is characterized architecturally but not included in the latency benchmarks, as it depends on model compilation and jstprove availability.

**Key findings:**

- ZeroClaw's regex guard is **2.9x faster** than IronClaw's full safety layer on injection inputs
- ZeroClaw achieves **higher recall** (52.1% vs 41.1%) with slightly better precision (82.6% vs 78.9%)
- IronClaw excels at special token and code injection detection; ZeroClaw covers more categories
- Feature extraction performance is comparable (~2.3 us for injection inputs)
- Both projects add ~2-4 us overhead for the ZK feature extraction step, with proof generation (Python/jstprove) expected to add 50-500 ms depending on circuit complexity

---

## Latency Benchmarks (Criterion)

All values from criterion 0.5 with 100 sample iterations at the `bench` optimization level.

### Feature Extraction (ZK input preparation)

Converts raw text into a fixed-length float vector via regex matching and builtin feature computation. This is the Rust-native portion of the ZK pipeline — it runs before any Python subprocess call.

| Benchmark | IronClaw | ZeroClaw | Ratio (IC/ZC) |
|-----------|----------|----------|---------------|
| Benign text | 2,309 ns (±458) | 1,619 ns (±46) | 1.43x slower |
| Injection text | 2,573 ns (±266) | 2,293 ns (±67) | 1.12x slower |

IronClaw's feature extractor shows higher variance (±458 ns vs ±46 ns on benign), likely due to its broader regex set (instruction override, role manipulation, system injection, special tokens, code injection, encoded payloads). ZeroClaw's extractor uses different categories (system override, role confusion, secret extraction, jailbreak, tool injection, command injection) with tighter performance.

### Pattern Guard (regex-only, no ZK)

Each project's existing prompt injection defense, measured independently.

| Benchmark | IronClaw | ZeroClaw | Ratio |
|-----------|----------|----------|-------|
| Benign text | 227 ns (±17) | 1,396 ns (±56) | **6.2x faster** |
| Injection text | 2,725 ns (±583) | 1,443 ns (±59) | 1.9x slower |

IronClaw's `Sanitizer` uses **Aho-Corasick** for fast multi-pattern string matching, giving it a dramatic advantage on benign text (early rejection at 227 ns). On injection text it slows down because it must process matches, escape content, and track warnings with severity levels.

ZeroClaw's `PromptGuard` uses **pure regex** across 6 categories with normalized scoring. Its performance is more consistent between benign and injection inputs (1,396 vs 1,443 ns) because it always evaluates all category checkers.

### Full Safety Pipeline

| Benchmark | IronClaw | ZeroClaw | Notes |
|-----------|----------|----------|-------|
| SafetyLayer (full) | 4,135 ns (±253) | — | Sanitizer + Validator + LeakDetector + Policy |
| PromptGuard (blocking mode) | — | 1,785 ns (±95) | Block action, 0.1 sensitivity |

IronClaw's `SafetyLayer` runs four subsystems in series (Sanitizer, Validator, PolicyRules, LeakDetector), totaling ~4.1 us. ZeroClaw's blocking-mode `PromptGuard` at max sensitivity adds only ~340 ns over the default warn mode.

### Summary Table

| Phase | IronClaw | ZeroClaw |
|-------|----------|----------|
| Feature extraction (injection) | 2.57 us | 2.29 us |
| Pattern guard (injection) | 2.72 us | 1.44 us |
| Full safety pipeline | 4.13 us | 1.79 us |
| **Combined (guard + ZK features)** | **~6.7 us** | **~3.7 us** |

The "combined" estimate is the sum of pattern guard + feature extraction, representing total Rust-side latency before the Python subprocess call. The ZK proof pipeline (witness + prove + verify) adds additional latency handled by the persistent `zkproxy_worker.py`.

---

## Detection Accuracy

Evaluated on a 143-prompt dataset (70 benign, 73 injection) spanning 9 injection categories.

### Overall Metrics

| Metric | IronClaw | ZeroClaw |
|--------|----------|----------|
| True Positives | 30 | 38 |
| False Positives | 8 | 8 |
| True Negatives | 62 | 62 |
| False Negatives | 43 | 35 |
| **Precision** | **78.9%** | **82.6%** |
| **Recall** | **41.1%** | **52.1%** |
| **F1 Score** | **0.541** | **0.639** |

ZeroClaw achieves a higher F1 score (0.639 vs 0.541) driven primarily by better recall (+11 percentage points) at equal or better precision. Both projects share the same false positive rate (8/70 = 11.4%).

### Per-Category Recall

| Category | Count | IronClaw | ZeroClaw | Winner |
|----------|-------|----------|----------|--------|
| system_override | 15 | **80.0%** | **80.0%** | Tie |
| jailbreak | 12 | 0.0% | **33.3%** | ZeroClaw |
| role_confusion | 11 | 45.5% | **81.8%** | ZeroClaw |
| secret_extraction | 10 | 0.0% | **10.0%** | ZeroClaw |
| command_injection | 7 | 0.0% | **42.9%** | ZeroClaw |
| tool_injection | 5 | 0.0% | **40.0%** | ZeroClaw |
| special_tokens | 5 | **100.0%** | 60.0% | IronClaw |
| multi_vector | 5 | **100.0%** | 80.0% | IronClaw |
| code_injection | 4 | **100.0%** | 0.0% | IronClaw |

**IronClaw strengths**: Special token detection (`<|endoftext|>`, `[INST]`), code injection (````system`, `sudo`), multi-vector attacks (these typically contain system injection markers that Aho-Corasick catches).

**ZeroClaw strengths**: Broader category coverage. Its dedicated regex categories for jailbreak (`DAN mode`, `developer mode`), role confusion (`you are now`, `your new role`), tool injection (`tool_calls`, `function_call`), command injection (`$(`, `&&`, `||`), and secret extraction (`show all secrets`) give it recall across more attack types.

**Gap analysis**: Both projects miss a significant fraction of injections. The ZK-attested ML model (ONNX guard) is designed to close this gap by learning non-linear combinations of feature signals that individual regex patterns miss.

---

## Architecture Comparison

| Dimension | IronClaw | ZeroClaw |
|-----------|----------|----------|
| Pattern engine | Aho-Corasick + Regex | Regex only |
| Pattern count | 17 string + 4 regex | 26 regex across 6 categories |
| Severity model | 4 levels (Critical/High/Medium/Low) | Normalized score 0.0-1.0 |
| Default threshold | Block on Critical/High | Warn at sensitivity 0.7 |
| Safety subsystems | Sanitizer + Validator + Policy + LeakDetector | PromptGuard (single pass) |
| ZK feature dimensions | 8 (instruction_override, role_manipulation, system_injection, special_tokens, code_injection, encoded_payload, normalized_length, whitespace_ratio) | 8 (system_override, role_confusion, secret_extraction, jailbreak, tool_injection, command_injection, digit_ratio, entropy) |
| ZK proof backend | DSperse/jstprove (shared) | DSperse/jstprove (shared) |
| TEE attestation | NoopTee (placeholder) | NoopTee (placeholder) |
| Audit logging | Append-only JSONL (new ZkAuditLog) | Extended existing AuditLogger |

### ZK Proxy Module (shared)

Both projects share an identical `src/zkproxy/` module behind the `zkproxy` feature flag:

| Component | File | Role |
|-----------|------|------|
| `worker.rs` | PersistentWorker | Tokio subprocess, JSON-RPC over stdin/stdout to Python |
| `feature.rs` | FeatureExtractor | Regex + builtins → `Vec<f32>` from `guard_config.json` |
| `guard.rs` | ZkProxy | Orchestrator: extract → call worker → parse proof → audit |
| `tee.rs` | TeeBackend | Attestation trait (NoopTee self-signs proof_hash + timestamp) |
| `audit.rs` | ZkAuditLog | Append-only JSONL with proof hash, timing, features |
| `types.rs` | — | ProofResult, GuardDecision, TimingBreakdown, FeatureConfig |
| `config.rs` | ZkProxyConfig | Env-driven config (model path, python bin, threshold) |

The Python-side `zkproxy_worker.py` keeps JSTprove loaded and circuits cached, eliminating per-request subprocess overhead. It communicates via JSON-RPC and returns timing breakdowns per operation (witness, prove, verify).

---

## ZK Proof Pipeline (Expected Overhead)

The ZK proof pipeline is not benchmarked in this report because it requires a compiled ONNX guard model and jstprove availability. Based on DSperse documentation and the `proof_runner.py` reference implementation, expected latencies for an 8-feature, 2-layer MLP guard model are:

| Phase | Expected Latency | Notes |
|-------|-----------------|-------|
| Feature extraction (Rust) | 2-3 us | Measured above |
| Witness generation | 5-20 ms | Depends on model complexity |
| Proof generation | 50-200 ms | jstprove circuit proving |
| Proof verification | 5-50 ms | Much faster than proving |
| **Total ZK pipeline** | **~60-270 ms** | Dominated by proof generation |

The persistent worker amortizes model loading and circuit compilation across requests. Health checks confirm readiness at startup.

---

## Recommendations

1. **Use ZeroClaw's pattern coverage as baseline**: Its broader category set (jailbreak, tool injection, command injection, secret extraction) catches more attack types. IronClaw should add these categories to its Sanitizer patterns.

2. **Merge IronClaw's Aho-Corasick for hot path**: ZeroClaw should consider Aho-Corasick for its most common string-match patterns (especially system override markers) to get IronClaw's sub-250 ns benign-text performance.

3. **ZK model training on labeled dataset**: The 143-prompt dataset can serve as initial training data for the guard ONNX model. The feature vectors from both projects' extractors should be compared to determine which feature set produces better separation between benign and injection classes.

4. **Proof generation caching**: For repeated identical inputs (same feature vector), proof results can be cached by feature hash to avoid redundant proof generation.

5. **TEE backend**: Replace `NoopTee` with AWS Nitro Enclave or Intel SGX attestation for production deployments where proof integrity must be verified by third parties.

---

## Reproducing These Results

```bash
# IronClaw benchmarks
cd ZKironclaw
rustup run 1.92.0 cargo bench --features zkproxy --bench zkproxy -- --output-format bencher

# ZeroClaw benchmarks
cd verifiedzeroclaw
cargo bench --features zkproxy --bench zkproxy -- --output-format bencher

# Accuracy evaluation
python3 benchmarks/run_accuracy.py

# Cross-project comparison (runs both benchmarks)
python3 benchmarks/compare.py
```

---

## Source Repositories

| Project | Repository | Branch |
|---------|-----------|--------|
| ZKironclaw (IronClaw) | https://github.com/shirin-shahabi/ZKironclaw | main |
| verifiedzeroclaw (ZeroClaw) | https://github.com/shirin-shahabi/verifiedzeroclaw | main |
| DSperse (benchmarks) | https://github.com/inference-labs-inc/dsperse | openclaw |
