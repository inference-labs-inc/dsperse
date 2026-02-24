"""Cross-project benchmark comparison.

Runs both project benchmarks and computes comparative statistics.
Outputs a markdown table comparing IronClaw vs ZeroClaw.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OPENCLAW_ROOT = Path(__file__).resolve().parent.parent
IRONCLAW_DIR = OPENCLAW_ROOT / "ZKironclaw"
ZEROCLAW_DIR = OPENCLAW_ROOT / "verifiedzeroclaw"
DATASET_PATH = OPENCLAW_ROOT / "benchmarks" / "dataset.jsonl"


@dataclass
class BenchResult:
    name: str
    mean_ns: float = 0.0
    median_ns: float = 0.0
    p95_ns: float = 0.0
    p99_ns: float = 0.0
    samples: int = 0


@dataclass
class DetectionResult:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run_criterion(project_dir: Path, bench_name: str) -> list[BenchResult]:
    cmd = [
        "cargo", "bench",
        "--features", "zkproxy",
        "--bench", bench_name,
        "--", "--output-format", "bencher",
    ]
    proc = subprocess.run(
        cmd, cwd=str(project_dir),
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"WARN: bench failed in {project_dir.name}: {proc.stderr[:500]}", file=sys.stderr)
        return []

    results = []
    for line in proc.stdout.splitlines():
        m = re.match(r"test\s+(\S+)\s+\.\.\.\s+bench:\s+([\d,]+)\s+ns/iter", line)
        if m:
            name = m.group(1)
            ns = float(m.group(2).replace(",", ""))
            results.append(BenchResult(name=name, mean_ns=ns, median_ns=ns))
    return results


def load_dataset() -> list[dict[str, str]]:
    entries = []
    with DATASET_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f} us"
    return f"{ns / 1_000_000:.2f} ms"


def print_comparison_table(
    ironclaw_results: list[BenchResult],
    zeroclaw_results: list[BenchResult],
) -> None:
    ic_map = {r.name: r for r in ironclaw_results}
    zc_map = {r.name: r for r in zeroclaw_results}

    print("\n## Latency Comparison\n")
    print("| Benchmark | IronClaw | ZeroClaw | Ratio |")
    print("|-----------|----------|----------|-------|")

    all_names = sorted(set(list(ic_map.keys()) + list(zc_map.keys())))
    for name in all_names:
        ic = ic_map.get(name)
        zc = zc_map.get(name)
        ic_str = format_ns(ic.mean_ns) if ic else "N/A"
        zc_str = format_ns(zc.mean_ns) if zc else "N/A"
        if ic and zc and zc.mean_ns > 0:
            ratio = f"{ic.mean_ns / zc.mean_ns:.2f}x"
        else:
            ratio = "N/A"
        print(f"| {name} | {ic_str} | {zc_str} | {ratio} |")


def print_detection_comparison() -> None:
    dataset = load_dataset()
    if not dataset:
        print("\nNo dataset loaded, skipping detection comparison.", file=sys.stderr)
        return

    benign_count = sum(1 for d in dataset if d["label"] == "benign")
    injection_count = sum(1 for d in dataset if d["label"] == "injection")
    categories = sorted(set(d.get("category", "unknown") for d in dataset if d["label"] == "injection"))

    print("\n## Dataset Summary\n")
    print(f"- Total prompts: {len(dataset)}")
    print(f"- Benign: {benign_count}")
    print(f"- Injection: {injection_count}")
    print(f"- Injection categories: {', '.join(categories)}")

    print("\n## Detection Dimensions\n")
    print("| Dimension | ZKironclaw (IronClaw) | verifiedzeroclaw (ZeroClaw) |")
    print("|-----------|----------------------|---------------------------|")
    print("| Pattern defense | Aho-Corasick + regex (17+4 patterns) | Regex only (26 patterns, 6 categories) |")
    print("| Feature extraction | 8 features from Sanitizer/Validator/Leak | 8 features from 6 category scores + builtins |")
    print("| ZK proof model | jstprove-compatible ONNX MLP | jstprove-compatible ONNX MLP |")
    print("| Guard threshold | Configurable (default 0.5) | Configurable sensitivity (default 0.7) |")
    print("| TEE attestation | NoopTee (self-signed) | NoopTee (self-signed) |")
    print("| Audit format | Append-only JSONL | Extended AuditLogger JSONL |")


def main() -> None:
    print("# ZK Proxy Benchmark Comparison: IronClaw vs ZeroClaw\n")

    print("Running IronClaw benchmarks...")
    ic_results = run_criterion(IRONCLAW_DIR, "zkproxy")

    print("Running ZeroClaw benchmarks...")
    zc_results = run_criterion(ZEROCLAW_DIR, "zkproxy")

    if ic_results or zc_results:
        print_comparison_table(ic_results, zc_results)
    else:
        print("\nNo benchmark results collected. Ensure both projects compile with `--features zkproxy`.")
        print("Generating comparison table with dataset and architecture info only.\n")

    print_detection_comparison()

    print("\n## Architecture Notes\n")
    print("Both projects share the same `zkproxy` module structure:")
    print("- `worker.rs`: Persistent Python subprocess (JSTprove backend)")
    print("- `feature.rs`: Rust-native regex feature extraction from guard_config.json")
    print("- `guard.rs`: Orchestrator (extract -> witness -> prove -> verify)")
    print("- `tee.rs`: TEE attestation (NoopTee for now)")
    print("- `audit.rs`: Append-only JSONL audit log with proof hashes and timing")
    print()
    print("Guard models are built from YAML definitions via `guard_builder.py`,")
    print("compiled to jstprove-compatible ONNX, and served by `zkproxy_worker.py`.")


if __name__ == "__main__":
    main()
