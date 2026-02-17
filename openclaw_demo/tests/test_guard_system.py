from __future__ import annotations

import json
from pathlib import Path

import onnx
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_injection_dataset_has_10_examples() -> None:
    data_path = ROOT / "data" / "prompt_injection_examples.jsonl"
    lines = [line.strip() for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 10
    parsed = [json.loads(line) for line in lines]
    assert {item["type"] for item in parsed} == {"direct_prompt_injection", "indirect_prompt_injection"}


def test_guard_onnx_model_uses_supported_ops() -> None:
    model_path = ROOT / "models" / "guard_filter.onnx"
    if not model_path.exists():
        raise AssertionError("Missing openclaw_demo/models/guard_filter.onnx. Run build_guard_onnx.py first.")
    model = onnx.load(model_path)
    ops = {node.op_type for node in model.graph.node}
    allowed = {"Gemm", "Relu", "Add", "Mul", "Sub", "Conv", "MaxPool", "BatchNormalization"}
    assert ops.issubset(allowed), f"Unsupported ops detected: {sorted(ops - allowed)}"


def test_grafana_and_prometheus_config_present_and_valid() -> None:
    prometheus_path = ROOT / "gateway" / "monitoring" / "prometheus.yml"
    datasource_path = ROOT / "gateway" / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    loki_ds_path = ROOT / "gateway" / "monitoring" / "grafana" / "provisioning" / "datasources" / "loki.yml"
    dashboard_path = ROOT / "gateway" / "monitoring" / "grafana" / "dashboards" / "openclaw-guard.json"

    prom_cfg = yaml.safe_load(prometheus_path.read_text(encoding="utf-8"))
    ds_cfg = yaml.safe_load(datasource_path.read_text(encoding="utf-8"))
    loki_cfg = yaml.safe_load(loki_ds_path.read_text(encoding="utf-8"))
    dash_cfg = json.loads(dashboard_path.read_text(encoding="utf-8"))

    assert "scrape_configs" in prom_cfg
    assert ds_cfg["datasources"][0]["type"] == "prometheus"
    assert loki_cfg["datasources"][0]["type"] == "loki"
    assert dash_cfg["title"] == "OpenClaw Guard Observability"
    panel_titles = {panel["title"] for panel in dash_cfg["panels"]}
    assert "Recent Requests (Table)" in panel_titles
    assert "Block Reasons (Rate)" in panel_titles
