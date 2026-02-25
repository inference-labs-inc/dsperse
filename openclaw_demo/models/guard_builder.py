from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper

from dsperse.src.backends.utils.jstprove_utils import JSTPROVE_SUPPORTED_OPS

BUILTIN_FEATURES = {
    "normalized_length",
    "digit_ratio",
    "whitespace_ratio",
    "uppercase_ratio",
    "special_char_ratio",
    "avg_word_length",
    "line_count_norm",
    "entropy",
}


def load_guard_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def validate_guard_spec(spec: dict[str, Any]) -> None:
    features = spec.get("features", [])
    if not features:
        raise ValueError("Guard spec must define at least one feature")

    for feat in features:
        kind = feat.get("type")
        if kind == "builtin":
            name = feat.get("name")
            if name not in BUILTIN_FEATURES:
                raise ValueError(f"Unknown builtin feature: {name}")
        elif kind == "regex_count":
            if "patterns" not in feat:
                raise ValueError(f"regex_count feature '{feat.get('name')}' requires 'patterns'")
        elif kind == "string_match":
            if "strings" not in feat:
                raise ValueError(f"string_match feature '{feat.get('name')}' requires 'strings'")
        else:
            raise ValueError(f"Unknown feature type: {kind}")

    model_spec = spec.get("model", {})
    layers = model_spec.get("layers", [])
    if not layers:
        raise ValueError("Guard spec must define at least one model layer")

    for layer in layers:
        op = layer.get("op")
        if op not in JSTPROVE_SUPPORTED_OPS:
            raise ValueError(f"Unsupported op '{op}' -- jstprove supports: {sorted(JSTPROVE_SUPPORTED_OPS)}")


def build_guard_config(spec: dict[str, Any]) -> dict[str, Any]:
    features = spec["features"]
    config_features = []
    for feat in features:
        entry: dict[str, Any] = {
            "name": feat["name"],
            "type": feat["type"],
            "index": feat.get("index", len(config_features)),
        }
        if feat["type"] == "regex_count":
            entry["patterns"] = feat["patterns"]
        elif feat["type"] == "string_match":
            entry["strings"] = feat["strings"]
        elif feat["type"] == "builtin":
            pass
        config_features.append(entry)

    return {
        "input_features": len(features),
        "features": config_features,
        "threshold": spec.get("threshold", 0.5),
        "model_name": spec.get("name", "guard"),
    }


def build_guard_onnx(spec: dict[str, Any], output_path: Path, seed: int = 42) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_features = len(spec["features"])
    model_spec = spec.get("model", {})
    layers = model_spec.get("layers", [])

    rng = np.random.default_rng(seed)
    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    prev_output = "features"
    prev_dim = n_features
    node_idx = 0

    for layer in layers:
        op = layer["op"]
        out_dim = layer.get("out_dim", prev_dim)
        out_name = f"x{node_idx}"

        if op == "Gemm":
            w_name = f"layer{node_idx}.weight"
            b_name = f"layer{node_idx}.bias"
            w = (rng.standard_normal((out_dim, prev_dim)).astype(np.float32) * 0.02)
            b = (rng.standard_normal((out_dim,)).astype(np.float32) * 0.02)
            inits.append(helper.make_tensor(w_name, TensorProto.FLOAT, w.shape, w.flatten().tolist()))
            inits.append(helper.make_tensor(b_name, TensorProto.FLOAT, b.shape, b.flatten().tolist()))
            nodes.append(helper.make_node("Gemm", [prev_output, w_name, b_name], [out_name], name=f"gemm_{node_idx}"))
            prev_dim = out_dim

        elif op == "Relu":
            nodes.append(helper.make_node("Relu", [prev_output], [out_name], name=f"relu_{node_idx}"))

        elif op == "Add":
            param_name = f"layer{node_idx}.param"
            val = layer.get("value", 0.05)
            p = np.full((prev_dim,), val, dtype=np.float32)
            inits.append(helper.make_tensor(param_name, TensorProto.FLOAT, p.shape, p.flatten().tolist()))
            nodes.append(helper.make_node("Add", [prev_output, param_name], [out_name], name=f"add_{node_idx}"))

        elif op == "Mul":
            param_name = f"layer{node_idx}.param"
            val = layer.get("value", 1.1)
            p = np.full((prev_dim,), val, dtype=np.float32)
            inits.append(helper.make_tensor(param_name, TensorProto.FLOAT, p.shape, p.flatten().tolist()))
            nodes.append(helper.make_node("Mul", [prev_output, param_name], [out_name], name=f"mul_{node_idx}"))

        elif op == "Sub":
            param_name = f"layer{node_idx}.param"
            val = layer.get("value", 0.02)
            p = np.full((prev_dim,), val, dtype=np.float32)
            inits.append(helper.make_tensor(param_name, TensorProto.FLOAT, p.shape, p.flatten().tolist()))
            nodes.append(helper.make_node("Sub", [prev_output, param_name], [out_name], name=f"sub_{node_idx}"))

        elif op == "Clip":
            min_name = f"layer{node_idx}.min"
            max_name = f"layer{node_idx}.max"
            min_val = np.array(layer.get("min", 0.0), dtype=np.float32)
            max_val = np.array(layer.get("max", 1.0), dtype=np.float32)
            inits.append(helper.make_tensor(min_name, TensorProto.FLOAT, [], [float(min_val)]))
            inits.append(helper.make_tensor(max_name, TensorProto.FLOAT, [], [float(max_val)]))
            nodes.append(helper.make_node("Clip", [prev_output, min_name, max_name], [out_name], name=f"clip_{node_idx}"))

        else:
            raise ValueError(f"Layer op '{op}' not yet implemented in guard_builder")

        prev_output = out_name
        node_idx += 1

    features_input = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_features])
    allow_score = helper.make_tensor_value_info("allow_score", TensorProto.FLOAT, [None, prev_dim])

    final_node = nodes[-1]
    final_node.output[0] = "allow_score"

    graph = helper.make_graph(nodes, "guard_model", [features_input], [allow_score], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def build_from_yaml(
    yaml_path: Path,
    onnx_output: Path | None = None,
    config_output: Path | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    spec = load_guard_yaml(yaml_path)
    validate_guard_spec(spec)

    name = spec.get("name", yaml_path.stem)
    out_dir = yaml_path.parent

    if onnx_output is None:
        onnx_output = out_dir / f"{name}.onnx"
    if config_output is None:
        config_output = out_dir / f"{name}_config.json"

    build_guard_onnx(spec, onnx_output, seed=seed)

    config = build_guard_config(spec)
    model_bytes = onnx_output.read_bytes()
    config["model_hash_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    config["onnx_path"] = str(onnx_output)

    with config_output.open("w") as f:
        json.dump(config, f, indent=2)

    return onnx_output, config_output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <guard.yaml> [output_dir]")
        sys.exit(1)

    yaml_file = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else yaml_file.parent

    onnx_path, config_path = build_from_yaml(
        yaml_file,
        onnx_output=out_dir / f"{yaml_file.stem}.onnx",
        config_output=out_dir / f"{yaml_file.stem}_config.json",
    )
    print(f"ONNX:   {onnx_path}")
    print(f"Config: {config_path}")

    model = onnx.load(str(onnx_path))
    ops = sorted({node.op_type for node in model.graph.node})
    print(f"Ops:    {', '.join(ops)}")
