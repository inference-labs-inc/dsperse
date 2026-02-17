from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def build_guard_filter_onnx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    fc1_w = (rng.standard_normal((16, 8)).astype(np.float32) * 0.02)
    fc1_b = (rng.standard_normal((16,)).astype(np.float32) * 0.02)
    add_bias = np.full((16,), 0.05, dtype=np.float32)
    mul_scale = np.full((16,), 1.1, dtype=np.float32)
    sub_shift = np.full((16,), 0.02, dtype=np.float32)
    fc2_w = (rng.standard_normal((1, 16)).astype(np.float32) * 0.02)
    fc2_b = (rng.standard_normal((1,)).astype(np.float32) * 0.02)

    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 8])
    allow_score = helper.make_tensor_value_info("allow_score", TensorProto.FLOAT, [None, 1])

    inits = [
        helper.make_tensor("fc1.weight", TensorProto.FLOAT, fc1_w.shape, fc1_w.flatten().tolist()),
        helper.make_tensor("fc1.bias", TensorProto.FLOAT, fc1_b.shape, fc1_b.flatten().tolist()),
        helper.make_tensor("add_bias", TensorProto.FLOAT, add_bias.shape, add_bias.flatten().tolist()),
        helper.make_tensor("mul_scale", TensorProto.FLOAT, mul_scale.shape, mul_scale.flatten().tolist()),
        helper.make_tensor("sub_shift", TensorProto.FLOAT, sub_shift.shape, sub_shift.flatten().tolist()),
        helper.make_tensor("fc2.weight", TensorProto.FLOAT, fc2_w.shape, fc2_w.flatten().tolist()),
        helper.make_tensor("fc2.bias", TensorProto.FLOAT, fc2_b.shape, fc2_b.flatten().tolist()),
    ]

    nodes = [
        helper.make_node("Gemm", ["features", "fc1.weight", "fc1.bias"], ["x1"], name="fc1_gemm"),
        helper.make_node("Relu", ["x1"], ["x2"], name="relu"),
        helper.make_node("Add", ["x2", "add_bias"], ["x3"], name="add"),
        helper.make_node("Mul", ["x3", "mul_scale"], ["x4"], name="mul"),
        helper.make_node("Sub", ["x4", "sub_shift"], ["x5"], name="sub"),
        helper.make_node("Gemm", ["x5", "fc2.weight", "fc2.bias"], ["allow_score"], name="fc2_gemm"),
    ]

    graph = helper.make_graph(nodes, "guard_filter", [features], [allow_score], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def print_ops(path: Path) -> None:
    model = onnx.load(path)
    ops = sorted({node.op_type for node in model.graph.node})
    print("ONNX ops:", ", ".join(ops))


if __name__ == "__main__":
    out = Path("openclaw_demo/models/guard_filter.onnx")
    build_guard_filter_onnx(out)
    print_ops(out)
