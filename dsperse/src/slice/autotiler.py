import onnx
from onnx import helper, TensorProto, numpy_helper
import json
import os
from pathlib import Path

ELEMENTWISE_OPS = {'Sigmoid', 'Mul', 'Add', 'Sub', 'Div', 'Relu', 'LeakyRelu', 'PRelu', 'Tanh', 'Clip', 'Neg', 'Abs', 'Sqrt', 'Exp', 'Log', 'Pow', 'Sin', 'Cos'}


def find_tile_size(spatial_dim: int, target: int) -> int | None:
    if spatial_dim <= target:
        return None
    for tile in range(min(target, spatial_dim), 7, -1):
        if spatial_dim % tile == 0:
            return tile
    return None


def is_tileable(model: onnx.ModelProto) -> bool:
    ops = {n.op_type for n in model.graph.node}
    if 'Conv' not in ops:
        return False
    return (ops - {'Conv'}).issubset(ELEMENTWISE_OPS)


def get_conv_params(model: onnx.ModelProto):
    for node in model.graph.node:
        if node.op_type == "Conv":
            attrs = {a.name: a for a in node.attribute}
            return {
                'node': node,
                'kernel': list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else [3, 3],
                'stride': list(attrs["strides"].ints) if "strides" in attrs else [1, 1],
                'dilation': list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1],
                'pads': list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0],
            }
    return None


def tile_conv_slice(slice_path: str, tile_size: int, output_path: str):
    m = onnx.load(slice_path)
    orig_input = m.graph.input[0]
    orig_dims = [d.dim_value for d in orig_input.type.tensor_type.shape.dim]
    _, c_in, h_in, w_in = orig_dims

    conv_params = get_conv_params(m)
    if not conv_params:
        return None

    conv_node = conv_params['node']
    kh, kw = conv_params['kernel']
    sh, sw = conv_params['stride']
    dh, dw = conv_params['dilation']
    pads = conv_params['pads']

    weights, bias = None, None
    for init in m.graph.initializer:
        if init.name == conv_node.input[1]:
            weights = numpy_helper.to_array(init)
        if len(conv_node.input) > 2 and init.name == conv_node.input[2]:
            bias = numpy_helper.to_array(init)

    if weights is None:
        return None

    c_out = weights.shape[0]
    halo_h, halo_w = pads[0], pads[1]
    effective_kh = (kh - 1) * dh + 1
    effective_kw = (kw - 1) * dw + 1

    tile_with_halo_h = tile_size + 2 * halo_h
    tile_with_halo_w = tile_size + 2 * halo_w
    conv_out_h = (tile_with_halo_h - effective_kh) // sh + 1
    conv_out_w = (tile_with_halo_w - effective_kw) // sw + 1
    out_tile_h = tile_size // sh
    out_tile_w = tile_size // sw

    X = helper.make_tensor_value_info(orig_input.name, TensorProto.FLOAT, [1, c_in, tile_with_halo_h, tile_with_halo_w])
    Y = helper.make_tensor_value_info(m.graph.output[0].name, TensorProto.FLOAT, [1, c_out, conv_out_h, conv_out_w])

    W = helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.flatten().tolist())
    initializers = [W]
    conv_inputs = [orig_input.name, "W"]
    if bias is not None:
        B = helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.flatten().tolist())
        initializers.append(B)
        conv_inputs.append("B")

    nodes = [helper.make_node("Conv", conv_inputs, ["conv_out"], kernel_shape=[kh, kw], strides=[sh, sw], pads=[0, 0, 0, 0], dilations=[dh, dw])]

    non_conv_ops = [n for n in m.graph.node if n.op_type != "Conv"]
    if non_conv_ops:
        for i, orig_node in enumerate(non_conv_ops):
            new_inputs = []
            for inp in orig_node.input:
                if any(inp == n.output[0] for n in m.graph.node if n.op_type == "Conv"):
                    new_inputs.append("conv_out")
                elif inp == orig_input.name:
                    new_inputs.append(orig_input.name)
                else:
                    new_inputs.append(inp)
            is_last = (i == len(non_conv_ops) - 1)
            new_outputs = [m.graph.output[0].name] if is_last else list(orig_node.output)
            nodes.append(helper.make_node(orig_node.op_type, new_inputs, new_outputs))
    else:
        nodes[-1].output[0] = m.graph.output[0].name

    graph = helper.make_graph(nodes, "tiled_slice", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    return {
        "original_input": orig_dims,
        "tile_size": tile_size,
        "halo": [halo_h, halo_w],
        "tile_with_halo": [tile_with_halo_h, tile_with_halo_w],
        "conv_out": [conv_out_h, conv_out_w],
        "out_tile": [out_tile_h, out_tile_w],
        "crop_offset": [(conv_out_h - out_tile_h) // 2, (conv_out_w - out_tile_w) // 2],
        "stride": [sh, sw],
        "kernel": [kh, kw],
        "c_in": c_in,
        "c_out": c_out,
    }


def autotile_slices(slices_dir: str, tile_size: int = 16) -> dict:
    """
    Scan slices directory, tile Conv slices with spatial > tile_size.
    Returns dict of {slice_idx: tiling_info} for tiled slices.
    """
    tiled_dir = os.path.join(slices_dir, "tiled")
    Path(tiled_dir).mkdir(exist_ok=True)

    tiled_info = {}
    for i in range(500):
        path = os.path.join(slices_dir, f"slice_{i}", "payload", f"slice_{i}.onnx")
        if not os.path.exists(path):
            continue

        m = onnx.load(path)
        if not is_tileable(m):
            continue

        inp = m.graph.input[0]
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if len(dims) != 4:
            continue

        _, c, h, w = dims
        if h <= tile_size:
            continue

        tile = find_tile_size(h, tile_size)
        if not tile:
            continue

        dst = os.path.join(tiled_dir, f"slice_{i}_tiled.onnx")
        info = tile_conv_slice(path, tile, dst)
        if info:
            info["tiled_path"] = dst
            tiled_info[i] = info

    if tiled_info:
        with open(os.path.join(tiled_dir, "tiled_info.json"), "w") as f:
            json.dump({"slices": {str(k): v for k, v in tiled_info.items()}, "tiled_indices": list(tiled_info.keys())}, f, indent=2)

    return tiled_info
