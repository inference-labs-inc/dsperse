"""
Autotiler for Conv slices.

Tiling breaks large convolution operations into smaller tiles that can be processed
independently, reducing memory/circuit size for ZK proofs. The key insight is that
convolutions are spatially local - each output pixel only depends on a small window
of input pixels (the kernel receptive field).

Tiling strategy:
  1. Split input spatially into tiles of size T x T
  2. Each tile needs extra "halo" pixels around edges for correct convolution boundary
  3. The halo size equals the original Conv padding (pixels the Conv would read beyond edges)
  4. Process each tile independently through the same Conv + elementwise ops
  5. Crop output tiles to remove boundary artifacts, then stitch together

Example: 64x64 input with 3x3 conv, stride=2, pad=1 -> tile_size=16
  - Input tile: 16x16 + 2*1 halo = 18x18 (tile_with_halo)
  - Output tile: 8x8 (after stride=2)
  - 64/16 = 4x4 = 16 tiles to process
"""
import json
from pathlib import Path

import onnx
from onnx import helper, TensorProto, numpy_helper

ELEMENTWISE_OPS = {
    'Sigmoid', 'Mul', 'Add', 'Sub', 'Div', 'Relu', 'LeakyRelu', 'PRelu',
    'Tanh', 'Clip', 'Neg', 'Abs', 'Sqrt', 'Exp', 'Log', 'Pow', 'Sin', 'Cos'
}


def find_tile_size(spatial_dim: int, target: int) -> int | None:
    """
    Find largest tile size <= target that evenly divides spatial_dim.
    Returns None if no valid tile size >= 8 exists.
    """
    if 7 <= target < spatial_dim:
        for tile in range(target, 7, -1):
            if spatial_dim % tile == 0:
                return tile
    return None


def is_tileable(model: onnx.ModelProto) -> bool:
    """
    A slice is tileable if it contains exactly one Conv followed by
    zero or more elementwise ops. Elementwise ops apply independently
    per-pixel, so they don't affect tiling boundaries.
    """
    ops = {n.op_type for n in model.graph.node}
    if 'Conv' not in ops:
        return False
    return (ops - {'Conv'}).issubset(ELEMENTWISE_OPS)


def get_conv_params(model: onnx.ModelProto) -> dict | None:
    """Extract Conv node parameters from model."""
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


def tile_conv_slice(slice_path: Path, tile_size: int, output_path: Path) -> dict | None:
    """
    Create a tiled version of a Conv slice.

    The tiled model expects input tiles with halo (overlap) regions that provide
    the extra context needed for convolution at tile boundaries. The halo size
    equals the original padding - this is the key insight that makes tiling work.

    Args:
        slice_path: Path to original slice ONNX model
        tile_size: Spatial size of each tile (before adding halo)
        output_path: Where to save the tiled model

    Returns:
        Dict with tiling metadata, or None if slice can't be tiled
    """
    m = onnx.load(slice_path)
    orig_input = m.graph.input[0]
    orig_dims = [d.dim_value for d in orig_input.type.tensor_type.shape.dim]
    c_in = orig_dims[1]

    conv_params = get_conv_params(m)
    if not conv_params:
        return None

    conv_node = conv_params['node']
    kh, kw = conv_params['kernel']
    sh, sw = conv_params['stride']
    if sh == 0 or sw == 0:
        return None
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

    # Halo = original padding. This is the overlap needed at tile boundaries
    # to produce correct output without edge artifacts.
    halo_h, halo_w = pads[0], pads[1]

    # Effective kernel size accounting for dilation
    effective_kh = (kh - 1) * dh + 1
    effective_kw = (kw - 1) * dw + 1

    # Input tile dimensions (tile + halo on each side)
    tile_with_halo_h = tile_size + 2 * halo_h
    tile_with_halo_w = tile_size + 2 * halo_w

    # Output dimensions from convolution formula: (input - kernel) / stride + 1
    # We use pads=[0,0,0,0] in tiled conv since halo already provides the context
    conv_out_h = (tile_with_halo_h - effective_kh) // sh + 1
    conv_out_w = (tile_with_halo_w - effective_kw) // sw + 1

    # Expected output tile size (what we need after cropping)
    out_tile_h = tile_size // sh
    out_tile_w = tile_size // sw

    # Build the tiled ONNX model
    X = helper.make_tensor_value_info(
        orig_input.name, TensorProto.FLOAT,
        [1, c_in, tile_with_halo_h, tile_with_halo_w]
    )
    Y = helper.make_tensor_value_info(
        m.graph.output[0].name, TensorProto.FLOAT,
        [1, c_out, conv_out_h, conv_out_w]
    )

    W = helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.flatten().tolist())
    initializers = [W]
    conv_inputs = [orig_input.name, "W"]
    if bias is not None:
        B = helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.flatten().tolist())
        initializers.append(B)
        conv_inputs.append("B")

    # Conv with zero padding (halo provides the needed context)
    nodes = [helper.make_node(
        "Conv", conv_inputs, ["conv_out"],
        kernel_shape=[kh, kw], strides=[sh, sw], pads=[0, 0, 0, 0], dilations=[dh, dw]
    )]

    # Copy non-Conv ops (Sigmoid, Mul, etc), rewiring inputs to use conv_out
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
            attr_kwargs = {a.name: helper.get_attribute_value(a) for a in orig_node.attribute}
            nodes.append(helper.make_node(orig_node.op_type, new_inputs, new_outputs, **attr_kwargs))
    else:
        nodes[-1].output[0] = m.graph.output[0].name

    graph = helper.make_graph(nodes, "tiled_slice", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

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


def autotile_slices(slices_dir: str | Path, tile_size: int = 16) -> dict:
    """
    Scan slices directory and create tiled versions of Conv slices.

    Args:
        slices_dir: Directory containing slice_N subdirectories
        tile_size: Target tile size (will find largest divisor <= this)

    Returns:
        Dict mapping slice index to tiling metadata for each tiled slice
    """
    slices_dir = Path(slices_dir)
    tiled_dir = slices_dir / "tiled"
    tiled_dir.mkdir(exist_ok=True)

    tiled_info = {}
    for entry in sorted(slices_dir.iterdir()):
        if not entry.name.startswith("slice_") or not entry.is_dir():
            continue
        try:
            i = int(entry.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        onnx_path = entry / "payload" / f"{entry.name}.onnx"
        if not onnx_path.exists():
            continue

        m = onnx.load(str(onnx_path))
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

        dst = tiled_dir / f"slice_{i}_tiled.onnx"
        info = tile_conv_slice(onnx_path, tile, dst)
        if info:
            info["tiled_path"] = str(dst)
            tiled_info[i] = info

    if tiled_info:
        info_path = tiled_dir / "tiled_info.json"
        with open(info_path, "w") as f:
            json.dump({
                "slices": {str(k): v for k, v in tiled_info.items()},
                "tiled_indices": list(tiled_info.keys())
            }, f, indent=2)

    return tiled_info
