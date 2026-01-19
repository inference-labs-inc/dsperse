"""
Autotiler for Conv slices.

Tiling breaks large convolution operations into smaller tiles that can be processed
independently, reducing memory/circuit size for ZK proofs.

Architecture:
  1. SPLIT slice: Pad input, extract N tiles with halo → N outputs
  2. TILE slices: Each processes one tile through Conv + elementwise → one output each
  3. CONCAT slice: Reassemble N tile outputs → full spatial output

Example: 1280x1280 input with 64x64 tiles, kernel=3, stride=2, pad=1
  - 20x20 = 400 tiles
  - Split: [1,3,1280,1280] → 400x [1,3,66,66]
  - Tiles: [1,3,66,66] → [1,32,32,32] each
  - Concat: 400x [1,32,32,32] → [1,32,640,640]
"""
import json
from pathlib import Path

import onnx
from onnx import helper, TensorProto, numpy_helper

ELEMENTWISE_OPS = {
    'Sigmoid', 'Mul', 'Add', 'Sub', 'Div', 'Relu', 'LeakyRelu', 'PRelu',
    'Tanh', 'Clip', 'Neg', 'Abs', 'Sqrt', 'Exp', 'Log', 'Pow', 'Sin', 'Cos'
}


def append_split_to_onnx(
    onnx_path: Path,
    split_input_name: str,
    c_in: int,
    h: int,
    w: int,
    tile_size: int,
    halo_h: int,
    halo_w: int,
    slice_idx: int
) -> list[str]:
    """
    Append split nodes to an existing ONNX model.
    The split consumes the tensor named `split_input_name` and produces tile outputs.
    Preserves ALL original outputs (for skip connections).

    Returns: list of output tensor names (tile inputs)
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    nodes, initializers, output_names, metadata = generate_split_nodes(
        split_input_name, c_in, h, w, tile_size, halo_h, halo_w, slice_idx, prefix=f"split_{slice_idx}_"
    )

    for init in initializers:
        graph.initializer.append(init)
    for node in nodes:
        graph.node.append(node)

    tile_with_halo_h, tile_with_halo_w = metadata["tile_with_halo"]

    # Preserve ALL original outputs (including split_input_name for skip connections)
    original_outputs = list(graph.output)
    while len(graph.output) > 0:
        graph.output.pop()
    for name in output_names:
        graph.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, [1, c_in, tile_with_halo_h, tile_with_halo_w]
        ))
    for out in original_outputs:
        graph.output.append(out)

    onnx.save(model, str(onnx_path))
    return output_names


def prepend_concat_to_onnx(
    onnx_path: Path,
    concat_output_name: str,
    num_tiles: int,
    tiles_y: int,
    tiles_x: int,
    c_out: int,
    out_tile_h: int,
    out_tile_w: int,
    slice_idx: int
) -> list[str]:
    """
    Prepend concat nodes to an existing ONNX model.
    The concat produces tensor named `concat_output_name` from tile outputs.
    Preserves other inputs (e.g., for residual connections).

    Returns: list of ALL input tensor names (tile outputs + skip connections)
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    nodes, initializers, input_names = generate_concat_nodes(
        num_tiles, tiles_y, tiles_x, slice_idx, concat_output_name, prefix=f"concat_{slice_idx}_"
    )

    for init in initializers:
        graph.initializer.append(init)
    for node in reversed(nodes):
        graph.node.insert(0, node)

    other_inputs = [inp for inp in graph.input if inp.name != concat_output_name]
    while len(graph.input) > 0:
        graph.input.pop()
    for name in input_names:
        graph.input.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, [1, c_out, out_tile_h, out_tile_w]
        ))
    for inp in other_inputs:
        graph.input.append(inp)

    onnx.save(model, str(onnx_path))

    all_input_names = input_names + [inp.name for inp in other_inputs]
    return all_input_names


def compute_halo(kernel: list[int], dilation: list[int]) -> tuple[int, int]:
    effective_kh = (kernel[0] - 1) * dilation[0] + 1
    effective_kw = (kernel[1] - 1) * dilation[1] + 1
    return effective_kh // 2, effective_kw // 2


def compute_min_tile_size(kernel: list[int], dilation: list[int]) -> int:
    effective_kh = (kernel[0] - 1) * dilation[0] + 1
    effective_kw = (kernel[1] - 1) * dilation[1] + 1
    return max(effective_kh, effective_kw) + 1


def find_tile_size(spatial_dim: int, target: int, min_tile: int = 7, stride: int = 1) -> int | None:
    if min_tile <= target < spatial_dim:
        for tile in range(target, min_tile - 1, -1):
            if spatial_dim % tile == 0 and tile % stride == 0:
                return tile
    return None


def calculate_tile_size_from_max_elements(
    channels: int,
    spatial_h: int,
    spatial_w: int,
    max_conv_size: int,
    min_tile: int = 7
) -> tuple[int | None, str | None]:
    """
    Calculate optimal tile size based on max element count constraint.

    The ZK circuit complexity depends on total tensor elements, not just resolution.
    Given max_conv_size (max elements per tile), calculate the largest tile size
    such that: channels * tile_h * tile_w <= max_conv_size

    For square tiles: tile_size <= sqrt(max_conv_size / channels)

    Args:
        channels: Number of channels (C dimension)
        spatial_h: Input height
        spatial_w: Input width
        max_conv_size: Maximum elements allowed per tile
        min_tile: Minimum tile size (must be > kernel effective size)

    Returns:
        Tuple of (tile_size, skip_reason):
        - (tile_size, None) if tiling is possible
        - (None, "already_fits") if total elements <= max_conv_size
        - (None, "min_tile_too_large") if required tile_size < min_tile (kernel constraint)
        - (None, "no_divisor") if no valid tile size divides spatial dim
    """
    import math

    total_elements = channels * spatial_h * spatial_w
    if total_elements <= max_conv_size:
        return None, "already_fits"

    max_tile_from_constraint = int(math.sqrt(max_conv_size / channels))

    if max_tile_from_constraint < min_tile:
        return None, "min_tile_too_large"

    target_tile = min(max_tile_from_constraint, spatial_h, spatial_w)

    tile_size = find_tile_size(spatial_h, target_tile, min_tile)
    if tile_size is None:
        return None, "no_divisor"

    return tile_size, None


def is_tileable(model: onnx.ModelProto) -> bool:
    if len(model.graph.input) > 1:
        return False
    conv_count = sum(1 for n in model.graph.node if n.op_type == "Conv")
    if conv_count != 1:
        return False
    conv_params = get_conv_params(model)
    if conv_params:
        kh, kw = conv_params['kernel']
        if kh % 2 == 0 or kw % 2 == 0:
            return False
        pads = conv_params['pads']
        dilation = conv_params['dilation']
        halo_h, halo_w = compute_halo([kh, kw], dilation)
        if pads != [halo_h, halo_w, halo_h, halo_w]:
            return False
    ops = {n.op_type for n in model.graph.node}
    return (ops - {'Conv'}).issubset(ELEMENTWISE_OPS)


def get_conv_params(model: onnx.ModelProto) -> dict | None:
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


def generate_split_nodes(
    input_name: str,
    c_in: int,
    h: int,
    w: int,
    tile_size: int,
    halo_h: int,
    halo_w: int,
    slice_idx: int,
    prefix: str = ""
) -> tuple[list, list, list[str], dict]:
    """
    Generate Pad + Slice nodes for splitting input into tiles.

    Returns: (nodes, initializers, output_names, metadata)
    """
    tiles_y = h // tile_size
    tiles_x = w // tile_size
    num_tiles = tiles_y * tiles_x
    tile_with_halo_h = tile_size + 2 * halo_h
    tile_with_halo_w = tile_size + 2 * halo_w

    nodes = []
    initializers = []
    output_names = []

    padded_name = f"{prefix}padded" if prefix else "padded"
    pads_name = f"{prefix}pads" if prefix else "pads"
    const_name = f"{prefix}constant_value" if prefix else "constant_value"

    pads_val = [0, 0, halo_h, halo_w, 0, 0, halo_h, halo_w]
    initializers.append(helper.make_tensor(pads_name, TensorProto.INT64, [8], pads_val))
    initializers.append(helper.make_tensor(const_name, TensorProto.FLOAT, [], [0.0]))

    nodes.append(helper.make_node(
        "Pad",
        inputs=[input_name, pads_name, const_name],
        outputs=[padded_name],
        mode="constant"
    ))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_idx = ty * tiles_x + tx
            out_name = f"tile_{slice_idx}_{tile_idx}_in"
            output_names.append(out_name)

            y_start = ty * tile_size
            x_start = tx * tile_size
            y_end = y_start + tile_with_halo_h
            x_end = x_start + tile_with_halo_w

            starts_name = f"{prefix}starts_{tile_idx}"
            ends_name = f"{prefix}ends_{tile_idx}"
            axes_name = f"{prefix}axes_{tile_idx}"

            initializers.append(helper.make_tensor(starts_name, TensorProto.INT64, [4], [0, 0, y_start, x_start]))
            initializers.append(helper.make_tensor(ends_name, TensorProto.INT64, [4], [1, c_in, y_end, x_end]))
            initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [4], [0, 1, 2, 3]))

            nodes.append(helper.make_node(
                "Slice",
                inputs=[padded_name, starts_name, ends_name, axes_name],
                outputs=[out_name]
            ))

    metadata = {
        "tiles_y": tiles_y,
        "tiles_x": tiles_x,
        "num_tiles": num_tiles,
        "tile_with_halo": [tile_with_halo_h, tile_with_halo_w],
        "c_in": c_in,
    }

    return nodes, initializers, output_names, metadata


def create_split_slice(
    input_name: str,
    c_in: int,
    h: int,
    w: int,
    tile_size: int,
    halo_h: int,
    halo_w: int,
    slice_idx: int,
    output_dir: Path,
    nested: bool = False
) -> dict:
    """
    Create a split slice that pads input and extracts all tiles.

    Input: [1, C, H, W]
    Output: N tensors of [1, C, tile_size + 2*halo, tile_size + 2*halo]
    """
    nodes, initializers, output_names, metadata = generate_split_nodes(
        input_name, c_in, h, w, tile_size, halo_h, halo_w, slice_idx
    )

    tile_with_halo_h, tile_with_halo_w = metadata["tile_with_halo"]

    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, c_in, h, w])
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, c_in, tile_with_halo_h, tile_with_halo_w])
        for name in output_names
    ]

    graph = helper.make_graph(nodes, f"split_slice_{slice_idx}", [X], outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    if nested:
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / "split.onnx"
    else:
        split_dir = output_dir / f"slice_{slice_idx}_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        payload_dir = split_dir / "payload"
        payload_dir.mkdir(exist_ok=True)
        onnx_path = payload_dir / f"slice_{slice_idx}_split.onnx"

    onnx.save(model, str(onnx_path))

    return {
        "path": str(onnx_path.resolve()),
        "input_name": input_name,
        "output_names": output_names,
        **metadata,
    }


def create_tile_slice(
    slice_path: Path,
    tile_size: int,
    slice_idx: int,
    output_dir: Path,
    nested: bool = False
) -> dict | None:
    """
    Create a single reusable tile processing slice.

    This creates ONE tile ONNX that can be executed N times with different inputs.
    Uses generic names (tile_in, tile_out) - the runner maps actual tensor names.

    Input: [1, C, tile_size + 2*halo, tile_size + 2*halo]
    Output: [1, C_out, out_tile, out_tile]
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

    weights, bias = None, None
    for init in m.graph.initializer:
        if init.name == conv_node.input[1]:
            weights = numpy_helper.to_array(init)
        if len(conv_node.input) > 2 and init.name == conv_node.input[2]:
            bias = numpy_helper.to_array(init)

    if weights is None:
        return None

    c_out = weights.shape[0]
    halo_h, halo_w = compute_halo([kh, kw], [dh, dw])
    effective_kh = (kh - 1) * dh + 1
    effective_kw = (kw - 1) * dw + 1
    tile_with_halo_h = tile_size + 2 * halo_h
    tile_with_halo_w = tile_size + 2 * halo_w
    conv_out_h = (tile_with_halo_h - effective_kh) // sh + 1
    conv_out_w = (tile_with_halo_w - effective_kw) // sw + 1
    out_tile_h = tile_size // sh
    out_tile_w = tile_size // sw

    input_name = "tile_in"
    output_name = "tile_out"

    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, c_in, tile_with_halo_h, tile_with_halo_w])
    Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, c_out, conv_out_h, conv_out_w])

    W = helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.flatten().tolist())
    initializers = [W]
    conv_inputs = [input_name, "W"]
    if bias is not None:
        B = helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.flatten().tolist())
        initializers.append(B)
        conv_inputs.append("B")

    nodes = [helper.make_node(
        "Conv", conv_inputs, ["conv_out"],
        kernel_shape=[kh, kw], strides=[sh, sw], pads=[0, 0, 0, 0], dilations=[dh, dw]
    )]

    non_conv_ops = [n for n in m.graph.node if n.op_type != "Conv"]
    if non_conv_ops:
        for init in m.graph.initializer:
            if init.name not in [conv_node.input[1]] + ([conv_node.input[2]] if len(conv_node.input) > 2 else []):
                initializers.append(init)

        for i, orig_node in enumerate(non_conv_ops):
            new_inputs = []
            for inp in orig_node.input:
                if any(inp == n.output[0] for n in m.graph.node if n.op_type == "Conv"):
                    new_inputs.append("conv_out")
                elif inp == orig_input.name:
                    new_inputs.append(input_name)
                else:
                    new_inputs.append(inp)
            is_last = (i == len(non_conv_ops) - 1)
            new_outputs = [output_name] if is_last else list(orig_node.output)
            attr_kwargs = {a.name: helper.get_attribute_value(a) for a in orig_node.attribute}
            nodes.append(helper.make_node(orig_node.op_type, new_inputs, new_outputs, **attr_kwargs))
    else:
        nodes[-1].output[0] = output_name

    graph = helper.make_graph(nodes, f"tile_{slice_idx}", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    if nested:
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / "tile.onnx"
    else:
        tile_dir = output_dir / f"slice_{slice_idx}_tile"
        tile_dir.mkdir(parents=True, exist_ok=True)
        payload_dir = tile_dir / "payload"
        payload_dir.mkdir(exist_ok=True)
        onnx_path = payload_dir / f"slice_{slice_idx}_tile.onnx"

    onnx.save(model, str(onnx_path))

    return {
        "path": str(onnx_path.resolve()),
        "input_name": input_name,
        "output_name": output_name,
        "conv_out": [conv_out_h, conv_out_w],
        "out_tile": [out_tile_h, out_tile_w],
    }


def generate_concat_nodes(
    num_tiles: int,
    tiles_y: int,
    tiles_x: int,
    slice_idx: int,
    output_name: str,
    prefix: str = ""
) -> tuple[list, list, list[str]]:
    """
    Generate Concat nodes for reassembling tile outputs.

    Returns: (nodes, initializers, input_names)
    """
    nodes = []
    initializers = []
    input_names = [f"tile_{slice_idx}_{i}_out" for i in range(num_tiles)]

    row_outputs = []
    for ty in range(tiles_y):
        row_inputs = [f"tile_{slice_idx}_{ty * tiles_x + tx}_out" for tx in range(tiles_x)]
        row_output = f"{prefix}row_{ty}" if prefix else f"row_{ty}"
        row_outputs.append(row_output)
        nodes.append(helper.make_node(
            "Concat",
            inputs=row_inputs,
            outputs=[row_output],
            axis=3
        ))

    nodes.append(helper.make_node(
        "Concat",
        inputs=row_outputs,
        outputs=[output_name],
        axis=2
    ))

    return nodes, initializers, input_names


def create_concat_slice(
    num_tiles: int,
    tiles_y: int,
    tiles_x: int,
    c_out: int,
    out_tile_h: int,
    out_tile_w: int,
    slice_idx: int,
    output_name: str,
    output_dir: Path,
    nested: bool = False
) -> dict:
    """
    Create a concat slice that reassembles tile outputs.

    Input: N tensors of [1, C_out, out_tile_h, out_tile_w]
    Output: [1, C_out, H', W']
    """
    nodes, initializers, input_names = generate_concat_nodes(
        num_tiles, tiles_y, tiles_x, slice_idx, output_name
    )

    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, c_out, out_tile_h, out_tile_w])
        for name in input_names
    ]

    full_h = tiles_y * out_tile_h
    full_w = tiles_x * out_tile_w
    Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, c_out, full_h, full_w])

    graph = helper.make_graph(nodes, f"concat_slice_{slice_idx}", inputs, [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    if nested:
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / "concat.onnx"
    else:
        concat_dir = output_dir / f"slice_{slice_idx}_concat"
        concat_dir.mkdir(parents=True, exist_ok=True)
        payload_dir = concat_dir / "payload"
        payload_dir.mkdir(exist_ok=True)
        onnx_path = payload_dir / f"slice_{slice_idx}_concat.onnx"

    onnx.save(model, str(onnx_path))

    return {
        "path": str(onnx_path.resolve()),
        "input_names": input_names,
        "output_name": output_name,
        "full_shape": [1, c_out, full_h, full_w],
    }


def tile_conv_slice(slice_path: Path, tile_size: int, output_path: Path) -> dict | None:
    """Legacy function - creates single tile model for backward compatibility."""
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

    weights, bias = None, None
    for init in m.graph.initializer:
        if init.name == conv_node.input[1]:
            weights = numpy_helper.to_array(init)
        if len(conv_node.input) > 2 and init.name == conv_node.input[2]:
            bias = numpy_helper.to_array(init)

    if weights is None:
        return None

    c_out = weights.shape[0]
    halo_h, halo_w = compute_halo([kh, kw], [dh, dw])
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

    nodes = [helper.make_node(
        "Conv", conv_inputs, ["conv_out"],
        kernel_shape=[kh, kw], strides=[sh, sw], pads=[0, 0, 0, 0], dilations=[dh, dw]
    )]

    non_conv_ops = [n for n in m.graph.node if n.op_type != "Conv"]
    if non_conv_ops:
        for init in m.graph.initializer:
            if init.name not in [conv_node.input[1]] + ([conv_node.input[2]] if len(conv_node.input) > 2 else []):
                initializers.append(init)

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


def autotile_slice(
    slice_idx: int,
    onnx_path: Path,
    tile_size: int,
    output_dir: Path,
    nested: bool = False,
    parallel: bool = False
) -> dict | None:
    """
    Create split + tile + concat slices for a single tileable slice.

    Returns metadata for all created slices.
    """
    m = onnx.load(str(onnx_path))
    if not is_tileable(m):
        return None

    inp = m.graph.input[0]
    out = m.graph.output[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4:
        return None

    _, c_in, h, w = dims
    if h <= tile_size or h != w:
        return None

    conv_params = get_conv_params(m)
    if not conv_params:
        return None

    kernel = conv_params['kernel']
    dilation = conv_params['dilation']
    sh, sw = conv_params['stride']

    halo_h, halo_w = compute_halo(kernel, dilation)
    min_tile = compute_min_tile_size(kernel, dilation)

    actual_tile_size = find_tile_size(h, tile_size, min_tile, stride=sh)
    if not actual_tile_size:
        return None

    tiles_y = h // actual_tile_size
    tiles_x = w // actual_tile_size
    num_tiles = tiles_y * tiles_x

    weights = None
    for init in m.graph.initializer:
        if init.name == conv_params['node'].input[1]:
            weights = numpy_helper.to_array(init)
            break
    if weights is None:
        return None
    c_out = weights.shape[0]

    out_tile_h = actual_tile_size // sh
    out_tile_w = actual_tile_size // sw

    input_name = inp.name
    output_name = out.name

    split_info = create_split_slice(
        input_name=input_name,
        c_in=c_in,
        h=h,
        w=w,
        tile_size=actual_tile_size,
        halo_h=halo_h,
        halo_w=halo_w,
        slice_idx=slice_idx,
        output_dir=output_dir,
        nested=nested
    )

    tile_info = create_tile_slice(
        slice_path=onnx_path,
        tile_size=actual_tile_size,
        slice_idx=slice_idx,
        output_dir=output_dir,
        nested=nested
    )
    if not tile_info:
        return None

    concat_info = create_concat_slice(
        num_tiles=num_tiles,
        tiles_y=tiles_y,
        tiles_x=tiles_x,
        c_out=c_out,
        out_tile_h=out_tile_h,
        out_tile_w=out_tile_w,
        slice_idx=slice_idx,
        output_name=output_name,
        output_dir=output_dir,
        nested=nested
    )

    return {
        "slice_idx": slice_idx,
        "original_onnx": str(onnx_path),
        "tile_size": actual_tile_size,
        "halo": [halo_h, halo_w],
        "tiles_y": tiles_y,
        "tiles_x": tiles_x,
        "num_tiles": num_tiles,
        "c_in": c_in,
        "c_out": c_out,
        "out_tile": [out_tile_h, out_tile_w],
        "split": split_info,
        "tile": tile_info,
        "concat": concat_info,
        "input_name": input_name,
        "output_name": output_name,
    }


def _check_tileable_slice(entry: Path, tile_size: int) -> tuple[int, Path] | None:
    if not entry.name.startswith("slice_") or not entry.is_dir():
        return None
    if "_split" in entry.name or "_tile_" in entry.name or "_concat" in entry.name:
        return None
    try:
        i = int(entry.name.split("_")[1])
    except (IndexError, ValueError):
        return None

    onnx_path = entry / "payload" / f"{entry.name}.onnx"
    if not onnx_path.exists():
        return None

    m = onnx.load(str(onnx_path))
    ops = {n.op_type for n in m.graph.node}
    num_inputs = len(m.graph.input)
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim] if m.graph.input else []
    print(f"Slice {i}: inputs={num_inputs}, ops={ops}, dims={dims}")

    if not is_tileable(m):
        non_elem = ops - {'Conv'} - ELEMENTWISE_OPS
        print(f"  -> Not tileable: num_inputs={num_inputs}, has_conv={'Conv' in ops}, non_elementwise={non_elem}")
        return None

    if len(dims) == 4:
        h, w = dims[2], dims[3]
        if h <= tile_size:
            print(f"  -> Spatial dim {h} <= tile_size {tile_size}, skipping")
            return None
        if h != w:
            print(f"  -> Non-square spatial dims {h}x{w}, skipping")
            return None

    return (i, onnx_path)


def autotile_slices(slices_dir: str | Path, tile_size: int = 16, parallel: bool = False) -> dict:
    """
    Scan slices directory and create tiled versions (split + tiles + concat) of Conv slices.
    """
    slices_dir = Path(slices_dir)
    tiled_dir = slices_dir / "tiled"
    tiled_dir.mkdir(exist_ok=True)

    tileable = []
    for entry in sorted(slices_dir.iterdir()):
        result = _check_tileable_slice(entry, tile_size)
        if result:
            tileable.append(result)

    tiled_info = {}
    if parallel and len(tileable) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        max_workers = min(len(tileable), multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(autotile_slice, i, onnx_path, tile_size, tiled_dir, False, True): i
                for i, onnx_path in tileable
            }
            for future in as_completed(futures):
                i = futures[future]
                info = future.result()
                if info:
                    tiled_info[i] = info
                    print(f"  -> Slice {i} tiled successfully: {info['num_tiles']} tiles")
    else:
        for i, onnx_path in tileable:
            info = autotile_slice(i, onnx_path, tile_size, tiled_dir, nested=False, parallel=parallel)
            if info:
                tiled_info[i] = info
                print(f"  -> Tiled successfully: {info['num_tiles']} tiles")

    if tiled_info:
        info_path = tiled_dir / "tiled_info.json"
        with open(info_path, "w") as f:
            json.dump({
                "slices": {str(k): v for k, v in tiled_info.items()},
                "tiled_indices": list(tiled_info.keys())
            }, f, indent=2)

    return tiled_info


def get_tiling_params(onnx_path: Path, max_conv_size: int = None, tile_size: int = None) -> dict | None:
    """
    Analyze a Conv slice and return tiling parameters if tileable.

    Args:
        onnx_path: Path to the ONNX slice
        max_conv_size: Maximum elements per tile (recommended). Tile size is calculated
                       dynamically based on channel count: tile_size = sqrt(max_conv_size / channels)
        tile_size: Fixed tile size (legacy). If both are provided, max_conv_size takes precedence.

    Returns None if not tileable or tiling not needed.
    """
    m = onnx.load(str(onnx_path))
    if not is_tileable(m):
        return None

    inp = m.graph.input[0]
    out = m.graph.output[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4:
        return None

    _, c_in, h, w = dims
    if h != w:
        return None

    conv_params = get_conv_params(m)
    if not conv_params:
        return None

    kernel = conv_params['kernel']
    dilation = conv_params['dilation']
    sh, sw = conv_params['stride']
    if sh == 0 or sw == 0:
        return None

    halo_h, halo_w = compute_halo(kernel, dilation)
    min_tile = compute_min_tile_size(kernel, dilation)

    if max_conv_size is not None:
        actual_tile_size, skip_reason = calculate_tile_size_from_max_elements(c_in, h, w, max_conv_size, min_tile)
        if actual_tile_size is None:
            if skip_reason == "already_fits":
                pass
            elif skip_reason == "min_tile_too_large":
                total_elements = c_in * h * w
                import math
                needed_tile = int(math.sqrt(max_conv_size / c_in))
                print(f"  WARNING: Conv {c_in}ch x {h}x{w} ({total_elements} elements) cannot be tiled to fit max_conv_size={max_conv_size}")
                print(f"           Needed tile_size={needed_tile} but min_tile={min_tile} (kernel constraint)")
            elif skip_reason == "no_divisor":
                print(f"  WARNING: Conv {c_in}ch x {h}x{w} - no valid tile size divides spatial dim")
            return None
    elif tile_size is not None:
        if h <= tile_size:
            return None
        actual_tile_size = find_tile_size(h, tile_size, min_tile, stride=sh)
        if not actual_tile_size:
            return None
    else:
        return None

    tiles_y = h // actual_tile_size
    tiles_x = w // actual_tile_size
    num_tiles = tiles_y * tiles_x

    weights = None
    for init in m.graph.initializer:
        if init.name == conv_params['node'].input[1]:
            weights = numpy_helper.to_array(init)
            break
    if weights is None:
        return None

    c_out = weights.shape[0]
    out_tile_h = actual_tile_size // sh
    out_tile_w = actual_tile_size // sw

    tile_elements = c_in * actual_tile_size * actual_tile_size
    total_elements = c_in * h * w

    return {
        "input_name": inp.name,
        "output_name": out.name,
        "c_in": c_in,
        "c_out": c_out,
        "h": h,
        "w": w,
        "tile_size": actual_tile_size,
        "halo": [halo_h, halo_w],
        "tiles_y": tiles_y,
        "tiles_x": tiles_x,
        "num_tiles": num_tiles,
        "out_tile": [out_tile_h, out_tile_w],
        "stride": [sh, sw],
        "tile_elements": tile_elements,
        "total_elements": total_elements,
    }


def create_standalone_bridge(
    nodes_list: list,
    initializers_list: list,
    inputs: list,
    outputs: list,
    name: str,
    output_path: Path
) -> str:
    """
    Create a standalone bridge ONNX with the given nodes.
    Used for edge cases like split-only or concat-only bridges.
    """
    graph = helper.make_graph(nodes_list, name, inputs, outputs, initializers_list)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return str(output_path)


def _create_tile_for_conv(args: tuple) -> tuple[int, dict | None]:
    """Helper for parallel tile creation."""
    conv_idx, conv_path, tile_size = args
    tiles_dir = Path(conv_path).parent / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    tile_info = create_tile_slice(
        slice_path=Path(conv_path),
        tile_size=tile_size,
        slice_idx=conv_idx,
        output_dir=tiles_dir,
        nested=True
    )
    return (conv_idx, tile_info)


def apply_tiling_to_slices(slices_dir: str | Path, max_conv_size: int = None, tile_size: int = None, parallel: bool = False) -> dict:
    """
    Apply tiling transform to slices, injecting split/concat into bridge slices.

    This implements the two-phase approach:
    1. Identify tileable Conv slices
    2. Create tile.onnx for each tileable Conv (can be parallelized)
    3. Inject split ops into preceding bridges, concat ops into following bridges

    Args:
        slices_dir: Directory containing sliced model
        max_conv_size: Maximum elements per tile (recommended). Tile size is calculated
                       dynamically per-slice based on channel count.
                       Formula: tile_size = sqrt(max_conv_size / channels)
        tile_size: Fixed tile size for all slices (legacy). Ignored if max_conv_size is set.
        parallel: If True, parallelize tile creation

    Returns: dict with tiling metadata
    """
    slices_dir = Path(slices_dir)
    metadata_path = slices_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"No metadata.json found in {slices_dir}")
        return {}

    with open(metadata_path) as f:
        metadata = json.load(f)

    slices_data = metadata.get("slices", [])
    if not slices_data:
        print("No slices found in metadata")
        return {}

    if max_conv_size:
        print(f"Using dynamic tiling with max_conv_size={max_conv_size} elements per tile")
    elif tile_size:
        print(f"Using fixed tile_size={tile_size} for all slices")
    else:
        print("No tiling parameters specified")
        return {}

    slice_info = []
    for idx, _ in enumerate(slices_data):
        slice_dir = slices_dir / f"slice_{idx}"
        onnx_path = slice_dir / "payload" / f"slice_{idx}.onnx"

        if not onnx_path.exists():
            slice_info.append({"idx": idx, "type": "missing", "path": None})
            continue

        tiling_params = get_tiling_params(onnx_path, max_conv_size=max_conv_size, tile_size=tile_size)
        if tiling_params:
            slice_info.append({
                "idx": idx,
                "type": "conv",
                "path": onnx_path,
                "tiling": tiling_params
            })
        else:
            slice_info.append({
                "idx": idx,
                "type": "bridge",
                "path": onnx_path
            })

    conv_slices = [(i, info) for i, info in enumerate(slice_info) if info["type"] == "conv"]
    if not conv_slices:
        print("No tileable Conv slices found")
        return {}

    tile_infos = {}
    tile_args = [(info["idx"], str(info["path"]), info["tiling"]["tile_size"]) for _, info in conv_slices]

    if parallel and len(tile_args) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        max_workers = min(len(tile_args), multiprocessing.cpu_count())
        print(f"Creating {len(tile_args)} tile models in parallel (workers={max_workers})...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_create_tile_for_conv, args): args[0] for args in tile_args}
            for future in as_completed(futures):
                conv_idx, tile_info = future.result()
                if tile_info:
                    tile_infos[conv_idx] = tile_info
                    print(f"  Created tile for slice {conv_idx}")
                else:
                    print(f"  Failed to create tile for slice {conv_idx}")
    else:
        for args in tile_args:
            conv_idx, tile_info = _create_tile_for_conv(args)
            if tile_info:
                tile_infos[conv_idx] = tile_info
            else:
                print(f"  Failed to create tile for slice {conv_idx}")

    tiled_results = {}

    for i, info in enumerate(slice_info):
        if info["type"] != "conv":
            continue

        conv_idx = info["idx"]
        if conv_idx not in tile_infos:
            continue

        conv_path = info["path"]
        tiling = info["tiling"]
        tile_info = tile_infos[conv_idx]
        tiles_dir = conv_path.parent / "tiles"

        c_in = tiling["c_in"]
        tile_sz = tiling["tile_size"]
        tile_elements = tiling.get("tile_elements", c_in * tile_sz * tile_sz)
        total_elements = tiling.get("total_elements", c_in * tiling["h"] * tiling["w"])
        print(f"Tiling Conv slice {conv_idx}: {c_in}ch x {tiling['h']}x{tiling['w']} -> tile_size={tile_sz} ({tiling['num_tiles']} tiles, {tile_elements} elements/tile)")

        prev_info = slice_info[i - 1] if i > 0 else None
        next_info = slice_info[i + 1] if i < len(slice_info) - 1 else None

        halo_h, halo_w = tiling["halo"]
        out_tile_h, out_tile_w = tile_info["conv_out"]

        split_path = None
        concat_path = None

        split_path = None
        concat_path = None

        if prev_info and prev_info["type"] == "bridge":
            print(f"  Appending split to bridge slice {prev_info['idx']}")
            append_split_to_onnx(
                onnx_path=prev_info["path"],
                split_input_name=tiling["input_name"],
                c_in=tiling["c_in"],
                h=tiling["h"],
                w=tiling["w"],
                tile_size=tiling["tile_size"],
                halo_h=halo_h,
                halo_w=halo_w,
                slice_idx=conv_idx
            )
            slices_data[prev_info["idx"]]["runtime_only"] = True
            split_path = prev_info["path"]
        elif prev_info and prev_info["type"] == "conv":
            print(f"  Creating concat+split bridge between Conv {prev_info['idx']} and Conv {conv_idx}")
            prev_tiling = prev_info["tiling"]
            prev_tile_info = tiled_results[prev_info["idx"]]["tile_info"]
            prev_out_tile_h, prev_out_tile_w = prev_tile_info["conv_out"]

            glue_dir = slices_dir / f"slice_{prev_info['idx']}_{conv_idx}_glue"
            glue_dir.mkdir(exist_ok=True)
            payload_dir = glue_dir / "payload"
            payload_dir.mkdir(exist_ok=True)
            glue_path = payload_dir / f"glue_{prev_info['idx']}_{conv_idx}.onnx"

            concat_nodes, concat_init, concat_inputs = generate_concat_nodes(
                prev_tiling["num_tiles"], prev_tiling["tiles_y"], prev_tiling["tiles_x"],
                prev_info["idx"], tiling["input_name"], prefix=f"concat_{prev_info['idx']}_"
            )
            split_nodes, split_init, split_outputs, split_meta = generate_split_nodes(
                tiling["input_name"], tiling["c_in"], tiling["h"], tiling["w"],
                tiling["tile_size"], halo_h, halo_w, conv_idx, prefix=f"split_{conv_idx}_"
            )

            all_nodes = concat_nodes + split_nodes
            all_init = concat_init + split_init

            tile_with_halo_h, tile_with_halo_w = split_meta["tile_with_halo"]
            inputs = [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, prev_tiling["c_out"], prev_out_tile_h, prev_out_tile_w])
                for name in concat_inputs
            ]
            outputs = [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, tiling["c_in"], tile_with_halo_h, tile_with_halo_w])
                for name in split_outputs
            ]

            create_standalone_bridge(all_nodes, all_init, inputs, outputs, f"glue_{prev_info['idx']}_{conv_idx}", glue_path)
            split_path = glue_path
            if "tiling" in slices_data[prev_info["idx"]]:
                slices_data[prev_info["idx"]]["tiling"]["concat"] = {"path": str(glue_path), "input_names": concat_inputs}
        else:
            print(f"  Creating split-only bridge before Conv {conv_idx}")
            split_dir = slices_dir / f"slice_{conv_idx}_split"
            split_dir.mkdir(exist_ok=True)

            split_result = create_split_slice(
                input_name=tiling["input_name"],
                c_in=tiling["c_in"],
                h=tiling["h"],
                w=tiling["w"],
                tile_size=tiling["tile_size"],
                halo_h=halo_h,
                halo_w=halo_w,
                slice_idx=conv_idx,
                output_dir=split_dir,
                nested=True
            )
            split_path = Path(split_result["path"])

        concat_input_names = None
        if next_info and next_info["type"] == "bridge":
            print(f"  Prepending concat to bridge slice {next_info['idx']}")
            concat_input_names = prepend_concat_to_onnx(
                onnx_path=next_info["path"],
                concat_output_name=tiling["output_name"],
                num_tiles=tiling["num_tiles"],
                tiles_y=tiling["tiles_y"],
                tiles_x=tiling["tiles_x"],
                c_out=tiling["c_out"],
                out_tile_h=out_tile_h,
                out_tile_w=out_tile_w,
                slice_idx=conv_idx
            )
            slices_data[next_info["idx"]]["runtime_only"] = True
            concat_path = next_info["path"]
        elif next_info is None or next_info["type"] != "conv":
            print(f"  Creating concat-only bridge after Conv {conv_idx}")
            concat_dir = slices_dir / f"slice_{conv_idx}_concat"
            concat_dir.mkdir(exist_ok=True)

            concat_result = create_concat_slice(
                num_tiles=tiling["num_tiles"],
                tiles_y=tiling["tiles_y"],
                tiles_x=tiling["tiles_x"],
                c_out=tiling["c_out"],
                out_tile_h=out_tile_h,
                out_tile_w=out_tile_w,
                slice_idx=conv_idx,
                output_name=tiling["output_name"],
                output_dir=concat_dir,
                nested=True
            )
            concat_path = Path(concat_result["path"])
            concat_input_names = concat_result["input_names"]

        tiled_results[conv_idx] = {
            "tile_path": str(tiles_dir / "tile.onnx"),
            "tiling": tiling,
            "tile_info": tile_info,
        }

        tiling_metadata = {
            "slice_idx": conv_idx,
            "tile_size": tiling["tile_size"],
            "num_tiles": tiling["num_tiles"],
            "tiles_y": tiling["tiles_y"],
            "tiles_x": tiling["tiles_x"],
            "halo": tiling["halo"],
            "out_tile": tiling["out_tile"],
            "c_out": tiling["c_out"],
            "tile": {
                "path": f"slice_{conv_idx}/payload/tiles/tile.onnx",
                "conv_out": tile_info["conv_out"],
            },
        }
        if split_path:
            try:
                split_rel = Path(split_path).relative_to(slices_dir)
            except ValueError:
                split_rel = split_path
            tiling_metadata["split"] = {"path": str(split_rel)}
        if concat_path:
            try:
                concat_rel = Path(concat_path).relative_to(slices_dir)
            except ValueError:
                concat_rel = concat_path
            concat_meta = {"path": str(concat_rel)}
            if concat_input_names:
                concat_meta["input_names"] = concat_input_names
            tiling_metadata["concat"] = concat_meta
        slices_data[conv_idx]["tiling"] = tiling_metadata

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return tiled_results
