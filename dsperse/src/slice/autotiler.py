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


def compute_halo(kernel: list[int], dilation: list[int]) -> tuple[int, int]:
    effective_kh = (kernel[0] - 1) * dilation[0] + 1
    effective_kw = (kernel[1] - 1) * dilation[1] + 1
    return effective_kh // 2, effective_kw // 2


def compute_min_tile_size(kernel: list[int], dilation: list[int]) -> int:
    effective_kh = (kernel[0] - 1) * dilation[0] + 1
    effective_kw = (kernel[1] - 1) * dilation[1] + 1
    return max(effective_kh, effective_kw) + 1


def find_tile_size(spatial_dim: int, target: int, min_tile: int = 7) -> int | None:
    if min_tile <= target < spatial_dim:
        for tile in range(target, min_tile - 1, -1):
            if spatial_dim % tile == 0:
                return tile
    return None


def is_tileable(model: onnx.ModelProto) -> bool:
    if len(model.graph.input) > 1:
        return False
    op_types = [n.op_type for n in model.graph.node]
    conv_count = op_types.count('Conv')
    if conv_count != 1:
        return False
    ops = set(op_types)
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
    tiles_y = h // tile_size
    tiles_x = w // tile_size
    num_tiles = tiles_y * tiles_x
    tile_with_halo_h = tile_size + 2 * halo_h
    tile_with_halo_w = tile_size + 2 * halo_w

    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, c_in, h, w])

    nodes = []
    outputs = []
    initializers = []

    pads_val = [0, 0, halo_h, halo_w, 0, 0, halo_h, halo_w]
    pads_tensor = helper.make_tensor("pads", TensorProto.INT64, [8], pads_val)
    initializers.append(pads_tensor)

    constant_value = helper.make_tensor("constant_value", TensorProto.FLOAT, [], [0.0])
    initializers.append(constant_value)

    nodes.append(helper.make_node(
        "Pad",
        inputs=[input_name, "pads", "constant_value"],
        outputs=["padded"],
        mode="constant"
    ))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_idx = ty * tiles_x + tx
            output_name = f"tile_{slice_idx}_{tile_idx}_in"

            y_start = ty * tile_size
            x_start = tx * tile_size
            y_end = y_start + tile_with_halo_h
            x_end = x_start + tile_with_halo_w

            starts_name = f"starts_{tile_idx}"
            ends_name = f"ends_{tile_idx}"
            axes_name = f"axes_{tile_idx}"

            initializers.append(helper.make_tensor(starts_name, TensorProto.INT64, [4], [0, 0, y_start, x_start]))
            initializers.append(helper.make_tensor(ends_name, TensorProto.INT64, [4], [1, c_in, y_end, x_end]))
            initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [4], [0, 1, 2, 3]))

            nodes.append(helper.make_node(
                "Slice",
                inputs=["padded", starts_name, ends_name, axes_name],
                outputs=[output_name]
            ))

            outputs.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, [1, c_in, tile_with_halo_h, tile_with_halo_w]
            ))

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
        "output_names": [f"tile_{slice_idx}_{i}_in" for i in range(num_tiles)],
        "tiles_y": tiles_y,
        "tiles_x": tiles_x,
        "num_tiles": num_tiles,
    }


def create_tile_slice(
    slice_path: Path,
    tile_size: int,
    tile_idx: int,
    slice_idx: int,
    output_dir: Path,
    nested: bool = False
) -> dict | None:
    """
    Create a single tile processing slice.

    Input: [1, C, tile_size + 2*halo, tile_size + 2*halo]
    Output: [1, C_out, out_tile, out_tile]
    """
    m = onnx.load(slice_path)
    orig_input = m.graph.input[0]
    orig_output = m.graph.output[0]
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

    input_name = f"tile_{slice_idx}_{tile_idx}_in"
    output_name = f"tile_{slice_idx}_{tile_idx}_out"

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

    graph = helper.make_graph(nodes, f"tile_{slice_idx}_{tile_idx}", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    if nested:
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / f"tile_{tile_idx}.onnx"
    else:
        tile_dir = output_dir / f"slice_{slice_idx}_tile_{tile_idx}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        payload_dir = tile_dir / "payload"
        payload_dir.mkdir(exist_ok=True)
        onnx_path = payload_dir / f"slice_{slice_idx}_tile_{tile_idx}.onnx"

    onnx.save(model, str(onnx_path))

    return {
        "path": str(onnx_path.resolve()),
        "input_name": input_name,
        "output_name": output_name,
        "tile_idx": tile_idx,
        "conv_out": [conv_out_h, conv_out_w],
        "out_tile": [out_tile_h, out_tile_w],
    }


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
    inputs = []
    nodes = []
    initializers = []

    for i in range(num_tiles):
        input_name = f"tile_{slice_idx}_{i}_out"
        inputs.append(helper.make_tensor_value_info(
            input_name, TensorProto.FLOAT, [1, c_out, out_tile_h, out_tile_w]
        ))

    row_outputs = []
    for ty in range(tiles_y):
        row_inputs = [f"tile_{slice_idx}_{ty * tiles_x + tx}_out" for tx in range(tiles_x)]
        row_output = f"row_{ty}"
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
        "input_names": [f"tile_{slice_idx}_{i}_out" for i in range(num_tiles)],
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

    actual_tile_size = find_tile_size(h, tile_size, min_tile)
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

    if parallel and num_tiles > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        max_workers = min(num_tiles, multiprocessing.cpu_count())
        tile_infos = [None] * num_tiles
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(create_tile_slice, onnx_path, actual_tile_size, tile_idx, slice_idx, output_dir, nested): tile_idx
                for tile_idx in range(num_tiles)
            }
            for future in as_completed(futures):
                tile_idx = futures[future]
                tile_info = future.result()
                if tile_info:
                    tile_infos[tile_idx] = tile_info
        tile_infos = [t for t in tile_infos if t is not None]
    else:
        tile_infos = []
        for tile_idx in range(num_tiles):
            tile_info = create_tile_slice(
                slice_path=onnx_path,
                tile_size=actual_tile_size,
                tile_idx=tile_idx,
                slice_idx=slice_idx,
                output_dir=output_dir,
                nested=nested
            )
            if tile_info:
                tile_infos.append(tile_info)

    if len(tile_infos) != num_tiles:
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
        "tiles": tile_infos,
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
