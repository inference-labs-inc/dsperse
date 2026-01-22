"""
Autotiler for Conv slices.

Tiling breaks large convolution operations into smaller tiles that can be processed
independently, reducing memory/circuit size for ZK proofs.

Architecture:
  1. SPLIT (Python): Pad input, extract N tiles with halo
  2. TILE (ONNX): Each tile processed through Conv + elementwise ops
  3. CONCAT (Python): Reassemble N tile outputs into full spatial output

The split/concat operations are done in pure Python at runtime, not as ONNX models.
Only the tile model is an ONNX file.
"""
import json
from pathlib import Path

import onnx
from onnx import helper, TensorProto, numpy_helper

import numpy as np

from dsperse.src.metadata.schema import TilingInfo, TileInfo, ChannelSplitInfo, ChannelGroupInfo
from dsperse.src.utils.utils import save_onnx_model

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
    min_tile: int = 7,
    stride: int = 1
) -> tuple[int | None, str | None]:
    import math

    total_elements = channels * spatial_h * spatial_w
    if total_elements <= max_conv_size:
        return None, "already_fits"

    max_tile_from_constraint = int(math.sqrt(max_conv_size / channels))

    if max_tile_from_constraint < min_tile:
        return None, "min_tile_too_large"

    target_tile = min(max_tile_from_constraint, spatial_h, spatial_w)

    tile_size = find_tile_size(spatial_h, target_tile, min_tile, stride=stride)
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
                'group': attrs["group"].i if "group" in attrs else 1,
            }
    return None


def is_channel_splittable(model: onnx.ModelProto) -> bool:
    if len(model.graph.input) > 1:
        return False
    conv_count = sum(1 for n in model.graph.node if n.op_type == "Conv")
    if conv_count != 1:
        return False
    conv_params = get_conv_params(model)
    if not conv_params:
        return False
    if conv_params['group'] != 1:
        return False
    ops = {n.op_type for n in model.graph.node}
    return (ops - {'Conv'}).issubset(ELEMENTWISE_OPS)


def calculate_channel_split_params(
    c_in: int,
    c_out: int,
    spatial_h: int,
    spatial_w: int,
    max_conv_size: int,
    min_tile: int = 7,
) -> tuple[int | None, int | None, str | None]:
    import math

    valid_tiles = sorted([t for t in range(min_tile, spatial_h + 1) if spatial_h % t == 0])
    if not valid_tiles:
        return None, None, "no_valid_tile_for_spatial_dims"

    for tile_candidate in valid_tiles:
        max_channels_for_tile = max_conv_size // (tile_candidate * tile_candidate)
        if max_channels_for_tile >= 1 and max_channels_for_tile < c_in:
            num_groups = math.ceil(c_in / max_channels_for_tile)
            if num_groups > 1:
                channels_per_group = math.ceil(c_in / num_groups)
                while channels_per_group * (num_groups - 1) >= c_in and num_groups > 1:
                    num_groups -= 1
                    channels_per_group = math.ceil(c_in / num_groups)
                if num_groups > 1:
                    return num_groups, channels_per_group, None

    return None, None, "channel_split_not_beneficial"


def create_channel_group_slice(
    slice_path: Path,
    group_idx: int,
    c_start: int,
    c_end: int,
    slice_idx: int,
    output_dir: Path,
) -> dict | None:
    m = onnx.load(str(slice_path))
    conv_params = get_conv_params(m)
    if not conv_params:
        return None

    conv_node = conv_params['node']
    kh, kw = conv_params['kernel']
    sh, sw = conv_params['stride']
    dh, dw = conv_params['dilation']
    pads = conv_params['pads']

    orig_input = m.graph.input[0]
    orig_dims = [d.dim_value for d in orig_input.type.tensor_type.shape.dim]
    h_in, w_in = orig_dims[2], orig_dims[3]

    weights, bias = None, None
    for init in m.graph.initializer:
        if init.name == conv_node.input[1]:
            weights = numpy_helper.to_array(init)
        if len(conv_node.input) > 2 and init.name == conv_node.input[2]:
            bias = numpy_helper.to_array(init)

    if weights is None:
        return None

    sliced_weights = weights[:, c_start:c_end, :, :]

    c_out = weights.shape[0]
    c_group = c_end - c_start

    effective_kh = (kh - 1) * dh + 1
    effective_kw = (kw - 1) * dw + 1
    h_out = (h_in + pads[0] + pads[2] - effective_kh) // sh + 1
    w_out = (w_in + pads[1] + pads[3] - effective_kw) // sw + 1

    input_name = f"group_{group_idx}_in"
    output_name = f"group_{group_idx}_out"

    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, c_group, h_in, w_in])
    Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, c_out, h_out, w_out])

    W = helper.make_tensor("W", TensorProto.FLOAT, sliced_weights.shape, sliced_weights.flatten().tolist())
    initializers = [W]
    conv_inputs = [input_name, "W"]

    nodes = [helper.make_node(
        "Conv", conv_inputs, [output_name],
        kernel_shape=[kh, kw], strides=[sh, sw], pads=pads, dilations=[dh, dw]
    )]

    graph = helper.make_graph(nodes, f"channel_group_{slice_idx}_{group_idx}", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    groups_dir = output_dir / "channel_groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = groups_dir / f"group_{group_idx}.onnx"

    save_onnx_model(model, onnx_path)

    return {
        "path": str(onnx_path.resolve()),
        "group_idx": group_idx,
        "c_start": c_start,
        "c_end": c_end,
        "output_shape": [1, c_out, h_out, w_out],
    }


def apply_channel_splitting_to_slice(
    slice_path: Path,
    split_params: dict,
    slice_idx: int,
    output_dir: Path,
) -> ChannelSplitInfo | None:
    c_in = split_params["c_in"]
    c_out = split_params["c_out"]
    num_groups = split_params["num_groups"]
    channels_per_group = split_params["channels_per_group"]

    m = onnx.load(str(slice_path))
    conv_params = get_conv_params(m)
    if not conv_params:
        return None

    groups = []
    for g in range(num_groups):
        c_start = g * channels_per_group
        c_end = min((g + 1) * channels_per_group, c_in)

        group_info = create_channel_group_slice(
            slice_path=slice_path,
            group_idx=g,
            c_start=c_start,
            c_end=c_end,
            slice_idx=slice_idx,
            output_dir=output_dir,
        )

        if group_info is None:
            return None

        groups.append(ChannelGroupInfo(
            group_idx=g,
            c_start=c_start,
            c_end=c_end,
            path=f"slice_{slice_idx}/payload/channel_groups/group_{g}.onnx",
        ))

    bias_path = None
    conv_node = conv_params['node']
    for init in m.graph.initializer:
        if len(conv_node.input) > 2 and init.name == conv_node.input[2]:
            bias = numpy_helper.to_array(init)
            bias_dir = output_dir / "channel_groups"
            bias_dir.mkdir(parents=True, exist_ok=True)
            bias_file = bias_dir / "bias.npy"
            np.save(str(bias_file), bias)
            bias_path = f"slice_{slice_idx}/payload/channel_groups/bias.npy"
            break

    return ChannelSplitInfo(
        slice_idx=slice_idx,
        c_in=c_in,
        c_out=c_out,
        num_groups=num_groups,
        channels_per_group=channels_per_group,
        input_name=split_params["input_name"],
        output_name=split_params["output_name"],
        h=split_params["h"],
        w=split_params["w"],
        groups=groups,
        bias_path=bias_path,
    )


def create_tile_slice(
    slice_path: Path,
    tile_size: int,
    slice_idx: int,
    output_dir: Path,
) -> dict | None:
    """
    Create a single tile processing ONNX model.

    Input: [1, C, tile_size + 2*halo, tile_size + 2*halo]
    Output: [1, C_out, out_tile_h, out_tile_w]
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

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = tiles_dir / "tile.onnx"

    save_onnx_model(model, onnx_path)

    return {
        "path": str(onnx_path.resolve()),
        "input_name": input_name,
        "output_name": output_name,
        "conv_out": [conv_out_h, conv_out_w],
    }


def get_tiling_params(onnx_path: Path, max_conv_size: int | None = None, tile_size: int | None = None) -> dict | None:
    """
    Analyze a Conv slice and return tiling parameters if tileable.
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

    weights = None
    for init in m.graph.initializer:
        if init.name == conv_params['node'].input[1]:
            weights = numpy_helper.to_array(init)
            break
    if weights is None:
        return None
    c_out = weights.shape[0]

    if max_conv_size is not None:
        actual_tile_size, skip_reason = calculate_tile_size_from_max_elements(c_in, h, w, max_conv_size, min_tile, stride=sh)
        if actual_tile_size is None:
            if skip_reason in ("min_tile_too_large", "no_divisor") and is_channel_splittable(m):
                num_groups, cpg, split_err = calculate_channel_split_params(c_in, c_out, h, w, max_conv_size, min_tile)
                if num_groups is not None and num_groups > 1:
                    return {
                        "needs_channel_split": True,
                        "c_in": c_in,
                        "c_out": c_out,
                        "num_groups": num_groups,
                        "channels_per_group": cpg,
                        "h": h,
                        "w": w,
                        "input_name": inp.name,
                        "output_name": out.name,
                    }
            return None
    elif tile_size is not None:
        if h <= tile_size:
            return None
        actual_tile_size = find_tile_size(h, tile_size, min_tile, stride=sh)
        if not actual_tile_size:
            return None
    else:
        return None

    if h % actual_tile_size != 0 or w % actual_tile_size != 0:
        return None

    tiles_y = h // actual_tile_size
    tiles_x = w // actual_tile_size
    num_tiles = tiles_y * tiles_x

    if num_tiles < 2:
        return None

    out_tile_h = actual_tile_size // sh
    out_tile_w = actual_tile_size // sw

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
    }


def apply_tiling_to_slices(slices_dir: str | Path, max_conv_size: int | None = None, tile_size: int | None = None) -> dict:
    """
    Apply tiling to Conv slices. Creates tile ONNX models and updates metadata.

    Split/concat operations are handled in pure Python at runtime, not as ONNX models.
    Only the tile model is created as an ONNX file.
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

    tiled_results = {}

    for idx, _slice_meta in enumerate(slices_data):
        slice_dir = slices_dir / f"slice_{idx}"
        onnx_path = slice_dir / "payload" / f"slice_{idx}.onnx"

        if not onnx_path.exists():
            continue

        tiling_params = get_tiling_params(onnx_path, max_conv_size=max_conv_size, tile_size=tile_size)
        if not tiling_params:
            continue

        if tiling_params.get("needs_channel_split"):
            c_in = tiling_params["c_in"]
            num_groups = tiling_params["num_groups"]
            cpg = tiling_params["channels_per_group"]
            print(f"Channel splitting Conv slice {idx}: {c_in}ch x {tiling_params['h']}x{tiling_params['w']} -> {num_groups} groups ({cpg} channels/group)")

            channel_split_info = apply_channel_splitting_to_slice(
                slice_path=onnx_path,
                split_params=tiling_params,
                slice_idx=idx,
                output_dir=slice_dir / "payload",
            )
            if not channel_split_info:
                print(f"  Failed to create channel groups for slice {idx}")
                continue

            tiled_results[idx] = {
                "channel_split": tiling_params,
            }
            slices_data[idx]["channel_split"] = channel_split_info.to_dict()

            slice_meta_path = slice_dir / "metadata.json"
            if slice_meta_path.exists():
                with open(slice_meta_path, "r") as f:
                    slice_meta = json.load(f)
                slice_slices = slice_meta.get("slices", [])
                if slice_slices:
                    slice_slices[0]["channel_split"] = channel_split_info.to_dict()
                with open(slice_meta_path, "w") as f:
                    json.dump(slice_meta, f, indent=2)
        else:
            c_in = tiling_params["c_in"]
            tile_sz = tiling_params["tile_size"]
            print(f"Tiling Conv slice {idx}: {c_in}ch x {tiling_params['h']}x{tiling_params['w']} -> tile_size={tile_sz} ({tiling_params['num_tiles']} tiles)")

            tile_info = create_tile_slice(
                slice_path=onnx_path,
                tile_size=tile_sz,
                slice_idx=idx,
                output_dir=slice_dir / "payload",
            )
            if not tile_info:
                print(f"  Failed to create tile model for slice {idx}")
                continue

            tiled_results[idx] = {
                "tiling": tiling_params,
                "tile_info": tile_info,
            }

            tiling_metadata = TilingInfo(
                slice_idx=idx,
                tile_size=tile_sz,
                num_tiles=tiling_params["num_tiles"],
                tiles_y=tiling_params["tiles_y"],
                tiles_x=tiling_params["tiles_x"],
                halo=tuple(tiling_params["halo"]),
                out_tile=tuple(tiling_params["out_tile"]),
                stride=tuple(tiling_params["stride"]),
                c_in=tiling_params["c_in"],
                c_out=tiling_params["c_out"],
                input_name=tiling_params["input_name"],
                output_name=tiling_params["output_name"],
                tile=TileInfo(
                    path=f"slice_{idx}/payload/tiles/tile.onnx",
                    conv_out=tuple(tile_info["conv_out"]),
                ),
            )
            slices_data[idx]["tiling"] = tiling_metadata.to_dict()

            slice_meta_path = slice_dir / "metadata.json"
            if slice_meta_path.exists():
                with open(slice_meta_path, "r") as f:
                    slice_meta = json.load(f)
                slice_slices = slice_meta.get("slices", [])
                if slice_slices:
                    slice_slices[0]["tiling"] = tiling_metadata.to_dict()
                with open(slice_meta_path, "w") as f:
                    json.dump(slice_meta, f, indent=2)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Tiled {len(tiled_results)} Conv slices")
    return tiled_results
