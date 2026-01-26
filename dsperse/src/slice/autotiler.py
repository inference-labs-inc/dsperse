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
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

from dsperse.src.analyzers.schema import TilingInfo, TileInfo, ChannelSplitInfo, ChannelGroupInfo
from dsperse.src.utils.utils import Utils

ELEMENTWISE_OPS = {
    'Sigmoid', 'Mul', 'Add', 'Sub', 'Div', 'Relu', 'LeakyRelu', 'PRelu',
    'Tanh', 'Clip', 'Neg', 'Abs', 'Sqrt', 'Exp', 'Log', 'Pow', 'Sin', 'Cos'
}

class Autotiler:
    """
    Automated tiling and channel splitting for ONNX Conv layers.
    
    Provides utilities for breaking large convolutions into manageable pieces
    to reduce memory footprint and circuit size in ZK-proof contexts.
    """

    # =========================================================================
    # Section 1: Halo & Tiling Math
    # =========================================================================

    @staticmethod
    def compute_halo_size(kernel: list[int], dilation: list[int]) -> tuple[int, int]:
        """Calculate the halo (overlap) size required for a convolution."""
        effective_kh = (kernel[0] - 1) * dilation[0] + 1
        effective_kw = (kernel[1] - 1) * dilation[1] + 1
        return effective_kh // 2, effective_kw // 2

    @staticmethod
    def compute_min_spatial_tile(kernel: list[int], dilation: list[int]) -> int:
        """Calculate the minimum viable spatial tile size for a kernel."""
        effective_kh = (kernel[0] - 1) * dilation[0] + 1
        effective_kw = (kernel[1] - 1) * dilation[1] + 1
        return max(effective_kh, effective_kw) + 1

    # =========================================================================
    # Section 2: Tile Size Calculation
    # =========================================================================

    @staticmethod
    def find_optimal_tile_size(spatial_dim: int, target: int, min_tile: int = 7, stride: int = 1) -> int | None:
        """Find the largest divisor of spatial_dim that is <= target and >= min_tile."""
        if min_tile <= target < spatial_dim:
            for tile in range(target, min_tile - 1, -1):
                if spatial_dim % tile == 0 and tile % stride == 0:
                    return tile
        return None

    @staticmethod
    def calculate_spatial_tile_config(
        channels: int,
        spatial_h: int,
        spatial_w: int,
        tile_size: int,
        min_tile: int = 7,
        stride: int = 1
    ) -> tuple[int | None, str | None]:
        """Determine optimal spatial tile dimension based on element count constraint."""
        total_elements = channels * spatial_h * spatial_w
        if total_elements <= tile_size:
            return None, "already_fits"

        max_tile_from_constraint = int(math.sqrt(tile_size / channels))

        if max_tile_from_constraint < min_tile:
            return None, "min_tile_too_large"

        target_tile = min(max_tile_from_constraint, spatial_h, spatial_w)

        actual_spatial_tile_size = Autotiler.find_optimal_tile_size(spatial_h, target_tile, min_tile, stride=stride)
        if actual_spatial_tile_size is None:
            return None, "no_divisor"

        return actual_spatial_tile_size, None

    @staticmethod
    def calculate_channel_split_config(
        c_in: int,
        c_out: int,
        spatial_h: int,
        spatial_w: int,
        tile_size: int,
        min_tile: int = 7,
    ) -> tuple[int | None, int | None, str | None]:
        """Calculate parameters for splitting a Conv by input channels."""
        valid_tiles = sorted([t for t in range(min_tile, spatial_h + 1) if spatial_h % t == 0])
        if not valid_tiles:
            return None, None, "no_valid_tile_for_spatial_dims"

        for tile_candidate in valid_tiles:
            max_channels_for_tile = tile_size // (tile_candidate * tile_candidate)
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

    # =========================================================================
    # Section 3: Model Analysis & Detection
    # =========================================================================

    @staticmethod
    def get_conv_params(model: onnx.ModelProto) -> dict | None:
        """Extract parameters from the first Conv node found in the model."""
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

    @staticmethod
    def _is_standard_conv_slice(model: onnx.ModelProto) -> dict | None:
        """Common validation for tileable/splittable Conv slices."""
        if len(model.graph.input) > 1:
            return None
        conv_count = sum(1 for n in model.graph.node if n.op_type == "Conv")
        if conv_count != 1:
            return None
        conv_params = Autotiler.get_conv_params(model)
        if not conv_params:
            return None
        ops = {n.op_type for n in model.graph.node}
        if not (ops - {'Conv'}).issubset(ELEMENTWISE_OPS):
            return None
        return conv_params

    @staticmethod
    def is_tileable(model: onnx.ModelProto) -> bool:
        """Check if a model slice can be spatially tiled."""
        conv_params = Autotiler._is_standard_conv_slice(model)
        if not conv_params:
            return False
        
        kh, kw = conv_params['kernel']
        if kh % 2 == 0 or kw % 2 == 0:
            return False
        pads = conv_params['pads']
        dilation = conv_params['dilation']
        halo_h, halo_w = Autotiler.compute_halo_size([kh, kw], dilation)
        if pads != [halo_h, halo_w, halo_h, halo_w]:
            return False
            
        return True

    @staticmethod
    def is_channel_splittable(model: onnx.ModelProto) -> bool:
        """Check if a model slice can be split by input channels."""
        conv_params = Autotiler._is_standard_conv_slice(model)
        if not conv_params:
            return False
        if conv_params['group'] != 1:
            return False
        return True

    @staticmethod
    def detect_tiling_needs(onnx_path: Path, tile_size: int | None = None) -> dict | None:
        """Analyze a Conv slice and return tiling or splitting parameters if applicable."""
        m = onnx.load(str(onnx_path))
        if not Autotiler.is_tileable(m):
            return None

        # Section 1: Extract model dimensions and basic validation
        dims = Autotiler._get_model_dimensions(m)
        if not dims:
            return None
        inp_name, out_name, c_in, h, w = dims

        # Section 2: Extract convolution parameters
        conv_params = Autotiler.get_conv_params(m)
        if not conv_params:
            return None

        sh, sw = conv_params['stride']
        if sh == 0 or sw == 0:
            return None

        # Section 3: Weight and output channel analysis
        weights, _ = Autotiler._get_weights_and_bias(m, conv_params['node'])
        if weights is None:
            return None
        c_out = weights.shape[0]

        # Section 4: Determine configuration (Tiling or Channel Splitting)
        if tile_size is None:
            return None

        min_tile = Autotiler.compute_min_spatial_tile(conv_params['kernel'], conv_params['dilation'])
        
        # Try spatial tiling first
        actual_tile_size, skip_reason = Autotiler.calculate_spatial_tile_config(c_in, h, w, tile_size, min_tile, stride=sh)
        
        if actual_tile_size is None:
            # Fallback to channel splitting if beneficial
            if skip_reason in ("min_tile_too_large", "no_divisor") and Autotiler.is_channel_splittable(m):
                return Autotiler._get_channel_split_config_params(
                    c_in, c_out, h, w, tile_size, min_tile, inp_name, out_name
                )
            return None

        # Section 5: Finalize spatial tiling parameters
        return Autotiler._finalize_spatial_tiling_params(
            inp_name, out_name, c_in, c_out, h, w, actual_tile_size, sh, sw, conv_params
        )

    @staticmethod
    def _get_model_dimensions(model: onnx.ModelProto) -> tuple[str, str, int, int, int] | None:
        """Extract input/output names and C, H, W dimensions."""
        inp = model.graph.input[0]
        out = model.graph.output[0]
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if len(dims) != 4 or dims[2] != dims[3]: # Expecting [1, C, H, W] where H == W
            return None
        return inp.name, out.name, dims[1], dims[2], dims[3]

    @staticmethod
    def _get_channel_split_config_params(
        c_in: int, c_out: int, h: int, w: int, tile_size: int, min_tile: int,
        inp_name: str, out_name: str
    ) -> dict | None:
        """Helper to build channel split configuration dictionary."""
        num_groups, cpg, split_err = Autotiler.calculate_channel_split_config(
            c_in, c_out, h, w, tile_size, min_tile
        )
        if num_groups is not None and num_groups > 1:
            return {
                "needs_channel_split": True,
                "c_in": c_in,
                "c_out": c_out,
                "num_groups": num_groups,
                "channels_per_group": cpg,
                "h": h,
                "w": w,
                "input_name": inp_name,
                "output_name": out_name,
            }
        return None

    @staticmethod
    def _finalize_spatial_tiling_params(
        inp_name: str, out_name: str, c_in: int, c_out: int, h: int, w: int,
        actual_tile_size: int, sh: int, sw: int, conv_params: dict
    ) -> dict | None:
        """Helper to calculate and build spatial tiling parameters dictionary."""
        if h % actual_tile_size != 0 or w % actual_tile_size != 0:
            return None

        tiles_y = h // actual_tile_size
        tiles_x = w // actual_tile_size
        num_tiles = tiles_y * tiles_x

        if num_tiles < 2:
            return None

        halo_h, halo_w = Autotiler.compute_halo_size(conv_params['kernel'], conv_params['dilation'])

        return {
            "input_name": inp_name,
            "output_name": out_name,
            "c_in": c_in,
            "c_out": c_out,
            "h": h,
            "w": w,
            "tile_size": actual_tile_size,
            "halo": [halo_h, halo_w],
            "tiles_y": tiles_y,
            "tiles_x": tiles_x,
            "num_tiles": num_tiles,
            "out_tile": [actual_tile_size // sh, actual_tile_size // sw],
            "stride": [sh, sw],
        }

    # =========================================================================
    # Section 4: Channel Splitting Generation
    # =========================================================================

    @staticmethod
    def _get_weights_and_bias(model: onnx.ModelProto, conv_node: onnx.NodeProto) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract weights and bias arrays from ONNX initializers."""
        weights, bias = None, None
        for init in model.graph.initializer:
            if init.name == conv_node.input[1]:
                weights = numpy_helper.to_array(init)
            if len(conv_node.input) > 2 and init.name == conv_node.input[2]:
                bias = numpy_helper.to_array(init)
        return weights, bias

    @staticmethod
    def create_channel_group_slice(
        slice_path: Path,
        group_idx: int,
        c_start: int,
        c_end: int,
        slice_idx: int,
        output_dir: Path,
    ) -> dict | None:
        """Create an ONNX model for a single channel group."""
        m = onnx.load(str(slice_path))
        
        # Section 1: Parameter Extraction
        conv_params = Autotiler.get_conv_params(m)
        if not conv_params:
            return None

        weights, _ = Autotiler._get_weights_and_bias(m, conv_params['node'])
        if weights is None:
            return None

        # Section 2: Dimension and Sliced Weight Calculation
        group_dims = Autotiler._calculate_channel_group_dims(m, c_start, c_end, conv_params)
        sliced_weights = weights[:, c_start:c_end, :, :]

        # Section 3: Build Graph Components (IO, Initializers, Nodes)
        input_name = f"group_{group_idx}_in"
        output_name = f"group_{group_idx}_out"
        
        X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, group_dims['c_group'], group_dims['h_in'], group_dims['w_in']])
        Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, weights.shape[0], group_dims['h_out'], group_dims['w_out']])

        W = helper.make_tensor("W", TensorProto.FLOAT, sliced_weights.shape, sliced_weights.flatten().tolist())
        node = helper.make_node(
            "Conv", [input_name, "W"], [output_name],
            kernel_shape=conv_params['kernel'], strides=conv_params['stride'], 
            pads=conv_params['pads'], dilations=conv_params['dilation']
        )

        # Section 4: Assemble and Save
        return Autotiler._save_channel_group_model(
            [node], slice_idx, group_idx, X, Y, [W], output_dir, c_start, c_end, weights.shape[0], group_dims
        )

    @staticmethod
    def _calculate_channel_group_dims(m: onnx.ModelProto, c_start: int, c_end: int, conv_params: dict) -> dict:
        """Calculate spatial dimensions for a channel group slice."""
        kh, kw = conv_params['kernel']
        sh, sw = conv_params['stride']
        dh, dw = conv_params['dilation']
        pads = conv_params['pads']

        dims = Autotiler._get_model_dimensions(m)
        _, _, _, h_in, w_in = dims

        effective_kh = (kh - 1) * dh + 1
        effective_kw = (kw - 1) * dw + 1
        
        return {
            'c_group': c_end - c_start,
            'h_in': h_in,
            'w_in': w_in,
            'h_out': (h_in + pads[0] + pads[2] - effective_kh) // sh + 1,
            'w_out': (w_in + pads[1] + pads[3] - effective_kw) // sw + 1
        }

    @staticmethod
    def _save_channel_group_model(
        nodes, slice_idx, group_idx, X, Y, initializers, output_dir, c_start, c_end, c_out, dims
    ) -> dict:
        """Assemble the channel group ONNX model and save it."""
        graph = helper.make_graph(nodes, f"channel_group_{slice_idx}_{group_idx}", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        groups_dir = output_dir / "channel_groups"
        groups_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = groups_dir / f"group_{group_idx}.onnx"

        Utils.save_onnx_model(model, onnx_path)

        return {
            "path": str(onnx_path.resolve()),
            "group_idx": group_idx,
            "c_start": c_start,
            "c_end": c_end,
            "output_shape": [1, c_out, dims['h_out'], dims['w_out']],
        }

    @staticmethod
    def apply_channel_splitting_to_slice(
        slice_path: Path,
        split_params: dict,
        slice_idx: int,
        output_dir: Path,
    ) -> ChannelSplitInfo | None:
        """Apply channel splitting to a single slice and save artifacts."""
        m = onnx.load(str(slice_path))
        conv_params = Autotiler.get_conv_params(m)
        if not conv_params:
            return None

        # Section 1: Process individual channel groups
        groups = Autotiler._process_channel_groups(slice_path, split_params, slice_idx, output_dir)
        if groups is None:
            return None

        # Section 2: Process bias if present
        bias_path = Autotiler._save_conv_bias(m, conv_params['node'], slice_idx, output_dir)

        # Section 3: Assembly final info
        return ChannelSplitInfo(
            slice_idx=slice_idx,
            c_in=split_params["c_in"],
            c_out=split_params["c_out"],
            num_groups=split_params["num_groups"],
            channels_per_group=split_params["channels_per_group"],
            input_name=split_params["input_name"],
            output_name=split_params["output_name"],
            h=split_params["h"],
            w=split_params["w"],
            groups=groups,
            bias_path=bias_path,
        )

    @staticmethod
    def _process_channel_groups(slice_path: Path, split_params: dict, slice_idx: int, output_dir: Path) -> list[ChannelGroupInfo] | None:
        """Create models for each channel group and return their info."""
        groups = []
        c_in = split_params["c_in"]
        num_groups = split_params["num_groups"]
        channels_per_group = split_params["channels_per_group"]

        for g in range(num_groups):
            c_start = g * channels_per_group
            c_end = min((g + 1) * channels_per_group, c_in)

            group_info = Autotiler.create_channel_group_slice(
                slice_path=slice_path, group_idx=g, c_start=c_start, c_end=c_end,
                slice_idx=slice_idx, output_dir=output_dir,
            )

            if group_info is None:
                return None

            groups.append(ChannelGroupInfo(
                group_idx=g, c_start=c_start, c_end=c_end,
                path=f"slice_{slice_idx}/payload/channel_groups/group_{g}.onnx",
            ))
        return groups

    @staticmethod
    def _save_conv_bias(model: onnx.ModelProto, conv_node: onnx.NodeProto, slice_idx: int, output_dir: Path) -> str | None:
        """Extract and save Conv bias to a .npy file if it exists."""
        _, bias = Autotiler._get_weights_and_bias(model, conv_node)
        if bias is None:
            return None
            
        bias_dir = output_dir / "channel_groups"
        bias_dir.mkdir(parents=True, exist_ok=True)
        bias_file = bias_dir / "bias.npy"
        np.save(str(bias_file), bias)
        return f"slice_{slice_idx}/payload/channel_groups/bias.npy"

    # =========================================================================
    # Section 5: Tiling Generation
    # =========================================================================

    @staticmethod
    def create_tile_slice(
        slice_path: Path,
        tile_size: int,
        slice_idx: int,
        output_dir: Path,
    ) -> dict | None:
        """Create a single spatial tile processing ONNX model."""
        m = onnx.load(slice_path)
        
        # Section 1: Parameter Extraction and Validation
        conv_params = Autotiler.get_conv_params(m)
        if not conv_params or conv_params['stride'][0] == 0 or conv_params['stride'][1] == 0:
            return None

        weights, bias = Autotiler._get_weights_and_bias(m, conv_params['node'])
        if weights is None:
            return None

        # Section 2: Dimension Calculation
        dims = Autotiler._calculate_tile_slice_dims(tile_size, conv_params, weights)
        
        # Section 3: Build Graph Components (IO, Initializers, Nodes)
        c_in = m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        X = helper.make_tensor_value_info("tile_in", TensorProto.FLOAT, [1, c_in, dims['tile_h'], dims['tile_w']])
        Y = helper.make_tensor_value_info("tile_out", TensorProto.FLOAT, [1, weights.shape[0], dims['out_h'], dims['out_w']])

        initializers, nodes = Autotiler._build_tile_graph_components(m, conv_params, weights, bias)

        # Section 4: Assemble and Save Model
        return Autotiler._save_tile_model(nodes, slice_idx, X, Y, initializers, output_dir, dims)

    @staticmethod
    def _calculate_tile_slice_dims(tile_size: int, conv_params: dict, weights: np.ndarray) -> dict:
        """Calculate spatial dimensions for the tile model."""
        kh, kw = conv_params['kernel']
        sh, sw = conv_params['stride']
        dh, dw = conv_params['dilation']
        
        halo_h, halo_w = Autotiler.compute_halo_size([kh, kw], [dh, dw])
        effective_kh = (kh - 1) * dh + 1
        effective_kw = (kw - 1) * dw + 1
        
        tile_h = tile_size + 2 * halo_h
        tile_w = tile_size + 2 * halo_w
        
        return {
            'tile_h': tile_h,
            'tile_w': tile_w,
            'out_h': (tile_h - effective_kh) // sh + 1,
            'out_w': (tile_w - effective_kw) // sw + 1,
            'halo': [halo_h, halo_w]
        }

    @staticmethod
    def _build_tile_graph_components(
        m: onnx.ModelProto, conv_params: dict, weights: np.ndarray, bias: np.ndarray | None
    ) -> tuple[list, list]:
        """Create initializers and nodes for the tiled ONNX model."""
        conv_node = conv_params['node']
        initializers = [helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.flatten().tolist())]
        conv_inputs = ["tile_in", "W"]
        
        if bias is not None:
            initializers.append(helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.flatten().tolist()))
            conv_inputs.append("B")

        nodes = [helper.make_node(
            "Conv", conv_inputs, ["conv_out"],
            kernel_shape=conv_params['kernel'], strides=conv_params['stride'], 
            pads=[0, 0, 0, 0], dilations=conv_params['dilation']
        )]

        # Handle post-Conv elementwise operations
        Autotiler._integrate_extra_ops(m, conv_node, initializers, nodes)
        return initializers, nodes

    @staticmethod
    def _integrate_extra_ops(m: onnx.ModelProto, conv_node: onnx.NodeProto, initializers: list, nodes: list):
        """Inject non-convolutional nodes from the original slice into the tiled graph."""
        orig_input_name = m.graph.input[0].name
        non_conv_ops = [n for n in m.graph.node if n.op_type != "Conv"]
        
        if not non_conv_ops:
            nodes[-1].output[0] = "tile_out"
            return

        # Add remaining initializers
        conv_weight_names = {conv_node.input[1]} | ({conv_node.input[2]} if len(conv_node.input) > 2 else set())
        for init in m.graph.initializer:
            if init.name not in conv_weight_names:
                initializers.append(init)

        # Reconnect elementwise ops
        for i, orig_node in enumerate(non_conv_ops):
            new_inputs = []
            for inp in orig_node.input:
                if any(inp == n.output[0] for n in m.graph.node if n.op_type == "Conv"):
                    new_inputs.append("conv_out")
                elif inp == orig_input_name:
                    new_inputs.append("tile_in")
                else:
                    new_inputs.append(inp)
            
            is_last = (i == len(non_conv_ops) - 1)
            new_outputs = ["tile_out"] if is_last else list(orig_node.output)
            attr_kwargs = {a.name: helper.get_attribute_value(a) for a in orig_node.attribute}
            nodes.append(helper.make_node(orig_node.op_type, new_inputs, new_outputs, **attr_kwargs))

    @staticmethod
    def _save_tile_model(nodes, slice_idx, X, Y, initializers, output_dir, dims) -> dict:
        """Assemble the ONNX model and save it to disk."""
        graph = helper.make_graph(nodes, f"tile_{slice_idx}", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        tiles_dir = output_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = tiles_dir / "tile.onnx"

        Utils.save_onnx_model(model, onnx_path)

        return {
            "path": str(onnx_path.resolve()),
            "input_name": "tile_in",
            "output_name": "tile_out",
            "conv_out": [dims['out_h'], dims['out_w']],
        }

    # =========================================================================
    # Section 6: Orchestration
    # =========================================================================

    @staticmethod
    def apply_tiling(slices_paths: dict[int, str | Path], tile_size: int) -> dict[int, Any]:
        """Apply tiling or channel splitting to a set of slices."""
        print(f"Using dynamic tiling with tile_size={tile_size} elements per tile")
        tiled_results = {}

        for idx, onnx_path in slices_paths.items():
            result = Autotiler._apply_transform_to_slice(idx, onnx_path, tile_size)
            if result:
                tiled_results[idx] = result

        print(f"Tiled {len(tiled_results)} Conv slices")
        return tiled_results

    @staticmethod
    def _apply_transform_to_slice(idx: int, onnx_path: str | Path, tile_size: int) -> dict | None:
        """Analyze a single slice and apply the appropriate tiling or splitting transform."""
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            return None

        tiling_params = Autotiler.detect_tiling_needs(onnx_path, tile_size=tile_size)
        if not tiling_params:
            return None

        if tiling_params.get("needs_channel_split"):
            return Autotiler._handle_channel_split_transform(idx, onnx_path, tiling_params)
        else:
            return Autotiler._handle_spatial_tiling_transform(idx, onnx_path, tiling_params)

    @staticmethod
    def _handle_channel_split_transform(idx: int, onnx_path: Path, tiling_params: dict) -> dict | None:
        """Orchestrate channel splitting for a single slice."""
        c_in = tiling_params["c_in"]
        num_groups = tiling_params["num_groups"]
        cpg = tiling_params["channels_per_group"]
        
        print(f"Channel splitting Conv slice {idx}: {c_in}ch x {tiling_params['h']}x{tiling_params['w']} -> {num_groups} groups ({cpg} channels/group)")

        channel_split_info = Autotiler.apply_channel_splitting_to_slice(
            slice_path=onnx_path,
            split_params=tiling_params,
            slice_idx=idx,
            output_dir=onnx_path.parent,
        )
        
        if not channel_split_info:
            print(f"  Failed to create channel groups for slice {idx}")
            return None

        return {"channel_split": channel_split_info.to_dict()}

    @staticmethod
    def _handle_spatial_tiling_transform(idx: int, onnx_path: Path, tiling_params: dict) -> dict | None:
        """Orchestrate spatial tiling for a single slice."""
        c_in = tiling_params["c_in"]
        tile_sz = tiling_params["tile_size"]
        
        print(f"Tiling Conv slice {idx}: {c_in}ch x {tiling_params['h']}x{tiling_params['w']} -> tile_size={tile_sz} ({tiling_params['num_tiles']} tiles)")

        tile_info = Autotiler.create_tile_slice(
            slice_path=onnx_path,
            tile_size=tile_sz,
            slice_idx=idx,
            output_dir=onnx_path.parent,
        )
        
        if not tile_info:
            print(f"  Failed to create tile model for slice {idx}")
            return None

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
        return {"tiling": tiling_metadata.to_dict()}

    @staticmethod
    def apply_tiling_to_slices(slices_dir: str | Path, tile_size: int | None = None) -> dict:
        """
        Apply tiling to Conv slices (backward compatibility wrapper).
        Creates tile ONNX models and updates metadata.
        """
        slices_dir = Path(slices_dir)
        metadata_path = slices_dir / "metadata.json"

        # Section 1: Validation and Metadata Loading
        if not metadata_path.exists():
            print(f"No metadata.json found in {slices_dir}")
            return {}

        with open(metadata_path) as f:
            metadata = json.load(f)

        slices_data = metadata.get("slices", [])
        if not slices_data or not tile_size:
            print("No slices found or no tiling parameters specified")
            return {}

        # Section 2: Map Indices to Paths and Apply Tiling
        slices_paths = Autotiler._map_slice_indices_to_paths(slices_dir, slices_data)
        tiled_results = Autotiler.apply_tiling(slices_paths, tile_size)

        # Section 3: Update Global and Per-Slice Metadata
        Autotiler._update_all_metadata(slices_dir, metadata, tiled_results)

        return tiled_results

    @staticmethod
    def _map_slice_indices_to_paths(slices_dir: Path, slices_data: list) -> dict[int, Path]:
        """Create a mapping of slice indices to their ONNX file paths."""
        slices_paths = {}
        for idx, _slice_meta in enumerate(slices_data):
            onnx_path = slices_dir / f"slice_{idx}" / "payload" / f"slice_{idx}.onnx"
            if onnx_path.exists():
                slices_paths[idx] = onnx_path
        return slices_paths

    @staticmethod
    def _update_all_metadata(slices_dir: Path, global_metadata: dict, tiled_results: dict):
        """Update both the global metadata and individual slice metadata files."""
        slices_data = global_metadata.get("slices", [])
        
        for idx, result in tiled_results.items():
            # Update global metadata structure in memory
            if "channel_split" in result:
                slices_data[idx]["channel_split"] = result["channel_split"]
            if "tiling" in result:
                slices_data[idx]["tiling"] = result["tiling"]

            # Update individual slice-level metadata files on disk
            Autotiler._update_single_slice_metadata(slices_dir, idx, result)

        # Save updated global metadata
        metadata_path = slices_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(global_metadata, f, indent=2)

    @staticmethod
    def _update_single_slice_metadata(slices_dir: Path, idx: int, result: dict):
        """Update a specific slice's metadata.json with transform results."""
        slice_meta_path = slices_dir / f"slice_{idx}" / "metadata.json"
        if not slice_meta_path.exists():
            return

        with open(slice_meta_path, "r") as f:
            slice_meta = json.load(f)
        
        slice_slices = slice_meta.get("slices", [])
        if slice_slices:
            if "channel_split" in result:
                slice_slices[0]["channel_split"] = result["channel_split"]
            if "tiling" in result:
                slice_slices[0]["tiling"] = result["tiling"]
        
        with open(slice_meta_path, "w") as f:
            json.dump(slice_meta, f, indent=2)

