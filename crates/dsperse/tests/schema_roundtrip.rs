use dsperse::schema::*;

#[test]
fn model_metadata_roundtrip() {
    let json = r#"{
        "original_model": "model.onnx",
        "model_type": "onnx",
        "input_shape": [[1, 3, 32, 32]],
        "output_shapes": [[1, 10]],
        "slice_points": [2, 5],
        "slices": [
            {
                "index": 0,
                "filename": "slice_0.onnx",
                "path": "/tmp/slices/slice_0/payload/slice_0.onnx",
                "relative_path": "slice_0/payload/slice_0.onnx",
                "shape": {
                    "tensor_shape": {
                        "input": [[1, 3, 32, 32]],
                        "output": [[1, 16, 16, 16]]
                    }
                },
                "dependencies": {
                    "input": ["input"],
                    "output": ["conv1_out"],
                    "filtered_inputs": ["input"]
                },
                "compilation": {
                    "jstprove": {
                        "compiled": true,
                        "tiled": false,
                        "weights_as_inputs": false,
                        "files": {
                            "compiled": "jstprove/circuit.txt",
                            "settings": "jstprove/settings.json"
                        }
                    }
                }
            },
            {
                "index": 1,
                "filename": "slice_1.onnx",
                "path": "/tmp/slices/slice_1/payload/slice_1.onnx",
                "relative_path": "slice_1/payload/slice_1.onnx",
                "shape": {
                    "tensor_shape": {
                        "input": [[1, 16, 16, 16]],
                        "output": [[1, 10]]
                    }
                },
                "dependencies": {
                    "input": ["conv1_out"],
                    "output": ["output"],
                    "filtered_inputs": ["conv1_out"]
                },
                "tiling": {
                    "slice_idx": 1,
                    "tile_size": 8,
                    "num_tiles": 4,
                    "tiles_y": 2,
                    "tiles_x": 2,
                    "halo": [1, 1],
                    "out_tile": [8, 8],
                    "stride": [1, 1],
                    "c_in": 16,
                    "c_out": 32,
                    "input_name": "conv1_out",
                    "output_name": "conv2_out",
                    "tile": {
                        "path": "tiles/tile.onnx",
                        "conv_out": [8, 8]
                    },
                    "tiles": [
                        {"path": "tiles/tile.onnx", "conv_out": [8, 8]},
                        {"path": "tiles/tile.onnx", "conv_out": [8, 8]}
                    ]
                },
                "compilation": {
                    "jstprove": {
                        "compiled": false,
                        "tiled": false,
                        "weights_as_inputs": false,
                        "files": {}
                    }
                }
            }
        ]
    }"#;

    let meta: ModelMetadata = serde_json::from_str(json).unwrap();
    assert_eq!(meta.original_model, "model.onnx");
    assert_eq!(meta.slices.len(), 2);
    assert_eq!(meta.slice_points, vec![2, 5]);

    let s0 = &meta.slices[0];
    assert_eq!(s0.index, 0);
    assert!(s0.compilation.jstprove.compiled);
    assert_eq!(
        s0.compilation.jstprove.files.compiled.as_deref(),
        Some("jstprove/circuit.txt")
    );
    assert!(s0.tiling.is_none());

    let s1 = &meta.slices[1];
    assert!(s1.tiling.is_some());
    let tiling = s1.tiling.as_ref().unwrap();
    assert_eq!(tiling.num_tiles, 4);
    assert_eq!(tiling.halo, [1, 1]);
    assert_eq!(tiling.tiles.as_ref().unwrap().len(), 2);

    let msgpack_bytes = rmp_serde::to_vec_named(&meta).unwrap();
    let meta2: ModelMetadata = rmp_serde::from_slice(&msgpack_bytes).unwrap();
    assert_eq!(meta2.slices.len(), 2);
    assert_eq!(meta2.slices[0].index, 0);
}

#[test]
fn run_metadata_roundtrip() {
    let json = r#"{
        "slices": {
            "slice_0": {
                "path": "slice_0/payload/slice_0.onnx",
                "input_shape": [[1, 3, 32, 32]],
                "output_shape": [[1, 16, 16, 16]],
                "dependencies": {
                    "input": ["input"],
                    "output": ["conv1_out"],
                    "filtered_inputs": ["input"]
                },
                "backend": "jstprove",
                "circuit_path": "slice_0/payload/jstprove/circuit.txt"
            }
        },
        "execution_chain": {
            "head": "slice_0",
            "nodes": {
                "slice_0": {
                    "slice_id": "slice_0",
                    "primary": "slice_0/payload/jstprove/circuit.txt",
                    "fallbacks": ["slice_0/payload/slice_0.onnx"],
                    "use_circuit": true,
                    "next": null,
                    "circuit_path": "slice_0/payload/jstprove/circuit.txt",
                    "onnx_path": "slice_0/payload/slice_0.onnx",
                    "backend": "jstprove"
                }
            },
            "fallback_map": {},
            "execution_results": [],
            "jstprove_proved_slices": 0,
            "jstprove_verified_slices": 0
        },
        "circuit_slices": {"slice_0": true},
        "overall_security": 100.0
    }"#;

    let meta: RunMetadata = serde_json::from_str(json).unwrap();
    assert_eq!(meta.slices.len(), 1);
    assert_eq!(meta.overall_security, 100.0);

    let slice = meta.get_slice("slice_0").unwrap();
    assert_eq!(slice.backend, "jstprove");
    assert_eq!(
        slice.jstprove_circuit_path.as_deref(),
        Some("slice_0/payload/jstprove/circuit.txt")
    );

    let chain = &meta.execution_chain;
    assert_eq!(chain.head.as_deref(), Some("slice_0"));
    assert!(chain.nodes["slice_0"].use_circuit);

    let circuit_slices: Vec<_> = meta.iter_circuit_slices().collect();
    assert_eq!(circuit_slices.len(), 1);
    assert_eq!(circuit_slices[0].0, "slice_0");

    let msgpack_bytes = rmp_serde::to_vec_named(&meta).unwrap();
    let meta2: RunMetadata = rmp_serde::from_slice(&msgpack_bytes).unwrap();
    assert_eq!(meta2.slices.len(), 1);
}

#[test]
fn execution_info_with_tiles() {
    let json = r#"{
        "method": "tiled",
        "success": true,
        "tile_exec_infos": [
            {"tile_idx": 0, "success": true, "method": "jstprove_gen_witness", "time_sec": 1.5},
            {"tile_idx": 1, "success": true, "method": "jstprove_gen_witness", "time_sec": 1.3},
            {"tile_idx": 2, "success": false, "error": "timeout", "time_sec": 30.0}
        ]
    }"#;

    let info: ExecutionInfo = serde_json::from_str(json).unwrap();
    assert!(info.success);
    assert_eq!(info.tile_exec_infos.len(), 3);
    assert!(!info.tile_exec_infos[2].success);
    assert_eq!(info.tile_exec_infos[2].error.as_deref(), Some("timeout"));
}

#[test]
fn channel_split_roundtrip() {
    let json = r#"{
        "slice_idx": 2,
        "c_in": 64,
        "c_out": 128,
        "num_groups": 4,
        "channels_per_group": 16,
        "input_name": "relu1_out",
        "output_name": "conv2_out",
        "h": 16,
        "w": 16,
        "groups": [
            {"group_idx": 0, "c_start": 0, "c_end": 16, "path": "channel_groups/group_0.onnx"},
            {"group_idx": 1, "c_start": 16, "c_end": 32, "path": "channel_groups/group_1.onnx"}
        ],
        "bias_path": "channel_groups/bias.msgpack"
    }"#;

    let info: ChannelSplitInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.num_groups, 4);
    assert_eq!(info.groups.len(), 2);
    assert_eq!(info.groups[0].c_end, 16);
    assert_eq!(info.bias_path.as_deref(), Some("channel_groups/bias.msgpack"));

    let msgpack_bytes = rmp_serde::to_vec_named(&info).unwrap();
    let info2: ChannelSplitInfo = rmp_serde::from_slice(&msgpack_bytes).unwrap();
    assert_eq!(info2.num_groups, 4);
}

#[test]
fn compilation_files_aliases() {
    let json1 = r#"{"compiled": "circuit.txt"}"#;
    let json2 = r#"{"compiled_circuit": "circuit.txt"}"#;
    let json3 = r#"{"circuit": "circuit.txt"}"#;

    let f1: CompilationFiles = serde_json::from_str(json1).unwrap();
    let f2: CompilationFiles = serde_json::from_str(json2).unwrap();
    let f3: CompilationFiles = serde_json::from_str(json3).unwrap();

    assert_eq!(f1.compiled.as_deref(), Some("circuit.txt"));
    assert_eq!(f2.compiled.as_deref(), Some("circuit.txt"));
    assert_eq!(f3.compiled.as_deref(), Some("circuit.txt"));
}

#[test]
fn backend_serde() {
    assert_eq!(
        serde_json::to_string(&Backend::Jstprove).unwrap(),
        r#""jstprove""#
    );
    assert_eq!(serde_json::to_string(&Backend::Onnx).unwrap(), r#""onnx""#);

    let b: Backend = serde_json::from_str(r#""jstprove""#).unwrap();
    assert_eq!(b, Backend::Jstprove);

    let b: Backend = serde_json::from_str(r#""JSTPROVE""#).unwrap();
    assert_eq!(b, Backend::Jstprove);
}

#[test]
fn tensor_shape_i64_deserialization() {
    let json = r#"{
        "input": [[1, 3, 224, 224]],
        "output": [[1, 1000]]
    }"#;

    let shape: TensorShape = serde_json::from_str(json).unwrap();
    assert_eq!(shape.input, vec![vec![1i64, 3, 224, 224]]);
    assert_eq!(shape.output, vec![vec![1i64, 1000]]);

    let msgpack_bytes = rmp_serde::to_vec_named(&shape).unwrap();
    let shape2: TensorShape = rmp_serde::from_slice(&msgpack_bytes).unwrap();
    assert_eq!(shape2.input, shape.input);
    assert_eq!(shape2.output, shape.output);
}

#[test]
fn tensor_shape_rejects_non_integer() {
    let json = r#"{"input": [[1, "hello", 3]], "output": []}"#;
    let result: std::result::Result<TensorShape, _> = serde_json::from_str(json);
    assert!(result.is_err());
}

#[test]
fn run_slice_metadata_i64_shapes() {
    let json = r#"{
        "path": "slice_0/payload/slice_0.onnx",
        "input_shape": [[1, 3, 32, 32]],
        "output_shape": [[1, 16, 16, 16]],
        "dependencies": {
            "input": ["input"],
            "output": ["conv1_out"],
            "filtered_inputs": ["input"]
        },
        "backend": "onnx"
    }"#;

    let meta: RunSliceMetadata = serde_json::from_str(json).unwrap();
    assert_eq!(meta.input_shape, vec![vec![1i64, 3, 32, 32]]);
    assert_eq!(meta.output_shape, vec![vec![1i64, 16, 16, 16]]);

    let msgpack_bytes = rmp_serde::to_vec_named(&meta).unwrap();
    let meta2: RunSliceMetadata = rmp_serde::from_slice(&msgpack_bytes).unwrap();
    assert_eq!(meta2.input_shape, meta.input_shape);
    assert_eq!(meta2.output_shape, meta.output_shape);
}
