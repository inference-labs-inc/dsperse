use std::collections::HashMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dsperse::schema::execution::{
    ExecutionChain, ExecutionInfo, ExecutionMethod, ExecutionNode, ExecutionResultEntry,
    RunMetadata, SliceResult, TileResult,
};
use dsperse::schema::metadata::{
    Backend, Compilation, Dependencies, ModelMetadata, RunSliceMetadata, SliceMetadata,
    SliceShapeWrapper, TensorShape,
};
use serde::{Deserialize, Serialize};

fn make_slice_metadata(index: usize) -> SliceMetadata {
    SliceMetadata {
        index,
        filename: format!("slice_{index}.onnx"),
        path: format!("/tmp/slices/slice_{index}/payload/slice_{index}.onnx"),
        relative_path: format!("slice_{index}/payload/slice_{index}.onnx"),
        shape: SliceShapeWrapper {
            tensor_shape: TensorShape {
                input: vec![vec![1, 3, 224, 224]],
                output: vec![vec![1, 64, 112, 112]],
            },
        },
        dependencies: Dependencies {
            input: vec![format!("input_{index}")],
            output: vec![format!("output_{index}")],
            filtered_inputs: vec![format!("input_{index}")],
        },
        tiling: None,
        channel_split: None,
        compilation: Compilation::default(),
        slice_metadata: Some(format!("slice_{index}/metadata.msgpack")),
        slice_metadata_relative_path: Some(format!("slice_{index}/metadata.msgpack")),
    }
}

fn make_model_metadata(num_slices: usize) -> ModelMetadata {
    let slices: Vec<SliceMetadata> = (0..num_slices).map(make_slice_metadata).collect();
    let slice_points: Vec<usize> = (0..=num_slices).collect();
    ModelMetadata {
        original_model: "/tmp/model.onnx".into(),
        model_type: "ONNX".into(),
        input_shape: vec![vec![1, 3, 224, 224]],
        output_shapes: vec![vec![1, 1000]],
        output_names: vec!["output".into()],
        slice_points,
        slices,
        dsperse_version: Some("0.0.0".into()),
        dsperse_rev: Some("abc1234".into()),
        jstprove_version: Some("0.1.0".into()),
        jstprove_rev: Some("def5678".into()),
        traced_shapes: None,
        original_model_path: None,
    }
}

fn make_run_metadata(num_slices: usize) -> RunMetadata {
    let mut slices = HashMap::new();
    let mut nodes = HashMap::new();
    let mut execution_results = Vec::new();

    for i in 0..num_slices {
        let slice_id = format!("slice_{i}");
        slices.insert(
            slice_id.clone(),
            RunSliceMetadata {
                path: format!("slice_{i}/payload/slice_{i}.onnx"),
                input_shape: vec![vec![1, 3, 224, 224]],
                output_shape: vec![vec![1, 64, 112, 112]],
                dependencies: Dependencies {
                    input: vec![format!("input_{i}")],
                    output: vec![format!("output_{i}")],
                    filtered_inputs: vec![format!("input_{i}")],
                },
                tiling: None,
                channel_split: None,
                backend: Backend::Jstprove,
                jstprove_circuit_path: Some(format!("slice_{i}/jstprove/circuit.bin")),
                jstprove_settings_path: None,
            },
        );
        nodes.insert(
            slice_id.clone(),
            ExecutionNode {
                slice_id: slice_id.clone(),
                primary: Some("jstprove_gen_witness".into()),
                fallbacks: vec!["onnx_only".into()],
                use_circuit: true,
                next: if i + 1 < num_slices {
                    Some(format!("slice_{}", i + 1))
                } else {
                    None
                },
                circuit_path: Some(format!("slice_{i}/jstprove/circuit.bin")),
                onnx_path: Some(format!("slice_{i}/payload/slice_{i}.onnx")),
                backend: Backend::Jstprove,
            },
        );
        execution_results.push(ExecutionResultEntry {
            slice_id: slice_id.clone(),
            witness_execution: Some(ExecutionInfo {
                method: ExecutionMethod::JstproveGenWitness,
                success: true,
                error: None,
                witness_file: Some(format!("runs/run_0/{slice_id}/witness.bin")),
                tile_exec_infos: vec![TileResult {
                    tile_idx: 0,
                    success: true,
                    error: None,
                    method: Some(ExecutionMethod::JstproveGenWitness),
                    time_sec: 1.23,
                    proof_path: None,
                }],
            }),
            proof_execution: Some(SliceResult {
                slice_id: slice_id.clone(),
                success: true,
                method: Some(ExecutionMethod::JstproveProve),
                error: None,
                proof_path: Some(format!("runs/run_0/{slice_id}/proof.bin")),
                time_sec: 45.67,
                tiles: Vec::new(),
            }),
            verification_execution: None,
        });
    }

    RunMetadata {
        slices,
        execution_chain: ExecutionChain {
            head: Some("slice_0".into()),
            nodes,
            fallback_map: HashMap::new(),
            execution_results,
            jstprove_proved_slices: num_slices,
            jstprove_verified_slices: 0,
        },
        overall_security: 128.0,
        packaging_type: Some("dsperse".into()),
        source_path: Some("/tmp/model.onnx".into()),
        run_directory: Some("/tmp/runs/run_0".into()),
        model_path: Some("/tmp/model.onnx".into()),
    }
}

fn bench_roundtrip<T: Serialize + for<'de> Deserialize<'de>>(
    c: &mut Criterion,
    name: &str,
    value: &T,
) {
    let json_bytes = serde_json::to_vec(value).unwrap();
    let msgpack_bytes = rmp_serde::to_vec_named(value).unwrap();

    let group_name = format!(
        "{name} (json={}, msgpack={})",
        json_bytes.len(),
        msgpack_bytes.len()
    );
    let mut group = c.benchmark_group(&group_name);

    group.bench_function("json_serialize", |b| {
        b.iter(|| serde_json::to_vec(black_box(value)).unwrap());
    });
    group.bench_function("msgpack_serialize", |b| {
        b.iter(|| rmp_serde::to_vec_named(black_box(value)).unwrap());
    });
    group.bench_function("json_deserialize", |b| {
        b.iter(|| serde_json::from_slice::<T>(black_box(&json_bytes)).unwrap());
    });
    group.bench_function("msgpack_deserialize", |b| {
        b.iter(|| rmp_serde::from_slice::<T>(black_box(&msgpack_bytes)).unwrap());
    });

    group.finish();
}

fn serialization_benchmarks(c: &mut Criterion) {
    let small_model = make_model_metadata(4);
    let large_model = make_model_metadata(64);
    let small_run = make_run_metadata(4);
    let large_run = make_run_metadata(64);

    bench_roundtrip(c, "ModelMetadata_4slices", &small_model);
    bench_roundtrip(c, "ModelMetadata_64slices", &large_model);
    bench_roundtrip(c, "RunMetadata_4slices", &small_run);
    bench_roundtrip(c, "RunMetadata_64slices", &large_run);
}

criterion_group!(benches, serialization_benchmarks);
criterion_main!(benches);
