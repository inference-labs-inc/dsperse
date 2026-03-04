use std::path::Path;

use dsperse::schema::metadata::ModelMetadata;

fn test_models_dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models"))
}

#[test]
fn slice_net_model() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(
        model_path.exists(),
        "test model not found at {}",
        model_path.display()
    );

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata = dsperse::slicer::slice_model(
        &model_path,
        Some(&output_dir),
        None,
        jstprove_circuits::ProofSystem::Expander.supported_ops(),
    )
    .expect("slice_model");

    assert!(!metadata.slices.is_empty());
    assert_eq!(metadata.model_type, "ONNX");
    assert!(!metadata.input_shape.is_empty());
    assert!(!metadata.output_shapes.is_empty());

    let meta_path = output_dir.join("metadata.msgpack");
    assert!(meta_path.exists(), "metadata.msgpack must be written");

    let model_onnx = output_dir.join("model.onnx");
    assert!(
        model_onnx.exists(),
        "model.onnx must be copied to output dir"
    );

    let loaded = ModelMetadata::load(&meta_path).expect("load metadata");
    assert_eq!(loaded.slices.len(), metadata.slices.len());

    assert!(loaded.traced_shapes.is_some());
    assert!(loaded.original_model_path.is_some());
    assert_eq!(loaded.original_model_path.as_deref(), Some("model.onnx"));
}

#[test]
fn slice_doom_model() {
    let model_path = test_models_dir().join("doom/model.onnx");
    assert!(
        model_path.exists(),
        "test model not found at {}",
        model_path.display()
    );

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata = dsperse::slicer::slice_model(
        &model_path,
        Some(&output_dir),
        None,
        jstprove_circuits::ProofSystem::Expander.supported_ops(),
    )
    .expect("slice_model");

    assert!(!metadata.slices.is_empty());

    for (i, slice) in metadata.slices.iter().enumerate() {
        assert_eq!(slice.index, i);
        assert!(!slice.dependencies.input.is_empty());
        assert!(!slice.dependencies.output.is_empty());
    }
}

#[test]
fn slice_net_model_remainder() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(
        model_path.exists(),
        "test model not found at {}",
        model_path.display()
    );

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata = dsperse::slicer::slice_model(
        &model_path,
        Some(&output_dir),
        None,
        jstprove_circuits::ProofSystem::Remainder.supported_ops(),
    )
    .expect("slice_model with Remainder");

    assert!(!metadata.slices.is_empty());
    assert_eq!(metadata.model_type, "ONNX");
}

#[test]
fn slice_with_tile_size() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(
        model_path.exists(),
        "test model not found at {}",
        model_path.display()
    );

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata = dsperse::slicer::slice_model(
        &model_path,
        Some(&output_dir),
        Some(8),
        jstprove_circuits::ProofSystem::Expander.supported_ops(),
    )
    .expect("slice_model");

    assert!(!metadata.slices.is_empty());

    let meta_path = output_dir.join("metadata.msgpack");
    assert!(meta_path.exists());
}

#[test]
fn slice_metadata_roundtrip_from_disk() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(
        model_path.exists(),
        "test model not found at {}",
        model_path.display()
    );

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let original = dsperse::slicer::slice_model(
        &model_path,
        Some(&output_dir),
        None,
        jstprove_circuits::ProofSystem::Expander.supported_ops(),
    )
    .expect("slice_model");

    let meta_path = output_dir.join("metadata.msgpack");
    let deserialized = ModelMetadata::load(&meta_path).expect("load metadata");

    assert_eq!(original.slices.len(), deserialized.slices.len());
    assert_eq!(original.original_model, deserialized.original_model);
    assert_eq!(original.input_shape, deserialized.input_shape);
    assert_eq!(original.output_shapes, deserialized.output_shapes);
    assert_eq!(original.traced_shapes, deserialized.traced_shapes);
}

#[test]
fn materialize_from_manifest() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(
        model_path.exists(),
        "test model not found at {}",
        model_path.display()
    );

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata = dsperse::slicer::slice_model(
        &model_path,
        Some(&output_dir),
        None,
        jstprove_circuits::ProofSystem::Expander.supported_ops(),
    )
    .expect("slice_model");

    dsperse::slicer::materializer::ensure_all_slices_materialized(&output_dir, &metadata)
        .expect("materialize all slices");

    for slice in &metadata.slices {
        let slice_dir = output_dir.join(format!("slice_{}", slice.index));
        assert!(
            slice_dir.exists(),
            "slice dir must exist after materialization: {}",
            slice_dir.display()
        );

        let payload_dir = slice_dir.join("payload");
        assert!(payload_dir.exists(), "payload dir must exist");

        let onnx_file = payload_dir.join(&slice.filename);
        assert!(
            onnx_file.exists(),
            "onnx file must exist: {}",
            onnx_file.display()
        );
    }
}

#[test]
fn resolve_onnx_points_to_existing_file_after_materialize() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(
        model_path.exists(),
        "test model not found at {}",
        model_path.display()
    );

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata = dsperse::slicer::slice_model(
        &model_path,
        Some(&output_dir),
        None,
        jstprove_circuits::ProofSystem::Expander.supported_ops(),
    )
    .expect("slice_model");

    dsperse::slicer::materializer::ensure_all_slices_materialized(&output_dir, &metadata)
        .expect("materialize all slices");

    let loaded = ModelMetadata::load(&output_dir.join("metadata.msgpack")).expect("load metadata");
    assert!(!loaded.slices.is_empty());

    for slice in &loaded.slices {
        let resolved = slice.resolve_onnx(&output_dir);
        assert!(
            resolved.is_file(),
            "resolve_onnx for slice {} must point to a regular file, got: {}",
            slice.index,
            resolved.display()
        );
        assert!(
            resolved.starts_with(&output_dir),
            "resolved path must start with output_dir"
        );
        let resolved_path = resolved.to_string_lossy();
        let output_str = output_dir.to_string_lossy();
        let count = resolved_path.matches(output_str.as_ref()).count();
        assert_eq!(
            count, 1,
            "output_dir must appear exactly once in resolved path, got {count}"
        );
    }
}
