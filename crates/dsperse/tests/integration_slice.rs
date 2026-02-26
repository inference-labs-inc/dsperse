use std::path::Path;

use dsperse::schema::metadata::ModelMetadata;

fn test_models_dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models"))
}

#[test]
fn slice_net_model() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(model_path.exists(), "test model not found at {}", model_path.display());

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata =
        dsperse::slicer::slice_model(&model_path, Some(&output_dir), None).expect("slice_model");

    assert!(!metadata.slices.is_empty());
    assert_eq!(metadata.model_type, "ONNX");
    assert!(!metadata.input_shape.is_empty());
    assert!(!metadata.output_shapes.is_empty());

    let meta_path = output_dir.join("metadata.msgpack");
    assert!(meta_path.exists(), "metadata.msgpack must be written");

    let loaded = ModelMetadata::load(&meta_path).expect("load metadata");
    assert_eq!(loaded.slices.len(), metadata.slices.len());

    for slice in &loaded.slices {
        let slice_dir = output_dir.join(format!("slice_{}", slice.index));
        assert!(slice_dir.exists(), "slice dir must exist: {}", slice_dir.display());

        let payload_dir = slice_dir.join("payload");
        assert!(payload_dir.exists(), "payload dir must exist");

        let onnx_file = payload_dir.join(&slice.filename);
        assert!(onnx_file.exists(), "onnx file must exist: {}", onnx_file.display());

        let per_slice_meta = slice_dir.join("metadata.msgpack");
        assert!(per_slice_meta.exists(), "per-slice metadata.msgpack must exist");
    }
}

#[test]
fn slice_doom_model() {
    let model_path = test_models_dir().join("doom/model.onnx");
    assert!(model_path.exists(), "test model not found at {}", model_path.display());

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata =
        dsperse::slicer::slice_model(&model_path, Some(&output_dir), None).expect("slice_model");

    assert!(!metadata.slices.is_empty());

    for (i, slice) in metadata.slices.iter().enumerate() {
        assert_eq!(slice.index, i);
        assert!(!slice.dependencies.input.is_empty());
        assert!(!slice.dependencies.output.is_empty());
    }
}

#[test]
fn slice_with_tile_size() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(model_path.exists(), "test model not found at {}", model_path.display());

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata =
        dsperse::slicer::slice_model(&model_path, Some(&output_dir), Some(8)).expect("slice_model");

    assert!(!metadata.slices.is_empty());

    let meta_path = output_dir.join("metadata.msgpack");
    assert!(meta_path.exists());
}

#[test]
fn slice_metadata_roundtrip_from_disk() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(model_path.exists(), "test model not found at {}", model_path.display());

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let original =
        dsperse::slicer::slice_model(&model_path, Some(&output_dir), None).expect("slice_model");

    let meta_path = output_dir.join("metadata.msgpack");
    let deserialized = ModelMetadata::load(&meta_path).expect("load metadata");

    assert_eq!(original.slices.len(), deserialized.slices.len());
    assert_eq!(original.original_model, deserialized.original_model);
    assert_eq!(original.input_shape, deserialized.input_shape);
    assert_eq!(original.output_shapes, deserialized.output_shapes);
}

#[test]
fn resolve_onnx_points_to_existing_file_after_slice() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(model_path.exists(), "test model not found at {}", model_path.display());

    let tmp = tempfile::tempdir().expect("create temp dir");
    let output_dir = tmp.path().join("slices");

    let metadata =
        dsperse::slicer::slice_model(&model_path, Some(&output_dir), None).expect("slice_model");

    let loaded = ModelMetadata::load(&output_dir.join("metadata.msgpack")).expect("load metadata");

    for slice in &loaded.slices {
        let resolved = slice.resolve_onnx(&output_dir);
        assert!(
            resolved.exists(),
            "resolve_onnx for slice {} must point to an existing file, got: {}",
            slice.index,
            resolved.display()
        );
        assert!(
            !resolved.to_string_lossy().contains(&format!(
                "slice_{}/{}",
                slice.index,
                output_dir.to_string_lossy()
            )),
            "resolved path must not contain doubled directory prefix"
        );
    }

    assert!(!metadata.slices.is_empty());
}
