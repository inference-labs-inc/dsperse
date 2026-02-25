use std::path::Path;

use dsperse::archive::converter::{self, FormatType};

fn test_models_dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models"))
}

#[test]
fn convert_dirs_to_dslice_roundtrip() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(model_path.exists(), "test model not found at {}", model_path.display());

    let tmp = tempfile::tempdir().expect("create temp dir");
    let slices_dir = tmp.path().join("slices");
    dsperse::slicer::slice_model(&model_path, Some(&slices_dir), None).expect("slice_model");

    let dslice_out = converter::convert(
        &slices_dir,
        FormatType::Dslice,
        None,
        false,
        true,
    )
    .expect("convert to dslice");
    assert!(dslice_out.exists(), "dslice output must exist");

    let has_dslice_files = std::fs::read_dir(&dslice_out)
        .expect("read dslice output dir")
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().is_some_and(|ext| ext == "dslice"));
    assert!(has_dslice_files, "dslice files must be produced");
}

#[test]
fn convert_dirs_to_dsperse_roundtrip() {
    let model_path = test_models_dir().join("net/model.onnx");
    assert!(model_path.exists(), "test model not found at {}", model_path.display());

    let tmp = tempfile::tempdir().expect("create temp dir");
    let slices_dir = tmp.path().join("slices");
    dsperse::slicer::slice_model(&model_path, Some(&slices_dir), None).expect("slice_model");

    let dsperse_path = converter::convert(
        &slices_dir,
        FormatType::Dsperse,
        None,
        false,
        true,
    )
    .expect("convert to dsperse");
    assert!(dsperse_path.exists(), "dsperse archive must exist");
    assert!(dsperse_path.extension().is_some_and(|e| e == "dsperse"));

    let restored_dir = tmp.path().join("restored");
    let restored = converter::convert(
        &dsperse_path,
        FormatType::Dirs,
        Some(&restored_dir),
        false,
        true,
    )
    .expect("convert back to dirs");
    assert!(restored.join("metadata.json").exists(), "restored metadata.json must exist");
}
