use std::path::Path;

use dsperse::archive::converter::{self, FormatType};

fn test_models_dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models"))
}

#[test]
fn convert_dirs_to_dslice_roundtrip() {
    let model_path = test_models_dir().join("net/model.onnx");
    if !model_path.exists() {
        eprintln!("skipping: test model not found at {}", model_path.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let slices_dir = tmp.path().join("slices");
    dsperse::slicer::slice_model(&model_path, Some(&slices_dir), None).unwrap();

    let dslice_out = converter::convert(
        &slices_dir,
        FormatType::Dslice,
        None,
        false,
        true,
    )
    .unwrap();
    assert!(dslice_out.exists());

    let has_dslice_files = std::fs::read_dir(&dslice_out)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().map_or(false, |ext| ext == "dslice"));
    assert!(has_dslice_files, "dslice files must be produced");
}

#[test]
fn convert_dirs_to_dsperse_roundtrip() {
    let model_path = test_models_dir().join("net/model.onnx");
    if !model_path.exists() {
        eprintln!("skipping: test model not found at {}", model_path.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let slices_dir = tmp.path().join("slices");
    dsperse::slicer::slice_model(&model_path, Some(&slices_dir), None).unwrap();

    let dsperse_path = converter::convert(
        &slices_dir,
        FormatType::Dsperse,
        None,
        false,
        true,
    )
    .unwrap();
    assert!(dsperse_path.exists());
    assert!(dsperse_path.extension().map_or(false, |e| e == "dsperse"));

    let restored_dir = tmp.path().join("restored");
    let restored = converter::convert(
        &dsperse_path,
        FormatType::Dirs,
        Some(&restored_dir),
        false,
        true,
    )
    .unwrap();
    assert!(restored.join("metadata.json").exists());
}
