use std::io::Write;
use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use rmpv::Value;
use zip::write::SimpleFileOptions;

fn create_dslice_zip(path: &Path, metadata_bytes: &[u8], payload_content: &[u8]) {
    let file = std::fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("metadata.msgpack", opts).unwrap();
    zip.write_all(metadata_bytes).unwrap();

    zip.add_directory("payload/", opts).unwrap();
    zip.start_file("payload/model.onnx", opts).unwrap();
    zip.write_all(payload_content).unwrap();

    zip.finish().unwrap();
}

fn metadata_msgpack() -> Vec<u8> {
    use serde_json::json;
    let val = json!({
        "original_model": "test.onnx",
        "model_type": "ONNX",
        "input_shape": [[1, 3, 28, 28]],
        "output_shapes": [[1, 10]],
        "slice_points": [0],
        "slices": [{
            "index": 0,
            "filename": "slice_0",
            "path": "payload/model.onnx",
            "relative_path": "slice_0",
            "shape": {"tensor_shape": {"input": [[1, 3, 28, 28]], "output": [[1, 10]]}},
            "dependencies": {"input": [], "output": [], "filtered_inputs": []},
            "compilation": {
                "jstprove": {
                    "compiled": true,
                    "tiled": false,
                    "weights_as_inputs": false,
                    "files": {
                        "compiled": "model.compiled"
                    }
                }
            }
        }]
    });
    let meta: dsperse::schema::metadata::ModelMetadata =
        serde_json::from_value(val).unwrap();
    rmp_serde::to_vec_named(&meta).unwrap()
}

#[test]
fn extract_single_slice_from_dslice_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let slices_dir = tmp.path().join("slices");
    std::fs::create_dir_all(&slices_dir).unwrap();

    create_dslice_zip(
        &slices_dir.join("slice_0.dslice"),
        &metadata_msgpack(),
        b"fake onnx bytes",
    );

    let result =
        dsperse::archive::extract_single_slice(&slices_dir, "slice_0", None).unwrap();

    assert_eq!(result, slices_dir.join("slice_0"));
    assert!(result.join("metadata.msgpack").exists());
    assert!(result.join("payload").join("model.onnx").exists());

    let onnx = std::fs::read(result.join("payload").join("model.onnx")).unwrap();
    assert_eq!(onnx, b"fake onnx bytes");

    let result2 =
        dsperse::archive::extract_single_slice(&slices_dir, "slice_0", None).unwrap();
    assert_eq!(result, result2);
    assert!(result2.join("metadata.msgpack").exists());
}

#[test]
fn read_dslice_slice_metadata_field_access_pattern() {
    let tmp = tempfile::tempdir().unwrap();
    let dslice_path = tmp.path().join("slice_0.dslice");

    create_dslice_zip(&dslice_path, &metadata_msgpack(), b"fake onnx");

    let slice_meta = dsperse::archive::read_dslice_slice_metadata(&dslice_path).unwrap();

    assert!(slice_meta.compilation.jstprove.compiled);
    assert_eq!(
        slice_meta.compilation.jstprove.files.compiled.as_deref(),
        Some("model.compiled")
    );
    assert_eq!(slice_meta.path, "payload/model.onnx");
    assert_eq!(slice_meta.index, 0);
    assert_eq!(slice_meta.filename, "slice_0");
}

fn make_value_array(vals: &[f64]) -> Value {
    Value::Array(vals.iter().map(|&v| Value::F64(v)).collect())
}

fn make_value_2d(rows: &[&[f64]]) -> Value {
    Value::Array(rows.iter().map(|row| make_value_array(row)).collect())
}

fn make_value_3d(planes: &[&[&[f64]]]) -> Value {
    Value::Array(planes.iter().map(|plane| make_value_2d(plane)).collect())
}

fn make_value_4d(blocks: &[&[&[&[f64]]]]) -> Value {
    Value::Array(blocks.iter().map(|block| make_value_3d(block)).collect())
}

#[test]
fn value_arrayd_roundtrip_1d() {
    let input = make_value_array(&[1.0, 2.0, 3.0, 4.0]);
    let arr = dsperse::utils::io::value_to_arrayd(&input).unwrap();
    assert_eq!(arr.shape(), &[4]);
    assert_eq!(arr[IxDyn(&[0])], 1.0);
    assert_eq!(arr[IxDyn(&[3])], 4.0);

    let output = dsperse::utils::io::arrayd_to_value(&arr);
    assert_eq!(output, input);
}

#[test]
fn value_arrayd_roundtrip_2d() {
    let input = make_value_2d(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let arr = dsperse::utils::io::value_to_arrayd(&input).unwrap();
    assert_eq!(arr.shape(), &[2, 2]);
    assert_eq!(arr[IxDyn(&[0, 0])], 1.0);
    assert_eq!(arr[IxDyn(&[1, 1])], 4.0);

    let output = dsperse::utils::io::arrayd_to_value(&arr);
    assert_eq!(output, input);
}

#[test]
fn value_arrayd_roundtrip_3d() {
    let input = make_value_3d(&[
        &[&[1.0, 2.0], &[3.0, 4.0]],
        &[&[5.0, 6.0], &[7.0, 8.0]],
    ]);
    let arr = dsperse::utils::io::value_to_arrayd(&input).unwrap();
    assert_eq!(arr.shape(), &[2, 2, 2]);
    assert_eq!(arr[IxDyn(&[0, 0, 0])], 1.0);
    assert_eq!(arr[IxDyn(&[1, 1, 1])], 8.0);

    let output = dsperse::utils::io::arrayd_to_value(&arr);
    assert_eq!(output, input);
}

#[test]
fn value_arrayd_roundtrip_4d() {
    let input = make_value_4d(&[&[
        &[&[0.5, 1.5], &[2.5, 3.5]],
        &[&[4.5, 5.5], &[6.5, 7.5]],
    ]]);
    let arr = dsperse::utils::io::value_to_arrayd(&input).unwrap();
    assert_eq!(arr.shape(), &[1, 2, 2, 2]);
    assert_eq!(arr[IxDyn(&[0, 0, 0, 0])], 0.5);
    assert_eq!(arr[IxDyn(&[0, 1, 1, 1])], 7.5);

    let output = dsperse::utils::io::arrayd_to_value(&arr);
    assert_eq!(output, input);
}

#[test]
fn value_arrayd_full_roundtrip_preserves_values() {
    let original = make_value_2d(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let arr = dsperse::utils::io::value_to_arrayd(&original).unwrap();
    let reconstructed = dsperse::utils::io::arrayd_to_value(&arr);
    let arr2 = dsperse::utils::io::value_to_arrayd(&reconstructed).unwrap();

    assert_eq!(arr.shape(), arr2.shape());
    assert_eq!(arr, arr2);
    assert_eq!(original, reconstructed);
}

#[test]
fn extract_input_data_key_precedence() {
    let val = Value::Map(vec![
        (Value::String("input_data".into()), make_value_array(&[1.0])),
        (Value::String("input".into()), make_value_array(&[2.0])),
        (Value::String("data".into()), make_value_array(&[3.0])),
        (Value::String("inputs".into()), make_value_array(&[4.0])),
    ]);
    let extracted = dsperse::utils::io::extract_input_data(&val).unwrap();
    assert_eq!(extracted, &make_value_array(&[1.0]));
}

#[test]
fn extract_input_data_fallback_to_input() {
    let val = Value::Map(vec![
        (Value::String("input".into()), make_value_array(&[2.0])),
        (Value::String("data".into()), make_value_array(&[3.0])),
        (Value::String("inputs".into()), make_value_array(&[4.0])),
    ]);
    let extracted = dsperse::utils::io::extract_input_data(&val).unwrap();
    assert_eq!(extracted, &make_value_array(&[2.0]));
}

#[test]
fn extract_input_data_fallback_to_data() {
    let val = Value::Map(vec![
        (Value::String("data".into()), make_value_array(&[3.0])),
        (Value::String("inputs".into()), make_value_array(&[4.0])),
    ]);
    let extracted = dsperse::utils::io::extract_input_data(&val).unwrap();
    assert_eq!(extracted, &make_value_array(&[3.0]));
}

#[test]
fn extract_input_data_fallback_to_inputs() {
    let val = Value::Map(vec![
        (Value::String("inputs".into()), make_value_array(&[4.0])),
    ]);
    let extracted = dsperse::utils::io::extract_input_data(&val).unwrap();
    assert_eq!(extracted, &make_value_array(&[4.0]));
}

#[test]
fn extract_input_data_returns_none_for_unrecognized_keys() {
    let val = Value::Map(vec![
        (Value::String("tensor".into()), make_value_array(&[1.0])),
        (Value::String("x".into()), make_value_array(&[2.0])),
    ]);
    assert!(dsperse::utils::io::extract_input_data(&val).is_none());
}

#[test]
fn slice_dir_path_formats_correctly() {
    let root = Path::new("/some/root");
    assert_eq!(
        dsperse::utils::paths::slice_dir_path(root, 0),
        Path::new("/some/root/slice_0")
    );
    assert_eq!(
        dsperse::utils::paths::slice_dir_path(root, 5),
        Path::new("/some/root/slice_5")
    );
    assert_eq!(
        dsperse::utils::paths::slice_dir_path(root, 42),
        Path::new("/some/root/slice_42")
    );
}

#[test]
fn extract_single_slice_combined_with_metadata_read() {
    let tmp = tempfile::tempdir().unwrap();
    let slices_dir = tmp.path().join("slices");
    std::fs::create_dir_all(&slices_dir).unwrap();

    let meta = metadata_msgpack();
    create_dslice_zip(&slices_dir.join("slice_0.dslice"), &meta, b"onnx payload");

    let slice_idx: usize = 0;
    let slice_id = format!("slice_{slice_idx}");

    dsperse::archive::extract_single_slice(&slices_dir, &slice_id, None).unwrap();

    let dslice_file = slices_dir.join(format!("{slice_id}.dslice"));
    let slice_meta = dsperse::archive::read_dslice_slice_metadata(&dslice_file).unwrap();

    assert!(slice_meta.compilation.jstprove.compiled);

    let slice_dir = dsperse::utils::paths::slice_dir_path(&slices_dir, slice_idx);

    let compiled = slice_meta
        .compilation
        .jstprove
        .files
        .compiled
        .as_ref()
        .unwrap();
    assert_eq!(compiled, "model.compiled");

    let onnx_path = slice_dir.join(&slice_meta.path);
    assert!(onnx_path.exists());
    assert_eq!(onnx_path, slices_dir.join("slice_0/payload/model.onnx"));
}

#[test]
fn arrayd_to_value_then_extract_input_data_integration() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    let tensor_val = dsperse::utils::io::arrayd_to_value(&arr);
    let wrapped = Value::Map(vec![
        (Value::String("input_data".into()), tensor_val),
    ]);

    let extracted = dsperse::utils::io::extract_input_data(&wrapped).unwrap();
    let roundtripped = dsperse::utils::io::value_to_arrayd(extracted).unwrap();
    assert_eq!(arr.shape(), roundtripped.shape());
    assert_eq!(arr, roundtripped);
}
