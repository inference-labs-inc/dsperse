use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use rmpv::Value;

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
    let input = make_value_3d(&[&[&[1.0, 2.0], &[3.0, 4.0]], &[&[5.0, 6.0], &[7.0, 8.0]]]);
    let arr = dsperse::utils::io::value_to_arrayd(&input).unwrap();
    assert_eq!(arr.shape(), &[2, 2, 2]);
    assert_eq!(arr[IxDyn(&[0, 0, 0])], 1.0);
    assert_eq!(arr[IxDyn(&[1, 1, 1])], 8.0);

    let output = dsperse::utils::io::arrayd_to_value(&arr);
    assert_eq!(output, input);
}

#[test]
fn value_arrayd_roundtrip_4d() {
    let input = make_value_4d(&[&[&[&[0.5, 1.5], &[2.5, 3.5]], &[&[4.5, 5.5], &[6.5, 7.5]]]]);
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
    let val = Value::Map(vec![(
        Value::String("inputs".into()),
        make_value_array(&[4.0]),
    )]);
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
fn arrayd_to_value_then_extract_input_data_integration() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    let tensor_val = dsperse::utils::io::arrayd_to_value(&arr);
    let wrapped = Value::Map(vec![(Value::String("input_data".into()), tensor_val)]);

    let extracted = dsperse::utils::io::extract_input_data(&wrapped).unwrap();
    let roundtripped = dsperse::utils::io::value_to_arrayd(extracted).unwrap();
    assert_eq!(arr.shape(), roundtripped.shape());
    assert_eq!(arr, roundtripped);
}
