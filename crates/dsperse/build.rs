fn main() {
    prost_build::Config::new()
        .compile_protos(&["proto/onnx.proto"], &["proto/"])
        .expect("Failed to compile ONNX proto");
}
