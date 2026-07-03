use dsperse::backend::jstprove::JstproveBackend;
use std::path::Path;

fn main() {
    let bundle = std::env::args()
        .nth(1)
        .expect("usage: inspect_bundle <circuit.bundle>");
    let backend = JstproveBackend::new();
    let params = backend
        .load_params(Path::new(&bundle))
        .expect("load")
        .expect("present");
    println!("weights_as_inputs={}", params.weights_as_inputs);
    for (i, io) in params.inputs.iter().enumerate() {
        println!(
            "input[{i}]: shape={:?} elems={}",
            io.shape,
            io.shape.iter().product::<usize>()
        );
    }
    for (i, io) in params.outputs.iter().enumerate() {
        println!(
            "output[{i}]: shape={:?} elems={}",
            io.shape,
            io.shape.iter().product::<usize>()
        );
    }
}
