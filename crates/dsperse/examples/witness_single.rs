use dsperse::backend::jstprove::JstproveBackend;
use dsperse::pipeline::CombinedRun;
use ndarray::{ArrayD, IxDyn};
use std::path::Path;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: witness_single <slices_dir> <slice_id>");
    let dir = Path::new(&dir);
    let slice_id = std::env::args().nth(2).expect("slice id");
    let norm: f64 = std::env::var("NORM_DIVISOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);

    let meta = dsperse::pipeline::runner::load_model_metadata(dir).expect("metadata");
    let input_shape: Vec<usize> = meta.input_shape[0].iter().map(|&d| d as usize).collect();
    let input = ArrayD::from_elem(IxDyn(&input_shape), 0.5_f64);
    let run = CombinedRun::new(dir, input).expect("combined run");
    let work = run.circuit_work_for(&slice_id).expect("slice work");

    let local_mag: f64 = work.input.iter().map(|v| v.abs()).sum::<f64>() / work.input.len() as f64;
    println!(
        "slice input elems={} mean_abs={local_mag:.4e} norm={norm}",
        work.input.len()
    );

    let activations: Vec<f64> = work.input.iter().map(|v| v / norm).collect();
    let backend = JstproveBackend::new();
    let circuit_path = work.circuit_path.as_deref().expect("circuit path");
    let params = backend
        .load_params(Path::new(circuit_path))
        .expect("params")
        .expect("params present");
    let inits = if params.weights_as_inputs {
        let onnx_path = work.onnx_path.as_deref().expect("onnx path");
        dsperse::pipeline::runner::extract_onnx_initializers(Path::new(onnx_path), &params)
            .expect("initializers")
    } else {
        Vec::new()
    };
    let w = backend
        .witness_f64(Path::new(circuit_path), &activations, &inits)
        .expect("witness");
    let dims = params.effective_input_dims();
    println!(
        "effective_input_dims={dims} activations={}",
        activations.len()
    );
    let outputs = backend.extract_outputs(&w, dims).expect("outputs");
    let nonzero = outputs.iter().filter(|v| **v != 0.0).count();
    let mag: f64 = outputs.iter().map(|v| v.abs()).sum::<f64>() / outputs.len().max(1) as f64;
    println!(
        "witness outputs={} nonzero={} mean_abs={mag:.4e} sample={:?}",
        outputs.len(),
        nonzero,
        &outputs[..outputs.len().min(5)]
    );
}
