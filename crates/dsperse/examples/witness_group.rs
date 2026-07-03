use dsperse::backend::jstprove::JstproveBackend;
use dsperse::pipeline::{CombinedRun, dim_split_group_payloads};
use ndarray::{ArrayD, IxDyn};
use std::path::Path;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: witness_group <slices_dir>");
    let dir = Path::new(&dir);

    let meta = dsperse::pipeline::runner::load_model_metadata(dir).expect("metadata");
    let input_shape: Vec<usize> = meta.input_shape[0].iter().map(|&d| d as usize).collect();
    let input = ArrayD::from_elem(IxDyn(&input_shape), 0.5_f64);
    let run = CombinedRun::new(dir, input).expect("combined run");

    let work = run.circuit_work_for("slice_11").expect("slice_11");
    let ds = work
        .slice_meta
        .dim_split
        .as_ref()
        .expect("dim_split metadata");

    let primary = work
        .named_inputs
        .iter()
        .find(|(n, _)| n == &ds.input_name)
        .map(|(_, t)| t)
        .expect("primary tensor in named_inputs");
    let secondaries: Vec<&ArrayD<f64>> = work
        .named_inputs
        .iter()
        .filter(|(n, _)| n != &ds.input_name)
        .map(|(_, t)| t)
        .collect();
    println!(
        "primary={:?} secondaries={}",
        primary.shape(),
        secondaries.len()
    );

    let payloads = dim_split_group_payloads(primary, &secondaries, ds).expect("group payloads");
    println!(
        "groups={} payload_len={}",
        payloads.len(),
        payloads[0].len()
    );

    let backend = JstproveBackend::new();
    let bundle = dir.join("slice_11/jstprove/circuit.bundle");
    for g in [0usize, 14, 28] {
        let result = backend.witness_f64(&bundle, &payloads[g], &[]);
        match result {
            Ok(w) => println!("group {g}: WITNESS OK ({} bytes)", w.len()),
            Err(e) => println!("group {g}: WITNESS FAILED: {e}"),
        }
    }
}
