use dsperse::backend::jstprove::JstproveBackend;
use dsperse::pipeline::CombinedRun;
use ndarray::{ArrayD, IxDyn};
use std::path::Path;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: witness_group <slices_dir>");
    let dir = Path::new(&dir);

    let meta = dsperse::pipeline::runner::load_model_metadata(dir).expect("metadata");
    let input_shape: Vec<usize> = meta.input_shape[0].iter().map(|&d| d as usize).collect();
    let norm: f64 = std::env::var("NORM_DIVISOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let input = ArrayD::from_elem(IxDyn(&input_shape), 0.5_f64 / norm);
    let run = CombinedRun::new(dir, input).expect("combined run");

    let slice_id = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "slice_11".to_string());
    let work = run.circuit_work_for(&slice_id).expect("slice work");
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

    let backend_for_params = JstproveBackend::new();
    let params = backend_for_params
        .load_params(&dir.join(&slice_id).join("jstprove/circuit.bundle"))
        .expect("params")
        .expect("params present");
    let manifest_shapes: Vec<Vec<usize>> = work
        .slice_meta
        .input_shape
        .iter()
        .map(|s| s.iter().map(|&d| d as usize).collect())
        .collect();
    let contract: Vec<(String, Vec<usize>)> = params
        .inputs
        .iter()
        .map(|io| (io.name.clone(), io.shape.clone()))
        .collect();
    let plan =
        dsperse::pipeline::plan_group_payload(&manifest_shapes, ds, &contract).expect("plan");
    let tensors: Vec<&ArrayD<f64>> = work.named_inputs.iter().map(|(_, t)| t).collect();
    let payloads = dsperse::pipeline::dim_split_group_payloads_planned(&tensors, &plan, ds)
        .expect("group payloads");
    let _ = (primary, &secondaries);
    println!(
        "groups={} payload_len={}",
        payloads.len(),
        payloads[0].len()
    );

    let backend = JstproveBackend::new();
    let bundle = dir.join(&slice_id).join("jstprove/circuit.bundle");
    let last = payloads.len() - 1;
    for g in [0usize, last / 2, last] {
        let result = backend.witness_f64(&bundle, &payloads[g], &[]);
        match result {
            Ok(w) => println!("group {g}: WITNESS OK ({} bytes)", w.len()),
            Err(e) => println!("group {g}: WITNESS FAILED: {e}"),
        }
    }
}
