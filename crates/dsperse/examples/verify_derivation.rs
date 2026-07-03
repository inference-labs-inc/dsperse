use dsperse::backend::jstprove::JstproveBackend;
use dsperse::pipeline::CombinedRun;
use ndarray::{ArrayD, IxDyn};
use std::path::Path;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: verify_derivation <slices_dir>");
    let dir = Path::new(&dir);

    let meta = dsperse::pipeline::runner::load_model_metadata(dir).expect("metadata");
    let input_shape: Vec<usize> = meta.input_shape[0].iter().map(|&d| d as usize).collect();
    let input = ArrayD::from_elem(IxDyn(&input_shape), 0.5_f64);

    let run = CombinedRun::new(dir, input).expect("combined run");

    let mut checked = 0usize;
    let mut mismatched = Vec::new();
    let all_ids: Vec<String> = (0..meta.slices.len())
        .map(|i| format!("slice_{}", meta.slices[i].index))
        .collect();
    for slice_id in all_ids {
        let work = match run.circuit_work_for(&slice_id) {
            Ok(w) => w,
            Err(e) => {
                mismatched.push((slice_id, 0usize, 0usize, format!("derive error: {e}")));
                continue;
            }
        };
        let expected: usize = work
            .slice_meta
            .input_shape
            .iter()
            .map(|shape| shape.iter().product::<i64>() as usize)
            .sum();
        let derived = work.input.len();
        checked += 1;
        if expected != 0 && expected != derived {
            mismatched.push((slice_id, expected, derived, String::new()));
        }
    }
    println!("checked={checked} mismatched={}", mismatched.len());
    for (id, e, d, err) in mismatched.iter().take(8) {
        println!("  {id}: expected={e} derived={d} {err}");
    }

    let bundle = dir.join("slice_11/jstprove/circuit.bundle");
    if bundle.exists() {
        let backend = JstproveBackend::new();
        let params = backend
            .load_params(&bundle)
            .expect("load params")
            .expect("params present");
        let work = run.circuit_work_for("slice_11").expect("slice_11 work");
        let initializer_count = if params.weights_as_inputs {
            let onnx = work.onnx_path.as_ref().expect("onnx path");
            dsperse::pipeline::extract_onnx_initializers(Path::new(onnx), &params)
                .expect("initializers")
                .len()
        } else {
            0
        };
        let activation_entries = params.inputs.len() - initializer_count;
        let bundle_expected: usize = params.inputs[..activation_entries]
            .iter()
            .map(|io| io.shape.iter().product::<usize>())
            .sum();
        println!(
            "slice_11 validator-preflight contract: weights_as_inputs={} total_entries={} initializers={} activation_expected={} derived_payload={} match={}",
            params.weights_as_inputs,
            params.inputs.len(),
            initializer_count,
            bundle_expected,
            work.input.len(),
            bundle_expected == work.input.len()
        );
    }
}
