use dsperse::pipeline::CombinedRun;
use ndarray::{ArrayD, IxDyn};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: shape_probe <slices_dir>");
    let dir = std::path::Path::new(&dir);
    let meta = dsperse::pipeline::runner::load_model_metadata(dir).expect("metadata");
    let input_shape: Vec<usize> = meta.input_shape[0].iter().map(|&d| d as usize).collect();
    let input = ArrayD::from_elem(IxDyn(&input_shape), 0.5_f64);
    let run = CombinedRun::new(dir, input).expect("combined run");
    for slice_id in ["slice_11", "slice_393", "slice_354"] {
        let work = match run.circuit_work_for(slice_id) {
            Ok(w) => w,
            Err(e) => {
                println!("{slice_id}: {e}");
                continue;
            }
        };
        let meta_shapes = &work.slice_meta.input_shape;
        println!("{slice_id}: circuit_path={:?}", work.circuit_path);
        for (i, (name, t)) in work.named_inputs.iter().enumerate() {
            let meta = meta_shapes.get(i).cloned().unwrap_or_default();
            println!(
                "  input {i} {name}: runtime={:?} metadata={:?}",
                t.shape(),
                meta
            );
        }
        let Some(circuit_path) = work.circuit_path.as_ref() else {
            println!("  -> circuit_path MISSING");
            continue;
        };
        let backend = dsperse::backend::jstprove::JstproveBackend::new();
        let params = match backend.load_params(std::path::Path::new(circuit_path)) {
            Ok(Some(p)) => p,
            other => {
                println!("  -> load_params FAILED: {:?}", other.map(|o| o.is_some()));
                continue;
            }
        };
        let Some(ds) = work.slice_meta.dim_split.as_ref() else {
            println!("  -> no dim_split");
            continue;
        };
        let manifest_shapes: Vec<Vec<usize>> = work
            .named_inputs
            .iter()
            .map(|(_, t)| t.shape().to_vec())
            .collect();
        let contract: Vec<(String, Vec<usize>)> = params
            .inputs
            .iter()
            .map(|io| (io.name.clone(), io.shape.clone()))
            .collect();
        match dsperse::pipeline::plan_group_payload(&manifest_shapes, ds, &contract) {
            Ok(plan) => println!("  -> PLAN OK: {plan:?}"),
            Err(e) => println!("  -> PLAN ERR: {e}"),
        }
    }
}
