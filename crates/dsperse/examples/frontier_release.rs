use dsperse::pipeline::CombinedRun;
use ndarray::{ArrayD, IxDyn};
use std::path::Path;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: frontier_release <slices_dir>");
    let dir = Path::new(&dir);
    let meta = dsperse::pipeline::runner::load_model_metadata(dir).expect("metadata");
    let input_shape: Vec<usize> = meta.input_shape[0].iter().map(|&d| d as usize).collect();
    let input = ArrayD::from_elem(IxDyn(&input_shape), 0.5_f64);
    let mut run = CombinedRun::new(dir, input).expect("combined run");

    let ids = run.circuit_work_ids();
    let start_tensors = run.tensor_count();
    let start_elems = run.tensor_elements();
    for id in &ids {
        run.circuit_work_for(id)
            .expect("work materializes before completion");
        run.mark_slice_done(id);
    }
    let end_tensors = run.tensor_count();
    let end_elems = run.tensor_elements();
    println!(
        "tensors {start_tensors} -> {end_tensors}, elements {start_elems} -> {end_elems}, freed {:.1}%",
        100.0 * (start_elems.saturating_sub(end_elems)) as f64 / start_elems.max(1) as f64
    );
    assert!(
        end_elems * 5 < start_elems.max(1),
        "expected at least 80% freed"
    );
    println!("FRONTIER RELEASE OK");
}
