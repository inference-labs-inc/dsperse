use dsperse::backend::jstprove::JstproveBackend;
use dsperse::pipeline::runner::load_model_metadata;
use std::fmt::Write as _;
use std::path::Path;

fn main() {
    let root = std::env::args()
        .nth(1)
        .expect("usage: extract_group_fixtures <audit_dir>");
    let backend = JstproveBackend::new();
    let mut out = String::from("[\n");
    let mut n = 0usize;
    let mut plan_failures = 0usize;

    let mut model_dirs: Vec<_> = std::fs::read_dir(&root)
        .expect("audit dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    model_dirs.sort();

    for model_dir in model_dirs {
        let slices_dir = model_dir.join("slices");
        if !slices_dir.join("metadata.msgpack").exists() {
            continue;
        }
        let meta = match load_model_metadata(&slices_dir) {
            Ok(m) => m,
            Err(_) => continue,
        };
        for slice in &meta.slices {
            let ds = match &slice.dim_split {
                Some(d) if d.weight_name.is_none() => d,
                _ => continue,
            };
            let inputs = &slice.dependencies.filtered_inputs;
            let shapes = &slice.shape.tensor_shape.input;
            if inputs.len() < 2 || shapes.len() != inputs.len() {
                continue;
            }
            let bundle = slices_dir
                .join(format!("slice_{}", slice.index))
                .join("jstprove/circuit.bundle");
            let params = match backend.load_params(Path::new(&bundle)) {
                Ok(Some(p)) => p,
                _ => continue,
            };
            let manifest_shapes: Vec<Vec<usize>> = shapes
                .iter()
                .map(|s| s.iter().map(|&d| d as usize).collect())
                .collect();
            let contract: Vec<(String, Vec<usize>)> = params
                .inputs
                .iter()
                .map(|io| (io.name.clone(), io.shape.clone()))
                .collect();
            match dsperse::pipeline::plan_group_payload(&manifest_shapes, ds, &contract) {
                Ok(_) => {}
                Err(e) => {
                    plan_failures += 1;
                    eprintln!(
                        "PLAN FAIL {:?} slice_{}: {e}",
                        model_dir.file_name(),
                        slice.index
                    );
                }
            }
            let model = model_dir.file_name().unwrap().to_string_lossy();
            if n > 0 {
                out.push_str(",\n");
            }
            let _ = write!(
                out,
                "{{\"model\":\"{model}\",\"slice\":{},\"split_dim\":{},\"dim_size\":{},\"elements_per_group\":{},\"num_groups\":{},\"manifest_shapes\":{:?},\"contract\":[{}]}}",
                slice.index,
                ds.split_dim,
                ds.dim_size,
                ds.elements_per_group,
                ds.num_groups,
                manifest_shapes,
                contract
                    .iter()
                    .map(|(name, shape)| format!("{{\"name\":\"{name}\",\"shape\":{shape:?}}}"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            n += 1;
        }
    }
    out.push_str("\n]\n");
    std::fs::write("group_contract_fixtures.json", &out).expect("write fixtures");
    eprintln!("fixtures={n} plan_failures={plan_failures}");
}
