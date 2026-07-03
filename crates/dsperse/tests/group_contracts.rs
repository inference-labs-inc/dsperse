use dsperse::pipeline::{GroupPayloadPart, plan_group_payload};
use dsperse::schema::tiling::{DimSplitInfo, DimSplitKind};
use serde::Deserialize;

#[derive(Deserialize)]
struct ContractEntry {
    name: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Fixture {
    model: String,
    slice: usize,
    split_dim: usize,
    dim_size: usize,
    elements_per_group: usize,
    num_groups: usize,
    manifest_shapes: Vec<Vec<usize>>,
    contract: Vec<ContractEntry>,
}

#[test]
fn every_production_group_contract_plans_and_sizes_exactly() {
    let raw = include_str!("fixtures/group_contract_fixtures.json");
    let fixtures: Vec<Fixture> = serde_json::from_str(raw).expect("fixture json");
    assert!(fixtures.len() >= 300, "fixture corpus unexpectedly small");

    for f in &fixtures {
        let ds = DimSplitInfo {
            split_kind: DimSplitKind::BatchDim,
            slice_idx: f.slice,
            split_dim: f.split_dim,
            dim_size: f.dim_size,
            elements_per_group: f.elements_per_group,
            num_groups: f.num_groups,
            ..Default::default()
        };
        let contract: Vec<(String, Vec<usize>)> = f
            .contract
            .iter()
            .map(|c| (c.name.clone(), c.shape.clone()))
            .collect();
        let plan = plan_group_payload(&f.manifest_shapes, &ds, &contract)
            .unwrap_or_else(|e| panic!("{} slice_{}: {e}", f.model, f.slice));

        let planned_size: usize = plan
            .iter()
            .map(|p| match p {
                GroupPayloadPart::Whole(i) => f.manifest_shapes[*i].iter().product::<usize>(),
                GroupPayloadPart::Split(i) => {
                    f.manifest_shapes[*i].iter().product::<usize>() / f.dim_size
                        * f.elements_per_group
                }
            })
            .sum();
        let activation_size: usize = f.contract[..plan.len()]
            .iter()
            .map(|c| c.shape.iter().product::<usize>())
            .sum();
        assert_eq!(
            planned_size, activation_size,
            "{} slice_{}: planned {} vs contract activation prefix {}",
            f.model, f.slice, planned_size, activation_size
        );
    }
}
