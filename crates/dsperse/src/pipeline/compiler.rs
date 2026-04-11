use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::backend::jstprove::JstproveBackend;
use crate::converter;
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;
use crate::slicer::autotiler::estimate_slice_constraints;
use crate::slicer::onnx_proto;
use crate::utils::paths::{find_metadata_path, slice_dir_path};

type CircuitCache = std::sync::Mutex<HashMap<String, PathBuf>>;

enum CompileOutcome {
    Compiled,
    CompiledChannelSplit {
        group_circuits: Vec<(usize, String)>,
    },
    CompiledDimSplit,
    Skipped,
    SkippedOverSize {
        estimated: u64,
        threshold: u64,
    },
}

/// Summary of a compile_slices invocation.  The pass returns Ok
/// even when individual slice compilations fail, so callers must
/// inspect `failed` to decide whether to proceed (e.g. allow
/// partial-coverage ONNX fallback) or abort.  Keeping the
/// compiled count explicit lets the CLI / analyze command
/// report a structured summary instead of inferring success from
/// log lines.
#[derive(Debug, Default)]
pub struct CompileReport {
    pub compiled: usize,
    pub failed: Vec<(usize, DsperseError)>,
}

impl CompileReport {
    pub fn ok_if_no_failures(self) -> Result<Self> {
        if self.failed.is_empty() {
            Ok(self)
        } else {
            Err(DsperseError::Pipeline(format!(
                "compile_slices: {} slice(s) failed to compile; set --allow-onnx-fallback to proceed with partial coverage",
                self.failed.len()
            )))
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compile_slices(
    slices_dir: &Path,
    backend: &JstproveBackend,
    proof_config: jstprove_circuits::api::ProofConfigType,
    parallel: usize,
    weights_as_inputs: bool,
    layers: Option<&[usize]>,
    jstprove_ops: &[&str],
    skip_compile_over_size: Option<u64>,
) -> Result<CompileReport> {
    let meta_path = find_metadata_path(slices_dir).ok_or_else(|| {
        DsperseError::Metadata(format!(
            "no {} found in slices directory",
            crate::utils::paths::METADATA_FILE
        ))
    })?;
    let mut metadata = ModelMetadata::load(&meta_path)?;

    if metadata.original_model_path.is_some() {
        crate::slicer::materializer::ensure_all_slices_materialized(slices_dir, &metadata)?;
    }

    let mut metadata_dirty = false;
    for slice in &mut metadata.slices {
        if let Some(ref mut cs) = slice.channel_split
            && cs.groups.is_empty()
        {
            let populated = populate_channel_split_groups(slices_dir, slice.index, cs)?;
            if populated {
                metadata_dirty = true;
            }
        }
        if let Some(ref mut ds) = slice.dim_split
            && ds.template_path.is_none()
        {
            let tmpl_rel = format!("slice_{}/payload/dim_template.onnx", slice.index);
            if slices_dir.join(&tmpl_rel).exists() {
                ds.template_path = Some(tmpl_rel);
                metadata_dirty = true;
            }
        }
    }
    // Strip dim_split metadata from slices where template creation failed
    // (axis-separability rejection, unsupported split kind). Leaving stale
    // dim_split entries in the metadata causes downstream runners and the
    // packager to emit bundles that fail at the strategy validation stage
    // ("dim_split present but template_path is missing").
    for slice in &mut metadata.slices {
        if slice
            .dim_split
            .as_ref()
            .is_some_and(|ds| ds.template_path.is_none())
        {
            tracing::info!(
                slice = slice.index,
                "stripping dim_split metadata (no template materialized)"
            );
            slice.dim_split = None;
            metadata_dirty = true;
        }
    }
    if metadata_dirty {
        metadata.save(&meta_path)?;
        tracing::info!("persisted materialized split groups to metadata");
    }

    let slices: Vec<_> = metadata
        .slices
        .iter()
        .filter(|s| layers.is_none_or(|l| l.contains(&s.index)))
        .cloned()
        .collect();

    tracing::info!(total = slices.len(), "compiling slices");

    let exclude_from_wai: std::collections::HashSet<String> =
        metadata.folded_constant_names.iter().cloned().collect();

    let traced_shapes = metadata.traced_shapes.clone();
    let traced_ref = traced_shapes.as_ref();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let compiled_count = std::sync::atomic::AtomicUsize::new(0);
    let meta_mutex = std::sync::Mutex::new((&mut metadata, false));
    let errors: std::sync::Mutex<Vec<(usize, DsperseError)>> = std::sync::Mutex::new(Vec::new());
    let circuit_cache: CircuitCache = std::sync::Mutex::new(HashMap::new());

    pool.install(|| {
        slices.par_iter().for_each(|slice| {
            let r = compile_single_slice(
                slices_dir,
                slice,
                backend,
                proof_config,
                weights_as_inputs,
                jstprove_ops,
                &exclude_from_wai,
                skip_compile_over_size,
                &circuit_cache,
                traced_ref,
            );
            match r {
                Ok(CompileOutcome::Compiled) => {
                    let count =
                        compiled_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    tracing::info!(slice = slice.index, count, "compiled");
                }
                Ok(CompileOutcome::CompiledChannelSplit { group_circuits }) => {
                    let count =
                        compiled_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    tracing::info!(
                        slice = slice.index,
                        groups = group_circuits.len(),
                        count,
                        "compiled channel split groups"
                    );
                    let mut guard = meta_mutex.lock().unwrap();
                    let (ref mut meta, ref mut dirty) = *guard;
                    if let Some(s) = meta.slices.iter_mut().find(|s| s.index == slice.index)
                        && let Some(ref mut cs) = s.channel_split
                    {
                        for (group_idx, circuit_path) in &group_circuits {
                            if let Some(group) =
                                cs.groups.iter_mut().find(|g| g.group_idx == *group_idx)
                            {
                                group.jstprove_circuit_path = Some(circuit_path.clone());
                            }
                        }
                        *dirty = true;
                    }
                }
                Ok(CompileOutcome::CompiledDimSplit) => {
                    let count =
                        compiled_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    tracing::info!(slice = slice.index, count, "compiled dim-split template");
                    let mut guard = meta_mutex.lock().unwrap();
                    let (ref mut meta, ref mut dirty) = *guard;
                    if let Some(s) = meta.slices.iter_mut().find(|s| s.index == slice.index)
                        && let Some(ref mut ds) = s.dim_split
                    {
                        ds.jstprove_circuit_path = Some(format!(
                            "slice_{}/jstprove/dim_split/circuit.bundle",
                            slice.index
                        ));
                        *dirty = true;
                    }
                }
                Ok(CompileOutcome::Skipped) => {
                    tracing::info!(slice = slice.index, "skipped (unsupported ops)")
                }
                Ok(CompileOutcome::SkippedOverSize {
                    estimated,
                    threshold,
                }) => {
                    tracing::info!(
                        slice = slice.index,
                        estimated,
                        threshold,
                        "skipped (estimated constraints exceed threshold)"
                    )
                }
                Err(e) => {
                    tracing::error!(slice = slice.index, error = %e, "compilation failed");
                    errors.lock().unwrap().push((slice.index, e));
                }
            }
        });
    });

    let errors = errors.into_inner().unwrap();
    let (metadata, cs_dirty) = meta_mutex.into_inner().unwrap();
    if cs_dirty {
        if let Err(e) = metadata.save(&meta_path) {
            tracing::error!(error = %e, "failed to persist split circuit paths");
        } else {
            tracing::info!("persisted split circuit paths to metadata");
        }
    }
    let compiled_count = compiled_count.load(std::sync::atomic::Ordering::Relaxed);

    if errors.is_empty() {
        tracing::info!(count = compiled_count, "all slices compiled");
    } else {
        tracing::warn!(
            compiled = compiled_count,
            failed = errors.len(),
            "compilation completed with errors; failed slices fall back to ONNX execution if the caller allows partial coverage"
        );
        for (idx, e) in &errors {
            tracing::warn!(slice = idx, error = %e, "slice compilation failed");
        }
    }
    Ok(CompileReport {
        compiled: compiled_count,
        failed: errors,
    })
}

struct SliceAnalysis {
    compatible: bool,
    data_movement_only: bool,
}

const DATA_MOVEMENT_OPS: &[&str] = &[
    "Reshape",
    "Transpose",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Identity",
    "Concat",
    "Split",
    "Gather",
    "Slice",
    "Expand",
    "Tile",
    "Cast",
];

fn analyze_slice_onnx(onnx_path: &Path, jstprove_ops: &[&str]) -> Result<SliceAnalysis> {
    let model = onnx_proto::load_model(onnx_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Slicer(format!("no graph in {}", onnx_path.display())))?;
    let compatible = graph
        .node
        .iter()
        .all(|n| jstprove_ops.contains(&n.op_type.as_str()));
    let data_movement_only = !graph.node.is_empty()
        && graph
            .node
            .iter()
            .all(|n| DATA_MOVEMENT_OPS.contains(&n.op_type.as_str()));
    Ok(SliceAnalysis {
        compatible,
        data_movement_only,
    })
}

pub(super) fn compute_circuit_signature(tmpl_path: &Path, curve: Option<&str>) -> Result<String> {
    use sha2::{Digest, Sha256};

    fn hash_bytes(hasher: &mut Sha256, b: &[u8]) {
        hasher.update((b.len() as u64).to_le_bytes());
        hasher.update(b);
    }

    let model = onnx_proto::load_model(tmpl_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Slicer("no graph for signature".into()))?;
    let mut hasher = Sha256::new();
    if let Some(c) = curve {
        hash_bytes(&mut hasher, c.as_bytes());
    }
    hasher.update((graph.node.len() as u64).to_le_bytes());
    for node in &graph.node {
        hash_bytes(&mut hasher, node.op_type.as_bytes());
        hasher.update((node.input.len() as u64).to_le_bytes());
        for inp in &node.input {
            hash_bytes(&mut hasher, inp.as_bytes());
        }
        hasher.update((node.output.len() as u64).to_le_bytes());
        for out in &node.output {
            hash_bytes(&mut hasher, out.as_bytes());
        }
        hasher.update((node.attribute.len() as u64).to_le_bytes());
        for attr in &node.attribute {
            hash_bytes(&mut hasher, attr.name.as_bytes());
            hasher.update(attr.r#type.to_le_bytes());
            hasher.update(attr.i.to_le_bytes());
            hasher.update(attr.f.to_le_bytes());
            hash_bytes(&mut hasher, &attr.s);
            hasher.update((attr.ints.len() as u64).to_le_bytes());
            for v in &attr.ints {
                hasher.update(v.to_le_bytes());
            }
            hasher.update((attr.floats.len() as u64).to_le_bytes());
            for v in &attr.floats {
                hasher.update(v.to_le_bytes());
            }
            hasher.update((attr.strings.len() as u64).to_le_bytes());
            for v in &attr.strings {
                hash_bytes(&mut hasher, v);
            }
        }
    }
    let init_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|i| i.name.as_str()).collect();
    for vi in &graph.input {
        if init_names.contains(vi.name.as_str()) {
            continue;
        }
        if let Some(shape) = onnx_proto::shape_from_value_info(vi) {
            hasher.update((shape.len() as u64).to_le_bytes());
            for d in &shape {
                hasher.update(d.to_le_bytes());
            }
        }
        if let Some(dt) = onnx_proto::elem_type_from_value_info(vi) {
            hasher.update(dt.to_le_bytes());
        }
    }
    for vi in &graph.output {
        if let Some(shape) = onnx_proto::shape_from_value_info(vi) {
            hasher.update((shape.len() as u64).to_le_bytes());
            for d in &shape {
                hasher.update(d.to_le_bytes());
            }
        }
        if let Some(dt) = onnx_proto::elem_type_from_value_info(vi) {
            hasher.update(dt.to_le_bytes());
        }
    }
    hasher.update((graph.initializer.len() as u64).to_le_bytes());
    for init in &graph.initializer {
        hasher.update((init.dims.len() as u64).to_le_bytes());
        for d in &init.dims {
            hasher.update(d.to_le_bytes());
        }
        hasher.update(init.data_type.to_le_bytes());
    }
    let hash = hasher.finalize();
    Ok(format!("{:x}", hash))
}

fn summarize_onnx_ops(onnx_path: &Path) -> String {
    let model = match onnx_proto::load_model(onnx_path) {
        Ok(m) => m,
        Err(_) => return String::from("?"),
    };
    let graph = match model.graph.as_ref() {
        Some(g) => g,
        None => return String::from("?"),
    };
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for node in &graph.node {
        *counts.entry(node.op_type.as_str()).or_default() += 1;
    }
    counts
        .iter()
        .map(|(op, n)| {
            if *n > 1 {
                format!("{op}x{n}")
            } else {
                op.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Debug, serde::Serialize)]
pub struct SliceAnalysisReport {
    pub index: usize,
    pub backend: String,
    pub reason: String,
    pub estimated_constraints: Option<u64>,
    pub ops: String,
    pub tiled: bool,
    pub channel_split: bool,
    pub dim_split: bool,
    pub circuit_signature: Option<String>,
}

pub fn analyze_slices(
    slices_dir: &Path,
    jstprove_ops: &[&str],
    skip_compile_over_size: Option<u64>,
    proof_config: Option<&str>,
) -> Result<Vec<SliceAnalysisReport>> {
    let meta_path = find_metadata_path(slices_dir).ok_or_else(|| {
        DsperseError::Metadata(format!(
            "no {} found in slices directory",
            crate::utils::paths::METADATA_FILE
        ))
    })?;
    let metadata = ModelMetadata::load(&meta_path)?;
    let mut reports = Vec::with_capacity(metadata.slices.len());

    for slice in &metadata.slices {
        let slice_dir = slice_dir_path(slices_dir, slice.index);
        if !slice_dir.exists() {
            reports.push(SliceAnalysisReport {
                index: slice.index,
                backend: "missing".into(),
                reason: "slice directory not found".into(),
                estimated_constraints: None,
                ops: String::new(),
                tiled: slice.tiling.is_some(),
                channel_split: slice.channel_split.is_some(),
                dim_split: slice.dim_split.is_some(),
                circuit_signature: None,
            });
            continue;
        }

        if slice
            .channel_split
            .as_ref()
            .is_some_and(|cs| !cs.groups.is_empty())
        {
            reports.push(SliceAnalysisReport {
                index: slice.index,
                backend: "jstprove".into(),
                reason: "channel-split".into(),
                estimated_constraints: None,
                ops: String::new(),
                tiled: slice.tiling.is_some(),
                channel_split: true,
                dim_split: false,
                circuit_signature: None,
            });
            continue;
        }

        if let Some(ref ds) = slice.dim_split
            && ds.template_path.is_some()
        {
            reports.push(SliceAnalysisReport {
                index: slice.index,
                backend: "jstprove".into(),
                reason: "dim-split".into(),
                estimated_constraints: None,
                ops: String::new(),
                tiled: slice.tiling.is_some(),
                channel_split: false,
                dim_split: true,
                circuit_signature: None,
            });
            continue;
        }

        let onnx_path = match resolve_compile_onnx(slices_dir, slice) {
            Ok(p) => p,
            Err(_) => {
                reports.push(SliceAnalysisReport {
                    index: slice.index,
                    backend: "onnx".into(),
                    reason: "onnx not found".into(),
                    estimated_constraints: None,
                    ops: String::new(),
                    tiled: slice.tiling.is_some(),
                    channel_split: false,
                    dim_split: false,
                    circuit_signature: None,
                });
                continue;
            }
        };

        if !onnx_path.exists() {
            reports.push(SliceAnalysisReport {
                index: slice.index,
                backend: "onnx".into(),
                reason: "onnx not found".into(),
                estimated_constraints: None,
                ops: String::new(),
                tiled: slice.tiling.is_some(),
                channel_split: false,
                dim_split: false,
                circuit_signature: None,
            });
            continue;
        }

        let ops = summarize_onnx_ops(&onnx_path);
        let analysis = analyze_slice_onnx(&onnx_path, jstprove_ops);
        let estimated = estimate_onnx_constraints(&onnx_path).ok();
        let sig = compute_circuit_signature(&onnx_path, proof_config).ok();

        let (backend, reason) = match analysis {
            Ok(a) if !a.compatible => ("onnx", "unsupported ops"),
            Ok(a) if a.data_movement_only => ("onnx", "data movement only"),
            Ok(_) => {
                if let (Some(est), Some(thresh)) = (estimated, skip_compile_over_size) {
                    if est > thresh {
                        ("onnx", "exceeds size threshold")
                    } else {
                        ("jstprove", "compilable")
                    }
                } else {
                    ("jstprove", "compilable")
                }
            }
            Err(_) => ("onnx", "analysis failed"),
        };

        reports.push(SliceAnalysisReport {
            index: slice.index,
            backend: backend.into(),
            reason: reason.into(),
            estimated_constraints: estimated,
            ops,
            tiled: slice.tiling.is_some(),
            channel_split: false,
            dim_split: slice.dim_split.is_some(),
            circuit_signature: sig,
        });
    }

    Ok(reports)
}

fn estimate_onnx_constraints(onnx_path: &Path) -> Result<u64> {
    let model = onnx_proto::load_model(onnx_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Slicer(format!("no graph in {}", onnx_path.display())))?;
    let shapes = extract_graph_shapes(graph);
    Ok(estimate_slice_constraints(&graph.node, &shapes))
}

fn extract_graph_shapes(
    graph: &onnx_proto::GraphProto,
) -> std::collections::HashMap<String, Vec<i64>> {
    let mut shapes = std::collections::HashMap::new();

    let extract_vi_shape = |vi: &onnx_proto::ValueInfoProto| -> Option<(String, Vec<i64>)> {
        let tp = vi.r#type.as_ref()?;
        if let Some(onnx_proto::onnx::type_proto::Value::TensorType(ref tt)) = tp.value {
            let dims: Vec<i64> = tt
                .shape
                .as_ref()?
                .dim
                .iter()
                .filter_map(|d| {
                    if let Some(onnx_proto::onnx::tensor_shape_proto::dimension::Value::DimValue(
                        v,
                    )) = d.value
                    {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect();
            if !dims.is_empty() {
                return Some((vi.name.clone(), dims));
            }
        }
        None
    };

    for vi in graph
        .input
        .iter()
        .chain(graph.output.iter())
        .chain(graph.value_info.iter())
    {
        if let Some((name, dims)) = extract_vi_shape(vi) {
            shapes.insert(name, dims);
        }
    }

    for init in &graph.initializer {
        if !init.name.is_empty() && !init.dims.is_empty() {
            shapes.insert(init.name.clone(), init.dims.clone());
        }
    }

    shapes
}

fn normalize_slice_for_backend(onnx_path: &Path) -> Result<Option<std::path::PathBuf>> {
    let mut model = onnx_proto::load_model(onnx_path)?;
    let changes = onnx_proto::normalize_for_circuit_backend(&mut model);
    if changes == 0 {
        return Ok(None);
    }
    let normalized = onnx_path.with_extension("backend.onnx");
    onnx_proto::save_model(&model, &normalized)?;
    Ok(Some(normalized))
}

#[allow(clippy::too_many_arguments)]
fn compile_single_slice(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    backend: &JstproveBackend,
    proof_config: jstprove_circuits::api::ProofConfigType,
    weights_as_inputs: bool,
    jstprove_ops: &[&str],
    exclude_from_wai: &std::collections::HashSet<String>,
    skip_compile_over_size: Option<u64>,
    circuit_cache: &CircuitCache,
    traced_shapes: Option<&std::collections::HashMap<String, Vec<i64>>>,
) -> Result<CompileOutcome> {
    let slice_dir = slice_dir_path(slices_dir, slice.index);
    if !slice_dir.exists() {
        return Err(DsperseError::Pipeline(format!(
            "slice directory not found: {}",
            slice_dir.display()
        )));
    }

    if let Some(ref cs) = slice.channel_split
        && !cs.groups.is_empty()
    {
        return compile_channel_split_slice(
            slices_dir,
            slice,
            cs,
            backend,
            proof_config,
            jstprove_ops,
            exclude_from_wai,
            skip_compile_over_size,
            circuit_cache,
            traced_shapes,
        );
    }

    if let Some(ref ds) = slice.dim_split
        && let Some(ref tmpl_rel) = ds.template_path
    {
        let tmpl_path = slices_dir.join(tmpl_rel);
        if tmpl_path.exists() {
            return compile_dim_split_template(
                slices_dir,
                slice,
                &tmpl_path,
                backend,
                proof_config,
                jstprove_ops,
                exclude_from_wai,
                skip_compile_over_size,
                circuit_cache,
                traced_shapes,
            );
        }
    }

    let onnx_path = resolve_compile_onnx(slices_dir, slice)?;
    if !onnx_path.exists() {
        return Err(DsperseError::Pipeline(format!(
            "ONNX model not found for slice {}: {}",
            slice.index,
            onnx_path.display()
        )));
    }

    let analysis = analyze_slice_onnx(&onnx_path, jstprove_ops)?;
    if !analysis.compatible {
        return Ok(CompileOutcome::Skipped);
    }
    if analysis.data_movement_only {
        tracing::info!(slice = slice.index, "skipped (data movement only)");
        return Ok(CompileOutcome::Skipped);
    }

    if let Some(threshold) = skip_compile_over_size {
        let estimated = estimate_onnx_constraints(&onnx_path)?;
        if estimated > threshold {
            return Ok(CompileOutcome::SkippedOverSize {
                estimated,
                threshold,
            });
        }
    }

    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let circuit_path = jst_dir.join("circuit.bundle");

    if circuit_path.is_dir() {
        match backend.load_params(&circuit_path) {
            Ok(_) => {
                tracing::info!(slice = slice.index, "already compiled, skipping");
                return Ok(CompileOutcome::Compiled);
            }
            Err(e) => {
                tracing::warn!(slice = slice.index, error = %e, "cached circuit invalid, recompiling");
                std::fs::remove_dir_all(&circuit_path)
                    .map_err(|e| DsperseError::io(e, &circuit_path))?;
            }
        }
    }

    let effective_wai = weights_as_inputs;

    let estimated = estimate_onnx_constraints(&onnx_path).ok();
    let op_summary = summarize_onnx_ops(&onnx_path);

    tracing::debug!(
        slice = slice.index,
        onnx = %onnx_path.display(),
        estimated_constraints = ?estimated,
        weights_as_inputs = effective_wai,
        ops = %op_summary,
        tiled = slice.tiling.is_some(),
        channel_split = slice.channel_split.is_some(),
        dim_split = slice.dim_split.is_some(),
        "compiling slice"
    );

    let compile_onnx = normalize_slice_for_backend(&onnx_path)?;

    let (params, architecture, wandb) = converter::prepare_jstprove_artifacts_filtered(
        compile_onnx.as_ref().unwrap_or(&onnx_path),
        effective_wai,
        exclude_from_wai,
        traced_shapes,
    )?;

    std::panic::catch_unwind(|| {
        backend.compile(&circuit_path, proof_config, params, architecture, wandb)
    })
    .map_err(|p| {
        let msg = p
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| p.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic");
        DsperseError::Backend(format!("jstprove panicked: {msg}"))
    })??;

    Ok(CompileOutcome::Compiled)
}

fn populate_channel_split_groups(
    slices_dir: &Path,
    slice_idx: usize,
    cs: &mut crate::schema::tiling::ChannelSplitInfo,
) -> Result<bool> {
    let groups_dir = slices_dir
        .join(format!("slice_{slice_idx}"))
        .join("payload")
        .join("channel_groups");
    if !groups_dir.exists() {
        return Ok(false);
    }

    let cpg = cs.channels_per_group;
    let mut groups = Vec::with_capacity(cs.num_groups);
    for g in 0..cs.num_groups {
        let c_start = g.checked_mul(cpg).ok_or_else(|| {
            DsperseError::Slicer(format!("overflow computing c_start for group {g}"))
        })?;
        let c_end = (g + 1)
            .checked_mul(cpg)
            .map(|v| v.min(cs.c_in))
            .ok_or_else(|| {
                DsperseError::Slicer(format!("overflow computing c_end for group {g}"))
            })?;
        let rel_path = format!("slice_{slice_idx}/payload/channel_groups/group_{g}.onnx");
        let abs_path = slices_dir.join(&rel_path);
        if !abs_path.exists() {
            tracing::warn!(
                slice = slice_idx,
                group = g,
                "expected group ONNX not found, skipping population"
            );
            return Ok(false);
        }
        groups.push(crate::schema::tiling::ChannelGroupInfo {
            group_idx: g,
            c_start,
            c_end,
            path: rel_path,
            jstprove_circuit_path: None,
            jstprove_settings_path: None,
        });
    }

    let bias_rel = format!("slice_{slice_idx}/payload/channel_groups/bias.msgpack");
    if slices_dir.join(&bias_rel).exists() {
        cs.bias_path = Some(bias_rel);
    }

    tracing::info!(
        slice = slice_idx,
        groups = groups.len(),
        "populated channel split groups from materialized files"
    );
    cs.groups = groups;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn compile_channel_split_slice(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    cs: &crate::schema::tiling::ChannelSplitInfo,
    backend: &JstproveBackend,
    proof_config: jstprove_circuits::api::ProofConfigType,
    jstprove_ops: &[&str],
    exclude_from_wai: &std::collections::HashSet<String>,
    skip_compile_over_size: Option<u64>,
    circuit_cache: &CircuitCache,
    traced_shapes: Option<&std::collections::HashMap<String, Vec<i64>>>,
) -> Result<CompileOutcome> {
    let slice_dir = slice_dir_path(slices_dir, slice.index);
    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let shared_circuit_rel = format!("slice_{}/jstprove/shared/circuit.bundle", slice.index);
    let shared_circuit_path = jst_dir.join("shared").join("circuit.bundle");

    if !shared_circuit_path.is_dir() {
        let first_group = cs.groups.first().ok_or_else(|| {
            DsperseError::Pipeline(format!("slice {} channel_split has no groups", slice.index))
        })?;
        let onnx_path = slices_dir.join(&first_group.path);
        if !onnx_path.exists() {
            return Err(DsperseError::Pipeline(format!(
                "channel group ONNX not found: {}",
                onnx_path.display()
            )));
        }

        let analysis = analyze_slice_onnx(&onnx_path, jstprove_ops)?;
        if !analysis.compatible {
            return Err(DsperseError::Pipeline(format!(
                "slice {} group 0 has unsupported ops for circuit compilation",
                slice.index
            )));
        }

        if let Some(threshold) = skip_compile_over_size {
            let estimated = estimate_onnx_constraints(&onnx_path)?;
            if estimated > threshold {
                return Ok(CompileOutcome::SkippedOverSize {
                    estimated,
                    threshold,
                });
            }
        }

        let sig = compute_circuit_signature(&onnx_path, None)?;

        let cached = circuit_cache.lock().unwrap().get(&sig).cloned();
        if let Some(ref cached_path) = cached
            && cached_path.is_dir()
        {
            let shared_dir = shared_circuit_path.parent().ok_or_else(|| {
                DsperseError::Pipeline("shared circuit path has no parent".into())
            })?;
            std::fs::create_dir_all(shared_dir).map_err(|e| DsperseError::io(e, shared_dir))?;
            copy_dir_recursive(cached_path, &shared_circuit_path)?;
            tracing::info!(
                slice = slice.index,
                sig = %sig,
                "reused cached channel-split circuit from prior slice"
            );
        } else {
            let shared_dir = shared_circuit_path.parent().ok_or_else(|| {
                DsperseError::Pipeline("shared circuit path has no parent".into())
            })?;
            std::fs::create_dir_all(shared_dir).map_err(|e| DsperseError::io(e, shared_dir))?;

            tracing::info!(
                slice = slice.index,
                groups = cs.groups.len(),
                sig = %sig,
                "compiling shared channel group circuit (weights-as-inputs)"
            );

            let (params, architecture, wandb) = converter::prepare_jstprove_artifacts_filtered(
                &onnx_path,
                true,
                exclude_from_wai,
                traced_shapes,
            )?;

            std::panic::catch_unwind(|| {
                backend.compile(
                    &shared_circuit_path,
                    proof_config,
                    params,
                    architecture,
                    wandb,
                )
            })
            .map_err(|p| {
                let msg = p
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| p.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("unknown panic");
                DsperseError::Backend(format!(
                    "jstprove panicked on slice {} shared circuit: {msg}",
                    slice.index
                ))
            })??;

            circuit_cache
                .lock()
                .unwrap()
                .insert(sig.clone(), shared_circuit_path.clone());
            tracing::info!(slice = slice.index, sig = %sig, "shared circuit compiled");
        }
    } else {
        backend.load_params(&shared_circuit_path).map_err(|e| {
            DsperseError::Pipeline(format!(
                "slice {} cached shared circuit invalid: {e}",
                slice.index
            ))
        })?;
        tracing::info!(
            slice = slice.index,
            "shared circuit already compiled, reusing"
        );
    }

    let group_circuits: Vec<(usize, String)> = cs
        .groups
        .iter()
        .map(|g| (g.group_idx, shared_circuit_rel.clone()))
        .collect();

    Ok(CompileOutcome::CompiledChannelSplit { group_circuits })
}

#[allow(clippy::too_many_arguments)]
fn compile_dim_split_template(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
    tmpl_path: &Path,
    backend: &JstproveBackend,
    proof_config: jstprove_circuits::api::ProofConfigType,
    jstprove_ops: &[&str],
    exclude_from_wai: &std::collections::HashSet<String>,
    skip_compile_over_size: Option<u64>,
    circuit_cache: &CircuitCache,
    _traced_shapes: Option<&std::collections::HashMap<String, Vec<i64>>>,
) -> Result<CompileOutcome> {
    let slice_dir = slice_dir_path(slices_dir, slice.index);
    let jst_dir = slice_dir.join("jstprove");
    std::fs::create_dir_all(&jst_dir).map_err(|e| DsperseError::io(e, &jst_dir))?;

    let circuit_path = jst_dir.join("dim_split").join("circuit.bundle");

    if circuit_path.is_dir() {
        match backend.load_params(&circuit_path) {
            Ok(_) => {
                tracing::info!(
                    slice = slice.index,
                    "dim-split template already compiled, reusing"
                );
                return Ok(CompileOutcome::CompiledDimSplit);
            }
            Err(e) => {
                tracing::warn!(slice = slice.index, error = %e, "cached dim-split circuit invalid, recompiling");
                std::fs::remove_dir_all(&circuit_path)
                    .map_err(|e| DsperseError::io(e, &circuit_path))?;
            }
        }
    }

    let analysis = analyze_slice_onnx(tmpl_path, jstprove_ops)?;
    if !analysis.compatible {
        return Ok(CompileOutcome::Skipped);
    }

    if let Some(threshold) = skip_compile_over_size {
        let estimated = slice
            .dim_split
            .as_ref()
            .map(|ds| ds.estimated_group_constraints)
            .filter(|&e| e > 0)
            .or_else(|| match estimate_onnx_constraints(tmpl_path) {
                Ok(e) => Some(e),
                Err(err) => {
                    // We can't turn an unknown cost into a safe
                    // gating decision, so fall through and let the
                    // compile attempt surface the real error rather
                    // than silently treating the slice as tiny.
                    tracing::warn!(
                        slice = slice.index,
                        onnx = %tmpl_path.display(),
                        error = %err,
                        "skip_compile_over_size: constraint estimate failed; proceeding to compile"
                    );
                    None
                }
            });
        if let Some(estimated) = estimated
            && estimated > threshold
        {
            return Ok(CompileOutcome::SkippedOverSize {
                estimated,
                threshold,
            });
        }
    }

    let sig = compute_circuit_signature(tmpl_path, None)?;

    let cached = circuit_cache.lock().unwrap().get(&sig).cloned();
    if let Some(ref cached_path) = cached
        && cached_path.is_dir()
    {
        let shared_dir = circuit_path
            .parent()
            .ok_or_else(|| DsperseError::Pipeline("dim-split circuit path has no parent".into()))?;
        std::fs::create_dir_all(shared_dir).map_err(|e| DsperseError::io(e, shared_dir))?;
        copy_dir_recursive(cached_path, &circuit_path)?;
        tracing::info!(
            slice = slice.index,
            sig = %sig,
            "reused cached dim-split circuit"
        );
        return Ok(CompileOutcome::CompiledDimSplit);
    }

    let shared_dir = circuit_path
        .parent()
        .ok_or_else(|| DsperseError::Pipeline("dim-split circuit path has no parent".into()))?;
    std::fs::create_dir_all(shared_dir).map_err(|e| DsperseError::io(e, shared_dir))?;

    tracing::info!(
        slice = slice.index,
        sig = %sig,
        "compiling dim-split template (weights-as-inputs)"
    );

    // Do NOT pass the original traced_shapes when compiling dim-split
    // templates. The template has rewritten shapes (dim_size → epg) that
    // differ from the original model's traced shapes. If traced_shapes
    // is passed, jstprove uses the original (larger) shapes and the
    // Transpose/Reshape validation fails on the mismatch.
    let (params, architecture, wandb) =
        converter::prepare_jstprove_artifacts_filtered(tmpl_path, true, exclude_from_wai, None)?;

    std::panic::catch_unwind(|| {
        backend.compile(&circuit_path, proof_config, params, architecture, wandb)
    })
    .map_err(|p| {
        let msg = p
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| p.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic");
        DsperseError::Backend(format!(
            "jstprove panicked on slice {} dim-split template: {msg}",
            slice.index
        ))
    })??;

    circuit_cache
        .lock()
        .unwrap()
        .insert(sig.clone(), circuit_path.clone());
    tracing::info!(slice = slice.index, sig = %sig, "dim-split template compiled");
    Ok(CompileOutcome::CompiledDimSplit)
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).map_err(|e| DsperseError::io(e, dst))?;
    for entry in std::fs::read_dir(src).map_err(|e| DsperseError::io(e, src))? {
        let entry = entry.map_err(|e| DsperseError::io(e, src))?;
        let ty = entry.file_type().map_err(|e| DsperseError::io(e, src))?;
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path).map_err(|e| DsperseError::io(e, &dst_path))?;
        }
    }
    Ok(())
}

fn resolve_compile_onnx(
    slices_dir: &Path,
    slice: &crate::schema::metadata::SliceMetadata,
) -> Result<std::path::PathBuf> {
    if let Some(ref tiling) = slice.tiling
        && let Some(ref tile) = tiling.tile
    {
        let tile_path = slices_dir.join(&tile.path);
        if tile_path.exists() {
            tracing::info!(
                slice = slice.index,
                path = %tile_path.display(),
                "using tile ONNX"
            );
            return Ok(tile_path);
        }
    }

    slice.resolve_onnx(slices_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::metadata::{
        Compilation, Dependencies, SliceMetadata, SliceShapeWrapper, TensorShape,
    };
    use crate::schema::tiling::{TileInfo, TilingInfo};

    fn test_models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models"))
    }

    fn make_slice_metadata(index: usize, path: &str) -> SliceMetadata {
        SliceMetadata {
            index,
            filename: format!("slice_{index}.onnx"),
            path: path.to_string(),
            relative_path: path.to_string(),
            shape: SliceShapeWrapper {
                tensor_shape: TensorShape::default(),
            },
            dependencies: Dependencies {
                input: vec![],
                output: vec![],
                filtered_inputs: vec![],
            },
            tiling: None,
            channel_split: None,
            dim_split: None,
            compilation: Compilation::default(),
            slice_metadata: None,
            slice_metadata_relative_path: None,
        }
    }

    const TEST_OPS: &[&str] = &["Conv", "Gemm", "MatMul"];

    #[test]
    fn analyze_slice_onnx_nonexistent() {
        let result = analyze_slice_onnx(Path::new("/nonexistent.onnx"), TEST_OPS);
        assert!(result.is_err());
    }

    #[test]
    fn analyze_slice_onnx_test_model() {
        let model_path = test_models_dir().join("net/model.onnx");
        assert!(
            model_path.exists(),
            "fixture missing: {}",
            model_path.display()
        );
        let analysis = analyze_slice_onnx(&model_path, TEST_OPS).unwrap();
        assert!(!analysis.compatible);
    }

    #[test]
    fn analyze_slice_onnx_with_initializers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("with_init.onnx");
        let model = onnx_proto::ModelProto {
            graph: Some(onnx_proto::GraphProto {
                node: vec![onnx_proto::make_node("Conv", vec![], vec![], vec![])],
                initializer: vec![onnx_proto::make_tensor(
                    "weight",
                    1,
                    &[3, 3, 3, 3],
                    vec![0.0; 81],
                )],
                ..Default::default()
            }),
            ..Default::default()
        };
        onnx_proto::save_model(&model, &path).unwrap();
        let analysis = analyze_slice_onnx(&path, &["Conv"]).unwrap();
        assert!(analysis.compatible);
    }

    #[test]
    fn analyze_slice_onnx_without_initializers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("no_init.onnx");
        let model = onnx_proto::ModelProto {
            graph: Some(onnx_proto::GraphProto {
                node: vec![onnx_proto::make_node("Relu", vec![], vec![], vec![])],
                initializer: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };
        onnx_proto::save_model(&model, &path).unwrap();
        let analysis = analyze_slice_onnx(&path, &["Relu"]).unwrap();
        assert!(analysis.compatible);
    }

    #[test]
    fn resolve_compile_onnx_no_tiling() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let slice_dir = slices_dir.join("slice_0");
        std::fs::create_dir_all(&slice_dir).unwrap();

        let meta = make_slice_metadata(0, "slice_0.onnx");
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("slice_0.onnx"));
    }

    #[test]
    fn resolve_compile_onnx_with_tile() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let tile_path = slices_dir.join("slice_0/payload/tiles/tile.onnx");
        std::fs::create_dir_all(tile_path.parent().unwrap()).unwrap();
        std::fs::write(&tile_path, b"dummy").unwrap();

        let mut meta = make_slice_metadata(0, "slice_0.onnx");
        meta.tiling = Some(TilingInfo {
            slice_idx: 0,
            tile_size: 8,
            num_tiles: 4,
            tiles_y: 2,
            tiles_x: 2,
            halo: [1, 1, 1, 1],
            out_tile: [4, 4],
            stride: [1, 1],
            c_in: 3,
            c_out: 16,
            input_name: "input".into(),
            output_name: "output".into(),
            input_names: vec![],
            ndim: 4,
            h: 16,
            w: 16,
            tile: Some(TileInfo {
                path: "slice_0/payload/tiles/tile.onnx".into(),
                conv_out: [4, 4],
                jstprove_circuit_path: None,
            }),
            tiles: None,
            segment_size: None,
            total_elements: None,
            original_shape: vec![],
        });
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("tile.onnx"));
    }

    #[test]
    fn resolve_compile_onnx_tile_missing_falls_back() {
        let tmp = tempfile::tempdir().unwrap();
        let slices_dir = tmp.path();
        let slice_dir = slices_dir.join("slice_0");
        std::fs::create_dir_all(&slice_dir).unwrap();

        let mut meta = make_slice_metadata(0, "slice_0.onnx");
        meta.tiling = Some(TilingInfo {
            slice_idx: 0,
            tile_size: 8,
            num_tiles: 4,
            tiles_y: 2,
            tiles_x: 2,
            halo: [1, 1, 1, 1],
            out_tile: [4, 4],
            stride: [1, 1],
            c_in: 3,
            c_out: 16,
            input_name: "input".into(),
            output_name: "output".into(),
            input_names: vec![],
            ndim: 4,
            h: 16,
            w: 16,
            tile: Some(TileInfo {
                path: "slice_0/payload/tiles/nonexistent.onnx".into(),
                conv_out: [4, 4],
                jstprove_circuit_path: None,
            }),
            tiles: None,
            segment_size: None,
            total_elements: None,
            original_shape: vec![],
        });
        let path = resolve_compile_onnx(slices_dir, &meta).unwrap();
        assert!(path.ends_with("slice_0.onnx"));
    }
}
