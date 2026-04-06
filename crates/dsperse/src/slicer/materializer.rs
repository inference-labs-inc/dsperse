use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::autotiler::{self, ChannelSplitParams};
use super::onnx_proto::{self, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};
use super::onnx_slicer::broadcast_shapes;
use crate::error::{DsperseError, Result};
use crate::schema::metadata::ModelMetadata;

const MAX_BACKWARD_DEPTH: usize = 64;

fn resolve_shape_backward(
    tensor_name: &str,
    graph: &GraphProto,
    traced_shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    resolve_shape_backward_inner(tensor_name, graph, traced_shapes, 0)
}

fn resolve_shape_backward_inner(
    tensor_name: &str,
    graph: &GraphProto,
    traced_shapes: &HashMap<String, Vec<i64>>,
    depth: usize,
) -> Option<Vec<i64>> {
    if depth > MAX_BACKWARD_DEPTH {
        return None;
    }

    if let Some(s) = traced_shapes.get(tensor_name) {
        return Some(s.clone());
    }

    if let Some(vi) = graph.value_info.iter().find(|v| v.name == tensor_name)
        && let Some(shape) = onnx_proto::shape_from_value_info(vi)
    {
        return Some(shape);
    }

    for init in &graph.initializer {
        if init.name == tensor_name {
            return Some(init.dims.to_vec());
        }
    }

    let producer = graph
        .node
        .iter()
        .find(|n| n.output.contains(&tensor_name.to_string()))?;
    let op = producer.op_type.as_str();

    if super::is_shape_preserving(op) {
        let inp = producer.input.first()?;
        return resolve_shape_backward_inner(inp, graph, traced_shapes, depth + 1);
    }

    if op == "Shape" {
        let inp = producer.input.first()?;
        let in_shape = resolve_shape_backward_inner(inp, graph, traced_shapes, depth + 1)?;
        return Some(vec![in_shape.len() as i64]);
    }

    if super::is_binary_arithmetic(op) {
        let resolved: Vec<Vec<i64>> = producer
            .input
            .iter()
            .filter_map(|inp| resolve_shape_backward_inner(inp, graph, traced_shapes, depth + 1))
            .collect();
        let refs: Vec<&Vec<i64>> = resolved.iter().collect();
        if let Some(broadcasted) = broadcast_shapes(&refs) {
            return Some(broadcasted);
        }
    }

    None
}

pub fn materialize_slice_model(
    model: &ModelProto,
    slice_points: &[usize],
    traced_shapes: &HashMap<String, Vec<i64>>,
    slice_idx: usize,
) -> Result<ModelProto> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DsperseError::Slicer("model.graph is None".into()))?;

    let total_nodes = graph.node.len();
    let segment_ranges = super::build_segment_ranges(slice_points, Some(total_nodes));
    let &(start, end) = segment_ranges.get(slice_idx).ok_or_else(|| {
        DsperseError::Slicer(format!(
            "slice index {slice_idx} out of range (have {} segments)",
            segment_ranges.len()
        ))
    })?;

    let init_map: HashMap<&str, &TensorProto> = graph
        .initializer
        .iter()
        .map(|i| (i.name.as_str(), i))
        .collect();

    let vi_map = onnx_proto::build_value_info_map(graph);

    let init_types: HashMap<&str, i32> = graph
        .initializer
        .iter()
        .map(|i| (i.name.as_str(), i.data_type))
        .collect();

    let node_output_types = build_node_output_types(graph);
    let future_deps = compute_future_dependencies(graph, &segment_ranges, &init_map);

    let constant_producers: HashMap<String, &TensorProto> = graph
        .node
        .iter()
        .filter(|n| n.op_type == "Constant")
        .flat_map(|n| {
            n.output.iter().filter_map(|out| {
                n.attribute
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.t.as_ref())
                    .map(|t| (out.clone(), t))
            })
        })
        .collect();

    let nodes: Vec<NodeProto> = graph.node[start..end].to_vec();

    let seg_outputs: HashSet<String> = nodes
        .iter()
        .flat_map(|n| n.output.iter().cloned())
        .collect();

    let seg_inputs_set: HashSet<String> = nodes
        .iter()
        .flat_map(|n| {
            let mut inputs: Vec<String> =
                n.input.iter().filter(|s| !s.is_empty()).cloned().collect();
            if super::is_control_flow(&n.op_type) {
                let outer_refs = super::collect_subgraph_outer_refs(n, graph);
                inputs.extend(outer_refs);
            }
            inputs
        })
        .collect();

    let future = future_deps.get(&slice_idx).cloned().unwrap_or_default();

    let query = SegmentQuery {
        nodes: &nodes,
        seg_outputs: &seg_outputs,
        seg_inputs_set: &seg_inputs_set,
        future_inputs: &future,
    };
    let ctx = ShapeContext {
        graph,
        init_map: &init_map,
        vi_map: &vi_map,
        traced_shapes,
        init_types: &init_types,
        node_output_types: &node_output_types,
        constant_producers: &constant_producers,
    };
    let (inputs, outputs, initializers) = get_segment_details(&query, &ctx)?;

    let opset_version = model
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .map(|o| o.version)
        .unwrap_or(13);

    let seg_graph = onnx_proto::make_graph(
        &format!("segment_{slice_idx}_graph"),
        nodes,
        inputs,
        outputs,
        initializers,
    );
    Ok(onnx_proto::make_model(seg_graph, opset_version))
}

pub fn materialize_slice_to_disk(
    model: &ModelProto,
    slice_points: &[usize],
    traced_shapes: &HashMap<String, Vec<i64>>,
    slice_idx: usize,
    output_path: &Path,
) -> Result<PathBuf> {
    let slice_model = materialize_slice_model(model, slice_points, traced_shapes, slice_idx)?;
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DsperseError::io(e, parent))?;
    }
    onnx_proto::save_model(&slice_model, output_path)?;
    Ok(output_path.to_path_buf())
}

pub fn ensure_slice_materialized(
    slices_dir: &Path,
    metadata: &ModelMetadata,
    slice_idx: usize,
) -> Result<PathBuf> {
    let slice_meta = metadata
        .slices
        .iter()
        .find(|s| s.index == slice_idx)
        .ok_or_else(|| DsperseError::Slicer(format!("no slice metadata for index {slice_idx}")))?;

    let slice_dir = slices_dir.join(format!("slice_{slice_idx}"));
    let payload_dir = slice_dir.join("payload");
    let onnx_path = payload_dir.join(format!("slice_{slice_idx}.onnx"));

    if onnx_path.exists() {
        materialize_tiling_artifacts(slices_dir, metadata, slice_meta, slice_idx)?;
        return Ok(onnx_path);
    }

    if !slice_dir.exists() {
        let archive = slices_dir.join(format!("slice_{slice_idx}.dslice"));
        if archive.exists() {
            extract_dslice_archive(&archive, &slice_dir)?;
            if onnx_path.exists() {
                materialize_tiling_artifacts(slices_dir, metadata, slice_meta, slice_idx)?;
                return Ok(onnx_path);
            }
        }
    }

    let traced_shapes = metadata.traced_shapes.as_ref().ok_or_else(|| {
        DsperseError::Slicer("metadata missing traced_shapes for materialization".into())
    })?;
    let original_path = metadata.original_model_path.as_ref().ok_or_else(|| {
        DsperseError::Slicer("metadata missing original_model_path for materialization".into())
    })?;

    let model_path = if Path::new(original_path).is_absolute() {
        PathBuf::from(original_path)
    } else {
        slices_dir.join(original_path)
    };

    let mut model = onnx_proto::load_model(&model_path)?;
    onnx_proto::normalize_opset(&mut model);
    let model_with_shapes = apply_traced_shapes(model, traced_shapes);

    std::fs::create_dir_all(&payload_dir).map_err(|e| DsperseError::io(e, &payload_dir))?;
    materialize_slice_to_disk(
        &model_with_shapes,
        &metadata.slice_points,
        traced_shapes,
        slice_idx,
        &onnx_path,
    )?;

    tracing::info!(slice = slice_idx, path = %onnx_path.display(), "materialized slice");

    materialize_tiling_artifacts(slices_dir, metadata, slice_meta, slice_idx)?;

    Ok(onnx_path)
}

fn materialize_tiling_artifacts(
    slices_dir: &Path,
    _metadata: &ModelMetadata,
    slice_meta: &crate::schema::metadata::SliceMetadata,
    slice_idx: usize,
) -> Result<()> {
    let payload_dir = slices_dir
        .join(format!("slice_{slice_idx}"))
        .join("payload");

    if let Some(ref tiling) = slice_meta.tiling
        && let Some(ref tile) = tiling.tile
    {
        let tile_path = slices_dir.join(&tile.path);
        if !tile_path.exists() {
            let onnx_path = payload_dir.join(format!("slice_{slice_idx}.onnx"));
            let slice_model = onnx_proto::load_model(&onnx_path)?;
            let is_ew = slice_model.graph.as_ref().is_some_and(|g| {
                !g.node.is_empty() && g.node.iter().all(|n| super::is_elementwise(&n.op_type))
            });
            let is_pool = slice_model
                .graph
                .as_ref()
                .is_some_and(|g| g.node.iter().any(|n| n.op_type == "MaxPool"));
            if is_ew {
                let seg_size = tiling.segment_size.ok_or_else(|| {
                    crate::error::DsperseError::Slicer(format!(
                        "slice {slice_idx}: elementwise tiling metadata missing segment_size; re-slice the model"
                    ))
                })? as i64;
                autotiler::create_elementwise_tile_slice(
                    &slice_model,
                    seg_size,
                    slice_idx,
                    &payload_dir,
                )?;
            } else if is_pool {
                autotiler::create_pool_tile_slice(
                    &slice_model,
                    tiling.tile_size as i64,
                    slice_idx,
                    &payload_dir,
                )?;
            } else {
                autotiler::create_tile_slice(
                    &slice_model,
                    tiling.tile_size as i64,
                    slice_idx,
                    &payload_dir,
                )?;
            }
            tracing::info!(slice = slice_idx, "materialized tile ONNX");
        }
    }

    if let Some(ref cs) = slice_meta.channel_split {
        let needs_materialization = cs.groups.is_empty()
            || cs.groups.iter().any(|g| {
                let group_path = slices_dir.join(&g.path);
                !group_path.exists()
            });

        if needs_materialization && cs.num_groups > 0 {
            let onnx_path = payload_dir.join(format!("slice_{slice_idx}.onnx"));
            let slice_model = onnx_proto::load_model(&onnx_path)?;

            let params = ChannelSplitParams {
                c_in: cs.c_in as i64,
                c_out: cs.c_out as i64,
                num_groups: cs.num_groups as i64,
                channels_per_group: cs.channels_per_group as i64,
                h: cs.h as i64,
                w: cs.w as i64,
                slice_idx,
            };
            let cs_info = autotiler::apply_channel_splitting(
                &slice_model,
                &params,
                &cs.input_name,
                &cs.output_name,
                &payload_dir,
            )?;
            tracing::info!(
                slice = slice_idx,
                groups = cs_info.groups.len(),
                "materialized channel groups"
            );
        }
    }

    Ok(())
}

pub fn ensure_all_slices_materialized(slices_dir: &Path, metadata: &ModelMetadata) -> Result<()> {
    use rayon::prelude::*;

    metadata.slices.par_iter().try_for_each(|slice| {
        ensure_slice_materialized(slices_dir, metadata, slice.index).map(|_| ())
    })
}

fn apply_traced_shapes(mut model: ModelProto, shapes: &HashMap<String, Vec<i64>>) -> ModelProto {
    fn set_shape(vi: &mut ValueInfoProto, shape: &[i64]) {
        if let Some(ref mut tp) = vi.r#type
            && let Some(onnx_proto::onnx::type_proto::Value::TensorType(ref mut tt)) = tp.value
        {
            tt.shape = Some(onnx_proto::onnx::TensorShapeProto {
                dim: shape
                    .iter()
                    .map(|&d| onnx_proto::onnx::tensor_shape_proto::Dimension {
                        denotation: String::new(),
                        value: Some(
                            onnx_proto::onnx::tensor_shape_proto::dimension::Value::DimValue(d),
                        ),
                    })
                    .collect(),
            });
        }
    }

    if let Some(ref mut graph) = model.graph {
        for inp in &mut graph.input {
            if let Some(shape) = shapes.get(&inp.name) {
                set_shape(inp, shape);
            }
        }
        for out in &mut graph.output {
            if let Some(shape) = shapes.get(&out.name) {
                set_shape(out, shape);
            }
        }
        for vi in &mut graph.value_info {
            if let Some(shape) = shapes.get(&vi.name) {
                set_shape(vi, shape);
            }
        }

        let existing: HashSet<String> = graph
            .input
            .iter()
            .chain(graph.output.iter())
            .chain(graph.value_info.iter())
            .map(|vi| vi.name.clone())
            .collect();

        let init_types: HashMap<&str, i32> = graph
            .initializer
            .iter()
            .map(|i| (i.name.as_str(), i.data_type))
            .collect();

        let node_output_types = build_node_output_types(graph);

        for (name, shape) in shapes {
            if !existing.contains(name) {
                let from_init = init_types.get(name.as_str()).copied();
                let from_node = node_output_types.get(name).copied();
                let elem_type = from_init.or(from_node).unwrap_or_else(|| {
                    tracing::debug!(tensor = %name, "no explicit dtype in initializers or node outputs, assuming FLOAT");
                    TensorProto::FLOAT
                });
                graph
                    .value_info
                    .push(onnx_proto::make_tensor_value_info(name, elem_type, shape));
            }
        }
    }
    model
}

fn compute_future_dependencies(
    graph: &GraphProto,
    segment_ranges: &[(usize, usize)],
    init_map: &HashMap<&str, &TensorProto>,
) -> HashMap<usize, HashSet<String>> {
    let mut seg_inputs: HashMap<usize, HashSet<String>> = HashMap::new();

    for (seg_idx, &(start, end)) in segment_ranges.iter().enumerate() {
        let seg_outputs: HashSet<String> = graph.node[start..end]
            .iter()
            .flat_map(|n| n.output.iter().cloned())
            .collect();

        let inputs: HashSet<String> = graph.node[start..end]
            .iter()
            .flat_map(|n| {
                if super::is_control_flow(&n.op_type) {
                    let outer_refs = super::collect_subgraph_outer_refs(n, graph);
                    return outer_refs
                        .into_iter()
                        .chain(n.input.iter().cloned())
                        .collect::<Vec<String>>();
                }
                n.input.to_vec()
            })
            .filter(|inp| {
                !inp.is_empty()
                    && !seg_outputs.contains(inp.as_str())
                    && !init_map.contains_key(inp.as_str())
            })
            .collect();

        seg_inputs.insert(seg_idx, inputs);
    }

    let mut future: HashMap<usize, HashSet<String>> = HashMap::new();
    for seg_idx in 0..segment_ranges.len() {
        let mut deps = HashSet::new();
        for future_idx in (seg_idx + 1)..segment_ranges.len() {
            if let Some(inputs) = seg_inputs.get(&future_idx) {
                deps.extend(inputs.iter().cloned());
            }
        }
        future.insert(seg_idx, deps);
    }
    future
}

struct SegmentQuery<'a> {
    nodes: &'a [NodeProto],
    seg_outputs: &'a HashSet<String>,
    seg_inputs_set: &'a HashSet<String>,
    future_inputs: &'a HashSet<String>,
}

struct ShapeContext<'a> {
    graph: &'a GraphProto,
    init_map: &'a HashMap<&'a str, &'a TensorProto>,
    vi_map: &'a HashMap<String, &'a ValueInfoProto>,
    traced_shapes: &'a HashMap<String, Vec<i64>>,
    init_types: &'a HashMap<&'a str, i32>,
    node_output_types: &'a HashMap<String, i32>,
    constant_producers: &'a HashMap<String, &'a TensorProto>,
}

impl ShapeContext<'_> {
    fn resolve_elem_type(&self, name: &str) -> i32 {
        self.init_types
            .get(name)
            .copied()
            .or_else(|| self.node_output_types.get(name).copied())
            .unwrap_or(TensorProto::FLOAT)
    }
}

fn get_segment_details(
    query: &SegmentQuery<'_>,
    ctx: &ShapeContext<'_>,
) -> Result<(Vec<ValueInfoProto>, Vec<ValueInfoProto>, Vec<TensorProto>)> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut initializers = Vec::new();

    let model_output_names: HashSet<String> =
        ctx.graph.output.iter().map(|o| o.name.clone()).collect();

    let mut added_inputs: HashSet<String> = HashSet::new();
    let mut sorted_inputs: Vec<_> = query.seg_inputs_set.iter().collect();
    sorted_inputs.sort();
    for inp_name in sorted_inputs {
        if query.seg_outputs.contains(inp_name) {
            continue;
        }
        if ctx.init_map.contains_key(inp_name.as_str()) {
            initializers.push((*ctx.init_map[inp_name.as_str()]).clone());
        } else if ctx.constant_producers.contains_key(inp_name) {
            let mut tensor = ctx.constant_producers[inp_name].clone();
            tensor.name = inp_name.clone();
            initializers.push(tensor);
        } else if !added_inputs.contains(inp_name) {
            if let Some(vi) = ctx.vi_map.get(inp_name) {
                inputs.push((*vi).clone());
            } else {
                let shape = ctx
                    .traced_shapes
                    .get(inp_name)
                    .cloned()
                    .or_else(|| resolve_shape_backward(inp_name, ctx.graph, ctx.traced_shapes))
                    .unwrap_or_else(|| {
                        tracing::warn!(tensor = %inp_name, "no traced shape; using placeholder");
                        vec![1]
                    });
                inputs.push(onnx_proto::make_tensor_value_info(
                    inp_name,
                    ctx.resolve_elem_type(inp_name),
                    &shape,
                ));
            }
            added_inputs.insert(inp_name.clone());
        }
    }

    let mut sorted_outputs: Vec<_> = query.seg_outputs.iter().collect();
    sorted_outputs.sort();
    for out_name in sorted_outputs {
        if ctx.constant_producers.contains_key(out_name) {
            continue;
        }
        let consumed_internally = query.nodes.iter().any(|n| n.input.contains(out_name));
        let needed_externally =
            query.future_inputs.contains(out_name) || model_output_names.contains(out_name);

        if !consumed_internally || needed_externally {
            if let Some(vi) = ctx.vi_map.get(out_name) {
                outputs.push((*vi).clone());
            } else {
                let shape = ctx
                    .traced_shapes
                    .get(out_name)
                    .cloned()
                    .or_else(|| resolve_shape_backward(out_name, ctx.graph, ctx.traced_shapes))
                    .unwrap_or_else(|| {
                        tracing::warn!(tensor = %out_name, "no traced shape; using placeholder");
                        vec![1]
                    });
                outputs.push(onnx_proto::make_tensor_value_info(
                    out_name,
                    ctx.resolve_elem_type(out_name),
                    &shape,
                ));
            }
        }
    }

    Ok((inputs, outputs, initializers))
}

pub fn build_node_output_types(graph: &GraphProto) -> HashMap<String, i32> {
    let mut types: HashMap<String, i32> = HashMap::new();
    for node in &graph.node {
        match node.op_type.as_str() {
            "Cast" => {
                if let Some(to) = onnx_proto::get_attribute_int(node, "to") {
                    for out in &node.output {
                        if !out.is_empty() {
                            types.insert(out.clone(), to as i32);
                        }
                    }
                }
            }
            "MaxPool" => {
                if let Some(idx_out) = node.output.get(1)
                    && !idx_out.is_empty()
                {
                    types.insert(idx_out.clone(), TensorProto::INT64);
                }
            }
            "Shape" | "NonZero" | "ArgMax" | "ArgMin" => {
                for out in &node.output {
                    if !out.is_empty() {
                        types.insert(out.clone(), TensorProto::INT64);
                    }
                }
            }
            _ => {}
        }
    }
    types
}

fn extract_dslice_archive(archive: &Path, dest: &Path) -> Result<()> {
    let tmp_dir = dest.with_file_name(format!(
        ".{}.extracting.{}",
        dest.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| DsperseError::io(e, &tmp_dir))?;
    let file = std::fs::File::open(archive).map_err(|e| DsperseError::io(e, archive))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| {
        DsperseError::Slicer(format!("reading dslice archive {}: {e}", archive.display()))
    })?;
    if let Err(e) = zip.extract(&tmp_dir) {
        std::fs::remove_dir_all(&tmp_dir).ok();
        return Err(DsperseError::Slicer(format!(
            "extracting {} to {}: {e}",
            archive.display(),
            tmp_dir.display()
        )));
    }
    if let Err(e) = std::fs::rename(&tmp_dir, dest) {
        std::fs::remove_dir_all(&tmp_dir).ok();
        if dest.exists() {
            return Ok(());
        }
        return Err(DsperseError::Slicer(format!(
            "renaming {} to {}: {e}",
            tmp_dir.display(),
            dest.display()
        )));
    }
    tracing::debug!(archive = %archive.display(), dest = %dest.display(), "extracted dslice archive");
    Ok(())
}

pub fn cleanup_extracted_slice(slices_dir: &Path, slice_id: &str) {
    let extract_dir = slices_dir.join(slice_id);
    if std::fs::remove_dir_all(&extract_dir).is_err() && extract_dir.exists() {
        tracing::warn!(dir = %extract_dir.display(), "failed to remove extracted slice dir");
    }
}
