use std::collections::{HashMap, HashSet};

use super::onnx_proto::GraphProto;

pub struct TensorInfo {
    pub name: String,
    pub producer_node: Option<String>,
    pub producer_node_idx: Option<usize>,
    pub consumers: Vec<(String, usize)>,
    pub is_input: bool,
    pub is_output: bool,
    pub is_initializer: bool,
}

pub struct TensorGraph {
    pub tensors: HashMap<String, TensorInfo>,
    pub node_to_idx: HashMap<String, usize>,
    pub idx_to_node: HashMap<usize, String>,
    num_nodes: usize,
}

impl TensorGraph {
    pub fn new(graph: &GraphProto) -> Self {
        let mut tg = Self {
            tensors: HashMap::new(),
            node_to_idx: HashMap::new(),
            idx_to_node: HashMap::new(),
            num_nodes: graph.node.len(),
        };
        tg.build(graph);
        tg
    }

    fn build(&mut self, graph: &GraphProto) {
        for inp in &graph.input {
            self.tensors.insert(
                inp.name.clone(),
                TensorInfo {
                    name: inp.name.clone(),
                    producer_node: None,
                    producer_node_idx: None,
                    consumers: Vec::new(),
                    is_input: true,
                    is_output: false,
                    is_initializer: false,
                },
            );
        }

        for init in &graph.initializer {
            if let Some(t) = self.tensors.get_mut(&init.name) {
                t.is_initializer = true;
            } else {
                self.tensors.insert(
                    init.name.clone(),
                    TensorInfo {
                        name: init.name.clone(),
                        producer_node: None,
                        producer_node_idx: None,
                        consumers: Vec::new(),
                        is_input: false,
                        is_output: false,
                        is_initializer: true,
                    },
                );
            }
        }

        for (idx, node) in graph.node.iter().enumerate() {
            let node_name = if node.name.is_empty() {
                format!("{}_{}", node.op_type, idx)
            } else {
                node.name.clone()
            };
            self.node_to_idx.insert(node_name.clone(), idx);
            self.idx_to_node.insert(idx, node_name.clone());

            for output in &node.output {
                if !output.is_empty() {
                    let t = self.tensors.entry(output.clone()).or_insert_with(|| TensorInfo {
                        name: output.clone(),
                        producer_node: None,
                        producer_node_idx: None,
                        consumers: Vec::new(),
                        is_input: false,
                        is_output: false,
                        is_initializer: false,
                    });
                    t.producer_node = Some(node_name.clone());
                    t.producer_node_idx = Some(idx);
                }
            }

            for inp in &node.input {
                if !inp.is_empty() {
                    let t = self.tensors.entry(inp.clone()).or_insert_with(|| TensorInfo {
                        name: inp.clone(),
                        producer_node: None,
                        producer_node_idx: None,
                        consumers: Vec::new(),
                        is_input: false,
                        is_output: false,
                        is_initializer: false,
                    });
                    t.consumers.push((node_name.clone(), idx));
                }
            }
        }

        for out in &graph.output {
            if let Some(t) = self.tensors.get_mut(&out.name) {
                t.is_output = true;
            }
        }
    }

    pub fn get_slice_inputs(&self, graph: &GraphProto, start_idx: usize, end_idx: usize) -> Vec<String> {
        let mut internal_outputs: HashSet<String> = HashSet::new();
        for idx in start_idx..end_idx {
            if let Some(node) = graph.node.get(idx) {
                for output in &node.output {
                    if !output.is_empty() {
                        internal_outputs.insert(output.clone());
                    }
                }
            }
        }

        let mut external_inputs = Vec::new();
        let mut seen = HashSet::new();
        for idx in start_idx..end_idx {
            if let Some(node) = graph.node.get(idx) {
                for inp in &node.input {
                    if !inp.is_empty() && !internal_outputs.contains(inp) && seen.insert(inp.clone()) {
                        if let Some(info) = self.tensors.get(inp) {
                            if !info.is_initializer {
                                external_inputs.push(inp.clone());
                            }
                        }
                    }
                }
            }
        }
        external_inputs
    }

    pub fn get_slice_outputs(&self, graph: &GraphProto, start_idx: usize, end_idx: usize) -> Vec<String> {
        let mut outputs = Vec::new();
        for idx in start_idx..end_idx {
            if let Some(node) = graph.node.get(idx) {
                for output in &node.output {
                    if output.is_empty() {
                        continue;
                    }
                    if let Some(info) = self.tensors.get(output) {
                        if info.is_output {
                            outputs.push(output.clone());
                            continue;
                        }
                        for (_consumer_name, consumer_idx) in &info.consumers {
                            if *consumer_idx >= end_idx {
                                outputs.push(output.clone());
                                break;
                            }
                        }
                    }
                }
            }
        }
        outputs
    }

    pub fn build_slice_dependencies(
        &self,
        graph: &GraphProto,
        slice_ranges: &[(usize, usize)],
    ) -> HashMap<usize, SliceDependencyInfo> {
        let mut result = HashMap::new();
        for (slice_idx, &(start, end)) in slice_ranges.iter().enumerate() {
            let inputs = self.get_slice_inputs(graph, start, end);
            let outputs = self.get_slice_outputs(graph, start, end);

            let mut skip_inputs = Vec::new();
            if slice_idx > 0 {
                let prev_end = slice_ranges[slice_idx - 1].1;
                for inp in &inputs {
                    if let Some(info) = self.tensors.get(inp) {
                        if let Some(prod_idx) = info.producer_node_idx {
                            if prod_idx < prev_end.saturating_sub(1) {
                                skip_inputs.push(inp.clone());
                            }
                        }
                    }
                }
            }

            result.insert(
                slice_idx,
                SliceDependencyInfo {
                    inputs,
                    outputs,
                    skip_inputs,
                    node_range: (start, end),
                },
            );
        }
        result
    }
}

impl std::fmt::Display for TensorGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorGraph({} tensors, {} nodes)",
            self.tensors.len(),
            self.num_nodes
        )
    }
}

pub struct SliceDependencyInfo {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub skip_inputs: Vec<String>,
    pub node_range: (usize, usize),
}
