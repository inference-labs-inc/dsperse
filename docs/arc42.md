# Dsperse Architecture (arc42)

This document describes the software architecture of Dsperse as it exists in this repository today. It follows the spirit of the arc42 template while focusing on what is implemented: an ONNX-only, CLI-driven pipeline integrating EZKL for circuit generation, proving, and verification.

---

## 1. Goals and Quality Objectives

- Provide a simple, local-first workflow to slice ONNX models into segments and run chained inference.
- Enable zkML workflows by compiling (EZKL), proving, and verifying per-segment execution.
- Keep the UX approachable via an interactive CLI with sensible defaults.
- Be explicit about support scope: ONNX models only (no .pth / TorchScript models).

Primary quality goals
- Reliability of the pipeline and artifacts (deterministic file layout, resumable steps).
- Operability: clear CLI, helpful prompts, actionable errors/logging.
- Extensibility: new backends or steps can be added behind well-defined interfaces.

---

## 2. Constraints

- Model format: ONNX only.
- Cryptographic backend: EZKL CLI (Halo2-based) for circuit compilation, setup, proving, verifying.
- Runtime environment: Python 3.9+, local file system; interactive CLI by default.
- Artifacts are JSON and file-based to allow inspection and reuse between steps.

---

## 3. Context and Scope

Dsperse is a single-node CLI application. It does not require a network, message bus, or external database. The CLI reads/writes files under a chosen model directory and orchestrates the EZKL command-line tool.

External interfaces
- EZKL CLI: invoked to generate settings, compile circuits, setup, prove, and verify.
- ONNX Runtime: used for non-zk per-segment inference and shape handling.

Inputs/outputs (high-level)
- Input: model.onnx, input.json.
- Outputs: slices/ (segment_*/ with ONNX segments and EZKL artifacts), analysis/model_metadata.json (optional), run/run_*/ (run_result.json, intermediate tensors, proofs).

---

## 4. Solution Strategy

- Slice the original ONNX graph into smaller ONNX segments with explicit I/O tensors.
- For each segment, compile EZKL circuits (settings, compiled model, proving/verification keys).
- Execute chained inference over segments; where EZKL witnesses are produced, later steps can generate proofs.
- Store everything as files to enable step-by-step inspection and reproducibility.

---

## 5. Building Block View

Top-level components
- CLI layer (src/cli/*): subcommands slice, compile, run, prove, verify, full-run (orchestration).
- Slicer (src/slicer.py): builds per-segment ONNX files and slices/metadata.json.
- Compiler (src/compiler.py + src/backends/ezkl.py): creates EZKL artifacts for selected segments.
- Runner (src/runner.py): chained execution across segments, producing run/run_*/ outputs and run_result.json.
- Prover (src/prover.py): generates proofs for segments with valid witnesses.
- Verifier (src/verifier.py): verifies produced proofs and records results.
- ONNX execution (src/backends/onnx_models.py): utility to run ONNXRuntime for segments.

Key data structures and files
- slices/metadata.json: operational metadata about segments and EZKL artifacts.
- analysis/model_metadata.json: optional analysis dump for human inspection.
- run/run_*/run_result.json: per-run execution summary and per-segment results.

---

## 6. Runtime View

Typical full run (local)
1) slice: create slices/segment_*/segment_i.onnx and slices/metadata.json.
2) compile: for chosen layers, create EZKL settings/compiled circuit/keys under each segment.
3) run: execute segments in order; produce inputs/outputs (and witnesses if EZKL path is used) under run/run_*/ and write run_result.json.
4) prove: for segments with witnesses, call ezkl prove; update run_result.json with proof_execution info.
5) verify: call ezkl verify per segment; update run_result.json with verification_execution info and overall counts.

The full-run command orchestrates the above and handles prompting/paths for a one-shot UX.

---

## 7. Deployment View

- Single machine execution; no daemon or server component.
- Requirements: Python, EZKL CLI installed and (optionally) EZKL SRS downloaded locally.
- Artifacts are stored next to the model so the workflow can be resumed or audited easily.

---

## 8. Crosscutting Concepts

- Logging and UX: colorized terminal output, interactive prompts with defaults, centralized logging utilities in src/cli/base.py.
- File layout contracts: stable directory structure for slices and runs; tools search for metadata.json to auto-detect context.
- Backend abstraction: EZKL integration is encapsulated; ONNXRuntime used for non-zk inference.

---

## 9. Architecture Decisions (selected)

- ONNX-only support: simplifies graph handling and avoids maintaining multiple model loaders.
- EZKL as ZK backend: leverage mature Halo2-based CLI and ecosystem.
- File-first orchestration: favor transparent artifacts over opaque state to aid debugging and reproducibility.
- Segmented execution: break models into smaller units to improve tractability and incremental zk workflows.

---

## 10. Quality Requirements

- Correctness: segmented I/O shape checks and validation before/after each step.
- Usability: commands prompt for missing inputs and suggest defaults; errors include recovery hints.
- Performance: compile/prove/verify only selected layers; reuse artifacts across runs.

---

## 11. Risks and Technical Debt

- Mismatch between docs and CLI options if commands evolve; mitigated by keeping README in sync with parsers.
- EZKL version drift impacting CLI flags or artifact formats; mitigated by pinning versions and testing common flows.
- Large models and memory pressure during slicing or runtime; mitigated by segment size selection and ONNX-only scope.

---

## 12. Glossary

- Segment: a sliced subgraph of the original ONNX model with explicit inputs/outputs.
- Slices metadata: operational JSON describing segments and their artifacts (settings, keys, etc.).
- Run: a single chained inference over segments, stored under run/run_*/.
- Witness: EZKL-generated data enabling proof creation for a segment execution.

---

This arc42 summary reflects the current repository (as of 2025-09-16).

1. **Layer Splitting in Different Model Types:**  
   - For convolutional layers, would you consider the unfolding operation (im2col) to convert them into a matrix form for circuitization, or is the initial focus solely on fully connected layers?
     - In the context of convolutional layers, one useful technique is called im2col. This operation—short for "image to column"—rearranges image data (or more generally, multi-dimensional feature maps) into a 2D matrix. The purpose is to transform the convolution operation into a matrix multiplication, which can be computed very efficiently using highly optimized linear algebra routines. While fully connected layers are naturally represented as dense matrices of weights and biases, convolutional layers are structured differently. In a convolutional layer, the weights are stored as a set of filters that slide across the input feature map. These filters have local connectivity and weight sharing, meaning that the same filter is applied across different regions of the input. By applying the im2col transformation, the input is rearranged into columns, and the convolution filters are flattened into rows, effectively turning the operation into a dense matrix multiplication. Although the resulting matrices may exhibit sparsity or repeated patterns, this transformation makes it easier to apply the same vertical splitting approach for circuitization.

2. **Resource Scoring and CLI Parameters:**  
   - What parameters (e.g., maximum layers per circuit, maximum nodes per circuit) should be exposed by default?  
     - To tailor workloads to individual miners' capabilities, we propose exposing parameters such as the maximum number of nodes (weights and biases) a miner’s machine can handle. Given the variability in miners' hardware configurations, these parameters should be configurable by the miners themselves. For example, the command-line interface might include options like: `Dsperse-miner --max-nodes 1000 --max-layers 1` In this example: `--max-nodes` sets the maximum number of nodes (individual weight and bias pairs) that the miner is willing to process. `--max-layers` specifies the default grouping of vertical layers per circuit. These parameters will be refined over time through discussions with the network’s miners.

3. **Proof Assembly Flow:**  
   - Can you add more details on the fallback mechanism when a miner’s assigned circuit fails, such as retry logic or threshold limits?
     - The proof assembly process begins when validators receive computed proofs for each assigned vertical layer from miners. Validators are aware of the number of proofs required to complete a full inference. If a miner's assigned circuit fails to produce a valid proof within a preset threshold time, the system initiates a fallback mechanism. This may involve reducing the circuit size by splitting it into smaller segments—potentially down to a single vertical layer—and reassigning that workload to the same or a different miner. Additionally, if a miner experiences repeated failures (e.g., failing three consecutive assignments), the incentive mechanism on the subnet could automatically reduce that miner’s allocation or temporarily exclude them from receiving assignments. This dynamic retry and threshold-based approach ensures robust proof assembly, maintaining the system's overall efficiency and reliability.

4. **Integration with External Tools:**  
   - Are there any modifications or wrappers needed for ezkl/Halo 2, or will Dsperse rely on their native support for circuit chaining?
     - Dsperse is designed to leverage existing CLI-based workflows provided by tools like ezkl and Halo 2 for circuit generation. We plan to rely on their native support for circuit chaining without significant modifications. Additionally, we are evaluating the "Expander Compiler Collection," a Rust-based zkML library, for potential integration via CLI commands. Our goal is to keep the integration straightforward, allowing Dsperse to utilize these tools without extensive custom wrappers.
