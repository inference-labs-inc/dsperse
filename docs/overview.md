# DSperse: Distributed zkML

## Overview

DSperse is a proving-system-agnostic intelligent slicer for verifiable AI. It decomposes ONNX neural network models into circuit-compatible segments and orchestrates compilation, inference, proving, and verification across pluggable ZK backends.

### Core Purpose
The project aims to solve a significant challenge in zkML (zero-knowledge machine learning) by introducing intelligent model slicing that enables distributed proof computation across heterogeneous hardware.

### Key Technical Innovation
The main innovation is the concept of "model slicing" where:
1. Instead of processing an entire neural network at once
2. The system splits the neural network into manageable segments
3. Each segment can be processed independently for analysis, inference, or proof generation

### Primary Goals
1. **Model Slicing**
    - Split neural network models into individual layers or custom segments
    - Support ONNX models
    - Enable detailed analysis of model components

2. **Distributed Computation**
    - Break down large ML models into manageable pieces
    - Enable parallel processing across multiple machines
    - Support both GPU and non-GPU nodes

3. **Resource Optimization**
    - Reduce RAM requirements through model splitting
    - Implement efficient inference pipelines
    - Better manage compute resources

4. **System Flexibility**
    - Support for different model types
    - Configurable slicing strategies
    - Adaptable to different hardware capabilities

5. **Zero-Knowledge Proofs**
    - Generate proofs for sliced model execution via JSTprove integration
    - Proving-system-agnostic design supporting Expander and Remainder backends
    - Optimize proof generation for distributed environments

### Implementation Framework
- Built on top of existing tools:
    - ONNX for model representation and interoperability
    - JSTprove (`jstprove_circuits` Rust crate) for zero-knowledge proof generation
    - Expander and Remainder as the underlying proving systems

- Comprehensive CLI interface for:
    - Model slicing
    - Inference
    - Proof generation
    - Proof verification

- Designed to work with various neural network architectures
- Focuses on practical applications of zkML technology
