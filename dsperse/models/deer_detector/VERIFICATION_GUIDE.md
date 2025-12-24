# JSTprove Verification Guide for Slices 41 & 52

This document describes the zero-knowledge verification artifacts for deer detector slices 41 and 52, compiled with the JSTprove backend.

## Overview

The YOLOv8n deer detection model is decomposed into 64 slices. Two representative slices from the detection neck (slices 41 and 52) have been compiled to zero-knowledge circuits using JSTprove for cryptographic verification of inference correctness.

## Slice 41: Feature Fusion Layer

**Location**: `slices/slice_41/`

### Model Details
- **Layer**: `/model.22/cv2.0/cv2.0.2/Conv`
- **Type**: Convolutional layer (1x1 kernel)
- **Parameters**: 4,160 total
  - Weight: [64, 64, 1, 1] = 4,096 params
  - Bias: [64] = 64 params
- **Input shape**: [1, 64, 80, 80]
- **Output shape**: [1, 64, 80, 80]
- **Function**: Feature fusion in detection neck (80x80 feature map)

### JSTprove Artifacts

```
slices/slice_41/payload/jstprove/
├── slice_41_circuit.txt                      # Compiled ZK circuit
├── slice_41_circuit_quantized_model.onnx     # Quantized ONNX model
├── slice_41_circuit_witness_solver.txt       # Witness generation solver
├── slice_41_circuit_architecture.json        # Circuit architecture metadata
├── slice_41_circuit_metadata.json            # Compilation metadata
├── slice_41_circuit_wandb.json               # Training/compilation logs
└── slice_41_settings.json                    # JSTprove settings
```

### Verification Workflow

1. **Input**: Intermediate activation from slice 40 (shape: [1, 64, 80, 80])
2. **Circuit execution**: Run slice_41_circuit.txt with quantized weights
3. **Witness generation**: Use slice_41_circuit_witness_solver.txt
4. **Proof generation**: JSTprove generates ZK proof of correct execution
5. **Verification**: Cryptographic verification (~10-50ms)
6. **Output**: Verified activation for slice 42 (shape: [1, 64, 80, 80])

### Settings (`slice_41_settings.json`)

```json
{
  "backend": "jstprove",
  "model_path": "slices/slice_41/payload/slice_41.onnx",
  "circuit_path": "slices/slice_41/payload/jstprove/slice_41_circuit.txt",
  "compiled_at": "slices/slice_41/payload/jstprove",
  "note": "JSTprove handles settings internally"
}
```

## Slice 52: Multi-Scale Detection Preparation

**Location**: `slices/slice_52/`

### Model Details
- **Layer**: `/model.22/cv2.1/cv2.1.2/Conv`
- **Type**: Convolutional layer (1x1 kernel)
- **Parameters**: 4,160 total
  - Weight: [64, 64, 1, 1] = 4,096 params
  - Bias: [64] = 64 params
- **Input shape**: [1, 64, 40, 40]
- **Output shape**: [1, 64, 40, 40]
- **Function**: Feature processing in detection neck (40x40 feature map)

### JSTprove Artifacts

```
slices/slice_52/payload/jstprove_circuitization/
├── slice_52_circuit.txt                      # Compiled ZK circuit
├── slice_52_circuit_quantized_model.onnx     # Quantized ONNX model
├── slice_52_circuit_witness_solver.txt       # Witness generation solver
├── slice_52_circuit_exc                      # Circuit executable
└── slice_52_settings.json                    # JSTprove settings
```

### Verification Workflow

1. **Input**: Intermediate activation from slice 51 (shape: [1, 64, 40, 40])
2. **Circuit execution**: Run slice_52_circuit.txt with quantized weights
3. **Witness generation**: Use slice_52_circuit_witness_solver.txt
4. **Proof generation**: JSTprove generates ZK proof of correct execution
5. **Verification**: Cryptographic verification (~10-50ms)
6. **Output**: Verified activation for slice 53 (shape: [1, 64, 40, 40])

### Settings (`slice_52_settings.json`)

```json
{
  "backend": "jstprove",
  "model_path": "slices/slice_52/payload/slice_52.onnx",
  "circuit_path": "slices/slice_52/payload/jstprove_circuitization/slice_52_circuit.txt",
  "compiled_at": "slices/slice_52/payload/jstprove_circuitization"
}
```

## Compilation Details

### JSTprove Backend
- **Quantization**: FP32 → Fixed-point representation for circuit compatibility
- **Circuit size**: ~10K-50K constraints per slice (depends on layer complexity)
- **Compilation time**: ~1-5 minutes per slice
- **Proof generation time**: ~1-5 seconds per slice (hardware dependent)
- **Verification time**: ~10-50ms per proof

### Why These Slices?

**Slice 41** and **Slice 52** were selected for ZK verification because:

1. **Representative operations**: Both are 1x1 convolutions in the detection neck
2. **Different scales**: Slice 41 operates on 80x80 feature maps, Slice 52 on 40x40
3. **Critical path**: Both are in the feature pyramid network (FPN) that enables multi-scale detection
4. **Moderate complexity**: Manageable circuit size while demonstrating real zkML capability

## Integration with Video Inference

During video processing (`run_deer_detection.py`), ZK verification is performed periodically:

1. **Frame processing**: Each video frame goes through all 64 slices
2. **Activation capture**: Intermediate outputs from slices 41 & 52 are captured
3. **Proof generation**: JSTprove generates proofs for these slices (every Nth frame)
4. **Watermarking**: Verified frames receive "JSTprove Verified" watermark
5. **Output**: Video contains both detections and cryptographic attestations

### Performance Considerations

- **Full verification**: ~2-10 seconds per frame (all slices)
- **Selective verification**: ~0.2-1 second per frame (slices 41 & 52 only)
- **Sampling strategy**: Verify every 10th frame to maintain real-time performance (35 FPS)
- **Batch verification**: Can verify multiple frames in parallel for throughput

## Verification Commands

### Compile Slice to Circuit
```bash
# Using Dsperse CLI
dsperse compile --backend jstprove \
  --model slices/slice_41/payload/slice_41.onnx \
  --output slices/slice_41/payload/jstprove
```

### Generate Proof
```bash
# Coming soon: JSTprove CLI integration
jstprove prove \
  --circuit slices/slice_41/payload/jstprove/slice_41_circuit.txt \
  --witness witness.json \
  --output proof.json
```

### Verify Proof
```bash
# Coming soon: JSTprove CLI integration
jstprove verify \
  --circuit slices/slice_41/payload/jstprove/slice_41_circuit.txt \
  --proof proof.json
```

## File Descriptions

### Circuit Files (`.txt`)
- Human-readable circuit representation
- Contains gates, wires, and constraints
- Used by JSTprove prover and verifier

### Quantized Models (`.onnx`)
- Fixed-point quantized version of the slice
- Ensures deterministic computation in ZK circuit
- Maintains accuracy while enabling efficient proving

### Witness Solvers (`.txt`)
- Describes how to compute circuit witness from inputs
- Maps ONNX operations to circuit constraints
- Used during proof generation

### Settings Files (`.json`)
- Configuration for Dsperse-JSTprove integration
- Paths to circuit, model, and compilation artifacts
- Backend-specific parameters

### Architecture Files (`.json`)
- Detailed circuit structure and statistics
- Layer-by-layer breakdown
- Constraint counts and complexity metrics

## Security Guarantees

### What is Proven?
- **Correctness**: The slice output was computed correctly from the input
- **Integrity**: No tampering with model weights or intermediate activations
- **Completeness**: All computations were performed as specified

### What is Hidden?
- **Model weights**: Remain private (not revealed in proof)
- **Intermediate activations**: Not disclosed during verification
- **Input data**: Can be kept private (depending on use case)

## Future Work

- **Full model verification**: Extend ZK proofs to all 64 slices
- **Optimized circuits**: Reduce proof generation time for real-time use
- **Batch verification**: Verify multiple frames simultaneously
- **On-chain verification**: Deploy verifier to blockchain for decentralized trust
- **Adaptive sampling**: Intelligently select which frames to verify based on confidence

## References

- **JSTprove**: [Zero-knowledge proof backend for neural networks]
- **Dsperse**: [Model slicing framework for distributed zkML]
- **YOLOv8**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)

## Contact

For questions about verification or to request additional slices to be compiled, please contact the Dsperse team.

