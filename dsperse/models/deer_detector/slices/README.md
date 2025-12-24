# Deer Detector: Training, Slicing, and ZK Verification

This document describes the complete pipeline for training a YOLOv8n deer detection model, slicing it into 64 computational segments using Dsperse, circuitizing selected slices with JStprove backend, and verifying execution with zero-knowledge proofs and watermarks.

## Model Training

The deer detector was trained using YOLOv8n (nano) architecture on the Kaggle wildlife 10-class object detection dataset, specifically focusing on deer detection as a single-class task. Training was performed on Google Colab with a Tesla T4 GPU over 30 epochs with a batch size of 16 and input resolution of 640×640 pixels. The model achieved a validation mAP50 of 0.941 and mAP50-95 of 0.781, demonstrating strong detection performance on deer images. The trained PyTorch model was exported to ONNX format (opset 18) with size optimization, resulting in an 11.7 MB model suitable for edge deployment and zkML integration.

To ensure JStprove compatibility, the model underwent post-processing to replace Sigmoid activation functions with ReLU operations, as JStprove's circuit compilation requires operations that can be expressed in its supported arithmetic circuit format. This modification maintains detection accuracy while enabling zero-knowledge proof generation for model execution verification.

## Dsperse Slicing and JStprove Circuitization

The ONNX model was sliced into 64 computational segments using Dsperse's ONNX graph slicing engine. Each slice represents a distinct computational subgraph with well-defined input/output dependencies, enabling distributed execution and selective zkML verification. The slicing process generated comprehensive metadata for each slice, including parameter counts, tensor shapes, and dependency graphs. Slices 41 and 52 were selected for JStprove circuitization based on their computational characteristics and compatibility with JStprove's supported operations.

Circuitization was performed using the JStprove backend, which compiles ONNX slice models into arithmetic circuits suitable for zero-knowledge proof generation. The circuitization process quantizes model weights and activations, generates circuit files, and produces witness solver specifications. For slice 41, the circuitization generated `slice_41_circuit.txt`, `slice_41_circuit_quantized_model.onnx`, and associated metadata files. Similarly, slice 52 was circuitized with corresponding circuit artifacts stored in the `jstprove_circuitization` directory. Both slices contain convolutional layers with 64 input/output channels and 1×1 kernel sizes, making them ideal candidates for zkML verification.

## Video Testing with ZK Verification and Watermarking

The sliced model was tested on `test.mp4`, a 9.8-second video (960×720 resolution, 35 FPS, 343 frames) containing deer footage. Inference was performed by executing all 64 slices sequentially, with slices 41 and 52 executed through their JStprove circuits to generate zero-knowledge proofs. The test results, saved to `test_result.mp4`, demonstrate successful deer detection across multiple frames with bounding box annotations and confidence scores.

Zero-knowledge verification was performed using JStprove's verification system, which validates that the circuit execution matches the expected outputs without revealing the intermediate computations. The verification process requires the circuit file, input JSON, output JSON, witness file, and proof file. Upon successful verification, JStproveN watermarks are embedded into the proof, providing cryptographic attestation of model execution integrity. These watermarks serve as tamper-evident markers that can be verified independently, ensuring that the model inference was performed correctly and that the results have not been manipulated.

The complete verification workflow demonstrates how distributed zkML systems can provide cryptographic guarantees for AI model execution, enabling trustless verification of inference results in decentralized and adversarial environments. The combination of model slicing, selective circuitization, and watermarking provides a scalable approach to zkML deployment for complex computer vision models like YOLOv8.

