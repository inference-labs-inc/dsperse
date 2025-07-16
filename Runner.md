# 🦜 Production Runner System

<div align="center">

![Parrot Logo](https://img.shields.io/badge/🦜-Runner%20System-blue?style=for-the-badge&logo=parrot)

**Secure • Adaptable • Fast • Reliable**

*Production inference with EzKL circuit integration & automatic fallback*

</div>

---

## 🚀 Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run production validation
cd src && python runners/deploy_production.py

# Basic usage
from runners.runner import Runner
runner = Runner("models/doom")
result = runner.infer()
print(f"🛡️ Security: {result['overall_security']}%")
```

---

## 📁 Core Files

```
src/runners/
├── runner_metadata.py     # Metadata generator with auto-validation
├── runner.py             # Production inference engine
├── test_runner_system.py # Comprehensive test suite
├── deploy_production.py  # Production validation
└── Runner.md            # This documentation
```

---

## 🛡️ Security & Adaptability

### Security Levels
- **100%**: All slices use verified EzKL circuits
- **80%**: 4/5 slices use circuits, 1 ONNX fallback  
- **0%**: Whole ONNX model (fastest, no verification)

### Automatic Fallback System
```python
# Smart fallback: EzKL → ONNX when needed
if circuit_available and verified and size_ok:
    use_ezkl_circuit()
else:
    fallback_to_onnx_slice()
```

---

## 📊 Metadata Structure

| Field | Description | Security Impact |
|-------|-------------|-----------------|
| `overall_security` | Security percentage (0-100%) | Core metric |
| `verified_slices` | Which slices use verified circuits | Trust level |
| `execution_chain` | Linked list of slice execution | Inference flow |
| `fallback_map` | Circuit → ONNX mapping | Reliability |
| `time_limit` / `size_limit` | Performance constraints | Auto-fallback triggers |

### Key Status Flags

| Flag | `true` | `false` / Warning |
|------|--------|-------------------|
| `Analyzer` | ✅ Ready | 🔧 Auto-runs analyzer |
| `Reconstruct` | ✅ Sliced | ⚠️ Warning only |
| `precircuit` | `"circuits compiled and ready"` | ⚠️ **No circuit authenticity guaranteed** |

**When `precircuit = "circuits compiled and ready"`**: System warns that circuits exist but verification is not guaranteed. Run `Model_circuitizer.py` for full authenticity.

---

## 🔬 Inference Method Comparison

Based on comprehensive testing with actual models (doom: 417,943 parameters, net: 62,006 parameters):

| Model | Method | Predicted Class | Security % | Time (s) | Classification Error | Time Overhead % | Security Gain % | Circuit Sizes (bytes) |
|-------|--------|----------------|------------|----------|---------------------|-----------------|-----------------|---------------------|
| **doom** | Whole Model | 2 | 80.0% | 0.006 | - | - | - | - |
| | Sliced Auto | 2 | 80.0% | 0.089 | ✅ | +1297.4% | +0.0% | [2827, 19048, 37728, 1607065, 7469] |
| |
| **net** | Whole Model | 3 | 100.0% | 0.002 | - | - | - | - |
| | Sliced Auto | 3 | 100.0% | 0.106 | ✅ | +4294.1% | +0.0% | [2508, 10523, 192885, 41057, 3671] |
| 

---

## 🧮 Circuit vs ONNX Computational Differences

**Production validation with 12-decimal precision error analysis showing realistic circuit computation trade-offs:**

### DOOM Model Results
```
📊 Max Absolute Error: 0.061248779297
📊 Mean Absolute Error: 0.053846086775  
📊 Sum Absolute Error: 0.376922607422
📊 Relative Error: 0.000219920020
⚙️  Execution: 4 circuits, 1 ONNX slice
🎯 Classification: ONNX=2, Circuit=2 ✅ MATCH
```

**Top 3 Largest Differences:**
1. Index 3: ONNX=270.841094970703, Circuit=270.902343750000, Diff=0.061248779297
2. Index 5: ONNX=276.001373291016, Circuit=276.062500000000, Diff=0.061126708984  
3. Index 1: ONNX=272.465393066406, Circuit=272.523437500000, Diff=0.058044433594

### NET Model Results
```
📊 Max Absolute Error: 0.003621816635
📊 Mean Absolute Error: 0.001755008101
📊 Sum Absolute Error: 0.017550081015  
📊 Relative Error: 0.001193855264
⚙️  Execution: 4 circuits, 1 ONNX slice
🎯 Classification: ONNX=3, Circuit=3 ✅ MATCH
```

**Top 3 Largest Differences:**
1. Index 7: ONNX=2.012003183365, Circuit=2.015625000000, Diff=0.003621816635
2. Index 9: ONNX=-1.425451755524, Circuit=-1.421875000000, Diff=0.003576755524
3. Index 6: ONNX=-0.695682823658, Circuit=-0.699218750000, Diff=0.003535926342



---

## 🎛️ User Control

### Time Limit Configuration

```python
# Edit metadata time_limit for automatic circuit fallback
import json

with open("models/doom/doom_Runner_Metadata.json", "r") as f:
    metadata = json.load(f)

# Set time limit (seconds)
metadata["time_limit"] = 0.05  # 50ms per slice


with open("models/doom/doom_Runner_Metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
```

### Size Limit Configuration

```python
# Edit metadata size_limit for circuit usage
metadata["size_limit"] = 500000   # 500KB limit

```



### Custom Verification Settings

```python
# Custom verification for specific slices
custom_verification = {
    "slice_0": True,   # Use verified circuit
    "slice_1": True,   # Use verified circuit  
    "slice_2": False,  # Force ONNX fallback
    "slice_3": False,  # Force ONNX fallback (size issue)
    "slice_4": True    # Use verified circuit
}

metadata["verified_slices"] = custom_verification

# Recalculate overall security
verified_count = sum(1 for v in custom_verification.values() if v)
total_slices = len(custom_verification)
metadata["overall_security"] = (verified_count / total_slices) * 100
```


### Runtime Override

```python
from runners.runner import Runner

# Runtime time limit override
runner = Runner("models/doom")
runner.time_limit = 0.01  # Very strict 10ms limit

# Runtime size limit override  
runner.size_limit = 50000  # 50KB limit (forces more fallbacks)

# Run with custom limits
result = runner.infer(mode="auto")
print(f"Fallbacks used: {sum(1 for r in result['execution_results'] if r['fallback_used'])}")
```

---


### Testing
```python  
# Run comprehensive tests (21 tests)
python -m pytest test/test_runner_system.py -v

# Test inference error tolerance
python -m pytest test/test_runner_system.py::TestInferenceErrorTolerance -v
```

### Basic Inference
```python
from runners.runner import Runner

# Auto mode (smart fallback)
runner = Runner("models/doom") 
result = runner.infer(mode="auto")

# ONNX mode (fastest)
result = runner.infer(mode="onnx_only")

# Results
print(f"Class: {result['predicted_class']}")
print(f"Security: {result['overall_security']}%") 
print(f"Method: {result['execution_method']}")
```

### Static Methods
```python
# One-liner inference
result = Runner.run_inference("models/doom")

# One-liner metadata
metadata_path = RunnerMetadata.generate_for_model("models/doom")
```


<div align="center">


*Kubz V.0.1*

</div>