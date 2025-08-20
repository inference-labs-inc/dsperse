# Bugbash Testing Document

## 📸 Scenario 1 Screenshots and Results

### Directory Structure and Setup
![Directory Structure - One Level Up](Bugbash/scenario1/onelevelupdir.png)
*Directory structure showing the relationship between doom, scenario1, and run directories*

![Successful Slice Operation](Bugbash/scenario1/successful_slice.png)
*Successful slicing of the doom model into 5 segments*

![Slice Directory Contents](Bugbash/scenario1/slice_dire.png)
*Contents of the slice directory after successful slicing*

![Metadata Shows Slices Exist Correctly](Bugbash/scenario1/metadata_slice_exist_correct.png)
*Metadata confirmation that all slices were created correctly*

### Circuitization Process
![Whole Model Circuitization - Not All Layers](Bugbash/scenario1/wholemodel%20circuitization!Not%20all%20layers.png)
*Initial whole model circuitization attempt showing layer limitations*

![Whole Model Circuitization - Layer Details](Bugbash/scenario1/wholemodel%20circuitization!Not%20all%20layers2.png)
*Detailed view of whole model circuitization process*

![Partial Circuitization - Layers 0 and 1](Bugbash/scenario1/partiallycircuitize%20layer%200%20and%201.png)
*Partial circuitization of specific layers (0 and 1)*

![Segment 3 Circuitization Fix](Bugbash/scenario1/circuitize_slice3.png)
*Fixing segment 3 circuitization with logrows=19 and kzg24.srs*

![EZKL Layer 3 Fix for Doom](Bugbash/scenario1/FIX_EZKLLAYER3_DOOM.png)
*Detailed fix for EZKL layer 3 in the doom model*

### Execution and Results
![Run Success with EZKL Circuits](Bugbash/scenario1/Run_success.png)
*Successful execution using EZKL circuits instead of ONNX fallback*

![Metadata Autocomplete Issue](Bugbash/scenario1/metadata_autocomplete.png)
*Issue with metadata autocomplete not working properly*

![RunnerAnalyzer Hardcoded Path Issue](Bugbash/scenario1/The%20RunnerAnalyzer%20is%20hardcoded%20to%20look%20for%20metadata%20in%20model_directory:slices:metadata.json..png)
*Bug where RunnerAnalyzer was hardcoded to look in wrong metadata location*

### Final Results
![Prove Command Success](Bugbash/scenario1/prove.png)
*Successful proof generation for all 5 segments (100% completion)*

![Verify Command Success](Bugbash/scenario1/verify.png)
*Successful verification of all 5 segment proofs (100% verification)*

## Version Information
**Project:** Dsperse v1.0.0  
**Branch:** dan/release  
**Commit:** e5d5a1847aaa377ab1eec5d7aeec9682abf64371  
**Date:** 2025-08-19 10:23:21 -0700  
**Author:** dan  
**Last Update:** Rename project from "Kubz" to "Dsperse" across all files, including CLI, documentation, installer, and internal references.

## Setup
 
 - `install.sh`    file will install through 'pip' , add 'uv' 
 - Set a working directory outside of the current working dir
 - `kzg25.srs` and `kzg26.srs` will not supported , will be required for larger slices. 
 - Missing packages : `onnxruntime_extensions`,`transformers`, `onnx_graphsurgeon`
 - `kubz --help` still existed instead of `dsperse --help`
 - a guide for Directory setup: ./doom, ./scenario1, ./run, ./slices(?)

## Set up ML Flow

- [ ] MLFLOW TO TRACK RUN


## Test Scenarios

### Scenario 1: doom_Desktop
**Screenshots folder:** `screenshots/scenario1_doom_desktop/`

#### Section 1: Slice 

`  --model-dir MODEL_DIR`  ✅
`  --output-dir OUTPUT_DIR` ✅
`  --input-file INPUT_FILE` ✅
    - Auto detect Input: ✅

#### Section 2: Run 

`  --model-dir MODEL_DIR`  ✅
`  --output-file OUTPUT_FILE` ✅
`  --input-file INPUT_FILE` ✅


1. metadata_autocomplete won't go through! the input address should set based on : `--model-dir ~/Desktop/scenario1`


2.The RunnerAnalyzer is hardcoded to look for metadata in model_directory/slices/metadata.json.
3.Run folder is not in the output directory, one level higher[instead of  `~/Desktop/scenario1/run` it is in `~/Desktop/run`]

#### Section 3: Circutize 

##### Partial Circuitize 0,1: ✅

##### Full Circuization: 

1. full model circuitization instead of all layers. [Tempfixed : layers 0-4]
2. Segment 3 doesn't gen vk, pk, FIX : Edits made: set "logrows": 19, recompiled, and ran setup with --srs-path ~/.ezkl/srs/kzg24.srs for segment_3; keys were generated successfully.


Successfully circuitized all 5 segments
Fixed segment 3 with logrows=19 and kzg24.srs
Generated compiled circuits, proving keys, and verification keys

#### Section 4: prove
    - Section 4-1 : run : run/run_20250820_132804
Proved segments: 5 of 5
Proof generation percentage: 100.0%
Overall security: 100.0%
Completed in 80.66 seconds

#### Section 5: verify 


### Scenario 2: net_to_CustomDirectory  
**Screenshots folder:** `screenshots/scenario2_net_custom_directory/`

### Scenario 3: Resnet_partial
**Screenshots folder:** `screenshots/scenario3_resnet_partial/`

### Scenario 4: Last Layer Slice
**Screenshots folder:** `screenshots/scenario4_last_layer_slice/`

## Quality of Life Files
<!-- Files and utilities that improve development experience -->

## README.md Feedback

 Awesome README!!!!!! probably the best documented in the space. 


__[Prefrence]__

- manual installation a seperate file (too many steps and scenario to follow)

- the workflow may be confusing: simple slice -> Run and clearly show in order to prove the model u need to slice->circuitize->gen prove-> Verify for the test and then provide more info on how to work with the calibrations and other settings. 

- Add a tracker like ml_flow for automated way to follow up. 

