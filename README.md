# GSoC Project Proposal Guide
**Project:** Optimize Quantized Model Inference Performance on ARM Devices with OpenVINO  
**Difficulty:** Medium  
**Required:** Strong C++ skills, Mac with ARM chip (Apple Silicon)

## 1) Project Context and Goal
This project focuses on improving OpenVINO Runtime performance for a **quantized Ultralytics/YOLO26 model** on ARM devices.

Expected impact:
- More int8 coverage for convolution-heavy quantized models
- Lower inference latency
- No correctness regressions

## 2) Required Reading / Code Familiarization
Review these PRs to understand the current state of int8 convolution support:
- Initial int8 convolution support: 
  https://github.com/openvinotoolkit/openvino/pull/30457
- Per-channel support: 
  https://github.com/openvinotoolkit/openvino/pull/33220
- Non-i32 convolution bias support: 
  https://github.com/openvinotoolkit/openvino/pull/33072

Also review the following OpenVINO documentation:
- How to build OpenVINO on macOS: https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_mac_arm.md
- Low Precision Transformations (LPT): https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library/advanced-guides/low-precision-transformations.html
- OpenVINO CPU plugin debug capabilities, including graph serialization, average counters, and blob dumping: https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/docs/debug_capabilities/README.md

## 3) What You Must Investigate Before Submitting a Proposal
You are **not expected to fully solve the problem before applying**.  
You are expected to demonstrate technical initiative, thorough investigation, and a realistic implementation plan.  
If time permits, you may also implement part of one technical gap described below (see Section 6).  

Before writing your proposal, do this:

- Build OpenVINO CPU plugin with `ENABLE_DEBUG_CAPS` and Python bindings: `openvino_intel_cpu_plugin`, `openvino_ir_frontend`, `py_ov_frontends`, `pyopenvino`
- Run both floating-point and quantized YOLO26 inference using the Python script [infer.py](https://github.com/alvoron/gsoc-2026-openvino/blob/main/scripts/infer.py).  
To run the script, create a dedicated Python virtual environment and install the following packages:
  ```
  pip install torch torchvision openvino ultralytics opencv-python
  ```
  How to run the script:
  ```
  python infer.py \
    --model models/yolo26n_quantized_openvino_model/yolo26n_quantized.xml \
    --image images/coco_bike.jpg \
    --output images/coco_bike_ov_int_result.jpg
  ```
  - YOLO26 floating-point model OpenVINO IR: https://github.com/alvoron/gsoc-2026-openvino/tree/main/models/yolo26n_openvino_model  
  - YOLO26 quantized model OpenVINO IR: https://github.com/alvoron/gsoc-2026-openvino/tree/main/models/yolo26n_quantized_openvino_model  
- Collect debug artifacts: execution model graph and graphs after each transformation phase.  
  Graph serialization docs: 
https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/docs/debug_capabilities/graph_serialization.md  
  **Note:** The Python package `ultralytics` checks whether the `openvino` package is installed before running inference. To force `ultralytics` to use a custom OpenVINO build, override the `PYTHONPATH` environment variable:
  ```
  export PYTHONPATH=<OPENVINO_REPO_ROOT>/bin/arm64/Release/python:<OPENVINO_REPO_ROOT>/tools/ovc
  ``` 
- Review the graphs and identify FP16 convolution nodes.

## 4) Technical Gaps You Should Analyze
Your analysis should address these blockers:

- **Pattern gap with Swish**
  - Observed sequence: `Convolution -> Multiply -> Add -> Swish -> FakeQuantize`
  - Current handled pattern: `Convolution -> Multiply -> Add -> FakeQuantize`
  - Result: Swish prevents int8 Convolution node creation

- **Activation zero-point representation**
  - A `Subtract` appears on convolution activation path
  - This zero-point should be represented through Convolution activation `QuantizationInfo` instead

Provide an implementation plan for one or both gaps.

## 5) Required Structure for Your Proposal
Please include these sections explicitly:

- Motivation and understanding of the problem
- Initial graph analysis findings
- Proposed implementation approach
- Validation plan
- Timeline with milestones

## 6) Start Addressing a Technical Gap (Optional)
You may try to start addressing the Swish pattern gap. To do this:

- Enable int8 convolution transformation by deleting `CPU_DISABLE_PASS_ARM(lptManager, ConvolutionTransformation)` in `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp`.
- After that, you may observe ACL issues described in:
  - https://github.com/ARM-software/ComputeLibrary/issues/1252
  - https://github.com/ARM-software/ComputeLibrary/issues/1253
- If you observe similar issues, update the `ComputeLibrary` submodule in `src/plugins/intel_cpu/thirdparty/ComputeLibrary`. Use the `main` branch instead of `v52.8.0` and apply https://github.com/ARM-software/ComputeLibrary/pull/1270 if it is not merged yet.
- If you do not observe any ACL error after deleting the line, you may avoid the ACL upgrade.
- You may need to remove the `ACLConvolutionExecutor` single post-op limitation and accept both `Activation` and `FakeQuantize`.
- Review how activations are fused into Convolution nodes and plan how to handle both the `Swish` node and `FakeQuantize`, so the execution graph contains one int8 convolution node with fused activation.
