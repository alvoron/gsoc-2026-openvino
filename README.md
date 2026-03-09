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

## 2) What You Must Investigate Before Submitting a Proposal
You are **not expected to fully solve the problem before applying**.  
You are expected to show technical initiative, good investigation, and a realistic implementation plan.

Before writing your proposal, do this investigation:

- Run quantized YOLO26 inference with OpenVINO `benchmark_app` tool  
  `benchmark_app` tool: https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp/benchmark_app  
  YOLO26 model OpenVINO IR: https://github.com/alvoron/gsoc-2026-openvino/tree/main/yolo26
- Collect debug artifacts: execution model graph and graphs after each transformation phase.  
  Graph serialization docs: 
  https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/docs/debug_capabilities/graph_serialization.md
- Review graphs and find FP16 convolutino nodes there.

## 3) Required Reading / Code Familiarization
You should review these PRs to get a picture of int8 Convolution support:
- Initial int8 convolution support: 
  https://github.com/openvinotoolkit/openvino/pull/30457
- Per-channel support: 
  https://github.com/openvinotoolkit/openvino/pull/33220
- Non-i32 convolution bias support: 
  https://github.com/openvinotoolkit/openvino/pull/33072

Also inspect relevant CPU plugin areas handling convolution, quantization patterns, and graph transformations.

## 4) Technical Gaps You Should Analyze
Your analysis should address these blockers:

- **Pattern gap with Swish**
  - Observed sequence: `Convolution -> Multiply -> Add -> Swish -> FakeQuantize`
  - Current handled pattern: `Convolution -> Multiply -> Add -> FakeQuantize`
  - Result: Swish prevents int8 Convolution node creation

- **Activation zero-point representation**
  - A `Subtract` appears on convolution activation path
  - This zero-point should be represented through Convolution activation `QuantizationInfo` instead

Provide an implementation plan for one and both gaps.

## 5) Required Structure for Your Proposal
Please include these sections explicitly:

- Motivation and understanding of the problem
- Initial graph analysis findings
- Proposed implementation approach
- Validation plan
- Timeline with milestones

