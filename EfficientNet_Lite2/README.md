# EfficientNet-Lite2 Edge Deployment Workflow

This directory documents the EfficientNet-Lite2 workflow used to:
- evaluate the original FP32 ONNX model on ImageNet-1k 2012 validation data
- locally quantize that model into an INT8/QDQ ONNX model
- measure how quantization changed accuracy, size, and throughput
- generate STM32N6 deployment outputs and record target-side benchmark results

The intended readers are:
- defense engineers evaluating edge deployment tradeoffs
- professors reviewing the technical path and reported results
- students who need to recreate the workflow and understand what each stage is doing

## Why This Matters

For defense-oriented edge AI systems, model quality by itself is not enough.

A model also has to:
- preserve accuracy after quantization
- fit real embedded memory limits
- produce a deployment artifact that can actually run on constrained target hardware

The important result in this workflow is:

`accuracy stayed essentially the same, while the model became small enough to support STM32N6-class deployment.`

## Main Results

### 1. Host-Side Validation And Compression

| Metric | Pre-Quantized ONNX | Locally Quantized ONNX | Change |
|---|---:|---:|---:|
| Model format | FP32 ONNX | INT8/QDQ ONNX | Quantized |
| Top-1 Accuracy | 76.09% | 76.01% | -0.08 pp |
| Top-5 Accuracy | 92.93% | 92.96% | +0.03 pp |
| Throughput Speedup | 1.00x | 2.86x | +185.88% |
| ONNX File Size | 24,353,308 B | 6,880,048 B | -71.75% |

### 2. Target Edge Deployment Metrics

| Metric | Quantized Model |
|---|---:|
| Deployable on STM32N6570-DK | Yes |
| Board latency / inference | 49.08 ms |
| Board throughput | 20.37 inf/s |
| Weights footprint | 6.892 MB |
| Activations footprint | 2.305 MB |
| Total runtime memory | 9.197 MB |
| Quantized weight blob | 7,226,801 B |

The pre-quantized FP32 ONNX baseline was not benchmarked on STM32N6570-DK because it was not deployable within the target memory/runtime footprint.

## Local Quantization Methodology

The local quantization step in this directory was not performed by ST Edge AI itself.

It was performed locally with ONNX Runtime static post-training quantization.

At a high level, the local quantization path used:
- ONNX Runtime `quantize_static()`
- `QuantFormat.QDQ`
- `QuantType.QInt8` for activations
- `QuantType.QInt8` for weights
- `CalibrationMethod.MinMax`
- `per_channel=True`
- a representative calibration dataset stored as `calibration_data.npz`

What that means in practice is:
- the original FP32 ONNX graph was rewritten into a QDQ graph
- `QuantizeLinear` and `DequantizeLinear` operators were inserted around internal tensors
- activation ranges were estimated offline from representative ImageNet-1k 2012 validation samples
- activations and weights were stored internally using signed INT8 quantization
- convolution-heavy weights used per-channel quantization to better preserve accuracy
- the external model interface still stayed FLOAT input / FLOAT output

So the model became quantized internally for edge deployment, while still looking like a standard float model from the outside.

## Repo Layout

This repo is intended to keep the model files, scripts, and result artifacts needed to understand and recreate the workflow.

ImageNet-1k 2012 validation data was used for:
- local evaluation
- representative calibration

The dataset itself is not redistributed through the repo.

The main benchmark artifact intended for the repo is:
- `results/developer_cloud_benchmark_stm32n6570dk.csv`

## Understanding The Quantization Settings

If you are trying to understand exactly what quantization was performed, these are the settings that matter most:

### 1. `QuantFormat.QDQ`

This means the ONNX model was converted into a quantization-aware graph representation by inserting:
- `QuantizeLinear`
- `DequantizeLinear`

around internal tensors.

### 2. `QuantType.QInt8` For Activations

This means intermediate activations inside the model were represented with signed INT8 quantization.

### 3. `QuantType.QInt8` For Weights

This means the learned weights were also represented with signed INT8 quantization.

### 4. `CalibrationMethod.MinMax`

This means activation ranges were estimated offline by observing representative data and recording minimum / maximum values.

Those ranges were then used to compute the quantization parameters written into the ONNX graph.

### 5. `per_channel=True`

This means weights were quantized per output channel instead of forcing one scale across an entire tensor.

That usually preserves accuracy better for convolution-heavy models like EfficientNet-Lite2.
