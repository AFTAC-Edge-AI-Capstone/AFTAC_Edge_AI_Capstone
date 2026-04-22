# Multi-Modal Edge AI Optimization for Resource-Constrained Devices

## Project Description
This project focuses on the development of a simulated multi-modal AI system designed for deployment on a power and compute-constrained edge device. The core objective is to train, prune, and quantize three distinct machine learning models: video object detection, audio classification, and sensor health monitoring—while rigorously evaluating the trade-offs between predictive accuracy, model size, and inference latency.

### Team Members
- Ian Blackburn
- Aaron Mathews
- MJ Rupprecht
- Joshua Trapp  

### Sponsor / Mentor
- AFTAC

## Setup

### EffecientAT Audio Model
Go to directory
```bash 
cd audio
```

Create the virtual environment:
```bash
python -m venv .venv
```

Activate it:
```bash
source .venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Run benchmark on best model
```bash
python3 benchmark-tflite.py models/aircraft_mn05_classifier_float16.tflite
```

### Vision
For information on the vision model, see [EfficientNet_Lite2/README.md](./EfficientNet_Lite2/README.md)

### Sensor Health
The sensor health model is contained in [sensor_health](./sensor_health/).
This model predicts the remaining useful life of engines given data from sensors.
Its dependencies are managed using [uv](https://github.com/astral-sh/uv).
When running a file with `uv run`, the dependencies will be automatically installed.

Some aspects of the model are configured via constants in [config.py](./sensor_health/config.py).
These include the window size, the maximum RUL, and the dataset(s) to train/test on.

To train the baseline model, use `uv run train.py`.
This file trains a CNN on the data, then performs inference on testing data and displays the results for analysis.
The resulting model is written to `maintenance_model.keras`.

To prune and tune this model, use `uv run optimize.py`.
This file prunes the model from the previous step and performs quantization aware training to prepare the model for the next step.
The model is written to `maintenance_model/`.

To convert this model to a quantized tflite model, use `uv run convert.py`.
This model converts the model to tflite, quantizes it to int8, then performs predictions.
It performs inference on testing data, displaying the score and results for analysis.
The model is written to `maintenance_model_int8.tflite`.
