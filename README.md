# Multi-Modal Edge AI Optimization for Resource-Constrained Devices

## Team Members
- Ian Blackburn
- Aaron Mathews
- MJ Rupprecht
- Joshua Trapp  

## Sponsor / Mentor
- AFTAC

## Project Description
This project focuses on the development of a simulated multi-modal AI system designed for deployment on a power and compute-constrained edge device. The core objective is to train, prune, and quantize three distinct machine learning models: video object detection, audio classification, and sensor health monitoring—while rigorously evaluating the trade-offs between predictive accuracy, model size, and inference latency.

## Setup
### EffecientAT audio model pipeline
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