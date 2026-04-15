import os
import sys
import glob
import torch
import torchaudio
import torch.nn as nn
import soundfile as sf
import numpy as np
import tensorflow as tf
import subprocess

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_PATH = os.path.join(BASE_DIR, "models", "aircraft_mn05_classifier.pth")

# TFLite requires STATIC input shapes. You must define the exact length
# of the audio clips you intend to feed into the model during inference.
AUDIO_LENGTH_SECONDS = 5.0  # <--- Change this to match your audio files!
SAMPLE_RATE = 32000
HOP_LENGTH = int(SAMPLE_RATE * 0.01)
N_MELS = 128
TARGET_SAMPLES = int(AUDIO_LENGTH_SECONDS * SAMPLE_RATE)
FRAMES = int((TARGET_SAMPLES / HOP_LENGTH)) + 1 

# Paths for intermediate and final models
ONNX_PATH = os.path.join(BASE_DIR, "models", "model.onnx")
TF_SAVED_MODEL_DIR = os.path.join(BASE_DIR, "models", "tf_saved_model")
TFLITE_PATH = os.path.join(BASE_DIR, "models", "aircraft_mn05_classifier_int8.tflite")

# --- CONNECT THE SUBMODULE PATH ---
EFFICIENTAT_DIR = os.path.join(BASE_DIR, "external", "EfficientAT")
if EFFICIENTAT_DIR not in sys.path:
    sys.path.append(EFFICIENTAT_DIR)

# --- 2. LOAD PYTORCH MODEL ---
print("Reconstructing PyTorch model...")
class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
num_classes = len(class_names)

original_cwd = os.getcwd()
try:
    os.chdir(EFFICIENTAT_DIR)
    from models.mn.model import get_model as get_mn
    
    # Suppress verbose print statements from get_mn
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    model = get_mn(width_mult=0.5, pretrained_name=None) # No need to download pretrained weights here
    sys.stdout = original_stdout
finally:
    os.chdir(original_cwd)

# Rebuild the head and load the state dictionary
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

print(f"Loading trained weights from {MODEL_PATH}...")
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval() # CRITICAL: Set to evaluation mode before export!

# --- 3. EXPORT TO ONNX ---
print(f"Exporting to ONNX format (Input shape: [1, 1, {N_MELS}, {FRAMES}])...")
# Create a dummy spectrogram tensor matching your inference dimensions
dummy_input = torch.randn(1, 1, N_MELS, FRAMES)

torch.onnx.export(
    model, 
    dummy_input, 
    ONNX_PATH,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# --- 4. CONVERT ONNX TO TENSORFLOW SAVEDMODEL ---
print("Converting ONNX to TensorFlow SavedModel using onnx2tf...")
# We use onnx2tf via subprocess as it handles complex graph operations beautifully
subprocess.run([
    "onnx2tf",
    "-i", ONNX_PATH,
    "-o", TF_SAVED_MODEL_DIR
], check=True)

# --- 5. INT8 QUANTIZATION AND TFLITE CONVERSION ---
print("Preparing Representative Dataset for INT8 Calibration...")

# Set up the exact same torchaudio transforms used in training
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=1024
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def representative_dataset_gen():
    """Yields real spectrograms to calibrate the INT8 activation ranges."""
    # Grab a handful of training files (approx 100 is standard for calibration)
    search_pattern = os.path.join(DATASET_DIR, '**', 'train', '*.wav')
    audio_files = glob.glob(search_pattern, recursive=True)[:100]
    
    for file_path in audio_files:
        audio_data, sr = sf.read(file_path)
        waveform = torch.from_numpy(audio_data).float()
        
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: waveform = waveform.t()
        
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # TFLite needs STRICT static padding/truncation to match FRAMES
        if waveform.shape[1] > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]
        else:
            pad_amount = TARGET_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # Compute Spectrogram exactly as in training
        spec = mel_transform(waveform)
        spec = amplitude_to_db(spec)
        
        # Add batch dimension and yield as float32 numpy array
        spec = spec.unsqueeze(0).numpy().astype(np.float32)
        
        # --- NEW LINE: Transpose from NCHW to NHWC ---
        spec = np.transpose(spec, (0, 2, 3, 1))
        
        yield [spec]

print("Converting to INT8 TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_DIR)

# Optimize for size/quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Provide the calibration dataset
converter.representative_dataset = representative_dataset_gen

# Restrict operations to strictly INT8 (forces fully quantized model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set input and output tensors to INT8 format for pure integer inference
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Success! Fully quantized INT8 TFLite model saved to: {TFLITE_PATH}")