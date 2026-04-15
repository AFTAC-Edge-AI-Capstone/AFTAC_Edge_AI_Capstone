import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- 0. PARSE COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description="Evaluate a custom EfficientAT INT8 TFLite classifier.")
parser.add_argument(
    "model_path", 
    type=str, 
    help="Path to the .tflite classifier model file to evaluate."
)
args = parser.parse_args()

# --- 1. CONFIGURATION & FIXES ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_DIR = os.path.join(BASE_DIR, "datasets") 
CLASSIFIER_PATH = args.model_path

# Audio processing constants (Must match your training/export script exactly)
AUDIO_LENGTH_SECONDS = 5.0  
SAMPLE_RATE = 32000
HOP_LENGTH = int(SAMPLE_RATE * 0.01)
N_MELS = 128
TARGET_SAMPLES = int(AUDIO_LENGTH_SECONDS * SAMPLE_RATE)

# --- 2. PREPARE TEST DATASET ---
print("Locating test dataset paths...")

def get_dataset_files(split='test'):
    file_paths = []
    labels = []
    
    class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for class_name in class_names:
        search_pattern = os.path.join(DATASET_DIR, class_name, split, '*.wav')
        
        for file_path in glob.glob(search_pattern):
            file_paths.append(file_path)
            labels.append(class_name)
            
    return file_paths, labels, sorted(class_names)

test_paths, test_labels_str, class_names = get_dataset_files('test')

if len(test_paths) == 0:
    print("⚠️ Error: No test files found. Check your directory structure.")
    exit()

le = LabelEncoder()
le.fit(class_names)
y_test = le.transform(test_labels_str)

# --- 3. LOAD TFLITE MODEL ---
print(f"Loading custom INT8 classifier: {os.path.basename(CLASSIFIER_PATH)}...")
try:
    interpreter = tf.lite.Interpreter(model_path=CLASSIFIER_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    
    # --- NEW LOGIC: Dynamically find the correct output tensor ---
    all_outputs = interpreter.get_output_details()
    output_details = None
    
    for out in all_outputs:
        # Look for the tensor where the last dimension matches our 5 classes
        if len(out['shape']) > 0 and out['shape'][-1] == len(class_names):
            output_details = out
            break
            
    if output_details is None:
        print(f"\n⚠️ FATAL ERROR: Could not find an output tensor matching {len(class_names)} classes.")
        print("Your TFLite model is outputting the following shapes:")
        for o in all_outputs:
            print(f" - {o['name']}: {o['shape']}")
        print("\nThis means the ONNX->TFLite conversion exported intermediate features instead of the final logits. You will need to check your export script.")
        exit()
    # -------------------------------------------------------------
    
    # Extract quantization parameters
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Failed to load classifier. Please check the file path. Error: {e}")
    exit()

# --- 4. PREPROCESSING & FEATURE EXTRACTION ---
# Setup torchaudio transforms exactly as calibrated
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=1024
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def preprocess_audio(file_path):
    """Loads, resamples, pads/truncates, and converts audio to a Mel-spectrogram."""
    audio_data, sr = sf.read(file_path)
    waveform = torch.from_numpy(audio_data).float()
    
    if waveform.ndim == 1: 
        waveform = waveform.unsqueeze(0)
    else: 
        waveform = waveform.t()
        
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
        
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if waveform.shape[1] > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]
    else:
        pad_amount = TARGET_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        
    spec = mel_transform(waveform)
    spec = amplitude_to_db(spec)
    
    # Add batch dim, convert to float32 numpy, and transpose to NHWC
    spec = spec.unsqueeze(0).numpy().astype(np.float32)
    spec = np.transpose(spec, (0, 2, 3, 1))
    
    return spec

# --- 5. RUN INFERENCE & BENCHMARK ---
print("\n" + "="*50)
print(f"BENCHMARKING RESULTS: {os.path.basename(CLASSIFIER_PATH)}")
print("="*50)

y_pred_probs = []

print(f"Running inference on {len(test_paths)} test files...")
for file_path in test_paths:
    if not os.path.exists(file_path):
        continue
        
    # 1. Preprocess to float32 spectrogram
    input_data_fp32 = preprocess_audio(file_path)
    
    # 2. Quantize to INT8 based on the model's input details
    if input_scale > 0:
        input_data_quantized = (input_data_fp32 / input_scale) + input_zero_point
        input_data = np.clip(np.round(input_data_quantized), -128, 127).astype(input_details['dtype'])
    else:
        input_data = input_data_fp32.astype(input_details['dtype'])
    
    # 3. Invoke interpreter
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    # 4. Extract and Dequantize output
    output_data = interpreter.get_tensor(output_details['index'])[0]
    
    if output_scale > 0:
        output_data_fp32 = (output_data.astype(np.float32) - output_zero_point) * output_scale
    else:
        output_data_fp32 = output_data
        
    y_pred_probs.append(output_data_fp32)

y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print stats
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Generate Multi-Class Confusion Matrix Plot


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {os.path.basename(CLASSIFIER_PATH)}')
plt.xlabel('Model Prediction')
plt.ylabel('Actual Truth')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot dynamically
plot_filename = f"cm_{os.path.splitext(os.path.basename(CLASSIFIER_PATH))[0]}.png"
plot_path = os.path.join(BASE_DIR, plot_filename)
plt.savefig(plot_path)

print(f"\nDone! Confusion matrix saved to: {plot_filename}")