import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- 0. PARSE COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description="Evaluate a custom YAMNet multi-class classifier.")
parser.add_argument(
    "model_path", 
    type=str, 
    help="Path to the .keras or .tflite classifier model file to evaluate."
)
args = parser.parse_args()

# --- 1. CONFIGURATION & FIXES ---
# Keep CPU mode to avoid the WSL XLA crash
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_DIR = os.path.join(BASE_DIR, "datasets") # Pointing to the root datasets folder
YAMNET_PATH = os.path.join(BASE_DIR, "models", "yamnet-v1")

# Use the path passed via command line
CLASSIFIER_PATH = args.model_path
is_tflite = CLASSIFIER_PATH.lower().endswith('.tflite')

# --- 2. PREPARE TEST DATASET ---
print("Locating test dataset paths...")

def get_dataset_files(split='test'):
    file_paths = []
    labels = []
    
    # Iterate through class directories dynamically
    class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for class_name in class_names:
        search_pattern = os.path.join(DATASET_DIR, class_name, split, '*.wav')
        
        for file_path in glob.glob(search_pattern):
            file_paths.append(file_path)
            labels.append(class_name)
            
    # We return sorted class names to ensure the LabelEncoder maps identically to the training script
    return file_paths, labels, sorted(class_names)

test_paths, test_labels_str, class_names = get_dataset_files('test')

if len(test_paths) == 0:
    print("⚠️ Error: No test files found. Check your directory structure.")
    exit()

# Recreate the LabelEncoder based on the sorted folder names
le = LabelEncoder()
le.fit(class_names)
y_test = le.transform(test_labels_str)
num_classes = len(le.classes_)

# --- 3. LOAD MODELS ---
print("Loading base YAMNet model for feature extraction...")
yamnet_model = tf.saved_model.load(YAMNET_PATH)
infer = yamnet_model.signatures["serving_default"]

print(f"Loading custom classifier: {os.path.basename(CLASSIFIER_PATH)}...")
try:
    if is_tflite:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=CLASSIFIER_PATH)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully.")
    else:
        # Load standard Keras multi-class model
        custom_model = tf.keras.models.load_model(CLASSIFIER_PATH)
        print("Keras model loaded successfully.")
except Exception as e:
    print(f"Failed to load classifier. Please check the file path. Error: {e}")
    exit()

# --- 4. FEATURE EXTRACTION ---
def extract_test_embeddings(file_paths):
    X = []
    print(f"Extracting embeddings for {len(file_paths)} test files...")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Missing file skipped: {file_path}")
            continue
            
        wav_data, _ = librosa.load(file_path, sr=16000, mono=True)
        waveform = tf.convert_to_tensor(wav_data, dtype=tf.float32)
        outputs = infer(waveform=waveform)
        
        embeddings = outputs['output_1']
        clip_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        X.append(clip_embedding)
        
    return np.array(X)

X_test = extract_test_embeddings(test_paths)

# --- 5. RUN INFERENCE & BENCHMARK ---
print("\n" + "="*50)
print(f"BENCHMARKING RESULTS: {os.path.basename(CLASSIFIER_PATH)}")
print("="*50)

# Predict probabilities based on model type
if is_tflite:
    y_pred_probs = []
    for x in X_test:
        input_data = np.expand_dims(x, axis=0).astype(input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Append the array of class probabilities
        y_pred_probs.append(output_data[0])
        
    y_pred_probs = np.array(y_pred_probs)
else:
    # Standard Keras bulk prediction
    y_pred_probs = custom_model.predict(X_test, verbose=0)

# Multi-class resolution: grab the index of the highest probability
y_pred = np.argmax(y_pred_probs, axis=1)

# Print stats
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Generate Multi-Class Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6)) # Made slightly larger to fit more classes
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {os.path.basename(CLASSIFIER_PATH)}')
plt.xlabel('Model Prediction')
plt.ylabel('Actual Truth')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot dynamically based on the model name
plot_filename = f"cm_{os.path.splitext(os.path.basename(CLASSIFIER_PATH))[0]}.png"
plot_path = os.path.join(BASE_DIR, plot_filename)
plt.savefig(plot_path)

print(f"\nDone! Confusion matrix saved to: {plot_filename}")