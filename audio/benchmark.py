import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- 0. PARSE COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description="Evaluate a custom YAMNet aircraft classifier.")
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
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "aircraft")
CSV_PATH = os.path.join(DATASET_DIR, "labels.csv")
YAMNET_PATH = os.path.join(BASE_DIR, "models", "yamnet-v1")

# Use the path passed via command line
CLASSIFIER_PATH = args.model_path
is_tflite = CLASSIFIER_PATH.lower().endswith('.tflite')

# --- 2. PREPARE TEST DATASET ---
print("Loading dataset paths...")
df = pd.read_csv(CSV_PATH)

def get_full_path(row):
    folder = f"{row['class']}-{row['split']}"
    return os.path.join(DATASET_DIR, folder, folder, row['filename'])

df['file_path'] = df.apply(get_full_path, axis=1)
df['label'] = df['class'].apply(lambda x: 1 if x == 'aircraft' else 0)

# We ONLY care about the test data here
test_df = df[df['split'] == 'test']

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
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully.")
    else:
        # Load standard Keras model
        custom_model = tf.keras.models.load_model(CLASSIFIER_PATH)
        print("Keras model loaded successfully.")
except Exception as e:
    print(f"Failed to load classifier. Please check the file path. Error: {e}")
    exit()

# --- 4. FEATURE EXTRACTION ---
def extract_test_embeddings(dataframe):
    X, y = [], []
    print(f"Extracting embeddings for {len(dataframe)} test files...")
    
    for idx, row in dataframe.iterrows():
        file_path = row['file_path']
        if not os.path.exists(file_path):
            print(f"Missing file skipped: {file_path}")
            continue
            
        wav_data, _ = librosa.load(file_path, sr=16000, mono=True)
        waveform = tf.convert_to_tensor(wav_data, dtype=tf.float32)
        outputs = infer(waveform=waveform)
        
        embeddings = outputs['output_1']
        clip_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        X.append(clip_embedding)
        y.append(row['label'])
        
    return np.array(X), np.array(y)

X_test, y_test = extract_test_embeddings(test_df)

# --- 5. RUN INFERENCE & BENCHMARK ---
print("\n" + "="*50)
print(f"BENCHMARKING RESULTS: {os.path.basename(CLASSIFIER_PATH)}")
print("="*50)

# Predict probabilities based on model type
if is_tflite:
    y_pred_probs = []
    for x in X_test:
        # Ensure the input shape matches the model's expected shape
        input_data = np.expand_dims(x, axis=0).astype(input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Extract the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_pred_probs.append(output_data[0][0])
        
    y_pred_probs = np.array(y_pred_probs)
else:
    # Standard Keras bulk prediction
    y_pred_probs = custom_model.predict(X_test, verbose=0).flatten()

# Convert probabilities to binary labels
y_pred = (y_pred_probs > 0.5).astype(int)

# Print stats
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Aircraft (1)']))

# Generate Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred Negative', 'Pred Aircraft'],
            yticklabels=['Actual Negative', 'Actual Aircraft'])
plt.title(f'Confusion Matrix - {os.path.basename(CLASSIFIER_PATH)}')
plt.xlabel('Model Prediction')
plt.ylabel('Actual Truth')
plt.tight_layout()

# Save the plot dynamically based on the model name
plot_filename = f"cm_{os.path.splitext(os.path.basename(CLASSIFIER_PATH))[0]}.png"
plot_path = os.path.join(BASE_DIR, plot_filename)
plt.savefig(plot_path)

print(f"\nDone! Confusion matrix saved to: {plot_filename}")