import os

# --- 1. CONFIGURATION & FIXES ---
# Bypass the WSL GPU/CUDA XLA compilation crash by forcing CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide the annoying info/warning logs

import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
from sklearn.utils import shuffle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "aircraft")
CSV_PATH = os.path.join(DATASET_DIR, "labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yamnet-v1")

# --- 2. PREPARE DATASET PATHS ---
df = pd.read_csv(CSV_PATH)

def get_full_path(row):
    # Handle the double-nested folder structure
    folder = f"{row['class']}-{row['split']}"
    return os.path.join(DATASET_DIR, folder, folder, row['filename'])

df['file_path'] = df.apply(get_full_path, axis=1)
df['label'] = df['class'].apply(lambda x: 1 if x == 'aircraft' else 0)

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# --- 3. LOAD MODEL ---
print("Loading local YAMNet model...")
yamnet_model = tf.saved_model.load(MODEL_PATH)
infer = yamnet_model.signatures["serving_default"]

# --- 4. FEATURE EXTRACTION ---
def extract_embeddings(dataframe):
    X, y = [], []
    
    for idx, row in dataframe.iterrows():
        file_path = row['file_path']
        
        # Safety check to prevent crashing on missing files
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file skipped: {file_path}")
            continue
            
        # Load standard 1D audio at 16kHz mono
        wav_data, _ = librosa.load(file_path, sr=16000, mono=True)
        waveform = tf.convert_to_tensor(wav_data, dtype=tf.float32)
        
        # Feed the 1D waveform. We explicitly pass it as a keyword argument 
        # based on the signature requirement revealed in your error log.
        outputs = infer(waveform=waveform)
        
        # Output 1 is the 1024-D embeddings. Output 0 is scores, Output 2 is spectrogram.
        embeddings = outputs['output_1']
        
        # Average across the time frames to get one vector per file
        clip_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        X.append(clip_embedding)
        y.append(row['label'])
        
    return np.array(X), np.array(y)

print("Extracting training embeddings (CPU mode active)...")
X_train, y_train = extract_embeddings(train_df)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print("Extracting testing embeddings...")
X_test, y_test = extract_embeddings(test_df)

# --- 5. BUILD & TRAIN CLASSIFIER ---
print("Training custom classifier...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=8,
    validation_data=(X_test, y_test)
)

model.save(os.path.join(BASE_DIR, "models", "aircraft_yamnet_classifier.keras"))
print("Training complete! Model saved.")
