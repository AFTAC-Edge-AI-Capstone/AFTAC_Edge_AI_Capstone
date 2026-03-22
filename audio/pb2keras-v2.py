import os
import glob
import numpy as np
import tensorflow as tf
import librosa
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION & FIXES ---
# Bypass the WSL GPU/CUDA XLA compilation crash by forcing CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide the annoying info/warning logs

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_DIR = os.path.join(BASE_DIR, "datasets") # Pointing to the root datasets folder
MODEL_PATH = os.path.join(BASE_DIR, "models", "yamnet-v1")

# --- 2. PREPARE DATASET PATHS ---
def get_dataset_files(split='train'):
    file_paths = []
    labels = []
    
    # Iterate through class directories (drone, neg, piston, etc.)
    class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for class_name in class_names:
        # Construct path: datasets/<class_name>/<split>/*.wav
        search_pattern = os.path.join(DATASET_DIR, class_name, split, '*.wav')
        
        for file_path in glob.glob(search_pattern):
            file_paths.append(file_path)
            labels.append(class_name)
            
    return file_paths, labels

print("Locating audio files...")
train_paths, train_labels_str = get_dataset_files('train')
test_paths, test_labels_str = get_dataset_files('test')

# Encode string labels into integers for multi-class classification
le = LabelEncoder()
train_labels = le.fit_transform(train_labels_str)
test_labels = le.transform(test_labels_str) # Use the same encoder for test set
num_classes = len(le.classes_)

print(f"Found {len(train_paths)} training files across {num_classes} classes: {le.classes_}")

# --- 3. LOAD MODEL ---
print("Loading local YAMNet model...")
yamnet_model = tf.saved_model.load(MODEL_PATH)
infer = yamnet_model.signatures["serving_default"]

# --- 4. FEATURE EXTRACTION ---
def extract_embeddings(file_paths, labels):
    X, y = [], []
    
    for file_path, label in zip(file_paths, labels):
        # Safety check to prevent crashing on missing files
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file skipped: {file_path}")
            continue
            
        # Load standard 1D audio at 16kHz mono
        wav_data, _ = librosa.load(file_path, sr=16000, mono=True)
        waveform = tf.convert_to_tensor(wav_data, dtype=tf.float32)
        
        # Feed the 1D waveform
        outputs = infer(waveform=waveform)
        
        # Output 1 is the 1024-D embeddings. Output 0 is scores, Output 2 is spectrogram.
        embeddings = outputs['output_1']
        
        # Average across the time frames to get one vector per file
        clip_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        X.append(clip_embedding)
        y.append(label)
        
    return np.array(X), np.array(y)

print("Extracting training embeddings (CPU mode active)...")
X_train, y_train = extract_embeddings(train_paths, train_labels)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print("Extracting testing embeddings...")
X_test, y_test = extract_embeddings(test_paths, test_labels)

# --- 5. BUILD & TRAIN MULTI-CLASS CLASSIFIER ---
print("Training custom multi-class classifier...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    # Changed to Softmax for multi-class and dynamically sized to the number of folders
    tf.keras.layers.Dense(num_classes, activation='softmax') 
])

# Changed loss to sparse_categorical_crossentropy to handle integer labels natively
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=8,
    validation_data=(X_test, y_test)
)

model.save(os.path.join(BASE_DIR, "models", "aircraft_yamnet_classifier.keras"))
print("Training complete! Model saved.")