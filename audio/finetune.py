import os
import sys
import glob
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import soundfile as sf

# --- 1. CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_DIR = os.path.join(BASE_DIR, "datasets") 

# --- CONNECT THE SUBMODULE PATH ---
EFFICIENTAT_DIR = os.path.join(BASE_DIR, "external", "EfficientAT")
if EFFICIENTAT_DIR not in sys.path:
    sys.path.append(EFFICIENTAT_DIR)

SAMPLE_RATE = 32000
HOP_LENGTH = int(SAMPLE_RATE * 0.01) # 10 ms hop size

# --- 2. PREPARE DATASET PATHS ---
def get_dataset_files(split='train'):
    file_paths = []
    labels = []
    
    class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for class_name in class_names:
        search_pattern = os.path.join(DATASET_DIR, class_name, split, '*.wav')
        for file_path in glob.glob(search_pattern):
            if os.path.exists(file_path):
                file_paths.append(file_path)
                labels.append(class_name)
            else:
                print(f"Missing file skipped: {file_path}")
                
    return file_paths, labels


# --- 3. OPTIMIZED DATASET CLASS ---
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        # Cache for resamplers to avoid recalculating filter weights per file
        self.resamplers = {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 1. Read the raw audio using soundfile
        audio_data, sr = sf.read(file_path)
        
        # 2. Convert to PyTorch tensor
        waveform = torch.from_numpy(audio_data).float()
        
        # 3. Fix dimensions to (channels, time)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) 
        else:
            waveform = waveform.t()
        
        # 4. Resample efficiently using cached resamplers
        if sr != SAMPLE_RATE:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = self.resamplers[sr](waveform)
            
        # 5. Convert to Mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Return RAW waveform; Spectrogram math is moved to the GPU loop below
        return waveform, torch.tensor(label, dtype=torch.long)


# --- MAIN EXECUTION BLOCK ---
# Required for PyTorch multiprocessing (num_workers > 0)
if __name__ == '__main__':
    print("Locating audio files...")
    train_paths, train_labels_str = get_dataset_files('train')
    test_paths, test_labels_str = get_dataset_files('test')

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels_str)
    test_labels = le.transform(test_labels_str) 
    num_classes = len(le.classes_)

    train_paths, train_labels = shuffle(train_paths, train_labels, random_state=42)
    print(f"Found {len(train_paths)} training files across {num_classes} classes: {le.classes_}")

    # Create Datasets
    train_dataset = AudioDataset(train_paths, train_labels)
    test_dataset = AudioDataset(test_paths, test_labels)

    # --- PERFORMANCE FIX: Upgraded DataLoaders ---
    # Note: Assumes your audio files are all the exact same length. 
    # If not, you'll need a custom collate_fn to pad them to the same size.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,       # Increased batch size
        shuffle=True, 
        num_workers=4,       # Parallel loading
        pin_memory=True      # Faster CPU-to-GPU transfer
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # --- PHASE 2: ARCHITECTURE MODIFICATION ---
    print("Loading mn05_as model via EfficientAT submodule...")
    original_cwd = os.getcwd()

    try:
        os.chdir(EFFICIENTAT_DIR)
        from models.mn.model import get_model as get_mn
        
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        model = get_mn(width_mult=0.5, pretrained_name="mn05_as")
        sys.stdout = original_stdout
    finally:
        os.chdir(original_cwd)

    # Freeze backbone & swap head
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # --- PERFORMANCE FIX: GPU Transforms ---
    # Define transforms once and push them to the GPU
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=128,
        hop_length=HOP_LENGTH,
        n_fft=1024
    ).to(device)
    
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    # --- TRAIN MULTI-CLASS CLASSIFIER ---
    print("Training custom multi-class classifier...")

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 25

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # 1. Move raw audio directly to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 2. Compute heavy Spectrogram math ON THE GPU
            with torch.no_grad():
                inputs = mel_transform(inputs)
                inputs = amplitude_to_db(inputs)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0] 
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                # 1. Move raw audio directly to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 2. Compute Spectrogram ON THE GPU
                inputs = mel_transform(inputs)
                inputs = amplitude_to_db(inputs)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "aircraft_mn05_classifier.pth"))
    print("Training complete! Model state_dict saved.")