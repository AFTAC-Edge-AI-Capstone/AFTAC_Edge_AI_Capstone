# Based on multiple python files sent by Eddie including these: W_AST_TrainingWcomments.py and W_AST_Distill_To_EffNet.py
# Modified by Aaron Mathews
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import soundfile as sf
import librosa
import glob
import math
from typing import Tuple, Dict, List
# Import Hugging Face components for model and feature extraction
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoConfig
# Import PEFT components for LoRA
from peft import get_peft_model, LoraConfig
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# Use the modern torch.amp for autocast and GradScaler
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import torch_pruning as tp


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

AUDIO_EXTENSIONS = ['*.wav']

# Configuration and data locations
DATA_DIRS = {
    "Neg": "modified_datasets/neg", # Directory containing negative (Noise/Background) samples (Label 0.0)
    "Drone": "modified_datasets/drone",
    "Piston": "modified_datasets/piston",
    "Turbofan": "modified_datasets/turbofan",
    "Turboprop": "modified_datasets/turboprop",

    "Turboshaft": "modified_datasets/turboshaft"
}

LABEL_MAP = {
    "Neg": 0,
    "Drone": 1,
    "Piston": 2,
    "Turbofan": 3,
    "Turboprop": 4,
    "Turboshaft": 5
}

class AudioAugmenter:
    """Augmentation with safety checks."""
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def augment(self, waveform):
        try:
            aug_waveform = waveform.copy()
            # Apply time stretch, pitch shift, noise, and gain with AUGMENTATION_PROBABILITY
            if np.random.random() < AUGMENTATION_PROBABILITY:
                rate = np.random.uniform(0.95, 1.05)
                # Ensure time_stretch input is float
                aug_waveform = librosa.effects.time_stretch(aug_waveform.astype(float), rate=rate, n_fft=2048)
            if np.random.random() < AUGMENTATION_PROBABILITY:
                n_steps = np.random.uniform(-1.5, 1.5)
                aug_waveform = librosa.effects.pitch_shift(aug_waveform.astype(float), sr=self.sr, n_steps=n_steps)
            if np.random.random() < AUGMENTATION_PROBABILITY:
                noise_level = np.random.uniform(0.001, 0.003)
                noise = np.random.normal(0, noise_level, aug_waveform.shape)
                aug_waveform = aug_waveform + noise
            if np.random.random() < AUGMENTATION_PROBABILITY:
                gain = np.random.uniform(0.85, 1.15)
                aug_waveform = aug_waveform * gain

            # Re-normalize and clip
            aug_waveform = np.clip(aug_waveform, -1.0, 1.0)
            if np.isnan(aug_waveform).any() or np.isinf(aug_waveform).any():
                return waveform
            return aug_waveform
        except Exception:
            return waveform

class AudioDataset(Dataset):
    """Dataset with augmentation and corrected 10.24s input length."""
    def __init__(self, data: list, sampling_rate: int, augment: bool = False):
        if augment:
            expanded_data = []
            for file_path, label in data:
                for _ in range(AUGMENTATION_MULTIPLIER):
                    expanded_data.append((file_path, label))
            self.data = expanded_data
        else:
            self.data = data
        self.sr = sampling_rate
        self.augment = augment
        self.augmenter = AudioAugmenter(sampling_rate) if augment else None

        # Use 10.24 seconds for AST compatibility!!!
        self.target_len = math.ceil(self.sr * TARGET_AUDIO_SECONDS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        file_path, label = self.data[idx]

        try:
            # Load audio
            waveform, sr = librosa.load(file_path, sr=self.sr, mono=True)

            # --- PADDING/CLIPPING TO 10.24 SECONDS ---
            if waveform.size < self.target_len:
                # Pad short audio
                padding = self.target_len - waveform.size
                waveform = np.pad(waveform, (0, padding), 'constant')
            else:
                # Clip longer audio to the exact target length (10.24s)
                waveform = waveform[:self.target_len]

        except Exception as e:
            # Fallback for corrupted/unreadable files
            print(f"Warning: Failed to load {file_path}. Using zero array. Error: {e}")
            waveform = np.zeros(self.target_len)

        if self.augment and self.augmenter:
            waveform = self.augmenter.augment(waveform)

        return {'waveform': waveform, 'labels': label}

def collate_fn_fsspec_free(batch: list, feature_extractor: AutoFeatureExtractor) -> Dict:
    """Collate function to process waveforms into spectrograms (input_values)."""
    waveforms = [item['waveform'] for item in batch]
    labels = [item['labels'] for item in batch]

    inputs = feature_extractor(
        waveforms,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    # Generate attention mask for AST (includes CLS token)
    if 'attention_mask' not in inputs:
        # AST attention mask is always one position longer than the spectrogram patches
        num_patches = inputs['input_values'].shape[-1]
        sequence_length = num_patches + 1
        inputs['attention_mask'] = torch.ones(
            (inputs['input_values'].shape[0], sequence_length),
            dtype=torch.long
        )

    inputs['labels'] = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Add a channel dimension for the CNN student: (B, F, T) -> (B, 1, F, T)
    inputs['input_values'] = inputs['input_values'].unsqueeze(1)

    return inputs

# --- Training Hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.1
GRAD_CLIP_NORM = 10.0
EMPTY_CACHE_EVERY_N_STEPS = 4
EARLY_STOPPING_PATIENCE = 10

# --- AST Teacher Configuration (Must match the pre-training script)
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
TEACHER_CHECKPOINT_PATH = "best_teacher_checkpoint.pt" # PATH TO THE FINE-TUNED AST TEACHER WEIGHTS!!!!!!<---------

# --- CNN Student Configuration
STUDENT_MODEL_NAME = 'efficientnet_b0'
DISTILLATION_TEMP = 2.0    # Temperature T for softening Teacher logits
DISTILLATION_ALPHA = 0.7 # Weight for KD loss (0.7 * KD Loss + 0.3 * BCE Loss)
TARGET_AUDIO_SECONDS = 10.24 # 10.24 seconds like AST
STUDENT_CHECKPOINT_PATH = "best_student_checkpoint.pt" # PATH TO SAVE THE TRAINED CNN STUDENT!!!!!!!!!<---------

def load_data_paths_and_labels():
    """Load file paths, assign labels, split into train/val, and calculate class weights."""
    train_files = []
    train_labels = []

    test_files = []
    test_labels = []

    print(f"\n--- Loading Dataset ---")

    for dataset_name, dataset_path in DATA_DIRS.items():
        label = LABEL_MAP[dataset_name]

        current_train_files = []
        current_test_files = []
        for ext in AUDIO_EXTENSIONS:
            # Use glob to find all audio files recursively in the directory
            current_train_files.extend(glob.glob(os.path.join(dataset_path + '/train', ext)))
            current_test_files.extend(glob.glob(os.path.join(dataset_path + '/test', ext)))

        train_files.extend(current_train_files)
        test_files.extend(current_test_files)

        train_labels.extend([label] * len(current_train_files))
        test_labels.extend([label] * len(current_test_files))

    train_data = list(zip(train_files, train_labels))
    val_data = list(zip(test_files, test_labels))

    # --- Calculate class weights ---
    # Needed for Categorical Cross-Entropy loss to handle class imbalance
    train_labels = [label for _, label in train_data]


    neg_count = len([e for e in train_labels if e == 0.0])
    drone_count = len([e for e in train_labels if e == 1.0])
    piston_count = len([e for e in train_labels if e == 2.0])
    turbofan_count = len([e for e in train_labels if e == 3.0])
    turboprop_count = len([e for e in train_labels if e == 4.0])
    turboshaft_count = len([e for e in train_labels if e == 5.0])

    # Inverse frequency weighting: weight_i = total_samples / (6 * class_count_i)

    total = len(train_labels)
    neg_weight = (total / (6.0 * neg_count)) if neg_count > 0 else 1.0
    drone_weight = (total / (6.0 * drone_count)) if drone_count > 0 else 1.0
    piston_weight = (total / (6.0 * piston_count)) if piston_count > 0 else 1.0
    turbofan_weight = (total / (6.0 * turbofan_count)) if turbofan_count > 0 else 1.0
    turboprop_weight = (total / (6.0 * turboprop_count)) if turboprop_count > 0 else 1.0
    turboshaft_weight = (total / (6.0 * turboshaft_count)) if turboshaft_count > 0 else 1.0

    print(f"\n--- Class Balance ---")
    print(f"Negative Count: {int(neg_count)} samples")
    print(f"Drone Count: {int(drone_count)} samples")
    print(f"Piston Count: {int(piston_count)} samples")
    print(f"Turbofan Count: {int(turbofan_count)} samples")
    print(f"Turboprop Count: {int(turboprop_count)} samples")
    print(f"Turboshaft Count: {int(turboshaft_count)} samples")
    print(f"{'='*70}\n")

    return train_data, val_data, torch.tensor([neg_weight, drone_weight, piston_weight, turbofan_weight, turboprop_weight, turboshaft_weight], dtype=torch.float32)

class EfficientNetSpectrogramStudent(nn.Module):
    """
    EfficientNet model adapted for single-channel spectrogram input (B, 1, F, T)
    using torchvision models.
    """
    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 6):
        super().__init__()

        if model_name == 'efficientnet_b0':
            self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {model_name} not supported via torchvision in this script.")

        # 1. Adapt the first convolutional layer (conv_stem) for 1 channel
        original_conv = self.efficientnet.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

        # Initialize the new 1-channel weights by averaging the 3-channel weights
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        self.efficientnet.features[0][0] = new_conv

        # 2. Adapt the classifier head
        num_in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_in_features, num_classes)

    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.efficientnet(input_values)

def validate(model, val_dataloader, compressed=False):
    model.eval()
    correct = 0
    total = 0

    predictions = []
    labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_data = {k: v.to(device) for k, v in batch.items()}

            # Use device.type as a POSITIONAL argument for autocast
            # with autocast(device.type, dtype=torch.float16):
            #     student_logits = model(input_data['input_values'])

            student_logits = model(input_data['input_values'])

            curr_predictions = torch.argmax((torch.sigmoid(student_logits)), dim=1).long().flatten()
            curr_labels = input_data['labels'].long().flatten()

            predictions.append(curr_predictions.detach().cpu())
            labels.append(curr_labels.detach().cpu())

            for i in range(len(curr_predictions)):
                if curr_predictions[i] == curr_labels[i]:
                    correct += 1
            total += curr_labels.size(0)

    val_accuracy = correct / total if total > 0 else 0

    # print(labels)
    # print(torch.cat(labels).numpy())
    # print(predictions)
    print(total)
    print(f"Accuracy: {val_accuracy}")
    c = confusion_matrix(torch.cat(labels).numpy(), torch.cat(predictions).numpy())
    data_labels = ["Negative", "Drone", "Piston", "Turbofan", "Turboprop", "Turboshaft"]
    sns.heatmap(c, annot=True, fmt='d', xticklabels=data_labels, yticklabels=data_labels)
    filename = ""
    if compressed:
        filename = "compressed_results.png"
    else:
        filename = "uncompressed_results.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(c)

if __name__ == '__main__':

    student_model_uncompressed = EfficientNetSpectrogramStudent(STUDENT_MODEL_NAME, num_classes=6).to(device)

    student_checkpoint_uncompressed = torch.load('best_student_checkpoint.pt')
    student_model_uncompressed.load_state_dict(student_checkpoint_uncompressed['model_state_dict'])

    # print(student_checkpoint_uncompressed[''])

    train_data_paths, val_data_paths, class_weights = load_data_paths_and_labels()
    
    # Feature Exatactor
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)

    val_dataset = AudioDataset(val_data_paths, feature_extractor.sampling_rate, augment=False)


    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn_fsspec_free(b, feature_extractor), num_workers=0
    )
    
    validate(student_model_uncompressed, val_dataloader)

    # Validate after pruning
    compressed_model = EfficientNetSpectrogramStudent('efficientnet_b0', num_classes=6).to(device)
    compressed_model.load_state_dict(student_checkpoint_uncompressed['model_state_dict'])
    example_inputs = torch.randn(1, 1, 224, 224).to(device)

    importance =  tp.importance.GroupMagnitudeImportance(p=2)

    pruner = tp.pruner.BasePruner(
        compressed_model,
        example_inputs,
        importance=importance,
        pruning_ratio=0.05
    )

    initial_macs, initial_params = tp.utils.count_ops_and_params(compressed_model, example_inputs)
    print(f"Initial MACs: {initial_macs}, Initial Params: {initial_params}")
    pruner.step()
    final_macs, final_params = tp.utils.count_ops_and_params(compressed_model, example_inputs)
    print(f"Final MACs: {final_macs}, Final Params: {final_params}")

    validate(compressed_model, val_dataloader)
    # Can use scikit learn
    # Accuracy



    # Confusion Matrix maybe