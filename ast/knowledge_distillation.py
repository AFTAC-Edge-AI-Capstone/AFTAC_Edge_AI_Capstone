# Based on this Python file sent by Eddie Robbins: W_AST_Distill_To_EffNet.py
# Modifications by Aaron Mathews

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
# Use the modern torch.amp for autocast and GradScaler
from torch.amp import autocast, GradScaler
import numpy as np
import librosa
import glob
import math
from typing import Tuple, Dict, List
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoConfig
from peft import get_peft_model, LoraConfig, set_peft_model_state_dict
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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


# --- Data Augmentation ---
AUGMENTATION_MULTIPLIER = 4
AUGMENTATION_PROBABILITY = 0.8

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

def load_and_configure_model(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    # There are 6 classes: drone, neg, piston, turbofan, turboprop, turboshaft
    config.num_labels = 6 # Set num_labels for classification
    config.problem_type = "single_label_classification"

    # Load the base AST model
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True # Ignore the mismatch when the pre-trained head (527 classes) is replaced by the new head again expediance here
    )    

    # --- A Minimal LoRA (Low-Rank Adaptation) Configuration ---
    lora_config = LoraConfig(
        r=8, # LoRA rank: smaller 'r' means fewer trainable parameters
        lora_alpha=16, # Scaling factor for the LoRA weights (alpha >= r is standard)
        target_modules=["query"], # Only apply LoRA to the query matrices in the attention blocks
        lora_dropout=0.2, # Dropout probability on the LoRA side
        bias="none", # Do not apply LoRA to bias terms
        modules_to_save=["classifier"], # Ensure the newly added classifier head is fully trainable (not LoRA-adapted)
    )
    # Apply the LoRA configuration to the model
    model = get_peft_model(model, lora_config)

     # --- Mixed precision (Casting requires_grad params to float32 for stability) ---
    # Convert the whole model to float16 (half precision) to save VRAM and speed up computation.
    model.half()

    # Ensure all newly added or trainable parameters (LoRA weights, MoE gates, classifier head)
    # are kept in float32 for training stability, as gradients in float16 can underflow.
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float16:
            # Handle PeftModel specific .data access for parameters
            if hasattr(param, 'data') and hasattr(param.data, 'data'):
                param.data = param.data.data.float()
            else:
                param.data = param.data.float()


    model.train()
    print("--- LoRA Configuration ---")
    model.print_trainable_parameters() # PEFT utility to show the dramatic reduction in trainable parameters
    print()

    return model

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

class DistillationLoss(nn.Module):
    """
    Combines Knowledge Distillation (KL Divergence on soft targets)
    and Standard Hard Target Loss (BCE).
    """
    def __init__(self, T: float, alpha: float, class_weights: torch.Tensor):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.loss = F.cross_entropy
        self.class_weights = class_weights


    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        #Standard Hard Target Loss (BCE)
        hard_loss = self.loss(student_logits, labels.long().squeeze(), weight=class_weights)

        # Distillation Loss (Soft Targets)
        # Binary Classification Soft Targets: [logit] -> [logit, -logit]
        teacher_logits_2d = torch.cat([teacher_logits, -teacher_logits], dim=-1)
        student_logits_2d = torch.cat([student_logits, -student_logits], dim=-1)

        # Calculate KL Divergence: log-prob of student vs prob of teacher
        kd_loss = F.kl_div(
            F.log_softmax(student_logits_2d / self.T, dim=-1),
            F.softmax(teacher_logits_2d / self.T, dim=-1),
            reduction='batchmean'
        ) * (self.T * self.T)

        #Combined Loss: alpha * KD Loss + (1-alpha) * Hard Loss
        total_loss = self.alpha * kd_loss + (1.0 - self.alpha) * hard_loss

        return total_loss, hard_loss, kd_loss

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

def save_student_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    """Saves the student model's state (CNN only)."""
    print(f"\n Saving new best CNN student checkpoint to {path}")
    model_to_save = model.module if hasattr(model, 'module') else model

    state = {
        'epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(state, path)

if __name__ == '__main__':
    # print(f"\n{'='*70}")
    # print(f"STARTING KNOWLEDGE DISTILLATION PIPELINE")
    # print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "Running on CPU")
    # print(f"Target Audio Length: {TARGET_AUDIO_SECONDS} seconds")
    # print(f"{'='*70}")

    # Feature Exatactor
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
    train_data_paths, val_data_paths, class_weights = load_data_paths_and_labels()
    class_weights = class_weights.to(device)

    train_dataset = AudioDataset(train_data_paths, feature_extractor.sampling_rate, augment=True)
    val_dataset = AudioDataset(val_data_paths, feature_extractor.sampling_rate, augment=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn_fsspec_free(b, feature_extractor), num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn_fsspec_free(b, feature_extractor), num_workers=0
    )

    #AST/MoE/LoRA Setup
    print("\n--- Teacher Model Setup (AST) ---")
    base_teacher_model = load_and_configure_model(AST_MODEL_NAME).to(device)

    if os.path.exists(TEACHER_CHECKPOINT_PATH):
        try:
            # Ignoring weights_only=False warning due to external library context
            checkpoint = torch.load(TEACHER_CHECKPOINT_PATH, map_location=device)
            # teacher_model.load_state_dict(checkpoint['model_state_dict'])
            set_peft_model_state_dict(base_teacher_model, checkpoint['lora_weights_state_dict'])
            teacher_model = base_teacher_model.merge_and_unload()

            teacher_model.eval() # Set to evaluation mode for soft target generation
            print(f"Teacher (AST) loaded successfully from {TEACHER_CHECKPOINT_PATH}.")
        except Exception as e:
            print(f"ERROR loading AST Teacher state_dict: {e}")
            exit()
    else:
        print(f"ERROR: AST Teacher checkpoint not found at {TEACHER_CHECKPOINT_PATH}. Cannot start distillation.")
        exit()

    torch.mps.empty_cache()

    #EfficientNet Setup
    print("\n--- Student Model Setup (EfficientNet) ---")
    student_model = EfficientNetSpectrogramStudent(STUDENT_MODEL_NAME, num_classes=6).to(device)

    #Loss, Optim config
    distillation_criterion = DistillationLoss(T=DISTILLATION_TEMP, alpha=DISTILLATION_ALPHA, class_weights=class_weights)


    student_optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    student_scaler = GradScaler()

    num_optimization_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    total_steps = num_optimization_steps_per_epoch * NUM_EPOCHS

    student_scheduler = OneCycleLR(
        student_optimizer, max_lr=LEARNING_RATE, total_steps=total_steps,
        pct_start=0.15, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
    )

    print(f"\n--- Distillation Configuration ---")
    print(f"KD Temp (T): {DISTILLATION_TEMP} | KD Alpha (α): {DISTILLATION_ALPHA}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"{'='*70}\n")

    # Distill train
    global_step = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        student_model.train()

        for step, batch in enumerate(train_dataloader):
            input_data = {k: v.to(device) for k, v in batch.items()}

            # --- AST FORWARD PASS (Generates Soft Targets) ---
            with torch.no_grad():
                #  Use device.type as a POSITIONAL argument for autocast
                with autocast(device.type, dtype=torch.float16):
                    # NOTE: AST expects (B, F, T) which so remove the channel dim (B, 1, F, T)
                    teacher_outputs = teacher_model(
                        input_values=input_data['input_values'].squeeze(1),
                        attention_mask=input_data['attention_mask']
                    )
                    teacher_logits = teacher_outputs.logits

            # --- EfficNet FORWARD PASS (Trained with Distillation) ---
            # Student model input (B, 1, F, T) includes the channel dimension
            # Use device.type as a POSITIONAL argument for autocast
            with autocast(device.type, dtype=torch.float16):
                student_logits = student_model(input_data['input_values'])

                # print(student_logits.shape)
                # print(teacher_logits.shape)
                # print(input_data['labels'].shape)

                total_loss, hard_loss, kd_loss = distillation_criterion(
                    student_logits,
                    teacher_logits,
                    input_data['labels']
                )

            final_loss = total_loss / GRADIENT_ACCUMULATION_STEPS
            student_scaler.scale(final_loss).backward()

            is_optimization_step = (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader)

            if is_optimization_step:
                student_scaler.unscale_(student_optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), GRAD_CLIP_NORM)

                student_scaler.step(student_optimizer)
                student_scaler.update()
                student_optimizer.zero_grad()

                if global_step < total_steps:
                    student_scheduler.step()

                global_step += 1
                current_loss = final_loss.item() * GRADIENT_ACCUMULATION_STEPS

                if global_step % 10 == 0 or (step + 1) == len(train_dataloader):
                    current_lr = student_scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Step {global_step:3d} | Total Loss: {current_loss:.4f} (KD: {kd_loss.item():.4f}, Hard Loss: {hard_loss.item():.4f}) | LR: {current_lr:.2e}")

            if (step + 1) % EMPTY_CACHE_EVERY_N_STEPS == 0:
                torch.mps.empty_cache()

        # --- Validation (Student Model only or EfficientNet) ---
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_data = {k: v.to(device) for k, v in batch.items()}

                # Use device.type as a POSITIONAL argument for autocast
                with autocast(device.type, dtype=torch.float16):
                    student_logits = student_model(input_data['input_values'])

                    # # Validation loss only uses the hard target BCE loss
                    # bce = F.binary_cross_entropy_with_logits(
                    #     student_logits, input_data['labels'], pos_weight=pos_weight
                    # )
                    # val_loss += bce.item()

                    loss = F.cross_entropy(
                        student_logits,
                        input_data['labels'].long().squeeze(),
                        weight=class_weights
                    )
                    val_loss += loss.item()

                # predictions = (torch.sigmoid(student_logits) > 0.5).long().flatten()
                # labels = input_data['labels'].long().flatten()
                # correct += (predictions == labels).sum().item()
                # total += labels.size(0)

                # torch.argmax(torch.sigmoid(outputs.logits))
                # print(outputs.logits)
                # print(labels)
                predictions = torch.argmax((torch.sigmoid(student_logits)), dim=1).long().flatten()
                labels = input_data['labels'].long().flatten()

                # print(predictions.shape)
                # print(labels.shape)
                for i in range(len(predictions)):
                  if predictions[i] == labels[i]:
                    correct += 1
                # correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total if total > 0 else 0
        val_loss = val_loss / len(val_dataloader)

        # Check for improvement and save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            save_student_checkpoint(student_model, student_optimizer, student_scheduler, epoch, best_val_loss, STUDENT_CHECKPOINT_PATH)
            print(f"NEW BEST CNN → Epoch {epoch+1:2d} | Val Acc: {val_accuracy:.4f} ({correct}/{total}) | Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1:2d} | Val Acc: {val_accuracy:.4f} ({correct}/{total}) | Val Loss: {val_loss:.4f} | Best: {best_val_acc:.4f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        print()

    print("="*70)
    print(f"DISTILLATION COMPLETE")
    print(f"Best CNN Student Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best CNN Student Validation Loss: {best_val_loss:.4f}")
    print("="*70)
