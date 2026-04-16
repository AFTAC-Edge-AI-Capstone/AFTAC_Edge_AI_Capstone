# Based on this python file sent by Eddie: W_AST_TrainingWcomments.py
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

# Initialize constants

AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

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

AUDIO_EXTENSIONS = ['*.wav'] # File extensions to search for
CHECKPOINT_PATH = "best_aircraft_moe_model1.pt" # Path for saving the best model checkpoint

# Training hyperparameters (optimized for stability...so far)
BATCH_SIZE = 8 # Actual batch size passed to the DataLoader
GRADIENT_ACCUMULATION_STEPS = 8 # Steps to accumulate gradients before optimization step (Effective Batch Size = 4 * 8 = 32)
LEARNING_RATE = 5e-5 # Maximum learning rate for the OneCycleLR scheduler
NUM_EPOCHS = 10 # Total number of training passes over the dataset
WEIGHT_DECAY = 0.1 # L2 regularization term for the optimizer
GRAD_CLIP_NORM = 10.0 # Maximum gradient norm for mixed precision

# Data Augmentation ( small dataset big model you may not have this problem)
AUGMENTATION_MULTIPLIER = 1 # How many times to duplicate each original training sample through augmentation
AUGMENTATION_PROBABILITY = 0.8 # Probability of applying any single augmentation type

# Memory and stability
EMPTY_CACHE_EVERY_N_STEPS = 4 # Calls torch.cuda.empty_cache() periodically to release unused memory
EARLY_STOPPING_PATIENCE = 10 # Number of epochs to wait for validation loss

device = None
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def load_data_paths_and_labels():
    """Load file paths, assign labels, split into train/val, and calculate class weights."""
    train_files = []
    train_labels = []

    test_files = []
    test_labels = []

    print(f"\n--- Dataset Loading ---")

    for dataset_name, dataset_path in DATA_DIRS.items():


        label = LABEL_MAP[dataset_name]
        print(f"Label: {label}")

        current_train_files = []
        current_test_files = []
        for ext in AUDIO_EXTENSIONS:
            # Use glob to find all audio files recursively in the directory
            current_train_files.extend(glob.glob(os.path.join(dataset_path + '/train', ext)))
            current_test_files.extend(glob.glob(os.path.join(dataset_path + '/test', ext)))
            print(len(current_train_files))
            print(len(current_test_files))
            print("\n")

        # all_files.extend(current_dir_files)

        train_files.extend(current_train_files)
        test_files.extend(current_test_files)

        train_labels.extend([label] * len(current_train_files))
        test_labels.extend([label] * len(current_test_files))
        print(train_labels)
        print(test_labels)

        # all_labels.extend([label] * len(current_dir_files))

    train_data = list(zip(train_files, train_labels))
    val_data = list(zip(test_files, test_labels))

    # --- Calculate class weights ---
    # Needed for Categorical Cross-Entropy loss to handle class imbalance
    train_labels = [label for _, label in train_data]


    drone_count = len([e for e in train_labels if e == 0.0])
    neg_count = len([e for e in train_labels if e == 1.0])
    piston_count = len([e for e in train_labels if e == 2.0])
    turbofan_count = len([e for e in train_labels if e == 3.0])
    turboprop_count = len([e for e in train_labels if e == 4.0])
    turboshaft_count = len([e for e in train_labels if e == 5.0])

    # Inverse frequency weighting: pos_weight = total_samples / (2 * pos_class_count)
    # This factor scales the loss of the positive class samples.

    total = len(train_labels)
    drone_weight = (total / (6.0 * drone_count)) if drone_count > 0 else 1.0
    neg_weight = (total / (6.0 * neg_count)) if neg_count > 0 else 1.0
    piston_weight = (total / (6.0 * piston_count)) if piston_count > 0 else 1.0
    turbofan_weight = (total / (6.0 * turbofan_count)) if turbofan_count > 0 else 1.0
    turboprop_weight = (total / (6.0 * turboprop_count)) if turboprop_count > 0 else 1.0
    turboshaft_weight = (total / (6.0 * turboshaft_count)) if turboshaft_count > 0 else 1.0

    print(f"\n--- Class Balance ---")
    print(f"Drone Count: {int(drone_count)} samples")
    print(f"Negative Count: {int(neg_count)} samples")
    print(f"Piston Count: {int(piston_count)} samples")
    print(f"Turbofan Count: {int(turbofan_count)} samples")
    print(f"Turboprop Count: {int(turboprop_count)} samples")
    print(f"Turboshaft Count: {int(turboshaft_count)} samples")
    print(f"{'='*70}\n")

    return train_data, val_data, torch.tensor([drone_weight, neg_weight, piston_weight, turbofan_weight, turboprop_weight, turboshaft_weight], dtype=torch.float32)

def collate_fn_fsspec_free(batch: list, feature_extractor: AutoFeatureExtractor) -> Dict:
    """Collate function for the DataLoader, converting raw waveforms to AST features."""
    waveforms = [item['waveform'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Use the Hugging Face feature extractor to convert raw audio into the model's required input format (spectrogram patches)
    inputs = feature_extractor(
        waveforms,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt", # Return PyTorch tensors
        padding=True, # Pad all sequences to the longest in the batch
        return_attention_mask=True
    )

    if 'attention_mask' not in inputs:
        # Manually construct attention mask for the Transformer model input (needed if feature_extractor doesn't generate it)
        # Sequence length = Number of patches + 1 for the CLS token
        num_patches = inputs['input_values'].shape[-1]
        sequence_length = num_patches + 1
        inputs['attention_mask'] = torch.ones(
            (inputs['input_values'].shape[0], sequence_length),
            dtype=torch.long
        )

    # Format labels for BCE loss: shape (B,) -> (B, 1)
    inputs['labels'] = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return inputs

class AudioDataset(Dataset):
    """Dataset with augmentation."""
    def __init__(self, data: list, sampling_rate: int, augment: bool = False):
        if augment:
            # --- Dataset Expansion for Training ---
            expanded_data = []
            for file_path, label in data:
                # Multiply the original samples to create a larger augmented dataset
                for _ in range(AUGMENTATION_MULTIPLIER):
                    expanded_data.append((file_path, label))
            self.data = expanded_data
            print(f" Dataset expanded: {len(data)} → {len(expanded_data)} samples")
        else:
            self.data = data

        self.sr = sampling_rate
        self.augment = augment
        self.augmenter = AudioAugmenter(sampling_rate) if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        file_path, label = self.data[idx]

        try:
            # Load audio file using librosa (uses numpy format)
            waveform, _ = librosa.load(file_path, sr=self.sr, mono=True)
        except Exception:
            # Handle corrupted files by returning a zero-filled waveform (5 seconds long)
            waveform = np.zeros(self.sr * 5)

        if self.augment and self.augmenter:
            waveform = self.augmenter.augment(waveform)

        return {'waveform': waveform, 'labels': label}

# Audio Augmentation
class AudioAugmenter:
    """Augmentation with safety checks."""
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def augment(self, waveform):
        """Apply random augmentations with error handling."""
        try:
            aug_waveform = waveform.copy()

            # Time stretch (alters speed/tempo without changing pitch, helpful for variable speeds of aircraft to me is unclear)
            if np.random.random() < AUGMENTATION_PROBABILITY:
                rate = np.random.uniform(0.95, 1.05)
                # librosa uses slightly different API for modern versions, but rate=rate is common
                aug_waveform = librosa.effects.time_stretch(aug_waveform, rate=rate)

            # Pitch shift (alters pitch without changing duration)
            if np.random.random() < AUGMENTATION_PROBABILITY:
                n_steps = np.random.uniform(-1.5, 1.5)
                aug_waveform = librosa.effects.pitch_shift(aug_waveform, sr=self.sr, n_steps=n_steps)

            # Add Gaussian noise (simulates real-world environmental noise maybe)
            if np.random.random() < AUGMENTATION_PROBABILITY:
                noise_level = np.random.uniform(0.001, 0.003)
                noise = np.random.normal(0, noise_level, aug_waveform.shape)
                aug_waveform = aug_waveform + noise

            # Random gain (simulates variable distance/loudness of aircraft or that is my guess)
            if np.random.random() < AUGMENTATION_PROBABILITY:
                gain = np.random.uniform(0.85, 1.15)
                aug_waveform = aug_waveform * gain

            # Safety: clip and check for NaN
            aug_waveform = np.clip(aug_waveform, -1.0, 1.0)
            if np.isnan(aug_waveform).any() or np.isinf(aug_waveform).any():
                return waveform # Return original if augmentation resulted in invalid data

            return aug_waveform
        except Exception as e:
            # Fallback to original waveform if any augmentation step fails
            return waveform

def load_and_configure_model(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 6 # Set num_labels for binary classification (output is a single logit)
    config.problem_type = "single_label_classification"

    # Load the base AST model
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True # Ignore the mismatch when the pre-trained head (527 classes) is replaced by the new head again expediance here
    )

    d_model = config.hidden_size # Dimension of the hidden state (Transformer embedding size)

    # There are 6 classes: drone, neg, piston, turbofan, turboprop, turboshaft
    # Ensure the final classification head is a simple Linear layer mapping d_model to 6 output logits.
    model.classifier = nn.Linear(d_model, 6)
    print(f"Classifier head installed\n")

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

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    """Saves the best model's state (including MoE layers and LoRA) and training progress."""
    print(f"\n Saving new best checkpoint to {path}")
    # Unwrap DDP or PeftModel for saving state dict
    model_to_save = model.module if hasattr(model, 'module') else model

    state = {
        'epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        # Save the model's entire state dict (includes LoRA weights and MoE experts/gates)
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(state, path)

def collate_function(b):
    return collate_fn_fsspec_free(b, feature_extractor)

if __name__ == '__main__':
    device = torch.device("mps")

    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    model = load_and_configure_model(model_name).to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    train_data_paths, val_data_paths, class_weights = load_data_paths_and_labels()

    # Datasets
    train_dataset = AudioDataset(train_data_paths, feature_extractor.sampling_rate, augment=True)
    val_dataset = AudioDataset(val_data_paths, feature_extractor.sampling_rate, augment=False)

    class_weights = class_weights.to(device) # Move the class weight tensor to the GPU


    # Dataloaders: num_workers=0 is used to prevent potential multiprocessing issues on Windows/specific setups
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # The lambda function wraps the custom collate function with the feature_extractor
        collate_fn=collate_function,
        num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_function,
        num_workers=2
    )

    # Optimizer with scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE, # This is the max LR used by OneCycleLR
        weight_decay=WEIGHT_DECAY
    )
    # GradScaler is essential for mixed-precision (autocast) training to prevent gradient underflow
    scaler = torch.amp.GradScaler("mps")

    # Calculate total steps for the OneCycleLR scheduler
    num_optimization_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    total_steps = num_optimization_steps_per_epoch * NUM_EPOCHS

    # OneCycleLR: A highly effective schedule that incorporates warmup and cosine decay
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE, # Peak learning rate
        total_steps=total_steps,
        pct_start=0.15, # 15% of steps used for the warmup phase
        anneal_strategy='cos',
        div_factor=10.0, # Initial LR is max_lr / div_factor (for warmup start)
        final_div_factor=100.0 # Final LR is max_lr / final_div_factor
    )

    print(f"{'='*70}")
    print(f"Training Configuration:")
    print(f"  Base samples: {len(train_data_paths)} train / {len(val_data_paths)} val")
    print(f"  Augmented: {len(train_dataset)} training samples")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE:.2e} (with warmup & cosine decay)")
    print(f"  Gradient clipping: {GRAD_CLIP_NORM}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
    print(f"{'='*70}\n")

    global_step = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Move the batch data to the GPU
            input_data = {k: v.to(device) for k, v in batch.items()}

            # --- Forward Pass with Mixed Precision ---
            with torch.autocast(device_type='mps'):
                # Model forward pass
                outputs = model(
                    input_values=input_data['input_values'],
                    attention_mask=input_data['attention_mask']
                )

                logits = outputs.logits

                loss = F.cross_entropy(
                    logits,
                    input_data['labels'].long().squeeze(),
                    weight=class_weights # Apply class imbalance correction
                )

                final_loss = loss

            # final_loss = bce_loss + (moe_aux_loss * LOAD_BALANCING_COEFF)

            # Normalize the loss by the accumulation steps before backpropagation
            final_loss = final_loss / GRADIENT_ACCUMULATION_STEPS
            # Scale the loss before backward pass (part of mixed precision)
            scaler.scale(final_loss).backward()

            # Check if this is the last step of the accumulation cycle or the last batch of the epoch
            is_optimization_step = (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader)

            if is_optimization_step: # Only optimize and step the scheduler on optimization steps
                # --- Optimization Step ---

                # Unscale the gradients before clipping (required by GradScaler)
                scaler.unscale_(optimizer)
                # Gradient clipping to prevent exploding gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                # Optimizer step (applies unscaled gradients)
                scaler.step(optimizer)
                # Updates the scaling factor for the next iteration
                scaler.update()
                # Zero the gradients for the next accumulation cycle
                optimizer.zero_grad()

                # Step the learning rate scheduler
                if global_step < total_steps:
                    scheduler.step()

                global_step += 1
                # Calculate the loss for logging (un-normalized by accumulation steps)
                current_loss = final_loss.item() * GRADIENT_ACCUMULATION_STEPS

                # Logging output
                if global_step % 10 == 0 or (step + 1) == len(train_dataloader):
                    bce_log = loss.item()
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Step {global_step:3d} | Loss: {current_loss:.4f} (BCE: {bce_log:.4f}) | LR: {current_lr:.2e} | GradNorm: {grad_norm:.2f}")

            # Periodically free up unused memory
            if (step + 1) % EMPTY_CACHE_EVERY_N_STEPS == 0:
                torch.mps.empty_cache()

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): # Disable gradient calculation for validation (saves memory and speeds up)
            for batch in val_dataloader:
                input_data = {k: v.to(device) for k, v in batch.items()}

                with torch.autocast(device_type='mps'):
                    outputs = model(
                        input_values=input_data['input_values'],
                        attention_mask=input_data['attention_mask']
                    )

                    logits = outputs.logits
                    loss = F.cross_entropy(
                        logits,
                        input_data['labels'].long().squeeze(),
                        weight=class_weights
                    )
                    val_loss += loss.item()

                # torch.argmax(torch.sigmoid(outputs.logits))
                # print(outputs.logits)
                # print(labels)
                predictions = torch.argmax((torch.sigmoid(outputs.logits)), dim=1).long().flatten()
                labels = input_data['labels'].long().flatten()

                # print(predictions.shape)
                # print(labels.shape)
                for i in range(len(predictions) - 1):
                  if predictions[i] == labels[i]:
                    correct += 1
                # correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total if total > 0 else 0
        val_loss = val_loss / len(val_dataloader) # Average validation loss

        # --- Checkpoint and Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save model state only if validation loss improves
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, CHECKPOINT_PATH)
            print(f"NEW BEST → Epoch {epoch+1:2d} | Val Acc: {val_accuracy:.4f} ({correct}/{total}) | Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1:2d} | Val Acc: {val_accuracy:.4f} ({correct}/{total}) | Val Loss: {val_loss:.4f} | Best: {best_val_acc:.4f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # Early stopping check
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            print(f"No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
            break

        print()

    print("="*70)
    print(f"TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*70)




    # finetune(train_dataloader, val_dataloader, class_weights)

    # Memory check
    # torch.cuda.empty_cache()
    # allocated = torch.cuda.memory_allocated(0) / 1024**3
    # reserved = torch.cuda.memory_reserved(0) / 1024**3
    # print(f"--- GPU Memory ---")
    # print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Free: {8-reserved:.2f} GB")
    # print(f"{'='*70}\n")
