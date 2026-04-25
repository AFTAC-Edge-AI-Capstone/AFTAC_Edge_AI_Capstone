# Based on this python file sent by Eddie: W_AST_TrainingWcomments.py
# Modified by Aaron Mathews
import os
import soundfile as sf
import librosa
import glob
import math
from typing import Tuple, Dict, List

DATA_DIRS = {
    "Drone": "./datasets/drone",
    "Piston": "./datasets/piston", 
    "Turbofan": "./datasets/turbofan",
    "Turboprop": "./datasets/turboprop"
}

MODIFIED_DATA_DIRS = {
    "Neg": "./modified_datasets/neg", # Directory containing negative (Noise/Background) samples (Label 0.0)
    "Drone": "./modified_datasets/drone",
    "Piston": "./modified_datasets/piston", 
    "Turbofan": "./modified_datasets/turbofan",
    "Turboprop": "./modified_datasets/turboprop",
    "Turboshaft": "./modified_datasets/turboshaft"
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

def process_audio():
    """Load file paths, assign labels, split into train/val, and calculate class weights."""
    
    train_files = []
    train_labels = []
    
    test_files = []
    test_labels = []

    print(f"\n--- Dataset Loading ---")

    for dataset_name, dataset_path in DATA_DIRS.items():
        label = LABEL_MAP[dataset_name]
        print(f"Label: {label}")
        
        modified_data_dir = MODIFIED_DATA_DIRS[dataset_name]

        modified_test_data_dir = f"{modified_data_dir}/test"
        modified_train_data_dir = f"{modified_data_dir}/train"

        if not os.path.isdir(modified_test_data_dir):
            os.makedirs(modified_test_data_dir)

        if not os.path.isdir(modified_train_data_dir):
            os.makedirs(modified_train_data_dir)

        current_train_files = []
        current_test_files = []
        for ext in AUDIO_EXTENSIONS:
            # Use glob to find all audio files recursively in the directory
            # current_train_files.extend(glob.glob(os.path.join(dataset_path + '/train', ext)))
            # current_test_files.extend(glob.glob(os.path.join(dataset_path + '/test', ext)))
            
            current_train_files = glob.glob(os.path.join(dataset_path + '/train', ext))
            modified_train_files = []

            for file_path in current_train_files:
                x, sampling_rate = librosa.load(file_path, sr=None)

                # All wav file lengths should be multiple of 10
                for i in range(0, len(x), 10*sampling_rate):
                    chunk = x[i:i+10*sampling_rate]

                    file = file_path.split("/")[-1]
                    modified_file_path = f"{modified_data_dir}/train/sample_{i}_{file}"
                    modified_train_files.append(modified_file_path)

                    sf.write(modified_file_path, chunk, sampling_rate)

            current_test_files = glob.glob(os.path.join(dataset_path + '/test', ext))
            modified_test_files = []

            for file_path in current_test_files:
                x, sampling_rate = librosa.load(file_path, sr=None)


                # All wav file lengths should be multiple of 10
                for i in range(0, len(x), 10*sampling_rate):
                    chunk = x[i:i+10*sampling_rate]
                    file = file_path.split("/")[-1]
                    modified_file_path = f"{modified_data_dir}/test/sample_{i}_{file}"
                    modified_test_files.append(modified_file_path)
                    sf.write(modified_file_path, chunk, sampling_rate)

        # current_train_files = current_train_files[0:int(len(current_train_files)/4)]
        # current_test_files = current_test_files[0:int(len(current_test_files)/4)]

        # all_files.extend(current_dir_files)

        train_files.extend(modified_train_files)
        test_files.extend(modified_test_files)

        train_labels.extend([label] * len(modified_train_files))
        test_labels.extend([label] * len(modified_test_files))

        # all_labels.extend([label] * len(current_dir_files))
        
    train_data = list(zip(train_files, train_labels))
    val_data = list(zip(test_files, test_labels))
    
    train_labels = [label for _, label in train_data]
    
    neg_count = len([e for e in train_labels if e == 0.0])
    drone_count = len([e for e in train_labels if e == 1.0])
    piston_count = len([e for e in train_labels if e == 2.0])
    turbofan_count = len([e for e in train_labels if e == 3.0])
    turboprop_count = len([e for e in train_labels if e == 4.0])
    turboshaft_count = len([e for e in train_labels if e == 5.0])
    
    print(f"\n--- Class Balance ---")
    print(f"Drone Count: {int(drone_count)} samples")
    print(f"Negative Count: {int(neg_count)} samples")
    print(f"Piston Count: {int(piston_count)} samples")
    print(f"Turbofan Count: {int(turbofan_count)} samples")
    print(f"Turboprop Count: {int(turboprop_count)} samples")
    print(f"Turboshaft Count: {int(turboshaft_count)} samples")
    print(f"{'='*70}\n")
    
if __name__ == '__main__':
    process_audio()