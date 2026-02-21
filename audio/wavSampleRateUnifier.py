import librosa
import soundfile as sf
import os
import csv

def standardize_audio(input_path, output_path, target_sr=16000):
    try:
        # librosa.load automatically resamples to target_sr if you specify it!
        # mono=True ensures it's 1-channel, which is standard for ML
        audio_data, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        # sf.write with subtype='PCM_16' forces the file into 16-bit (int16) format
        sf.write(output_path, audio_data, target_sr, subtype='PCM_16')
        print(f"Successfully converted: {os.path.basename(input_path)} -> {os.path.basename(os.path.dirname(output_path))}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_directory_with_labels(input_dir, output_base_dir, csv_path):
    # 1. Define how (class, split) combinations map to subdirectories
    folder_mapping = {
        ('aircraft', 'train'): 'air-train',
        ('aircraft', 'test'):  'air-test',
        ('negative', 'train'): 'neg-train',
        ('negative', 'test'):  'neg-test'
    }

    # 2. Create the target subdirectories if they don't exist
    for folder_name in folder_mapping.values():
        os.makedirs(os.path.join(output_base_dir, folder_name), exist_ok=True)

    # 3. Read the CSV and build a routing dictionary { "filename.wav": "target_folder" }
    routing_dict = {}
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                audio_class = row['class']
                data_split = row['split']
                
                # Look up the target folder based on class and split
                target_folder = folder_mapping.get((audio_class, data_split))
                if target_folder:
                    routing_dict[filename] = target_folder
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # 4. Iterate over all .wav files in the input directory
    for file in os.listdir(input_dir):
        if file.lower().endswith('.wav'):
            if file in routing_dict:
                input_path = os.path.join(input_dir, file)
                target_folder = routing_dict[file]
                output_path = os.path.join(output_base_dir, target_folder, file)
                
                # Standardize and save to the routed destination
                standardize_audio(input_path, output_path)
            else:
                print(f"Skipped {file}: Not found in CSV or invalid class/split.")

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    # Change these paths to match your actual directory structures
    RAW_AUDIO_DIR = "/home/hackerman/CSCE482/datasets/aircraft/newData/audio/1" 
    PROCESSED_AUDIO_DIR = "./processed_audio"
    CSV_FILE = "/home/hackerman/CSCE482/datasets/aircraft/labels.csv"
    
    # Process the dataset
    process_directory_with_labels(RAW_AUDIO_DIR, PROCESSED_AUDIO_DIR, CSV_FILE)