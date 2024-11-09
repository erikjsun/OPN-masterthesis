import json
import torch

# Import the classes used in Dataset1
from preprocessed_temporal_four_data_class import PreprocessedTemporalFourData
from prepared_dataset_class import PreparedDataset

# Load configuration from JSON file
def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

# Load dataset1 from the provided path
def load_dataset1(dataset_train_path):
    print(f"Loading Dataset1 from: {dataset_train_path}...")

    try:
        # Load dataset1 from the specified path
        dataset1 = torch.load(dataset_train_path)
        print("Dataset1 loaded successfully.")
    except Exception as e:
        print(f"Error loading Dataset1: {e}")
        return None

    return dataset1

# Analyze and print detailed information for specific elements in dataset1
def analyze_dataset_shape(dataset):
    if dataset is None:
        print("Dataset is not loaded. Shape analysis skipped.")
        return

    print(f"\nAnalyzing shape of Dataset1 with {len(dataset)} samples:")

    # Print detailed information for specific elements in each sample
    num_samples_to_print = 5
    for i in range(min(num_samples_to_print, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}: Type = {type(sample)}")
        
        if isinstance(sample, torch.Tensor):
            print(f"  Shape: {sample.shape}")
        elif isinstance(sample, (list, tuple)):
            print(f"  Length of sample: {len(sample)}")
            for j, elem in enumerate(sample):
                # Skip printing elements 5 and 7
                if j == 5 or j == 7:
                    print(f"    Element {j}: Skipped (too long)")
                    continue

                # Provide detailed information for elements 1, 2, 3, 4, and 6
                if j in [1, 2, 3, 4, 6]:
                    print(f"    Element {j}: Type = {type(elem)}")
                    if isinstance(elem, torch.Tensor):
                        print(f"      Shape = {elem.shape}, Value = {elem.item() if elem.numel() == 1 else elem}")
                    elif isinstance(elem, str):
                        print(f"      Value = {elem}")
                    elif isinstance(elem, list) or isinstance(elem, tuple):
                        print(f"      Value = {elem}")
                    else:
                        print(f"      Value: {elem}")
                else:
                    # Provide basic info for other elements
                    if isinstance(elem, torch.Tensor):
                        print(f"    Element {j}: Type = {type(elem)}, Shape = {elem.shape}")
                    else:
                        print(f"    Element {j}: Type = {type(elem)}, Value: {elem}")

# Main function to load and analyze dataset1
def main():
    # Load configuration
    config = load_config()

    # Load dataset1
    dataset_train_path = config['dataset_train_path']
    dataset1 = load_dataset1(dataset_train_path)

    # Analyze the shape and details of dataset1
    analyze_dataset_shape(dataset1)

# Entry point
if __name__ == "__main__":
    main()