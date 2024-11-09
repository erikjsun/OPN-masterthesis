# Import the classes used in Dataset1
from preprocessed_temporal_four_data_class import PreprocessedTemporalFourData
from prepared_dataset_class import PreparedDataset
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio.v3 as iio
from PIL import Image
import random

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

# Define the PreprocessedTemporalFourData class
def get_video_data(self, index):
    video = self.dataset.predata_train[index] if self.trainval == 'train' else self.dataset.predata_test[index]
    action_label = video['action_label']
    video_name = video['video_name']
    return video, action_label, video_name

class PreprocessedTemporalFourData(DataLoader):
    def __init__(self, dataset, trainval='train', pixel_mean=96.5, patch_size=160):
        self.dataset = dataset
        self.trainval = trainval
        self.pixel_mean = pixel_mean
        self.patch_size = patch_size

    def __len__(self):
        if self.trainval == 'train':
            return len(self.dataset.predata_train)
        else:
            return len(self.dataset.predata_test)

    def __getitem__(self, index):
        video, action_label, video_name = self.get_video_data(index)
        # Read the frames from the video
        frames = iio.imread(video['data'], index=None, format_hint=".avi")
        # Drop every other frame
        frames = frames[::2]

        # Compute the optical flow weights for frame selection
        weights, flows = self.compute_optical_flow_weights(frames)

        # Select the four frames from the video based on the weights of the optical flow – random order
        indices = np.random.choice(frames.shape[0], size=4, replace=False, p=weights)
        selected_frames = frames[indices]

        # Normalize the indices to their respective order
        ranked_indices = indices.argsort().argsort()

        # Saving the ordered frames for visualization
        ordered_indices = np.sort(indices)
        ordered_frames = frames[ordered_indices]

        # Create label list for the frame order
        frame_order_label, frames_canonical_order = self.get_frame_order_label(ranked_indices)

        # Applying random mirroring to all selected frames at once
        mirror = random.randint(0, 1)
        if mirror == 1:
            selected_frames_list = [np.array(Image.fromarray(frame.astype('uint8')).transpose(Image.FLIP_LEFT_RIGHT)) for frame in selected_frames]
            selected_frames = np.array(selected_frames_list)
            ordered_frames_list = [np.array(Image.fromarray(frame.astype('uint8')).transpose(Image.FLIP_LEFT_RIGHT)) for frame in ordered_frames]
            ordered_frames = np.array(ordered_frames_list)

        # Select the best patch from the frames
        best_patch = self.select_best_patch(selected_frames)

        # Preprocess the frames, including spatial jittering and channel splitting
        preprocessed_frames, preprocessed_frames_coordinates = self.preprocess_frames(selected_frames, best_patch)

        # Convert the action_label, flows, and indices to PyTorch tensors
        action_label = torch.tensor(action_label)
        flows = torch.tensor(np.array(flows))
        indices = torch.tensor(indices)
        # print("Shape of preprocessed_frames:", preprocessed_frames.shape)
        return preprocessed_frames, frame_order_label, action_label, video_name, frames_canonical_order, selected_frames, preprocessed_frames_coordinates, ordered_frames

# Main function to load dataset1 and analyze the DataLoader's first few elements
def main():
    # Load configuration
    config = load_config()

    # Load dataset1 from the specified path
    dataset_train_path = config['dataset_train_path']
    dataset1_data = load_dataset1(dataset_train_path)

    if dataset1_data is None:
        print("Error: Could not load dataset1.")
        return

    # Create PreprocessedTemporalFourData instance for Dataset1
    dataset1 = PreprocessedTemporalFourData(dataset1_data, trainval='train')

    # Create DataLoader instance
    train_loader = DataLoader(dataset1, batch_size=4, shuffle=True, num_workers=0)

    # Analyze the first few batches and print details
    max_batches_to_print = 5
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        
        # Unpack the batch (which contains tuples of elements)
        try:
            preprocessed_frames, frame_order_label, action_label, video_name, frames_canonical_order, selected_frames, preprocessed_frames_coordinates, ordered_frames = batch
        except ValueError as e:
            print(f"Error unpacking batch {batch_idx}: {e}")
            continue

        # Print details for the first two samples in the batch
        for i in range(2):
            print(f"\nSample {i} in Batch {batch_idx}:")
            if isinstance(preprocessed_frames, torch.Tensor):
                print(f"  Preprocessed Frames shape: {preprocessed_frames[i].shape}")
                print(f"  Preprocessed Frames values (first 5 elements of flattened tensor): {preprocessed_frames[i].flatten()[:5]}")

            # Print Frame Order Label, Action Label, and Video Name
            print(f"  Frame Order Label: {frame_order_label[i]}")
            print(f"  Action Label: {action_label[i]}")
            print(f"  Video Name: {video_name[i]}")

            # Print Frames Canonical Order if available
            if isinstance(frames_canonical_order, torch.Tensor):
                print(f"  Frames Canonical Order: {frames_canonical_order[i]}")

        # Only print up to a few batches to keep the output manageable
        if batch_idx >= max_batches_to_print - 1:
            break

if __name__ == "__main__":
    main()
