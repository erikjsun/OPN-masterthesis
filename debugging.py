import io
import torch
import numpy as np
from azure.storage.blob import BlobServiceClient
from torch.utils.data import Dataset
from data_prep import visualize_frames, test_temporal_four  # Ensure this imports your necessary functions

class BlobPreprocessedDataset(Dataset):
    def __init__(self, blob_service_client, container_name, blob_names):
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.blob_names = blob_names
        self.data = self.load_and_unbatch_blobs()  # Load and unbatch data from blobs

    def load_and_unbatch_blobs(self):
        all_data = []

        for blob_name in self.blob_names:
            # Get the blob client
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)

            # Download the blob content (this should be a batch of items)
            downloaded_blob = blob_client.download_blob().readall()
            buffer = io.BytesIO(downloaded_blob)
            buffer.seek(0)

            # Load the preprocessed batch of data
            preprocessed_batch = torch.load(buffer)  # This is a list of items in the batch

            # Debugging information about the batch
            print(f"Loaded batch from Blob Storage {blob_name}: Type={type(preprocessed_batch)}, Length={len(preprocessed_batch)}")

            # Unbatch the items (append each item separately to the list)
            for item in preprocessed_batch:
                all_data.append(item)

        return all_data

    def __len__(self):
        return len(self.data)  # Return the total number of items across all batches

    def __getitem__(self, idx):
        return self.data[idx]

def load_preprocessed_data_from_blob(blob_service_client, container_name, folder_name):
    """
    Function to load preprocessed data from Azure Blob Storage and return a dataset.
    
    Parameters:
        blob_service_client: Initialized Azure BlobServiceClient instance.
        container_name: The name of the container in Azure Blob Storage.
        folder_name: The folder containing the .pth preprocessed files.
    
    Returns:
        dataset: BlobPreprocessedDataset containing the loaded data.
    """
    container_client = blob_service_client.get_container_client(container_name)

    # List the blobs in the preprocessed-data folder
    print(f"Listing blobs in: {folder_name}")
    blob_list = container_client.list_blobs(name_starts_with=folder_name + '/')
    blob_names = [blob.name for blob in blob_list if blob.name.endswith('.pth')]

    if not blob_names:
        print("No preprocessed data files found in the blob storage.")
        return None
    else:
        print(f"Found {len(blob_names)} preprocessed data files.")

    # Return the BlobPreprocessedDataset
    return BlobPreprocessedDataset(blob_service_client, container_name, blob_names)

# Debugging function for frame order
def debug_frame_order(preprocessed_data):
    """
    Debugging function to validate the frame order labels and verify the processing steps.
    
    Parameters:
        preprocessed_data: List containing tuples with the following elements:
            - preprocessed_frames: Preprocessed frames (cropped and jittered).
            - frame_order_label: The label indicating the frame order.
            - action_label: The action label for the video.
            - video_name: The name of the video.
            - frames_canonical_order: The canonical order of the frames.
            - selected_frames: The original selected frames before preprocessing.
            - preprocessed_frames_coordinates: Coordinates used for cropping.
            - ordered_frames: Chronologically ordered frames.
    """
    for i, (preprocessed_frames, frame_order_label, action_label, video_name, 
            frames_canonical_order, selected_frames, preprocessed_frames_coordinates, ordered_frames) in enumerate(preprocessed_data):
        
        print(f"\n### Video {i+1}: {video_name} ###")
        
        # Frame order label
        print(f"Frame Order Label: {frame_order_label}")
        
        # Canonical order of the frames
        print(f"Canonical Frame Order: {frames_canonical_order}")
        
        # Selected frame indices (before preprocessing)
        print(f"Selected Frame Indices (Preprocessing): {np.argsort(frames_canonical_order)}")
        
        # Preprocessed frame coordinates
        print(f"Preprocessed Frame Coordinates: {preprocessed_frames_coordinates}")
        
        # Debug the actual frames (optional): show them if needed
        visualize_frames(preprocessed_frames, selected_frames, frames_canonical_order, video_name, preprocessed_frames_coordinates, ordered_frames)
        
        # Verify if frame order label corresponds to the actual frame order
        if verify_frame_order_label(frames_canonical_order, frame_order_label):
            print("Frame order label is correct.")
        else:
            print("Frame order label is incorrect.")

def verify_frame_order_label(canonical_order, frame_order_label):
    """
    Verifies if the frame order label matches the expected order of frames.

    Parameters:
        canonical_order: The canonical order of the frames.
        frame_order_label: The label indicating the frame order.

    Returns:
        bool: True if the label matches the order, False otherwise.
    """
    # Example logic: Check against the frame_order_to_label_dict
    frame_order_to_label_dict = {
        (0, 1, 2, 3): 0,
        (0, 2, 1, 3): 1,
        (0, 3, 2, 1): 2,
        (0, 1, 3, 2): 3,
        (0, 3, 1, 2): 4,
        (0, 2, 3, 1): 5,
        (1, 0, 2, 3): 6,
        (1, 0, 3, 2): 7,
        (1, 2, 0, 3): 8,
        (1, 3, 0, 2): 9,
        (2, 0, 1, 3): 10,
        (2, 1, 0, 3): 11
    }

    canonical_tuple = tuple(canonical_order.numpy())  # Ensure it's a tuple
    expected_label = frame_order_to_label_dict.get(canonical_tuple)

    return expected_label == frame_order_label.item()

# Initialize Blob Service Client
STORAGEACCOUNTURL = "https://exjobbssl1863219591.blob.core.windows.net"
STORAGEACCOUNTKEY = "PuL1QY8bQvIyGi653lr/9CPvyHLnip+cvsu62YAipDjB7onPDxfME156z5/O2NwY0PRLMTZc86/6+ASt5Vts8w=="
CONTAINERNAME = "exjobbssl"
PREPROCESSEDDATA_FOLDERNAME = "ucf-preprocessed-data"
blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

# Load the dataset from Blob Storage
dataset = load_preprocessed_data_from_blob(
    blob_service_client_instance,
    CONTAINERNAME,
    PREPROCESSEDDATA_FOLDERNAME
)

# Example usage: debug the frame order of the dataset
if dataset is not None:
    debug_frame_order(dataset.data)