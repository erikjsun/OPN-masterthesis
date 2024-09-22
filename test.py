import io
import torch
from azure.storage.blob import BlobServiceClient, ContainerClient
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from PIL import Image

# Import the necessary functions for visualization
from data_prep_withblobsaving_nobatches import PreprocessedTemporalFourData, PreparedDataset # Import classes
from data_prep_withblobsaving_nobatches import test_temporal_four, visualize_frames  # Ensure this imports from your updated data prep code

# Azure Blob Storage configuration
STORAGEACCOUNTURL = "https://exjobbssl1863219591.blob.core.windows.net"
STORAGEACCOUNTKEY = "PuL1QY8bQvIyGi653lr/9CPvyHLnip+cvsu62YAipDjB7onPDxfME156z5/O2NwY0PRLMTZc86/6+ASt5Vts8w=="
CONTAINERNAME = "exjobbssl"
PREPROCESSEDDATA_FOLDERNAME = "preprocessed-data"

# Initialize Azure Blob Storage client
blob_service_client = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
container_client = blob_service_client.get_container_client(CONTAINERNAME)

def basic_check_and_visualize():
    # List the blobs in the preprocessed-data folder
    print(f"Listing blobs in: {PREPROCESSEDDATA_FOLDERNAME}")
    blob_list = container_client.list_blobs(name_starts_with=PREPROCESSEDDATA_FOLDERNAME + '/')
    blob_names = [blob.name for blob in blob_list]

    # Filter to get only .pth files
    blob_names = [blob_name for blob_name in blob_names if blob_name.endswith('.pth')]

    if not blob_names:
        print("No preprocessed data files found in the blob storage.")
        return

    # Select a random preprocessed data file to download
    blob_name = random.choice(blob_names)
    print(f"Downloading blob: {blob_name}")

    try:
        # Get the blob client
        blob_client = container_client.get_blob_client(blob_name)

        # Download the blob content
        downloaded_blob = blob_client.download_blob().readall()

        # Load the preprocessed data
        buffer = io.BytesIO(downloaded_blob)
        buffer.seek(0)
        preprocessed_data = torch.load(buffer)  # Load the preprocessed data tuple

        print("Data successfully downloaded and loaded from Blob Storage.")

        # Visualize the preprocessed data
        test_temporal_four(preprocessed_data)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Run the basic check and visualization
    basic_check_and_visualize()
