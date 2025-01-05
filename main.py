# MAIN.PY

# 1. IMPORTS
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import psutil
from azure.storage.blob import BlobServiceClient, ContainerClient
import io
import gc
import random
from sklearn.model_selection import train_test_split
import sys

# 2. CUSTOM IMPORTS (LOCAL MODULES)
from model import CustomOPN
from data_prep import PreparedDataset, PreprocessedTemporalFourData

# 3. GLOBAL CONFIG / CONSTANTS
# Azure Blob Storage configuration
STORAGEACCOUNTURL = "https://exjobbssl1863219591.blob.core.windows.net"
STORAGEACCOUNTKEY = "PuL1QY8bQvIyGi653lr/9CPvyHLnip+cvsu62YAipDjB7onPDxfME156z5/O2NwY0PRLMTZc86/6+ASt5Vts8w=="
CONTAINERNAME = "exjobbssl"
PREPROCESSEDDATA_FOLDERNAME = "ucf-preprocessed-data-100"

epoch_amount = 5  # For testing
chunk_size = 4
training_batch_size = 32
num_workers = 1

##################################################
# 4. DATASET CLASS FOR FULLY EXTRACTED .PTH FILES
##################################################
class FullyExtractedBlobDataset(Dataset):
    """
    Loads multiple fully-extracted .pth files (the 8-tuple samples) from Azure Blob Storage
    into memory as a single list. Each chunk is passed a list of .pth blob names.
    """
    def __init__(self, blob_service_client, container_name, pth_blob_names):
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.pth_blob_names = pth_blob_names
        self.samples = []  # store all 8-tuple samples from these pth files
        self._load_pth_files()

    def _load_pth_files(self):
        # Download each .pth blob, load the final list of 8-tuples, and extend self.samples
        for i, blob_name in enumerate(self.pth_blob_names, start=1):
            print(f"    => Downloading file {i}/{len(self.pth_blob_names)}: {blob_name}")
            dl_start = time.time()

            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            downloaded_blob = blob_client.download_blob().readall()
            dl_end = time.time()
            print(f"       Download took {dl_end - dl_start:.2f} seconds. Size: {len(downloaded_blob)} bytes.")

            buffer = io.BytesIO(downloaded_blob)
            # This should be a list of final samples (8-tuples)
            data_in_this_file = torch.load(buffer)
            self.samples.extend(data_in_this_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Each item is an 8-tuple:
        # (preprocessed_frames, frame_order_label, action_label, video_name,
        #  frames_canonical_order, selected_frames, preprocessed_coords, ordered_frames)
        return self.samples[idx]

##################################################
# 5. HELPER FUNCTIONS
##################################################
def chunkify(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def train_one_chunk(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # Timer initialization
    batch_start_time = None

    for batch_idx, (inputs, frame_order_labels, *rest) in enumerate(train_loader):
        # Print timing from previous batch
        if batch_start_time is not None:
            batch_end_time = time.time()
            batch_processing_time = batch_end_time - batch_start_time
            print(f"Batch {batch_idx}/{len(train_loader)} processed in {batch_processing_time:.2f} seconds")
        else:
            print(f"Batch {batch_idx}/{len(train_loader)} processed (timing starts from next batch)")

        # Reset timer for the current batch
        batch_start_time = time.time()

        # Preprocessing: shape => (batch, 4, H, W, ???) => permute => (batch, 4, ???, H, W)
        # Then flatten frames+channels => (batch, 4*C, H, W)
        inputs = inputs.float().permute(0, 1, 4, 2, 3).contiguous()
        inputs = inputs.view(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3], inputs.shape[4])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, frame_order_labels)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == frame_order_labels).sum().item()
        total_samples += inputs.size(0)

    print()  # newline

    chunk_loss = running_loss / total_samples
    chunk_acc  = running_corrects / total_samples
    return chunk_loss, running_corrects, total_samples

def validate_in_chunks(model, blob_service_client, val_blob_names, batch_size, chunk_size):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # We chunkify the val_blob_names so we load fewer .pth files at once
    for blob_chunk_idx, blob_chunk in enumerate(chunkify(val_blob_names, chunk_size), start=1):
        print(f"Validating on blob_chunk {blob_chunk_idx} with {len(blob_chunk)} file(s)...")

        blob_chunk_start_time = time.time()
        # Create dataset with those .pth files
        val_dataset_blob_chunk = FullyExtractedBlobDataset(
            blob_service_client, CONTAINERNAME, blob_chunk
        )
        val_loader = DataLoader(val_dataset_blob_chunk, batch_size=batch_size, shuffle=False)
        blob_chunk_end_time = time.time()
        print(f"    Blob chunk {blob_chunk_idx} loaded in {blob_chunk_end_time - blob_chunk_start_time:.2f} sec")

        with torch.no_grad():
            for inputs, frame_order_labels, *rest in val_loader:
                # Same dimension logic as in train
                inputs = inputs.float().permute(0, 1, 4, 2, 3).contiguous()
                inputs = inputs.view(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3], inputs.shape[4])

                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, frame_order_labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == frame_order_labels).sum().item()
                total_samples += inputs.size(0)

        del val_dataset_blob_chunk
        del val_loader
        gc.collect()
        torch.cuda.empty_cache()

    val_loss = running_loss / total_samples
    val_acc  = running_corrects / total_samples
    return val_loss, val_acc

def plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, plot_save_path):
    # (unchanged) your existing code
    epochs = range(1, epoch_amount + 1)
    plt.figure(figsize=(12, 12))
    # ...
    # rest of your code for plotting
    plt.savefig(plot_save_path)
    plt.show()

def evaluate_model(test_dataset):
    # (unchanged) your existing code
    # ...
    pass  # if you want to keep it

##################################################
# 6. TRAINING PIPELINE
##################################################
def train_model(model_save_path, 
                blob_service_client, 
                train_blob_names, 
                val_blob_names, 
                epochs=5, 
                chunk_size=2, 
                batch_size=32):

    model = CustomOPN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    train_loss_history = []
    train_acc_history  = []
    val_loss_history   = []
    val_acc_history    = []

    print("\nStarting Training Loop...")

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        # Shuffle the training file list each epoch for randomness
        train_files_shuffled = train_blob_names[:]
        random.shuffle(train_files_shuffled)

        epoch_train_loss     = 0.0
        epoch_train_corrects = 0
        epoch_train_samples  = 0

        # (B) chunkify the shuffled list
        for blob_chunk_idx, blob_chunk in enumerate(chunkify(train_files_shuffled, chunk_size), start=1):
            print(f"  Loading train blob_chunk {blob_chunk_idx} with {len(blob_chunk)} file(s)...")

            # measure loading time
            blob_chunk_load_start_time = time.time()
            train_dataset_blob_chunk = FullyExtractedBlobDataset(
                blob_service_client, CONTAINERNAME, blob_chunk
            )
            blob_chunk_load_end_time   = time.time()
            blob_chunk_loading_time    = blob_chunk_load_end_time - blob_chunk_load_start_time
            print(f"  train blob_chunk {blob_chunk_idx} loaded in {blob_chunk_loading_time:.2f} seconds")

            train_loader = DataLoader(
                train_dataset_blob_chunk,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )

            blob_chunk_train_start_time = time.time()
            chunk_loss, chunk_corrects, chunk_samples = train_one_chunk(
                model, criterion, optimizer, train_loader
            )
            blob_chunk_train_end_time = time.time()
            blob_chunk_train_time     = blob_chunk_train_end_time - blob_chunk_train_start_time
            print(f"  train blob_chunk {blob_chunk_idx} processed in {blob_chunk_train_time:.2f} sec")

            # accumulate
            epoch_train_loss     += chunk_loss * chunk_samples
            epoch_train_corrects += chunk_corrects
            epoch_train_samples  += chunk_samples

            # cleanup
            del train_dataset_blob_chunk
            del train_loader
            gc.collect()
            torch.cuda.empty_cache()

        # final epoch stats
        epoch_train_loss = epoch_train_loss / epoch_train_samples
        epoch_train_acc  = epoch_train_corrects / epoch_train_samples

        # validation
        val_loss, val_acc = validate_in_chunks(model, blob_service_client, val_blob_names, batch_size, chunk_size)

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1} => Train Loss: {epoch_train_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), model_save_path)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


##################################################
# 7. MAIN EXECUTION
##################################################
def main():
    # 1) Load config
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_save_path = config['model_save_path']
    plot_save_path  = config['plot_save_path']

    # 2) Connect to BlobService
    blob_service_client = BlobServiceClient(account_url=STORAGEACCOUNTURL,
                                            credential=STORAGEACCOUNTKEY)
    container_client = blob_service_client.get_container_client(CONTAINERNAME)

    # 3) List .pth blobs
    blob_list = container_client.list_blobs(name_starts_with=PREPROCESSEDDATA_FOLDERNAME + '/')
    # IMPORTANT: We only want the "fully extracted" .pth files.
    # If you stored them with a pattern like "ucf101_preprocessed_fullyextracted_batch_..."
    # you can filter accordingly:
    all_blob_names = [
        blob.name for blob in blob_list
        if blob.name.endswith('.pth') and "fullyextracted" in blob.name
    ]
    if not all_blob_names:
        print("No fully extracted .pth data found in the blob storage.")
        return

    print(f"Found {len(all_blob_names)} fully extracted .pth files in the container.")

    # 4) Train/Val split
    train_blob_names, val_blob_names = train_test_split(all_blob_names, test_size=0.2, random_state=42)
    print(f"{len(train_blob_names)} train files, {len(val_blob_names)} val files.")

    # 5) Train
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = train_model(
        model_save_path=model_save_path,
        blob_service_client=blob_service_client,
        train_blob_names=train_blob_names,
        val_blob_names=val_blob_names,
        epochs=epoch_amount,       # from your global
        chunk_size=chunk_size,     # how many .pth files to load at once
        batch_size=training_batch_size
    )

    # 6) Plot
    plot_loss_and_accuracy(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, plot_save_path)

if __name__ == '__main__':
    main()