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
PREPROCESSEDDATA_FOLDERNAME = "ucf-preprocessed-data-1000"
## Define Global Variables
epoch_amount = 5 ##TODO make it 17000
chunk_size = 4
training_batch_size = 32
num_workers = 1

# 4. CLASSES
class BlobDataset(Dataset):
    def __init__(self, blob_service_client, container_name, blob_names):
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.blob_names = blob_names
        self.datasets = []
        
        # Load all batch files
        for blob_name in blob_names:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            downloaded_blob = blob_client.download_blob().readall()
            buffer = io.BytesIO(downloaded_blob)
            buffer.seek(0)
            self.datasets.append(torch.load(buffer))
            print(f"    Loaded {blob_name}")

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of range")
    
# 5. HELPER FUNCTIONS
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
        # For the first batch, skip timing calculation
        if batch_start_time is not None:
            batch_end_time = time.time()
            batch_processing_time = batch_end_time - batch_start_time
            print(f"Batch {batch_idx}/{len(train_loader)} processed in {batch_processing_time:.2f} seconds")
        else:
            print(f"Batch {batch_idx}/{len(train_loader)} processed (timing starts from next batch)")

        # Reset the timer for the next batch
        batch_start_time = time.time()

        # Preprocessing
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

        # Calculate and print batch processing time
        #batch_end_time = time.time()
        #batch_processing_time = batch_end_time - batch_start_time
        #print(f"Batch {batch_idx + 1}/{len(train_loader)} processed in {batch_processing_time:.2f} seconds")

    # After the loop ends, print a final newline so it doesn't run onto the same line
    print()  

    chunk_loss = running_loss / total_samples
    chunk_acc = running_corrects / total_samples
    return chunk_loss, running_corrects, total_samples

def validate_in_chunks(model, blob_service_client, val_blob_names, batch_size, chunk_size):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # Loop through blob_chunks
    for blob_chunk_idx, blob_chunk in enumerate(chunkify(val_blob_names, chunk_size), start=1):
        print(f"Validating on blob_chunk {blob_chunk_idx} with {len(blob_chunk)} blob_batches(s)...")

        # Track the time to load the blob chunk
        blob_chunk_start_time = time.time()
        
        # Create dataset for this blob chunk
        val_dataset_blob_chunk = BlobDataset(blob_service_client, CONTAINERNAME, blob_chunk)
        val_loader = DataLoader(val_dataset_blob_chunk, batch_size=batch_size, shuffle=False)

        blob_chunk_end_time = time.time()
        blob_chunk_loading_time = blob_chunk_end_time - blob_chunk_start_time
        print(f"    Blob chunk {blob_chunk_idx} loaded in {blob_chunk_loading_time:.2f} seconds")

        # Perform validation on the current blob chunk
        with torch.no_grad():
            for inputs, frame_order_labels, *rest in val_loader:
                inputs = inputs.float().permute(0, 1, 4, 2, 3).contiguous()
                inputs = inputs.view(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3], inputs.shape[4])

                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, frame_order_labels)

                # Accuracy
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == frame_order_labels).sum().item()
                total_samples += inputs.size(0)

        # Cleanup after processing the blob chunk
        del val_dataset_blob_chunk
        del val_loader
        torch.cuda.empty_cache()  # or gc.collect() if CPU

    # Compute validation metrics
    val_loss = running_loss / total_samples
    val_acc = running_corrects / total_samples
    return val_loss, val_acc

def plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, plot_save_path):
    # After training, plot loss and accuracy
    epochs = range(1, epoch_amount + 1)

    plt.figure(figsize=(12, 12))
    
    # Plotting Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_history, label='Training Loss')
    # Training Loss Trend Line
    m_train_loss, b_train_loss = np.polyfit(epochs, train_loss_history, 1)
    plt.plot(epochs, m_train_loss*np.array(epochs) + b_train_loss, 'r--', label='Train Loss Trend Line')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Training Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracy_history, label='Training Accuracy')
    # Training Accuracy Trend Line
    m_train_acc, b_train_acc = np.polyfit(epochs, train_accuracy_history, 1)
    plt.plot(epochs, m_train_acc*np.array(epochs) + b_train_acc, 'r--', label='Train Acc Trend Line')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Validation Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_loss_history, label='Validation Loss')
    # Validation Loss Trend Line
    m_val_loss, b_val_loss = np.polyfit(epochs, val_loss_history, 1)
    plt.plot(epochs, m_val_loss*np.array(epochs) + b_val_loss, 'r--', label='Val Loss Trend Line')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Validation Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_accuracy_history, label='Validation Accuracy')
    # Validation Accuracy Trend Line
    m_val_acc, b_val_acc = np.polyfit(epochs, val_accuracy_history, 1)
    plt.plot(epochs, m_val_acc*np.array(epochs) + b_val_acc, 'r--', label='Val Acc Trend Line')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Calculate and print the correlation coefficients
    corr_train_loss = np.corrcoef(epochs, train_loss_history)[0, 1]
    corr_val_loss = np.corrcoef(epochs, val_loss_history)[0, 1]
    corr_train_acc = np.corrcoef(epochs, train_accuracy_history)[0, 1]
    corr_val_acc = np.corrcoef(epochs, val_accuracy_history)[0, 1]
    print(f'Training Loss correlation coefficient: {corr_train_loss:.2f}')
    print(f'Validation Loss correlation coefficient: {corr_val_loss:.2f}')
    print(f'Training Accuracy correlation coefficient: {corr_train_acc:.2f}')
    print(f'Validation Accuracy correlation coefficient: {corr_val_acc:.2f}') 

    # Save the plot before showing
    plt.savefig(plot_save_path)

    plt.show()

def evaluate_model(test_dataset):
    # Load the trained model
    model_path = f'model_epoch_{epoch_amount-1}.pt'  # replace with your model path
    model = CustomOPN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a DataLoader for the test data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_batch_size, shuffle=False)

    # Initialize the running accuracy
    running_corrects = 0

    # Iterate over the test data
    for inputs, frame_order_labels, action_labels, video_name, _ in test_loader:
        # Convert inputs to float
        inputs = inputs.to(torch.float32)

        # Change the order of dimensions to (batch_size, frames, channels, height, width)
        inputs = inputs.permute(0, 1, 4, 2, 3)
        # Concatenate frames and channels
        inputs = inputs.contiguous().view(-1, inputs.shape[1]*inputs.shape[2], inputs.shape[3], inputs.shape[4])

        # Make predictions
        outputs = model(inputs)

        # Get predicted class labels
        _, predicted_labels = torch.max(outputs, 1)

        # Update the running accuracy
        running_corrects += torch.sum(predicted_labels == frame_order_labels.data)

    # Calculate the final accuracy
    accuracy = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy}')

# 6. TRAINING PIPELINE
def train_model(model_save_path, 
                blob_service_client, 
                train_blob_names, 
                val_blob_names, 
                epochs=5, 
                chunk_size=2, 
                batch_size=32):

    # 1) Instantiate the model and define criterion, optimizer, etc.
    model = CustomOPN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    # We'll store the metrics each epoch
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    print("\nStarting Training Loop...")

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        # (A) Shuffle the training file list each epoch for randomness
        train_files_shuffled = train_blob_names[:]
        random.shuffle(train_files_shuffled)

        # Track epoch-level metrics
        epoch_train_loss = 0.0
        epoch_train_corrects = 0
        epoch_train_samples = 0

        # (B) Now chunkify the shuffled list
        for blob_chunk_idx, blob_chunk in enumerate(chunkify(train_files_shuffled, chunk_size), start=1):
            print(f"  Loading blob_chunk {blob_chunk_idx} with {len(blob_chunk)} blob_batch(es)...")

            # Track the time to load the blob chunk
            blob_chunk_load_start_time = time.time()
            
            # Create a dataset for the blob chunk
            dataset_blob_chunk = BlobDataset(blob_service_client, CONTAINERNAME, blob_chunk)
            blob_chunk_load_end_time = time.time()
            blob_chunk_loading_time = blob_chunk_load_end_time - blob_chunk_load_start_time
            print(f"  Blob_chunk {blob_chunk_idx} loaded in {blob_chunk_loading_time:.2f} seconds")

            # Create a DataLoader from the blob chunk dataset
            train_loader = DataLoader(dataset_blob_chunk, batch_size=batch_size, shuffle=True)
             # Track the time to train the blob chunk
            blob_chunk_train_start_time = time.time()
            
            blob_chunk_train_end_time = time.time()
            # Train on this blob chunk
            blob_chunk_loss, blob_chunk_corrects, blob_chunk_samples = train_one_chunk(
                model, criterion, optimizer, train_loader
            )
            blob_chunk_train_time = blob_chunk_train_end_time - blob_chunk_train_start_time
            print(f"  Blob_chunk {blob_chunk_idx} trained in {blob_chunk_train_time:.2f} seconds")

            # Accumulate blob chunk metrics into epoch totals
            epoch_train_loss += blob_chunk_loss * blob_chunk_samples
            epoch_train_corrects += blob_chunk_corrects
            epoch_train_samples += blob_chunk_samples


            # Discard blob batch data from memory
            del dataset_blob_batch
            del train_loader
            torch.cuda.empty_cache()  # or gc.collect() if CPU

        # (D) Compute final epoch stats
        epoch_train_loss = epoch_train_loss / epoch_train_samples
        epoch_train_acc = epoch_train_corrects / epoch_train_samples

        # Now do validation
        val_loss, val_acc = validate_in_chunks(model, blob_service_client, val_blob_names, batch_size, chunk_size)

        # Log
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1} => Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), model_save_path)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history

# 7. MAIN EXECUTION 
def main():
    # 1. Load config
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_save_path = config['model_save_path']
    plot_save_path = config['plot_save_path']

    # 2. Initialize BlobServiceClient
    blob_service_client = BlobServiceClient(
        account_url=STORAGEACCOUNTURL,
        credential=STORAGEACCOUNTKEY
    )
    container_client = blob_service_client.get_container_client(CONTAINERNAME)

    # 3. List .pth blobs
    blob_list = container_client.list_blobs(name_starts_with=PREPROCESSEDDATA_FOLDERNAME + '/')
    all_blob_names = [blob.name for blob in blob_list if blob.name.endswith('.pth')]
    if not all_blob_names:
        print("No preprocessed data files found in the blob storage.")
        return

    print(f"Found {len(all_blob_names)} blob batch .pth files in the container.")

    # 4. Split into train/val
    train_blob_names, val_blob_names = train_test_split(all_blob_names, test_size=0.2, random_state=42)
    print(f"{len(train_blob_names)} train files, {len(val_blob_names)} val files.")

    # 5. Train the model using chunked approach
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = train_model(
        model_save_path=model_save_path,
        blob_service_client=blob_service_client,
        train_blob_names=train_blob_names,
        val_blob_names=val_blob_names,
        epochs=epoch_amount,       # from your global variable
        chunk_size=chunk_size,     # tweak as needed for memory
        batch_size=training_batch_size
    )

    # 6. Plot training curves
    plot_loss_and_accuracy(
        train_loss_hist,
        train_acc_hist,
        val_loss_hist,
        val_acc_hist,
        plot_save_path
    )
if __name__ == '__main__':
    main()