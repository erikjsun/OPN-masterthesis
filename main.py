import os
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import CustomOPN
from data_prep import PreparedDataset, PreprocessedTemporalFourData
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
from azure.storage.blob import BlobServiceClient

# Azure Blob Storage configuration
STORAGEACCOUNTURL = "https://exjobbssl1863219591.blob.core.windows.net"
STORAGEACCOUNTKEY = "PuL1QY8bQvIyGi653lr/9CPvyHLnip+cvsu62YAipDjB7onPDxfME156z5/O2NwY0PRLMTZc86/6+ASt5Vts8w=="
CONTAINERNAME = "exjobbssl"
PREPROCESSEDDATA_FOLDERNAME = "preprocessed-data"

# Define global variables
epoch_amount = 100  # Adjust as needed
batch_size = 32
num_workers = 2

# Initialize Azure Blob Storage client
blob_service_client = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
container_client = blob_service_client.get_container_client(CONTAINERNAME)

def main():
    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_save_path = config['model_save_path']
    plot_save_path = config['plot_save_path']

    # Ensure save directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    # Define a save interval (e.g., save every 10 epochs)
    save_interval = 10

    # Start training
    model, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train_model(
        model_save_path, save_interval
    )

    # Plot training performance
    plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, plot_save_path)

def train_model(model_save_path, save_interval):
    # Initialize the model
    model = CustomOPN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    # Initialize lists to store loss and accuracy at each epoch
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    print('Starting Training')

    total_time = 0

    # Create 'modelsnapshots' subfolder in 'saved'
    snapshot_dir = os.path.join('saved', 'modelsnapshots')
    os.makedirs(snapshot_dir, exist_ok=True)

    # List all preprocessed data blobs
    all_blob_list = list(container_client.list_blobs(name_starts_with=f"{PREPROCESSEDDATA_FOLDERNAME}/"))
    preprocessed_blobs = [blob for blob in all_blob_list if 'preprocessed_batch' in blob.name]
    print(f"Found {len(preprocessed_blobs)} preprocessed data blobs.")

    if not preprocessed_blobs:
        print("No preprocessed data blobs found. Please check your Azure Blob Storage path and blob names.")
        return None, [], [], [], []

    # Shuffle and split blobs into training and validation sets
    np.random.shuffle(preprocessed_blobs)
    split_index = int(0.8 * len(preprocessed_blobs))
    train_blobs = preprocessed_blobs[:split_index]
    val_blobs = preprocessed_blobs[split_index:]

    for epoch in range(epoch_amount):
        epoch_start_time = time.time()

        # Train on all batches for the current epoch
        train_loss, train_acc = train_on_batches(model, criterion, optimizer, train_blobs)
        val_loss, val_acc = validate_on_batches(model, criterion, val_blobs)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time

        print(f"Time for 1 epoch: {epoch_time:.2f} seconds")

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_acc)

        print(f'Epoch {epoch + 1}/{epoch_amount}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        # Save model snapshot every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            snapshot_path = os.path.join(snapshot_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), snapshot_path)
            print(f"Model saved at {snapshot_path}")

        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved at {model_save_path}")

    return model, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

def train_on_batches(model, criterion, optimizer, train_blobs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for blob in train_blobs:
        print(f"Processing training blob: {blob.name}")
        blob_client = container_client.get_blob_client(blob.name)
        downloaded_blob = blob_client.download_blob().readall()
        buffer = io.BytesIO(downloaded_blob)
        buffer.seek(0)
        preprocessed_data = torch.load(buffer)

        # Create a DataLoader for the preprocessed data
        train_loader = DataLoader(preprocessed_data, batch_size=1, shuffle=True, num_workers=num_workers)

        for batch_data in train_loader:
            # Each batch_data is a preprocessed batch
            preprocessed_batch = batch_data[0]  # Extract the batch from the list

            inputs_list = []
            labels_list = []

            for data in preprocessed_batch:
                (
                    preprocessed_frames,
                    frame_order_label,
                    action_label,
                    video_name,
                    frames_canonical_order,
                    selected_frames,
                    preprocessed_frames_coordinates,
                    ordered_frames,
                ) = data

                inputs_list.append(preprocessed_frames)
                labels_list.append(frame_order_label)

            # Stack inputs and labels
            inputs = torch.stack(inputs_list)
            frame_order_labels = torch.stack(labels_list)

            # Proceed with the training steps
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)
            loss = criterion(outputs, frame_order_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted_labels == frame_order_labels.data)
            total_samples += inputs.size(0)

        # Free up memory
        del preprocessed_data, buffer, train_loader
        torch.cuda.empty_cache()

    if total_samples > 0:
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
    else:
        epoch_loss = 0
        epoch_acc = 0
        print("Warning: No data processed in this epoch.")

    return epoch_loss, epoch_acc

def validate_on_batches(model, criterion, val_blobs):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    print(f"Validating on {len(val_blobs)} validation blobs.")

    if not val_blobs:
        print("No validation blobs found.")
        return 0, 0  # Return zeros to prevent division by zero

    with torch.no_grad():
        for blob in val_blobs:
            print(f"Processing validation blob: {blob.name}")
            blob_client = container_client.get_blob_client(blob.name)

            # Download the preprocessed batch as bytes
            downloaded_blob = blob_client.download_blob().readall()

            # Load the preprocessed data from the downloaded bytes
            buffer = io.BytesIO(downloaded_blob)
            buffer.seek(0)
            preprocessed_data = torch.load(buffer)

            # Create a DataLoader from the preprocessed data
            val_loader = DataLoader(preprocessed_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            for inputs, frame_order_labels, *rest in val_loader:
                inputs = inputs.to(torch.float32)
                inputs = inputs.permute(0, 1, 4, 2, 3)
                inputs = inputs.contiguous().view(
                    -1, inputs.shape[1] * inputs.shape[2], inputs.shape[3], inputs.shape[4]
                )

                outputs = model(inputs)
                _, predicted_labels = torch.max(outputs, 1)
                loss = criterion(outputs, frame_order_labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicted_labels == frame_order_labels.data)
                total_samples += inputs.size(0)

            # Free up memory
            del preprocessed_data, buffer, val_loader
            torch.cuda.empty_cache()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc

def plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, plot_save_path):
    epochs = range(1, epoch_amount + 1)

    plt.figure(figsize=(12, 12))

    # Plotting Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_history, label='Training Loss')
    m_train_loss, b_train_loss = np.polyfit(epochs, train_loss_history, 1)
    plt.plot(epochs, m_train_loss * np.array(epochs) + b_train_loss, 'r--', label='Train Loss Trend Line')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Training Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracy_history, label='Training Accuracy')
    m_train_acc, b_train_acc = np.polyfit(epochs, train_accuracy_history, 1)
    plt.plot(epochs, m_train_acc * np.array(epochs) + b_train_acc, 'r--', label='Train Acc Trend Line')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Validation Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_loss_history, label='Validation Loss')
    m_val_loss, b_val_loss = np.polyfit(epochs, val_loss_history, 1)
    plt.plot(epochs, m_val_loss * np.array(epochs) + b_val_loss, 'r--', label='Val Loss Trend Line')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Validation Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_accuracy_history, label='Validation Accuracy')
    m_val_acc, b_val_acc = np.polyfit(epochs, val_accuracy_history, 1)
    plt.plot(epochs, m_val_acc * np.array(epochs) + b_val_acc, 'r--', label='Val Acc Trend Line')
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
    model_path = f'model_epoch_{epoch_amount}.pt'  # Adjust as needed
    model = CustomOPN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a DataLoader for the test data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    running_corrects = 0

    with torch.no_grad():
        for inputs, frame_order_labels, *rest in test_loader:
            inputs = inputs.to(torch.float32)
            inputs = inputs.permute(0, 1, 4, 2, 3)
            inputs = inputs.contiguous().view(
                -1, inputs.shape[1] * inputs.shape[2], inputs.shape[3], inputs.shape[4]
            )

            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)
            running_corrects += torch.sum(predicted_labels == frame_order_labels.data)

    accuracy = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()