import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import CustomOPN
from preprocessed_temporal_four_data_class import PreprocessedTemporalFourData
from prepared_dataset_class import PreparedDataset
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import psutil

## DEFINE GLOBAL VARIABLES
epoch_amount = 100 ##TODO make it 17000
batch_size = 32
num_workers = 0

def print_memory_usage(stage):
    memory = psutil.virtual_memory()
    print(f"[{stage}] Total Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"[{stage}] Available Memory: {memory.available / (1024 ** 3):.2f} GB")
    print(f"[{stage}] Memory Usage: {memory.percent}%")

def main():
    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Access paths from configuration
    dataset_train_path = config['dataset_train_path']
    model_save_path = config['model_save_path']
    plot_save_path = config['plot_save_path']

    print("Loading temporal four dataset")
    dataset = torch.load(dataset_train_path)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history  = train_model(train_dataset, val_dataset, model_save_path)
    plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, plot_save_path)

def train_model(train_dataset, val_dataset, model_save_path):
    #Initialize the model
    model = CustomOPN()
    criterion = nn.CrossEntropyLoss()

    #Setting optimizer and scheduler
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1) #TODO make it milestones=[130000, 170000]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Number of batches in train_ loader: {len(train_loader)}')
    
    # Initialize lists to store loss and accuracy at each epoch
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    print('Starting Training')

    total_time = 0

    for epoch in range(epoch_amount):
        #print_memory_usage(f"Before Epoch {epoch + 1}")
        # Measure training time for 1 epoch
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader)
        val_loss, val_acc = validate_model(model, criterion, val_loader)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time
        print(f"Time for 1 epoch: {epoch_time:.2f} seconds")
        #print_memory_usage(f"After Epoch {epoch + 1}")

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_acc)

        print(f'Epoch {epoch+1}/{epoch_amount}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        # TODO Every X (e.g. 100) iterations, save a snapshot of the model
        # if epoch % 100 == 0:
        #     torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')
        scheduler.step()

        # Print total time for every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Total time for {epoch + 1} epochs: {total_time:.2f} seconds")
    
    torch.save(model.state_dict(), model_save_path)
    return model, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, frame_order_labels, *rest in train_loader:
        inputs = inputs.to(torch.float32)  # Convert inputs to float
        inputs = inputs.permute(0, 1, 4, 2, 3)  # Change the order of dimensions to (batch_size, frames, channels, height, width)
        inputs = inputs.contiguous().view(-1, inputs.shape[1]*inputs.shape[2], inputs.shape[3], inputs.shape[4])  # Concatenate frames and channels
        
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
        loss = criterion(outputs, frame_order_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predicted_labels == frame_order_labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

def validate_model(model, criterion, val_loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, frame_order_labels, *rest in val_loader:
            inputs = inputs.to(torch.float32) # Convert inputs to float
            inputs = inputs.permute(0, 1, 4, 2, 3)  # Change the order of dimensions to (batch_size, frames, channels, height, width)
            inputs = inputs.contiguous().view(-1, inputs.shape[1]*inputs.shape[2], inputs.shape[3], inputs.shape[4])  # Concatenate frames and channels

            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)
            loss = criterion(outputs, frame_order_labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted_labels == frame_order_labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    return epoch_loss, epoch_acc

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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

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

def visualize_network():
    from torchviz import make_dot

    # Assuming `inputs` is a batch of your input data
    inputs = torch.randn(1, 4, 80, 80, 3)  # Adjust the size as needed

    model = CustomOPN()

    # Permute the dimensions to (batch_size, frames, channels, height, width)
    inputs = inputs.permute(0, 1, 4, 2, 3)

    # Flatten the frames and channels dimensions
    inputs = inputs.contiguous().view(-1, inputs.shape[1]*inputs.shape[2], inputs.shape[3], inputs.shape[4])

    # Ensure there is more than one value per channel
    if inputs.shape[0] == 1:
        inputs = inputs.repeat(2, 1, 1, 1)

    outputs = model(inputs)

    # Create the graph
    dot = make_dot(outputs, params=dict(model.named_parameters()))

    # Display the graph
    dot.view()

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def show_dataset_images(dataset, num_images=5):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        # Assuming each item in dataset gives a batch of images
        img_tensor = dataset[i][0]  # This needs to be the tensor containing the image data

        # Check the number of dimensions and rearrange if necessary
        if img_tensor.ndim == 4 and img_tensor.shape[-1] in {1, 3}:
            img_tensor = img_tensor.permute(0, 3, 1, 2)  # Rearrange from NHWC to NCHW if needed
            img_tensor = img_tensor[0]  # Take the first image of the batch

        # Handle single channel images (grayscale)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)  # Remove channel dimension if it's grayscale

        img = TF.to_pil_image(img_tensor)
        axs[i].imshow(img, cmap='gray' if img_tensor.shape[0] == 1 else None)
        axs[i].axis('off')
    plt.show()

if __name__ == '__main__':
    main()