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

## DEFINE GLOBAL VARIABLES
epoch_amount = 50 ##TODO make it 17000

def main():
    print("Loading temporal four dataset")
    train_dataset = torch.load('dataset_train.pth')
    #test_dataset = torch.load('dataset_test.pth')
    model, loss_history, accuracy_history = train_model(train_dataset)
    plot_loss_and_accuracy(loss_history, accuracy_history)

def train_model(train_dataset):
    #Initialize the model
    model = CustomOPN()
    criterion = nn.CrossEntropyLoss()

    #Setting optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1) #TODO make it milestones=[130000, 170000]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)

    print(f'Number of batches in train_loader: {len(train_loader)}')

    # Assuming validation_data is your validation dataset
    #validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False) TODO add validation data
    
    # Initialize lists to store loss and accuracy at each epoch
    loss_history = []
    accuracy_history = []

    print('Starting Training')
    for epoch in range(epoch_amount):
        # Training phase
        
        model.train()
        running_loss = 0.0
        running_corrects = 0

        #start_time = time.time()
        for inputs, frame_order_labels, *rest in train_loader:
            #end_time = time.time()
            #print(f'Time taken to retrieve elements: {end_time - start_time} seconds') # Expected: ~40 seconds
            # Convert inputs to float
            inputs = inputs.to(torch.float32)

            #TODO check if this is correct
            inputs = inputs.permute(0, 1, 4, 2, 3)  # Change the order of dimensions to (batch_size, frames, channels, height, width)
            inputs = inputs.contiguous().view(-1, inputs.shape[1]*inputs.shape[2], inputs.shape[3], inputs.shape[4])  # Concatenate frames and channels

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            #start_time = time.time()
            outputs = model(inputs)
            #end_time = time.time()
            #print(f'Time taken for forward pass: {end_time - start_time} seconds') #EXTREMELY FAST

            # Get predicted class labels
            _, predicted_labels = torch.max(outputs, 1)
            #print("Predicted labels: ", predicted_labels)
                    
            # Calculate loss
            loss = criterion(outputs, frame_order_labels)

            # Backward pass and optimize
            #start_time = time.time()
            loss.backward()
            #end_time = time.time()
            #print(f'Time taken for backward pass: {end_time - start_time} seconds') #EXTREMELY FAST
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted_labels == frame_order_labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_acc)

        print(f'Epoch {epoch+1}/{epoch_amount}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')
        
        # TODO Every X (e.g. 100) iterations, save a snapshot of the model
        #if epoch % 100 == 0:
            #torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')
        scheduler.step()
    return model, loss_history, accuracy_history

def plot_loss_and_accuracy(loss_history, accuracy_history):
    # After training, plot loss and accuracy
    epochs = range(1, epoch_amount + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # Plot correlation lines
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss')
    m, b = np.polyfit(epochs, loss_history, 1)

def create_validation_dataset():
    #TODO
    pass

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

if __name__ == '__main__':
    main()