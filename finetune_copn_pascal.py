import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import json

# Define PASCAL VOC classes and mapping to indices
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class_to_idx = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# Pruned CustomOPN model without pairwise feature extraction and order prediction
class CustomOPNPruned(nn.Module):
    def __init__(self):
        super(CustomOPNPruned, self).__init__()

        # 1. FEATURE EXTRACTION
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        #Pairwise Feature Extraction and Order Prediction layers from the original CustomOPN are removed

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x
    
# Load pre-trained COPN model with feature extraction layers only
def load_pretrained_copn(pretrained_model_path='./saved/model_weights.pth'):
    model = CustomOPNPruned()  # Use pruned version of the model
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path)
        
        # Handle the input channel mismatch for the first conv layer (12 channels vs 3 channels)
        pretrained_conv1_weight = state_dict['conv_layers.0.weight']  # Shape: (96, 12, 11, 11)
        
        # Average the weights over the 4 frames (each with 3 channels)
        averaged_conv1_weight = pretrained_conv1_weight.view(96, 4, 3, 11, 11).mean(1)  # Reduce channels to 3
        
        # Update the state dict with the averaged weights for the first layer
        state_dict['conv_layers.0.weight'] = averaged_conv1_weight
        
        # Load the updated state_dict into the model, allowing only matching layers to be loaded
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"No pre-trained model found at {pretrained_model_path}, using random initialization")
    return model

# Define the fine-tuning model
class COPN_FineTune(nn.Module):
    def __init__(self, copn_model, num_classes):
        super(COPN_FineTune, self).__init__()
        self.features = copn_model.conv_layers  # Use pre-trained layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),  # Adjust based on the output size of your conv_layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def custom_collate_fn(batch):
    """
    Custom collate function to process a batch of data from the VOCDetection dataset.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]

    # Process targets to be in the correct format
    # Assuming that 'annotation' -> 'object' contains the necessary labels
    targets_tensor = []
    for target in targets:
        target_labels = torch.zeros(20)  # Assuming there are 20 classes
        objects = target['annotation']['object']
        if isinstance(objects, list):
            for obj in objects:
                class_name = obj['name']
                class_idx = class_to_idx[class_name]
                target_labels[class_idx] = 1
        else:
            class_name = objects['name']
            class_idx = class_to_idx[class_name]
            target_labels[class_idx] = 1
        targets_tensor.append(target_labels)
    
    targets_tensor = torch.stack(targets_tensor)

    return images, targets_tensor

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = 'data/VOCdevkit'
if not os.path.exists(data_dir):
    train_dataset = datasets.VOCDetection(root='data/VOCdevkit', year='2007', image_set='trainval', download=True, transform=transform)
    val_dataset = datasets.VOCDetection(root='data/VOCdevkit', year='2007', image_set='val', download=True, transform=transform)
else:
    train_dataset = datasets.VOCDetection(root='data/VOCdevkit', year='2007', image_set='trainval', transform=transform)
    val_dataset = datasets.VOCDetection(root='data/VOCdevkit', year='2007', image_set='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=3, collate_fn=custom_collate_fn)

# Initialize model with transferred features and new classifier layers
copn_model = load_pretrained_copn()  # Load the pruned model with pre-trained weights
model = COPN_FineTune(copn_model, num_classes=20)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

# Training Function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    all_outputs = []
    all_targets = []
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Collecting outputs and targets for AP computation
        all_outputs.append(outputs.detach())
        all_targets.append(targets)

        #if i % 10 == 0:
        #    print(f'Epoch [{epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Calculate Average Precision (AP) for training set
    train_mAP = compute_mAP(all_outputs, all_targets)
    #print(f'Epoch [{epoch}] Training mAP: {train_mAP:.4f}')
    return train_mAP

# Evaluation Function
def evaluate(model, val_loader):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            all_outputs.append(outputs.detach())
            all_targets.append(targets)
    
    # Calculate mAP for validation set
    val_mAP = compute_mAP(all_outputs, all_targets)
    #print(f'Validation mAP: {val_mAP:.4f}')
    return val_mAP

def compute_mAP(outputs, targets):
    """
    Compute mean Average Precision (mAP) for multi-label classification.

    Parameters:
    outputs (list of torch.Tensor): List of model outputs for the validation set.
    targets (list of torch.Tensor): List of true labels for the validation set.

    Returns:
    float: The computed mAP score.
    """
    # Convert lists to numpy arrays after detaching from the computation graph
    outputs = [output.detach().cpu().numpy() for output in outputs]
    targets = [target.detach().cpu().numpy() for target in targets]

    # Concatenate all predictions and true labels
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)

    num_classes = targets.shape[1]
    average_precisions = []

    for class_idx in range(num_classes):
        # Get the ground truth and predictions for the current class
        true_labels = targets[:, class_idx]
        scores = outputs[:, class_idx]

        # Compute Average Precision (AP)
        average_precision = average_precision_score(true_labels, scores)
        average_precisions.append(average_precision)

    # Compute mean Average Precision (mAP)
    mAP = np.mean(average_precisions)
    return mAP

# Save mAP data
def save_map_data(train_mAPs, val_mAPs, path='./saved/map_data.json'):
    data = {
        'train_mAPs': train_mAPs,
        'val_mAPs': val_mAPs
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"mAP data saved to {path}")

# Visualization function
def save_plot_metrics(train_mAPs, val_mAPs):
    plt.figure(figsize=(10, 5))
    plt.plot(train_mAPs, label='Training mAP', marker='o')
    plt.plot(val_mAPs, label='Validation mAP', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Training and Validation mAP over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('./saved/mAP_plot.png')  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

if __name__ == '__main__':
    # Training Loop
    print("Beginning finetuning and evaluation on pretrained model")
    num_epochs = 1
    train_mAPs = []
    val_mAPs = []

    for epoch in range(num_epochs):
        train_mAP = train(model, train_loader, criterion, optimizer, epoch)
        val_mAP = evaluate(model, val_loader)
        
        print(f'Epoch {epoch}: train_mAP = {train_mAP}, val_mAP = {val_mAP}')  # Debugging line

        train_mAPs.append(train_mAP)
        val_mAPs.append(val_mAP)

    # Save the mAP data to a JSON file
    save_map_data(train_mAPs, val_mAPs, path='./saved/map_data_with_pretraining.json')
    save_plot_metrics(train_mAPs, val_mAPs)

    print("Training completed!")