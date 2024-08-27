import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score

# Define PASCAL VOC classes and mapping to indices
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class_to_idx = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

class CustomOPN(nn.Module):
    def __init__(self):
        super(CustomOPN, self).__init__()

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
        return x

# Load pre-trained COPN model with feature extraction layers only
def load_pretrained_copn():
    model = CustomOPN()
    # No weight loading, using random initialization
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
copn_model = load_pretrained_copn()
model = COPN_FineTune(copn_model, num_classes=20)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

# Training Function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'Epoch [{epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluation Function
def evaluate(model, val_loader):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            all_outputs.append(outputs)
            all_targets.append(targets)
    # Calculate mAP
    mAP = compute_mAP(all_outputs, all_targets)
    print(f'mAP: {mAP:.4f}')

def compute_mAP(outputs, targets):
    """
    Compute mean Average Precision (mAP) for multi-label classification.

    Parameters:
    outputs (list of torch.Tensor): List of model outputs for the validation set.
    targets (list of torch.Tensor): List of true labels for the validation set.

    Returns:
    float: The computed mAP score.
    """
    # Convert lists to numpy arrays
    outputs = [output.cpu().numpy() for output in outputs]
    targets = [target.cpu().numpy() for target in targets]

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

if __name__ == '__main__':
    # Training Loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        evaluate(model, val_loader)