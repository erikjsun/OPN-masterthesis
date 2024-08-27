import os
import json
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

# Define the modified CustomOPN model
class CustomOPN(nn.Module):
    def __init__(self, input_channels=12):
        super(CustomOPN, self).__init__()

        # 1. FEATURE EXTRACTION
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=0),
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

        # 2. PAIRWISE FEATURE EXTRACTION
        self.fc6 = nn.Linear(256 * 3 * 3, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)

        self.fc7_layers = nn.ModuleList()
        for i in range(6):
            self.fc7_layers.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
            ))

        # 3. ORDER PREDICTION
        self.fc8 = nn.Linear(512 * 6, 12)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x_concat = [torch.cat((x1, x2), dim=1), torch.cat((x2, x3), dim=1), torch.cat((x3, x4), dim=1),
                    torch.cat((x1, x3), dim=1), torch.cat((x2, x4), dim=1), torch.cat((x1, x4), dim=1)]
        
        out = []
        for i in range(6):
            out_i = self.fc7_layers[i](x_concat[i])
            out.append(out_i)

        out = torch.cat(out, dim=1)
        out = self.fc8(out)
        return out

# Load pre-trained model and adjust first layer
def load_pretrained_copn(path):
    model = CustomOPN(input_channels=12)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    
    # Adjust first layer to handle 3 channels instead of 12
    new_first_layer = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
    with torch.no_grad():
        new_first_layer.weight[:, :3, :, :] = model.conv_layers[0].weight[:, :3, :, :]
        new_first_layer.weight[:, 3:, :, :].zero_()  # Zero out the remaining channels
    model.conv_layers[0] = new_first_layer

    return model

# Define the fine-tuning model
class COPN_FineTune(nn.Module):
    def __init__(self, copn_model, num_classes):
        super(COPN_FineTune, self).__init__()
        self.features = copn_model.conv_layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),  # Adjust based on the output size of your conv_layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
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
    
# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def fine_tune_model(config_path):
    # Load configuration
    config = load_config(config_path)
    pretrained_path = config['model_save_path']  # Path to the saved pre-trained model

    # Load the pre-trained model
    copn_model = load_pretrained_copn(pretrained_path)
    model = COPN_FineTune(copn_model, num_classes=20)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Training function
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

    # Evaluation function
    def evaluate(model, val_loader):
        model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images)
                all_outputs.append(outputs)
                all_targets.append(targets)
        mAP = compute_mAP(all_outputs, all_targets)
        print(f'mAP: {mAP:.4f}')

    # Compute mean Average Precision (mAP)
    def compute_mAP(outputs, targets):
        outputs = [output.cpu().numpy() for output in outputs]
        targets = [target.cpu().numpy() for target in targets]
        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        num_classes = targets.shape[1]
        average_precisions = []
        for class_idx in range(num_classes):
            true_labels = targets[:, class_idx]
            scores = outputs[:, class_idx]
            average_precision = average_precision_score(true_labels, scores)
            average_precisions.append(average_precision)
        mAP = np.mean(average_precisions)
        return mAP

    # Fine-tune the model
    num_epochs = 5
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        evaluate(model, val_loader)

# Adjust the dataset and dataloader part
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

if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    config_path = 'path/to/your/config.json'  # Update this path
    fine_tune_model(config_path)