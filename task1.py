import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary

# Import functions
from data_loader import get_loaders, get_data
from model_eval import test_model
from model_train import train_model
from samples import show_samples

dataset_loc = './data/dataset_1/'
num_classes = 3
class_names = ['MUS', 'NORM', 'STR']

torch.manual_seed(42)

train_loader, valid_loader, test_loader = get_loaders(dataset_loc,224, 18)

# Initialize model
resnet_18 = models.resnet18(weights=None)
resnet_18.fc = nn.Linear(resnet_18.fc.in_features, num_classes)

# Visualize model summary
# summary(resnet_18, input_size=(3, 224, 224))

# Train model
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(resnet_18.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

# train_model(resnet_18, train_loader, valid_loader, criterion, optimizer, max_epoch=50, device='cpu', class_names=class_names)

# Test model
resnet_18.load_state_dict(torch.load('best_weights.pth'))
test_model(resnet_18, test_loader, device='cpu', class_names=class_names)








