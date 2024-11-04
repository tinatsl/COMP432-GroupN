import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorflow.python.layers.core import dropout
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchsummary import summary

# Best parameter search - Libraries
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Import functions
from data_loader import get_loaders, get_data
from model_eval import test_model
from model_train import train_model
from samples import show_samples

dataset_loc = './dataset_1/'


torch.manual_seed(42)

train_loader, valid_loader, test_loader = get_loaders(dataset_loc,224, 18)
# X_train, y_train = get_data(train_loader)
# X_valid, y_valid = get_data(valid_loader)
# X_test, y_test = get_data(test_loader)

# show_samples(X_train, y_train, 3)

# Instantiate model
num_classes = 3
resnet_18 = models.resnet18(weights=None)
resnet_18.fc = nn.Linear(resnet_18.fc.in_features, num_classes)

# Visualize model summary
# summary(resnet_18, input_size=(3, 224, 224))

# Train model

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_18.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
# optimizer = optim.SGD(resnet_18.parameters(), lr=0.001)
# optimizer = optim.Adam(resnet_18.parameters(), lr=0.0001)

# train_model(resnet_18, train_loader, valid_loader, criterion, optimizer, max_epoch=50, device='cpu', patience=20) # 25 epochs seems better with normalized data

# Test model
resnet_18.load_state_dict(torch.load('best_weights.pth'))
test_model(resnet_18, test_loader, device='cpu')

# net = NeuralNetClassifier(
#     resnet_18,      # Replace with model being tested
#     max_epochs=50,
#     lr=0.001,
#     criterion=nn.CrossEntropyLoss,
#     optimizer=optim.Adam,   # Already tried SGD and Adam variations
#     device='cuda' if torch.cuda.is_available() else 'cpu'
#     # callbacks=[EarlyStopping(patience=20)]  # high patience since data is complex
# )
#
# param_grid = {
#     'lr': [0.0001, 0.0005, 0.001],
#     'max_epochs': [25, 30, 50],
#     'batch_size': [32, 64],
#     'optimizer': [optim.Adam, optim.SGD],
#     'optimizer__weight_decay': [0.01, 0.1],
#     'optimizer__momentum': [0.9, 0.95],
#     'optimizer__nesterov': [True, False],
#     'module__dropout': [0.2, 0.3]
# }
#
# random_search = RandomizedSearchCV(net, param_grid, n_iter=10, refit=True, cv=3, scoring='accuracy', verbose=2)
# random_search.fit(X_train, y_train)
#
# print("Best Parameters:", random_search.best_params_)
# print("Best Score:", random_search.best_score_)



