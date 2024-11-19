import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


# Best parameter search - Libraries
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV

# Import functions
from data_loader import get_loaders, get_data


# Get data

dataset_loc = '../data/1/'
num_classes = 3

torch.manual_seed(42)

train_loader, valid_loader, test_loader = get_loaders(dataset_loc,224, 18)
X_train, y_train = get_data(train_loader)
X_valid, y_valid = get_data(valid_loader)

# Combine data
X_combined = np.concatenate((X_train, X_valid), axis=0)
y_combined = np.concatenate((y_train, y_valid), axis=0)

# Initialize model
resnet_18 = models.resnet18(weights=None)
resnet_18.fc = nn.Linear(resnet_18.fc.in_features, num_classes)

# Initialize net

net = NeuralNetClassifier(
    resnet_18,
    max_epochs=50,
    lr=0.001,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[EarlyStopping(patience=20)]
)

param_grid = {
    'lr': [0.0001, 0.001, 0.01],
    'max_epochs': [25, 30, 50],
    'batch_size': [18, 32],
    'optimizer': [optim.Adam, optim.SGD],
    'optimizer__weight_decay': [0.01, 0.1],
    'optimizer__momentum': [0.9, 0.95],
    'optimizer__nesterov': [True, False]
}

random_search = RandomizedSearchCV(net, param_grid, n_iter=10, refit=True, cv=3, scoring='accuracy', verbose=2)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)