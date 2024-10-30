import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
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
from samples import show_samples

dataset_loc = './dataset_1/'


torch.manual_seed(42)

train_loader, test_loader = get_loaders(dataset_loc, 0.3, 224, 32)
X_train, y_train = get_data(train_loader)
X_test, y_test = get_data(test_loader)

# Only needed for parameter testing - Comment out if training only
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# show_samples(X, y, 3)

# Instantiate model
num_classes = 3
resnet_18 = models.resnet18(weights=None)
resnet_18.fc = nn.Linear(resnet_18.fc.in_features, num_classes)

# Visualize model summary
# summary(resnet_18, input_size=(3, 224, 224))

net = NeuralNetClassifier(
    resnet_18,      # Replace with model being tested
    max_epochs=50,
    lr=0.001,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,   # Already tried SGD and Adam variations
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # callbacks=[EarlyStopping(patience=20)]  # high patience since data is complex
)

param_grid = {
    'lr': [0.0001, 0.001, 0.01,],
    'max_epochs': [25, 50],
    'batch_size': [32, 64],
    'optimizer': [optim.Adam]
}

grid_search = GridSearchCV(net, param_grid, refit=True, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)



