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

show_samples(X, y, 3)


