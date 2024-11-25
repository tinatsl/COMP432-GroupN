import torch
import torch.nn as nn
import torchvision.models as models

# Import functions
from data_loader import get_loader
from task2.analyze_model import save_analysis

num_classes = 3
dataset2_loc = './data/2/'
dataset3_loc = './data/2/'
class_names2 = ['GLAND', 'NONGL', 'TU']
class_names3 = ['CAT', 'DOG', 'WILD']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

data_loader2 = get_loader(dataset2_loc, 224, 64)
data_loader3 = get_loader(dataset3_loc, 224, 64)

# Loading our model
resnet_18 = models.resnet18(weights=None)
resnet_18.fc = nn.Linear(resnet_18.fc.in_features, num_classes)
resnet_18.load_state_dict(torch.load('./task1/best_weights.pth', weights_only=True))
resnet_18.to(device)
# model = torch.nn.Sequential(*(list(resnet_18.children())[:-1]))

# ImageNet model
resnet_18_ImageNet = models.resnet18(weights='IMAGENET1K_V1')
resnet_18_ImageNet.fc = nn.Linear(resnet_18_ImageNet.fc.in_features, num_classes)
resnet_18_ImageNet.to(device)
# model_ImageNet = torch.nn.Sequential(*(list(resnet_18_ImageNet.children())[:-1]))
# model_ImageNet.to(device)

# Test models
save_analysis(resnet_18, data_loader2, device, class_names2, './task2/out_TrainedModel/dataset2')
print('Done analyzing our model with data set 2\n')
save_analysis(resnet_18, data_loader3, device, class_names3, './task2/out_TrainedModel/dataset3')
print('Done analyzing our model with data set 3\n')

save_analysis(resnet_18_ImageNet, data_loader2, device, class_names2, './task2/out_ImageNet/dataset2')
print('Done analyzing ImageNet model with data set 2\n')
save_analysis(resnet_18_ImageNet, data_loader3, device, class_names3, './task2/out_ImageNet/dataset3')
print('Done analyzing ImageNet model with data set 3\n')


