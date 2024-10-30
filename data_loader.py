import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_loaders(data_path, test_ratio, img_size, batch_size=32):

    """ This function generates train and test data loaders

    Parameters
    ----------
    data_path: str
            path to target dataset
    test_ratio: float
            ratio for the test set with relation to the train set
    img_size: int
            specifies expected img size for the model
    batch_size: int
            batch size for data loader

    Returns
    -------
    train_loader: DataLoader
        loader for the train set

    test_loader: DataLoader
            loader for the test set

    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # get images as tensors
    dataset = ImageFolder(data_path, transform=transform)

    # define train and test ratio
    train_size = int((1.0 - test_ratio) * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(10)

    # split dataset using random_split
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Extract features and labels compatible with grid search
def get_data(loader):

    X, y = zip(*[(data, labels) for data, labels in loader])

    # Concatenate along the first dimension
    X = torch.cat(X, dim=0).cpu().numpy()
    y = torch.cat(y, dim=0).long().cpu().numpy()

    return X, y