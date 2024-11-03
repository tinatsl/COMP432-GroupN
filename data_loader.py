import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_loaders(data_path, img_size=224, batch_size=32):
    """ This function generates train, validation, and test data loaders

    Parameters
    ----------
    data_path: str
        path to target dataset
    img_size: int
        expected size of one data sample
    batch_size: int
        batch size for data loader

    Returns
    -------
    train_loader: DataLoader
        loader for the train set
    val_loader: DataLoader
        loader for the validation set
    test_loader: DataLoader
        loader for the test set
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7626, 0.5242, 0.7113], std=[0.1492, 0.1949, 0.1401])
    ])

    # Load the full dataset
    dataset = ImageFolder(data_path, transform=transform)

    # Calculate the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    generator = torch.Generator().manual_seed(10)

    # Split dataset into train + validation and test
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size], generator=generator)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Extract features and labels compatible with grid search
def get_data(loader):
    X, y = zip(*[(data, labels) for data, labels in loader])

    # Concatenate along the first dimension
    X = torch.cat(X, dim=0).cpu().numpy()
    y = torch.cat(y, dim=0).long().cpu().numpy()

    return X, y