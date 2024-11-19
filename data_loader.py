import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_norm_param(train_loader):
    """Calculate the mean and std of a dataset from the given loader.

    Parameters
    ----------
    train_loader: DataLoader
        Temp training set loader

    Returns
    -------
    mean: tensor
        Mean values of the dataset channels
    std: tensor
        Standard deviation values of the dataset channels
    """

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in train_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

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
    mean: list or tensor, optional
        Mean for normalization; if None, no normalization is applied
    std: list or tensor, optional
        Standard deviation for normalization; if None, no normalization is applied

    Returns
    -------
    train_loader: DataLoader
        loader for the train set
    val_loader: DataLoader
        loader for the validation set
    test_loader: DataLoader
        loader for the test set
    """

    transform_default = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Load the full dataset
    dataset = ImageFolder(data_path, transform=transform_default)

    # Calculate the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    generator = torch.Generator().manual_seed(10)

    # Split dataset into train, validation and test
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size],
                                                                   generator=generator)

    train_loader_temp = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Compute mean and std from the training set
    mean, std = get_norm_param(train_loader_temp)

    # Create a transform with normalization
    transform_normalized = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Apply the normalized transform to the datasets
    dataset.transform = transform_normalized
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Extract features and labels compatible with best param search
def get_data(loader):
    x, y = zip(*[(data, labels) for data, labels in loader])

    # Concatenate along the first dimension
    x = torch.cat(x, dim=0).cpu().numpy()
    y = torch.cat(y, dim=0).long().cpu().numpy()

    return x, y
