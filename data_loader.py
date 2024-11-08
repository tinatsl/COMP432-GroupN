import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_norm_param(data_path, img_size=224, batch_size=32):
    """Calculate the mean and std of a dataset from the given image path.

    Parameters
    ----------
    data_path: str
        Path to the target dataset
    img_size: int
        Expected size of one data sample for resizing
    batch_size: int
        Batch size for data loader to iterate over dataset

    Returns
    -------
    mean: tensor
        Mean values of the dataset channels
    std: tensor
        Standard deviation values of the dataset channels
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Create an ImageFolder dataset
    dataset = ImageFolder(data_path, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


def get_loaders(data_path, img_size=224, batch_size=32, mean=None, std=None):
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

    if mean is None or std is None:
        mean, std = get_norm_param(data_path, img_size, batch_size)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load the full dataset
    dataset = ImageFolder(data_path, transform=transform)

    # Calculate the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    generator = torch.Generator().manual_seed(10)

    # Split dataset into train, validation and test
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size],
                                                                   generator=generator)

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
