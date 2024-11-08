import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import numpy as np

dataset_loc = './1/'

dataset = ImageFolder(root=dataset_loc)
class_names = dataset.classes


def show_samples(X, y, num_samples=3):
    classes = np.unique(y)
    samples = {label: [] for label in classes}

    # Randomly select samples for each class
    for label in classes:
        indices = np.where(y == label)[0]
        selected_indices = np.random.choice(indices, size=num_samples, replace=False)
        samples[label] = X[selected_indices]

    # Plotting
    num_classes = len(classes)
    fig, axs = plt.subplots(num_classes, num_samples, figsize=(num_samples * 2, num_classes * 2))
    fig.suptitle("Random Samples Per Class", fontsize=16)

    for row, (label, images) in enumerate(samples.items()):
        for col, img in enumerate(images):
            ax = axs[row, col]
            ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.axis('off')
            if col == 0:
                ax.set_title(f"'{class_names[label]}'")

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    plt.show()
