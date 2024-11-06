import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_tsne(features, labels, class_names):
    # Apply t-SNE - 2D
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    # cmap = plt.get_cmap('plasma', len(class_names))
    # colors = [cmap(i) for i in range(len(class_names))]

    custom_colors = ['#5b1e51', '#cb1b4f', '#f6b38e']

    # Plotting
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        idxs = labels == i
        plt.scatter(
            features_2d[idxs, 0],
            features_2d[idxs, 1],
            label=class_name,
            alpha=0.6,
            color=custom_colors[i]
        )

    plt.legend()
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

def report_metrics(test_labels, test_predictions, class_names):
    # Scikit-learn classification report
    report = classification_report(test_labels, test_predictions, target_names=class_names)
    print(report)

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_predictions, class_names=class_names)

def test_model(model, test_loader, device, class_names):
    model.eval()
    features = []
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get model outputs and predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Collect features, predictions, and labels
            features.extend(outputs.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Report metrics and plot confusion matrix
    report_metrics(test_labels, test_predictions, class_names)

    # Plot t-SNE visualization: added into test program as it is reused in Task 2
    plot_tsne(np.array(features), np.array(test_labels), class_names)