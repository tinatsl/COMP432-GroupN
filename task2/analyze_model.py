import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

def save_analysis(model, data_loader, device, class_names, save_path):
    model.eval()
    features = []
    predictions = []
    labels = []

    # Extract features
    with torch.no_grad():
        for images, label in data_loader:
            images = images.to(device)
            label = label.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            features.append(outputs.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            labels.extend(label.cpu().numpy())
    #combine and convert
    features = np.vstack(features)  
    labels = np.array(labels)       
    predictions = np.array(predictions) 

    # Calculate tsne
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot tsne
    custom_colors = ['r', 'g', 'b']

    # Plotting
    plt.figure()
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

    #Save outputs:
    plt.savefig((save_path + '.png'), bbox_inches='tight')
    print('Saved tsne plot at:' + save_path + '.png')

    np.save((save_path + '_features.npy'), features)
    np.save((save_path + '_features2D.npy'), features_2d)
    np.save((save_path +'_labels.npy'), labels)
    print('Saved features and labels')

    report = classification_report(labels, predictions, target_names=class_names)
    print(report)
    with open((save_path + '_report.txt'), 'w') as file:
        file.write(report)
    