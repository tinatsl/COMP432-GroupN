import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from matplotlib import pyplot as plt

# Loss and accuracy graph
def plot_metrics(epochs, train_losses, train_accuracies, val_losses=None, val_accuracies=None, save_path=None):
    plt.figure(figsize=(12, 5))

    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='train_loss', color='#ec1a24')
    if val_losses is not None:
        plt.plot(epochs, val_losses, label='valid_loss', color='#11469b')
    plt.title('Loss per Epoch')
    plt.xlabel('# Epoch')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.legend()

    # Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='train_accuracy', color='#ec1a24')
    if val_accuracies is not None:
        plt.plot(epochs, val_accuracies, label='valid_accuracy', color='#11469b')
    plt.title('Accuracy per Epoch')
    plt.xlabel('# Epoch')
    plt.ylabel('Accuracy')
    plt.grid(False)
    plt.legend()

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)

    plt.show()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, max_epoch, device, class_names, patience=20):
    best_loss = float('inf')
    early_stop_counter = 0

    # Table columns
    print(f"{'epoch':<6}{'train loss':<12}{'train acc':<12}{'valid loss':<12}{'valid acc':<12}")
    print("=" * 54)

    # Lists to store metrics for plotting
    train_losses_per_epoch, train_accuracies_per_epoch = [], []
    val_losses_per_epoch, val_accuracies_per_epoch = [], []

    for epoch in range(max_epoch):
        # Training Phase
        model.train()
        train_losses, train_predictions, train_labels = [], [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            predictions = model(features)
            loss = criterion(predictions, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            train_losses.append(loss.item())

            # Store predictions and labels for accuracy calculation
            _, predicted = torch.max(predictions, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_accuracy = accuracy_score(train_labels, train_predictions)
        epoch_train_loss = np.mean(train_losses)
        train_losses_per_epoch.append(epoch_train_loss)
        train_accuracies_per_epoch.append(train_accuracy)

        # Validation Phase
        model.eval()
        val_losses, val_predictions, val_labels = [], [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                predictions = model(features)
                loss = criterion(predictions, labels)
                val_losses.append(loss.item())

                # Store predictions and labels for accuracy calculation
                _, predicted = torch.max(predictions, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_accuracy = accuracy_score(val_labels, val_predictions)
        epoch_val_loss = np.mean(val_losses)
        val_losses_per_epoch.append(epoch_val_loss)
        val_accuracies_per_epoch.append(val_accuracy)

        print(f"{epoch + 1:<6}{epoch_train_loss:<12.4f}{train_accuracy:<12.4f}{epoch_val_loss:<12.4f}{val_accuracy:<12.4f}")

        # Early stopping check based on validation loss
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_weights.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final Training and Validation Reports
    print("\nTRAINING REPORT:")
    print(classification_report(train_labels, train_predictions, target_names=class_names))

    print("\nVALIDATION REPORT:")
    print(classification_report(val_labels, val_predictions, target_names=class_names))

    print("TRAINING COMPLETE.")

    plot_metrics(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, train_accuracies_per_epoch, val_losses_per_epoch, val_accuracies_per_epoch)
