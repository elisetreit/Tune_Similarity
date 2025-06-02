# visualization.py

import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training and validation loss over epochs."""
    train_loss = history.history['loss']
    val_loss = history.history.get('val_loss', None)
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label='Train Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(metrics):
    """Plot evaluation metrics."""
    plt.figure(figsize=(8, 4))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()