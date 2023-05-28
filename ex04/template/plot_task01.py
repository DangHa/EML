import json
import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot(base_data, dropout01_data, dropout04_data, dropout07_data):

    # Base network plots
    epochs = np.arange(1, len(base_data["train_losses"]) + 1)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(epochs, base_data["train_losses"], label="Train loss")
    axs[0].plot(epochs, base_data["test_losses"], label="Test loss")
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[1].plot(epochs, base_data["test_accuracies"])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Test accuracy")
    fig.suptitle("No dropout VGG11")
    plt.show()

    # Dropout plots
    fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, base_data["test_accuracies"], label="dropout=0.0")
    ax.plot(epochs, dropout01_data["test_accuracies"], label="dropout=0.1")
    ax.plot(epochs, dropout04_data["test_accuracies"], label="dropout=0.4")
    ax.plot(epochs, dropout07_data["test_accuracies"], label="dropout=0.7")
    ax.legend()
    fig.suptitle("VGG11 with varying dropout")
    plt.show()


if __name__ == "__main__":

    with open("task01_base.json", "r") as f:
        base_data = json.load(f)

    with open("task01_dropout01.json", "r") as f:
        dropout01_data = json.load(f)

    with open("task01_dropout04.json", "r") as f:
        dropout04_data = json.load(f)

    with open("task01_dropout07.json", "r") as f:
        dropout07_data = json.load(f)

    plot(base_data, dropout01_data, dropout04_data, dropout07_data)
