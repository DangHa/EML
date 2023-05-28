import json
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np


def plot(l2_0001, l2_00001, l2_000001, weights_0001, weights_00001, weights_000001):

    epochs = np.arange(1, len(l2_0001["train_losses"]) + 1)

    # different l2 strength plots
    plt.plot(epochs, l2_0001["test_accuracies"], label="l2=0.001")
    plt.plot(epochs, l2_00001["test_accuracies"], label="l2=0.0001")
    plt.plot(epochs, l2_000001["test_accuracies"], label="l2=0.00001")

    plt.title("VGG11 with varying l2 strengths")
    plt.xlabel("Epochs")
    plt.ylabel("Test accuracy [%]")
    plt.legend()
    plt.show()

    # the historgram of the last 
    fig, axs = plt.subplots(1, 3, figsize=(9, 5))
    _ = axs[0].hist(weights_0001, density=True, bins=30)
    axs[0].legend()
    axs[0].set_xlabel("l2=0.001")
    axs[0].set_ylabel("Number of elements in this range")

    _ = axs[1].hist(weights_00001, density=True, bins=30)
    axs[1].legend()
    axs[1].set_xlabel("l2=0.0001")

    _ = axs[2].hist(weights_000001, density=True, bins=30)
    axs[2].legend()
    axs[2].set_xlabel("l2=0.00001")

    fig.suptitle("VGG11 with varying l2 strengths")
    plt.show()

if __name__ == "__main__":

    with open("task02_l2_0001.json", "r") as f:
        l2_0001 = json.load(f)

    with open("task02_l2_00001.json", "r") as f:
        l2_00001 = json.load(f)

    with open("task02_l2_000001.json", "r") as f:
        l2_000001 = json.load(f)

    weights_0001 = torch.load('trained_VGG11_L2-0.001.pt', map_location=torch.device('cpu'))['layers.16.weight'].reshape(-1).detach().numpy()
    weights_00001 = torch.load('trained_VGG11_L2-0.0001.pt', map_location=torch.device('cpu'))['layers.16.weight'].reshape(-1).detach().numpy()
    weights_000001 = torch.load('trained_VGG11_L2-1e-05.pt', map_location=torch.device('cpu'))['layers.16.weight'].reshape(-1).detach().numpy()

    print(weights_0001.shape)

    plot(l2_0001, l2_00001, l2_000001, weights_0001, weights_00001, weights_000001)