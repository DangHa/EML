import json
import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot(resnet, vgg, cnn):

    # Resnet network plots
    epochs = np.arange(1, len(resnet["test_losses"]) + 1)
    fig, axs = plt.subplots(1, 1)
    axs.plot(epochs, resnet["test_losses"], label="ResNet")
    axs.plot(epochs[:30], vgg["test_losses"], label="VGG")
    axs.plot(epochs[:30], cnn["test_losses"], label="CNN")
    axs.legend()
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    fig.suptitle("Resnet training for 50 epoches")
    plt.show()


if __name__ == "__main__":

    with open("task01_resnet.json", "r") as f:
        resnet = json.load(f)

    with open("task01_vgg.json", "r") as f:
        vgg = json.load(f)

    with open("task01_cnn.json", "r") as f:
        cnn = json.load(f)

    plot(resnet, vgg, cnn)
