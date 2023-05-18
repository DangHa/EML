import json
import argparse
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"


def plot(mlp_data, cnn_data):
    dummy_date = datetime.datetime(1970, 1, 1)

    fig, axs = plt.subplots(1, 2)

    for data, device, color in zip([mlp_data, cnn_data], ["MLP", "CNN"], ["r", "g"]):
        x_time = [dummy_date + datetime.timedelta(seconds=s) for s in data["times"]]
        x_epoch = data["epochs"]
        y = data["accuracies"]
        axs[0].plot(x_time, y, color, label=device)
        axs[1].plot(x_epoch, y, color, label=device)

    fig.suptitle("MLP vs CNN training speed comparison on SVHN")
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
    axs[0].set_xlabel("Time [mm:ss]")
    axs[0].set_ylabel("Test accuracy [%]")
    axs[0].legend()
    axs[1].xaxis.get_major_locator().set_params(integer=True)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Test accuracy [%]")
    axs[1].legend()
    #fig.autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mlp_logfile", type=str,
                        help="path to the output file of the MLP experiment")
    parser.add_argument("cnn_logfile", type=str,
                        help="path to the output file of the CNN experiment")
    args = parser.parse_args()

    with open(args.mlp_logfile, "r") as f:
        mlp_data = json.load(f)

    with open(args.cnn_logfile, "r") as f:
        cnn_data = json.load(f)

    plot(mlp_data, cnn_data)
