import json
import argparse
import datetime
import matplotlib.pyplot as plt

plt.rcParams["date.autoformatter.minute"] = "%M:%S"


def plot(cpu_data, gpu_data):
    dummy_date = datetime.datetime(1970, 1, 1)

    for data, device, color in zip([cpu_data, gpu_data], ["CPU", "GPU"], ["r", "g"]):
        x = [dummy_date + datetime.timedelta(seconds=s) for s in data["times"]]
        y = data["accuracies"]
        plt.plot(x, y, color, label=device)

    plt.title("CPU vs GPU training speed comparison on MNIST")
    plt.xlabel("Time [mm:ss]")
    plt.ylabel("Test accuracy [%]")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_logfile", type=str,
                        help="path to the output file of the CPU experiment")
    parser.add_argument("gpu_logfile", type=str,
                        help="path to the output file of the GPU experiment")
    args = parser.parse_args()

    with open(args.cpu_logfile, "r") as f:
        cpu_data = json.load(f)

    with open(args.gpu_logfile, "r") as f:
        gpu_data = json.load(f)

    plot(cpu_data, gpu_data)
