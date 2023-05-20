import json
import argparse
import datetime
import matplotlib.pyplot as plt

plt.rcParams["date.autoformatter.minute"] = "%M:%S"


def plot(optimizer, lr_01, lr_005, lr_001, lr_0005, lr_0001):

    for data, device, color in zip([lr_01, lr_005, lr_001, lr_0005, lr_0001], 
                                   ["lr_01", "lr_005", "lr_001", "lr_0005", "lr_0001"], 
                                   ["r", "g--", "b", "purple", "orange"]):
        x = data["epochs"]
        y = data["accuracies"]
        plt.plot(x, y, color, label=device)

    plt.title("Different learning rates with " + optimizer.upper() + " optimizer.")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy [%]")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default="sgd", metavar='OP',
                        choices=["sgd", "adam", "adamax"],
                        help='optimizer')

    parser.add_argument("lr_01", type=str,
                        help="learning rate is 0.1")
    parser.add_argument("lr_005", type=str,
                        help="learning rate is 0.05")
    parser.add_argument("lr_001", type=str,
                        help="learning rate is 0.01")
    parser.add_argument("lr_0005", type=str,
                        help="learning rate is 0.005")
    parser.add_argument("lr_0001", type=str,
                        help="learning rate is 0.001")
    args = parser.parse_args()

    with open(args.lr_01, "r") as f:
        lr_01 = json.load(f)

    with open(args.lr_005, "r") as f:
        lr_005 = json.load(f)
    
    with open(args.lr_001, "r") as f:
        lr_001 = json.load(f)

    with open(args.lr_0005, "r") as f:
        lr_0005 = json.load(f)
    
    with open(args.lr_0001, "r") as f:
        lr_0001 = json.load(f)

    plot(args.optimizer, lr_01, lr_005, lr_001, lr_0005, lr_0001)
