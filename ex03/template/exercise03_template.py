from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import json
import time


# TODO: Implement the MLP class, to be equivalent to the MLP from the last exercise!
# NOTE: We use ReLU instead of Sigmoid in this exercise
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear0 = nn.Linear(in_features, 512)
        self.relu0 = nn.ReLU()
        self.linear1 = nn.Linear(512, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
      x = torch.flatten(x, 1)
      x = self.linear0(x)
      x = self.relu0(x)
      x = self.linear1(x)
      x = self.relu1(x)
      x = self.linear2(x)
      x = F.log_softmax(x, dim=1)
      return x


# TODO: Implement the CNN class, as defined in the exercise!
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten4 = nn.Flatten()
        self.linear5 = nn.Linear(18432, 128)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(128, 10)

    def forward(self, x):
      x = self.conv1(x)
      x = self.relu1(x)
      x = self.conv2(x)
      x = self.relu2(x)
      x = self.conv3(x)
      x = self.relu3(x)
      x = self.flatten4(x)
      x = self.linear5(x)
      x = self.relu5(x)
      x = self.linear6(x)
      x = F.log_softmax(x, dim=1)
      return x



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    return test_loss, accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default="mnist", metavar='D',
                        choices=["mnist", "svhn"],
                        help='dataset on which to run the experiment')
    parser.add_argument('--model', type=str, default="mlp", metavar='M',
                        choices=["mlp", "cnn"],
                        help='model which to train')
    parser.add_argument('--optimizer', type=str, default="adam", metavar='OP',
                        choices=["sgd", "adam", "adamax"],
                        help='optimizer')
    parser.add_argument('--output-file', type=str, default=None, metavar='O',
                        help='path to the file where the results should be saved to')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset == "mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset_train = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset_test = datasets.MNIST('../data', train=False,
                        transform=transform)
    else:
        # Normalization values taken from
        # https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/_modules/deepobs/pytorch/datasets/svhn.html
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
        dataset_train = datasets.SVHN('../data', split="train", download=True,
                        transform=transform)
        dataset_test = datasets.SVHN('../data', split="test", download=True,
                        transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    if args.model == "mlp":
        in_features = 28*28 if args.dataset == "mnist" else 32*32*3
        model = MLP(in_features).to(device)
    else:
        model = CNN().to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    epochs = []
    times = []
    accuracies = []
    time_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)
        
        time_diff = time.time() - time_start
        times.append(time_diff)
        epochs.append(epoch)
        accuracies.append(test_accuracy)

    if args.output_file:
        with open(args.output_file, "w") as f:
            data = {"epochs": epochs,
                    "times": times,
                    "test": test_loss,
                    "accuracies": accuracies}
            json.dump(data, f)


if __name__ == '__main__':
    main()
