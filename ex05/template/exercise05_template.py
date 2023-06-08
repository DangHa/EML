from __future__ import print_function
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torchvision.ops as tv_nn
from typing import Any, Callable, List, Optional, Type, Union

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity,
    ) -> None:
        super().__init__()
        # TODO: Implement the basic residual block!
        self.expansion = 1

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = norm_layer(planes)    
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the basic residual block!
        shortcut = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x + shortcut
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity):
        super().__init__()
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.block1_1 = BasicBlock(32, 32, 1, self._norm_layer)
        self.block1_2 = BasicBlock(32, 32, 1, self._norm_layer)
        self.block1_3 = BasicBlock(32, 32, 1, self._norm_layer)
        self.block2_1 = BasicBlock(32, 64, 2, self._norm_layer)
        self.block2_2 = BasicBlock(64, 64, 1, self._norm_layer)
        self.block2_3 = BasicBlock(64, 64, 1, self._norm_layer)
        self.block3_1 = BasicBlock(64, 128, 2, self._norm_layer)
        self.block3_2 = BasicBlock(128, 128, 1, self._norm_layer)
        self.block3_3 = BasicBlock(128, 128, 1, self._norm_layer)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = F.relu(x)
        x = torch.sum(x, [2,3])
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


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
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.time(),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()/data.shape[0] ))

def test(model, device, test_loader, epoch):
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
    print('Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        time.time(),
        epoch,
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    return test_loss, accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--L2_reg', type=float, default=None,
                        help='L2_reg (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output-file', type=str, default=None, metavar='O',
                        help='path to the file where the results should be saved to')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_transforms = [transforms.ToTensor()]
    train_transforms = transforms.Compose(train_transforms)


    dataset_train = datasets.SVHN('../data', split='train', download=True,
                       transform=train_transforms)
    dataset_test = datasets.SVHN('../data', split='test', download=True,
                       transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    norm_layer = nn.Identity
    model = ResNet(norm_layer=norm_layer)
    model = model.to(device)

    if args.L2_reg is not None:
        L2_reg = args.L2_reg
    else:
        L2_reg = 0.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=L2_reg)

    train_losses = []
    test_losses = []
    test_accuracies = []   

    start_time = time.time()
    print(f'Starting training at: {start_time:.4f}')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    end_time = time.time()
    print(f"Finished training. Took: {end_time - start_time} seconds.")

    if args.output_file:
        with open(args.output_file, "w") as f:
            data = {"train_losses": train_losses,
                    "test_losses": test_losses,
                    "test_accuracies": test_accuracies}
            json.dump(data, f)


if __name__ == '__main__':
    main()
