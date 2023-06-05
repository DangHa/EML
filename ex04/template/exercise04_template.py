from __future__ import print_function
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torchvision.ops as tv_nn


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.Dropout(dropout_p),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
    

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout_p),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class VGG11(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.layers = self._make_layers(dropout_p)

    def _conv_block(in_features, out_features):
        return nn.ModuleList([
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.ReLU()
        ])

    def _make_layers(self, dropout_p):
        layers = [
            ConvBlock(3, 64, dropout_p),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, dropout_p),
            nn.MaxPool2d(2),
            ConvBlock(128, 256, dropout_p),
            ConvBlock(256, 256, dropout_p),
            nn.MaxPool2d(2),
            ConvBlock(256, 512, dropout_p),
            ConvBlock(512, 512, dropout_p),
            nn.MaxPool2d(2),
            ConvBlock(512, 512, dropout_p),
            ConvBlock(512, 512, dropout_p),
            nn.MaxPool2d(2),
            nn.Flatten(),
            LinearBlock(512, 4096, dropout_p),
            LinearBlock(4096, 4096, dropout_p),
            nn.Linear(4096, 10)
        ]
        return nn.ModuleList(layers)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
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
                100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.sum().item()
    
    train_loss /= len(train_loader.dataset)
    return train_loss

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
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='dropout_p (default: 0.0)')
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
    parser.add_argument('--selectaug', type=int, default=1, metavar='N',
                        help='Type of data augmentation')                        
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

    if args.selectaug == 1:
      train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly image crop
        transforms.RandomHorizontalFlip(),  # Randomly horizontally flip
        transforms.ToTensor()
      ])

    elif args.selectaug == 2:
      train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly image crop
        transforms.RandomHorizontalFlip(),  # Randomly horizontally flip
        transforms.RandomRotation(10), # Randomly image rotate by 10 degree
        transforms.ToTensor()
      ])
      
    elif args.selectaug == 3:
      train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly image crop
        transforms.RandomHorizontalFlip(),  # Randomly horizontally flip
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # Randomly  brightness, contrast, and saturation
        transforms.ToTensor()
      ])
      
    test_transforms = transforms.Compose([
       transforms.ToTensor()
    ])

    dataset_train = datasets.SVHN('../data', split='train', download=True,
                       transform=train_transforms)
    dataset_test = datasets.SVHN('../data', split='test', download=True,
                       transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = VGG11(dropout_p=args.dropout_p).to(device)
    
    # L2 regulization
    if args.L2_reg is not None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2_reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

    if (args.L2_reg is not None):
        f_name = f'trained_VGG11_L2-{args.L2_reg}.pt'
        torch.save(model.state_dict(), f_name)
        print(f'Saved model to: {f_name}')

    if args.output_file:
        with open(args.output_file, "w") as f:
            data = {"train_losses": train_losses,
                    "test_losses": test_losses,
                    "test_accuracies": test_accuracies}
            json.dump(data, f)


if __name__ == '__main__':
    main()
