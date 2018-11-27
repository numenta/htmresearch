# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from __future__ import print_function
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from htmresearch.frameworks.pytorch.image_transforms import RandomNoise
from htmresearch.frameworks.pytorch.sparse_mnist_net import SparseMNISTNet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def train(args, model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    model.rezeroWeights()  # Only allow weight changes to the non-zero weights
    if batch_idx % args.log_interval == 0:
      bins = np.linspace(0.0, 0.8, 200)
      plt.hist(model.dutyCycle, bins, alpha=0.5, label='All cols')
      plt.xlabel("Duty cycle")
      plt.ylabel("Number of units")
      plt.savefig("images/figure_"+str(epoch)+"_"+str(model.learningIterations))
      plt.close()
      print("")
      model.printMetrics()
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
      pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  model.printMetrics()
  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  return correct


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=20, metavar='N',
                      help='number of epochs to train (default: 20)')
  parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                      help='learning rate (default: 0.02)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--boost-strength', type=float, default=2.0,
                      metavar='boost',
                      help='Boost strength. Zero implies no boosting.'
                      ' (default: 2.0)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=42, metavar='S',
                      help='random seed (default: 42)')
  parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                      help='how many minibatches to wait before logging' 
                           ' training status (default: 1000)')
  args = parser.parse_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  print("Running with args:")
  print(args)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


  model = SparseMNISTNet(boostStrength=args.boost_strength).to(device)
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

  # Do the training
  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)

  # Test with noise
  total_correct = 0
  for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        RandomNoise(noise,
                    # logDir="data/debug"
                    ),
        transforms.Normalize((0.1307,), (0.3081,))
      ])),
      batch_size=args.test_batch_size, shuffle=True, **kwargs)

    print("Testing with noise level=",noise)
    total_correct += test(args, model, device, test_loader)

  print("k=200, n=2000, potential pool = ",model.weightSparsity)
  print("Total correct:",total_correct)

if __name__ == '__main__':
  main()