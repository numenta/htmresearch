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
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from htmresearch.support.expsuite import PyExperimentSuite

from htmresearch.frameworks.pytorch.image_transforms import RandomNoise
from htmresearch.frameworks.pytorch.sparse_mnist_net import SparseMNISTNet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MNISTSparseExperiment(PyExperimentSuite):
  """
  Allows running multiple sparse MNIST experiments in parallel
  """

  def reset(self, params, repetition):
    """
    Called once at the beginning of each experiment.
    """
    print(params)
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    self.resultsDir = os.path.join(params["path"], params["name"], "plots")
    if not os.path.exists(self.resultsDir):
      os.makedirs(self.resultsDir)

    self.use_cuda = not params["no_cuda"] and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

    self.train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('data', train=True, download=True,
                     transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                     ])),
      batch_size=params["batch_size"], shuffle=True, **kwargs)

    self.test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
      ])),
      batch_size=params["test_batch_size"], shuffle=True, **kwargs)

    self.model = SparseMNISTNet(n=params["n"],
                                k=params["k"],
                                boostStrength=params["boost_strength"],
                                weightSparsity=params["weight_sparsity"]
                                ).to(self.device)
    self.optimizer = optim.SGD(self.model.parameters(),
                               lr=params["learning_rate"],
                               momentum=params["momentum"])


  def iterate(self, params, repetition, iteration):
    """
    Called once for each training iteration.
    """
    self.train(params, epoch=iteration)
    return self.runNoiseTests(params)


  def finalize(self, params, rep):
    """
    Called once at the end of each experiment. Not sure what to do here!
    """
    pass


  def train(self, params, epoch):
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):
      data, target = data.to(self.device), target.to(self.device)
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      self.optimizer.step()
      self.model.rezeroWeights()  # Only allow weight changes to the non-zero weights
      if batch_idx % params["log_interval"] == 0:
        bins = np.linspace(0.0, 0.8, 200)
        plt.hist(self.model.dutyCycle, bins, alpha=0.5, label='All cols')
        plt.xlabel("Duty cycle")
        plt.ylabel("Number of units")
        plt.savefig(self.resultsDir + "/figure_"+str(epoch)+"_"+str(
                    self.model.learningIterations))
        plt.close()
        # print("")
        # self.model.printMetrics()
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #   epoch, batch_idx * len(data), len(self.train_loader.dataset),
        #          100. * batch_idx / len(self.train_loader), loss.item()))


  def test(self, params, test_loader):
    self.model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_error = 100. * correct / len(test_loader.dataset)
    # self.model.printMetrics()
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #   test_loss, correct, len(test_loader.dataset),
    #   test_error))

    ret = {"num_correct": correct,
           "test_loss": test_loss,
           "testerror": test_error}

    return ret


  def runNoiseTests(self, params):
    ret = {}
    kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

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
        batch_size=params["test_batch_size"], shuffle=True, **kwargs)

      # print("Testing with noise level=",noise)
      testResult = self.test(params, test_loader)
      total_correct += testResult["num_correct"]
      ret[noise]= testResult

    # self.model.printParameters()
    # print("Total noise correctness score:",total_correct)
    ret["totalCorrect"] = total_correct

    return ret

if __name__ == '__main__':
  suite = MNISTSparseExperiment()
  suite.start()
