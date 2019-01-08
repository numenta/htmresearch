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
import os
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from htmresearch.support.expsuite import PyExperimentSuite

from htmresearch.frameworks.pytorch.image_transforms import RandomNoise
from htmresearch.frameworks.pytorch.sparse_mnist_net import SparseMNISTNet
from htmresearch.frameworks.pytorch.sparse_mnist_cnn import SparseMNISTCNN
from htmresearch.frameworks.pytorch.duty_cycle_metrics import plotDutyCycles


class MNISTSparseExperiment(PyExperimentSuite):
  """
  Allows running multiple sparse MNIST experiments in parallel
  """


  def parse_cfg(self):
    super(MNISTSparseExperiment, self).parse_cfg()
    # Change the current working directory to be relative to 'experiments.cfg'
    projectDir = os.path.dirname(self.options.config)
    projectDir = os.path.abspath(projectDir)
    os.chdir(projectDir)


  def reset(self, params, repetition):
    """
    Called once at the beginning of each experiment.
    """
    self.startTime = time.time()
    print(params)
    torch.manual_seed(params["seed"] + repetition)
    np.random.seed(params["seed"] + repetition)

    # Get our directories correct
    self.dataDir = params["datadir"]
    self.resultsDir = os.path.join(params["path"], params["name"], "plots")

    if not os.path.exists(self.resultsDir):
      os.makedirs(self.resultsDir)

    self.use_cuda = not params["no_cuda"] and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

    self.train_loader = torch.utils.data.DataLoader(
      datasets.MNIST(self.dataDir, train=True, download=True,
                     transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                     ])),
      batch_size=params["batch_size"], shuffle=True, **kwargs)

    self.test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(self.dataDir, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
      ])),
      batch_size=params["test_batch_size"], shuffle=True, **kwargs)

    if params["use_cnn"]:
      sp_model = SparseMNISTCNN(
        c1OutChannels=params["c1_out_channels"],
        c1k=params["c1_k"],
        dropout=params["dropout"],
        n=params["n"],
        k=params["k"],
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
      )
      print("c1OutputLength=", sp_model.c1OutputLength)
    else:
      sp_model = SparseMNISTNet(
        n=params["n"],
        k=params["k"],
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
        dropout=params["dropout"],
      )
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs")
      sp_model = torch.nn.DataParallel(sp_model)

    self.model = sp_model.to(self.device)
    self.learningRate = params["learning_rate"]
    self.createOptimizer(params, self.learningRate)


  def iterate(self, params, repetition, iteration):
    """
    Called once for each training iteration (== epoch here).
    """
    t1 = time.time()
    ret = {}
    self.train(params, epoch=iteration)

    # Run noise test
    if (params["test_noise_every_epoch"] or 
        iteration == params["iterations"] - 1):
      ret.update(self.runNoiseTests(params))
      print("Noise test results: totalCorrect=", ret["totalCorrect"],
            "Test error=", ret["testerror"], ", entropy=", ret["entropy"])
      if ret["totalCorrect"] > 100000 and ret["testerror"] > 98.3:
        print("*******")
        print(params)
    else:
      ret.update(self.test(params, self.test_loader))
      print("Test error=", ret["testerror"], ", entropy=", ret["entropy"])

    ret.update({"elapsedTime": time.time() - self.startTime})
    ret.update({"learningRate": self.learningRate})

    print("Iteration =", iteration,
          ", iteration time= {0:.3f} secs, "
          "total elapsed time= {1:.3f} mins".format(
            time.time() - t1,ret["elapsedTime"]/60.0))

    self.learningRate = self.learningRate * params["learning_rate_factor"]
    self.createOptimizer(params, self.learningRate)

    print("dutycycle:", self.model.dutyCycle.min(),
          self.model.dutyCycle.max(),
          self.model.dutyCycle.mean())

    return ret


  def finalize(self, params, rep):
    """
    Save the full model once we are done.
    """
    saveDir = os.path.join(params["path"], params["name"], "model.pt")
    torch.save(self.model, saveDir)


  def createOptimizer(self, params, lr):
    """
    Create a new instance of the optimizer with the given learning rate.
    """
    print("Creating optimizer with learning rate=",lr)
    if params["optimizer"] == "SGD":
      self.optimizer = optim.SGD(self.model.parameters(),
                                 lr=lr,
                                 momentum=params["momentum"])
    elif params["optimizer"] == "Adam":
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    else:
      raise LookupError("Incorrect optimizer value")


  def train(self, params, epoch):
    """
    Train one epoch of this model by iterating through mini batches. An epoch
    ends after one pass through the training set, or if the number of mini
    batches exceeds the parameter "batches_in_epoch".
    """
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):

      data, target = data.to(self.device), target.to(self.device)
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      self.optimizer.step()
      self.model.rezeroWeights()

      # Log info every log_interval mini batches
      if batch_idx % params["log_interval"] == 0:
        entropy = self.model.entropy()
        print("logging: ",self.model.learningIterations,
              " learning iterations, elapsedTime", time.time() - self.startTime,
              " entropy:", float(entropy)," / ", self.model.maxEntropy())
        if params["create_plots"]:
          plotDutyCycles(self.model.dutyCycle,
                         self.resultsDir + "/figure_"+str(epoch)+"_"+str(
                           self.model.learningIterations))

      if batch_idx >= params["batches_in_epoch"]:
        break

    self.model.postEpoch()


  def test(self, params, test_loader):
    """
    Test the model using the given loader and return test metrics
    """
    self.model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_error = 100. * correct / len(test_loader.dataset)

    entropy = self.model.entropy()
    ret = {"num_correct": correct,
           "test_loss": test_loss,
           "testerror": test_error,
           "entropy": float(entropy)}

    return ret


  def runNoiseTests(self, params):
    """
    Test the model with different noise values and return test metrics.
    """
    ret = {}
    kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

    # Test with noise
    total_correct = 0
    for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
      test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(self.dataDir, train=False, transform=transforms.Compose([
          transforms.ToTensor(),
          RandomNoise(noise,
                      whiteValue=0.1307 + 2*0.3081,
                      # logDir="data/debug"
                      ),
          transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=params["test_batch_size"], shuffle=True, **kwargs)

      testResult = self.test(params, test_loader)
      total_correct += testResult["num_correct"]
      ret[noise]= testResult

    ret["totalCorrect"] = total_correct
    ret["testerror"] = ret[0.0]["testerror"]
    ret["entropy"] = ret[0.0]["entropy"]

    return ret


  def pruneWeights(self, params):
    """
    TODO: Prune the weights whose absolute magnitude is <= params["minWeight"]
    """
    pass


if __name__ == '__main__':
  suite = MNISTSparseExperiment()
  suite.start()
