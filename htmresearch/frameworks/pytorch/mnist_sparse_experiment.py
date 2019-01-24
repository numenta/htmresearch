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
import sys
import traceback
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from htmresearch.frameworks.pytorch.benchmark_utils import (
  register_nonzero_counter, unregister_counter_nonzero)
from htmresearch.support.expsuite import PyExperimentSuite

from htmresearch.frameworks.pytorch.image_transforms import RandomNoise
from htmresearch.frameworks.pytorch.sparse_net import SparseNet
from htmresearch.frameworks.pytorch.duty_cycle_metrics import plotDutyCycles
from htmresearch.frameworks.pytorch.dataset_utils import createValidationDataSampler


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

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(self.dataDir, train=True, download=True,
                                   transform=transform)

    # Create training and validation sampler from MNIST dataset by training on
    # random X% of the training set and validating on the remaining (1-X)%,
    # where X can be tuned via the "validation" parameter
    validation = params.get("validation", 50000.0 / 60000.0)
    if validation < 1.0:
      self.train_sampler, self.validation_sampler = createValidationDataSampler(
        train_dataset, validation)

      self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=params["batch_size"],
                                                      sampler=self.train_sampler)

      self.validation_loader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=params["batch_size"],
                                                           sampler=self.validation_sampler)
    else:
      # No validation. Normal training dataset
      self.validation_loader = None
      self.validation_sampler = None
      self.train_sampler = None
      self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=params["batch_size"],
                                                      shuffle=True)


    self.test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(self.dataDir, train=False, transform=transform),
      batch_size=params["test_batch_size"], shuffle=True)

    # Parse 'n' and 'k' parameters
    n = params["n"]
    k = params["k"]
    if isinstance(n, basestring):
      n = map(int, n.split("_"))
    if isinstance(k, basestring):
      k = map(int, k.split("_"))

    if params["use_cnn"]:
      c1_out_channels = params["c1_out_channels"]
      c1_k = params["c1_k"]
      if isinstance(c1_out_channels, basestring):
        c1_out_channels = map(int, c1_out_channels.split("_"))
      if isinstance(c1_k, basestring):
        c1_k = map(int, c1_k.split("_"))

      sp_model = SparseNet(
        inputSize=(1, 28, 28),
        outChannels=c1_out_channels,
        c_k=c1_k,
        dropout=params["dropout"],
        n=n,
        k=k,
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
        useBatchNorm=params["use_batch_norm"],
      )
      print("c1OutputLength=", sp_model.cnnSdr[0].outputLength)
    else:
      sp_model = SparseNet(
        n=n,
        k=k,
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
        dropout=params["dropout"],
        useBatchNorm=params["use_batch_norm"],
      )
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs")
      sp_model = torch.nn.DataParallel(sp_model)

    self.model = sp_model.to(self.device)
    self.learningRate = params["learning_rate"]
    self.optimizer = self.createOptimizer(params, self.model)
    self.lr_scheduler = self.createLearningRateScheduler(params, self.optimizer)


  def iterate(self, params, repetition, iteration):
    """
    Called once for each training iteration (== epoch here).
    """
    try:
      print("\nStarting iteration",iteration)
      t1 = time.time()
      ret = {}

      # Update learning rate using learning rate scheduler if configured
      if self.lr_scheduler is not None:
        # ReduceLROnPlateau lr_scheduler step should be called after validation,
        # all other lr_schedulers should be called before training
        if params["lr_scheduler"] != "ReduceLROnPlateau":
          self.lr_scheduler.step()

      self.train(params, epoch=iteration)

      # Run validation test
      if self.validation_loader is not None:
        validation = self.test(params, self.validation_loader)

        # ReduceLROnPlateau step should be called after validation
        if params["lr_scheduler"] == "ReduceLROnPlateau":
          self.lr_scheduler.step(validation["test_loss"])

        ret["validation"] = validation
        print("Validation: Test error=", validation["testerror"],
              "entropy=", validation["entropy"])

      # Run noise test
      if (params["test_noise_every_epoch"] or
          iteration == params["iterations"] - 1):
        ret.update(self.runNoiseTests(params))
        print("Noise test results: totalCorrect=", ret["totalCorrect"],
              "Test error=", ret["testerror"], ", entropy=", ret["entropy"])
        if ret["totalCorrect"] > 100000 and ret["testerror"] > 98.3:
          print("*******")
          print(params)

      ret.update({"elapsedTime": time.time() - self.startTime})
      ret.update({"learningRate": self.learningRate if self.lr_scheduler is None
                                                    else self.lr_scheduler.get_lr()})

      print("Iteration time= {0:.3f} secs, "
            "total elapsed time= {1:.3f} mins".format(
              time.time() - t1,ret["elapsedTime"]/60.0))

    except Exception as e:
      # Tracebacks are not printed if using multiprocessing so we do it here
      tb = sys.exc_info()[2]
      traceback.print_tb(tb)
      raise RuntimeError("Something went wrong in iterate")

    return ret


  def finalize(self, params, rep):
    """
    Called once we are done.
    """
    if params.get("saveNet", True):
      self.pruneWeights(params)
      self.pruneDutyCycles(params)

      # Save the full model once we are done.
      saveDir = os.path.join(params["path"], params["name"], "model.pt")
      torch.save(self.model, saveDir)


  def createLearningRateScheduler(self, params, optimizer):
    """
    Creates the learning rate scheduler and attach the optimizer
    """
    lr_scheduler = params.get("lr_scheduler", None)
    if lr_scheduler is None:
      return None

    if lr_scheduler == "StepLR":
      lr_scheduler_params = "{'step_size': 1, 'gamma':" + str(params["learning_rate_factor"]) + "}"

    else:
      lr_scheduler_params = params.get("lr_scheduler_params", None)
      if lr_scheduler_params is None:
        raise ValueError("Missing 'lr_scheduler_params' for {}".format(lr_scheduler))

    # Get lr_scheduler class by name
    clazz = eval("torch.optim.lr_scheduler.{}".format(lr_scheduler))

    # Parse scheduler parameters from config
    lr_scheduler_params = eval(lr_scheduler_params)

    return clazz(optimizer, **lr_scheduler_params)

  def createOptimizer(self, params, model):
    """
    Create a new instance of the optimizer
    """
    lr = params["learning_rate"]
    print("Creating optimizer with learning rate=", lr)
    if params["optimizer"] == "SGD":
      optimizer = optim.SGD(model.parameters(), lr=lr,
                            momentum=params["momentum"])
    elif params["optimizer"] == "Adam":
      optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
      raise LookupError("Incorrect optimizer value")

    return optimizer

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

      # Log info every log_interval mini batches
      if batch_idx % params["log_interval"] == 0:
        entropy = self.model.entropy()
        print("logging: ",self.model.getLearningIterations(),
              " learning iterations, elapsedTime", time.time() - self.startTime,
              " entropy:", float(entropy)," / ", self.model.maxEntropy())
        if params["create_plots"]:
          plotDutyCycles(self.model.dutyCycle,
                         self.resultsDir + "/figure_"+str(epoch)+"_"+str(
                           self.model.getLearningIterations()))

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

    nonzeros = None
    count_nonzeros = params.get("count_nonzeros", False)
    if count_nonzeros:
      nonzeros = {}
      register_nonzero_counter(self.model, nonzeros)

    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        # count nonzeros only once
        if count_nonzeros:
          count_nonzeros = False
          unregister_counter_nonzero(self.model)

    test_loss /= len(test_loader.sampler)
    test_error = 100. * correct / len(test_loader.sampler)

    entropy = self.model.entropy()
    ret = {"num_correct": correct,
           "test_loss": test_loss,
           "testerror": test_error,
           "entropy": float(entropy)}

    if nonzeros is not None:
      ret["nonzeros"] = nonzeros

    return ret


  def runNoiseTests(self, params):
    """
    Test the model with different noise values and return test metrics.
    """
    ret = {}

    # Noise on validation data
    validation = {} if self.validation_sampler is not None else None

    # Test with noise
    total_correct = 0
    validation_total_correct = 0
    for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
      transform = transforms.Compose([
        transforms.ToTensor(),
        RandomNoise(noise, whiteValue=0.1307 + 2*0.3081),
        transforms.Normalize((0.1307,), (0.3081,))
      ])
      test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(self.dataDir, train=False, transform=transform),
        batch_size=params["test_batch_size"], shuffle=True)

      testResult = self.test(params, test_loader)
      total_correct += testResult["num_correct"]
      ret[noise]= testResult

      if validation is not None:
        validation_loader = torch.utils.data.DataLoader(
          datasets.MNIST(self.dataDir, train=True, transform=transform),
          sampler=self.validation_sampler,
          batch_size=params["test_batch_size"])

        validationResult = self.test(params, validation_loader)
        validation_total_correct += validationResult["num_correct"]
        validation[noise] = validationResult

    ret["totalCorrect"] = total_correct
    ret["testerror"] = ret[0.0]["testerror"]
    ret["entropy"] = ret[0.0]["entropy"]

    if "nonzeros" in ret[0.0]:
      ret["nonzeros"] = ret[0.0]["nonzeros"]

    if validation is not None:
      validation["totalCorrect"] = validation_total_correct
      validation["testerror"] = validation[0.0]["testerror"]
      validation["entropy"] = validation[0.0]["entropy"]
      ret["validation"] = validation

    return ret


  def pruneWeights(self, params):
    """
    Prune the weights whose absolute magnitude is < params["min_weight"]
    """
    self.model.pruneWeights(params.get("min_weight", 0.0))


  def pruneDutyCycles(self, params):
    """
    Prune the DutyCycles whose absolute magnitude is < params["min_dutycycle"]
    """
    self.model.pruneDutyCycles(params.get("min_dutycycle", 0.0))


if __name__ == '__main__':
  suite = MNISTSparseExperiment()
  suite.start()
