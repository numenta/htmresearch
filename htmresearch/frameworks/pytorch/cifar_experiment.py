# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
from __future__ import print_function, division

import os
import random
import time

import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms, datasets

from htmresearch.frameworks.pytorch.model_utils import trainModel, evaluateModel
from htmresearch.frameworks.pytorch.image_transforms import RandomNoise
from htmresearch.frameworks.pytorch.modules.k_winners import getEntropies
from htmresearch.frameworks.pytorch.modules import (
  SparseWeights2d, KWinners2d, updateBoostStrength, rezeroWeights,
  Flatten
)
from htmresearch.frameworks.pytorch.modules.not_so_densenet import (
  NotSoDenseNet
)
from htmresearch.support.expsuite import PyExperimentSuite



class CIFARExperiment(PyExperimentSuite):
  """
  Experiment for training on CIFAR-10
  """


  def __init__(self):
    super(CIFARExperiment, self).__init__()

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def do_experiment(self, params):
    self.restore_supported = ("restore_supported" in self.cfgparser.defaults() and
                              self.cfgparser.getboolean("DEFAULT", "restore_supported"))
    super(CIFARExperiment, self).do_experiment(params)


  def reset(self, params, repetition):
    """Called at the beginning of each experiment and each repetition"""

    pprint.pprint(params)

    self.initialize(params, repetition)

    # Load CIFAR dataset
    dataDir = params.get('dataDir', 'data')
    self.transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    self.trainset = datasets.CIFAR10(root=dataDir, train=True, download=True,
                                transform=self.transform_train)

    self.createModel(params, repetition)

    print("Torch reports", torch.cuda.device_count(), "GPUs available")
    if torch.cuda.device_count() > 1:
      self.model = torch.nn.DataParallel(self.model)

    self.model.to(self.device)

    self.optimizer = self.createOptimizer(self.model)
    self.lr_scheduler = self.createLearningRateScheduler(self.optimizer)
    self.test_loaders = self.createTestLoaders(self.noise_values)


  def initialize(self, params, repetition):
    """
    Initialize experiment parameters and default values from configuration file.
    Called by reset() at the beginning of each experiment and each repetition.
    """
    self.name = params["name"]
    self.dataDir = params.get("datadir", "data")
    self.seed = params.get("seed", 42) + repetition

    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    random.seed(self.seed)

    # Training
    self.epochs = params.get("epochs", 1)
    self.batch_size = params.get("batch_size", 128)
    self.batches_in_epoch = params.get("batches_in_epoch", 100000)
    self.first_epoch_batch_size = params.get("first_epoch_batch_size",
                                             self.batch_size)
    self.batches_in_first_epoch = params.get("batches_in_first_epoch",
                                             self.batches_in_epoch)

    # Testing
    self.test_batch_size = params.get("test_batch_size", 1000)
    self.test_batches_in_epoch = params.get("test_batches_in_epoch", 100000)
    self.noise_values = map(float,
                            params.get("noise_values", "0.0, 0.1").split(", "))
    self.best_noise_score = 0.0
    self.best_epoch = -1

    # Optimizer
    self.optimizer_class = eval(params.get("optimizer", "torch.optim.SGD"))
    self.lr = params.get("learning_rate", 0.05)
    self.momentum = params.get("momentum", 0.9)
    self.weight_decay = params.get("weight_decay", 0.0)
    self.optimizer_params = eval(params.get("optimizer_params", "{}"))
    self.lr_scheduler_gamma = params.get("lr_scheduler_gamma", 0.9)
    self.loss_function = eval(params.get("loss_function",
                                         "torch.nn.functional.cross_entropy"))

    # Network parameters
    self.conv1_sparsity = params.get("conv1_sparsity", 1.0)
    self.network_type = params.get("network_type", "sparse")
    self.growth_rate = params.get("growth_rate", 12)
    self.nblocks = map(int,
                       params.get("nblocks", "6, 12, 24, 16").split(", "))
    self.k_inference_factor = params.get("k_inference_factor", 1.5)

    self.dense_sparsities = map(float,
                                params.get("dense_sparsities",
                                           "1.0, 1.0, 1.0, 1.0").split(", "))
    self.transition_sparsities = map(float,
                                     params.get("transition_sparsities",
                                                "0.1, 0.1, 0.2").split(", "))
    self.linear_sparsity = params.get("linear_sparsity", 0.0)
    self.linear_weight_sparsity = params.get("linear_weight_sparsity", 0.3)
    self.linear_n = params.get("linear_n", 500)
    self.avg_pool_size = params.get("avg_pool_size", 2)
    self.dense_c1_out_planes = params.get("dense_c1_out_planes", 4*self.growth_rate)


  def createLearningRateScheduler(self, optimizer):
    """
    Creates the learning rate scheduler and attach the optimizer
    """
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=1,
                                           gamma=self.lr_scheduler_gamma)


  def createOptimizer(self, model):
    """
    Create a new instance of the optimizer
    """
    return torch.optim.SGD(model.parameters(),
                           lr=self.lr,
                           momentum=self.momentum,
                           weight_decay=self.weight_decay)


  def iterate(self, params, repetition, iteration):

    print("\nEpoch: {:d}".format(iteration))
    self.epoch_start_time = time.time()

    if iteration == 0:
      batch_size = self.first_epoch_batch_size
      batches_in_epoch = self.batches_in_first_epoch
    else:
      batch_size = self.batch_size
      batches_in_epoch = self.batches_in_epoch

    # Train on a single task
    train_loader = torch.utils.data.DataLoader(dataset=self.trainset,
                                               batch_size=batch_size,
                                               shuffle=True)
    self.preEpoch()
    trainModel(model=self.model, loader=train_loader,
               optimizer=self.optimizer, device=self.device,
               batches_in_epoch=batches_in_epoch,
               criterion=self.loss_function)
    self.postEpoch()

    self.epoch_train_time = time.time() - self.epoch_start_time

    # Test on all trained tasks combined
    ret = self.runNoiseTests(noiseValues=self.noise_values,
                             loaders=self.test_loaders)
    self.epoch_time = time.time() - self.epoch_start_time

    # Include learning rate stats
    ret.update({"learning_rate": self.lr_scheduler.get_lr()[0]})

    self.logger(iteration, ret)

    return ret


  def logger(self, iteration, ret):
    """Print out relevant information"""
    print("Learning rate: {:f}".format(self.lr_scheduler.get_lr()[0]))
    entropies = getEntropies(self.model)
    print("Entropy and max entropy: ", float(entropies[0]), entropies[1])
    print("Training time for epoch=", self.epoch_train_time)
    for noise in self.noise_values:
      print("Noise= {:3.2f}, loss = {:5.4f}, Accuracy = {:5.3f}%".format(
        noise, ret[noise]["loss"], 100.0*ret[noise]["accuracy"]))
    print("Full epoch time =", self.epoch_time)
    if ret[0.0]["accuracy"] > 0.91:
      self.best_noise_score = max(ret[0.1]["accuracy"], self.best_noise_score)
      self.best_epoch = iteration


  def createTestLoaders(self, noise_values):
    """
    Create a list of data loaders, one for each noise value
    """
    print("Creating test loaders for noise values:", noise_values)
    loaders = []
    for noise in noise_values:

      transform_noise_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        RandomNoise(noise,
                    whiteValue=0.5 + 2 * 0.20,
                    blackValue=0.5 - 2 * 0.2),
      ])

      testset = datasets.CIFAR10(root=self.dataDir,
                                 train=False,
                                 download=True,
                                 transform=transform_noise_test)
      loaders.append(
        DataLoader(testset, batch_size=self.test_batch_size, shuffle=False)
      )

    return loaders


  def runNoiseTests(self, noiseValues, loaders):
    """
    Test the model with different noise values and return test metrics.
    """
    ret = {}
    for noise, loader in zip(noiseValues, loaders):
      testResult = evaluateModel(
        model=self.model,
        loader=loader,
        device=self.device,
        batches_in_epoch=self.test_batches_in_epoch,
        criterion=self.loss_function
      )
      ret[noise] = testResult

    return ret


  def createModel(self, params, repetition):

    if self.network_type == "dense":
      self.model = NotSoDenseNet(
        nblocks=self.nblocks,
        growth_rate=self.growth_rate,
        dense_c1_out_planes=self.dense_c1_out_planes,
        avg_pool_size=self.avg_pool_size,
        conv1_sparsity=1.0,
        dense_sparsities=[1.0] * 4,
        transition_sparsities=[1.0] * 3,
        linear_sparsity=0.0,
      )

    else:
      self.model = NotSoDenseNet(
        nblocks=self.nblocks,
        growth_rate=self.growth_rate,
        conv1_sparsity=self.conv1_sparsity,
        dense_c1_out_planes=self.dense_c1_out_planes,
        dense_sparsities=self.dense_sparsities,
        transition_sparsities=self.transition_sparsities,
        linear_sparsity=self.linear_sparsity,
        linear_weight_sparsity=self.linear_weight_sparsity,
        linear_n=self.linear_n,
        avg_pool_size=self.avg_pool_size,
        k_inference_factor=self.k_inference_factor,
      )


  def preEpoch(self):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()


  def postEpoch(self):
    if hasattr(self.model, "postEpoch"):
      self.model.postEpoch()
    else:
      self.model.apply(updateBoostStrength)
      self.model.apply(rezeroWeights)

  def save_state(self, params, rep, n):
    print("Saving state")
    saveDir = os.path.join(params["path"], params["name"], "model.{}.{}.pt".format(rep, n))
    torch.save(self.model, saveDir)


  def restore_state(self, params, rep, n):
    saveDir = os.path.join(params["path"], params["name"], "model.{}.{}.pt".format(rep, n))
    self.model = torch.load(saveDir, map_location=self.device)


  def finalize(self, params, rep):
    print("\nBest noise score = {:5.2f}%".format(100.0*self.best_noise_score),
          "at epoch {:d}".format(self.best_epoch))
    if params.get("restore_supported", False):
      print("Saving state")
      saveDir = os.path.join(params["path"], params["name"], "model.final.pt")
      torch.save(self.model, saveDir)



if __name__ == '__main__':
  suite = CIFARExperiment()
  suite.start()
