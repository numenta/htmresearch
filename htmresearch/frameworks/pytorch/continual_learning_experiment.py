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

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import ConcatDataset
from torchvision import transforms, datasets

from htmresearch.frameworks.pytorch.dataset_utils import splitDataset
from htmresearch.frameworks.pytorch.model_utils import trainModel, evaluateModel
from htmresearch.frameworks.pytorch.modules import (
  Flatten, SparseWeights, KWinners2d, KWinners, updateBoostStrength, rezeroWeights)
from htmresearch.support.expsuite import PyExperimentSuite



class BaselineContinualLearningExperiment(PyExperimentSuite):
  """
  Baseline continual learning experiment based on incremental class learning
  scenario using mnist dataset where we train on two MNIST categories at the
  time while testing of all previous categories:

      - train on [0, 1], test on [0, 1]
      - train on [2, 3], test on [0, 1, 2, 3]
      - ...
      - train on [8, 9], test on [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  """


  def __init__(self):
    super(BaselineContinualLearningExperiment, self).__init__()
    self.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def do_experiment(self, params):
    self.restore_supported = ("restore_supported" in self.cfgparser.defaults() and
                              self.cfgparser.getboolean("DEFAULT", "restore_supported"))
    super(BaselineContinualLearningExperiment, self).do_experiment(params)


  def reset(self, params, repetition):

    self.initialize(params, repetition)

    # Load MNIST dataset
    dataDir = params.get('dataDir', 'data')
    train = datasets.MNIST(dataDir, train=True, download=True, transform=self.transform)
    test = datasets.MNIST(dataDir, train=False, download=True, transform=self.transform)

    # Split mnist dataset into 5 separate datasets.
    # One dataset for each task: [0,1], [2,3], ..., [8,9]
    groupByTask = lambda x: x[1] // 2
    self.train_datasets = splitDataset(train, groupByTask)
    self.test_datasets = splitDataset(test, groupByTask)

    # Assume weight_sparsity == 1.0 for dense networks
    if self.weight_sparsity < 1.0:
      self.model = self.createSparseCNNModel()
    else:
      self.model = self.createDenseCNNModel()

    self.optimizer = self.createOptimizer(self.model)
    self.lr_scheduler = self.createLearningRateScheduler(self.optimizer)


  def initialize(self, params, repetition):
    """
    Initialize experiment parameters and default values from configuration file
    """
    self.name = params["name"]
    self.dataDir = params.get("datadir", "data")
    self.seed = params.get("seed", 42) + repetition

    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    random.seed(self.seed)

    # Training
    self.epochs = params.get("epochs", 1)
    self.batch_size = params.get("batch_size", 16)
    self.batches_in_epoch = params.get("batches_in_epoch", 60000)
    self.first_epoch_batch_size = params.get("first_epoch_batch_size", self.batch_size)
    self.batches_in_first_epoch = params.get("batches_in_first_epoch", self.batches_in_epoch)

    # Testing
    self.test_batch_size = params.get("test_batch_size", 1000)

    # Optimizer
    self.optimizer_class = eval(params.get("optimizer", "torch.optim.SGD"))
    self.optimizer_params = eval(params.get("optimizer_params", "{}"))
    self.lr_scheduler_class = eval(params.get("lr_scheduler", None))
    self.lr_scheduler_params = eval(params.get("lr_scheduler_params", "{}"))
    self.loss_function = eval(params.get("loss_function", "torch.nn.functional.nll_loss"))

    # CNN parameters
    c, h, w = map(int, params.get("input_shape", "1_28_28").split("_"))
    self.in_channels = c
    self.out_channels = map(int, params.get("out_channels", "30_30").split("_"))
    self.kernel_size = map(int, params.get("kernel_size", "5_5").split("_"))
    self.stride = map(int, params.get("stride", "1_1").split("_"))
    self.padding = map(int, params.get("padding", "0_0").split("_"))

    # Compute Flatten CNN output len
    self.maxpool = []
    self.maxpool.append(
      ((w + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1) // 2)
    self.maxpool.append(
      ((self.maxpool[0] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1) // 2)

    self.cnn_output_len = [self.maxpool[i] * self.maxpool[i] * self.out_channels[i]
                           for i in range(len(self.maxpool))]

    # Linear parameteers
    self.n = params.get("n", 1000)
    self.output_size = params.get("output_size", 10)

    # Sparse parameters
    if "c1_k" in params:
      self.cnn_k = map(int, params["c1_k"].split("_"))
    else:
      self.cnn_k = self.cnn_output_len

    self.k = params.get("k", self.n)
    self.k_inference_factor = params.get("k_inference_factor", 1.0)
    self.boost_strength = params.get("boost_strength", 1.0)
    self.boost_strength_factor = params.get("boost_strength_factor", 1.0)
    self.weight_sparsity = params.get("weight_sparsity", 1.0)
    self.weight_sparsity_cnn = params.get("weight_sparsity_cnn", 1.0)


  def createDenseCNNModel(self):
    """
    Create a standard network composed of two CNN / MaxPool layers followed by a
    linear layer with using ReLU activation between the layers
    """

    # Create denseCNN2 model
    model = nn.Sequential(
      nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels[0],
                kernel_size=self.kernel_size[0], stride=self.stride[0],
                padding=self.padding[0]),
      nn.MaxPool2d(kernel_size=2),
      nn.ReLU(),

      nn.Conv2d(in_channels=self.out_channels[0], out_channels=self.out_channels[1],
                kernel_size=self.kernel_size[1], stride=self.stride[1],
                padding=self.padding[1]),
      nn.MaxPool2d(kernel_size=2),
      nn.ReLU(),

      Flatten(),

      nn.Linear(self.cnn_output_len[1], self.n),
      nn.ReLU(),

      nn.Linear(self.n, self.output_size),
      nn.LogSoftmax(dim=1)
    )
    model.to(self.device)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

    return model


  def createSparseCNNModel(self):
    """
    Create a sparse network composed of two CNN / MaxPool layers followed by a
    sparse linear layer with using k-winner activation between the layers
    """
    # Create sparseCNN2 model
    model = nn.Sequential(
      nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels[0],
                kernel_size=self.kernel_size[0], stride=self.stride[0],
                padding=self.padding[0]),
      nn.MaxPool2d(kernel_size=2),
      KWinners2d(n=self.cnn_output_len[0], k=self.cnn_k[0],
                 channels=self.out_channels[0],
                 kInferenceFactor=self.k_inference_factor,
                 boostStrength=self.boost_strength,
                 boostStrengthFactor=self.boost_strength_factor),

      nn.Conv2d(in_channels=self.out_channels[0], out_channels=self.out_channels[1],
                kernel_size=self.kernel_size[1], stride=self.stride[1],
                padding=self.padding[1]),
      nn.MaxPool2d(kernel_size=2),
      KWinners2d(n=self.cnn_output_len[1], k=self.cnn_k[1],
                 channels=self.out_channels[1],
                 kInferenceFactor=self.k_inference_factor,
                 boostStrength=self.boost_strength,
                 boostStrengthFactor=self.boost_strength_factor),

      Flatten(),

      SparseWeights(
        nn.Linear(self.cnn_output_len[1], self.n), self.weight_sparsity),
      KWinners(n=self.n, k=self.k, kInferenceFactor=self.k_inference_factor,
               boostStrength=self.boost_strength,
               boostStrengthFactor=self.boost_strength_factor),

      nn.Linear(self.n, self.output_size),
      nn.LogSoftmax(dim=1)
    )

    model.to(self.device)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

    return model


  def createLearningRateScheduler(self, optimizer):
    """
    Creates the learning rate scheduler and attach the optimizer
    """
    if self.lr_scheduler_class is None:
      return None

    return self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)


  def createOptimizer(self, model):
    """
    Create a new instance of the optimizer
    """
    return self.optimizer_class(model.parameters(), **self.optimizer_params)


  def iterate(self, params, repetition, iteration):

    # Use 'iterations' to represent the task (0=[0-1], ..,5=[8-9])
    task = iteration

    position = self.cfgparser.sections().index(self.name) * 2
    for epoch in tqdm.trange(self.epochs, position=position,
                             desc="{}:{}".format(self.name, task)):
      if epoch == 0:
        batch_size = self.first_epoch_batch_size
        batches_in_epoch = self.batches_in_first_epoch
      else:
        batch_size = self.batch_size
        batches_in_epoch = self.batches_in_epoch

      # Train on a single task
      train_loader = torch.utils.data.DataLoader(dataset=self.train_datasets[task],
                                                 batch_size=batch_size,
                                                 shuffle=True)
      self.preEpoch()
      trainModel(model=self.model, loader=train_loader,
                 optimizer=self.optimizer, device=self.device,
                 batches_in_epoch=batches_in_epoch,
                 criterion=self.loss_function,
                 progress_bar={"desc": "training", "position": position + 1})
      self.postEpoch()

    # Test on all trained tasks combined
    combined_datasets = ConcatDataset(self.test_datasets[:task + 1])
    test_loader = torch.utils.data.DataLoader(dataset=combined_datasets,
                                              batch_size=self.test_batch_size,
                                              shuffle=True)
    return evaluateModel(model=self.model, device=self.device,
                         loader=test_loader,
                         criterion=self.loss_function,
                         progress={"desc": "testing", "position": position + 1})


  def preEpoch(self):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()


  def postEpoch(self):
    self.model.apply(updateBoostStrength)
    self.model.apply(rezeroWeights)


  def save_state(self, params, rep, n):
    saveDir = os.path.join(params["path"], params["name"], "model.{}.{}.pt".format(rep, n))
    torch.save(self.model, saveDir)


  def restore_state(self, params, rep, n):
    saveDir = os.path.join(params["path"], params["name"], "model.{}.{}.pt".format(rep, n))
    self.model = torch.load(saveDir, map_location=self.device)


  def finalize(self, params, rep):
    if params.get("restore_supported", False):
      saveDir = os.path.join(params["path"], params["name"], "model.final.pt")
      torch.save(self.model, saveDir)



if __name__ == '__main__':
  suite = BaselineContinualLearningExperiment()
  suite.start()
