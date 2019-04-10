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
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms, datasets

from htmresearch.frameworks.pytorch.model_utils import trainModel, evaluateModel
from htmresearch.frameworks.pytorch.image_transforms import RandomNoise
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

    self.transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    self.transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def do_experiment(self, params):
    self.restore_supported = ("restore_supported" in self.cfgparser.defaults() and
                              self.cfgparser.getboolean("DEFAULT", "restore_supported"))
    super(CIFARExperiment, self).do_experiment(params)


  def reset(self, params, repetition):

    pprint.pprint(params)

    self.initialize(params, repetition)

    # Load CIFAR dataset
    dataDir = params.get('dataDir', 'data')
    self.trainset = datasets.CIFAR10(root=dataDir, train=True, download=True,
                                transform=self.transform_train)
    self.testset = datasets.CIFAR10(root=dataDir, train=False, download=True,
                               transform=self.transform_test)

    if self.dense:
      self.model = NotSoDenseNet(
        nblocks=self.nblocks,
        growth_rate=self.growth_rate,
        dense_c1_out_planes=self.dense_c1_out_planes,
        avg_pool_size=self.avg_pool_size,
        dense_sparsities=[1.0] * 4,
        transition_sparsities=[1.0] * 3,
        linear_sparsity=0.0,
      )
    else:
      self.model = NotSoDenseNet(
        nblocks=self.nblocks,
        growth_rate=self.growth_rate,
        dense_c1_out_planes=self.dense_c1_out_planes,
        dense_sparsities=self.dense_sparsities,
        transition_sparsities=self.transition_sparsities,
        linear_sparsity=self.linear_sparsity,
        linear_weight_sparsity=self.linear_weight_sparsity,
        linear_n=self.linear_n,
        avg_pool_size=self.avg_pool_size,
      )


    print("Torch reports", torch.cuda.device_count(), "GPUs available")
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs")
      self.model = torch.nn.DataParallel(self.model)

    print("Setting device to", self.device)
    self.model.to(self.device)

    self.optimizer = self.createOptimizer(self.model)
    self.lr_scheduler = self.createLearningRateScheduler(self.optimizer)


  def initialize(self, params, repetition):
    """
    Initialize experiment parameters and default values from configuration file.
    Called at the beginning of each experiment and each repetition.
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
    self.batches_in_first_epoch = params.get("batches_in_first_epoch", 100000)

    # Testing
    self.test_batch_size = params.get("test_batch_size", 1000)
    self.test_batches_in_epoch = params.get("test_batches_in_epoch", 100000)

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
    self.dense = params.get("dense", False)
    self.growth_rate = params.get("growth_rate", 12)
    self.nblocks = map(int,
                       params.get("nblocks", "6, 12, 24, 16").split(", "))
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
    t1 = time.time()

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
    print("Learning rate: {:f}".format(self.lr_scheduler.get_lr()[0]))


    trainModel(model=self.model, loader=train_loader,
               optimizer=self.optimizer, device=self.device,
               batches_in_epoch=batches_in_epoch,
               criterion=self.loss_function)
    self.postEpoch()

    print("Training time for epoch=", time.time() - t1)

    # Test on all trained tasks combined
    test_loader = torch.utils.data.DataLoader(dataset=self.testset,
                                              batch_size=self.test_batch_size,
                                              shuffle=True)
    ret = evaluateModel(model=self.model, device=self.device,
                         loader=test_loader,
                         batches_in_epoch=self.test_batches_in_epoch,
                         criterion=self.loss_function)

    print("Test loss = {:5.4f}, Accuracy = {:5.3f}%".format(ret["loss"], 100.0*ret["accuracy"]))
    print("Full epoch time =", time.time() - t1)

    if iteration==params["iterations"]-1:
      self.runNoiseTests()

    # Include learning rate stats
    ret.update({"learning_rate": self.lr_scheduler.get_lr()[0]})

    return ret


  def runNoiseTests(self):
    """
    Test the model with different noise values and return test metrics.
    """

    print("\nRunning noise tests")
    ret = {}

    # Test with noise
    total_correct = 0
    for noise in [0.0, 0.1]:

      transform_noise_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        RandomNoise(noise,
                    whiteValue=0.5 + 2 * 0.20,
                    blackValue=0.5 - 2 * 0.2),
      ])

      testset = datasets.CIFAR10(root=self.dataDir,
                                 train=False,
                                 download=True,
                                 transform=transform_noise_test)
      test_loader = DataLoader(testset,
                               batch_size=self.test_batch_size,
                               shuffle=False)

      testResult = evaluateModel(
        model=self.model,
        loader=test_loader,
        device=self.device,
        batches_in_epoch=self.test_batches_in_epoch,
        criterion=self.loss_function
      )
      total_correct += testResult["total_correct"]
      ret[noise] = testResult


    ret["total_correct"] = total_correct

    print("Noise results:")
    pprint.pprint(ret)

    return ret


  def preEpoch(self):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()


  def postEpoch(self):
    self.model.postEpoch()


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
  suite = CIFARExperiment()
  suite.start()
