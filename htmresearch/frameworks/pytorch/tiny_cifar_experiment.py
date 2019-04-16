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

import torch
import torch.nn as nn

from htmresearch.frameworks.pytorch.modules.k_winners import getEntropies
from htmresearch.frameworks.pytorch.modules import (
  SparseWeights2d, KWinners2d, Flatten
)
from htmresearch.frameworks.pytorch.cifar_experiment import CIFARExperiment


class TinyCIFARExperiment(CIFARExperiment):
  """
  Experiment for training on CIFAR-10
  """


  def __init__(self):
    super(TinyCIFARExperiment, self).__init__()

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def initialize(self, params, repetition):
    """
    Initialize experiment parameters and default values from configuration file.
    Called at the beginning of each experiment and each repetition.
    """
    super(TinyCIFARExperiment, self).initialize(params, repetition)
    self.network_type = params.get("network_type", "sparse")


  def logger(self, iteration, ret):
    """Print out relevant information at each epoch"""
    print("Learning rate: {:f}".format(self.lr_scheduler.get_lr()[0]))
    entropies = getEntropies(self.model)
    print("Entropy and max entropy: ", float(entropies[0]), entropies[1])
    print("Training time for epoch=", self.epoch_train_time)
    for noise in self.noise_values:
      print("Noise= {:3.2f}, loss = {:5.4f}, Accuracy = {:5.3f}%".format(
        noise, ret[noise]["loss"], 100.0*ret[noise]["accuracy"]))
    print("Full epoch time =", self.epoch_time)
    if ret[0.0]["accuracy"] > 0.7:
      self.best_noise_score = max(ret[0.1]["accuracy"], self.best_noise_score)
      self.best_epoch = iteration



  def createModel(self, params, repetition):
    if self.network_type == "tiny":
      self.model = self.createTinyModel(self.dense_c1_out_planes)

    elif self.network_type == "tiny_sparse":
      self.model = self.createTinySparseModel(self.dense_c1_out_planes)

    else:
      print("Unknown network type")


  def createTinyModel(self, out_planes):
    return nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=out_planes,
                kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_planes),
      nn.AvgPool2d(kernel_size=2),
      nn.ReLU(),

      nn.Conv2d(in_channels=out_planes, out_channels=out_planes,
                kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_planes),
      nn.AvgPool2d(kernel_size=2),
      nn.ReLU(),

      Flatten(),
      nn.Linear(out_planes*32*2, 10),  # planes * 32 * 32 / 16
    )


  def createTinySparseModel(self, out_planes):
    return nn.Sequential(
      # First layer
      nn.Conv2d(in_channels=3, out_channels=out_planes,
                kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_planes),
      nn.AvgPool2d(kernel_size=2),
      KWinners2d(n=out_planes*32*8, channels=out_planes,
                 k=int(0.1*out_planes*32*8), kInferenceFactor=1.5,
                 boostStrength=1.5, boostStrengthFactor=0.9),

      # Second layer
      SparseWeights2d(
        nn.Conv2d(in_channels=out_planes, out_channels=out_planes,
                  kernel_size=3, padding=1, bias=False),
        weightSparsity=0.5
      ),
      nn.BatchNorm2d(out_planes),
      nn.AvgPool2d(kernel_size=2),
      KWinners2d(n=out_planes * 32 * 2, channels=out_planes,
                 k=int(0.1 * out_planes * 32 * 2), kInferenceFactor=1.5,
                 boostStrength=1.5, boostStrengthFactor=0.9),

      # Output layer
      Flatten(),
      nn.Linear(out_planes*32*2, 10),  # planes * 32 * 32 / 16
    )



if __name__ == '__main__':
  suite = TinyCIFARExperiment()
  suite.start()
