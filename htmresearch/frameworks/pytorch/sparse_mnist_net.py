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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from htmresearch.frameworks.pytorch.k_winners import KWinners

import matplotlib
matplotlib.use('Agg')

class SparseMNISTNet(nn.Module):
  """
  This class implements a sparse MNIST net.
  """

  def __init__(self, n=2000, k=200, weightSparsity=0.25, boostStrength=1.0):
    super(SparseMNISTNet, self).__init__()

    self.k = k
    self.n = n
    self.l1 = nn.Linear(28*28, self.n)
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.zeroWts = []
    self.l2 = nn.Linear(self.n, 10)
    self.learningIterations = 0

    # Boosting related variables
    self.dutyCyclePeriod = 2000
    self.boostStrength = boostStrength
    self.dutyCycle = torch.zeros(self.n)
    self.boostFactors = torch.ones(self.n)

    # For each L1 unit, decide which weights are going to be zero
    numZeros = int(round((1.0 - self.weightSparsity) * self.l1.weight.shape[1]))
    for i in range(self.n):
      self.zeroWts.append(
        np.random.permutation(self.l1.weight.shape[1])[0:numZeros])


    self.rezeroWeights()


  def rezeroWeights(self):
    # print("non zero before:",self.l1.weight.data.nonzero().shape)
    for i in range(self.n):
      self.l1.weight.data[i, self.zeroWts[i]] = 0.0
    # print("non zero after:",self.l1.weight.data.nonzero().shape)


  def forward(self, x):

    # First hidden layer
    x = x.view(-1, 28*28)
    if self.learningIterations > self.dutyCyclePeriod:
      x = KWinners.apply(self.l1(x), self.dutyCycle, self.k, self.boostStrength)
    else:
      x = KWinners.apply(self.l1(x), self.dutyCycle, self.k, 0.0)

    if self.training:
      # Update moving average of duty cycle for training iterations only
      batchSize = x.shape[0]
      self.dutyCycle = (self.dutyCycle * (self.dutyCyclePeriod - batchSize) +
                        ((x > 0).sum(dim=0)).float()) / self.dutyCyclePeriod
      self.learningIterations += batchSize

    # Output layer
    x = self.l2(x)
    x = F.log_softmax(x, dim=1)

    return x


  def printMetrics(self):
    print("Learning Iterations:", self.learningIterations)
    print("non zero weights:", self.l1.weight.data.nonzero().shape,
          "all weights:", self.l1.weight.data.shape)
    print("duty cycles min/max/mean:",
          self.dutyCycle.min(), self.dutyCycle.max(), self.dutyCycle.mean())

