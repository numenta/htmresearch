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
from htmresearch.frameworks.pytorch.duty_cycle_metrics import (
  maxEntropy, binaryEntropy
)

class SparseMNISTNet(nn.Module):

  def __init__(self, n=2000,
               k=200,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               boostStrength=1.0,
               boostStrengthFactor=1.0,
               dropout=0.0):
    """
    A network with one hidden layer, which is a k-sparse linear layer, designed
    for MNIST.

    :param n:
      Number of units in the hidden layer.

    :param k:
      Number of ON (non-zero) units per iteration.

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero.

    :param boostStrength:
      boost strength (0.0 implies no boosting).

    :param boostStrengthFactor:
      boost strength is multiplied by this factor after each epoch.
      A value < 1.0 will decrement it every epoch.
    :param dropout:
      dropout probability used to train the second and subsequent layers.
      A value 0.0 implies no dropout
    """
    super(SparseMNISTNet, self).__init__()

    assert(weightSparsity >= 0)
    assert(k <= n)

    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.n = n
    self.l1 = nn.Linear(28*28, self.n)
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.l2 = nn.Linear(self.n, 10)
    self.learningIterations = 0
    self.dropout = dropout

    # Boosting related variables
    self.dutyCyclePeriod = 1000
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.register_buffer("dutyCycle", torch.zeros(self.n))

    # For each L1 unit, decide which weights are going to be zero
    if self.weightSparsity < 1.0:
      outputSize, inputSize = self.l1.weight.shape
      numZeros = int(round((1.0 - self.weightSparsity) * inputSize))

      outputIndices = np.arange(outputSize)
      inputIndices = np.array([np.random.permutation(inputSize)[:numZeros]
                               for _ in outputIndices], dtype=np.long)

      # Create tensor indices for all non-zero weights
      zeroIndices = np.empty((outputSize, numZeros, 2), dtype=np.long)
      zeroIndices[:, :, 0] = outputIndices[:, None]
      zeroIndices[:, :, 1] = inputIndices
      zeroIndices = torch.LongTensor(zeroIndices.reshape(-1, 2))

      self.zeroWts = (zeroIndices[:, 0], zeroIndices[:, 1])
      self.rezeroWeights()


  def rezeroWeights(self):
    if self.weightSparsity < 1.0:
      # print("non zero before:",self.l1.weight.data.nonzero().shape)
      self.l1.weight.data[self.zeroWts] = 0.0
      # print("non zero after:",self.l1.weight.data.nonzero().shape)


  def postEpoch(self):
    """
    Call this once after each training epoch.
    """
    self.boostStrength = self.boostStrength * self.boostStrengthFactor
    print("boostStrength is now:", self.boostStrength)


  def forward(self, x):

    if not self.training:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)
    else:
      k = self.k

    # First hidden layer
    x = x.view(-1, 28*28)

    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    if k != self.n:
      x = KWinners.apply(self.l1(x), self.dutyCycle, k, self.boostStrength)
    else:
      x = F.relu(self.l1(x))

    if self.training:
      # Update moving average of duty cycle for training iterations only
      # During inference this is kept static.
      batchSize = x.shape[0]
      self.learningIterations += batchSize

      # Only need to update dutycycle if if k < n
      if k != self.n:
        period = min(self.dutyCyclePeriod, self.learningIterations)
        self.dutyCycle.mul_(period - batchSize)
        self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.dutyCycle.div_(period)

    # Dropout
    if self.dropout > 0.0:
      x = F.dropout(x, p=self.dropout, training=self.training)

    # Output layer
    x = self.l2(x)
    x = F.log_softmax(x, dim=1)

    return x


  def maxEntropy(self):
    """
    Returns the maximum entropy we can expect from level 1
    """
    return maxEntropy(self.n, self.k)


  def entropy(self):
    """
    Returns the current entropy
    """
    _, entropy = binaryEntropy(self.dutyCycle)
    return entropy

