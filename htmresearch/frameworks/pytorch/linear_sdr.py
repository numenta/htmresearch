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

class LinearSDR(nn.Module):

  def __init__(self,
               inputFeatures,
               n=2000,
               k=200,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               boostStrength=1.0,
               ):

    super(LinearSDR, self).__init__()
    self.in_features = inputFeatures
    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.n = n
    self.l1 = nn.Linear(inputFeatures, self.n)
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.learningIterations = 0

    # Boosting related variables
    self.dutyCyclePeriod = 1000
    self.boostStrength = boostStrength
    self.register_buffer("dutyCycle", torch.zeros(self.n))

    # For each unit, decide which weights are going to be zero
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
      self.l1.weight.data[self.zeroWts] = 0.0


  def forward(self, x):

    if not self.training:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)
    else:
      k = self.k

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

    return x


  def setBoostStrength(self, b):
    self.boostStrength = b


  def maxEntropy(self):
    """
    Returns the maximum entropy we can expect from level 1
    """
    return maxEntropy(self.n, self.k)


  def entropy(self):
    """
    Returns the current entropy of this layer
    """
    _, entropy = binaryEntropy(self.dutyCycle)
    return entropy


