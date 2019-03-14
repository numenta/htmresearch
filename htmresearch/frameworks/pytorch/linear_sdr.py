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
import math
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
               n=500,
               k=50,
               kInferenceFactor=1.5,
               weightSparsity=0.4,
               boostStrength=1.0,
               useBatchNorm=True,
               normalizeWeights=False
               ):
    """
    A sparse linear layer with fixed sparsity, weight sparsity, and boosting.

    :param inputFeatures:
      The size of the input to this layer

    :param n:
      Number of units in this layer

    :param k:
      Number of ON units in this layer. The sparsity of this layer will be
      k / n. If k <= 0 or k >= n, the layer acts as a traditional fully
      connected RELU layer.

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero in the layer.

    :param boostStrength:
      Boost strength (0.0 implies no boosting).

    :param useBatchNorm:
      If True, applies batchNorm1D after the linear step, before k-winners
      is applied.
    :param normalizeWeights:
      If True, initialize weights normalizing to the number of non-zeros

    """
    super(LinearSDR, self).__init__()
    self.in_features = inputFeatures
    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.n = n
    self.l1 = nn.Linear(inputFeatures, self.n)
    self.weightSparsity = weightSparsity
    self.learningIterations = 0
    if self.k <= 0:
      self.k = self.n

    self.bn = None
    if useBatchNorm:
      self.bn = nn.LayerNorm(self.n)

    if normalizeWeights:
      self.normalizeWeights()

    # Boosting related variables
    self.dutyCyclePeriod = 1000
    self.boostStrength = boostStrength
    if k < n:
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


  def normalizeWeights(self):
    """
    Initialize the weights using kaiming_uniform initialization normalized to
    the number of non-zeros in the layer instead of the whole input size.
    Similar to torch.nn.Linear.reset_parameters() but applying weight sparsity
    to the input size
    """
    _, inputSize = self.l1.weight.shape
    fan = int(inputSize * self.weightSparsity)
    gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    nn.init.uniform_(self.l1.weight, -bound, bound)
    if self.l1.bias is not None:
        bound = 1 / math.sqrt(fan)
        nn.init.uniform_(self.l1.bias, -bound, bound)


  def rezeroWeights(self):
    if self.weightSparsity < 1.0:
      self.l1.weight.data[self.zeroWts] = 0.0


  def forward(self, x):
    if self.training:
      self.rezeroWeights()
      k = self.k
    else:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)

    x = self.l1(x)

    # Use batch norm if requested
    if self.bn is not None:
      x = self.bn(x)

    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    if k < self.n:
      x = KWinners.apply(x, self.dutyCycle, k, self.boostStrength)
    else:
      x = F.relu(x)

    # Update moving average of duty cycle for training iterations only
    # During inference this is kept static.
    if self.training:
      batchSize = x.shape[0]
      self.learningIterations += batchSize

      # Only need to update dutycycle if k < n
      if k < self.n:
        period = min(self.dutyCyclePeriod, self.learningIterations)
        self.dutyCycle.mul_(period - batchSize)
        self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.dutyCycle.div_(period)

    return x


  def setBoostStrength(self, b):
    self.boostStrength = b


  def getLearningIterations(self):
    return self.learningIterations


  def maxEntropy(self):
    """
    Returns the maximum total entropy we can expect from this layer
    """
    return maxEntropy(self.n, self.k)


  def entropy(self):
    """
    Returns the current total entropy of this layer
    """
    if self.k < self.n:
      _, entropy = binaryEntropy(self.dutyCycle)
      return entropy
    else:
      return 0

  def pruneDutycycles(self, threshold=0.0):
    """
    Prune all the units whose dutycycles absolute magnitude is less than
    `threshold * k/n`
    :param threshold: min threshold to prune. If less than zero then no pruning
    :type threshold: float
    """
    if threshold < 0.0:
      return

    if not hasattr(self, 'dutyCycle'):
      return

    # See KWinners
    targetDesity = float(self.k) / float(self.n)

    # Units to keep
    mask = torch.ge(torch.abs(self.dutyCycle), targetDesity * threshold)
    mask = mask.type(torch.float32).view(self.n, 1)

    # Zero weights with low dutycycles
    self.l1.weight.data.mul_(mask)
