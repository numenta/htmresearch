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
import torch.nn as nn
import torch.nn.functional as F

from htmresearch.frameworks.pytorch.duty_cycle_metrics import (
  maxEntropy
)
from htmresearch.frameworks.pytorch.linear_sdr import LinearSDR


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
    self.linearSdr1 = LinearSDR(inputFeatures=28 * 28,
                                n=n,
                                k=k,
                                kInferenceFactor=kInferenceFactor,
                                weightSparsity=weightSparsity,
                                boostStrength=boostStrength
                                )

    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.l2 = nn.Linear(self.n, 10)
    self.dropout = dropout
    self.boostStrengthFactor = boostStrengthFactor
    self.boostStrength = boostStrength


  def postEpoch(self):
    """
    Call this once after each training epoch.
    """
    self.boostStrength = self.boostStrength * self.boostStrengthFactor
    self.linearSdr1.setBoostStrength(self.boostStrength)
    print("boostStrength is now:", self.boostStrength)

    if self.training:
      # The optimizer is updating the weights during training after the forward
      # step. Therefore we need to re-zero the weights after every epoch
      self.linearSdr1.rezeroWeights()


  def forward(self, x):

    # First hidden layer
    x = x.view(-1, 28*28)
    x = self.linearSdr1(x)

    # Dropout
    if self.dropout > 0.0:
      x = F.dropout(x, p=self.dropout, training=self.training)

    # Output layer
    x = self.l2(x)
    x = F.log_softmax(x, dim=1)

    return x


  def getLearningIterations(self):
    return self.linearSdr1.learningIterations
  

  def maxEntropy(self):
    """
    Returns the maximum entropy we can expect from level 1
    """
    return maxEntropy(self.n, self.k)


  def entropy(self):
    """
    Returns the current entropy
    """
    return self.linearSdr1.entropy()

