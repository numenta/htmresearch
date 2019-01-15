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

from htmresearch.frameworks.pytorch.duty_cycle_metrics import (
  maxEntropy
)
from htmresearch.frameworks.pytorch.linear_sdr import LinearSDR


class SparseLinearNet(nn.Module):

  def __init__(self,
               n=2000,
               k=200,
               inputSize=28*28,
               outputSize=10,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               boostStrength=1.0,
               boostStrengthFactor=1.0,
               dropout=0.0):
    """
    A network with one or more hidden layers, which is a k-sparse linear layer.

    :param n:
      Number of units in each hidden layer.
    :type n: int or list[int]

    :param k:
      Number of ON (non-zero) units per iteration in each hidden layer.
    :type k: int or list[int]

    :param inputSize:
      Total dimensionality of input vector. We apply view(-1, inputSize)
      to the data before passing it to LinearSDR.
    :type inputSize: int

    :param outputSize:
      Total dimensionality of output vector
    :type outputSize: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero.
    :type weightSparsity: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      boost strength is multiplied by this factor after each epoch.
      A value < 1.0 will decrement it every epoch.
    :type boostStrengthFactor: float

    :param dropout:
      dropout probability used to train the second and subsequent layers.
      A value 0.0 implies no dropout
    :type dropout: float
    """
    super(SparseLinearNet, self).__init__()

    if type(k) is not list:
      k = [k]
    if type(n) is not list:
      n = [n]
    assert(len(n) == len(k))
    assert(weightSparsity >= 0)
    for i in range(len(n)):
      assert(k[i] <= n[i])

    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.n = n
    self.inputSize = inputSize
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.boostStrengthFactor = boostStrengthFactor
    self.boostStrength = boostStrength

    # Hidden layers
    inputFeatures = inputSize
    for i in range(len(n)):
      self.add_module("linearSdr{}".format(i+1),
                      LinearSDR(inputFeatures=inputFeatures,
                                n=n[i],
                                k=k[i],
                                kInferenceFactor=kInferenceFactor,
                                weightSparsity=weightSparsity,
                                boostStrength=boostStrength))
      # Add dropout after each hidden layer
      if dropout > 0.0:
        self.add_module("dropout", nn.Dropout(dropout))

      # Feed this layer output into next layer input
      inputFeatures = n[i]

    # Add one fully connected layer after all hidden layers
    self.add_module("fc", nn.Linear(self.n[-1], outputSize))
    self.add_module("logSoftmax", nn.LogSoftmax(dim=1))


  def postEpoch(self):
    """
    Call this once after each training epoch.
    """
    if self.training:
      self.boostStrength = self.boostStrength * self.boostStrengthFactor
      for module in self.children():
        if type(module) == LinearSDR:
          module.setBoostStrength(self.boostStrength)

          # The optimizer is updating the weights during training after the forward
          # step. Therefore we need to re-zero the weights after every epoch
          module.rezeroWeights()

      print("boostStrength is now:", self.boostStrength)


  def forward(self, x):
    x = x.view(-1, self.inputSize)
    for module in self.children():
      x = module.forward(x)

    return x


  def getLearningIterations(self):
    return self.linearSdr1.learningIterations
  

  def maxEntropy(self):
    """
    Returns the maximum entropy we can expect
    """
    return sum([maxEntropy(self.n[i], self.k[i])
                for i in range(len(self.n))])


  def entropy(self):
    """
    Returns the current entropy
    """
    entropy = 0
    for module in self.children():
      if type(module) == LinearSDR:
        entropy += module.entropy()

    return entropy

