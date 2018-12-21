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

class SparseMNISTCNN(nn.Module):

  def __init__(self,
               c1OutChannels=20,
               c1k=20,
               n=50,
               useDropout=True,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               boostStrength=1.0,
               boostStrengthFactor=1.0):
    """
    A network with hidden CNN layers, which can be k-sparse linear layers. The
    CNN layers are followed by a fully connected hidden layer followed by an
    output layer. Designed for MNIST.

    :param c1OutChannels:
      Number of channels (filters) in the first convolutional layer C1.

    :param c1k:
      Number of ON (non-zero) filters per iteration in the first convolutional
      layer C1.

    :param n:
      Number of units in the fully connected hidden layer

    :param useDropout:
      If True, dropout will be used to train the second and subsequent layers.

    :param kInferenceFactor:
      During inference (training=False) we increase c1k and l2k by this factor.

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero in the convolutional layer.

    :param boostStrength:
      boost strength (0.0 implies no boosting).

    :param boostStrengthFactor:
      boost strength is multiplied by this factor after each epoch.
      A value < 1.0 will decrement it every epoch.

    """
    super(SparseMNISTCNN, self).__init__()

    assert(weightSparsity >= 0)
    assert(c1k <= c1OutChannels)

    # Hyperparameters
    self.c1k = c1k
    self.c1OutChannels = c1OutChannels
    self.n = n
    self.kInferenceFactor = kInferenceFactor
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.useDropout = useDropout

    # First convolutional layer
    self.c1 = nn.Conv2d(1, c1OutChannels, kernel_size=5)

    # Compute the number of outputs of c1 after maxpool. We always use a stride
    # of 1 for CNN1, 2 for maxpool, with no padding for either.
    self.c1MaxpoolWidth = ((28 - self.c1.kernel_size[0]) + 1)/ 2

    # First fully connected layer and the fully connected output layer
    self.fc1 = nn.Linear(self.c1MaxpoolWidth * self.c1MaxpoolWidth
                         * c1OutChannels, n)
    self.fc2 = nn.Linear(n, 10)

    self.learningIterations = 0

    # Boosting related variables
    self.dutyCyclePeriod = 1000
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.register_buffer("dutyCycle", torch.zeros(self.c1OutChannels))

    #
    # # Weight sparsification. For each unit, decide which weights are going to be zero
    self.zeroWts = []
    # if self.weightSparsity < 1.0:
    #   numZeros = int(round((1.0 - self.weightSparsity) * self.l1.weight.shape[1]))
    #   for i in range(self.n):
    #     self.zeroWts.append(
    #       np.random.permutation(self.l1.weight.shape[1])[0:numZeros])
    #
    #   self.rezeroWeights()


  def rezeroWeights(self):
    pass
    # if self.weightSparsity < 1.0:
    #   # print("non zero before:",self.l1.weight.data.nonzero().shape)
    #   for i in range(self.n):
    #     self.l1.weight.data[i, self.zeroWts[i]] = 0.0
    #   # print("non zero after:",self.l1.weight.data.nonzero().shape)


  def postEpoch(self):
    """
    Call this once after each training epoch. Currently just updates
    boostStrength
    """
    self.boostStrength = self.boostStrength * self.boostStrengthFactor
    print("boostStrength is now:", self.boostStrength)


  def forward(self, x):
    batchSize = x.shape[0]

    x = self.c1(x)
    x = F.max_pool2d(x, 2)
    x = F.relu(x)

    x = x.view(-1, self.c1MaxpoolWidth * self.c1MaxpoolWidth *
                   self.c1OutChannels)
    x = self.fc1(x)
    x = F.relu(x)
    if self.useDropout:
      x = F.dropout(x, training=self.training)
    x = self.fc2(x)

    if self.training:
      # Update moving average of duty cycle for training iterations only
      # During inference this is kept static.
      self.learningIterations += batchSize

    x = F.log_softmax(x, dim=1)

    return x


  def printMetrics(self):
    print("Learning Iterations:", self.learningIterations)
    # print("non zero weights:", self.l1.weight.data.nonzero().shape,
    #       "all weights:", self.l1.weight.data.shape)
    # print("duty cycles min/max/mean:",
    #       self.dutyCycle.min(), self.dutyCycle.max(), self.dutyCycle.mean())


  def printParameters(self):
    print("                1k :", self.l1k)
    print("                 n :", self.n)
    print("    weightSparsity :", self.weightSparsity)
    print("     boostStrength :", self.boostStrength)
    print("   dutyCyclePeriod :", self.dutyCyclePeriod)
    print("   kInferenceFactor:", self.kInferenceFactor)
    print("boostStrengthFactor:", self.boostStrengthFactor)