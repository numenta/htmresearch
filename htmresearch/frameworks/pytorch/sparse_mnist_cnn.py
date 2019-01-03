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

from htmresearch.frameworks.pytorch.k_winners_cnn import KWinners

import matplotlib
matplotlib.use('Agg')

def CNNOutputSize(imageShape, outChannels, kernelSize, stride=1, padding=0):
  """
  Computes the output shape of the CNN for a given image before maxPooling,
  ignoring dilation and groups.

  math::
  H_{out} = \lfloor
    \frac{H_{in} + 2 \times \text{padding} - \text{kernelSize}} {\text{stride}}
    + 1 \rfloor

  W_{out} = \lfloor
    \frac{W_{in} + 2 \times \text{padding} - \text{kernelSize}} {\text{stride}}
    + 1 \rfloor

  :param imageShape: tuple: (H_in, W_in)

  :return: (C_out, H_out, W_out, N) where N = C_out * H_out * W_out)

  """
  hout = math.floor(
    (imageShape[0] + 2 * padding - kernelSize) / stride + 1)
  wout = math.floor(
    (imageShape[1] + 2 * padding - kernelSize) / stride + 1)

  return outChannels, hout, wout, outChannels*hout*wout


class SparseMNISTCNN(nn.Module):

  def __init__(self,
               c1OutChannels=20,
               c1k=20,
               n=50,
               useDropout=True,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               boostStrength=1.0,
               boostStrengthFactor=1.0,
               imageSize=(1,28,28)):
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


    We consider three possibilities for sparse CNNs:

    1) Treat the output as a sparse linear layer as if the weights were not
       shared. Do global inhibition across the whole layer, and accumulate
       duty cycles across all units as if they were all distinct. This makes
       little sense.

    2) Treat the output as a sparse linear layer but do consider weight sharing.
       Do global inhibition across the whole layer, but accumulate duty cycles
       across the c1OutChannels filters (it is possible that a given filter has
       multiple active outputs per image). This is simpler to implement and may
       be a decent approach for smaller images such as MNIST. It requires fewer
       filters to get our SDR properties.

    3) Do local inhibition. Do inhibition within each set of filters such
       that each location has at least k active units. Accumulate duty cycles
       across the c1OutChannels filters (it is possible that a given filter has
       multiple active outputs per image). The downside of this approach is that
       we will force activity even in blank areas of the image, which could even
       be negative. To counteract that we would want something like the spatial
       pooler's stimulusThreshold, so that only positive activity gets
       transmitted. Another downside is that we may need a large number of
       filters to get SDR properties. Overall this may be a good approach for
       larger color images and complex domains but may be too heavy handed for
       MNIST.

    In all of the above cases we would ensure that each filter has a sparse
    set of weights. In addition we can optionally enforce sparsity in the
    linear layers after the CNN layers.

    """
    super(SparseMNISTCNN, self).__init__()

    assert(weightSparsity >= 0)
    # assert(c1k <= c1OutChannels)

    # Hyperparameters
    self.c1k = c1k
    self.c1OutChannels = c1OutChannels
    self.n = n
    self.kInferenceFactor = kInferenceFactor
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.useDropout = useDropout
    self.kernelSize = 5

    # First convolutional layer
    self.c1 = nn.Conv2d(imageSize[0], c1OutChannels, kernel_size=5)

    # Compute the number of outputs of c1 after maxpool. We always use a stride
    # of 1 for CNN1, 2 for maxpool, with no padding for either.
    self.c1Shape = CNNOutputSize((imageSize[1], imageSize[2]), c1OutChannels,
                                 kernelSize=self.kernelSize)
    self.c1MaxpoolWidth = int(math.floor(self.c1Shape[2]/2.0))
    self.c1OutputLength = int(self.c1MaxpoolWidth * self.c1MaxpoolWidth
                           * c1OutChannels)

    # First fully connected layer and the fully connected output layer
    self.fc1 = nn.Linear(self.c1OutputLength, n)
    self.fc2 = nn.Linear(n, 10)

    self.learningIterations = 0

    # Boosting related variables
    self.dutyCyclePeriod = 1000
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.register_buffer("dutyCycle", torch.zeros((1, self.c1OutChannels, 1, 1)))

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

    if not self.training:
      k = min(int(round(self.c1k * self.kInferenceFactor)), self.c1OutChannels)
    else:
      k = self.c1k

    x = self.c1(x)
    x = F.max_pool2d(x, 2)

    if self.c1k < self.c1OutputLength:
      x = KWinners.apply(x, self.dutyCycle, k, self.boostStrength)
    else:
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

      # Only need to update C1 dutycycle if c1k < c1OutChannels
      # if k < self.c1OutChannels:
      #   period = min(self.dutyCyclePeriod, self.learningIterations)
      #   self.dutyCycle.mul_(period - batchSize)
      #   self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
      #   self.dutyCycle.div_(period)

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
