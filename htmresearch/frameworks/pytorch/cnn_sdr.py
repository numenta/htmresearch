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

import torch
import torch.nn as nn
import torch.nn.functional as F

from htmresearch.frameworks.pytorch.k_winners import (
  KWinnersCNN, updateDutyCycleCNN
)

from htmresearch.frameworks.pytorch.duty_cycle_metrics import (
  maxEntropy, binaryEntropy
)

class CNNSDR2d(nn.Module):

  def __init__(self,
               imageShape=(1, 28, 28),
               outChannels=20,
               k=20,
               kernelSize=5,
               kInferenceFactor=1.5,
               boostStrength=1.0,
               ):
    """
    A sparse CNN layer with fixed sparsity and boosting. We do not yet support
    weight sparsity for CNNs.

    :param imageShape:
      A tuple representing (in_channels,height,width).

    :param outChannels:
      Number of channels (filters) in this convolutional layer.

    :param k:
      Number of ON (non-zero) units per iteration in this convolutional layer.
      The sparsity of this layer will be k / self.outputLength. If k >=
      self.outputLength, the layer acts as a traditional convolutional layer.

    :param kernelSize:
      Size of the CNN kernel.

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    """

    super(CNNSDR2d, self).__init__()
    self.outChannels = outChannels
    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.kernelSize = kernelSize
    self.imageShape = imageShape
    self.stride = 1
    self.padding = 0

    self.cnn = nn.Conv2d(imageShape[0], outChannels, kernel_size=kernelSize)

    # Compute the number of outputs of c1 after maxpool. We always use a stride
    # of 1 for CNN, 2 for maxpool, with no padding for either.
    shape = self.outputSize()
    self.maxpoolWidth = int(math.floor(shape[2] / 2.0))
    self.outputLength = int(self.maxpoolWidth * self.maxpoolWidth * outChannels)

    # Boosting related variables
    self.learningIterations = 0
    self.dutyCyclePeriod = 1000
    self.boostStrength = boostStrength
    self.register_buffer("dutyCycle", torch.zeros((1, self.outChannels, 1, 1)))


  def forward(self, x):
    batchSize = x.shape[0]
    self.learningIterations += batchSize

    if self.training:
      k = self.k
    else:
      k = min(int(round(self.k * self.kInferenceFactor)), self.outputLength)

    x = self.cnn(x)
    x = F.max_pool2d(x, 2)

    if k < self.outputLength:
      x = KWinnersCNN.apply(x, self.dutyCycle, k, self.boostStrength)

      # Update moving average of duty cycle for training iterations only
      # During inference this is kept static.
      updateDutyCycleCNN(x, self.dutyCycle,
                         self.dutyCyclePeriod, self.learningIterations)
    else:
      x = F.relu(x)

    return x


  def outputSize(self):
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
      (self.imageShape[0] + 2 * self.padding - self.kernelSize) / self.stride + 1)
    wout = math.floor(
      (self.imageShape[1] + 2 * self.padding - self.kernelSize) / self.stride + 1)

    return self.outChannels, hout, wout, self.outChannels * hout * wout



  def setBoostStrength(self, b):
    self.boostStrength = b


  def getLearningIterations(self):
    return self.learningIterations


  def maxEntropy(self):
    """
    Returns the maximum entropy we can expect from this layer
    """
    return maxEntropy(self.outputLength, self.k)


  def entropy(self):
    """
    Returns the current total entropy of this layer
    """
    _, entropy = binaryEntropy(self.dutyCycle)
    return entropy * self.maxpoolWidth * self.maxpoolWidth


