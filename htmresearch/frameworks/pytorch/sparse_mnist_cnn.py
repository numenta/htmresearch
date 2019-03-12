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

from htmresearch.frameworks.pytorch.linear_sdr import LinearSDR
from htmresearch.frameworks.pytorch.cnn_sdr import CNNSDR2d


class SparseMNISTCNN(nn.Module):

  def __init__(self,
               c1OutChannels=20,
               c1k=20,
               n=50,
               k=50,
               dropout=0.5,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               boostStrength=1.0,
               boostStrengthFactor=1.0,
               imageShape=(1, 28, 28)):
    """
    A network with hidden CNN layers, which can be k-sparse linear layers. The
    CNN layers are followed by a fully connected hidden layer followed by an
    output layer. Designed for MNIST.

    :param c1OutChannels:
      Number of channels (filters) in the first convolutional layer C1.

    :param c1k:
      Number of ON (non-zero) units per iteration in the first convolutional
      layer C1. The sparsity of this layer will be
      c1k / self.cnnSdr1.outputLength. If c1k >= self.cnnSdr1.outputLength, the
      layer acts as a traditional convolutional layer.

    :param n:
      Number of units in the fully connected hidden layer

    :param k:
      Number of ON units in the fully connected hidden layer. The sparsity of
      this layer will be k / n. If k >= n, the layer acts as a traditional
      fully connected RELU layer.

    :param dropout:
      dropout probability used to train the second and subsequent layers.
      A value 0.0 implies no dropout

    :param kInferenceFactor:
      During inference (training=False) we increase c1k and l2k by this factor.

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero in the fully connected
      layers.

    :param boostStrength:
      boost strength (0.0 implies no boosting).

    :param boostStrengthFactor:
      boost strength is multiplied by this factor after each epoch.
      A value < 1.0 will decrement it every epoch.

    :param imageShape:
      A tuple representing (in_channels,height,width).


    We considered three possibilities for sparse CNNs. The second one is
    currently implemented.

    1) Treat the output as a sparse linear layer as if the weights were not
       shared. Do global inhibition across the whole layer, and accumulate
       duty cycles across all units as if they were all distinct. This makes
       little sense.

    2) Treat the output as a sparse global layer but do consider weight sharing.
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

    """
    super(SparseMNISTCNN, self).__init__()

    assert(weightSparsity >= 0)
    assert(imageShape[1] == imageShape[2],
           "sparseCNN only supports square images")

    # Hyperparameters
    self.c1k = c1k
    self.c1OutChannels = c1OutChannels
    self.kInferenceFactor = kInferenceFactor
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.dropout = dropout
    self.kernelSize = 5

    # First convolutional layer
    self.cnnSdr1 = CNNSDR2d(imageShape=imageShape, outChannels=c1OutChannels,
                            k=c1k, kernelSize=self.kernelSize,
                            kInferenceFactor=kInferenceFactor,
                            boostStrength=boostStrength)

    # First linear SDR layer
    self.linearSdr1 = LinearSDR(inputFeatures=self.cnnSdr1.outputLength,
                                n=n,
                                k=k,
                                kInferenceFactor=kInferenceFactor,
                                weightSparsity=weightSparsity,
                                boostStrength=boostStrength
                                )

    # ...and the fully connected linear output layer
    self.linearOutput = nn.Linear(n, 10)

    # Boosting related variables
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor


  def postEpoch(self):
    """
    Call this once after each training epoch. Currently just updates
    boostStrength
    """
    if self.training:
      self.setBoostStrength(self.boostStrength * self.boostStrengthFactor)

      # The optimizer is updating the weights during training after the forward
      # step. Therefore we need to re-zero the weights after every epoch
      self.linearSdr1.rezeroWeights()
      self.cnnSdr1.rezeroWeights()


  def setBoostStrength(self, b):
    self.boostStrength = b
    self.linearSdr1.setBoostStrength(b)
    self.cnnSdr1.setBoostStrength(b)


  def forward(self, x):
    # CNN layer
    x = self.cnnSdr1(x)

    # Linear layer
    x = x.view(-1, self.cnnSdr1.outputLength)
    x = self.linearSdr1(x)

    # If requested, apply dropout to fully connected layer
    if self.dropout > 0.0:
      x = F.dropout(x, p=self.dropout, training=self.training)

    # Linear output layer
    x = self.linearOutput(x)
    x = F.log_softmax(x, dim=1)

    return x


  def getLearningIterations(self):
    return self.linearSdr1.getLearningIterations()


  def maxEntropy(self):
    """
    Returns the maximum entropy we can expect from level 1
    """
    return self.cnnSdr1.maxEntropy() + self.linearSdr1.maxEntropy()


  def entropy(self):
    """
    Returns the current entropy, scaled properly
    """
    return self.cnnSdr1.entropy() + self.linearSdr1.entropy()

