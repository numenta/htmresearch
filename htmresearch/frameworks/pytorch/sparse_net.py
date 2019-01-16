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

import collections

import torch.nn as nn

from htmresearch.frameworks.pytorch.linear_sdr import LinearSDR
from htmresearch.frameworks.pytorch.cnn_sdr import CNNSDR2d



class Flatten(nn.Module):
  """
  Simple module used to flatten the tensors before passing data from CNN layer
  to the linear layer
  """
  def __init__(self, size):
    super(Flatten, self).__init__()
    self.size = size

  def forward(self, x):
    x = x.view(-1, self.size)
    return x

class SparseNet(nn.Module):

  def __init__(self,
               n=2000,
               k=200,
               outChannels=0,
               c_k=0,
               inputSize=28*28,
               outputSize=10,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               boostStrength=1.0,
               boostStrengthFactor=1.0,
               dropout=0.0):
    """
    A network with one or more hidden layers, which can be a sequence of
    k-sparse CNN followed by a sequence of k-sparse linear layer.

    :param n:
      Number of units in each fully connected k-sparse linear layer.
      Use 0 to disable the linear layer
    :type n: int or list[int]

    :param k:
      Number of ON (non-zero) units per iteration in each k-sparse linear layer.
      The sparsity of this layer will be k / n. If k >= n, the layer acts as a
      traditional fully connected RELU layer
    :type k: int or list[int]

    :param outChannels:
      Number of channels (filters) in each k-sparse convolutional layer.
      Use 0 to disable the CNN layer
    :type outChannels: int or list[int]

    :param c_k:
      Number of ON (non-zero) units per iteration in each k-sparse convolutional
      layer. The sparsity of this layer will be c_k / c_n. If c_k >= c_n, the
      layer acts as a traditional convolutional layer.
    :type c_k: int or list[int]

    :param inputSize:
      If the CNN layer is enable this parameter holds a tuple representing
      (in_channels,height,width). Otherwise it will hold the total
      dimensionality of input vector of the first linear layer. We apply
      view(-1, inputSize) to the data before passing it to Linear layers.
    :type inputSize: int or tuple[int,int,int]

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


  .. note::

    We considered three possibilities for sparse CNNs. The second one is
    currently implemented.

    1. Treat the output as a sparse linear layer as if the weights were not
       shared. Do global inhibition across the whole layer, and accumulate
       duty cycles across all units as if they were all distinct. This makes
       little sense.

    2. Treat the output as a sparse global layer but do consider weight sharing.
       Do global inhibition across the whole layer, but accumulate duty cycles
       across the outChannels filters (it is possible that a given filter has
       multiple active outputs per image). This is simpler to implement and may
       be a decent approach for smaller images such as MNIST. It requires fewer
       filters to get our SDR properties.

    3. Do local inhibition. Do inhibition within each set of filters such
       that each location has at least k active units. Accumulate duty cycles
       across the outChannels filters (it is possible that a given filter has
       multiple active outputs per image). The downside of this approach is that
       we will force activity even in blank areas of the image, which could even
       be negative. To counteract that we would want something like the spatial
       pooler's stimulusThreshold, so that only positive activity gets
       transmitted. Another downside is that we may need a large number of
       filters to get SDR properties. Overall this may be a good approach for
       larger color images and complex domains but may be too heavy handed for
       MNIST.

    """
    super(SparseNet, self).__init__()

    assert(weightSparsity >= 0)

    # Validate CNN sdr params
    if isinstance(inputSize, collections.Sequence):
      assert(inputSize[1] == inputSize[2],
             "sparseCNN only supports square images")

    if type(outChannels) is not list:
      outChannels = [outChannels]
    if type(c_k) is not list:
      c_k = [c_k]
    assert(len(outChannels) == len(c_k))

    # Validate linear sdr params
    if type(n) is not list:
      n = [n]
    if type(k) is not list:
      k = [k]
    assert(len(n) == len(k))
    for i in range(len(n)):
      assert(k[i] <= n[i])




    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.n = n
    self.outChannels = outChannels
    self.c_k = c_k
    self.inputSize = inputSize
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.boostStrengthFactor = boostStrengthFactor
    self.boostStrength = boostStrength
    self.kernelSize = 5
    self.learningIterations = 0

    inputFeatures = inputSize
    self.cnnSdr = nn.Sequential()
    # CNN Layers
    for i in range(len(outChannels)):
      if outChannels[i] != 0:
        module = CNNSDR2d(imageShape=inputFeatures,
                          outChannels=outChannels[i],
                          k=c_k[i],
                          kernelSize=self.kernelSize,
                          kInferenceFactor=kInferenceFactor,
                          boostStrength=boostStrength)
        self.cnnSdr.add_module("cnnSdr{}".format(i), module)
        # Feed this layer output into next layer input
        inputFeatures = (outChannels[i], module.maxpoolWidth, module.maxpoolWidth)

    # Linear layers
    self.linearSdr = nn.Sequential()
    if len(self.cnnSdr) > 0:
      inputFeatures = self.cnnSdr[-1].outputLength

    self.linearSdr.add_module("flatten", Flatten(inputFeatures))
    for i in range(len(n)):
      if n[i] != 0:
        self.linearSdr.add_module("linearSdr{}".format(i+1),
                        LinearSDR(inputFeatures=inputFeatures,
                                  n=n[i],
                                  k=k[i],
                                  kInferenceFactor=kInferenceFactor,
                                  weightSparsity=weightSparsity,
                                  boostStrength=boostStrength))
        # Add dropout after each hidden layer
        if dropout > 0.0:
          self.linearSdr.add_module("dropout", nn.Dropout(dropout))

        # Feed this layer output into next layer input
        inputFeatures = n[i]

    # Add one fully connected layer after all hidden layers
    self.fc = nn.Sequential(
      nn.Linear(self.n[-1], outputSize),
      nn.LogSoftmax(dim=1)
    )


  def postEpoch(self):
    """
    Call this once after each training epoch.
    """
    if self.training:
      self.boostStrength = self.boostStrength * self.boostStrengthFactor
      for module in self.cnnSdr.children():
        if hasattr(module, "setBoostStrength"):
          module.setBoostStrength(self.boostStrength)

      for module in self.linearSdr.children():
        if hasattr(module, "setBoostStrength"):
          module.setBoostStrength(self.boostStrength)
        if hasattr(module, "rezeroWeights"):
          # The optimizer is updating the weights during training after the forward
          # step. Therefore we need to re-zero the weights after every epoch
          module.rezeroWeights()

      print("boostStrength is now:", self.boostStrength)


  def forward(self, x):
    for module in self.children():
      x = module.forward(x)

    if self.training:
      batchSize = x.shape[0]
      self.learningIterations += batchSize

    return x


  def getLearningIterations(self):
    return self.learningIterations

  def maxEntropy(self):
    """
    Returns the maximum entropy we can expect
    """
    maxEntropy = 0
    for module in self.cnnSdr.children():
      if hasattr(module, "maxEntropy"):
        maxEntropy += module.maxEntropy()
    for module in self.linearSdr.children():
      if hasattr(module, "maxEntropy"):
        maxEntropy += module.maxEntropy()

    return maxEntropy


  def entropy(self):
    """
    Returns the current entropy
    """
    entropy = 0
    for module in self.cnnSdr.children():
      if hasattr(module, "entropy"):
        entropy += module.entropy()

    for module in self.linearSdr.children():
      if hasattr(module, "entropy"):
        entropy += module.entropy()

    return entropy

