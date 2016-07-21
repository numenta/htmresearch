# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

"""
Implementation of Bruno Olshausen's sparse coding algorithm.
It relies on the formulation developed by Olshausen and Field (1996), slightly
modified so that it uses a Locally Competitive Algorithm (LCA) to compute the
coefficients, rather than a vanilla gradient descent, as proposed by Rozell
et al. (2008).

The algorithm expresses image patches on a basis of filter functions, with the
constraint that only a few of the coefficients on this basis (also called
activations) are non-zero. Thus, it tries to find the basis decomposition such
that the image projection is as close as possible to the original image,
with as few non-zero activations as possible.

The algorithm for solving the resulting objective function solves a dynamical
system by using thresholding functions to induce local competition between
dimensions.
"""

import random
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


EPSILON = 0.000001


class SparseNet(object):
  """
  Base class for SparseNet implementation, which provides public methods for
  training and encoding data, as well as methods for plotting the network's
  basis and loss history.

  The method to get data batches must be implemented in sub-classes specific to
  each data type.
  """

  __metaclass__ = ABCMeta


  def __init__(self,
               filterDim=64,
               outputDim=64,
               batchSize=100,
               numLcaIterations=75,
               learningRate=2.0,
               decayCycle=100,
               learningRateDecay=1.0,
               lcaLearningRate=0.1,
               thresholdDecay=0.95,
               minThreshold=0.1,
               thresholdType='soft',
               verbosity=0,
               showEvery=500,
               seed=42):
    """
    Initializes the SparseNet.
    :param filterDim:               (int)   (Flattened) dimension of filters
    :param outputDim:               (int)   Output dimension
    :param batchSize:               (int)   Batch size for training
    :param numLcaIterations:        (int)   Number of iterations in LCA
    :param learningRate:            (float) Learning rate
    :param decayCycle:              (int)   Number of iterations between decays
    :param learningRateDecay        (float) Learning rate decay rate
    :param lcaLearningRate          (float) Learning rate in LCA
    :param minThreshold:            (float) Minimum activation threshold
                                            during decay
    :param verbosity:               (int)   Verbosity level
    :param seed:                    (int)   Seed for random number generators
    """
    self.filterDim = filterDim
    self.outputDim = outputDim
    self.batchSize = batchSize
    self._reset()

    # training parameters
    self.learningRate = learningRate
    self.decayCycle = decayCycle
    self.learningRateDecay = learningRateDecay

    # LCA parameters
    self.numLcaIterations = numLcaIterations
    self.lcaLearningRate = lcaLearningRate
    self.thresholdDecay = thresholdDecay
    self.minThreshold = minThreshold
    self.thresholdType = thresholdType

    # debugging
    self.verbosity = verbosity
    self.showEvery = showEvery
    self.seed = seed
    if seed is not None:
      np.random.seed(seed)
      random.seed(seed)


  def train(self, inputData, numIterations, reset=False):
    """
    Trains the SparseNet, with the provided data.

    The reset parameter can be set to False if the network should not be
    reset before training (for example for continuing a previous started
    training).
    :param inputData:     (array) Input data, of dimension (inputDim, numPoints)
    :param numIterations: (int)   Number of training iterations
    :param reset:         (bool)  If set to True, reset basis and history
    """
    if not isinstance(inputData, np.ndarray):
      inputData = np.array(inputData)

    if reset:
      self._reset()

    for _ in xrange(numIterations):
      self._iteration += 1

      batch = self._getDataBatch(inputData)

      # check input dimension, change if necessary
      if batch.shape[0] != self.filterDim:
        raise ValueError("Batches and filter dimesions don't match!")

      activations = self.encode(batch)
      self._learn(batch, activations)

      if self._iteration % self.decayCycle == 0:
        self.learningRate *= self.learningRateDecay

    if self.verbosity >= 1:
      self.plotLoss()
      self.plotBasis()


  def encode(self, data, flatten=False):
    """
    Encodes the provided input data, returning a sparse vector of activations.

    It solves a dynamic system to find optimal activations, as proposed by
    Rozell et al. (2008).
    :param data:          (array) Data to be encoded (single point or multiple)
    :param flatten        (bool)  Whether or not the data needs to be flattened,
                                  in the case of images for example. Does not
                                  need to be enabled during training.
    :return:              (array) Array of sparse activations (dimOutput,
                                  numPoints)
    """
    if not isinstance(data, np.ndarray):
      data = np.array(data)

    # flatten if necessary
    if flatten:
      try:
        data = np.reshape(data, (self.filterDim, data.shape[-1]))
      except ValueError:
        # only one data point
        data = np.reshape(data, (self.filterDim, 1))

    if data.shape[0] != self.filterDim:
      raise ValueError("Data does not have the correct dimension!")

    # if single data point, convert to 2-dimensional array for consistency
    if len(data.shape) == 1:
      data = data[:, np.newaxis]


    projection = self.basis.T.dot(data)
    representation = self.basis.T.dot(self.basis) - np.eye(self.outputDim)
    states = np.zeros((self.outputDim, data.shape[1]))

    threshold = 0.5 * np.max(np.abs(projection), axis=0)
    activations = self._thresholdNonLinearity(states, threshold)

    for _ in xrange(self.numLcaIterations):
      # update dynamic system
      states *= (1 - self.lcaLearningRate)
      states += self.lcaLearningRate * (projection - representation.dot(activations))
      activations = self._thresholdNonLinearity(states, threshold)

      # decay threshold
      threshold *= self.thresholdDecay
      threshold[threshold < self.minThreshold] = self.minThreshold

    return activations


  def plotLoss(self, filename=None):
    """
    Plots the loss history.

    :param filename    (string) Can be provided to save the figure
    """
    plt.figure()
    plt.plot(self.losses.keys(), self.losses.values())
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Learning curve for {}".format(self))

    if filename is not None:
      plt.savefig(filename)


  def plotBasis(self, filename=None):
    """
    Plots the basis functions, reshaped in 2-dimensional arrays.

    This representation makes the most sense for visual input.
    :param:  filename    (string) Can be provided to save the figure
    """
    if np.floor(np.sqrt(self.filterDim)) ** 2 != self.filterDim:
      print "Basis visualization is not available if filterDim is not a square."
      return

    dim = int(np.sqrt(self.filterDim))

    if np.floor(np.sqrt(self.outputDim)) ** 2 != self.outputDim:
      outDimJ = np.sqrt(np.floor(self.outputDim / 2))
      outDimI = np.floor(self.outputDim / outDimJ)
      if outDimI > outDimJ:
        outDimI, outDimJ = outDimJ, outDimI
    else:
      outDimI = np.floor(np.sqrt(self.outputDim))
      outDimJ = outDimI

    outDimI, outDimJ = int(outDimI), int(outDimJ)
    basis = - np.ones((1 + outDimI * (dim + 1), 1 + outDimJ * (dim + 1)))

    # populate array with basis values
    k = 0
    for i in xrange(outDimI):
      for j in xrange(outDimJ):
        colorLimit = np.max(np.abs(self.basis[:, k]))
        mat = np.reshape(self.basis[:, k], (dim, dim)) / colorLimit
        basis[1 + i * (dim + 1) : 1 + i * (dim + 1) + dim, \
              1 + j * (dim + 1) : 1 + j * (dim + 1) + dim] = mat
        k += 1

    plt.figure()
    plt.subplot(aspect="equal")
    plt.pcolormesh(basis)

    plt.axis([0, 1 + outDimJ * (dim + 1), 0, 1 + outDimI * (dim + 1)])
    # remove ticks
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.title("Basis functions for {0}".format(self))

    if filename is not None:
      plt.savefig(filename)


  def _reset(self):
    """
    Reinitializes basis functions, iteration number and loss history.
    """
    self.basis = np.random.randn(self.filterDim, self.outputDim)
    self.basis /= np.sqrt(np.sum(self.basis ** 2, axis=0))
    self._iteration = 0
    self.losses = {}


  def _learn(self, batch, activations):
    """
    Learns a single iteration on the provided batch and activations.
    :param batch:      (array)  Training batch, of dimension (filterDim,
                                batchSize)
    :param activations:(array)  Computed activations, of dimension (outputDim,
                                batchSize)
    """
    batchResiduals = batch - self.basis.dot(activations)
    loss = np.mean(np.sqrt(np.sum(batchResiduals ** 2, axis=0)))
    self.losses[self._iteration] = loss

    if self.verbosity >= 2:
      if self._iteration % self.showEvery == 0:
        print "At iteration {0}, loss is {1:.3f}".format(self._iteration, loss)

    # update basis
    gradBasis = batchResiduals.dot(activations.T) / self.batchSize
    self.basis += self.learningRate * gradBasis

    # normalize basis
    self.basis /= np.sqrt(np.sum(self.basis ** 2, axis=0))


  def _thresholdNonLinearity(self, input, threshold, thresholdType=None):
    """
    Non linearity function, to transform the activations during training and
    encoding.
    :param input:          (array)  Activations
    :param threshold:      (array)  Thresholds
    :param thresholdType:  (string) 'soft', 'absoluteHard' or 'relativeHard'
    """
    if thresholdType == None:
      thresholdType = self.thresholdType

    activation = np.copy(input)

    if thresholdType == 'soft':
      return np.maximum(np.abs(activation) - threshold, 0.) * np.sign(activation)

    if thresholdType == 'absoluteHard':
      activation[np.abs(activation) < threshold] = 0.
      return activation

    if thresholdType == 'relativeHard':
      activation[activation < threshold] = 0.
      return activation


  @abstractmethod
  def _getDataBatch(self, inputData):
    """
    Returns an array of dimensions (filterDim, batchSize), to be used as
    batch for training data.

    Must be implemented in sub-classes specific to different data types

    :param: inputData:     (array)   Array of dimension (inputDim, numPoints)
    :returns:              (array)   Batch of dimension (filterDim, batchSize)
    """


  @classmethod
  def read(cls, proto):
    """
    Reads deserialized data from proto object

    :param proto: (DynamicStructBuilder) Proto object
    :return       (SparseNet)            SparseNet instance
    """
    sparsenet = object.__new__(cls)

    sparsenet.filterDim = proto.filterDim
    sparsenet.outputDim = proto.outputDim
    sparsenet.batchSize = proto.batchSize

    lossHistoryProto = proto.losses
    sparsenet.losses = {}
    for i in xrange(len(lossHistoryProto)):
      sparsenet.losses[lossHistoryProto[i].iteration] = lossHistoryProto[i].loss
    sparsenet._iteration = proto.iteration

    sparsenet.basis = np.reshape(proto.basis, newshape=(sparsenet.filterDim,
                                                        sparsenet.outputDim))

    # training parameters
    sparsenet.learningRate = proto.learningRate
    sparsenet.decayCycle = proto.decayCycle
    sparsenet.learningRateDecay = proto.learningRateDecay

    # LCA parameters
    sparsenet.numLcaIterations = proto.numLcaIterations
    sparsenet.lcaLearningRate = proto.lcaLearningRate
    sparsenet.thresholdDecay = proto.thresholdDecay
    sparsenet.minThreshold = proto.minThreshold
    sparsenet.thresholdType = proto.thresholdType

    # debugging
    sparsenet.verbosity = proto.verbosity
    sparsenet.showEvery = proto.showEvery
    sparsenet.seed = int(proto.seed)
    if sparsenet.seed is not None:
      np.random.seed(sparsenet.seed)
      random.seed(sparsenet.seed)

    return sparsenet


  def write(self, proto):
    """
    Writes serialized data to proto object

    :param proto: (DynamicStructBuilder) Proto object
    """
    proto.filterDim = self.filterDim
    proto.outputDim = self.outputDim
    proto.batchSize = self.batchSize

    lossHistoryProto = proto.init("losses", len(self.losses))
    i = 0
    for iteration, loss in self.losses.iteritems():
      iterationLossHistoryProto = lossHistoryProto[i]
      iterationLossHistoryProto.iteration = iteration
      iterationLossHistoryProto.loss = float(loss)
      i += 1

    proto.iteration = self._iteration

    proto.basis = list(
      self.basis.flatten().astype(type('float', (float,), {}))
    )

    # training parameters
    proto.learningRate = self.learningRate
    proto.decayCycle = self.decayCycle
    proto.learningRateDecay = self.learningRateDecay

    # LCA parameters
    proto.numLcaIterations = self.numLcaIterations
    proto.lcaLearningRate = self.lcaLearningRate
    proto.thresholdDecay = self.thresholdDecay
    proto.minThreshold = self.minThreshold
    proto.thresholdType = self.thresholdType

    # debugging
    proto.verbosity = self.verbosity
    proto.showEvery = self.showEvery
    proto.seed = self.seed


  def __eq__(self, other):
    """
    :param other:     (SparseNet) Other SparseNet to compare to
    :return:          (bool)      True if both networks are equal
    """
    if self.filterDim != other.filterDim:
      return False
    if self.outputDim != other.outputDim:
      return False
    if self._iteration != other._iteration:
      return False

    for iteration, loss in self.losses.iteritems():
      if iteration not in other.losses:
        return False
      if abs(loss - other.losses[iteration]) > EPSILON:
        return False

    if np.mean(np.abs(self.basis - other.basis)) > EPSILON:
      return False

    if self.learningRate != other.learningRate:
      return False
    if self.decayCycle != other.decayCycle:
      return False
    if self.learningRateDecay != other.learningRateDecay:
      return False

    if self.numLcaIterations != other.numLcaIterations:
      return False
    if self.lcaLearningRate != other.lcaLearningRate:
      return False
    if self.thresholdDecay != other.thresholdDecay:
      return False
    if self.minThreshold != other.minThreshold:
      return False
    if self.thresholdType != other.thresholdType:
      return False

    if self.seed != other.seed:
      return False

    return True


  def __ne__(self, other):
    """
    :param other:     (SparseNet)  Other SparseNet to compare to
    :return:          (bool)       True if both networks are not equal
    """
    return not self == other


  def __repr__(self):
    """
    Custom representation method.
    """
    className = self.__class__.__name__
    return className + "({0}, {1})".format(self.filterDim, self.outputDim)
