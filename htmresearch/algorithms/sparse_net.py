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
It uses the formulation developed by Olshausen and Field (1996), slightly
modified so that it uses a Locally Competitive Algorithm (LCA) to compute the
coefficients, rather than a vanilla gradient descent, as proposed by Rozell
et al. (2008).

This file also proposes a sub-class specific for natural images, which uses
random sub-parts of images as training batches, and proposes a method for
loading images from a .mat file. This is inspired by Bruno Olshausen's
Neural Computation course.

Example use:
  # create and train network
  net = ImageSparseNet(inputDim=64,
                       outputDim=64,
                       verbosity=2,
                       learningRate=0.1,
                       numIterations=400,
                       batchSize=100)
  images = net.loadMatlabImages("IMAGES.mat", "IMAGES")
  net.train(images)

  # visualize loss history and basis
  net.plotLoss(filename="loss_history.png")
  net.plotBasis(filename="basis_functions.png")

  # encode images
  encodings = net.encode(images)
"""

import random

import numpy as np
import scipy.io as sc
import matplotlib.pyplot as plt

class SparseNet(object):
  """Default SparseNet implementation, which works on most common data types.

  It provides public methods for training and encoding data, as well as methods
  for plotting the network and loss.
  """

  def __init__(self,
               inputDim=64,
               outputDim=64,
               batchSize=100,
               numIterations=1000,
               numLCAIterations=75,
               learningRate=0.2,
               decayCycle=50,
               learningRateDecay=0.95,
               thresholdDecay=0.95,
               minThreshold=0.1,
               thresholdType='soft',
               verbosity=0):
    """
    Initializes the SparseNet.
    :param inputDim:                (int) (Flattened) dimension of input data
    :param outputDim:               (int) Output dimension
    :param batchSize:               (int) Batch size for training
    :param numIterations:           (int) Number of training iterations
    :param numLCAIterations:        (int) Number of iterations in LCA
    :param learningRate:            (float) Learning rate
    :param minThreshold:            (float) Minimum activation threshold
                                    during decay
    :param verbosity:               (int) Verbosity level
    """
    self.inputDim = inputDim
    self.outputDim = outputDim
    self.batchSize = batchSize
    self._reset()

    self.numIterations = numIterations
    self.numLCAIterations = numLCAIterations
    self.learningRate = learningRate
    self.decayCycle = decayCycle
    self.learningRateDecay = learningRateDecay
    self.thresholdDecay = thresholdDecay
    self.minThreshold = minThreshold
    self.thresholdType = thresholdType

    self.verbosity = verbosity


  def train(self, inputData, reset=True):
    """
    Trains the SparseNet, with the provided data.

    The reset parameter can be set to False if the network should not be
    reset before training (for example for continuing a previous started
    training).
    """
    if not isinstance(inputData, np.ndarray):
      inputData = np.array(inputData)

    if reset:
      self._reset()

    for t in xrange(self.numIterations):
      batch = self._getDataBatch(inputData)
      if batch.shape[0] != self.inputDim:
        print "Changing input dimension"
        self.inputDim = batch.shape[0]

      activations = self.encode(batch)
      try:
        self._learn(batch, activations, t)
      except RuntimeWarning:
        raise RuntimeWarning("Overflowed, try with a lower learning rate!")

      if t % self.decayCycle == 0:
        self.learningRate *= self.learningRateDecay

    if self.verbosity >= 1:
      self.plotLoss()
      self.plotBasis()


  def encode(self, data):
    """
    Encodes the provided input data, returning a sparse vector of activations.

    It solves a dynamic system to find optimal activations, as proposed by
    Rozell et al. (2008).
    :param data:          (array) Input data, dimensions (inputDim, numPoints)
    :return:              (array) Array of sparse activations
    """
    if not isinstance(data, np.ndarray):
      data = np.array(data)
    if data.shape[0] != self.inputDim:
      raise ValueError("Input data does not have the correct dimension!")
    # if single data point, convert to 2-dimensional array for consistency
    if len(data.shape) == 1:
      data = data[:, np.newaxis]

    projection = self.basis.T.dot(data)
    representation = self.basis.T.dot(self.basis) - np.eye(self.outputDim)
    states = np.zeros((self.outputDim, data.shape[1]))

    threshold = 0.5 * np.max(np.abs(projection), axis=0)
    activations = self._activation(states, threshold)
    for t in xrange(self.numLCAIterations):
      # update dynamic system
      states *= (1 - self.learningRate)
      states += self.learningRate * (projection - representation.dot(activations))
      activations = self._activation(states, threshold)

      # decay threshold
      threshold *= self.thresholdDecay
      threshold[threshold < self.minThreshold] = self.minThreshold

    return activations


  def plotLoss(self, filename=None):
    """
    Plots the loss history.

    :param filename            (string) Can be provided to save the figure
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
    :param:  filename            (string) Can be provided to save the figure
    """
    if np.floor(np.sqrt(self.outputDim)) ** 2 != self.outputDim:
      print "Basis visualization is not available if outputDim is not a square."
      return
    if np.floor(np.sqrt(self.inputDim)) ** 2 != self.inputDim:
      print "Basis visualization is not available if inputDim is not a square."
      return

    dim = int(np.sqrt(self.inputDim))
    outDim = int(np.sqrt(self.outputDim))
    basis = - np.ones((outDim * (dim + 1) + 1, outDim * (dim + 1) + 1))

    # populate array with basis values
    k = 0
    for i in xrange(outDim):
      for j in xrange(outDim):
        colorLimit = np.max(np.abs(self.basis[:, k]))
        mat = np.reshape(self.basis[:, k], (dim, dim)) / colorLimit
        basis[1 + i * (dim + 1) : 1 + i * (dim + 1) + dim, \
              1 + j * (dim + 1) : 1 + j * (dim + 1) + dim] = mat
        k += 1

    plt.figure()
    plt.pcolor(basis)
    plt.axis([0, 1 + (outDim + 1) * dim, 0, 1 + (outDim + 1) * dim])
    # remove ticks
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.title("Basis functions for {0}".format(self))

    if filename is not None:
      plt.savefig(filename)


  def _reset(self):
    """
    Reinitializes basis functions and loss history.
    """
    self.basis = np.random.randn(self.inputDim, self.outputDim)
    self.basis /= np.sqrt(np.sum(self.basis ** 2, axis=0))
    self._iteration = 0
    self.losses = dict()


  def _learn(self, batch, activations, iteration):
    """
    Learns a single iteration on the provided batch and activations.
    :param batch        (array) Training batch, of dimension (inputDim, batchSize)
    :param coefficients (array) Computed activations, of dimension (outputDim, batchSize)
    """
    batchResiduals = batch - self.basis.dot(activations)
    loss = np.mean(np.sqrt(np.sum(batchResiduals ** 2, axis=0)))
    self.losses[iteration] = loss

    if self.verbosity >= 2:
      print "At iteration {0}, loss is {1:.3f}".format(iteration, loss)

    # update basis
    gradBasis = batchResiduals.dot(activations.T) / self.batchSize
    self.basis += self.learningRate * gradBasis

    # normalize basis
    self.basis /= np.sqrt(np.sum(self.basis ** 2, axis=0))


  def _activation(self, input, threshold, thresholdType=None):
    """
    Activation function, to transform the activations during training and
    encoding.
    :param input:          (array) Activations
    :param theta:          (float) Threshold
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


  def _getDataBatch(self, inputData):
    """
    Returns an array of dimensions (inputDim, batchSize), to be used as
    batch for training data.

    The basis implementation simply samples a batch from the training data,
    and sub-implementations should be used for particular data types (e.g.
    natural images).
    """
    numSamples = inputData.shape[1]
    return inputData[:, random.sample(range(numSamples), self.batchSize)]


  def __repr__(self):
    """Custom representation."""
    className = self.__class__.__name__
    return className + "({0}, {1})".format(self.inputDim, self.outputDim)



class ImageSparseNet(SparseNet):
  """
  Implementation of SparseNet specifically suited for natural images.

  It's particularity is a batch is composed with sub-components of one
  particular input image.
  """

  BUFFER = 4

  def loadMatlabImages(self, path, name):
    """
    Loads images from a .mat file.
    :param path:      (string) Path to .mat file
    :param name:      (string) Object name in the .mat file, just before .mat

    Also stores image dimensions to later the original images. If there are
    multiple channels, self.numChannels will store the number of channels,
    otherwise it will be set to None.
    """
    try:
      images = sc.loadmat(path)[name]
    except IndexError:
      raise IndexError('Wrong filename for provided images.')

    self._initializeDimensions(images)

    return images


  def _initializeDimensions(self, inputData):
    """
    Stores the training images' dimensions, for convenience.
    """
    if len(inputData.shape) == 2:
      self.imageHeight, self.numImages = inputData.shape
      self.imageWidth, self.numChannels = None, None

    elif len(inputData.shape) == 3:
      self.imageHeight, \
      self.imageWidth, \
      self.numImages = inputData.shape
      self.numChannels = None

    elif len(inputData.shape) == 4:
      self.imageHeight, \
      self.imageWidth, \
      self.numChannels, \
      self.numImages = inputData.shape

    else:
      raise ValueError("The provided image set has more than 4 dimensions.")


  def _getDataBatch(self, inputData):
    """
    Returns an array of dimensions (inputDim, batchSize), to be used as
    batch for training data.

    This implementation uses random sub-images from one random image as
    training batch.

    Images are flattened to get a 2-dimensional batch.
    """
    if not hasattr(self, 'numImages'):
      self._initializeDimensions(inputData)

    batch = np.zeros((self.inputDim, self.batchSize))
    imageIdx = np.random.choice(range(self.numImages))
    miniImageSize = int(np.sqrt(self.inputDim))

    for i in xrange(self.batchSize):
      rows = self.imageHeight - miniImageSize - 2 * self.BUFFER
      rowNumber = self.BUFFER + np.random.choice(range(rows))

      if self.imageHeight is None:
        miniImage = inputData[rowNumber : rowNumber + miniImageSize,
                              imageIdx]
      else:
        cols = self.imageWidth - miniImageSize - 2 * self.BUFFER
        colNumber = self.BUFFER + np.random.choice(range(cols))

        if self.numChannels is None:
          miniImage = inputData[rowNumber : rowNumber + miniImageSize,
                                colNumber : colNumber + miniImageSize,
                                imageIdx]
        else:
          channelNumber = np.random.choice(range(self.numChannels))
          miniImage = inputData[rowNumber : rowNumber + miniImageSize,
                                colNumber : colNumber + miniImageSize,
                                channelNumber,
                                imageIdx]

      miniImage = np.reshape(miniImage, (self.inputDim, ))
      batch[:, i] = miniImage

    return batch


if __name__ == '__main__':
  sn = ImageSparseNet(inputDim=64,
                      outputDim=64,
                      verbosity=5,
                      numIterations=400,
                      numLCAIterations=100,
                      batchSize=100)
  ds = sn.loadMatlabImages("../data/IMAGES.mat", "IMAGES")
  sn.train(ds)
  sn.plotBasis("fig.png")
  sn.plotLoss("fig2.png")
