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
This file also proposes a sub-class of SparseNet specific for natural images,
It uses random sub-parts of images of arbitrary size at
training, but must be fed with images of correct dimensions when encoding.
The regular SparseNet can also be used with images, but must be fed with
flattened image, both at training and when encoding.

Example use:
  # create and train network
  net = ImageSparseNet(inputDim=64,
                       outputDim=64,
                       verbosity=2,
                       numIterations=1000,
                       batchSize=100)
  images = net.loadMatlabImages("../data/IMAGES.mat", "IMAGES")
  net.train(images)

  # visualize loss history and basis
  net.plotLoss(filename="loss_history.png")
  net.plotBasis(filename="basis_functions.png")

  # encode images, must have inputDim pixels (also works with single image)
  encodings = net.encode(imagesToEncode)

"""

import numpy as np
import scipy.io as sc

from htmresearch.algorithms.sparse_net import SparseNet


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
    :param path:      (string)   Path to .mat file
    :param name:      (string)   Object name in the .mat file, just before .mat

    Also stores image dimensions to later the original images. If there are
    multiple channels, self.numChannels will store the number of channels,
    otherwise it will be set to None.
    """
    try:
      images = sc.loadmat(path)[name]
    except KeyError:
      raise KeyError('Wrong filename for provided images.')

    self._initializeDimensions(images)

    return images


  def loadNumpyImages(self, path, key=None):
    """
    Loads images using numpy.

    :param path:      (string)   Path to data file
    :param key:       (string)   Object key in data file if it's a dict

    Also stores image dimensions to later the original images. If there are
    multiple channels, self.numChannels will store the number of channels,
    otherwise it will be set to None.
    """
    data = np.load(path)

    if isinstance(data, dict):
      if key is None:
        raise ValueError("Images are stored as a dict, a key must be provided!")
      try:
        data = data[key]
      except KeyError:
        raise KeyError("Wrong key for provided data.")

    if not isinstance(data, np.ndarray):
      raise TypeError("Data must be stored as a dict or numpy array.")

    self._initializeDimensions(data)

    return data


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

    This implementation uses random sub-images from a random image as
    training batch.

    Images are flattened to get a 2-dimensional batch.
    """
    if not hasattr(self, 'numImages'):
      self._initializeDimensions(inputData)

    batch = np.zeros((self.inputDim, self.batchSize))
    miniImageSize = int(np.sqrt(self.inputDim))
    # choose random image
    imageIdx = np.random.choice(range(self.numImages))

    for i in xrange(self.batchSize):
      # pick random starting row
      rows = self.imageHeight - miniImageSize - 2 * self.BUFFER
      rowNumber = self.BUFFER + np.random.choice(range(rows))

      if self.imageHeight is None:
        miniImage = inputData[rowNumber : rowNumber + miniImageSize,
                              imageIdx]
      else:
        # pick random starting column
        cols = self.imageWidth - miniImageSize - 2 * self.BUFFER
        colNumber = self.BUFFER + np.random.choice(range(cols))

        if self.numChannels is None:
          miniImage = inputData[rowNumber : rowNumber + miniImageSize,
                                colNumber : colNumber + miniImageSize,
                                imageIdx]
        else:
          # pick random channel
          channelNumber = np.random.choice(range(self.numChannels))
          miniImage = inputData[rowNumber : rowNumber + miniImageSize,
                                colNumber : colNumber + miniImageSize,
                                channelNumber,
                                imageIdx]

      miniImage = np.reshape(miniImage, (self.inputDim, ))
      batch[:, i] = miniImage

    return batch
