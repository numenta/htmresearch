#!/usr/bin/env python
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


import copy
import random

import matplotlib.pyplot as plt
import numpy as np

from nupic.bindings.math import GetNTAReal
# !/usr/bin/env python
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


import random
import numpy as np
import pandas as pd




uintType = "uint32"


def getConnectedSyns(sp):
  numInputs = sp.getNumInputs()
  numColumns = np.prod(sp.getColumnDimensions())
  connectedSyns = np.zeros((numColumns, numInputs), dtype=uintType)
  for columnIndex in range(numColumns):
    sp.getConnectedSynapses(columnIndex, connectedSyns[columnIndex, :])
  connectedSyns = connectedSyns.astype('float32')
  return connectedSyns



def getMovingBar(startLocation,
                 direction,
                 imageSize=(20, 20),
                 steps=5,
                 barHalfLength=3,
                 orientation='horizontal'):
  """
  Generate a list of bars
  :param startLocation:
         (list) start location of the bar center, e.g. (10, 10)
  :param direction:
         direction of movement, e.g., (1, 0)
  :param imageSize:
         (list) number of pixels on horizontal and vertical dimension
  :param steps:
         (int) number of steps
  :param barHalfLength:
         (int) length of the bar
  :param orientation:
         (string) "horizontal" or "vertical"
  :return:
  """
  startLocation = np.array(startLocation)
  direction = np.array(direction)
  barMovie = []
  for step in range(steps):
    barCenter = startLocation + step * direction
    barMovie.append(getBar(imageSize,
                           barCenter,
                           barHalfLength,
                           orientation))

  return barMovie



def getBar(imageSize, barCenter, barHalfLength, orientation='horizontal'):
  """
  Generate a single horizontal or vertical bar
  :param imageSize
         a list of (numPixelX. numPixelY). The number of pixels on horizontal
         and vertical dimension, e.g., (20, 20)
  :param barCenter:
         (list) center of the bar, e.g. (10, 10)
  :param barHalfLength
         (int) half length of the bar. Full length is 2*barHalfLength +1
  :param orientation:
         (string) "horizontal" or "vertical"
  :return:
  """
  (nX, nY) = imageSize
  (xLoc, yLoc) = barCenter
  bar = np.zeros((nX, nY), dtype=uintType)
  if orientation == 'horizontal':
    xmin = max(0, (xLoc - barHalfLength))
    xmax = min(nX - 1, (xLoc + barHalfLength + 1))
    bar[xmin:xmax, yLoc] = 1
  elif orientation == 'vertical':
    ymin = max(0, (yLoc - barHalfLength))
    ymax = min(nY - 1, (yLoc + barHalfLength + 1))
    bar[xLoc, ymin:ymax] = 1
  else:
    raise RuntimeError("orientation has to be horizontal or vertical")
  return bar



def getCross(nX, nY, barHalfLength):
  cross = np.zeros((nX, nY), dtype=uintType)
  xLoc = np.random.randint(barHalfLength, nX - barHalfLength)
  yLoc = np.random.randint(barHalfLength, nY - barHalfLength)
  cross[(xLoc - barHalfLength):(xLoc + barHalfLength + 1), yLoc] = 1
  cross[xLoc, (yLoc - barHalfLength):(yLoc + barHalfLength + 1)] = 1
  return cross



def generateRandomSDR(numSDR, numDims, numActiveInputBits, seed=42):
  """
  Generate a set of random SDR's
  @param numSDR:
  @param nDim:
  @param numActiveInputBits:
  """
  randomSDRs = np.zeros((numSDR, numDims), dtype=uintType)
  indices = np.array(range(numDims))
  np.random.seed(seed)
  for i in range(numSDR):
    randomIndices = np.random.permutation(indices)
    activeBits = randomIndices[:numActiveInputBits]
    randomSDRs[i, activeBits] = 1

  return randomSDRs



def getRandomBar(imageSize, barHalfLength, orientation='horizontal'):
  (nX, nY) = imageSize
  if orientation == 'horizontal':
    xLoc = np.random.randint(barHalfLength, nX - barHalfLength)
    yLoc = np.random.randint(0, nY)
    bar = getBar(imageSize, (xLoc, yLoc), barHalfLength, orientation)
  elif orientation == 'vertical':
    xLoc = np.random.randint(0, nX)
    yLoc = np.random.randint(barHalfLength, nY - barHalfLength)
    bar = getBar(imageSize, (xLoc, yLoc), barHalfLength, orientation)
  else:
    raise RuntimeError("orientation has to be horizontal or vertical")

  # shift bar with random phases
  bar = np.roll(bar, np.random.randint(10 * nX), 0)
  bar = np.roll(bar, np.random.randint(10 * nY), 1)
  return bar



def generateCorrelatedSDRPairs(numInputVectors,
                               inputSize,
                               numInputVectorPerSensor,
                               numActiveInputBits,
                               corrStrength=0.1,
                               seed=42):
  inputVectors1 = generateRandomSDR(
    numInputVectorPerSensor, int(inputSize / 2), numActiveInputBits, seed)
  inputVectors2 = generateRandomSDR(
    numInputVectorPerSensor, int(inputSize / 2), numActiveInputBits, seed + 1)

  # for each input on sensor 1, how many inputs on the 2nd sensor are
  # strongly correlated with it?
  numCorrPairs = 2
  numInputVector1 = numInputVectorPerSensor
  numInputVector2 = numInputVectorPerSensor
  corrPairs = np.zeros((numInputVector1, numInputVector2))
  for i in range(numInputVector1):
    idx = np.random.choice(np.arange(numInputVector2),
                           size=(numCorrPairs,), replace=False)
    corrPairs[i, idx] = 1.0 / numCorrPairs

  uniformDist = np.ones((numInputVector1, numInputVector2)) / numInputVector2
  sampleProb = corrPairs * corrStrength + uniformDist * (1 - corrStrength)
  inputVectors = np.zeros((numInputVectors, inputSize))
  for i in range(numInputVectors):
    vec1 = np.random.randint(numInputVector1)
    vec2 = np.random.choice(np.arange(numInputVector2), p=sampleProb[vec1, :])

    inputVectors[i][:] = np.concatenate((inputVectors1[vec1],
                                         inputVectors2[vec2]))

  return inputVectors, inputVectors1, inputVectors2, corrPairs



def generateDenseVectors(numVectors, inputSize, seed):
  np.random.seed(seed)
  inputVectors = np.zeros((numVectors, inputSize), dtype=uintType)
  for i in range(numVectors):
    for j in range(inputSize):
      inputVectors[i][j] = random.randrange(2)
  return inputVectors



def convertToBinaryImage(image, thresh=75):
  binaryImage = np.zeros(image.shape)
  binaryImage[image > np.percentile(image, thresh)] = 1
  return binaryImage



def getImageData(numInputVectors):
  from htmresearch.algorithms.image_sparse_net import ImageSparseNet

  DATA_PATH = "../sparse_net/data/IMAGES.mat"
  DATA_NAME = "IMAGES"

  DEFAULT_SPARSENET_PARAMS = {
    "filterDim": 64,
    "outputDim": 64,
    "batchSize": numInputVectors,
    "numLcaIterations": 75,
    "learningRate": 2.0,
    "decayCycle": 100,
    "learningRateDecay": 1.0,
    "lcaLearningRate": 0.1,
    "thresholdDecay": 0.95,
    "minThreshold": 1.0,
    "thresholdType": 'soft',
    "verbosity": 0,  # can be changed to print training loss
    "showEvery": 500,
    "seed": 42,
  }

  network = ImageSparseNet(**DEFAULT_SPARSENET_PARAMS)

  print "Loading training data..."
  images = network.loadMatlabImages(DATA_PATH, DATA_NAME)

  nDim1, nDim2, numImages = images.shape
  binaryImages = np.zeros(images.shape)
  for i in range(numImages):
    binaryImages[:, :, i] = convertToBinaryImage(images[:, :, i])

  inputVectors = network._getDataBatch(binaryImages)
  inputVectors = inputVectors.T
  return inputVectors



class SDRDataSet(object):
  """
  Generate, store, and manipulate SDR dataset
  """


  def __init__(self,
               params):

    self._params = params
    self._inputVectors = []
    self._dataType = params['dataType']
    self._additionalInfo = {}
    self.generateInputVectors(params)


  def generateInputVectors(self, params):

    if params['dataType'] == 'randomSDR':
      self._inputVectors = generateRandomSDR(
        params['numInputVectors'],
        params['inputSize'],
        params['numActiveInputBits'],
        params['seed'])

    elif params['dataType'] == 'denseVectors':
      self._inputVectors = generateDenseVectors(
        params['numInputVectors'],
        params['inputSize'],
        params['seed'])

    elif params['dataType'] == 'randomBarPairs':
      inputSize = params['nX'] * params['nY']
      numInputVectors = params['numInputVectors']
      self._inputVectors = np.zeros((numInputVectors, inputSize),
                                    dtype=uintType)
      for i in range(numInputVectors):
        bar1 = getRandomBar((params['nX'], params['nY']),
                            params['barHalfLength'], 'horizontal')
        bar2 = getRandomBar((params['nX'], params['nY']),
                            params['barHalfLength'], 'vertical')
        data = bar1 + bar2
        data[data > 0] = 1
        self._inputVectors[i, :] = np.reshape(data, newshape=(1, inputSize))

    elif params['dataType'] == 'randomBarSets':
      inputSize = params['nX'] * params['nY']
      numInputVectors = params['numInputVectors']
      self._inputVectors = np.zeros((numInputVectors, inputSize),
                                    dtype=uintType)
      for i in range(numInputVectors):
        data = 0
        for barI in range(params['numBarsPerInput']):
          orientation = np.random.choice(['horizontal', 'vertical'])
          bar = getRandomBar((params['nX'], params['nY']),
                             params['barHalfLength'], orientation)
          data += bar
        data[data > 0] = 1
        self._inputVectors[i, :] = np.reshape(data, newshape=(1, inputSize))

    elif params['dataType'] == 'randomCross':
      inputSize = params['nX'] * params['nY']
      numInputVectors = params['numInputVectors']
      self._inputVectors = np.zeros((numInputVectors, inputSize),
                                    dtype=uintType)
      for i in range(numInputVectors):
        data = getCross(params['nX'], params['nY'], params['barHalfLength'])
        self._inputVectors[i, :] = np.reshape(data, newshape=(1, inputSize))

    elif params['dataType'] == 'correlatedSDRPairs':
      (inputVectors, inputVectors1, inputVectors2, corrPairs) = \
        generateCorrelatedSDRPairs(
          params['numInputVectors'],
          params['inputSize'],
          params['numInputVectorPerSensor'],
          params['numActiveInputBits'],
          params['corrStrength'],
          params['seed'])
      self._inputVectors = inputVectors
      self._additionalInfo = {"inputVectors1": inputVectors1,
                              "inputVectors2": inputVectors2,
                              "corrPairs": corrPairs}
    elif params['dataType'] == 'nyc_taxi':
      from nupic.encoders.scalar import ScalarEncoder
      df = pd.read_csv('./data/nyc_taxi.csv', header=0, skiprows=[1, 2])
      inputVectors = np.zeros((5000, params['n']))
      for i in range(5000):
        inputRecord = {
          "passenger_count": float(df["passenger_count"][i]),
          "timeofday": float(df["timeofday"][i]),
          "dayofweek": float(df["dayofweek"][i]),
        }

        enc = ScalarEncoder(w=params['w'],
                            minval=params['minval'],
                            maxval=params['maxval'],
                            n=params['n'])
        inputSDR = enc.encode(inputRecord["passenger_count"])
        inputVectors[i, :] = inputSDR
      self._inputVectors = inputVectors


  def getInputVectors(self):
    return self._inputVectors


  def getAdditionalInfo(self):
    return self._additionalInfo



from nupic.math.topology import coordinatesFromIndex

realDType = GetNTAReal()
uintType = "uint32"


def percentOverlap(x1, x2):
  """
  Computes the percentage of overlap between vectors x1 and x2.

  @param x1   (array) binary vector
  @param x2   (array) binary vector
  @param size (int)   length of binary vectors

  @return percentOverlap (float) percentage overlap between x1 and x2
  """
  nonZeroX1 = np.count_nonzero(x1)
  nonZeroX2 = np.count_nonzero(x2)

  percentOverlap = 0
  minX1X2 = min(nonZeroX1, nonZeroX2)
  if minX1X2 > 0:
    overlap = float(np.dot(x1.T, x2))
    percentOverlap = overlap / minX1X2

  return percentOverlap



def addNoiseToVector(inputVector, noiseLevel, vectorType):
  """
  Add noise to SDRs
  @param inputVector (array) binary vector to be corrupted
  @param noiseLevel  (float) amount of noise to be applied on the vector.
  @param vectorType  (string) "sparse" or "dense"
  """
  if vectorType == 'sparse':
    corruptSparseVector(inputVector, noiseLevel)
  elif vectorType == 'dense':
    corruptDenseVector(inputVector, noiseLevel)
  else:
    raise ValueError("vectorType must be 'sparse' or 'dense' ")


def corruptDenseVector(vector, noiseLevel):
  """
  Corrupts a binary vector by inverting noiseLevel percent of its bits.

  @param vector     (array) binary vector to be corrupted
  @param noiseLevel (float) amount of noise to be applied on the vector.
  """
  size = len(vector)
  for i in range(size):
    rnd = random.random()
    if rnd < noiseLevel:
      if vector[i] == 1:
        vector[i] = 0
      else:
        vector[i] = 1



def corruptSparseVector(sdr, noiseLevel):
  """
  Add noise to sdr by turning off numNoiseBits active bits and turning on
  numNoiseBits in active bits
  @param sdr        (array) Numpy array of the  SDR
  @param noiseLevel (float) amount of noise to be applied on the vector.
  """

  numNoiseBits = int(noiseLevel * np.sum(sdr))
  if numNoiseBits <= 0:
    return sdr
  activeBits = np.where(sdr > 0)[0]
  inActiveBits = np.where(sdr == 0)[0]

  turnOffBits = np.random.permutation(activeBits)
  turnOnBits = np.random.permutation(inActiveBits)
  turnOffBits = turnOffBits[:numNoiseBits]
  turnOnBits = turnOnBits[:numNoiseBits]

  sdr[turnOffBits] = 0
  sdr[turnOnBits] = 1



def calculateOverlapCurve(sp, inputVectors):
  """
  Evalulate noise robustness of SP for a given set of SDRs
  @param sp a spatial pooler instance
  @param inputVectors list of arrays.
  :return:
  """
  columnNumber = np.prod(sp.getColumnDimensions())
  numInputVector, inputSize = inputVectors.shape

  outputColumns = np.zeros((numInputVector, columnNumber), dtype=uintType)
  outputColumnsCorrupted = np.zeros((numInputVector, columnNumber),
                                    dtype=uintType)

  noiseLevelList = np.linspace(0, 1.0, 21)
  inputOverlapScore = np.zeros((numInputVector, len(noiseLevelList)))
  outputOverlapScore = np.zeros((numInputVector, len(noiseLevelList)))
  for i in range(numInputVector):
    for j in range(len(noiseLevelList)):
      inputVectorCorrupted = copy.deepcopy(inputVectors[i][:])
      corruptSparseVector(inputVectorCorrupted, noiseLevelList[j])

      sp.compute(inputVectors[i][:], False, outputColumns[i][:])
      sp.compute(inputVectorCorrupted, False,
                 outputColumnsCorrupted[i][:])

      inputOverlapScore[i][j] = percentOverlap(inputVectors[i][:],
                                               inputVectorCorrupted)
      outputOverlapScore[i][j] = percentOverlap(outputColumns[i][:],
                                                outputColumnsCorrupted[i][:])

  return noiseLevelList, inputOverlapScore, outputOverlapScore



def classifySPoutput(targetOutputColumns, outputColumns):
  """
  Classify the SP output
  @param targetOutputColumns (list) The target outputs, corresponding to
                                    different classes
  @param outputColumns (array) The current output
  @return classLabel (int) classification outcome
  """
  numTargets, numDims = targetOutputColumns.shape
  overlap = np.zeros((numTargets,))
  for i in range(numTargets):
    overlap[i] = percentOverlap(outputColumns, targetOutputColumns[i, :])
  classLabel = np.argmax(overlap)
  return classLabel



def classificationAccuracyVsNoise(sp, inputVectors, noiseLevelList):
  """
  Evaluate whether the SP output is classifiable, with varying amount of noise
  @param sp a spatial pooler instance
  @param inputVectors (list) list of input SDRs
  @param noiseLevelList (list) list of noise levels
  :return:
  """
  numInputVector, inputSize = inputVectors.shape

  if sp is None:
    targetOutputColumns = copy.deepcopy(inputVectors)
  else:
    columnNumber = np.prod(sp.getColumnDimensions())
    # calculate target output given the uncorrupted input vectors
    targetOutputColumns = np.zeros((numInputVector, columnNumber),
                                   dtype=uintType)
    for i in range(numInputVector):
      sp.compute(inputVectors[i][:], False, targetOutputColumns[i][:])

  outcomes = np.zeros((len(noiseLevelList), numInputVector))
  for i in range(len(noiseLevelList)):
    for j in range(numInputVector):
      corruptedInputVector = copy.deepcopy(inputVectors[j][:])
      corruptSparseVector(corruptedInputVector, noiseLevelList[i])

      if sp is None:
        outputColumns = copy.deepcopy(corruptedInputVector)
      else:
        outputColumns = np.zeros((columnNumber, ), dtype=uintType)
        sp.compute(corruptedInputVector, False, outputColumns)

      predictedClassLabel = classifySPoutput(targetOutputColumns, outputColumns)
      outcomes[i][j] = predictedClassLabel == j

  predictionAccuracy = np.mean(outcomes, 1)
  return predictionAccuracy


def plotExampleInputOutput(sp, inputVectors, saveFigPrefix=None):
  """
  Plot example input & output
  @param sp: an spatial pooler instance
  @param inputVectors: a set of input vectors
  """
  numInputVector, inputSize = inputVectors.shape
  numColumns = np.prod(sp.getColumnDimensions())

  outputColumns = np.zeros((numInputVector, numColumns), dtype=uintType)
  inputOverlap = np.zeros((numInputVector, numColumns), dtype=uintType)

  connectedCounts = np.zeros((numColumns,), dtype=uintType)
  sp.getConnectedCounts(connectedCounts)

  winnerInputOverlap = np.zeros(numInputVector)
  for i in range(numInputVector):
    sp.compute(inputVectors[i][:], False, outputColumns[i][:])
    inputOverlap[i][:] = sp.getOverlaps()
    activeColumns = np.where(outputColumns[i][:] > 0)[0]
    if len(activeColumns) > 0:
      winnerInputOverlap[i] = np.mean(
        inputOverlap[i][np.where(outputColumns[i][:] > 0)[0]])

  fig, axs = plt.subplots(2, 1)
  axs[0].imshow(inputVectors[:, :200], cmap='gray', interpolation="nearest")
  axs[0].set_ylabel('input #')
  axs[0].set_title('input vectors')
  axs[1].imshow(outputColumns[:, :200], cmap='gray', interpolation="nearest")
  axs[1].set_ylabel('input #')
  axs[1].set_title('output vectors')
  if saveFigPrefix is not None:
    plt.savefig('figures/{}_example_input_output.pdf'.format(saveFigPrefix))

  inputDensity = np.sum(inputVectors, 1) / float(inputSize)
  outputDensity = np.sum(outputColumns, 1) / float(numColumns)
  fig, axs = plt.subplots(2, 1)
  axs[0].plot(inputDensity)
  axs[0].set_xlabel('input #')
  axs[0].set_ylim([0, 0.2])

  axs[1].plot(outputDensity)
  axs[1].set_xlabel('input #')
  axs[1].set_ylim([0, 0.05])

  if saveFigPrefix is not None:
    plt.savefig('figures/{}_example_input_output_density.pdf'.format(saveFigPrefix))



def inspectSpatialPoolerStats(sp, inputVectors, saveFigPrefix=None):
  """
  Inspect the statistics of a spatial pooler given a set of input vectors
  @param sp: an spatial pooler instance
  @param inputVectors: a set of input vectors
  """
  numInputVector, inputSize = inputVectors.shape
  numColumns = np.prod(sp.getColumnDimensions())

  outputColumns = np.zeros((numInputVector, numColumns), dtype=uintType)
  inputOverlap = np.zeros((numInputVector, numColumns), dtype=uintType)

  connectedCounts = np.zeros((numColumns, ), dtype=uintType)
  sp.getConnectedCounts(connectedCounts)

  winnerInputOverlap = np.zeros(numInputVector)
  for i in range(numInputVector):
    sp.compute(inputVectors[i][:], False, outputColumns[i][:])
    inputOverlap[i][:] = sp.getOverlaps()
    activeColumns = np.where(outputColumns[i][:] > 0)[0]
    if len(activeColumns) > 0:
      winnerInputOverlap[i] = np.mean(
        inputOverlap[i][np.where(outputColumns[i][:] > 0)[0]])
  avgInputOverlap = np.mean(inputOverlap, 0)

  activationProb = np.mean(outputColumns.astype(realDType), 0)

  dutyCycleDist, binEdge = np.histogram(activationProb,
                                        bins=10, range=[-0.005, 0.095])
  dutyCycleDist = dutyCycleDist.astype('float32') / np.sum(dutyCycleDist)
  binCenter = (binEdge[1:] + binEdge[:-1])/2

  fig, axs = plt.subplots(2, 2)
  axs[0, 0].hist(connectedCounts)
  axs[0, 0].set_xlabel('# Connected Synapse')

  axs[0, 1].hist(winnerInputOverlap)
  axs[0, 1].set_xlabel('# winner input overlap')

  axs[1, 0].bar(binEdge[:-1]+0.001, dutyCycleDist, width=.008)
  axs[1, 0].set_xlim([-0.005, .1])
  axs[1, 0].set_xlabel('Activation Frequency')

  axs[1, 1].plot(connectedCounts, activationProb, '.')
  axs[1, 1].set_xlabel('connection #')
  axs[1, 1].set_ylabel('activation freq')
  plt.tight_layout()

  if saveFigPrefix is not None:
    plt.savefig('figures/{}_network_stats.pdf'.format(saveFigPrefix))
  return fig


def getRFCenters(sp, params, type='connected'):
  numColumns = np.product(sp.getColumnDimensions())
  dimensions = (params['nX'], params['nY'])

  meanCoordinates = np.zeros((numColumns, 2))
  avgDistToCenter = np.zeros((numColumns, 2))
  for columnIndex in range(numColumns):
    receptiveField = np.zeros((sp.getNumInputs(), ))
    if type == 'connected':
      sp.getConnectedSynapses(columnIndex, receptiveField)
    elif type == 'potential':
      sp.getPotential(columnIndex, receptiveField)
    else:
      raise RuntimeError('unknown RF type')

    connectedSynapseIndex = np.where(receptiveField)[0]
    if len(connectedSynapseIndex) == 0:
      continue
    coordinates = []
    for synapseIndex in connectedSynapseIndex:
      coordinate = coordinatesFromIndex(synapseIndex, dimensions)
      coordinates.append(coordinate)
    coordinates = np.array(coordinates)

    coordinates = coordinates.astype('float32')
    angularCoordinates = np.array(coordinates)
    angularCoordinates[:, 0] = coordinates[:, 0] / params['nX'] * 2 * np.pi
    angularCoordinates[:, 1] = coordinates[:, 1] / params['nY'] * 2 * np.pi


    for i in range(2):
      meanCoordinate = np.arctan2(
        np.sum(np.sin(angularCoordinates[:, i])),
        np.sum(np.cos(angularCoordinates[:, i])))
      if meanCoordinate < 0:
        meanCoordinate += 2 * np.pi

      dist2Mean = angularCoordinates[:, i] - meanCoordinate
      dist2Mean = np.arctan2(np.sin(dist2Mean), np.cos(dist2Mean))
      dist2Mean = np.max(np.abs(dist2Mean))

      meanCoordinate *= dimensions[i] / (2 * np.pi)
      dist2Mean *= dimensions[i] / (2 * np.pi)

      avgDistToCenter[columnIndex, i] = dist2Mean
      meanCoordinates[columnIndex, i] = meanCoordinate

  return meanCoordinates, avgDistToCenter


def entropy(x):
  """
  Calculate entropy of a binary random variable.
  (https://en.wikipedia.org/wiki/Entropy_(information_theory))
  @param x (float) the probability of the variable to be 1.
  @return (float) entropy
  """
  if  x*(1 - x) == 0:
    return 0;
  else:
    return - x*np.log2(x) - (1-x)*np.log2(1-x)

entropyVectorized = np.vectorize(entropy)

def calculateEntropy(activeColumns):
  """
  calculate the mean entropy given activation history
  @param activeColumns (array) 2D numpy array of activation history
  @return entropy (float) mean entropy
  """
  activationProb   = np.mean(activeColumns, 0)
  entropy          = np.sum(entropyVectorized(activationProb))
  numberOfColumns  = activeColumns.shape[1]
  # return mean entropy
  return entropy/numberOfColumns



def calculateInputOverlapMat(inputVectors, sp):
  numColumns = np.product(sp.getColumnDimensions())
  numInputVector, inputSize = inputVectors.shape
  overlapMat = np.zeros((numColumns, numInputVector))
  for c in range(numColumns):
    connectedSynapses = np.zeros((inputSize, ), dtype=uintType)
    sp.getConnectedSynapses(c, connectedSynapses)
    for i in range(numInputVector):
      overlapMat[c, i] = percentOverlap(connectedSynapses, inputVectors[i, :])
  return overlapMat



def calculateStability(activeColumnsCurrentEpoch, activeColumnsPreviousEpoch):
  activeColumnsStable = np.logical_and(activeColumnsCurrentEpoch,
                                       activeColumnsPreviousEpoch)
  stability = np.mean(np.sum(activeColumnsStable, 1))/\
              np.mean(np.sum(activeColumnsCurrentEpoch, 1))
  return stability


def calculateInputSpaceCoverage(sp):
  numInputs = np.prod(sp.getInputDimensions())
  numColumns = np.prod(sp.getColumnDimensions())
  inputSpaceCoverage = np.zeros(numInputs)

  connectedSynapses = np.zeros((numInputs), dtype=uintType)
  for columnIndex in range(numColumns):
    sp.getConnectedSynapses(columnIndex, connectedSynapses)
    inputSpaceCoverage += connectedSynapses
  inputSpaceCoverage = np.reshape(inputSpaceCoverage, sp.getInputDimensions())
  return inputSpaceCoverage


def reconstructionError(sp, inputVectors, activeColumnVectors, threshold=0.):
  """
  Computes a reconstruction error. The reconstuction $r(x)$ of an input vector $x$
  is given by the sum of the active column's connected synapses vector of 
  the SDR representation $sdr(x)$ of $x$ normalized by $1/numActiveColumns$. 
  The error is the normalized sum over the "hamming distance" (i.e. distance 
  induced by L1 norm) of $x$ and its reconstruction $r(x)$, i.e. (mathy stuff in LaTex)
  \[
      Reconstruction Error = (1/batchSize) * \sum_{x \in InputBatch} \| x - r(x) \|_1 .
  \]
  Note that $r(x)$ can be expressed as
  \[
      r(x) = (1/numActiveColumns) * C * sdr(x) ,
  \]
  where we view $sdr(x)$ as a binary column vector and $C$ is the 
  binary matrix whose jth column encodes the synaptic connectivity of 
  the pooler's columns and the input bits, i.e. 
  \[
        c_{i,j} = 1   :<=>  column j has a stable synaptic 
                            connection to input bit i.
  \]
  Note: Turns out that in our setting (x and syn(i) binary vectors) we have 
  \[
        Reconstruction Error = Witness Error.
  \]
  It can be shown that the error is optimized by the Hebbian-like update rule 
  of the spatial pooler. 
  
  @param sp (SpatialPooler) the spatial pooler instance
  @param inputVectors (array) 2D numpy array of input vectors
  @param activeColumnVectors (array) 2D numpy array of activation history
  @param threshold (float) if set > 0 it serves as threshold for a step function 
                           applied to the reconstruction vectors (values smaller than 
                           threshold are set to zero, and values bigger to one)
  @return error (float) the reconstruction error
  """
  batchSize        = inputVectors.shape[0]
  connectionMatrix = getConnectedSyns(sp)

  reconstructionVectors = np.dot(activeColumnVectors, connectionMatrix)
  numActiveColumns      = np.sum(activeColumnVectors, 1)[0]
  reconstructionVectors = reconstructionVectors/numActiveColumns

  if threshold > 0.:
    reconstructionVectors =np.where( 
                            reconstructionVectors > threshold, 
                            np.ones( reconstructionVectors.shape), 
                            np.zeros(reconstructionVectors.shape))

  Err = np.sum(np.absolute(reconstructionVectors  - inputVectors))

  return Err/batchSize


def witnessError(sp, inputVectors, activeColumnsCurrentEpoch):
  """
  Computes a variation of a reconstruction error. It measures the average 
  hamming distance of an active column's connected synapses vector and its witnesses. 
  An input vector is called witness for a column, iff the column is among 
  the active columns for the input computed by the spatial pooler. 
  The error is given by
  \[     
      Witness Error = (1/batchSize) * \sum_{x \in InputBatch} 
                          (1/numActiveColumns) * \sum_{i active column of sdr(x)} \| x -  syn(i) \|_1.
  \]
  Note: Turns out that in our setting (x and syn(i) binary vectors) we have 
  \[
      Witness Error = Reconstruction Error.
  \]
  It can be shown that the error is optimized by the Hebbian-like update rule 
  of the spatial pooler. 
  """
  connectionMatrix = getConnectedSyns(sp)
  batchSize        = inputVectors.shape[0]

  # 1st sum... over each input in batch
  Err = 0.
  for i in range(batchSize):
    activeColumns    = np.where(activeColumnsCurrentEpoch[i] > 0.)[0]
    numActiveColumns = activeColumns.shape[0]
    # 2nd sum... over each active colum
    err = 0.
    for j in activeColumns:
      # Compute hamming distance and accumulate
      err += np.sum(np.absolute(connectionMatrix[j] - inputVectors[i]))

    Err += err/numActiveColumns

  return Err/batchSize



def mutualInformation(sp, activeColumnsCurrentEpoch, column_1, column_2):
  """
  Computes the mutual information of the binary variables that represent 
  the activation probabilities of two columns. The mutual information I(X,Y) 
  of two random variables is given by
  \[
       I (X,Y)  = \sum_{x,y} p(x,y) log( p(x,y) / ( p(x) p(y) ) ).
  \]
  (https://en.wikipedia.org/wiki/Mutual_information)
  """
  i, j        = column_1, column_2
  batchSize   = activeColumnsCurrentEpoch.shape[0]

  # Activity Counts
  ci, cj, cij = 0., 0., dict([((0,0),0.), ((1,0),0.), ((0,1),0.), ((1,1),0.)])
  for t in range(batchSize):
    ai = activeColumnsCurrentEpoch[t, i]
    aj = activeColumnsCurrentEpoch[t, j]
    cij[(ai, aj)] += 1.
    ci += ai
    cj += aj

  # Mutual information calculation 
  Iij = 0
  for a,b in [(0,0), (1,0), (0,1), (1,1)]:
    # Compute probabilities
    pij = cij[(a,b)]/batchSize
    pi  = ci/batchSize if a == 1 else 1. - ci/batchSize
    pj  = cj/batchSize if b == 1 else 1. - cj/batchSize
    # Add current term of mutual information 
    Iij += pij * np.log2(pij/(pi*pj)) if pij > 0 else 0

  return Iij



def meanMutualInformation(sp, activeColumnsCurrentEpoch, columnsUnderInvestigation = []):
  """
  Computes the mean of the mutual information 
  of pairs taken from a list of columns. 
  """
  if len(columnsUnderInvestigation) == 0:
    columns = range(np.prod(sp.getColumnDimensions()))
  else:
    columns = columnsUnderInvestigation
  numCols = len(columns)
  sumMutualInfo = 0
  normalizingConst = numCols*(numCols - 1)/2
  for i in range(numCols):
    for j in range(i+1, numCols):
      sumMutualInfo += mutualInformation(sp, activeColumnsCurrentEpoch, columns[i], columns[j])

  return sumMutualInfo/normalizingConst



