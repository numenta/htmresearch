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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from nupic.research.spatial_pooler import SpatialPooler

# from nupic.bindings.algorithms import SpatialPooler

uintType = "uint32"
plt.ion()
mpl.rcParams['pdf.fonttype'] = 42


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
  minX1X2 = min(nonZeroX1, nonZeroX2)
  percentOverlap = 0
  if minX1X2 > 0:
    percentOverlap = float(np.dot(x1.T, x2)) / float(minX1X2)
  return percentOverlap



def generateDenseVectors(numVectors, inputSize):
  inputVectors = np.zeros((numVectors, inputSize), dtype=uintType)
  for i in range(numVectors):
    for j in range(inputSize):
      inputVectors[i][j] = random.randrange(2)
  return inputVectors



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



def addNoiseToVector(inputVector, noiseLevel, vectorType):
  if vectorType == 'sparse':
    corruptSparseVector(inputVector, noiseLevel)
  elif vectorType == 'dense':
    corruptDenseVector(inputVector, noiseLevel)
  else:
    raise ValueError("vectorType must be 'sparse' or 'dense' ")



def calculateOverlapCurve(sp, inputVectors, inputVectorType):
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
      # addNoiseToVector(inputVectorCorrupted, noiseLevelList[j], inputVectorType)

      sp.compute(inputVectors[i][:], False, outputColumns[i][:])
      sp.compute(inputVectorCorrupted, False,
                 outputColumnsCorrupted[i][:])

      inputOverlapScore[i][j] = percentOverlap(inputVectors[i][:],
                                               inputVectorCorrupted)
      outputOverlapScore[i][j] = percentOverlap(outputColumns[i][:],
                                                outputColumnsCorrupted[i][:])

  return inputOverlapScore, outputOverlapScore



def classifySPoutput(targetOutputColumns, outputColumns):
  numTargets, numDims = targetOutputColumns.shape
  overlap = np.zeros((numTargets,))
  for i in range(numTargets):
    overlap[i] = percentOverlap(outputColumns, targetOutputColumns[i, :])
  classLabel = np.argmax(overlap)
  return classLabel



def classificationAccuracyVsNoise(sp, inputVectors, noiseLevelList):
  numInputVector, inputSize = inputVectors.shape

  if sp is None:
    targetOutputColumns = copy.deepcopy(inputVectors)
  else:
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
      # addNoiseToVector(corruptedInputVector, noiseLevelList[i], inputVectorType)

      if sp is None:
        outputColumns = copy.deepcopy(corruptedInputVector)
      else:
        outputColumns = np.zeros(sp.getColumnDimensions(), dtype=uintType)
        sp.compute(corruptedInputVector, False, outputColumns)

      predictedClassLabel = classifySPoutput(targetOutputColumns, outputColumns)
      outcomes[i][j] = predictedClassLabel == j

  predictionAccuracy = np.mean(outcomes, 1)
  return predictionAccuracy



def plotAccuracyVsNoise(noiseLevelList, predictionAccuracy):
  plt.figure()
  plt.plot(noiseLevelList, predictionAccuracy, '-o')
  plt.ylim([0, 1.05])
  plt.xlabel('Noise level')
  plt.ylabel('Prediction Accuracy')



def inspectSpatialPoolerStats(sp, inputVectors, saveFigPrefix=None):
  """
  Inspect the statistics of a spatial pooler given a set of input vectors
  :param sp: an spatial pooler instance
  :param inputVectors: a set of input vectors
  :return:
  """
  numInputVector, inputSize = inputVectors.shape
  numColumns = np.prod(sp.getColumnDimensions())

  outputColumns = np.zeros((numInputVector, numColumns), dtype=uintType)
  inputOverlap = np.zeros((numInputVector, numColumns), dtype=uintType)

  connectedCounts = np.zeros((numColumns, ), dtype=uintType)
  sp.getConnectedCounts(connectedCounts)

  for i in range(numInputVector):
    sp.compute(inputVectors[i][:], False, outputColumns[i][:])
    inputOverlap[i][:] = sp.getOverlaps()

  avgInputOverlap = np.mean(inputOverlap, 0)
  activationProb = np.mean(outputColumns.astype('float32'), 0)

  fig, axs = plt.subplots(2, 1)
  axs[0].imshow(inputVectors[:, :200], cmap='gray')
  axs[0].set_ylabel('sample #')
  axs[0].set_title('input vectors')
  axs[1].imshow(outputColumns[:, :200], cmap='gray')
  axs[1].set_ylabel('sample #')
  axs[1].set_title('output vectors')
  if saveFigPrefix is not None:
    plt.savefig('figures/{}_example_input_output.pdf'.format(saveFigPrefix))

  fig, axs = plt.subplots(2, 2)
  axs[0, 0].hist(connectedCounts)
  axs[0, 0].set_xlabel('# Connected Synapse')

  axs[0, 1].hist(avgInputOverlap)
  axs[0, 1].set_xlabel('# avgInputOverlap')

  axs[1, 0].hist(activationProb)
  axs[1, 0].set_xlabel('activation prob')

  axs[1, 1].plot(connectedCounts, activationProb, '.')
  if saveFigPrefix is not None:
    plt.savefig('figures/{}_network_stats.pdf'.format(saveFigPrefix))


def calculateEntropy(activeColumns):
  MIN_ACTIVATION_PROB = 0.000001
  activationProb = np.mean(activeColumns, 0)
  activationProb[activationProb < MIN_ACTIVATION_PROB] = MIN_ACTIVATION_PROB
  activationProb = activationProb / np.sum(activationProb)

  entropy = -np.dot(activationProb, np.log2(activationProb))
  return entropy



def generateCorrelatedInputs():
  numInputVector1 = 50
  numInputVector2 = 50
  w = 20
  inputSize1 = w * numInputVector1
  inputSize2 = w * numInputVector2

  # inputVectors1 = np.zeros((numInputVector1, inputSize1))
  # for i in range(numInputVector1):
  #   inputVectors1[i][i*w:(i+1)*w] = 1
  #
  # inputVectors2 = np.zeros((numInputVector2, inputSize2))
  # for i in range(numInputVector2):
  #   inputVectors2[i][i*w:(i+1)*w] = 1

  inputVectors1 = generateRandomSDR(numInputVector1, inputSize1, w, seed=1)
  inputVectors2 = generateRandomSDR(numInputVector2, inputSize2, w, seed=2)

  corrMatSparsity = 0.1
  corrMat = np.random.rand(numInputVector1, numInputVector2) < corrMatSparsity

  numInputVector = np.sum(corrMat)
  inputSize = inputSize1 + inputSize2
  inputVectors = np.zeros((numInputVector, inputSize))
  counter = 0
  for i in range(numInputVector1):
    for j in range(numInputVector2):
      if corrMat[i, j]:
        inputVectors[counter][:] = np.concatenate((inputVectors1[i],
                                                  inputVectors2[j]))
        counter += 1

  randomOrder = np.random.permutation(range(numInputVector))
  inputVectors = inputVectors[randomOrder, :]
  return inputVectors, inputVectors1, inputVectors2



def plotReceptiveFields(sp, nDim1=8, nDim2=8):
  """
  Plot 2D receptive fields for 16 randomly selected columns
  :param sp:
  :return:
  """
  fig, ax = plt.subplots(nrows=4, ncols=4)
  for rowI in range(4):
    for colI in range(4):
      col = np.random.randint(columnNumber)
      connectedSynapses = np.zeros((inputSize,), dtype=uintType)
      sp.getConnectedSynapses(col, connectedSynapses)
      receptiveField = connectedSynapses.reshape((nDim1, nDim2))
      ax[rowI, colI].imshow(receptiveField, cmap='gray')
      ax[rowI, colI].set_title("col: {}".format(col))


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



def analyzeReceptiveFieldSparseInputs(inputVectors, sp):
  numColumns = np.product(sp.getColumnDimensions())
  overlapMat = calculateInputOverlapMat(inputVectors, sp)

  plt.figure()
  plt.imshow(overlapMat[:100, :], interpolation="nearest", cmap="magma")
  plt.xlabel("Input Vector #")
  plt.ylabel("SP Column #")
  plt.colorbar()
  plt.title('percent overlap')

  sortedOverlapMat = np.zeros(overlapMat.shape)
  for c in range(numColumns):
    sortedOverlapMat[c, :] = np.sort(overlapMat[c, :])

  avgSortedOverlaps = np.flipud(np.mean(sortedOverlapMat, 0))
  plt.figure()
  plt.plot(avgSortedOverlaps, '-o')
  plt.xlabel('sorted input vector #')
  plt.ylabel('percent overlap')


def analyzeReceptiveFieldCorrelatedInputs(
        inputVectors, inputVectors1, inputVectors2, sp):

  numInputVector, inputSize = inputVectors.shape
  numInputVector1 = 50
  numInputVector2 = 50
  w = 20
  inputSize1 = w * numInputVector1
  inputSize2 = w * numInputVector2

  connectedCounts = np.zeros((columnNumber,), dtype=uintType)
  sp.getConnectedCounts(connectedCounts)

  numColumns = np.product(sp.getColumnDimensions())
  overlapMat1 = np.zeros((numColumns, inputVectors1.shape[0]))
  overlapMat2 = np.zeros((numColumns, inputVectors2.shape[0]))
  numColumns = np.product(sp.getColumnDimensions())
  numInputVector, inputSize = inputVectors.shape

  for c in range(numColumns):
    connectedSynapses = np.zeros((inputSize,), dtype=uintType)
    sp.getConnectedSynapses(c, connectedSynapses)
    for i in range(inputVectors1.shape[0]):
      overlapMat1[c, i] = percentOverlap(connectedSynapses[:inputSize1],
                                         inputVectors1[i, :inputSize1])
    for i in range(inputVectors2.shape[0]):
      overlapMat2[c, i] = percentOverlap(connectedSynapses[inputSize1:],
                                         inputVectors2[i, :inputSize2])

  sortedOverlapMat1 = np.zeros(overlapMat1.shape)
  sortedOverlapMat2 = np.zeros(overlapMat2.shape)
  for c in range(numColumns):
    sortedOverlapMat1[c, :] = np.sort(overlapMat1[c, :])
    sortedOverlapMat2[c, :] = np.sort(overlapMat2[c, :])
  fig, ax = plt.subplots(nrows=2, ncols=2)
  ax[0, 0].plot(np.mean(sortedOverlapMat1, 0), '-o')
  ax[0, 1].plot(np.mean(sortedOverlapMat2, 0), '-o')

  fig, ax = plt.subplots(nrows=1, ncols=2)
  ax[0].imshow(overlapMat1[:100, :], interpolation="nearest", cmap="magma")
  ax[0].set_xlabel('# Input 1')
  ax[0].set_ylabel('SP Column #')
  ax[1].imshow(overlapMat2[:100, :], interpolation="nearest", cmap="magma")
  ax[1].set_xlabel('# Input 2')
  ax[1].set_ylabel('SP Column #')



if __name__ == "__main__":
  plt.close('all')

  inputVectorType = 'correlate-input'  # 'sparse' or 'dense'
  trackOverlapCurveOverTraining = False

  if inputVectorType == 'sparse':
    numInputVector = 100
    inputSize = 1024
    numActiveBits = int(0.02 * inputSize)
    inputVectors = generateRandomSDR(numInputVector, inputSize, numActiveBits)
  elif inputVectorType == 'dense':
    numInputVector = 100
    inputSize = 1000
    inputVectors = generateDenseVectors(numInputVector, inputSize)
  elif inputVectorType == 'correlate-input':
    inputVectors, inputVectors1, inputVectors2 = generateCorrelatedInputs()
  elif inputVectorType == 'natural_images':
    numInputVector = 100
    inputVectors = getImageData(numInputVector)
  else:
    raise ValueError

  numInputVector, inputSize = inputVectors.shape

  print "Training Data Type {}".format(inputVectorType)
  print "Training Data Size {} Dimensions {}".format(numInputVector, inputSize)

  columnNumber = 2048
  spatialPoolerParameters = {
    "inputDimensions": (inputSize, 1),
    "columnDimensions": (columnNumber, 1),
    "potentialRadius": int(0.5 * inputSize),
    "globalInhibition": True,
    "numActiveColumnsPerInhArea": int(0.02 * columnNumber),
    "stimulusThreshold": 0,
    "synPermInactiveDec": 0.005,
    "synPermActiveInc": 0.001,
    "synPermConnected": 0.1,
    "minPctOverlapDutyCycle": 0.01,
    "minPctActiveDutyCycle": 0.01,
    "dutyCyclePeriod": 1000,
    "maxBoost": 1.0,
    "seed": 1936
  }
  sp = SpatialPooler(**spatialPoolerParameters)

  inspectSpatialPoolerStats(sp, inputVectors, inputVectorType+"beforeTraining")

  if inputVectorType == "sparse":
    analyzeReceptiveFieldSparseInputs(inputVectors, sp)
    plt.savefig('figures/{}_inputOverlap_before_learning.pdf'.format(inputVectorType))

  # classification Accuracy before training
  noiseLevelList = np.linspace(0, 1.0, 21)
  accuracyBeforeTraining = classificationAccuracyVsNoise(
    sp, inputVectors, noiseLevelList)

  accuracyWithoutSP = classificationAccuracyVsNoise(
    None, inputVectors, noiseLevelList)

  epochs = 800

  activeColumnsCurrentEpoch = np.zeros((numInputVector, columnNumber))
  activeColumnsPreviousEpoch = np.zeros((numInputVector, columnNumber))
  connectedCounts = np.zeros((columnNumber,), dtype=uintType)
  numBitDiffTrace = []
  numConnectedSynapsesTrace = []
  numNewlyConnectedSynapsesTrace = []
  numEliminatedSynapsesTrace = []
  entropyTrace = []

  fig, ax = plt.subplots()
  cmap = cm.get_cmap('jet')
  for epoch in range(epochs):
    print "training SP epoch {}".format(epoch)
    # calcualte overlap curve here
    if epoch % 50 == 0 and trackOverlapCurveOverTraining:
      inputOverlapScore, outputOverlapScore = calculateOverlapCurve(
        sp, inputVectors, inputVectorType)
      plt.plot(np.mean(inputOverlapScore, 0), np.mean(outputOverlapScore, 0),
               color=cmap(float(epoch) / epochs))

    activeColumnsPreviousEpoch = copy.copy(activeColumnsCurrentEpoch)
    connectedCountsPreviousEpoch = copy.copy(connectedCounts)

    # train SP here,
    # Learn is turned off at the first epoch to gather stats of untrained SP
    learn = False if epoch == 0 else True

    # randomize the presentation order of input vectors
    sdrOrders = np.random.permutation(np.arange(numInputVector))
    for i in range(numInputVector):
      outputColumns = np.zeros(sp.getColumnDimensions(), dtype=uintType)
      inputVector = copy.deepcopy(inputVectors[sdrOrders[i]][:])
      # addNoiseToVector(inputVector, 0.05, inputVectorType)
      sp.compute(inputVector, learn, outputColumns)

      activeColumnsCurrentEpoch[sdrOrders[i]][:] = np.reshape(outputColumns,
                                                              (1, columnNumber))

    connectedCounts = connectedCounts.astype(uintType)
    sp.getConnectedCounts(connectedCounts)
    connectedCounts = connectedCounts.astype('float32')

    entropyTrace.append(calculateEntropy(activeColumnsCurrentEpoch))

    if epoch >= 1:
      activeColumnsDiff = activeColumnsCurrentEpoch > activeColumnsPreviousEpoch
      numBitDiffTrace.append(np.mean(np.sum(activeColumnsDiff, 1)))

      numConnectedSynapsesTrace.append(np.sum(connectedCounts))

      numNewSynapses = connectedCounts - connectedCountsPreviousEpoch
      numNewSynapses[numNewSynapses < 0] = 0
      numNewlyConnectedSynapsesTrace.append(np.sum(numNewSynapses))

      numEliminatedSynapses = connectedCountsPreviousEpoch - connectedCounts
      numEliminatedSynapses[numEliminatedSynapses < 0] = 0
      numEliminatedSynapsesTrace.append(np.sum(numEliminatedSynapses))

  plt.xlabel('Input overlap')
  plt.ylabel('Output overlap')

  cax = fig.add_axes([0.05, 0.95, 0.4, 0.05])

  fig2, ax2 = plt.subplots()
  data = np.arange(0, 800).reshape((20, 40))
  im = ax2.imshow(data, cmap='jet')

  cbar = fig.colorbar(im, cax=cax, orientation='horizontal',
                      ticks=[0, 400, 800])
  plt.close(fig2)
  plt.savefig('figures/overlap_over_training_{}_.pdf'.format(inputVectorType))

  # plot stats over training
  fig, axs = plt.subplots(nrows=5, ncols=1)

  axs[0].plot(numConnectedSynapsesTrace)
  axs[0].set_ylabel('Syn #')

  axs[1].plot(numNewlyConnectedSynapsesTrace)
  axs[1].set_ylabel('New Syn #')

  axs[2].plot(numEliminatedSynapsesTrace)
  axs[2].set_ylabel('Remove Syns #')

  axs[3].plot(numBitDiffTrace)
  axs[3].set_ylabel('Active Column Diff')

  axs[4].plot(entropyTrace)
  axs[4].set_ylabel('entropy (bits)')
  plt.savefig('figures/network_stats_over_training_{}.pdf'.format(inputVectorType))

  # inspect SP again
  inspectSpatialPoolerStats(sp, inputVectors, inputVectorType+"afterTraining")
  # classify SDRs with noise
  noiseLevelList = np.linspace(0, 1.0, 21)
  accuracyAfterTraining = classificationAccuracyVsNoise(
    sp, inputVectors, noiseLevelList)

  plt.figure()
  plt.plot(noiseLevelList, accuracyBeforeTraining, 'r-x')
  plt.plot(noiseLevelList, accuracyAfterTraining, 'b-o')
  plt.plot(noiseLevelList, accuracyWithoutSP, 'k--')
  plt.ylim([0, 1.05])
  plt.legend(['Before Training', 'After Training'], loc=3)
  plt.xlabel('Noise level')
  plt.ylabel('Prediction Accuracy')
  plt.savefig('figures/noise_robustness_{}_.pdf'.format(inputVectorType))

  # analyze RF properties
  if inputVectorType == "sparse":
    analyzeReceptiveFieldSparseInputs(inputVectors, sp)
    plt.savefig('figures/{}_inputOverlap_after_learning.pdf'.format(inputVectorType))
  elif inputVectorType == 'correlate-input':
    analyzeReceptiveFieldCorrelatedInputs(
      inputVectors, inputVectors1, inputVectors2, sp)
    plt.savefig(
      'figures/{}_inputOverlap_after_learning.pdf'.format(inputVectorType))
