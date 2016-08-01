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
      addNoiseToVector(inputVectorCorrupted, noiseLevelList[j], inputVectorType)

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
      addNoiseToVector(corruptedInputVector, noiseLevelList[i], inputVectorType)

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



def inspectSpatialPoolerStats(sp, inputVectors):
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

  connectedCounts = np.zeros((numColumns, ))
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
  axs[1].set_title('input vectors')

  fig, axs = plt.subplots(2, 2)
  axs[0, 0].hist(connectedCounts)
  axs[0, 0].set_xlabel('# Connected Synapse')

  axs[0, 1].hist(avgInputOverlap)
  axs[0, 1].set_xlabel('# avgInputOverlap')

  axs[1, 0].hist(activationProb)
  axs[1, 0].set_xlabel('activation prob')

  axs[1, 1].plot(connectedCounts, activationProb, '.')



if __name__ == "__main__":
  plt.close('all')

  numInputVector = 100
  inputVectorType = 'sparse'  # 'sparse' or 'dense'
  trackOverlapCurveOverTraining = True

  if inputVectorType == 'sparse':
    inputSize = 1024
    numActiveBits = int(0.02 * inputSize)
    inputVectors = generateRandomSDR(numInputVector, inputSize, numActiveBits)
  elif inputVectorType == 'dense':
    inputSize = 1000
    inputVectors = generateDenseVectors(numInputVector, inputSize)
  else:
    raise ValueError

  columnNumber = 2048
  sp = SpatialPooler((inputSize, 1),
                     (columnNumber, 1),
                     potentialRadius=int(0.5 * inputSize),
                     numActiveColumnsPerInhArea=int(0.02 * columnNumber),
                     globalInhibition=True,
                     seed=1936,
                     maxBoost=1,
                     dutyCyclePeriod=1000,
                     synPermActiveInc=0.001,
                     synPermInactiveDec=0.001)

  inspectSpatialPoolerStats(sp, inputVectors)

  # classification Accuracy before training
  noiseLevelList = np.linspace(0, 1.0, 21)
  accuracyBeforeTraining = classificationAccuracyVsNoise(
    sp, inputVectors, noiseLevelList)

  accuracyWithoutSP = classificationAccuracyVsNoise(
    None, inputVectors, noiseLevelList)

  epochs = 800

  activeColumnsCurrentEpoch = np.zeros((numInputVector, columnNumber))
  activeColumnsPreviousEpoch = np.zeros((numInputVector, columnNumber))
  connectedCounts = np.zeros((columnNumber,))
  numBitDiffTrace = []
  numConnectedSynapsesTrace = []
  numNewlyConnectedSynapsesTrace = []
  numEliminatedSynapsesTrace = []

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

    sp.getConnectedCounts(connectedCounts)

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
  fig, axs = plt.subplots(nrows=4, ncols=1)
  axs[0].plot(numBitDiffTrace)
  axs[0].set_ylabel('Active Column Diff')

  axs[1].plot(numConnectedSynapsesTrace)
  axs[1].set_ylabel('Syn #')

  axs[3].plot(numNewlyConnectedSynapsesTrace)
  axs[3].set_ylabel('Newly Syn #')

  axs[2].plot(numEliminatedSynapsesTrace)
  axs[2].set_ylabel('Eliminated Syns #')
  plt.savefig('figures/network_stats_over_training_{}.pdf'.format(inputVectorType))

  # inspect SP again
  inspectSpatialPoolerStats(sp, inputVectors)
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

