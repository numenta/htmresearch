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

  return inputOverlapScore, outputOverlapScore



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
    winnerInputOverlap[i] = np.mean(inputOverlap[i][np.where(outputColumns[i][:] > 0)[0]])
  avgInputOverlap = np.mean(inputOverlap, 0)

  activationProb = np.mean(outputColumns.astype(realDType), 0)

  # fig, axs = plt.subplots(2, 1)
  # axs[0].imshow(inputVectors[:, :200], cmap='gray')
  # axs[0].set_ylabel('sample #')
  # axs[0].set_title('input vectors')
  # axs[1].imshow(outputColumns[:, :200], cmap='gray')
  # axs[1].set_ylabel('sample #')
  # axs[1].set_title('output vectors')
  # if saveFigPrefix is not None:
  #   plt.savefig('figures/{}_example_input_output.pdf'.format(saveFigPrefix))

  fig, axs = plt.subplots(2, 2)
  axs[0, 0].hist(connectedCounts)
  axs[0, 0].set_xlabel('# Connected Synapse')

  axs[0, 1].hist(winnerInputOverlap)
  axs[0, 1].set_xlabel('# winner input overlap')

  axs[1, 0].hist(activationProb)
  axs[1, 0].set_xlabel('activation prob')

  axs[1, 1].plot(connectedCounts, activationProb, '.')
  axs[1, 1].set_xlabel('connection #')
  axs[1, 1].set_ylabel('activation freq')
  plt.tight_layout()

  if saveFigPrefix is not None:
    plt.savefig('figures/{}_network_stats.pdf'.format(saveFigPrefix))
  return fig


def calculateEntropy(activeColumns):
  """
  calculate entropy given activation history
  @param activeColumns (array) 2D numpy array of activation history
  @return entropy (flaot) entropy
  """
  MIN_ACTIVATION_PROB = 0.000001
  activationProb = np.mean(activeColumns, 0)
  activationProb[activationProb < MIN_ACTIVATION_PROB] = MIN_ACTIVATION_PROB
  activationProb = activationProb / np.sum(activationProb)

  entropy = -np.dot(activationProb, np.log2(activationProb))
  return entropy



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
              np.mean(np.sum(activeColumnsCurrentEpoch))
  return stability