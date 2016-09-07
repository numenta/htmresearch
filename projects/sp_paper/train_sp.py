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
from optparse import OptionParser
import random
import pprint


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from nupic.research.spatial_pooler import SpatialPooler as PYSpatialPooler

from htmresearch.support.spatial_pooler_monitor_mixin import (
  SpatialPoolerMonitorMixin)

class MonitoredSpatialPooler(SpatialPoolerMonitorMixin,
                             PYSpatialPooler): pass

from nupic.bindings.algorithms import SpatialPooler as CPPSpatialPooler
from nupic.bindings.math import GetNTAReal
from generate_sdr_dataset import SDRDataSet

realDType = GetNTAReal()
uintType = "uint32"
plt.ion()
mpl.rcParams['pdf.fonttype'] = 42



def getSpatialPoolerParams(inputSize, boosting=False):
  if boosting is False:
    from sp_params import spParamNoBoosting as spatialPoolerParameters
  else:
    from sp_params import spParamWithBoosting as spatialPoolerParameters

  spatialPoolerParameters['inputDimensions'] = (inputSize, 1)
  spatialPoolerParameters['potentialRadius'] = inputSize

  print "Spatial Pooler Parameters: "
  pprint.pprint(spatialPoolerParameters)
  return spatialPoolerParameters



def createSpatialPooler(spatialImp, spatialPoolerParameters):
  if spatialImp == 'py':
    sp = PYSpatialPooler(**spatialPoolerParameters)
  elif spatialImp == 'cpp':
    sp = CPPSpatialPooler(**spatialPoolerParameters)
  elif spatialImp == 'monitored_sp':
    sp = MonitoredSpatialPooler(**spatialPoolerParameters)
  else:
    raise RuntimeError("Invalide spatialImp")
  return sp



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
  activationProb = np.mean(outputColumns.astype(realDType), 0)

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
  axs[1, 1].set_xlabel('connection #')
  axs[1, 1].set_ylabel('activation freq')
  plt.tight_layout()

  if saveFigPrefix is not None:
    plt.savefig('figures/{}_network_stats.pdf'.format(saveFigPrefix))


def calculateEntropy(activeColumns):
  MIN_ACTIVATION_PROB = 0.000001
  activationProb = np.mean(activeColumns, 0)
  activationProb[activationProb < MIN_ACTIVATION_PROB] = MIN_ACTIVATION_PROB
  activationProb = activationProb / np.sum(activationProb)

  entropy = -np.dot(activationProb, np.log2(activationProb))
  return entropy



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
        inputVectors, sp, params, inputVectors1, inputVectors2):

  numInputVector, inputSize = inputVectors.shape
  numInputVector1 = params['numInputVectorPerSensor']
  numInputVector2 = params['numInputVectorPerSensor']
  w = params['numActiveInputBits']
  inputSize1 = int(params['inputSize']/2)
  inputSize2 = int(params['inputSize']/2)

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



def plotReceptiveFields2D(sp, Nx, Ny):
  inputSize = Nx * Ny
  numColumns = np.product(sp.getColumnDimensions())

  nrows = 4
  ncols = 4
  fig, ax = plt.subplots(nrows, ncols)
  for r in range(nrows):
    for c in range(ncols):
      colID = np.random.randint(numColumns)
      connectedSynapses = np.zeros((inputSize,), dtype=uintType)
      sp.getConnectedSynapses(colID, connectedSynapses)
      receptiveField = np.reshape(connectedSynapses, (Nx, Ny))
      ax[r, c].imshow(receptiveField, interpolation="nearest", cmap='gray')
      ax[r, c].set_title('col {}'.format(colID))



def _getArgs():
  parser = OptionParser(usage="Train HTM Spatial Pooler")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='randomSDR',
                    dest="dataSet",
                    help="DataSet Name, choose from sparse, correlated-input"
                         "bar, cross, image")

  parser.add_option("-b",
                    "--boosting",
                    type=int,
                    default=0,
                    dest="boosting",
                    help="Whether to use boosting")

  parser.add_option("-e",
                    "--numEpochs",
                    type=int,
                    default=100,
                    dest="numEpochs",
                    help="number of epochs")

  parser.add_option("-c",
                    "--runClassification",
                    type=int,
                    default=0,
                    dest="classification",
                    help="Whether to run classification experiment")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder



def plotBoostTrace(sp, inputVectors):
  numInputVector, inputSize = inputVectors.shape
  columnNumber = np.prod(sp.getColumnDimensions())
  boostFactorsTrace = np.zeros((columnNumber, numInputVector))
  activeDutyCycleTrace = np.zeros((columnNumber, numInputVector))
  minActiveDutyCycleTrace = np.zeros((columnNumber, numInputVector))
  for i in range(numInputVector):
    outputColumns = np.zeros(sp.getColumnDimensions(), dtype=uintType)
    inputVector = copy.deepcopy(inputVectors[i][:])
    sp.compute(inputVector, learn, outputColumns)

    boostFactors = np.zeros((columnNumber, ), dtype=realDType)
    sp.getBoostFactors(boostFactors)
    boostFactorsTrace[:, i] = boostFactors

    activeDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getActiveDutyCycles(activeDutyCycle)
    activeDutyCycleTrace[:, i] = activeDutyCycle

    minActiveDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getMinActiveDutyCycles(minActiveDutyCycle)
    minActiveDutyCycleTrace[:, i] = minActiveDutyCycle

  c = 103
  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(boostFactorsTrace[c, :])
  plt.ylabel('Boost Factor')
  plt.subplot(2, 1, 2)
  plt.plot(activeDutyCycleTrace[c, :])
  plt.plot(minActiveDutyCycleTrace[c, :])
  plt.xlabel(' Time ')
  plt.ylabel('Active Duty Cycle')



def plotPermInfo(permInfo):
  fig, ax = plt.subplots(5, 1, sharex=True)
  ax[0].plot(permInfo['numConnectedSyn'])
  ax[0].set_title('connected syn #')
  ax[1].plot(permInfo['numNonConnectedSyn'])
  ax[2].plot(permInfo['avgPermConnectedSyn'])
  ax[2].set_title('perm connected')
  ax[3].plot(permInfo['avgPermNonConnectedSyn'])
  ax[3].set_title('perm unconnected')
  # plt.figure()
  # plt.subplot(3, 1, 1)
  # plt.plot(perm - initialPermanence[columnIndex, :])
  # plt.subplot(3, 1, 2)
  # plt.plot(truePermanence - initialPermanence[columnIndex, :], 'r')
  # plt.subplot(3, 1, 3)
  # plt.plot(truePermanence - perm, 'r')



if __name__ == "__main__":
  plt.close('all')

  trackOverlapCurveOverTraining = False

  (_options, _args) = _getArgs()
  inputVectorType = _options.dataSet
  numEpochs = _options.numEpochs
  classification = _options.classification

  if inputVectorType == 'randomSDR':
    params = {'dataType': 'randomSDR',
              'numInputVectors': 100,
              'inputSize': 1024,
              'numActiveInputBits': 20,
              'seed': 41}
  elif inputVectorType == 'dense':
    params = {'dataType': 'denseVectors',
              'numInputVectors': 100,
              'inputSize': 1024,
              'seed': 41}
  elif inputVectorType == 'correlatedSDRPairs':
    params = {'dataType': 'correlatedSDRPairs',
              'numInputVectors': 100,
              'inputSize': 1024,
              'numInputVectorPerSensor': 50,
              'corrStrength': 0.5,
              'numActiveInputBits': 20,
              'seed': 41}
  elif inputVectorType == 'randomBarPairs':
    params = {'dataType': 'randomBarPairs',
              'numInputVectors': 100,
              'nX': 20,
              'nY': 20,
              'barHalfLength': 3,
              'seed': 41}
  elif inputVectorType == 'randomCross':
    params = {'dataType': 'randomCross',
              'numInputVectors': 100,
              'nX': 20,
              'nY': 20,
              'barHalfLength': 3,
              'seed': 41}
  elif inputVectorType == 'nyc_taxi':
    params = {'dataType': 'nyc_taxi',
              'n': 109,
              'w': 21,
              'minval': 0,
              'maxval': 40000}
  else:
    raise ValueError('unknown data type')

  sdrData = SDRDataSet(params)

  inputVectors = sdrData.getInputVectors()
  numInputVector, inputSize = inputVectors.shape

  print "Training Data Type {}".format(inputVectorType)
  print "Training Data Size {} Dimensions {}".format(numInputVector, inputSize)

  spParams = getSpatialPoolerParams(inputSize, _options.boosting)
  sp = createSpatialPooler('monitored_sp', spParams)
  columnNumber = np.prod(sp.getColumnDimensions())

  inspectSpatialPoolerStats(sp, inputVectors, inputVectorType+"beforeTraining")

  if inputVectorType == "sparse":
    analyzeReceptiveFieldSparseInputs(inputVectors, sp)
    plt.savefig('figures/{}_inputOverlap_before_learning.pdf'.format(inputVectorType))

  # classification Accuracy before training
  if classification:
    noiseLevelList = np.linspace(0, 1.0, 21)
    accuracyBeforeTraining = classificationAccuracyVsNoise(
      sp, inputVectors, noiseLevelList)

    accuracyWithoutSP = classificationAccuracyVsNoise(
      None, inputVectors, noiseLevelList)

  activeColumnsCurrentEpoch = np.zeros((numInputVector, columnNumber))
  activeColumnsPreviousEpoch = np.zeros((numInputVector, columnNumber))
  connectedCounts = np.zeros((columnNumber,), dtype=uintType)
  stabilityTrace = []
  numConnectedSynapsesTrace = []
  numNewlyConnectedSynapsesTrace = []
  numEliminatedSynapsesTrace = []
  entropyTrace = []
  meanBoostFactorTrace = []


  if trackOverlapCurveOverTraining:
    fig, ax = plt.subplots()
    cmap = cm.get_cmap('jet')

  sp.mmClearHistory()

  for epoch in range(numEpochs):
    print "training SP epoch {}".format(epoch)
    # calcualte overlap curve here
    if epoch % 50 == 0 and trackOverlapCurveOverTraining:
      inputOverlapScore, outputOverlapScore = calculateOverlapCurve(
        sp, inputVectors, inputVectorType)
      plt.plot(np.mean(inputOverlapScore, 0), np.mean(outputOverlapScore, 0),
               color=cmap(float(epoch) / numEpochs))

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
    connectedCounts = connectedCounts.astype(realDType)

    entropyTrace.append(calculateEntropy(activeColumnsCurrentEpoch))

    boostFactors = np.zeros((columnNumber, ), dtype=realDType)
    sp.getBoostFactors(boostFactors)
    meanBoostFactorTrace.append(np.mean(boostFactors))

    activeDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getActiveDutyCycles(activeDutyCycle)
    if epoch >= 1:
      activeColumnsStable = np.logical_and(activeColumnsCurrentEpoch,
                                           activeColumnsPreviousEpoch)
      stabilityTrace.append(np.mean(np.sum(activeColumnsStable, 1))/
                             spParams['numActiveColumnsPerInhArea'])

      numConnectedSynapsesTrace.append(np.sum(connectedCounts))

      numNewSynapses = connectedCounts - connectedCountsPreviousEpoch
      numNewSynapses[numNewSynapses < 0] = 0
      numNewlyConnectedSynapsesTrace.append(np.sum(numNewSynapses))

      numEliminatedSynapses = connectedCountsPreviousEpoch - connectedCounts
      numEliminatedSynapses[numEliminatedSynapses < 0] = 0
      numEliminatedSynapsesTrace.append(np.sum(numEliminatedSynapses))

  columnIndex = 240
  permInfo = sp.recoverPermanence(columnIndex)
  plotPermInfo(permInfo)

  if trackOverlapCurveOverTraining:
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
  fig, axs = plt.subplots(nrows=5, ncols=1, sharex=True)

  axs[0].plot(numConnectedSynapsesTrace)
  axs[0].set_ylabel('Syn #')

  axs[1].plot(numNewlyConnectedSynapsesTrace)
  axs[1].set_ylabel('New Syn #')

  axs[2].plot(numEliminatedSynapsesTrace)
  axs[2].set_ylabel('Remove Syns #')

  axs[3].plot(stabilityTrace)
  axs[3].set_ylabel('Stability')

  axs[4].plot(entropyTrace)
  axs[4].set_ylabel('entropy (bits)')
  axs[4].set_xlabel('epochs')
  plt.savefig('figures/network_stats_over_training_{}.pdf'.format(inputVectorType))

  # inspect SP again
  inspectSpatialPoolerStats(sp, inputVectors, inputVectorType+"afterTraining")
  if classification:
    # classify SDRs with noise
    noiseLevelList = np.linspace(0, 1.0, 21)
    accuracyAfterTraining = classificationAccuracyVsNoise(
      sp, inputVectors, noiseLevelList)
  #
  # plt.figure()
  # plt.plot(noiseLevelList, accuracyBeforeTraining, 'r-x')
  # plt.plot(noiseLevelList, accuracyAfterTraining, 'b-o')
  # plt.plot(noiseLevelList, accuracyWithoutSP, 'k--')
  # plt.ylim([0, 1.05])
  # plt.legend(['Before Training', 'After Training'], loc=3)
  # plt.xlabel('Noise level')
  # plt.ylabel('Classification Accuracy')
  # plt.savefig('figures/noise_robustness_{}_.pdf'.format(inputVectorType))

  # analyze RF properties
  if inputVectorType == "randomSDR":
    analyzeReceptiveFieldSparseInputs(inputVectors, sp)
    plt.savefig('figures/{}_inputOverlap_after_learning.pdf'.format(inputVectorType))
  elif inputVectorType == 'correlate-input':
    additionalInfo = sdrData.getAdditionalInfo()
    inputVectors1 = additionalInfo["inputVectors1"]
    inputVectors2 = additionalInfo["inputVectors2"]
    corrPairs = additionalInfo["corrPairs"]
    analyzeReceptiveFieldCorrelatedInputs(
      inputVectors, sp, params, inputVectors1, inputVectors2)
    plt.savefig(
      'figures/{}_inputOverlap_after_learning.pdf'.format(inputVectorType))
  elif inputVectorType == "randomBarPairs" or inputVectorType == "randomCross":
    plotReceptiveFields2D(sp, params['nX'], params['nY'])


