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
import pprint


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from nupic.research.spatial_pooler import SpatialPooler as PYSpatialPooler

from htmresearch.frameworks.sp_paper.sp_metrics import (
  calculateEntropy, calculateInputOverlapMat, inspectSpatialPoolerStats,
  classificationAccuracyVsNoise, percentOverlap, calculateOverlapCurve,
  calculateStability
)
from htmresearch.support.spatial_pooler_monitor_mixin import (
  SpatialPoolerMonitorMixin)

class MonitoredSpatialPooler(SpatialPoolerMonitorMixin,
                             PYSpatialPooler): pass

from nupic.bindings.algorithms import SpatialPooler as CPPSpatialPooler
from nupic.bindings.math import GetNTAReal
from htmresearch.support.generate_sdr_dataset import SDRDataSet

realDType = GetNTAReal()
uintType = "uint32"
plt.ion()
mpl.rcParams['pdf.fonttype'] = 42



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



def getSpatialPoolerParams(inputSize, boosting=False):
  if boosting == 0:
    from sp_params import spParamNoBoosting as spatialPoolerParameters
  else:
    from sp_params import spParamWithBoosting as spatialPoolerParameters

  spatialPoolerParameters['inputDimensions'] = (inputSize, 1)
  spatialPoolerParameters['potentialRadius'] = inputSize

  print "Spatial Pooler Parameters: "
  pprint.pprint(spatialPoolerParameters)
  return spatialPoolerParameters



def getSDRDataSetParams(inputVectorType):
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
  elif inputVectorType == 'randomBarSets':
    params = {'dataType': 'randomBarSets',
              'numInputVectors': 100,
              'nX': 40,
              'nY': 40,
              'barHalfLength': 3,
              'numBarsPerInput': 10,
              'seed': 41}
  elif inputVectorType == 'nyc_taxi':
    params = {'dataType': 'nyc_taxi',
              'n': 109,
              'w': 21,
              'minval': 0,
              'maxval': 40000}
  else:
    raise ValueError('unknown data type')
  return params


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



def plotReceptiveFields(sp, nDim1=8, nDim2=8):
  """
  Plot 2D receptive fields for 16 randomly selected columns
  :param sp:
  :return:
  """
  columnNumber = np.product(sp.getColumnDimensions())
  fig, ax = plt.subplots(nrows=4, ncols=4)
  for rowI in range(4):
    for colI in range(4):
      col = np.random.randint(columnNumber)
      connectedSynapses = np.zeros((inputSize,), dtype=uintType)
      sp.getConnectedSynapses(col, connectedSynapses)
      receptiveField = connectedSynapses.reshape((nDim1, nDim2))
      ax[rowI, colI].imshow(receptiveField, cmap='gray')
      ax[rowI, colI].set_title("col: {}".format(col))



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

  parser.add_option("--spatialImp",
                    type=str,
                    default="cpp",
                    dest="spatialImp",
                    help="spatial pooler implementations: py, c++, or "
                         "monitored_sp")

  parser.add_option("--trackOverlapCurve",
                    type=int,
                    default=0,
                    dest="trackOverlapCurve",
                    help="whether to track overlap curve during learning")

  parser.add_option("--changeDataSetContinuously",
                    type=int,
                    default=0,
                    dest="changeDataSetContinuously",
                    help="whether to change data set at every epoch")

  (options, remainder) = parser.parse_args()
  print options
  return options, remainder



def plotBoostTrace(sp, inputVectors, columnIndex):
  """
  Plot boostfactor for a selected column

  Note that learning is ON for SP here

  :param sp: sp instance
  :param inputVectors: input data
  :param columnIndex: index for the column of interest
  """
  numInputVector, inputSize = inputVectors.shape
  columnNumber = np.prod(sp.getColumnDimensions())
  boostFactorsTrace = np.zeros((columnNumber, numInputVector))
  activeDutyCycleTrace = np.zeros((columnNumber, numInputVector))
  minActiveDutyCycleTrace = np.zeros((columnNumber, numInputVector))
  for i in range(numInputVector):
    outputColumns = np.zeros(sp.getColumnDimensions(), dtype=uintType)
    inputVector = copy.deepcopy(inputVectors[i][:])
    sp.compute(inputVector, True, outputColumns)

    boostFactors = np.zeros((columnNumber, ), dtype=realDType)
    sp.getBoostFactors(boostFactors)
    boostFactorsTrace[:, i] = boostFactors

    activeDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getActiveDutyCycles(activeDutyCycle)
    activeDutyCycleTrace[:, i] = activeDutyCycle

    minActiveDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getMinActiveDutyCycles(minActiveDutyCycle)
    minActiveDutyCycleTrace[:, i] = minActiveDutyCycle

  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(boostFactorsTrace[columnIndex, :])
  plt.ylabel('Boost Factor')
  plt.subplot(2, 1, 2)
  plt.plot(activeDutyCycleTrace[columnIndex, :])
  plt.plot(minActiveDutyCycleTrace[columnIndex, :])
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



def plotAccuracyVsNoise(noiseLevelList, predictionAccuracy):
  plt.figure()
  plt.plot(noiseLevelList, predictionAccuracy, '-o')
  plt.ylim([0, 1.05])
  plt.xlabel('Noise level')
  plt.ylabel('Prediction Accuracy')



def plotSPstatsOverTime(numConnectedSynapsesTrace,
                        numNewlyConnectedSynapsesTrace,
                        numEliminatedSynapsesTrace,
                        stabilityTrace,
                        entropyTrace,
                        fileName=None):
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
  if fileName is not None:
    plt.savefig(fileName)



if __name__ == "__main__":
  plt.close('all')

  (_options, _args) = _getArgs()
  inputVectorType = _options.dataSet
  numEpochs = _options.numEpochs
  classification = _options.classification
  spatialImp = _options.spatialImp
  trackOverlapCurveOverTraining = _options.trackOverlapCurve
  changeDataSetContinuously = _options.changeDataSetContinuously

  params = getSDRDataSetParams(inputVectorType)

  sdrData = SDRDataSet(params)

  inputVectors = sdrData.getInputVectors()
  numInputVector, inputSize = inputVectors.shape

  print "Training Data Type {}".format(inputVectorType)
  print "Training Data Size {} Dimensions {}".format(numInputVector, inputSize)

  spParams = getSpatialPoolerParams(inputSize, _options.boosting)
  sp = createSpatialPooler(spatialImp, spParams)
  columnNumber = np.prod(sp.getColumnDimensions())

  expName = "dataType_{}_boosting_{}".format(inputVectorType, _options.boosting)
  # inspect SP stats before learning
  inspectSpatialPoolerStats(sp, inputVectors, expName+"beforeTraining")


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
  inputOverlapWinnerTrace = []

  if trackOverlapCurveOverTraining:
    fig, ax = plt.subplots()
    cmap = cm.get_cmap('jet')

  if spatialImp == "monitored_sp":
    sp.mmClearHistory()

  for epoch in range(numEpochs):
    if changeDataSetContinuously:
      params['seed'] = epoch
      sdrData.generateInputVectors(params)
      inputVectors = sdrData.getInputVectors()
      numInputVector, inputSize = inputVectors.shape

    print "training SP epoch {}".format(epoch)
    # calcualte overlap curve here
    if epoch % 50 == 0 and trackOverlapCurveOverTraining:
      inputOverlapScore, outputOverlapScore = calculateOverlapCurve(
        sp, inputVectors)
      plt.plot(np.mean(inputOverlapScore, 0), np.mean(outputOverlapScore, 0),
               color=cmap(float(epoch) / numEpochs))

    activeColumnsPreviousEpoch = copy.copy(activeColumnsCurrentEpoch)
    connectedCountsPreviousEpoch = copy.copy(connectedCounts)

    # Learn is turned off at the first epoch to gather stats of untrained SP
    learn = False if epoch == 0 else True

    # randomize the presentation order of input vectors
    sdrOrders = np.random.permutation(np.arange(numInputVector))

    # train SP here,
    for i in range(numInputVector):
      outputColumns = np.zeros(sp.getColumnDimensions(), dtype=uintType)
      inputVector = copy.deepcopy(inputVectors[sdrOrders[i]][:])
      # addNoiseToVector(inputVector, 0.05, inputVectorType)
      sp.compute(inputVector, learn, outputColumns)

      activeColumnsCurrentEpoch[sdrOrders[i]][:] = np.reshape(outputColumns,
                                                              (1, columnNumber))

      overlaps = sp.getOverlaps()
      inputOverlapWinner = overlaps[np.where(outputColumns > 0)[0]]
      inputOverlapWinnerTrace.append(np.mean(inputOverlapWinner))


    # gather trace stats here
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
      stability = calculateStability(activeColumnsCurrentEpoch,
                                     activeColumnsPreviousEpoch)
      stabilityTrace.append(stability)

      numConnectedSynapsesTrace.append(np.sum(connectedCounts))

      numNewSynapses = connectedCounts - connectedCountsPreviousEpoch
      numNewSynapses[numNewSynapses < 0] = 0
      numNewlyConnectedSynapsesTrace.append(np.sum(numNewSynapses))

      numEliminatedSynapses = connectedCountsPreviousEpoch - connectedCounts
      numEliminatedSynapses[numEliminatedSynapses < 0] = 0
      numEliminatedSynapsesTrace.append(np.sum(numEliminatedSynapses))

  if spatialImp == "monitored_sp":
    # plot permanence for a single column when monitored sp is used
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
    plt.savefig('figures/overlap_over_training_{}_.pdf'.format(expName))

  # plot stats over training
  fileName = 'figures/network_stats_over_training_{}.pdf'.format(expName)
  plotSPstatsOverTime(numConnectedSynapsesTrace,
                      numNewlyConnectedSynapsesTrace,
                      numEliminatedSynapsesTrace,
                      stabilityTrace,
                      entropyTrace,
                      fileName)

  # inspect SP again
  inspectSpatialPoolerStats(sp, inputVectors, expName+"afterTraining")

  if classification:
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
    plt.ylabel('Classification Accuracy')
    plt.savefig('figures/noise_robustness_{}_.pdf'.format(expName))

  # analyze RF properties
  if inputVectorType == "randomSDR":
    analyzeReceptiveFieldSparseInputs(inputVectors, sp)
    plt.savefig('figures/{}_inputOverlap_after_learning.pdf'.format(expName))
  elif inputVectorType == 'correlate-input':
    additionalInfo = sdrData.getAdditionalInfo()
    inputVectors1 = additionalInfo["inputVectors1"]
    inputVectors2 = additionalInfo["inputVectors2"]
    corrPairs = additionalInfo["corrPairs"]
    analyzeReceptiveFieldCorrelatedInputs(
      inputVectors, sp, params, inputVectors1, inputVectors2)
    plt.savefig(
      'figures/{}_inputOverlap_after_learning.pdf'.format(expName))
  elif (inputVectorType == "randomBarPairs" or
            inputVectorType == "randomCross" or
            inputVectorType == "randomBarSets"):
    plotReceptiveFields2D(sp, params['nX'], params['nY'])


