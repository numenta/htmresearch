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

from optparse import OptionParser
import pprint
import os
import pickle
from tabulate import tabulate

from nupic.research.spatial_pooler import SpatialPooler as PYSpatialPooler
from htmresearch.algorithms.faulty_spatial_pooler import FaultySpatialPooler
from htmresearch.frameworks.sp_paper.sp_metrics import (
  calculateEntropy, calculateInputOverlapMat, inspectSpatialPoolerStats,
  classificationAccuracyVsNoise, percentOverlap, calculateOverlapCurve,
  calculateStability, plotExampleInputOutput,
  reconstructionError, witnessError
)
from htmresearch.support.spatial_pooler_monitor_mixin import (
  SpatialPoolerMonitorMixin)

from htmresearch.support.sp_paper_utils import *

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
  elif spatialImp == "faulty_sp":
    sp = FaultySpatialPooler(**spatialPoolerParameters)
  else:
    raise RuntimeError("Invalide spatialImp")
  return sp



def getSpatialPoolerParams(inputSize, boosting=0):
  if boosting == 0:
    from model_params.sp_params import spParamNoBoosting as spatialPoolerParameters
  else:
    from model_params.sp_params import spParamWithBoosting as spatialPoolerParameters

  spatialPoolerParameters['inputDimensions'] = (inputSize, 1)
  spatialPoolerParameters['potentialRadius'] = inputSize

  print "Spatial Pooler Parameters: "
  pprint.pprint(spatialPoolerParameters)
  return spatialPoolerParameters



def getSDRDataSetParams(inputVectorType, seed):
  if inputVectorType == 'randomSDR':
    params = {'dataType': 'randomSDR',
              'numInputVectors': 100,
              'inputSize': 1024,
              'numActiveInputBits': 20,
              'seed': seed}
  elif inputVectorType == 'randomSDRVaryingSparsity':
    params = {'dataType': 'randomSDRVaryingSparsity',
              'numInputVectors': 100,
              'inputSize': 512,
              'minSparsity': 0.02,
              'maxSparsity': 0.2,
              'seed': seed}
  elif inputVectorType == 'dense':
    params = {'dataType': 'denseVectors',
              'numInputVectors': 100,
              'inputSize': 1024,
              'seed': seed}
  elif inputVectorType == 'correlatedSDRPairs':
    params = {'dataType': 'correlatedSDRPairs',
              'numInputVectors': 100,
              'inputSize': 1024,
              'numInputVectorPerSensor': 50,
              'corrStrength': 0.5,
              'numActiveInputBits': 20,
              'seed': seed}
  elif inputVectorType == 'randomBarPairs':
    params = {'dataType': 'randomBarPairs',
              'numInputVectors': 50,
              'nX': 10,
              'nY': 10,
              'barHalfLength': 2,
              'seed': seed}
  elif inputVectorType == 'randomCross':
    params = {'dataType': 'randomCross',
              'numInputVectors': 50,
              'numCrossPerInput': 1,
              'nX': 10,
              'nY': 10,
              'barHalfLength': 2,
              'seed': seed}
  elif inputVectorType == 'randomBarSets':
    params = {'dataType': 'randomBarSets',
              'numInputVectors': 50,
              'nX': 40,
              'nY': 40,
              'barHalfLength': 3,
              'numBarsPerInput': 10,
              'seed': seed}
  elif inputVectorType == 'nyc_taxi':
    params = {'dataType': 'nyc_taxi',
              'n': 109,
              'w': 21,
              'minval': 0,
              'maxval': 40000}
  else:
    raise ValueError('unknown data type')
  print
  print "dataset parameters"
  pprint.pprint(params)
  return params



def _getArgs():
  parser = OptionParser(usage="Train HTM Spatial Pooler")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='randomSDR',
                    dest="dataSet",
                    help="DataSet Name, choose from sparse, correlatedSDRPairs"
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
                    help="spatial pooler implementations: py, c++, "
                         "monitored_sp, faulty_sp")

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

  parser.add_option("--changeDataSetAt",
                    type=int,
                    default=0,
                    dest="changeDataSetAt",
                    help="regenerate dataset at this iteration")

  parser.add_option("--killCellsAt",
                    type=int,
                    default=0,
                    dest="killCellsAt",
                    help="kill a fraction of sp cells at this iteration")

  parser.add_option("--killCellPrct",
                    type=float,
                    default=0.0,
                    dest="killCellPrct",
                    help="the fraction of sp cells that will be removed")

  parser.add_option("--name",
                    type=str,
                    default='defaultName',
                    dest="expName",
                    help="the fraction of sp cells that will be removed")

  parser.add_option("--seed",
                    type=str,
                    default=41,
                    dest="seed",
                    help="random seed for SP and dataset")

  (options, remainder) = parser.parse_args()
  print options
  return options, remainder



if __name__ == "__main__":
  plt.close('all')

  (_options, _args) = _getArgs()
  inputVectorType = _options.dataSet
  numEpochs = _options.numEpochs
  classification = _options.classification
  spatialImp = _options.spatialImp
  trackOverlapCurveOverTraining = _options.trackOverlapCurve
  changeDataSetContinuously = _options.changeDataSetContinuously
  changeDataSetAt = _options.changeDataSetAt
  killCellsAt = _options.killCellsAt
  killCellPrct = _options.killCellPrct
  expName = _options.expName
  seed = _options.seed
  if expName == 'defaultName':
    expName = "dataType_{}_boosting_{}".format(
      inputVectorType, _options.boosting)
  expName = expName + '_seed_{}'.format(seed)
  createDirectories(expName)

  params = getSDRDataSetParams(inputVectorType, seed)

  sdrData = SDRDataSet(params)

  inputVectors = sdrData.getInputVectors()
  numInputVector, inputSize = inputVectors.shape

  print "Training Data Type {}".format(inputVectorType)
  print "Training Data Size {} Dimensions {}".format(numInputVector, inputSize)

  spParams = getSpatialPoolerParams(inputSize, _options.boosting)
  sp = createSpatialPooler(spatialImp, spParams)
  columnNumber = np.prod(sp.getColumnDimensions())
  aliveColumns = np.arange(columnNumber)
  # inspect SP stats before learning
  inspectSpatialPoolerStats(sp, inputVectors, expName+"beforeTraining")

  testInputs = inputVectors[:40, :]
  numTestInputs = testInputs.shape[0]

  # classification Accuracy before training
  if classification:
    noiseLevelList = np.linspace(0, 1.0, 21)
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
  classificationRobustnessTrace = []
  noiseRobustnessTrace = []
  reconstructionErrorTrace = []
  witnessErrorTrace = []

  connectedSyns = getConnectedSyns(sp)

  if spatialImp == "monitored_sp":
    sp.mmClearHistory()

  checkPoints = [0, changeDataSetAt-1, changeDataSetAt, numEpochs-1]
  for epoch in range(numEpochs):
    if changeDataSetContinuously or (epoch == changeDataSetAt):
      params['seed'] = epoch
      sdrData.generateInputVectors(params)
      inputVectors = sdrData.getInputVectors()
      numInputVector, inputSize = inputVectors.shape

    if epoch == killCellsAt and spatialImp == "faulty_sp":
        sp.killCells(killCellPrct)
        aliveColumns = sp.getAliveColumns()

    print "training SP epoch {}".format(epoch)

    # calculate overlap curve here
    if trackOverlapCurveOverTraining:
      noiseLevelList, inputOverlapScore, outputOverlapScore = \
        calculateOverlapCurve(sp, testInputs)
      noiseRobustnessTrace.append(np.trapz(np.flipud(np.mean(outputOverlapScore, 0)),
                                           noiseLevelList))
      np.savez('./results/input_output_overlap/{}_{}'.format(expName, epoch),
               noiseLevelList, inputOverlapScore, outputOverlapScore)

    if classification:
      # classify SDRs with noise
      noiseLevelList = np.linspace(0, 1.0, 21)
      classification_accuracy = classificationAccuracyVsNoise(
        sp, inputVectors, noiseLevelList)
      classificationRobustnessTrace.append(
        np.trapz(classification_accuracy, noiseLevelList))
      np.savez('./results/classification/{}/epoch_{}'.format(expName, epoch),
              noiseLevelList, classification_accuracy)

    activeColumnsPreviousEpoch = copy.copy(activeColumnsCurrentEpoch)
    connectedSynsPreviousEpoch = copy.copy(connectedSyns)

    # Learn is turned off at the first epoch to gather stats of untrained SP
    learn = False if epoch == 0 else True

    # randomize the presentation order of input vectors
    sdrOrders = np.random.permutation(np.arange(numInputVector))

    # train SP here,
    activeColumnsCurrentEpoch = runSPOnBatch(
      sp, inputVectors, learn, sdrOrders)

    # gather trace stats here
    sp.getConnectedCounts(connectedCounts)
    connectedSyns = getConnectedSyns(sp)

    entropyTrace.append(calculateEntropy(activeColumnsCurrentEpoch[:, aliveColumns]))

    boostFactors = np.zeros((columnNumber, ), dtype=realDType)
    sp.getBoostFactors(boostFactors)
    meanBoostFactorTrace.append(np.mean(boostFactors))

    activeDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getActiveDutyCycles(activeDutyCycle)

    reconstructionErrorTrace.append(reconstructionError(sp, inputVectors, activeColumnsCurrentEpoch))
    witnessErrorTrace.append(witnessError(sp, inputVectors, activeColumnsCurrentEpoch))

    if epoch >= 1:
      stability = calculateStability(activeColumnsCurrentEpoch,
                                     activeColumnsPreviousEpoch)
      stabilityTrace.append(stability)

      numConnectedSynapsesTrace.append(np.sum(connectedCounts))

      numNewSynapses = connectedSyns - connectedSynsPreviousEpoch
      numNewSynapses[numNewSynapses < 0] = 0
      numNewlyConnectedSynapsesTrace.append(np.sum(numNewSynapses))

      numEliminatedSynapses = connectedSynsPreviousEpoch - connectedSyns
      numEliminatedSynapses[numEliminatedSynapses < 0] = 0
      numEliminatedSynapsesTrace.append(np.sum(numEliminatedSynapses))

      metrics = {'connected syn': [numConnectedSynapsesTrace[-1]],
                 'new syn': [numNewlyConnectedSynapsesTrace[-1]],
                 'remove syn': [numEliminatedSynapsesTrace[-1]],
                 'stability': [stabilityTrace[-1]],
                 'entropy': [entropyTrace[-1]],
                 'reconstr error': [reconstructionErrorTrace[-1]],
                 'witness error': [witnessErrorTrace[-1]]}
      if trackOverlapCurveOverTraining:
        metrics['noise-robustness'] = [noiseRobustnessTrace[-1]]
      if classification:
        metrics['classification'] = [classificationRobustnessTrace[-1]]
      print tabulate(metrics, headers="keys")
    else:
      stabilityTrace.append(np.nan)
      numNewlyConnectedSynapsesTrace.append(np.nan)
      numConnectedSynapsesTrace.append(np.nan)
      numEliminatedSynapsesTrace.append(np.nan)


    if epoch in checkPoints:
      # inspect SP again
      inspectSpatialPoolerStats(sp, inputVectors, expName+"epoch{}".format(epoch))

      # analyze RF properties
      if inputVectorType == "randomSDR":
        analyzeReceptiveFieldSparseInputs(inputVectors, sp)
        plt.savefig('figures/inputOverlaps/{}/epoch{}.pdf'.format(expName, epoch))
      elif inputVectorType == 'correlatedSDRPairs':
        additionalInfo = sdrData.getAdditionalInfo()
        inputVectors1 = additionalInfo["inputVectors1"]
        inputVectors2 = additionalInfo["inputVectors2"]
        corrPairs = additionalInfo["corrPairs"]
        analyzeReceptiveFieldCorrelatedInputs(
          inputVectors, sp, params, inputVectors1, inputVectors2)
        plt.savefig(
          'figures/inputOverlaps/{}/epoch{}.pdf'.format(expName, epoch))
      elif (inputVectorType == "randomBarPairs" or
                inputVectorType == "randomCross" or
                inputVectorType == "randomBarSets"):
        plotReceptiveFields2D(sp, params['nX'], params['nY'])
        plt.savefig(
          'figures/inputOverlaps/{}/epoch{}.pdf'.format(expName, epoch))


  if spatialImp == "monitored_sp":
    # plot permanence for a single column when monitored sp is used
    columnIndex = 240
    permInfo = sp.recoverPermanence(columnIndex, getPermTrace=1)
    permTrace = permInfo['permTrace']
    plotPermInfo(permInfo)

  plotExampleInputOutput(sp, inputVectors, expName + "final")

  # plot stats over training
  fileName = 'figures/network_stats_over_training_{}.pdf'.format(expName)
  axs = plotSPstatsOverTime(numNewlyConnectedSynapsesTrace,
                            numEliminatedSynapsesTrace,
                            noiseRobustnessTrace,
                            stabilityTrace,
                            entropyTrace,
                            fileName)

  if classification:
    # classify SDRs with noise
    for epoch in range(numEpochs):
      npzfile = np.load(
        './results/classification/{}/epoch_{}.npz'.format(expName, epoch))

  traces = {'numNewlyConnectedSynapsesTrace': numNewlyConnectedSynapsesTrace,
            'numEliminatedSynapsesTrace': numEliminatedSynapsesTrace,
            'noiseRobustnessTrace': noiseRobustnessTrace,
            'stabilityTrace': stabilityTrace,
            'entropyTrace': entropyTrace,
            'expName': expName}
  pickle.dump(traces, open('./results/traces/{}/trace'.format(expName), 'wb'))