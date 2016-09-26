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
from tabulate import tabulate
from pylab import rcParams


from nupic.research.spatial_pooler import SpatialPooler as PYSpatialPooler
from htmresearch.algorithms.faulty_spatial_pooler import FaultySpatialPooler
from htmresearch.frameworks.sp_paper.sp_metrics import (
  calculateEntropy, inspectSpatialPoolerStats,
  classificationAccuracyVsNoise, getRFCenters, calculateOverlapCurve,
  calculateStability, calculateInputSpaceCoverage
)
from htmresearch.support.spatial_pooler_monitor_mixin import (
  SpatialPoolerMonitorMixin)

from htmresearch.support.sp_paper_utils import *

class MonitoredSpatialPooler(SpatialPoolerMonitorMixin,
                             PYSpatialPooler): pass

class MonitoredFaultySpatialPooler(SpatialPoolerMonitorMixin,
                                   FaultySpatialPooler): pass


from nupic.bindings.algorithms import SpatialPooler as CPPSpatialPooler
from nupic.bindings.math import GetNTAReal

from nupic.math.topology import indexFromCoordinates, coordinatesFromIndex

from htmresearch.support.generate_sdr_dataset import SDRDataSet, getBar

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
  elif spatialImp == "monitored_faulty_sp":
    sp = MonitoredFaultySpatialPooler(**spatialPoolerParameters)

  else:
    raise RuntimeError("Invalide spatialImp")
  return sp



def getSpatialPoolerParams(params, boosting=0, inputVectorType=None):
  if boosting > 0:
    from model_params.sp_params import spParamTopologyWithBoosting as spatialPoolerParameters
  else:
    from model_params.sp_params import spParamTopologyNoBoosting as spatialPoolerParameters

  if inputVectorType in ['randomCross']:
    from model_params.sp_params import spParamTopologyWithBoostingCross as spatialPoolerParameters

  spatialPoolerParameters['inputDimensions'] = (params['nX'], params['nY'])

  print "Spatial Pooler Parameters: "
  pprint.pprint(spatialPoolerParameters)
  return spatialPoolerParameters



def plotReceptiveFields2D(sp, Nx, Ny, seed=42):
  inputSize = Nx * Ny
  numColumns = np.product(sp.getColumnDimensions())

  nrows = 4
  ncols = 4
  fig, ax = plt.subplots(nrows, ncols)
  np.random.seed(seed)
  for r in range(nrows):
    for c in range(ncols):
      colID = np.random.randint(numColumns)
      connectedSynapses = np.zeros((inputSize,), dtype=uintType)
      sp.getConnectedSynapses(colID, connectedSynapses)

      potentialSyns = np.zeros((inputSize,), dtype=uintType)
      sp.getPotential(colID, potentialSyns)

      receptiveField = np.zeros((inputSize,), dtype=uintType)
      receptiveField[potentialSyns > 0] = 1
      receptiveField[connectedSynapses > 0] = 5
      receptiveField = np.reshape(receptiveField, (Nx, Ny))

      ax[r, c].imshow(receptiveField, interpolation="nearest", cmap='gray')
      ax[r, c].set_xticks([])
      ax[r, c].set_yticks([])
      ax[r, c].set_title('col {}'.format(colID))
  return fig



def _getArgs():
  parser = OptionParser(usage="Train HTM Spatial Pooler")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='randomBarSets',
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

  parser.add_option("--checkRFCenters",
                    type=int,
                    default=0,
                    dest="checkRFCenters",
                    help="whether to track RF cneters")


  parser.add_option("--checkTestInput",
                    type=int,
                    default=0,
                    dest="checkTestInput",
                    help="whether to check response to test inputs")


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
                    default=-1,
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

  (options, remainder) = parser.parse_args()
  print options
  return options, remainder



def updatePotentialRadius(sp, newPotentialRadius):
  """
  Change the potential radius for all columns
  :return:
  """
  oldPotentialRadius = sp._potentialRadius
  sp._potentialRadius = newPotentialRadius
  numColumns = np.prod(sp.getColumnDimensions())
  for columnIndex in xrange(numColumns):
    potential = sp._mapPotential(columnIndex)
    sp._potentialPools.replace(columnIndex, potential.nonzero()[0])

  sp._updateInhibitionRadius()



def initializeSPConnections(sp, potentialRaidus=10, initConnectionRadius=5):
  numColumns = np.prod(sp.getColumnDimensions())

  updatePotentialRadius(sp, newPotentialRadius=initConnectionRadius)
  for columnIndex in xrange(numColumns):
    potential = sp._mapPotential(columnIndex)
    sp._potentialPools.replace(columnIndex, potential.nonzero()[0])
    perm = sp._initPermanence(potential, 0.5)
    sp._updatePermanencesForColumn(perm, columnIndex, raisePerm=True)

  updatePotentialRadius(sp, newPotentialRadius=potentialRaidus)


def getSDRDataSetParams(inputVectorType):
  if inputVectorType == 'randomBarSets':
    params = {'dataType': 'randomBarSets',
              'numInputVectors': 100,
              'nX': 32,
              'nY': 32,
              'barHalfLength': 6,
              'numBarsPerInput': 6,
              'seed': 41}
  elif inputVectorType == 'randomBarPairs':
    params = {'dataType': 'randomBarSets',
              'numInputVectors': 1000,
              'nX': 32,
              'nY': 32,
              'barHalfLength': 3,
              'numBarsPerInput': 6,
              'seed': 41}
  elif inputVectorType == 'randomCross':
    params = {'dataType': 'randomCross',
              'numInputVectors': 200,
              'nX': 32,
              'nY': 32,
              'barHalfLength': 3,
              'numCrossPerInput': 6,
              'seed': 41}
  else:
    raise ValueError('unknown data type')

  return params


if __name__ == "__main__":
  plt.close('all')

  (_options, _args) = _getArgs()
  inputVectorType = _options.dataSet
  numEpochs = _options.numEpochs
  classification = _options.classification
  spatialImp = _options.spatialImp
  trackOverlapCurveOverTraining = _options.trackOverlapCurve
  changeDataSetContinuously = _options.changeDataSetContinuously
  boosting = _options.boosting
  checkTestInput = _options.checkTestInput
  changeDataSetAt = _options.changeDataSetAt
  checkRFCenters = _options.checkRFCenters
  checkRFCenters = _options.checkRFCenters
  killCellsAt = _options.killCellsAt
  killCellPrct = _options.killCellPrct
  expName = _options.expName
  inputVectorType = _options.dataSet

  params = getSDRDataSetParams(inputVectorType)

  if expName == 'defaultName':
    expName = "dataType_{}_boosting_{}".format(
      inputVectorType, _options.boosting)

  sdrData = SDRDataSet(params)
  inputVectors = sdrData.getInputVectors()
  numInputVector, inputSize = inputVectors.shape

  plt.imshow(np.reshape(inputVectors[0], (params['nX'], params['nY'])),
             interpolation='nearest', cmap='gray')
  print "Training Data Size {} Dimensions {}".format(numInputVector, inputSize)

  spParams = getSpatialPoolerParams(params, boosting, inputVectorType)
  sp = createSpatialPooler(spatialImp, spParams)

  if spatialImp in ['faulty_sp', 'py']:
    initializeSPConnections(sp, potentialRaidus=10, initConnectionRadius=5)

  columnNumber = np.prod(sp.getColumnDimensions())

  numTestInputs = 5
  testInputs = np.zeros((numTestInputs, inputSize))
  for i in range(numTestInputs):
    orientation = np.random.choice(['horizontal', 'vertical'])
    xLoc = np.random.randint(16, 17)
    yLoc = np.random.randint(15, 16)
    bar = getBar((params['nX'], params['nY']),
                 (xLoc, yLoc), 1, orientation)
    testInputs[i, :] = np.reshape(bar, newshape=(1, inputSize))

  activeColumnsTestInputs = np.zeros((numTestInputs, columnNumber))
  for i in range(numTestInputs):
    outputColumns = np.zeros((columnNumber, 1), dtype=uintType)
    sp.compute(testInputs[i, :], False, outputColumns)
    # sp.growRandomSynapses()
    activeColumnsTestInputs[i][:] = np.reshape(outputColumns,
                                               (1, columnNumber))

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
  noiseRobustnessTrace = []
  classificationRobustnessTrace = []
  activityTrace = []

  epoch = 0
  while epoch < numEpochs:
    print "training SP epoch {} ".format(epoch)
    if changeDataSetContinuously or (epoch == changeDataSetAt):
      params['seed'] = epoch
      sdrData.generateInputVectors(params)
      inputVectors = sdrData.getInputVectors()
      numInputVector, inputSize = inputVectors.shape
      plt.figure(10)
      plt.clf()
      plt.imshow(np.reshape(np.sum(inputVectors, 0), (params['nX'], params['nY'])),
                 interpolation='nearest', cmap='jet')
      plt.colorbar()
      plt.savefig('figures/avgInputs/{}_epoch_{}'.format(expName, epoch))

    if epoch == killCellsAt:
      if spatialImp == "faulty_sp" or spatialImp == "monitored_faulty_sp":
        # sp.killCells(killCellPrct)
        centerColumn = indexFromCoordinates((15, 15), sp._columnDimensions)
        sp.killCellRegion(centerColumn, 5)

    if trackOverlapCurveOverTraining:
      noiseLevelList, inputOverlapScore, outputOverlapScore = calculateOverlapCurve(
        sp, testInputs)
      noiseRobustnessTrace.append(np.trapz(np.flipud(np.mean(outputOverlapScore, 0)),
                                           noiseLevelList))
      np.savez('./results/input_output_overlap/{}_{}'.format(expName, epoch),
               noiseLevelList, inputOverlapScore, outputOverlapScore)

    if classification:
      # classify SDRs with noise
      noiseLevelList = np.linspace(0, 1.0, 21)
      classification_accuracy = classificationAccuracyVsNoise(
        sp, testInputs, noiseLevelList)
      classificationRobustnessTrace.append(
        np.trapz(classification_accuracy, noiseLevelList))
      np.savez('./results/classification/{}_{}'.format(expName, epoch),
              noiseLevelList, classification_accuracy)

    activeColumnsPreviousEpoch = copy.copy(activeColumnsCurrentEpoch)
    connectedCountsPreviousEpoch = copy.copy(connectedCounts)

    # train SP here,
    # Learn is turned off at the first epoch to gather stats of untrained SP
    learn = False if epoch == 0 else True

    # randomize the presentation order of input vectors
    sdrOrders = np.random.permutation(np .arange(numInputVector))
    activeColumnsCurrentEpoch = runSPOnBatch(
      sp, inputVectors[sdrOrders, :], learn=True)

    connectedCounts = connectedCounts.astype(uintType)
    sp.getConnectedCounts(connectedCounts)
    connectedCounts = connectedCounts.astype(realDType)

    entropyTrace.append(calculateEntropy(activeColumnsCurrentEpoch))

    boostFactors = np.zeros((columnNumber, ), dtype=realDType)
    sp.getBoostFactors(boostFactors)
    meanBoostFactorTrace.append(np.mean(boostFactors))

    activeDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getActiveDutyCycles(activeDutyCycle)

    overlaps = sp.getOverlaps()
    inputOverlapWinner = overlaps[np.where(outputColumns > 0)[0]]
    inputOverlapWinnerTrace.append(np.mean(inputOverlapWinner))

    if checkRFCenters:
      RFcenters, avgDistToCenter = getRFCenters(sp, params, type='connected')
      if spatialImp == 'faulty_sp':
        aliveColumns = sp.getAliveColumns()
      else:
        aliveColumns = np.arange(columnNumber)
      fig = plotReceptiveFieldCenter(RFcenters[aliveColumns, :],
                                     connectedCounts[aliveColumns],
                                     (params['nX'], params['nY']))
      plt.savefig('./figures/RFcenters/{}_epoch_{}.png'.format(expName, epoch))
      plt.close(fig)
      np.savez('./results/RFcenters/{}_{}'.format(expName, epoch),
               RFcenters, avgDistToCenter)

    if checkTestInput:
      RFcenters, avgDistToCenter = getRFCenters(sp, params, type='connected')
      inputIdx = 0
      outputColumns = np.zeros((columnNumber, 1), dtype=uintType)
      sp.compute(testInputs[inputIdx, :], False, outputColumns)
      activeColumns = np.where(outputColumns > 0)[0]
      plt.figure(1)
      plt.clf()
      plt.imshow(
        1 - np.transpose(
          np.reshape(testInputs[inputIdx], (params['nX'], params['nY']))),
        interpolation='nearest', cmap='gray')
      plt.scatter(RFcenters[:, 0], RFcenters[:, 1])
      plt.scatter(RFcenters[activeColumns, 0], RFcenters[activeColumns, 1],
                  color='r')
      plt.xlim([-1, params['nX'] + 1])
      plt.ylim([-1, params['nY'] + 1])

      plt.savefig(
        './figures/ResponseToTestInputs/{}_epoch_{}.png'.format(expName, epoch))

    fig = plotReceptiveFields2D(sp, params['nX'], params['nY'])
    plt.savefig('figures/exampleRFs/{}/epoch_{}'.format(expName, epoch))
    plt.close(fig)

    if epoch >= 0:
      inputSpaceCoverage = calculateInputSpaceCoverage(sp)
      np.savez('./results/InputCoverage/{}_{}'.format(expName, epoch),
               inputSpaceCoverage, connectedCounts)

      plt.figure(2)
      plt.clf()
      plt.imshow(inputSpaceCoverage, interpolation='nearest', cmap="jet")
      plt.colorbar()
      plt.savefig(
        './figures/InputCoverage/{}_epoch_{}.png'.format(expName, epoch))

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

      activityTrace.append(np.mean(np.sum(activeColumnsCurrentEpoch, 1)))
      metrics = {'connected syn': [numConnectedSynapsesTrace[-1]],
                 'new syn': [numNewlyConnectedSynapsesTrace[-1]],
                 'remove syn': [numEliminatedSynapsesTrace[-1]],
                 'stability': [stabilityTrace[-1]],
                 'entropy': [entropyTrace[-1]],
                 'activity': [activityTrace[-1]]}
      if trackOverlapCurveOverTraining:
        metrics['noise-robustness'] = [noiseRobustnessTrace[-1]]
      if classification:
        metrics['classification'] = [classificationRobustnessTrace[-1]]
      print tabulate(metrics, headers="keys")
    epoch += 1

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
  plt.savefig(
    'figures/network_stats_over_training_{}.pdf'.format(inputVectorType))

  plotReceptiveFields2D(sp, params['nX'], params['nY'])
  inspectSpatialPoolerStats(sp, inputVectors,
                            inputVectorType + "afterTraining")

  if trackOverlapCurveOverTraining:
    plt.figure()
    plt.plot(noiseRobustnessTrace)
    plt.xlabel('epochs')
    plt.ylabel('noise robustness')
    plt.savefig('figures/noise_robustness_over_training_{}.pdf'.format(expName))
