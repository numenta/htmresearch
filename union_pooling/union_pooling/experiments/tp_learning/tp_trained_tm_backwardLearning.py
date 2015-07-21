# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import copy
import sys
import time
import os
import yaml
from optparse import OptionParser

import numpy
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.ion()

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from experiments.capacity import data_utils
from union_pooling.experiments.union_pooler_experiment import (
    UnionPoolerExperiment)

_SHOW_PROGRESS_INTERVAL = 200


"""
Experiment 1
Runs UnionPooler on input from a Temporal Memory after training
on a long sequence

Enables learning in UnionPooler
Tests different learning rules
- Forward learning is Hebbian learning on union pooled cells
- Backward learning is Reinforcement-like learning that allows cells to
  connect to inputs from the previous few time steps
- Several metrics are measured before and after learning, including
  average response latency, total size of union, overlap between learned &
  naive representations
"""


ncol = 1024
learnType = 'ForwardBackward'
learningPasses = 100
paramDir = 'params/5_trainingPasses_'+str(ncol)+'_columns_'+learnType+'.yaml'
outputDir = 'results/'+str(ncol)+learnType+'/'

if not os.path.exists(outputDir):
  os.makedirs(outputDir)

params = yaml.safe_load(open(paramDir, 'r'))
options = {'plotVerbosity': 2, 'consoleVerbosity': 2}
plotVerbosity = 2
consoleVerbosity = 1


print "Running SDR overlap experiment...\n"
print "Params dir: {0}".format(paramDir)
print "Output dir: {0}\n".format(outputDir)

# Dimensionality of sequence patterns
patternDimensionality = params["patternDimensionality"]

# Cardinality (ON / true bits) of sequence patterns
patternCardinality = params["patternCardinality"]

# Length of sequences shown to network
sequenceLength = params["sequenceLength"]

# Number of sequences used. Sequences may share common elements.
numberOfSequences = params["numberOfSequences"]

# Number of sequence passes for training the TM. Zero => no training.
trainingPasses = params["trainingPasses"]

# Generate a sequence list and an associated labeled list (both containing a
# set of sequences separated by None)
print "\nGenerating sequences..."
patternAlphabetSize = sequenceLength * numberOfSequences
patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                patternAlphabetSize)
sequenceMachine = SequenceMachine(patternMachine)

numbers = sequenceMachine.generateNumbers(numberOfSequences, sequenceLength)
generatedSequences = sequenceMachine.generateFromNumbers(numbers)
sequenceLabels = [str(numbers[i + i*sequenceLength: i + (i+1)*sequenceLength])
                  for i in xrange(numberOfSequences)]
labeledSequences = []
for label in sequenceLabels:
  for _ in xrange(sequenceLength):
    labeledSequences.append(label)
  labeledSequences.append(None)


def initializeNetwork():
  tmParamOverrides = params["temporalMemoryParams"]
  upParamOverrides = params["unionPoolerParams"]

  # Set up the Temporal Memory and Union Pooler network
  print "\nCreating network..."
  experiment = UnionPoolerExperiment(tmParamOverrides, upParamOverrides)
  return experiment


def runTMtrainingPhase(experiment):

  # Train only the Temporal Memory on the generated sequences
  if trainingPasses > 0:

    print "\nTraining Temporal Memory..."
    if consoleVerbosity > 0:
      print "\nPass\tBursting Columns Mean\tStdDev\tMax"

    for i in xrange(trainingPasses):
      experiment.runNetworkOnSequences(generatedSequences,
                                       labeledSequences,
                                       tmLearn=True,
                                       upLearn=None,
                                       verbosity=consoleVerbosity,
                                       progressInterval=_SHOW_PROGRESS_INTERVAL)

      if consoleVerbosity > 0:
        stats = experiment.getBurstingColumnsStats()
        print "{0}\t{1}\t{2}\t{3}".format(i, stats[0], stats[1], stats[2])

      # Reset the TM monitor mixin's records accrued during this training pass
      # experiment.tm.mmClearHistory()

    print
    print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics())
    print



def SDRsimilarity(SDR1, SDR2):
  return float(len(SDR1 & SDR2)) / float(max(len(SDR1), len(SDR2) ))


def getUnionSDRSimilarityCurve(activeCellsTrace, trainingPasses, sequenceLength, maxSeparation, skipBeginningElements=0):

  similarityVsSeparation = numpy.zeros((trainingPasses, maxSeparation))
  for rpts in xrange(trainingPasses):
    for sep in xrange(maxSeparation):
      similarity = []
      for i in xrange(rpts*sequenceLength+skipBeginningElements, rpts*sequenceLength+sequenceLength-sep):
        similarity.append(SDRsimilarity(activeCellsTrace[i], activeCellsTrace[i+sep]))

      similarityVsSeparation[rpts, sep] = numpy.mean(similarity)

  return similarityVsSeparation


def plotSDRsimilarityVsTemporalSeparation(similarityVsSeparationBefore, similarityVsSeparationAfter):
  # plot SDR similarity as a function of temporal separation
  f, (axs) = plt.subplots(nrows=2,ncols=2)
  rpt = 0
  ax1 = axs[0,0]
  ax1.plot(similarityVsSeparationBefore[rpt,:],label='Before')
  ax1.plot(similarityVsSeparationAfter[rpt,:],label='After')
  ax1.set_xlabel('Separation in time between SDRs')
  ax1.set_ylabel('SDRs overlap')
  # ax1.set_title('Initial Cycle')
  ax1.set_ylim([0,1])
  ax1.legend(loc='upper right')
  # rpt=4
  # ax2.plot(similarityVsSeparationBefore[rpt,:],label='Before')
  # ax2.plot(similarityVsSeparationAfter[rpt,:],label='After')
  # ax2.set_xlabel('Separation in time between SDRs')
  # ax2.set_ylabel('SDRs overlap')
  # ax2.set_title('Last Cycle')
  # ax2.set_ylim([0,1])
  # ax2.legend(loc='upper right')
  f.savefig(outputDir+'UnionSDRoverlapVsTemporalSeparation.eps',format='eps')


def plotSimilarityMatrix(similarityMatrixBefore, similarityMatrixAfter):
  f, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
  im = ax1.imshow(similarityMatrixBefore[0:sequenceLength, 0:sequenceLength],interpolation="nearest")
  ax1.set_xlabel('Time (steps)')
  ax1.set_ylabel('Time (steps)')
  ax1.set_title('Overlap - Before Learning')

  im = ax2.imshow(similarityMatrixAfter[0:sequenceLength, 0:sequenceLength],interpolation="nearest")
  ax2.set_xlabel('Time (steps)')
  ax2.set_ylabel('Time (steps)')
  ax2.set_title('Overlap - After Learning')
  # cax,kw = mpl.colorbar.make_axes([ax1, ax2])
  # plt.colorbar(im, cax=cax, **kw)
  # plt.tight_layout()
  f.savefig(outputDir+'/UnionSDRoverlapBeforeVsAfterLearning.eps',format='eps')


def calculateSimilarityMatrix(activeCellsTraceBefore, activeCellsTraceAfter):
  nSteps = sequenceLength # len(activeCellsTraceBefore)
  similarityMatrixBeforeAfter = numpy.zeros((nSteps, nSteps))
  similarityMatrixBefore = numpy.zeros((nSteps, nSteps))
  similarityMatrixAfter = numpy.zeros((nSteps, nSteps))
  for i in xrange(nSteps):
    for j in xrange(nSteps):
      similarityMatrixBefore[i,j] = SDRsimilarity(activeCellsTraceBefore[i], activeCellsTraceBefore[j])
      similarityMatrixAfter[i,j] = SDRsimilarity(activeCellsTraceAfter[i], activeCellsTraceAfter[j])
      similarityMatrixBeforeAfter[i,j] = SDRsimilarity(activeCellsTraceBefore[i], activeCellsTraceAfter[j])

  return (similarityMatrixBefore, similarityMatrixAfter, similarityMatrixBeforeAfter)


def plotTPRvsUPROverlap(similarityMatrix):
  f = plt.figure()
  plt.subplot(2,2,1)
  im = plt.imshow(similarityMatrix[0:sequenceLength, 0:sequenceLength],
                  interpolation="nearest",aspect='auto', vmin=0, vmax=0.3)
  plt.colorbar(im)
  plt.xlabel('UPR over time')
  plt.ylabel('TPR over time')
  plt.title(' Overlap between UPR & TPR')
  f.savefig(outputDir+'OverlapTPRvsUPR.eps',format='eps')


def bitLifeVsLearningCycles(activeCellsTrace, numColumns,learningPasses, sequenceLength):
  bitLifeVsLearning = numpy.zeros(learningPasses)
  for i in xrange(learningPasses):
    bitLifeList = []
    bitLifeCounter = numpy.zeros(numColumns)

    for t in xrange(sequenceLength):
      preActiveCells = set(numpy.where(bitLifeCounter>0)[0])
      currentActiveCells = activeCellsTrace[i*sequenceLength+t]
      newActiveCells = list(currentActiveCells - preActiveCells)
      stopActiveCells = list(preActiveCells - currentActiveCells)
      if t == sequenceLength-1:
        stopActiveCells = list(currentActiveCells)
      continuousActiveCells = list(preActiveCells & currentActiveCells)
      bitLifeList += list(bitLifeCounter[stopActiveCells])

      bitLifeCounter[stopActiveCells] = 0
      bitLifeCounter[newActiveCells] = 1
      bitLifeCounter[continuousActiveCells] += 1

    bitLifeVsLearning[i] = numpy.mean(bitLifeList)

  return bitLifeVsLearning


def showSequenceStartLine(ax, trainingPasses, sequenceLength):
  for i in xrange(trainingPasses):
    ax.vlines(i*sequenceLength, 0, ax.get_ylim()[0], linestyles='--')


def runTestPhase(experiment, tmLearn=False, upLearn=True, outputfileName='results/TemporalPoolingOutputs.pdf'):

  print "\nRunning test phase..."
  print "tmLearn: ", tmLearn
  print "upLearn: ", upLearn
  inputSequences = generatedSequences
  inputCategories = labeledSequences

  experiment.tm.mmClearHistory()
  experiment.up.mmClearHistory()
  experiment.tm.reset()
  experiment.up.reset()

  # Persistence levels across time
  poolingActivationTrace = numpy.zeros((experiment.up._numColumns, 0))
  # union SDR across time
  activeCellsTrace = numpy.zeros((experiment.up._numColumns, 0))
  # active cells in SP across time
  activeSPTrace = numpy.zeros((experiment.up._numColumns, 0))
  # number of connections for SP cells
  connectionCountTrace = numpy.zeros((experiment.up._numColumns, 0))
  # number of active inputs per SP cells
  activeOverlapsTrace = numpy.zeros((experiment.up._numColumns, 0))
  # number of predicted active inputs per SP cells
  predictedActiveOverlapsTrace = numpy.zeros((experiment.up._numColumns, 0))

  for _ in xrange(trainingPasses):
    experiment.tm.reset()
    experiment.up.reset()
    for i in xrange(len(inputSequences)):
      sensorPattern = inputSequences[i]
      inputCategory = inputCategories[i]
      if sensorPattern is None:
        pass
      else:
        experiment.tm.compute(sensorPattern,
                        formInternalConnections=True,
                        learn=tmLearn,
                        sequenceLabel=inputCategory)


        activeCells, predActiveCells, burstingCols, = experiment.getUnionPoolerInput()

        overlapsActive = experiment.up._calculateOverlap(activeCells)
        overlapsPredictedActive = experiment.up._calculateOverlap(predActiveCells)
        activeOverlapsTrace = numpy.concatenate((activeOverlapsTrace, overlapsActive.reshape((experiment.up._numColumns,1))), 1)
        predictedActiveOverlapsTrace = numpy.concatenate((predictedActiveOverlapsTrace, overlapsPredictedActive.reshape((experiment.up._numColumns,1))), 1)

        experiment.up.compute(activeCells,
                        predActiveCells,
                        learn=upLearn,
                        sequenceLabel=inputCategory)


        currentPoolingActivation = experiment.up._poolingActivation.reshape((experiment.up._numColumns, 1))
        poolingActivationTrace = numpy.concatenate((poolingActivationTrace, currentPoolingActivation), 1)

        currentUnionSDR = numpy.zeros((experiment.up._numColumns, 1))
        currentUnionSDR[experiment.up._unionSDR] = 1
        activeCellsTrace = numpy.concatenate((activeCellsTrace, currentUnionSDR), 1)

        currentSPSDR = numpy.zeros((experiment.up._numColumns, 1))
        currentSPSDR[experiment.up._activeCells] = 1
        activeSPTrace = numpy.concatenate((activeSPTrace, currentSPSDR), 1)

        connectionCountTrace = numpy.concatenate((connectionCountTrace,
                                                  experiment.up._connectedCounts.reshape((experiment.up._numColumns, 1))), 1)

    print "\nPass\tBursting Columns Mean\tStdDev\tMax"
    stats = experiment.getBurstingColumnsStats()
    print "{0}\t{1}\t{2}\t{3}".format(_, stats[0], stats[1], stats[2])
    # print
    # print MonitorMixinBase.mmPrettyPrintMetrics(\
    #     experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
    # print
    experiment.tm.mmClearHistory()

  newConnectionCountTrace = numpy.zeros(connectionCountTrace.shape)
  n = newConnectionCountTrace.shape[1]
  newConnectionCountTrace[:,0:n-2] = connectionCountTrace[:,1:n-1] - connectionCountTrace[:,0:n-2]

  # estimate fraction of shared bits across adjacent time point
  unionSDRshared = experiment.up._mmComputeUnionSDRdiff()

  bitLifeList = experiment.up._mmComputeBitLifeStats()
  bitLife = numpy.array(bitLifeList)


  # Plot SP outputs, UP persistence and UP outputs in testing phase
  ncolShow = 100
  f, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3)
  ax1.imshow(activeSPTrace[1:ncolShow,:], cmap=cm.Greys,interpolation="nearest",aspect='auto')
  showSequenceStartLine(ax1, trainingPasses, sequenceLength)
  ax1.set_title('SP SDR')
  ax1.set_ylabel('Columns')

  ax2.imshow(poolingActivationTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
  showSequenceStartLine(ax2, trainingPasses, sequenceLength)
  ax2.set_title('Persistence')

  ax3.imshow(activeCellsTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
  showSequenceStartLine(ax3, trainingPasses, sequenceLength)
  ax3.set_title('Union SDR')

  # ax4.imshow(newConnectionCountTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
  # showSequenceStartLine(ax4, trainingPasses, sequenceLength)
  # ax4.set_title('New Connection #')
  # ax2.set_xlabel('Time (steps)')

  pp = PdfPages(outputfileName)
  pp.savefig()
  pp.close()




def runTPLearnPhase(experiment, learningPasses):
  tmLearn = False
  upLearn = True

  inputSequences = generatedSequences
  inputCategories = labeledSequences

  experiment.tm.mmClearHistory()
  experiment.up.mmClearHistory()
  experiment.tm.reset()
  experiment.up.reset()

  # Persistence levels across time
  poolingActivationTrace = numpy.zeros((experiment.up._numColumns, 0))
  # union SDR across time
  activeCellsTrace = numpy.zeros((experiment.up._numColumns, 0))
  # active cells in SP across time
  activeSPTrace = numpy.zeros((experiment.up._numColumns, 0))
  # number of connections for SP cells
  connectionCountTrace = numpy.zeros((experiment.up._numColumns, 0))
  # number of active inputs per SP cells
  activeOverlapsTrace = numpy.zeros((experiment.up._numColumns, 0))
  # number of predicted active inputs per SP cells
  predictedActiveOverlapsTrace = numpy.zeros((experiment.up._numColumns, 0))

  permamenceTrace = numpy.zeros((experiment.tm.numberOfCells(), 0))

  for _ in xrange(learningPasses):
    experiment.tm.reset()
    experiment.up.reset()
    for i in xrange(len(inputSequences)):
      sensorPattern = inputSequences[i]
      inputCategory = inputCategories[i]
      if sensorPattern is None:
        pass
      else:
        experiment.tm.compute(sensorPattern,
                        formInternalConnections=True,
                        learn=tmLearn,
                        sequenceLabel=inputCategory)


        activeCells, predActiveCells, burstingCols, = experiment.getUnionPoolerInput()

        overlapsActive = experiment.up._calculateOverlap(activeCells)
        overlapsPredictedActive = experiment.up._calculateOverlap(predActiveCells)
        activeOverlapsTrace = numpy.concatenate((activeOverlapsTrace, overlapsActive.reshape((experiment.up._numColumns,1))), 1)
        predictedActiveOverlapsTrace = numpy.concatenate((predictedActiveOverlapsTrace, overlapsPredictedActive.reshape((experiment.up._numColumns,1))), 1)

        # print ' step: ', i
        # print 'current predicted input:  ',numpy.where(predActiveCells)[0]
        # print 'previous predicted input: ',numpy.where(experiment.up._prePredictedActiveInput)[0]
        # print 'overlapsPredictedActive: ', overlapsPredictedActive[31]

        experiment.up.compute(activeCells,
                        predActiveCells,
                        learn=upLearn,
                        sequenceLabel=inputCategory)


        # print 'active UP cells: ', experiment.up._activeCells

        # permamenceTrace = numpy.concatenate((permamenceTrace,
        #                                      experiment.up._permanences.getRow(31).reshape((experiment.tm.numberOfCells(),1))),1)
        #
        # currentPoolingActivation = experiment.up._poolingActivation.reshape((experiment.up._numColumns, 1))
        # poolingActivationTrace = numpy.concatenate((poolingActivationTrace, currentPoolingActivation), 1)
        #
        # currentUnionSDR = numpy.zeros((experiment.up._numColumns, 1))
        # currentUnionSDR[experiment.up._unionSDR] = 1
        # activeCellsTrace = numpy.concatenate((activeCellsTrace, currentUnionSDR), 1)
        #
        # currentSPSDR = numpy.zeros((experiment.up._numColumns, 1))
        # currentSPSDR[experiment.up._activeCells] = 1
        # activeSPTrace = numpy.concatenate((activeSPTrace, currentSPSDR), 1)
        #
        # connectionCountTrace = numpy.concatenate((connectionCountTrace,
        #                                           experiment.up._connectedCounts.reshape((experiment.up._numColumns, 1))), 1)

    print "\nPass\tBursting Columns Mean\tStdDev\tMax"
    stats = experiment.getBurstingColumnsStats()
    print "{0}\t{1}\t{2}\t{3}".format(_, stats[0], stats[1], stats[2])
    # print
    # print MonitorMixinBase.mmPrettyPrintMetrics(\
    #     experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
    # print
    experiment.tm.mmClearHistory()


  # Plot SP outputs, UP persistence and UP outputs in testing phase

  ncolShow = 50
  f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1,ncols=4)
  ax1.imshow(activeSPTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest", aspect='auto')
  showSequenceStartLine(ax1, trainingPasses, sequenceLength)
  ax1.set_title('SP SDR')
  ax1.set_ylabel('Columns')

  ax2.imshow(poolingActivationTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest", aspect='auto')
  showSequenceStartLine(ax2, trainingPasses, sequenceLength)
  ax2.set_title('Persistence')

  ax3.imshow(activeCellsTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest", aspect='auto')
  showSequenceStartLine(ax3, trainingPasses, sequenceLength)
  ax3.set_title('Union SDR')

  ax4.imshow(predictedActiveOverlapsTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
  showSequenceStartLine(ax4, trainingPasses, sequenceLength)
  ax4.set_title('New Connection #')
  ax2.set_xlabel('Time (steps)')


def plotSummaryResults(upBeforeLearning, upDuringLearning, upAfterLearning, learningPasses):
  maxSeparation = 30
  skipBeginningElements = 10
  activeCellsTraceBefore = upBeforeLearning._mmTraces['activeCells'].data
  similarityVsSeparationBefore = getUnionSDRSimilarityCurve(activeCellsTraceBefore,  trainingPasses, sequenceLength,
                                                            maxSeparation, skipBeginningElements)

  activeCellsTraceAfter = upAfterLearning._mmTraces['activeCells'].data
  similarityVsSeparationAfter = getUnionSDRSimilarityCurve(activeCellsTraceAfter,  trainingPasses, sequenceLength,
                                                           maxSeparation, skipBeginningElements)

  plotSDRsimilarityVsTemporalSeparation(similarityVsSeparationBefore, similarityVsSeparationAfter)

  (similarityMatrixBefore, similarityMatrixAfter, similarityMatrixBeforeAfter) = \
    calculateSimilarityMatrix(activeCellsTraceBefore, activeCellsTraceAfter)

  plotTPRvsUPROverlap(similarityMatrixBeforeAfter)

  plotSimilarityMatrix(similarityMatrixBefore, similarityMatrixAfter)


  activeCellsTrace = upDuringLearning._mmTraces["activeCells"].data
  meanBitLifeVsLearning = bitLifeVsLearningCycles(activeCellsTrace, experiment.up._numColumns, learningPasses, sequenceLength)

  numBitsUsed = []
  avgBitLatency = []
  for rpt in xrange(learningPasses):
    allActiveBits = set()
    for i in xrange(sequenceLength):
      allActiveBits |= (set(activeCellsTrace[rpt*sequenceLength+i]))

    bitActiveTime = numpy.ones(experiment.up._numColumns) * -1
    for i in xrange(sequenceLength):
      curActiveCells = list(activeCellsTrace[rpt*sequenceLength+i])
      for j in xrange(len(curActiveCells)):
        if bitActiveTime[curActiveCells[j]] < 0:
          bitActiveTime[curActiveCells[j]] = i

    bitActiveTimeSummary = bitActiveTime[bitActiveTime>0]
    print 'pass ', rpt, ' num bits: ', len(allActiveBits), ' latency : ',numpy.mean(bitActiveTimeSummary)

    numBitsUsed.append(len(allActiveBits))
    avgBitLatency.append(numpy.mean(bitActiveTimeSummary))


  f = plt.figure()
  plt.subplot(2,2,1)
  plt.plot(numBitsUsed)
  plt.xlabel(' learning pass #')
  plt.ylabel(' number of cells in UPR')
  plt.ylim([100,600])
  plt.subplot(2,2,2)
  plt.plot(avgBitLatency)
  plt.xlabel(' learning pass #')
  plt.ylabel(' average latency ')
  plt.ylim([11,19])
  plt.subplot(2,2,3)
  plt.plot(meanBitLifeVsLearning)
  plt.xlabel(' learning pass #')
  plt.ylabel(' average lifespan ')
  plt.ylim([10,30])
  f.savefig(outputDir+'SDRpropertyOverLearning.eps',format='eps')


if __name__ == "__main__":
  experiment = initializeNetwork()
  runTMtrainingPhase(experiment)
  runTestPhase(experiment, tmLearn=False, upLearn=False, outputfileName=outputDir+'TemporalPoolingBeforeLearning.pdf')
  upBeforeLearning = copy.deepcopy(experiment.up)

  runTPLearnPhase(experiment, learningPasses)
  upDuringLearning = copy.deepcopy(experiment.up)

  runTestPhase(experiment, tmLearn=False, upLearn=False, outputfileName=outputDir+'TemporalPoolingAfterLearning.pdf')
  upAfterLearning = copy.deepcopy(experiment.up)

  plotSummaryResults(upBeforeLearning, upDuringLearning, upAfterLearning, learningPasses)
