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

import csv
import sys
import time
import os
import yaml
from optparse import OptionParser

import numpy
from pylab import rcParams

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from experiments.capacity import data_utils
from union_pooling.experiments.union_pooler_experiment import (
    UnionPoolerExperiment)

"""
Experiment 1
Runs UnionPooler on input from a Temporal Memory after training
on a long sequence
"""



_SHOW_PROGRESS_INTERVAL = 200
_PLOT_RESET_SHADING = 0.2
_PLOT_HEIGHT = 6
_PLOT_WIDTH = 9



def writeDefaultMetrics(outputDir, experiment, patternDimensionality,
                        patternCardinality, patternAlphabetSize, sequenceLength,
                        numberOfSequences, trainingPasses, elapsedTime):
  fileName = "defaultMetrics_{0}learningPasses.csv".format(trainingPasses)
  filePath = os.path.join(outputDir, fileName)
  with open(filePath, "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    header = ["n", "w", "alphabet", "seq length", "num sequences",
              "training passes", "experiment time"]
    row = [patternDimensionality, patternCardinality, patternAlphabetSize,
           sequenceLength, numberOfSequences, trainingPasses, elapsedTime]
    for metric in (experiment.tm.mmGetDefaultMetrics() +
                   experiment.up.mmGetDefaultMetrics()):
      header += ["{0} ({1})".format(metric.prettyPrintTitle(), x) for x in
                ["min", "max", "sum", "mean", "stddev"]]
      row += [metric.min, metric.max, metric.sum, metric.mean,
              metric.standardDeviation]
    csvWriter.writerow(header)
    csvWriter.writerow(row)
    outputFile.flush()



def writeMetricTrace(experiment, traceName, outputDir, outputFileName):
  """
  Assume trace elements can be converted to list.
  :param experiment: UnionPoolerExperiment instance
  :param traceName: name of the metric trace
  :param outputDir: dir where output file will be written
  :param outputFileName: filename of output file
  """
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  filePath = os.path.join(outputDir, outputFileName)
  with open(filePath, "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    dataTrace = experiment.up._mmTraces[traceName].data
    rows = [list(datum) for datum in dataTrace]
    csvWriter.writerows(rows)
    outputFile.flush()



def plotNetworkState(experiment, plotVerbosity, trainingPasses, phase=""):

  if plotVerbosity >= 1:
    rcParams["figure.figsize"] = _PLOT_WIDTH, _PLOT_HEIGHT

    # Plot Union SDR trace if it is not empty
    dataTrace = experiment.up._mmTraces["activeCells"].data
    unionSizeTrace = [len(datum) for datum in dataTrace]
    if unionSizeTrace != []:
      x = [i for i in xrange(len(unionSizeTrace))]
      stdDevs = None
      title = "Union Size; {0} training passses vs. Time".format(trainingPasses)
      data_utils.getErrorbarFigure(title, x, unionSizeTrace, stdDevs, "Time",
                                   "Union Size")
    if plotVerbosity >= 2:
      title = "training passes: {0}, phase: {1}".format(trainingPasses, phase)
      experiment.up.mmGetPlotConnectionsPerColumn(title=title)
      experiment.tm.mmGetCellActivityPlot(title=title,
                                          activityType="activeCells",
                                          showReset=True,
                                          resetShading=_PLOT_RESET_SHADING)
      experiment.tm.mmGetCellActivityPlot(title=title,
                                          activityType="predictedActiveCells",
                                          showReset=True,
                                          resetShading=_PLOT_RESET_SHADING)
      experiment.up.mmGetCellActivityPlot(title=title,
                                          showReset=True,
                                          resetShading=_PLOT_RESET_SHADING)


paramDir = 'params/1024_baseline/5_trainingPasses_long_sequence.yaml';
outputDir = 'results/'
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

# TODO If this parameter is to be supported, the sequence generation code
# below must change
# Number of unique patterns from which sequences are built
# patternAlphabetSize = params["patternAlphabetSize"]

# Length of sequences shown to network
sequenceLength = params["sequenceLength"]

# Number of sequences used. Sequences may share common elements.
numberOfSequences = params["numberOfSequences"]

# Number of sequence passes for training the TM. Zero => no training.
trainingPasses = params["trainingPasses"]

tmParamOverrides = params["temporalMemoryParams"]
upParamOverrides = params["unionPoolerParams"]

# Generate a sequence list and an associated labeled list (both containing a
# set of sequences separated by None)
start = time.time()
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

# Set up the Temporal Memory and Union Pooler network
print "\nCreating network..."
experiment = UnionPoolerExperiment(tmParamOverrides, upParamOverrides)

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
    experiment.tm.mmClearHistory()

  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
    experiment.tm.mmGetDefaultMetrics())
  print

  if plotVerbosity >= 2:
    plotNetworkState(experiment, plotVerbosity, trainingPasses, phase="Training")

experiment.tm.mmClearHistory()

experiment.up.mmClearHistory()
print "\nRunning test phase..."
experiment.runNetworkOnSequences(generatedSequences,
                                 labeledSequences,
                                 tmLearn=False,
                                 upLearn=False,
                                 verbosity=consoleVerbosity,
                                 progressInterval=_SHOW_PROGRESS_INTERVAL)

print "\nPass\tBursting Columns Mean\tStdDev\tMax"
stats = experiment.getBurstingColumnsStats()
print "{0}\t{1}\t{2}\t{3}".format(0, stats[0], stats[1], stats[2])
if trainingPasses > 0 and stats[0] > 0:
  print "***WARNING! MEAN BURSTING COLUMNS IN TEST PHASE IS GREATER THAN 0***"

print
print MonitorMixinBase.mmPrettyPrintMetrics(\
    experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
print
plotNetworkState(experiment, plotVerbosity, trainingPasses, phase="Testing")

cellTrace = experiment.up._mmTraces["activeCells"].data
experiment.up.mmGetSubsetCellTracePlot(cellTrace, 100, "activeCells","UP representation subset")

experiment.up.mmGetCellActivityPlot(title="UP representation",
                                    showReset=True,
                                    resetShading=_PLOT_RESET_SHADING)

elapsed = int(time.time() - start)
print "Total time: {0:2} seconds.".format(elapsed)


inputSequences = generatedSequences
inputCategories = labeledSequences
tmLearn = False
upLearn = False
classifierLearn = False
currentTime = time.time()

experiment.tm.reset()
experiment.up.reset()

poolingActivationTrace = numpy.zeros((experiment.up._numColumns, 1))
activeCellsTrace = numpy.zeros((experiment.up._numColumns, 1))
activeSPTrace = numpy.zeros((experiment.up._numColumns, 1))

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

    if upLearn is not None:
      activeCells, predActiveCells, burstingCols, = experiment.getUnionPoolerInput()
      experiment.up.compute(activeCells,
                      predActiveCells,
                      learn=upLearn,
                      sequenceLabel=inputCategory)

      currentPoolingActivation = experiment.up._poolingActivation

      currentPoolingActivation = experiment.up._poolingActivation.reshape((experiment.up._numColumns, 1))
      poolingActivationTrace = numpy.concatenate((poolingActivationTrace, currentPoolingActivation), 1)

      currentUnionSDR = numpy.zeros((experiment.up._numColumns, 1))
      currentUnionSDR[experiment.up._unionSDR] = 1
      activeCellsTrace = numpy.concatenate((activeCellsTrace, currentUnionSDR), 1)

      currentSPSDR = numpy.zeros((experiment.up._numColumns, 1))
      currentSPSDR[experiment.up._activeCells] = 1
      activeSPTrace = numpy.concatenate((activeSPTrace, currentSPSDR), 1)      

      
# estimate fraction of shared bits across adjacent time point      
unionSDRdiff = []
unionSDRshared = []
for t in xrange(activeCellsTrace.shape[1] - 1):
  totalBits = sum(numpy.logical_or(activeCellsTrace[:,t], activeCellsTrace[:,t+1]))
  sharedBits = sum(numpy.logical_and(activeCellsTrace[:,t], activeCellsTrace[:,t+1]))
  numDiffBits = totalBits - sharedBits
  unionSDRdiff.append(numDiffBits)
  unionSDRshared.append( sharedBits/sum(activeCellsTrace[:,t+1]))

bitLifeList = []
bitLifeCounter = numpy.ones(experiment.up._numColumns) * -1
for t in xrange(activeCellsTrace.shape[1] ):
  newActiveCells = numpy.where(numpy.logical_and(activeCellsTrace[:,t]>0, bitLifeCounter==-1))
  continuousActiveCells = numpy.where(numpy.logical_and(activeCellsTrace[:,t]>0, bitLifeCounter>0))
  stopActiveCells = numpy.where(numpy.logical_and(activeCellsTrace[:,t]==0, bitLifeCounter>0))

  bitLifeList.append(list(bitLifeCounter[stopActiveCells]))
  bitLifeCounter[stopActiveCells] = -1
  bitLifeCounter[newActiveCells] = 1
  bitLifeCounter[continuousActiveCells] += 1

bitLife = numpy.zeros((0))
for t in xrange(len(bitLifeList)):
  bitLife = numpy.concatenate((bitLife, numpy.array(bitLifeList[t])), 0)

  # if classifierLearn and sensorPattern is not None:
  #   unionSDR = experiment.up.getUnionSDR()
  #   upCellCount = experiment.up.getColumnDimensions()
  #   experiment.classifier.learn(unionSDR, inputCategory, isSparse=upCellCount)
  #   if verbosity > 0:
  #     pprint.pprint("{0} is category {1}".format(unionSDR, inputCategory))

  # if progressInterval is not None and i > 0 and i % progressInterval == 0:
  #   elapsed = (time.time() - currentTime) / 60.0
  #   print ("Ran {0} / {1} elements of sequence in "
  #          "{2:0.2f} minutes.".format(i, len(inputSequences), elapsed))
  #   currentTime = time.time()
  #   print MonitorMixinBase.mmPrettyPrintMetrics(
  #     experiment.tm.mmGetDefaultMetrics())

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

# from nupic.research.monitor_mixin.plot import Plot
# plot = Plot(experiment.up, "persistence over time")    
# plot.add2DArray(poolingActivationTrace[1:100,:], xlabel="time", ylabel="Cells")


# plot = Plot(experiment.up, "unionSDR over time")    
# plot.add2DArray(activeCellsTrace[1:100,:], xlabel="time",ylabel='Cells')

# plot = Plot(experiment.up, "SP SDR over time")    
# plot.add2DArray(activeSPTrace[1:100,:], xlabel="time", ylabel="Cells")

plt.figure()
plt.subplot(1,3,1)
plt.imshow(activeSPTrace[1:100,:], cmap=cm.Greys,interpolation="nearest")
plt.title('SP SDR')
plt.ylabel('Cells')
plt.subplot(1,3,2)
plt.imshow(poolingActivationTrace[1:100,:], cmap=cm.Greys, interpolation="nearest")
plt.title('Persistence')
plt.xlabel('Time (steps)')
plt.subplot(1,3,3)
plt.imshow(activeCellsTrace[1:100,:], cmap=cm.Greys,interpolation="nearest")
plt.title('Union SDR')
pp = PdfPages('results/UnionSDRexample.pdf')
pp.savefig()
pp.close()

plt.figure()
plt.subplot(2,2,1)
plt.plot(sum(activeCellsTrace))
plt.ylabel('Union SDR size')
plt.xlabel('Time (steps)')

plt.subplot(2,2,2)
plt.plot(unionSDRshared)
plt.ylabel('Shared Bits with previous union SDRs')
plt.xlabel('Time (steps)')
plt.subplot(2,2,3)
plt.hist(bitLife)
plt.xlabel('Life duration for each bit')
pp = PdfPages('results/UnionSDRproperty.pdf')
pp.savefig()
pp.close()

