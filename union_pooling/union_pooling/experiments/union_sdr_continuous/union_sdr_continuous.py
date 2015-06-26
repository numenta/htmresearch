#!/usr/bin/env python
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
Runs UnionPooler on input from a Temporal Memory with and
without training. Compute overlap between Union SDR representations in two
conditions over time.
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



def run(params, paramDir, outputDir, plotVerbosity=0, consoleVerbosity=0):
  """
  Runs the union overlap experiment.

  :param params: A dict of experiment parameters
  :param paramDir: Path of parameter file
  :param outputDir: Output will be written to this path
  :param plotVerbosity: Plotting verbosity
  :param consoleVerbosity: Console output verbosity
  """
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
      # experiment.tm.mmClearHistory()

    print
    print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics())
    print
    if plotVerbosity >= 2:
      plotNetworkState(experiment, plotVerbosity, trainingPasses,
                       phase="Training")

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
  print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
  print
  plotNetworkState(experiment, plotVerbosity, trainingPasses, phase="Testing")

  elapsed = int(time.time() - start)
  print "Total time: {0:2} seconds.".format(elapsed)

  # Write Union SDR trace
  metricName = "activeCells"
  outputFileName = "unionSdrTrace_{0}learningPasses.csv".format(trainingPasses)
  writeMetricTrace(experiment, metricName, outputDir, outputFileName)

  if plotVerbosity >= 1:
    raw_input("Press any key to exit...")



def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nRun union overlap experiment using params in"
                              " PARAMS_DIR and outputting results to "
                              "OUTPUT_DIR.")
  parser.add_option("-p",
                    "--plot",
                    type=int,
                    default=0,
                    dest="plotVerbosity",
                    help="Plotting verbosity: 0 => none, 1 => summary plots, "
                         "2 => detailed plots")
  parser.add_option("-c",
                    "--console",
                    type=int,
                    default=0,
                    dest="consoleVerbosity",
                    help="Console message verbosity: 0 => none")
  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) < 2:
    parser.print_help(sys.stderr)
    sys.exit()

  with open(args[0]) as paramsFile:
    params = yaml.safe_load(paramsFile)

  return options, args, params



if __name__ == "__main__":
  (options, args, params) = _getArgs()

  paramDir = 'params/1024_baseline/5_trainingPasses.yaml';
  outputDir = 'results/'
  params = yaml.safe_load(paramDir)
  options = {'plotVerbosity': 2, 'consoleVerbosity': 2}
  run(params, args[0], args[1], options.plotVerbosity, options.consoleVerbosity)
