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

from pylab import rcParams

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from union_pooling.experiments.union_pooler_experiment import (
    UnionPoolerExperiment)

"""
Experiment 1
Union Pooling with and without Temporal Memory training
Compute overlap between Union SDR representations in two conditions over time
"""



_SHOW_PROGRESS_INTERVAL = 100
_VERBOSITY = 0
PLOT_RESET_SHADING = 0.2
PLOT_HEIGHT = 6
PLOT_WIDTH = 9



def writeMetricTrace(experiment, traceName, outputDir, outputFileName):
  """
  Assume trace elements can be converted to list.
  :param experiment:
  :param traceName:
  :param outputDir:
  :param outputFileName:
  :return:
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



def plotNetworkState(experiment, plotVerbosity, trainingPasses, phase=""):
  if plotVerbosity >= 1:
    rcParams["figure.figsize"] = PLOT_WIDTH, PLOT_HEIGHT
    title = "training passes: {0}, phase: {1}".format(trainingPasses, phase)
    experiment.up.mmGetPlotConnectionsPerColumn(title=title)
    if plotVerbosity >= 2:
      experiment.tm.mmGetCellActivityPlot(title=title,
                                          activityType="activeCells",
                                          showReset=True,
                                          resetShading=PLOT_RESET_SHADING)
      experiment.tm.mmGetCellActivityPlot(title=title,
                                          activityType="predictedActiveCells",
                                          showReset=True,
                                          resetShading=PLOT_RESET_SHADING)
      experiment.up.mmGetCellActivityPlot(title=title, showReset=True,
                                          resetShading=PLOT_RESET_SHADING)



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

  # Number of unique patterns from which sequences are built
  patternAlphabetSize = params["patternAlphabetSize"]

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
    print "Training Temporal Memory..."
    for i in xrange(trainingPasses):
      if consoleVerbosity > 0:
        print "\nTraining pass: {0}".format(i)
      experiment.runNetworkOnSequence(generatedSequences,
                                      labeledSequences,
                                      tmLearn=True,
                                      upLearn=None,
                                      verbosity=_VERBOSITY,
                                      progressInterval=_SHOW_PROGRESS_INTERVAL)

    print
    print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics())
    print
    plotNetworkState(experiment, plotVerbosity, trainingPasses,
                     phase="Training")

  print "\nRunning test phase..."
  for i in xrange(numberOfSequences):
    sequence = generatedSequences[i + i * sequenceLength:
                                  (i + 1) + (i + 1) * sequenceLength]
    labeledSequence = labeledSequences[i + i * sequenceLength:
                                       (i + 1) + (i + 1) * sequenceLength]
    experiment.runNetworkOnSequence(sequence,
                                    labeledSequence,
                                    tmLearn=False,
                                    upLearn=False,
                                    verbosity=_VERBOSITY,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)

  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
  print
  plotNetworkState(experiment, plotVerbosity, trainingPasses,
                     phase="Testing")

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
  run(params, args[0], args[1], options.plotVerbosity, options.consoleVerbosity)
