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
import time

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



def runTestPhase(experiment, consoleVerbosity):
  # TODO
  pass



def outputNetworkState(experiment, plotVerbosity, trainingPasses, phase=""):
  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
    experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
  print
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



def run(trainingPasses, numberOfSequences, sequenceLength,
        patternDimensionality, patternCardinality, patternAlphabetSize,
        consoleVerbosity=0, plotVerbosity=0):
  """
  Runs the union overlap experiment.

  :param isTrainTemporalMemory: If true the temporal memory will be trained
  :param numberOfSequences: Number of unique sequences shown to network
  :param sequenceLength: Length of sequences shown to network
  :param patternDimensionality: Dimensionality of sequence patterns
  :param patternCardinality: Cardinality (ON / true bits) of sequence patterns
  :param patternAlphabetSize: Number of unique patterns from which sequences
  are built
  :param consoleVerbosity: Console output verbosity
  :param plotVerbosity: Plotting verbosity
  """
  start = time.time()

  # Generate a sequence list and an associated labeled list (both containing a
  # set of sequences separated by None)
  print "Generating sequences..."
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
  print "Creating network..."
  tmParamOverrides = {}
  upParamOverrides = {}
  experiment = UnionPoolerExperiment(tmParamOverrides, upParamOverrides)

  # Train only the Temporal Memory on the generated sequences
  if trainingPasses > 0:
    print "Training Temporal Memory..."
    for _ in xrange(trainingPasses):
      experiment.runNetworkOnSequence(generatedSequences,
                                      labeledSequences,
                                      tmLearn=True,
                                      upLearn=None,
                                      verbosity=_VERBOSITY,
                                      progressInterval=_SHOW_PROGRESS_INTERVAL)

    outputNetworkState(experiment, plotVerbosity, trainingPasses,
                       phase="Training")

  print "Running test phase..."
  for i in xrange(numberOfSequences):
    sequence = generatedSequences[i + i * sequenceLength:
                                  i + (i + 1) * sequenceLength]
    labeledSequence = labeledSequences[i + i * sequenceLength:
                                       i + (i + 1) * sequenceLength]
    experiment.runNetworkOnSequence(sequence,
                                    labeledSequence,
                                    tmLearn=False,
                                    upLearn=False,
                                    verbosity=_VERBOSITY,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)

  outputNetworkState(experiment, plotVerbosity, trainingPasses,
                     phase="Testing")

  elapsed = int(time.time() - start)
  print "Total time: {0:2} seconds.".format(elapsed)

  # Write results to output file
  # TODO writeOutput(outputDir, runner, numElems, numWorlds, elapsed)
  if plotVerbosity >= 1:
    raw_input("Press any key to exit...")



if __name__ == "__main__":
  # trainingPasses = 50
  trainingPasses = 0
  numberOfSequences = 4
  sequenceLength = 4
  patternDimensionality = 1024
  patternCardinality = 20
  patternAlphabetSize = 100
  run(trainingPasses, numberOfSequences, sequenceLength,
      patternDimensionality, patternCardinality, patternAlphabetSize)
