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
import random
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
Experiment 2a
Explore the limits of the classifiability of Union Pooler's union SDR.

Data: Sequences of unique patterns.
Train Phase: Train network on sequences for some number of repetitions
Test phase: Each sequence is shown once and the Union SDRs at the end are
recorded. Compute the distinctness between the final SDRs of each sequence.
"""



_SHOW_PROGRESS_INTERVAL = 200



def getDistinctness(unionSdrs):
    distinctnessConfusion = []

    for i in xrange(len(unionSdrs)):
      for j in xrange(i+1):
        numOverlap = len(unionSdrs[i] & unionSdrs[j])
        distinctnessConfusion.append(numOverlap)

    mean = numpy.mean(distinctnessConfusion)
    stdDev = numpy.std(distinctnessConfusion)
    maximum = max(distinctnessConfusion)
    return mean, stdDev, maximum



def run(params, paramDir, outputDir, plotVerbosity=0, consoleVerbosity=0):
  """
  Runs the Union Pooler capacity experiment.

  :param params: A dict containing the following experiment parameters:

        patternDimensionality - Dimensionality of sequence patterns
        patternCardinality - Cardinality (# ON bits) of sequence patterns
        sequenceLength - Length of sequences shown to network
        sequenceCount - Number of unique sequences used
        trainingPasses - Number of times Temporal Memory is trained on each
        sequence
        temporalMemoryParams - A dict of Temporal Memory parameter overrides
        unionPoolerParams - A dict of Union Pooler parameter overrides

  :param paramDir: Path of parameter file
  :param outputDir: Output will be written to this path
  :param plotVerbosity: Plotting verbosity
  :param consoleVerbosity: Console output verbosity
  """
  print "Running Union Pooler Capacity Experiment...\n"
  print "Params dir: {0}".format(os.path.join(os.path.dirname(__file__),
                                              paramDir))
  print "Output dir: {0}\n".format(os.path.join(os.path.dirname(__file__),
                                                outputDir))

  patternDimensionality = params["patternDimensionality"]
  patternCardinality = params["patternCardinality"]
  sequenceLength = params["sequenceLength"]
  sequenceCount = params["numberOfSequences"]
  trainingPasses = params["trainingPasses"]
  tmParamOverrides = params["temporalMemoryParams"]
  upParamOverrides = params["unionPoolerParams"]

  # Generate a sequence list and an associated labeled list (both containing a
  # set of sequences separated by None)
  start = time.time()
  print "Generating sequences..."
  patternAlphabetSize = sequenceLength * sequenceCount
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)

  numbers = sequenceMachine.generateNumbers(sequenceCount, sequenceLength)
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)
  sequenceLabels = [str(numbers[i + i*sequenceLength: i + (i+1)*sequenceLength])
                    for i in xrange(sequenceCount)]
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
      print "\nPass\tMean\t\tStdDev\t\tMax\t\t(Bursting Columns)"

    for i in xrange(trainingPasses):

      experiment.runNetworkOnSequence(generatedSequences,
                                      labeledSequences,
                                      tmLearn=True,
                                      upLearn=None,
                                      verbosity=consoleVerbosity,
                                      progressInterval=_SHOW_PROGRESS_INTERVAL)

      if consoleVerbosity > 0:
        stats = experiment.getBurstingColumnsStats()
        print "{0}\t{1}\t{2}\t{3}".format(i, stats[0], stats[1], stats[2])

    print
    print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics())
    print
    # if plotVerbosity >= 2:
    #   plotNetworkState(experiment, plotVerbosity, trainingPasses,
    #                    phase="Training")
  experiment.tm.mmClearHistory()
  experiment.up.mmClearHistory()

  print "\nRunning test phase..."

  unionSDRS = []
  for i in xrange(sequenceCount):
    seq = generatedSequences[i + i*sequenceLength: i+1 + (i+1)*sequenceLength]
    lblSeq = labeledSequences[i + i*sequenceLength: i+1 + (i+1)*sequenceLength]
    experiment.runNetworkOnSequence(seq,
                                    lblSeq,
                                    tmLearn=False,
                                    upLearn=False,
                                    verbosity=consoleVerbosity,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)
    unionSDRS.append(experiment.up.getUnionSDR())

  distinctness = getDistinctness(unionSDRS)


  # runTestPhase(experiment, generatedSequences, sequenceCount, sequenceLength,
  #              testPresentations, perturbationChance)

  print "\nPass\tBursting Columns Mean\tStdDev\tMax"
  stats = experiment.getBurstingColumnsStats()
  print "{0}\t{1}\t{2}\t{3}".format(0, stats[0], stats[1], stats[2])
  if trainingPasses > 0 and stats[0] > 0:
    print "***WARNING! MEAN BURSTING COLUMNS IN TEST PHASE IS GREATER THAN 0***"

  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
  print
  # plotNetworkState(experiment, plotVerbosity, trainingPasses, phase="Testing")

  elapsed = int(time.time() - start)
  print "Total time: {0:2} seconds.".format(elapsed)

  ## Write Union SDR trace
  # metricName = "activeCells"
  # outputFileName = "unionSdrTrace_{0}learningPasses.csv".format(trainingPasses)
  # writeMetricTrace(experiment, metricName, outputDir, outputFileName)

  if plotVerbosity >= 1:
    raw_input("\nPress any key to exit...")



def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nRun Capacity experiment using "
                              "params in PARAMS_DIR (relative to this file) "
                              "and outputting results to OUTPUT_DIR.")
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

  absPath = os.path.join(os.path.dirname(__file__), args[0])
  with open(absPath) as paramsFile:
    params = yaml.safe_load(paramsFile)

  return options, args, params



if __name__ == "__main__":
  (_options, _args, _params) = _getArgs()
  run(_params, _args[0], _args[1], _options.plotVerbosity,
      _options.consoleVerbosity)
