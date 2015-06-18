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

"""
Experiment 2a
Explore the limits of the distinctness of Union Pooler's union SDR.

Data: Sequences of unique patterns.
Train Phase: Train network on some number of sequences having moderate length.
Test phase: Each sequence is shown once and the Union SDRs at the end are
recorded. Compute the distinctness between the final SDRs of each sequence.
"""

from optparse import OptionParser
import os
import sys
import time
import yaml

import numpy

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from union_pooling.experiments.union_pooler_experiment import (
    UnionPoolerExperiment)



_SHOW_PROGRESS_INTERVAL = 5000



def getDistinctness(sdrList):
  """
  Gets the distinctness of a list of unionSdrs, that is, a measure of the
  overlap between the SDRs.
  :param sdrList: a list of SDRs
  :return: The average, standard deviation, and max distinctness of the set
  """
  distinctness = []
  for i in xrange(len(sdrList)):
    for j in xrange(i):
      intersection = numpy.intersect1d(sdrList[i], sdrList[j])
      distinctness.append(len(intersection))

  return numpy.mean(distinctness), numpy.std(distinctness), max(distinctness)



def generateSequences(patternDimensionality, patternCardinality, sequenceLength,
                      sequenceCount):
  # Generate a sequence list and an associated labeled list
  # (both containing a set of sequences separated by None)
  print "Generating sequences..."
  patternAlphabetSize = sequenceLength * sequenceCount
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)
  numbers = sequenceMachine.generateNumbers(sequenceCount, sequenceLength)
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)
  sequenceLabels = [
    str(numbers[i + i * sequenceLength: i + (i + 1) * sequenceLength])
    for i in xrange(sequenceCount)]
  labeledSequences = []
  for label in sequenceLabels:
    for _ in xrange(sequenceLength):
      labeledSequences.append(label)
    labeledSequences.append(None)

  return generatedSequences, labeledSequences


def runTestPhase(experiment, inputSequences, seqLabels, sequenceCount,
                 sequenceLength, consoleVerbosity):
  print "\nRunning Test Phase..."
  unionSdrs = []
  for i in xrange(sequenceCount):

    # Extract next sequence
    begin = i + i * sequenceLength
    end = i + 1 + (i + 1) * sequenceLength
    seq = inputSequences[begin: end]
    lblSeq = seqLabels[begin: end]

    # Present sequence (minus reset element)
    experiment.runNetworkOnSequences(seq[:-1],
                                    lblSeq[:-1],
                                    tmLearn=False,
                                    upLearn=False,
                                    verbosity=consoleVerbosity,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)

    # Record Union SDR at end of sequence
    seqUnionSdr = experiment.up.getUnionSDR()
    unionSdrs.append(numpy.sort(seqUnionSdr))

    # Run reset element
    experiment.runNetworkOnSequences(seq[-1:],
                                    lblSeq[-1:],
                                    tmLearn=False,
                                    upLearn=False,
                                    verbosity=consoleVerbosity,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)

  return unionSdrs


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
  start = time.time()
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

  # Generate input data
  inputSequences, seqLabels = generateSequences(patternDimensionality,
                                                patternCardinality,
                                                sequenceLength,
                                                sequenceCount)

  print "\nCreating Network..."
  experiment = UnionPoolerExperiment(tmParamOverrides, upParamOverrides)

  # Train the Temporal Memory on the generated sequences
  print "\nTraining Temporal Memory..."
  for i in xrange(trainingPasses):
    print "\nTraining pass {0} ...\n".format(i)
    experiment.runNetworkOnSequences(inputSequences,
                                     seqLabels,
                                     tmLearn=True,
                                     upLearn=None,
                                     verbosity=consoleVerbosity,
                                     progressInterval=_SHOW_PROGRESS_INTERVAL)

    if consoleVerbosity > 0:
      stats = experiment.getBurstingColumnsStats()
      print "\nPass\tMean\t\tStdDev\t\tMax\t\t(Bursting Columns)"
      print "{0}\t{1}\t{2}\t{3}".format(i, stats[0], stats[1], stats[2])

  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
    experiment.tm.mmGetDefaultMetrics())
  print
  experiment.tm.mmClearHistory()

  # Run test phase recording Union SDRs
  unionSdrs = runTestPhase(experiment, inputSequences, seqLabels, sequenceCount,
                           sequenceLength, consoleVerbosity)

  # Output distinctness metric
  print "\nSequences\tDistinctness Ave\tStdDev\tMax"
  ave, stdDev, maxDist = getDistinctness(unionSdrs)
  print "{0}\t{1}\t{2}\t{3}".format(sequenceCount, ave, stdDev, maxDist)

  # Check bursting columns metric during test phase
  print "\nSequences\tBursting Columns Mean\tStdDev\tMax"
  stats = experiment.getBurstingColumnsStats()
  print "{0}\t{1}\t{2}\t{3}".format(sequenceCount, stats[0], stats[1], stats[2])
  if trainingPasses > 0 and stats[0] > 0:
    print "***Warning! Mean bursing columns > 0 in test phase***"

  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
      experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
  print
  print "Total time: {0:2} seconds.".format(int(time.time() - start))



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
