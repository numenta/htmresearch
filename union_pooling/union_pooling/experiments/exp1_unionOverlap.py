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



def runTestPhase(experiment, consoleVerbosity):
  # TODO
  pass



# TODO: Consider moving this to union_pooler_experiment
def trainNetwork(experiment, sequences, sequencesLabels, repetitions,
                 verbosity, onlineLearn=False):
  print "Training network..."
  if onlineLearn:
    experiment.runNetworkOnSequence(sequences,
                                    sequencesLabels,
                                    tmLearn=True,
                                    upLearn=True,
                                    verbosity=verbosity,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)
  else:
    experiment.runNetworkOnSequence(sequences,
                                    sequencesLabels,
                                    tmLearn=True,
                                    upLearn=False,
                                    verbosity=verbosity,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)
    experiment.runNetworkOnSequence(sequences,
                                    sequencesLabels,
                                    tmLearn=False,
                                    upLearn=True,
                                    verbosity=verbosity,
                                    progressInterval=_SHOW_PROGRESS_INTERVAL)

  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
    experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
  print



def plotNetworkState(experiment, plotVerbosity, onlineLearning,
                     experimentPhase):
  # TODO
  pass



def main():
  start = time.time()
  onlineLearning = False
  consoleVerbosity = 0
  plotVerbosity = 0
  patternDimensionality = 1024
  patternCardinality = 20
  patternAlphabetSize = 100
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)

  numSequences = 4
  sequenceLength = 4
  numbers = sequenceMachine.generateNumbers(numSequences, sequenceLength)
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)
  sequenceLabels = [str(numbers[i + i*sequenceLength: i + (i+1)*sequenceLength])
                    for i in xrange(numSequences)]
  labeledSequence = []
  for label in sequenceLabels:
    for i in xrange(sequenceLength):
      labeledSequence.append(label)
    labeledSequence.append(None)

  tmParamOverrides = {}
  upParamOverrides = {}
  experiment = UnionPoolerExperiment(tmParamOverrides, upParamOverrides)

  trainingRepetitions = 1
  trainNetwork(experiment, generatedSequences, labeledSequence,
               trainingRepetitions, _VERBOSITY, onlineLearn=onlineLearning)

  runTestPhase(experiment, consoleVerbosity)
  plotNetworkState(experiment, plotVerbosity, onlineLearning, "Testing")

  elapsed = int(time.time() - start)
  print "Total time: {0:2} seconds.".format(elapsed)

  # TODO output Union SDR sequence
  # Write results to output file
  # writeOutput(outputDir, runner, numElems, numWorlds, elapsed)
  # if plotVerbosity >= 1:
  #   raw_input("Press any key to exit...")



if __name__ == "__main__":
  main()
