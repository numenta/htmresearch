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

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine

from union_pooling.experiments.union_pooler_experiment import (
    UnionPoolerExperiment)

"""
Union Pooling with and without Temporal Memory training
Compute overlap between Union SDR representations in two conditions over time
"""


_SHOW_PROGRESS_INTERVAL = 100
_VERBOSITY = 0


def trainNetwork(experiment, sequences, repetitions, verbosity):
  print "Training network..."
  experiment.feedLayers(sequences, tmLearn=True, tpLearn=True,
                        verbosity=verbosity,
                        progressInterval=_SHOW_PROGRESS_INTERVAL)
  print
  print MonitorMixinBase.mmPrettyPrintMetrics(runner.tp.mmGetDefaultMetrics() +
                                              runner.tm.mmGetDefaultMetrics())
  print



def main():
  tmLearning = False
  patternDimensionality = 1024
  patternCardinality = 20
  patternAlphabetSize = 100
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)

  numSequences = 10
  sequenceLength = 10
  numbers = sequenceMachine.generateNumbers(numSequences, sequenceLength)
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)
  # print sequenceMachine.prettyPrintSequence(generatedSequences)

  temporalMemoryParamOverrides = {}
  unionPoolerParamOverrides = {}
  experiment = UnionPoolerExperiment(temporalMemoryParamOverrides,
                                     unionPoolerParamOverrides)

  trainingRepetitions = 1
  trainNetwork(experiment, generatedSequences, trainingRepetitions, _VERBOSITY)
  # TODO run training

  # TODO run testing

  # TODO output Union SDR sequence



if __name__ == "__main__":
  main()
