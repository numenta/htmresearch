#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
from itertools import product
import sys

from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent
from sensorimotor.exhaustive_one_d_agent import ExhaustiveOneDAgent

from sensorimotor.sensorimotor_experiment_runner import (
  SensorimotorExperimentRunner
)



print """
--------------------------------------------------------------------------------
This program tests the memorization capacity of L4+L3.

The independent variables (that we change) are:
    - # of distinct worlds (images)
    - # of unique elements (fixation points)

The dependent variables (that we monitor) are:
    - temporal pooler stability
    - temporal pooler distinctness

Each world will be composed of unique elements that are not shared between
worlds, to test the raw memorization capacity without generalization.

The output of this program is a data sheet (CSV) showing the relationship
between these variables.
--------------------------------------------------------------------------------
"""



# Reference constants
OUTPUT_HEADERS = [
  "num_worlds",
  "num_elements",
  "tp_stability_min",
  "tp_stability_max",
  "tp_stability_sum",
  "tp_stability_mean",
  "tp_stability_stddev",
  "tp_distinctness_min",
  "tp_distinctness_max",
  "tp_distinctness_sum",
  "tp_distinctness_mean",
  "tp_distinctness_stddev"
]
OUTPUT_FILE = ("memorization_capacity_test_results.csv" if len(sys.argv) <= 1
                                                        else sys.argv[1])



# Set constants
numWorldsRange = range(2, 100, 5)
numElementsRange = range(2, 100, 5)

VERBOSITY = 0
PLOT = 0
SHOW_PROGRESS_INTERVAL = 10



# Initialize experiment
universe = OneDUniverse(nSensor=512, wSensor=20,
                        nMotor=512, wMotor=20)
wTotal = universe.wSensor + universe.wMotor



# Run the experiment
with open(OUTPUT_FILE, 'wb') as outFile:
  csvWriter = csv.writer(outFile)
  headerWritten = False

  combinations = sorted(product(numWorldsRange, numElementsRange),
    key=lambda x: x[0]*x[1])  # sorted by total # of elements

  for numWorlds, numElements in combinations:
    print "Setting up a new experiment..."
    runner = SensorimotorExperimentRunner(
      tmOverrides={
        "columnDimensions": [universe.nSensor],
        "minThreshold": wTotal,
        "activationThreshold": wTotal,
        "maxNewSynapseCount": wTotal
      },
      tpOverrides={
        "columnDimensions": [universe.nSensor],
        "numActiveColumnsPerInhArea": universe.wSensor
      }
    )
    print "Done setting up experiment.\n"

    exhaustiveAgents = []
    randomAgents = []
    completeSequenceLength = numElements ** 2

    for world in xrange(numWorlds):
      elements = range(world * numElements, world * numElements + numElements)

      exhaustiveAgents.append(
        ExhaustiveOneDAgent(OneDWorld(universe, elements), 0))

      possibleMotorValues = range(-numElements, numElements+1)
      possibleMotorValues.remove(0)
      randomAgents.append(
        RandomOneDAgent(OneDWorld(universe, elements), numElements / 2,
                        possibleMotorValues=possibleMotorValues))


    print "Training (worlds: {0}, elements: {1})...".format(numWorlds,
                                                            numElements)
    sequences = runner.generateSequences(completeSequenceLength * 2,
                                         exhaustiveAgents,
                                         verbosity=VERBOSITY)
    runner.feedLayers(sequences, tmLearn=True, tpLearn=True,
                      verbosity=VERBOSITY,
                      showProgressInterval=SHOW_PROGRESS_INTERVAL)
    print "Done training.\n"

    print MonitorMixinBase.mmPrettyPrintMetrics(
      runner.tp.mmGetDefaultMetrics() + runner.tm.mmGetDefaultMetrics())
    print

    if PLOT >= 1:
      runner.tp.mmGetPlotConnectionsPerColumn(
        title="worlds: {0}, elements: {1}".format(numWorlds, numElements))


    print "Testing (worlds: {0}, elements: {1})...".format(numWorlds,
                                                           numElements)
    sequences = runner.generateSequences(completeSequenceLength,
                                         randomAgents,
                                         verbosity=VERBOSITY)
    runner.feedLayers(sequences, tmLearn=False, tpLearn=False,
                      verbosity=VERBOSITY,
                      showProgressInterval=SHOW_PROGRESS_INTERVAL)
    print "Done testing.\n"

    if VERBOSITY >= 2:
      print "TP Stability:"
      print
      print runner.tp.mmPrettyPrintDataStabilityConfusion()
      print "TP Distinctness:"
      print
      print runner.tp.mmPrettyPrintDataDistinctnessConfusion()
      print

    print MonitorMixinBase.mmPrettyPrintMetrics(
      runner.tp.mmGetDefaultMetrics() + runner.tm.mmGetDefaultMetrics())
    print

    header = ["# worlds", "# elements"] if not headerWritten else None

    row = [numWorlds, numElements]

    for metric in (runner.tp.mmGetDefaultMetrics() +
                   runner.tm.mmGetDefaultMetrics()):
      row += [metric.min, metric.max, metric.sum,
              metric.mean,  metric.standardDeviation]

      if header:
        header += ["{0} ({1})".format(metric.title, x) for x in [
                   "min", "max", "sum", "mean", "stddev"]]

    if header:
      csvWriter.writerow(header)
      headerWritten = True
    csvWriter.writerow(row)
    outFile.flush()

  raw_input("Press any key to exit...")
