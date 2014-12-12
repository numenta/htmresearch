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
"""
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
"""

import csv
import os
import sys

from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent
from sensorimotor.exhaustive_one_d_agent import ExhaustiveOneDAgent

from sensorimotor.sensorimotor_experiment_runner import (
  SensorimotorExperimentRunner
)



# Constants
DEFAULTS = {
  "n": 512,
  "w": 20,
  "tmParams": {
    "columnDimensions": [512],
    "minThreshold": 40,
    "activationThreshold": 40,
    "maxNewSynapseCount": 40
  },
  "tpParams": {
    "columnDimensions": [512],
    "numActiveColumnsPerInhArea": 20,
    "potentialPct": 0.9,
    "initConnectedPct": 0.5
  }
}
VERBOSITY = 0
PLOT = 0
SHOW_PROGRESS_INTERVAL = 10



def runExperiment(numWorlds, numElements,
                  n, w,
                  tmParams, tpParams,
                  outputDir):
  # Initialize output
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  csvFilePath = os.path.join(outputDir, "{0}x{1}.csv".format(numWorlds,
                                                             numElements))

  # Initialize experiment
  universe = OneDUniverse(nSensor=n, wSensor=w,
                          nMotor=n, wMotor=w)
  wTotal = universe.wSensor + universe.wMotor

  # Run the experiment
  with open(csvFilePath, 'wb') as csvFile:
    csvWriter = csv.writer(csvFile)
    headerWritten = False

    print "Setting up a new experiment..."
    runner = SensorimotorExperimentRunner(tmOverrides=tmParams,
                                          tpOverrides=tpParams)
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
    csvFile.flush()

  if PLOT >= 1:
    raw_input("Press any key to exit...")



if __name__ == "__main__":
  if len(sys.argv) < 4:
    print "Usage: ./capacity.py NUM_WORLDS NUM_ELEMENTS OUTPUT_DIR"
    sys.exit()

  runExperiment(int(sys.argv[1]), int(sys.argv[2]),
                DEFAULTS["n"],
                DEFAULTS["w"],
                DEFAULTS["tmParams"],
                DEFAULTS["tpParams"],
                sys.argv[3])