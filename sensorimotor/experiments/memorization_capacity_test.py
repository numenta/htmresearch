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

The output of this program is a set of graphs showing the relationship between
these variables.
--------------------------------------------------------------------------------
"""



# Set constants
numWorldsRange = range(2, 10, 3)
numElementsRange = range(2, 10, 3)

VERBOSITY = 0
SHOW_PROGRESS_INTERVAL = 10



# Set up the experiment
print "Setting up the experiment..."
universe = OneDUniverse(nSensor=512, wSensor=20,
                        nMotor=512, wMotor=20)
wTotal = universe.wSensor + universe.wMotor
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
print "Done setting up the experiment.\n"



# Run the experiment
for numWorlds in numWorldsRange:

  for numElements in numElementsRange:
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
      print runner.tp.prettyPrintDataStabilityConfusion()
      print "TP Distinctness:"
      print
      print runner.tp.prettyPrintDataDistinctnessConfusion()
      print

    print runner.tp.prettyPrintMetrics(runner.tp.getDefaultMetrics())
    print
