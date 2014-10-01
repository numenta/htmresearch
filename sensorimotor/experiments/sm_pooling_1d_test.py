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

from sensorimotor.sensorimotor_experiment_runner import (
  SensorimotorExperimentRunner
)


print """
This program runs sensorimotor inference and pooling with several static worlds.
"""

############################################################
# Initialize the universe, worlds, and agents
nElements = 20
wEncoders = 21
universe = OneDUniverse(debugSensor=False,
                        debugMotor=False,
                        nSensor=512, wSensor=wEncoders,
                        nMotor=wEncoders*7, wMotor=wEncoders)

# Initialize a bunch of worlds, each with at most 8 elements
agents = [
  RandomOneDAgent(OneDWorld(universe, range(8)), 4,
                         possibleMotorValues=(-2, -1, 1, 2), seed=23),
  RandomOneDAgent(OneDWorld(universe, range(8-1, -1, -1)), 4,
                         possibleMotorValues=(-2, -1, 1, 2), seed=42),

  RandomOneDAgent(OneDWorld(universe, range(0,16,2)), 4,
                         possibleMotorValues=(-2, -1, 1, 2), seed=10),
  RandomOneDAgent(OneDWorld(universe, range(0,15,3)), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),
  RandomOneDAgent(OneDWorld(universe, range(0,20,4)), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),

  RandomOneDAgent(OneDWorld(universe, [0, 8, 3, 1, 6]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),
  RandomOneDAgent(OneDWorld(universe, [6, 1, 3, 8, 0]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),

  RandomOneDAgent(OneDWorld(universe, [3, 7, 4, 2, 5]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),
  RandomOneDAgent(OneDWorld(universe, [5, 2, 4, 7, 3]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),

  RandomOneDAgent(OneDWorld(universe, [8, 12, 9, 7, 10]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),
  RandomOneDAgent(OneDWorld(universe, [10, 7, 9, 12, 8]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),

  RandomOneDAgent(OneDWorld(universe, [15, 19, 16, 14, 17]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),
  RandomOneDAgent(OneDWorld(universe, [17, 14, 16, 19, 15]), 2,
                         possibleMotorValues=(-2, -1, 1, 2), seed=5),

  ]

l3NumColumns = 512
l3NumActiveColumnsPerInhArea = 20

############################################################
# Initialize the experiment runner with relevant parameters
print "Initializing experiment runner"
smer = SensorimotorExperimentRunner(
          tmOverrides={
              "columnDimensions": [universe.nSensor],
              "minThreshold": wEncoders*2,
              "maxNewSynapseCount": wEncoders*2,
              "activationThreshold": wEncoders*2
            },
          tpOverrides={
              "columnDimensions": [l3NumColumns],
              "numActiveColumnsPerInhArea": l3NumActiveColumnsPerInhArea,
            }
)

############################################################
# Temporal memory training

print "Training TemporalMemory on sequences"
sequences = smer.generateSequences(500, agents, verbosity=0)
smer.feedLayers(sequences, tmLearn=True, verbosity=0)


# Check if TM learning went ok

print "Testing TemporalMemory on novel sequences"
testSequenceLength=100
sequences = smer.generateSequences(testSequenceLength, agents, verbosity=0)
stats = smer.feedLayers(sequences, tmLearn=False, verbosity=0)

print smer.tm.mmPrettyPrintMetrics(smer.tm.mmGetDefaultMetrics())

unpredictedActiveColumnsMetric = smer.tm.getMetricFromTrace(
  smer.tm.getTraceUnpredictedActiveColumns())
predictedActiveColumnsMetric = smer.tm.getMetricFromTrace(
  smer.tm.getTracePredictedActiveColumns())
if (unpredictedActiveColumnsMetric.sum == 0) and (
        predictedActiveColumnsMetric.sum ==
            universe.wSensor*(testSequenceLength-1)*len(agents)):
  print "TM training successful!!"
else:
  print "TM training unsuccessful"


############################################################
# Temporal pooler training

print "Training TemporalPooler on sequences"
sequences = smer.generateSequences(100, agents, verbosity=0)
smer.feedLayers(sequences, tmLearn=False, tpLearn=True, verbosity=0)

print "Testing TemporalPooler on sequences"
sequences = smer.generateSequences(10, agents, verbosity=1)
smer.feedLayers(sequences, tmLearn=False, tpLearn=False, verbosity=2)

