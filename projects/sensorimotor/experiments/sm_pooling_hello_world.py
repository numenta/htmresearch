#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from nupic.algorithms.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent

from sensorimotor.sensorimotor_experiment_runner import (
  SensorimotorExperimentRunner
)


print """
This program forms the simplest test of sensorimotor sequence inference with
pooling. We present sequences from two small 1D worlds. We then test to ensure
the number of predicted columns matches the actual columns.
"""

############################################################
# Initialize the universe, worlds, and agents
nElements = 5
wEncoders = 7
universe = OneDUniverse(debugSensor=True,
                        debugMotor=True,
                        nSensor=nElements*wEncoders, wSensor=wEncoders,
                        nMotor=wEncoders*7, wMotor=wEncoders)
agents = [
  RandomOneDAgent(OneDWorld(universe, range(nElements)), 4,
                  possibleMotorValues=(-1,1), seed=23),
  RandomOneDAgent(OneDWorld(universe, list(reversed(range(nElements)))), 4,
                  possibleMotorValues=(-1,1), seed=23),
]

l3NumColumns = 512
l3NumActiveColumnsPerInhArea = 20

############################################################
# Initialize the experiment runner with relevant parameters
smer = SensorimotorExperimentRunner(
          tmOverrides={
              "columnDimensions": [nElements*wEncoders],
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
sequences = smer.generateSequences(40, agents, verbosity=1)
smer.feedLayers(sequences, tmLearn=True, verbosity=1)


# Check if TM learning went ok

print "Testing TemporalMemory on novel sequences"
testSequenceLength=10
sequences = smer.generateSequences(testSequenceLength, agents, verbosity=1)
smer.feedLayers(sequences, tmLearn=False, verbosity=2)

print smer.tm.mmPrettyPrintMetrics(smer.tm.mmGetDefaultMetrics())

unpredictedActiveColumnsMetric = smer.tm.mmGetMetricFromTrace(
  smer.tm.mmGetTraceUnpredictedActiveColumns())
predictedActiveColumnsMetric = smer.tm.mmGetMetricFromTrace(
  smer.tm.mmGetTracePredictedActiveColumns())
if (unpredictedActiveColumnsMetric.sum == 0) and (
      predictedActiveColumnsMetric.sum ==
            universe.wSensor*(testSequenceLength-1)*len(agents)):
  print "TM training successful!!"
else:
  print "TM training unsuccessful"


############################################################
# Temporal pooler training

print "Training TemporalPooler on sequences"
sequences = smer.generateSequences(10, agents, verbosity=1)
smer.feedLayers(sequences, tmLearn=False, tpLearn=True, verbosity=2)

print MonitorMixinBase.mmPrettyPrintMetrics(smer.tm.mmGetDefaultMetrics() +
                                            smer.tp.mmGetDefaultMetrics())
