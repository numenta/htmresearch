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

# A simple example of sensorimotor inference that shows how to use the
# various classes. We feed in 1D sequences from 1D worlds with overlapping
# patterns.  We then test to ensure the number of predicted columns matches
# the actual columns.


import numpy

from nupic.bindings.math import GetNTAReal
from nupic.research.temporal_memory_inspect_mixin import  (
  TemporalMemoryInspectMixin)

from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent
from sensorimotor.general_temporal_memory import (
            GeneralTemporalMemory
)


"""

This program forms the simplest test of sensorimotor sequence inference
with 1D patterns. We present a sequence from a single 1D pattern. The
TM is initialized with multiple cells per column but should form a
first order representation of this sequence.

"""

realDType = GetNTAReal()

# Mixin class for TM statistics
class TMI(TemporalMemoryInspectMixin,GeneralTemporalMemory): pass


def feedTM(tm, length, agents,
           verbosity=0, learn=True):
  """Feed the given sequence to the TM instance."""
  tm.mmClearHistory()
  for agent in agents:
    tm.reset()
    if verbosity > 0:
      print "\nGenerating sequence for world:", agent.world.toString()
    sensorSequence, motorSequence, sensorimotorSequence = (
      agent.generateSensorimotorSequence(length,verbosity=verbosity)
    )
    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      sensorimotorPattern = sensorimotorSequence[i]
      tm.compute(sensorPattern,
                activeExternalCells=sensorimotorPattern,
                formInternalConnections=False,
                learn=learn)

  if verbosity >= 2:
    print tm.prettyPrintHistory(verbosity=verbosity)
    print

  if learn and verbosity >= 3:
    print tm.prettyPrintConnections()

  return tm.getStatistics()


# Initialize the universe, worlds, and agents
nElements = 10
wEncoders = 7
universe = OneDUniverse(debugSensor=True,
                        debugMotor=True,
                        nSensor=nElements*wEncoders, wSensor=wEncoders,
                        nMotor=wEncoders*7, wMotor=wEncoders)
agents = [
  RandomOneDAgent(OneDWorld(universe, range(nElements)), 4,
                         possibleMotorValues=(-1,1), seed=23),
  RandomOneDAgent(OneDWorld(universe, range(nElements-1, -1, -1)), 4,
                         possibleMotorValues=(-1,1), seed=42),
  RandomOneDAgent(OneDWorld(universe, range(0,nElements,2)), 4,
                         possibleMotorValues=(-1,1), seed=10),
  RandomOneDAgent(OneDWorld(universe, range(0,nElements,3)), 2,
                         possibleMotorValues=(-1,1), seed=5),
  ]

# The TM parameters
DEFAULT_TM_PARAMS = {
  "columnDimensions": [nElements*wEncoders],
  "cellsPerColumn": 8,
  "initialPermanence": 0.5,
  "connectedPermanence": 0.6,
  "minThreshold": wEncoders*2,
  "maxNewSynapseCount": wEncoders*2,
  "permanenceIncrement": 0.1,
  "permanenceDecrement": 0.02,
  "activationThreshold": wEncoders*2
}

tm = TMI(**dict(DEFAULT_TM_PARAMS))

# Train and test
print "Training TM on sequences"
feedTM(tm, length=700, agents=agents, verbosity=0, learn=True)

print "Testing TM on sequences"
stats = feedTM(tm, length=200, agents=agents, verbosity=2,
               learn=False)

# Debug
print "cells for column 0",tm.connections.cellsForColumn(0)
print "cells for column 1",tm.connections.cellsForColumn(1)
print "cells for column 20",tm.connections.cellsForColumn(20)

print "Unpredicted columns: min, max, sum, average, stdev",stats[4]
print "Predicted columns: min, max, sum, average, stdev",stats[2]
print "Predicted inactive cells:",stats[1]

if (stats[4][2]== 0) and (stats[2][2] == universe.wSensor*199*len(agents)):
  print "Test successful!!"
else:
  print "Test unsuccessful"
