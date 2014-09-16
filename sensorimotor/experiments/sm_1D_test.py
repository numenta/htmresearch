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
# patterns.  We then test to ensure the number of

# The input to CLA contains both sensory inputs ("A","B","C","D") 
# and motor commands that encodes the eye velocity vector e.g. (1,-1,2,-2,...)
# CLA will be trained on copies of valid transitions


from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent

from nupic.research.temporal_memory_inspect_mixin import  (
  TemporalMemoryInspectMixin)

from sensorimotor.learn_on_one_cell_temporal_memory import (
  LearnOnOneCellTemporalMemory)

"""

This program forms the simplest test of sensorimotor sequence inference
with 1D patterns. We present a sequence from a single 1D pattern. The
TM is initialized with multiple cells per column but should form a
first order representation of this sequence.

"""

# Mixin class for TM statistics
class TMInspect(TemporalMemoryInspectMixin,LearnOnOneCellTemporalMemory): pass


def feedTM(tm, length, agents,
           verbosity=0, learn=True):
  """Feed the given sequence to the TM instance."""
  tm.clearHistory()
  for agent in agents:
    tm.reset()
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
universe = OneDUniverse(debugSensor=True,
                        debugMotor=True,
                        nSensor=nElements*7, wSensor=7,
                        nMotor=49, wMotor=7)
agents = [
  RandomOneDAgent(OneDWorld(universe, range(nElements), 2),
                         possibleMotorValues=(-1,1)),
  RandomOneDAgent(OneDWorld(universe, range(nElements-1, -1, -1), 2),
                         possibleMotorValues=(-1,1)),
  RandomOneDAgent(OneDWorld(universe, range(0,nElements,2), 2),
                         possibleMotorValues=(-1,1)),
  ]

# The TM parameters
DEFAULT_TM_PARAMS = {
  "columnDimensions": [nElements*7],
  "cellsPerColumn": 8,
  "initialPermanence": 0.5,
  "connectedPermanence": 0.6,
  "minThreshold": 10,
  "maxNewSynapseCount": 50,
  "permanenceIncrement": 0.1,
  "permanenceDecrement": 0.02,
  "activationThreshold": 10
}

tm = TMInspect(**dict(DEFAULT_TM_PARAMS))

# Train and test
print "Training TM on sequences"
feedTM(tm, length=100, agents=agents, verbosity=0, learn=True)

print "Testing TM on sequences"
stats = feedTM(tm, length=70, agents=agents, verbosity=2,
               learn=False)

print "Unpredicted columns: min, max, sum, average, stdev",stats[4]
print "Predicted columns: min, max, sum, average, stdev",stats[2]

if (stats[4][2]== 0) and (stats[2][2] == universe.wSensor*69*len(agents)):
  print "Test successful!!"