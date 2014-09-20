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

from sensorimotor.TP_New import SPTP as TP_New


"""

This program forms the simplest test of sensorimotor sequence inference
with 1D patterns. We present a sequence from a single 1D pattern. The
TM is initialized with multiple cells per column but should form a
first order representation of this sequence.

"""

realDType = GetNTAReal()

# Mixin class for TM statistics
class TMI(TemporalMemoryInspectMixin,GeneralTemporalMemory): pass

def formatRow(x, formatString = "%d", rowSize = 700):
  """
  Utility routine for pretty printing large vectors
  """
  s = ''
  for c,v in enumerate(x):
    if c > 0 and c % 7 == 0:
      s += ' '
    if c > 0 and c % rowSize == 0:
      s += '\n'
    s += formatString % v
  s += ' '
  return s



def formatInputForTP(tm):
  """
  Given an instance of the TM, format the information we need to send to the
  TP.
  """
  # all currently active cells in layer 4
  tpInputVector = numpy.zeros(tm.connections.numberOfCells()).astype(realDType)
  tpInputVector[list(tm.activeCells)] = 1

  # bursting columns in layer 4
  burstingColumns = numpy.zeros(
    tm.connections.numberOfColumns()).astype(realDType)
  burstingColumns[list(tm.unpredictedActiveColumnsList[-1])] = 1

  # correctly predicted cells in layer 4
  correctlyPredictedCells = numpy.zeros(
    tm.connections.numberOfCells()).astype(realDType)
  correctlyPredictedCells[list(tm.predictedActiveCellsList[-1])] = 1

  return (tpInputVector, burstingColumns, correctlyPredictedCells)



def feedTM(tm, length, agents,
           verbosity=0, learn=True):
  """Feed the given sequence to the TM instance."""
  tm.clearHistory()
  for agent in agents:
    tm.reset()
    if verbosity > 0:
      print "\nGenerating sequence for world:",str(agent._world)
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
      tpInputVector, burstingColumns, correctlyPredictedCells = (
          formatInputForTP(tm))

  if verbosity >= 2:
    print tm.prettyPrintHistory(verbosity=verbosity)
    print

  if learn and verbosity >= 3:
    print tm.prettyPrintConnections()

  return tm.getStatistics()


def feedTMTP(tm, tp, length, agents,
           verbosity=0, learn=True):
  """Feed the given sequence to the TM instance and the TP instance."""
  tm.clearHistory()
  for agent in agents:
    tm.reset()
    if verbosity > 0:
      print "\nGenerating sequence for world:",str(agent._world)
    sensorSequence, motorSequence, sensorimotorSequence = (
      agent.generateSensorimotorSequence(length,verbosity=verbosity)
    )
    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      sensorimotorPattern = sensorimotorSequence[i]

      # Feed the TM
      tm.compute(sensorPattern,
                activeExternalCells=sensorimotorPattern,
                formInternalConnections=False,
                learn=learn)

      # Feed the TP
      tpInputVector, burstingColumns, correctlyPredictedCells = (
          formatInputForTP(tm))
      activeArray = numpy.zeros(tp.getNumColumns())

      tp.compute(tpInputVector, learn=True, activeArray=activeArray,
                   burstingColumns=burstingColumns,
                   predictedCells=correctlyPredictedCells)

      if verbosity >= 2:
        print "L3 Active Cells \n",formatRow(activeArray.nonzero()[0],
                                             formatString="%4d")


  if verbosity >= 2:
    print tm.prettyPrintHistory(verbosity=verbosity)
    print

  if learn and verbosity >= 3:
    print tm.prettyPrintConnections()

  return tm.getStatistics()


# Initialize the universe, worlds, and agents
nElements = 5
wEncoders = 7
universe = OneDUniverse(debugSensor=True,
                        debugMotor=True,
                        nSensor=nElements*wEncoders, wSensor=wEncoders,
                        nMotor=wEncoders*7, wMotor=wEncoders)
agents = [
  RandomOneDAgent(OneDWorld(universe, range(nElements), 4),
                         possibleMotorValues=(-1,1), seed=23),
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
feedTM(tm, length=40, agents=agents, verbosity=2, learn=True)

print "Testing TM on sequences"
stats = feedTM(tm, length=10, agents=agents, verbosity=2,
               learn=False)

# Check if TM learning went ok
print "Unpredicted columns: min, max, sum, average, stdev",stats[4]
print "Predicted columns: min, max, sum, average, stdev",stats[2]
print "Predicted inactive cells:",stats[1]

if (stats[4][2]== 0) and (stats[2][2] == universe.wSensor*199*len(agents)):
  print "Test successful!!"
else:
  print "Test unsuccessful"

print "Training TP on sequences"

l3NumColumns = 512
l3NumActiveColumnsPerInhArea = 20
tp = TP_New(
      inputDimensions  = [tm.connections.numberOfCells()],
      columnDimensions = [l3NumColumns],
      potentialRadius  = tm.connections.numberOfCells(),
      globalInhibition = True,
      numActiveColumnsPerInhArea=l3NumActiveColumnsPerInhArea,
      synPermInactiveDec=0,
      synPermActiveInc=0.001,
      synPredictedInc = 0.5,
      maxBoost=1.0,
      seed=4,
      potentialPct=0.9,
      stimulusThreshold = 2,
      useBurstingRule = False,
      minPctActiveDutyCycle = 0.1,
      synPermConnected = 0.3,
      initConnectedPct=0.2,
      spVerbosity=0
    )

print "Testing TM on sequences"
stats = feedTMTP(tm, tp, length=10, agents=agents, verbosity=2)
