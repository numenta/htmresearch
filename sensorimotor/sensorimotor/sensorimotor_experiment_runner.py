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

import numpy

from nupic.bindings.math import GetNTAReal

from sensorimotor.general_temporal_memory import (
            InspectGeneralTemporalMemory
)
from sensorimotor.temporal_pooler import TemporalPooler

"""

Experiment runner class for running networks with layer 4 and layer 3. The
client is responsible for setting up universes, agents, and worlds.  This
class just sets up and runs the HTM learning algorithms.

"""

realDType = GetNTAReal()


class SensorimotorExperimentRunner(object):
  DEFAULT_TM_PARAMS = {
    # These should be decent for most experiments, shouldn't need to override
    # these too often. Might want to increase cellsPerColumn for capacity
    # experiments.
    "cellsPerColumn": 8,
    "initialPermanence": 0.5,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,

    # We will force client to override these
    "columnDimensions": "Sorry",
    "minThreshold": "Sorry",
    "maxNewSynapseCount": "Sorry",
    "activationThreshold": "Sorry",
  }

  DEFAULT_TP_PARAMS = {
    # Need to check these parameters and find stable values that will be
    # consistent across most experiments.
    "synPermInactiveDec": 0,   # TODO: Check we can use class default here.
    "synPermActiveInc": 0.001, # TODO: Check we can use class default here.
    "synPredictedInc": 0.5,  # TODO: Why so high??
    "initConnectedPct": 0.2,  # TODO: need to check impact of this for pooling

    # We will force client to override these
    "numActiveColumnsPerInhArea": "Sorry",
  }

  def __init__(self, tmOverrides=None, tpOverrides=None, seed=42, verbosity=0):
    # Initialize Layer 4 temporal memory
    params = dict(self.DEFAULT_TM_PARAMS)
    params.update(tmOverrides or {})
    self._checkParams(params)
    self.tm = InspectGeneralTemporalMemory(**params)

    # Initialize Layer 3 temporal pooler
    params = dict(self.DEFAULT_TP_PARAMS)
    params["inputDimensions"] = [self.tm.connections.numberOfCells()]
    params["potentialRadius"] = self.tm.connections.numberOfCells()
    params.update(tpOverrides or {})
    self._checkParams(params)
    self.tp = TemporalPooler(**params)


  def _checkParams(self, params):
    for k,v in params.iteritems():
      if v == "Sorry":
        raise RuntimeError("Param "+k+" must be specified")


  def feedLayers(self, sequences, tmLearn, tpLearn=None, verbosity=0):
    """
    Feed the given set of sequences to the HTM algorithms.

    @param tmLearn:   (bool)      Either False, or True
    @param tpLearn:   (None,bool) Either None, False, or True. If None,
                                  temporal pooler will be skipped.
    """
    self.tm.clearHistory()

    for s,seq in enumerate(sequences):
      self.tm.reset()

      for i in xrange(len(seq[0])):
        sensorPattern = seq[0][i]
        sensorimotorPattern = seq[2][i]

        # Feed the TM
        self.tm.compute(sensorPattern,
                  activeExternalCells=sensorimotorPattern,
                  formInternalConnections=False,
                  learn=tmLearn)

        # If requested, feed the TP
        if tpLearn is not None:
          tpInputVector, burstingColumns, correctlyPredictedCells = (
              self.formatInputForTP())
          activeArray = numpy.zeros(self.tp.getNumColumns())

          self.tp.compute(tpInputVector, learn=True, activeArray=activeArray,
                       burstingColumns=burstingColumns,
                       predictedCells=correctlyPredictedCells)

          if verbosity >= 2:
            print "L3 Active Cells \n",self.formatRow(activeArray.nonzero()[0],
                                                 formatString="%4d")


    if verbosity >= 2:
      print self.tm.prettyPrintHistory(verbosity=verbosity)
      print

    return self.tm.getStatistics()


  def generateSequences(self, length, agents, verbosity=0):
    """
    Generate sequences of the given length for each of the agents.

    Returns a list containing one tuple for each agent. Each tuple contains
    (sensorSequence, motorSequence, and sensorimotorSequence) as returned by
    the agent's generateSensorimotorSequence() method.

    """
    sequences = []
    for agent in agents:
      if verbosity > 0:
        print "\nGenerating sequence for world:",str(agent.world)
      sequences.append(
          agent.generateSensorimotorSequence(length, verbosity=verbosity)
      )

    return sequences


  def formatInputForTP(self):
    """
    Given an instance of the TM, format the information we need to send to the
    TP.
    """
    # all currently active cells in layer 4
    tpInputVector = numpy.zeros(
                  self.tm.connections.numberOfCells()).astype(realDType)
    tpInputVector[list(self.tm.activeCells)] = 1

    # bursting columns in layer 4
    burstingColumns = numpy.zeros(
      self.tm.connections.numberOfColumns()).astype(realDType)
    burstingColumns[list(self.tm.unpredictedActiveColumnsList[-1])] = 1

    # correctly predicted cells in layer 4
    correctlyPredictedCells = numpy.zeros(
      self.tm.connections.numberOfCells()).astype(realDType)
    correctlyPredictedCells[list(self.tm.predictedActiveCellsList[-1])] = 1

    return (tpInputVector, burstingColumns, correctlyPredictedCells)


  def formatRow(self, x, formatString = "%d", rowSize = 700):
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


