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
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)

from sensorimotor.general_temporal_memory import GeneralTemporalMemory
from sensorimotor.temporal_pooler import TemporalPooler
from sensorimotor.temporal_pooler_monitor_mixin import (
  TemporalPoolerMonitorMixin)
class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                     GeneralTemporalMemory): pass
class MonitoredTemporalPooler(TemporalPoolerMonitorMixin, TemporalPooler): pass

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
    self.tm = MonitoredGeneralTemporalMemory(__name__="TM", **params)

    # Initialize Layer 3 temporal pooler
    params = dict(self.DEFAULT_TP_PARAMS)
    params["inputDimensions"] = [self.tm.connections.numberOfCells()]
    params["potentialRadius"] = self.tm.connections.numberOfCells()
    params.update(tpOverrides or {})
    self._checkParams(params)
    self.tp = MonitoredTemporalPooler(__name__="TP", **params)


  def _checkParams(self, params):
    for k,v in params.iteritems():
      if v == "Sorry":
        raise RuntimeError("Param "+k+" must be specified")


  def feedLayers(self, sequences, tmLearn, tpLearn=None, verbosity=0):
    """
    Feed the given sequences to the HTM algorithms.

    @param tmLearn:   (bool)      Either False, or True
    @param tpLearn:   (None,bool) Either None, False, or True. If None,
                                  temporal pooler will be skipped.
    """
    (sensorSequence,
     motorSequence,
     sensorimotorSequence,
     sequenceLabels) = sequences

    self.tm.clearHistory()
    self.tp.clearHistory()

    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      sensorimotorPattern = sensorimotorSequence[i]
      sequenceLabel = sequenceLabels[i]

      if sensorPattern is None:
        self.tm.reset()
        self.tp.reset()

      else:
        # Feed the TM
        self.tm.compute(sensorPattern,
                  activeExternalCells=sensorimotorPattern,
                  formInternalConnections=False,
                  learn=tmLearn,
                  sequenceLabel=sequenceLabel)

        # If requested, feed the TP
        if tpLearn is not None:
          tpInputVector, burstingColumns, correctlyPredictedCells = (
              self.formatInputForTP())
          activeArray = numpy.zeros(self.tp.getNumColumns())

          self.tp.compute(tpInputVector,
                          tpLearn,
                          activeArray,
                          burstingColumns,
                          correctlyPredictedCells,
                          sequenceLabel=sequenceLabel)

    if verbosity >= 2:
      traces = []
      traces += self.tm.getDefaultTraces(verbosity=verbosity)
      if tpLearn is not None:
        traces += self.tp.getDefaultTraces(verbosity=verbosity)
      print MonitorMixinBase.prettyPrintTraces(
        traces, breakOnResets=self.tm.getTraceResets())
      print


  @staticmethod
  def generateSequences(length, agents, verbosity=0):
    """
    @param length (int)           Length of each sequence to generate, one for
                                  each agent
    @param agents (AbstractAgent) Agents acting in their worlds

    @return (tuple) (sensor sequence, motor sequence, sensorimotor sequence,
                     sequence labels)
    """
    sensorSequence = []
    motorSequence = []
    sensorimotorSequence = []
    sequenceLabels = []

    for agent in agents:
      s,m,sm = agent.generateSensorimotorSequence(length, verbosity=verbosity)
      sensorSequence += s
      motorSequence += m
      sensorimotorSequence += sm
      sequenceLabels += [str(agent.world)] * length

      sensorSequence.append(None)
      motorSequence.append(None)
      sensorimotorSequence.append(None)
      sequenceLabels.append(None)

    return (sensorSequence, motorSequence, sensorimotorSequence, sequenceLabels)


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
    burstingColumns[
      list(self.tm.getTraceUnpredictedActiveColumns().data[-1])] = 1

    # correctly predicted cells in layer 4
    correctlyPredictedCells = numpy.zeros(
      self.tm.connections.numberOfCells()).astype(realDType)
    correctlyPredictedCells[
      list(self.tm.getTracePredictedActiveCells().data[-1])] = 1

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


