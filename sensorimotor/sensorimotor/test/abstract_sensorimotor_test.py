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

import unittest2 as unittest

import numpy

from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)
from sensorimotor.general_temporal_memory import GeneralTemporalMemory
# Uncomment the lines below to use FastGeneralTemporalMemory
# from sensorimotor.fast_general_temporal_memory import (
#   FastGeneralTemporalMemory as GeneralTemporalMemory)
class MonitoredSensorimotorTemporalMemory(TemporalMemoryMonitorMixin,
                                          GeneralTemporalMemory): pass



class AbstractSensorimotorTest(unittest.TestCase):

  VERBOSITY = 1
  DEFAULT_TM_PARAMS = {}
  SEED = 42


  def _init(self, tmOverrides=None):
    """
    Initialize Sensorimotor Temporal Memory, and other member variables.

    :param tmOverrides: overrides for default Temporal Memory parameters
    """
    params = self._computeTMParams(tmOverrides)
    # Uncomment the line below to disable learn on one cell mode
    # params["learnOnOneCell"] = False
    self.tm = MonitoredSensorimotorTemporalMemory(**params)


  def _feedTM(self, sequence, learn=True):
    (sensorSequence,
     motorSequence,
     sensorimotorSequence,
     sequenceLabels) = sequence

    self.tm.mmClearHistory()

    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      motorPattern = motorSequence[i]
      sensorimotorPattern = sensorimotorSequence[i]
      sequenceLabel = sequenceLabels[i]
      if sensorPattern is None:
        self.tm.reset()
      else:
        self.tm.compute(sensorPattern,
                        activeExternalCells=motorPattern,
                        formInternalConnections=True,
                        learn=learn,
                        sequenceLabel=sequenceLabel)

    if self.VERBOSITY >= 2:
      print self.tm.mmPrettyPrintTraces(
        self.tm.mmGetDefaultTraces(verbosity=self.VERBOSITY-1),
        breakOnResets=self.tm.mmGetTraceResets())
      print

    if self.VERBOSITY >= 2:
      print self.tm.mmPrettyPrintSequenceCellRepresentations()
      print

    if learn and self.VERBOSITY >= 3:
      print self.tm.mmPrettyPrintConnections()


  def _testTM(self, sequence):

    self._feedTM(sequence, learn=False)

    print self.tm.mmPrettyPrintMetrics(self.tm.mmGetDefaultMetrics())


  # ==============================
  # Overrides
  # ==============================


  def setUp(self):
    self.tm = None
    self._random = numpy.random.RandomState(self.SEED)

    print ("\n"
           "======================================================\n"
           "Test: {0} \n"
           "{1}\n"
           "======================================================\n"
    ).format(self.id(), self.shortDescription())


  # ==============================
  # Helper functions
  # ==============================

  @classmethod
  def _generateSensorimotorSequences(cls, length, agents):
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
      s,m,sm = agent.generateSensorimotorSequence(length,
                                                  verbosity=cls.VERBOSITY-1)
      sensorSequence += s
      motorSequence += m
      sensorimotorSequence += sm
      sequenceLabels += [agent.world.toString()] * length

      sensorSequence.append(None)
      motorSequence.append(None)
      sensorimotorSequence.append(None)
      sequenceLabels.append(None)

    return sensorSequence, motorSequence, sensorimotorSequence, sequenceLabels


  def _computeTMParams(self, overrides):
    params = dict(self.DEFAULT_TM_PARAMS)
    params.update(overrides or {})
    return params

