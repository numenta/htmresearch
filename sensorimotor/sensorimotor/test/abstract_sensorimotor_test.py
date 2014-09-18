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
from prettytable import PrettyTable

from sensorimotor.general_temporal_memory import (
  InspectGeneralTemporalMemory as SensorimotorTemporalMemory)



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
    self.tm = SensorimotorTemporalMemory(**params)


  def _feedTM(self, sequence, learn=True):
    (sensorSequence,
     motorSequence,
     sensorimotorSequence,
     sequenceLabels) = sequence

    self.tm.clearHistory()

    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      sensorimotorPattern = sensorimotorSequence[i]
      sequenceLabel = sequenceLabels[i]
      if sensorPattern is None:
        self.tm.reset()
      else:
        self.tm.compute(sensorPattern,
                        activeExternalCells=sensorimotorPattern,
                        formInternalConnections=False,
                        learn=learn,
                        sequenceLabel=sequenceLabel)

    if self.VERBOSITY >= 2:
      print self.tm.prettyPrintHistory(verbosity=self.VERBOSITY-2)
      print

    if learn and self.VERBOSITY >= 3:
      print self.tm.prettyPrintConnections()


  def _testTM(self, sequence):

    self._feedTM(sequence, learn=False)
    stats = self.tm.getStatistics()

    self.allStats.append((self.id(), stats))

    return stats


  # ==============================
  # Overrides
  # ==============================

  @classmethod
  def setUpClass(cls):
    cls.allStats = []


  @classmethod
  def tearDownClass(cls):
    cols = ["Test",
            "predicted active cells (stats)",
            "predicted inactive cells (stats)",
            "predicted active columns (stats)",
            "predicted inactive columns (stats)",
            "unpredicted active columns (stats)"]

    table = PrettyTable(cols)

    for stats in cls.allStats:
      row = [stats[0]] + [tuple(x) for x in list(stats[1])]
      table.add_row(row)

    print
    print table
    print "(stats) => (min, max, sum, average, standard deviation)"


  def setUp(self):
    self.tm = None
    self.tmTestMachine = None
    self._random = numpy.random.RandomState(self.SEED)

    if self.VERBOSITY >= 2:
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
      sequenceLabels += [str(agent.world)] * length

      sensorSequence.append(None)
      motorSequence.append(None)
      sensorimotorSequence.append(None)
      sequenceLabels.append(None)

    return (sensorSequence, motorSequence, sensorimotorSequence, sequenceLabels)


  def _computeTMParams(self, overrides):
    params = dict(self.DEFAULT_TM_PARAMS)
    params.update(overrides or {})
    return params

