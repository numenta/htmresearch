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

"""
Utilities for running data through the Sensorimotor TM, and analyzing the
results.
"""

from nupic.test.temporal_memory_test_machine import TemporalMemoryTestMachine



class SensorimotorTemporalMemoryTestMachine(TemporalMemoryTestMachine):
  """
  Sensorimotor TM test machine class.
  """

  def feedSensorimotorSequence(self,
                               sensorSequence,
                               sensorimotorSequence,
                               learn=True):
    """
    Feed a sensorimotor sequence through the Sensorimotor TM.

    @param sensorSequence       (list) List of sensor patterns, with None for
                                       resets
    @param sensorimotorSequence (list) List of sensor+motor patterns, with None
                                       for resets
    @param learn                (bool) Learning enabled

    @return (list) List of sets containing predictive cells,
                   one for each element in `sequence`
    """
    results = []

    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      sensorimotorPattern = sensorimotorSequence[i]
      if sensorPattern is None:
        self.tm.reset()
      else:
        self.tm.compute(sensorPattern,
                        activeExternalCells=sensorimotorPattern,
                        formInternalConnections=False,
                        learn=learn)

      results.append(self.tm.predictiveCells)

    return results

