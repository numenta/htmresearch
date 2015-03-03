#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

from nupic.bindings.math import GetNTAReal

from nupic.data.pattern_machine import PatternMachine
from nupic.data.sequence_machine import SequenceMachine

from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)
from nupic.research.temporal_memory import TemporalMemory
from sensorimotor.spatial_temporal_pooler import (
  SpatialTemporalPooler as TemporalPooler)
from sensorimotor.temporal_pooler_monitor_mixin import (
  TemporalPoolerMonitorMixin)

class MonitoredTemporalMemory(TemporalMemoryMonitorMixin, TemporalMemory): pass
class MonitoredTemporalPooler(TemporalPoolerMonitorMixin, TemporalPooler): pass



realDType = GetNTAReal()



VERBOSITY = 2
PLOT = 1



class SpatialTemporalPoolerTest(unittest.TestCase):

  def setUp(self):
    self.patternMachine = PatternMachine(1024, 20, num=100)
    self.sequenceMachine = SequenceMachine(self.patternMachine)

    self.tm = MonitoredTemporalMemory(
      mmName="TM",
      columnDimensions=[1024],
      cellsPerColumn=16,
      initialPermanence=0.5,
      connectedPermanence=0.7,
      minThreshold=20,
      maxNewSynapseCount=30,
      permanenceIncrement=0.1,
      permanenceDecrement=0.02,
      activationThreshold=20)

    self.tp = MonitoredTemporalPooler(
      inputDimensions=[1024, 16],
      columnDimensions=[1024],
      mmName="TP")


  def testOverlapping(self):
    sequences = [
      [ 0,  1,  2,  3,  4,  5,  6,  7],
      [ 8,  9, 10, 11, 12, 13, 14, 15],
      [ 0, 16, 17, 18, 19, 20, 21,  7],
      [22, 23, 24, 11, 12, 25, 26, 27]
    ]
    labels = ["A", "B", "C", "D"]

    sequences = [self.sequenceMachine.generateFromNumbers(s) for s in sequences]

    for _ in xrange(10):
      self._feedSequences(sequences, sequenceLabels=labels)

    self._printInfo()
    self._showPlots()


  def tearDown(self):
    if PLOT >= 1:
      raw_input("Press any key to exit...")


  def _feedSequences(self, sequences,
                     tmLearn=True, tpLearn=True, sequenceLabels=None):
    for i in xrange(len(sequences)):
      sequence = sequences[i]
      label = sequenceLabels[i] if sequenceLabels is not None else None
      for pattern in sequence:
        self._feedPattern(pattern, tmLearn=tmLearn, tpLearn=tpLearn,
                          sequenceLabel=label)
      self.tm.reset()
      self.tp.reset()


  def _printInfo(self):
    if VERBOSITY >= 2:
      print MonitorMixinBase.mmPrettyPrintTraces(
        self.tp.mmGetDefaultTraces(verbosity=3) +
        self.tm.mmGetDefaultTraces(verbosity=3),
        breakOnResets=self.tm.mmGetTraceResets())
      print

    print MonitorMixinBase.mmPrettyPrintMetrics(
      self.tp.mmGetDefaultMetrics() + self.tm.mmGetDefaultMetrics())
    print


  def _showPlots(self):
    if PLOT >= 1:
      self.tp.mmGetCellActivityPlot(showReset=True)


  def _feedPattern(self, pattern,
                   tmLearn=True, tpLearn=True, sequenceLabel=None):
    # Feed the TM
    predictedCells = self.tm.predictiveCells
    self.tm.compute(pattern, learn=tmLearn, sequenceLabel=sequenceLabel)

    # If requested, feed the TP
    if tpLearn is not None:
      tpInputVector, correctlyPredictedCells = (
        self._formatInputForTP(predictedCells))
      activeArray = numpy.zeros(self.tp.getNumColumns())

      self.tp.compute(tpInputVector,
                      tpLearn,
                      activeArray,
                      None,  # not needed
                      correctlyPredictedCells,
                      sequenceLabel=sequenceLabel)


  def _formatInputForTP(self, predictedCells):
    """
    Given an instance of the TM, format the information we need to send to the
    TP.
    """
    # all currently active cells in layer 4
    tpInputVector = numpy.zeros(
                  self.tm.numberOfCells()).astype(realDType)
    tpInputVector[list(self.tm.activeCells)] = 1

    # correctly predicted cells in layer 4
    correctlyPredictedCells = numpy.zeros(
      self.tm.numberOfCells()).astype(realDType)
    correctlyPredictedCells[list(predictedCells & self.tm.activeCells)] = 1

    return tpInputVector, correctlyPredictedCells



if __name__ == "__main__":
  unittest.main()
