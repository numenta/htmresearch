#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
"""
Unit tests for Extended Temporal Memory.
"""

import unittest

from htmresearch_core.experimental import ExtendedTemporalMemory

import numpy as np



class ExtendedTemporalMemoryUnitTest(unittest.TestCase):

  def constructTM(self, **params):
    return ExtendedTemporalMemory(**params)


  def testDepolarizeWithExternalBasalInput(self):
    """
    Verify that external basal input can depolarize a cell.
    """
    tm = self.constructTM(
      columnCount=10,
      basalInputSize=10,
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2)

    segment = tm.createBasalSegment(4)
    tm.basalConnections.createSynapse(segment, 0, 0.5)
    tm.basalConnections.createSynapse(segment, 1, 0.5)
    tm.basalConnections.createSynapse(segment, 2, 0.5)

    tm.compute(activeColumns=[1],
               basalInput= [0, 1, 2])

    np.testing.assert_equal(tm.getPredictedCells(), [4],
                            "External basal input should depolarize cells.")
    self.assertEqual(tm.getActiveCells(), [4],
                     "The column should be predicted.")


  def testDontDepolarizeWithOnlyApicalInput(self):
    """
    Verify that apical input can't depolarize a cell by itself.
    """
    tm = self.constructTM(
      columnCount=10,
      apicalInputSize=10,
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2)

    segment = tm.createApicalSegment(4)
    tm.apicalConnections.createSynapse(segment, 0, 0.5)
    tm.apicalConnections.createSynapse(segment, 1, 0.5)
    tm.apicalConnections.createSynapse(segment, 2, 0.5)

    tm.compute(activeColumns=[1],
               apicalInput=[0, 1, 2])

    np.testing.assert_equal(tm.getPredictedCells(), [],
                            "Apical input isn't enough to predict a cell.")
    np.testing.assert_equal(tm.getActiveCells(), [4, 5, 6, 7],
                            "The column should burst.")


  def testApicalInputDisambiguatesPredictiveCells(self):
    """
    Apical input should give cells an extra boost. If a cell in a column has
    active apical and basal segments, other cells in that column with only
    active basal segments are not depolarized.
    """
    tm = self.constructTM(
      columnCount=10,
      basalInputSize=10,
      apicalInputSize=10,
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2)

    basalSegment1 = tm.createBasalSegment(4)
    tm.basalConnections.createSynapse(basalSegment1, 0, 0.5)
    tm.basalConnections.createSynapse(basalSegment1, 1, 0.5)
    tm.basalConnections.createSynapse(basalSegment1, 2, 0.5)

    basalSegment2 = tm.createBasalSegment(5)
    tm.basalConnections.createSynapse(basalSegment2, 0, 0.5)
    tm.basalConnections.createSynapse(basalSegment2, 1, 0.5)
    tm.basalConnections.createSynapse(basalSegment2, 2, 0.5)

    apicalSegment = tm.createApicalSegment(4)
    tm.apicalConnections.createSynapse(apicalSegment, 0, 0.5)
    tm.apicalConnections.createSynapse(apicalSegment, 1, 0.5)
    tm.apicalConnections.createSynapse(apicalSegment, 2, 0.5)

    tm.compute(activeColumns=[1],
               basalInput=[0, 1, 2],
               apicalInput=[0, 1, 2])

    np.testing.assert_equal(tm.getPredictedCells(), [4],
                            "Cells with active basal and apical should inhibit "
                            "cells with just basal input.")
    np.testing.assert_equal(tm.getActiveCells(), [4],
                            "Cells with active basal and apical should inhibit "
                            "cells with just basal input.")


  def testGrowBasalSynapses(self):
    """
    Grow basal synapses from winner cells to basal inputs.
    """
    tm = self.constructTM(
      columnCount=10,
      basalInputSize=20,
      apicalInputSize=20,
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      sampleSize=4,
      permanenceIncrement=.10)

    tm.compute(activeColumns=[2], basalInput=[10, 11])

    winnerCell = tm.getWinnerCells()[0]
    segments = list(tm.basalConnections.segmentsForCell(winnerCell))
    self.assertEquals(len(segments), 1,
                      "A basal segment should grow on the winner cell.")

    synapses = tm.basalConnections.synapsesForSegment(segments[0])
    presynapticCells = set(tm.basalConnections.dataForSynapse(s).presynapticCell
                           for s in synapses)

    self.assertEqual(presynapticCells, set([10, 11]),
                     "It should grow synapses to all growth candidates.")


  def testGrowApicalSynapses(self):
    """
    Grow apical synapses from winner cells to apical inputs.
    """
    tm = self.constructTM(
      columnCount=10,
      basalInputSize=20,
      apicalInputSize=20,
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      sampleSize=4,
      permanenceIncrement=.10)

    tm.compute(activeColumns=[2], apicalInput=[10, 11])

    winnerCell = tm.getWinnerCells()[0]
    segments = list(tm.apicalConnections.segmentsForCell(winnerCell))
    self.assertEquals(len(segments), 1,
                      "A apical segment should grow on the winner cell.")

    synapses = tm.apicalConnections.synapsesForSegment(segments[0])
    presynapticCells = set(tm.apicalConnections.dataForSynapse(s).presynapticCell
                           for s in synapses)

    self.assertEqual(presynapticCells, set([10, 11]),
                     "It should grow synapses to all growth candidates.")


  def testReinforceActiveBasalSynapses(self):
    """
    Reinforce basal synapses from active segments to basalInput.
    """
    tm = self.constructTM(
      columnCount=10,
      basalInputSize=20,
      apicalInputSize=20,
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2,
      permanenceIncrement=.10)

    segment = tm.createBasalSegment(4)
    tm.basalConnections.createSynapse(segment, 10, 0.5)
    tm.basalConnections.createSynapse(segment, 11, 0.5)
    tm.basalConnections.createSynapse(segment, 12, 0.5)

    tm.compute(activeColumns=[1],
               basalInput=[10, 11, 12])

    np.testing.assert_equal(tm.getActiveCells(), [4])

    segments = list(tm.basalConnections.segmentsForCell(4))
    self.assertEquals(len(segments), 1,
                      "There should still only be one segment.")

    synapses = tm.basalConnections.synapsesForSegment(segments[0])
    synapseData = list(tm.basalConnections.dataForSynapse(s)
                       for s in synapses)

    self.assertEqual(set(d.presynapticCell for d in synapseData),
                     set([10, 11, 12]))

    for d in synapseData:
      self.assertAlmostEqual(d.permanence, 0.6)


  def testReinforceActiveApicalSynapses(self):
    """
    Reinforce apical synapses from active segments to apicalInput.
    """
    tm = self.constructTM(
      columnCount=10,
      basalInputSize=20,
      apicalInputSize=20,
      cellsPerColumn=4,
      activationThreshold=2,
      connectedPermanence=.5,
      minThreshold=2,
      permanenceIncrement=.10)

    # Make sure the cell is predicted.
    basalSegment = tm.createBasalSegment(4)
    tm.basalConnections.createSynapse(basalSegment, 16, 0.5)
    tm.basalConnections.createSynapse(basalSegment, 17, 0.5)

    segment = tm.createApicalSegment(4)
    tm.apicalConnections.createSynapse(segment, 10, 0.5)
    tm.apicalConnections.createSynapse(segment, 11, 0.5)
    tm.apicalConnections.createSynapse(segment, 12, 0.5)

    tm.compute(activeColumns=[1],
               basalInput=[16, 17],
               apicalInput=[10, 11, 12])

    np.testing.assert_equal(tm.getActiveCells(), [4])

    segments = list(tm.apicalConnections.segmentsForCell(4))
    self.assertEquals(len(segments), 1,
                      "There should still only be one segment.")

    synapses = tm.apicalConnections.synapsesForSegment(segments[0])
    synapseData = list(tm.apicalConnections.dataForSynapse(s)
                       for s in synapses)

    self.assertEqual(set(d.presynapticCell for d in synapseData),
                     set([10, 11, 12]))

    for d in synapseData:
      self.assertAlmostEqual(d.permanence, 0.6)
