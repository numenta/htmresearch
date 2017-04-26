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
      columnDimensions=(10,),
      basalInputDimensions=(10,),
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2)

    numCells = 40

    segment = tm.basalConnections.createSegment(4)
    tm.basalConnections.createSynapse(segment, numCells + 0, 0.5)
    tm.basalConnections.createSynapse(segment, numCells + 1, 0.5)
    tm.basalConnections.createSynapse(segment, numCells + 2, 0.5)

    tm.depolarizeCells(activeCellsExternalBasal = [0, 1, 2])

    self.assertEqual(tm.getPredictiveCells(), [4],
                     "External basal input should depolarize cells.")

    activeColumns = [1]
    tm.activateCells(activeColumns = [1],
                     reinforceCandidatesExternalBasal = [0, 1, 2],
                     growthCandidatesExternalBasal = [0, 1, 2])

    self.assertEqual(tm.getActiveCells(), [4],
                     "The column should be predicted.")


  def testDontDepolarizeWithOnlyApicalInput(self):
    """
    Verify that apical input can't depolarize a cell by itself.
    """
    tm = self.constructTM(
      columnDimensions=(10,),
      apicalInputDimensions=(10,),
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2)

    numCells = 40

    segment = tm.apicalConnections.createSegment(4)
    tm.apicalConnections.createSynapse(segment, numCells + 0, 0.5)
    tm.apicalConnections.createSynapse(segment, numCells + 1, 0.5)
    tm.apicalConnections.createSynapse(segment, numCells + 2, 0.5)

    tm.depolarizeCells(activeCellsExternalApical = [0, 1, 2])

    np.testing.assert_equal(tm.getPredictiveCells(), [],
                            "Apical input isn't enough to predict a cell.")

    activeColumns = [1]
    tm.activateCells(activeColumns = [1],
                     reinforceCandidatesExternalApical = [0, 1, 2],
                     growthCandidatesExternalApical = [0, 1, 2])

    np.testing.assert_equal(tm.getActiveCells(), [4, 5, 6, 7],
                            "The column should burst.")


  def testApicalInputDisambiguatesPredictiveCells(self):
    """
    Apical input should give cells an extra boost. If a cell in a column has
    active apical and basal segments, other cells in that column with only
    active basal segments are not depolarized.
    """
    tm = self.constructTM(
      columnDimensions=(10,),
      basalInputDimensions=(10,),
      apicalInputDimensions=(10,),
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2)

    numCells = 40

    basalSegment1 = tm.basalConnections.createSegment(4)
    tm.basalConnections.createSynapse(basalSegment1, numCells + 0, 0.5)
    tm.basalConnections.createSynapse(basalSegment1, numCells + 1, 0.5)
    tm.basalConnections.createSynapse(basalSegment1, numCells + 2, 0.5)

    basalSegment2 = tm.basalConnections.createSegment(5)
    tm.basalConnections.createSynapse(basalSegment2, numCells + 0, 0.5)
    tm.basalConnections.createSynapse(basalSegment2, numCells + 1, 0.5)
    tm.basalConnections.createSynapse(basalSegment2, numCells + 2, 0.5)

    apicalSegment = tm.apicalConnections.createSegment(4)
    tm.apicalConnections.createSynapse(apicalSegment, numCells + 0, 0.5)
    tm.apicalConnections.createSynapse(apicalSegment, numCells + 1, 0.5)
    tm.apicalConnections.createSynapse(apicalSegment, numCells + 2, 0.5)

    tm.depolarizeCells(activeCellsExternalBasal = [0, 1, 2],
                       activeCellsExternalApical = [0, 1, 2])

    np.testing.assert_equal(tm.getPredictiveCells(), [4],
                            "Cells with active basal and apical should inhibit "
                            "cells with just basal input.")

    activeColumns = [1]
    tm.activateCells(activeColumns = [1],
                     reinforceCandidatesExternalBasal = [0, 1, 2],
                     growthCandidatesExternalBasal = [0, 1, 2],
                     reinforceCandidatesExternalApical = [0, 1, 2],
                     growthCandidatesExternalApical = [0, 1, 2])

    np.testing.assert_equal(tm.getActiveCells(), [4],
                            "Cells with active basal and apical should inhibit "
                            "cells with just basal input.")


  def testGrowBasalSynapses(self):
    """
    Grow basal synapses from winner cells to previous winner cells and to
    'growthCandidatesExternalBasal'.
    """
    tm = self.constructTM(
      columnDimensions=(10,),
      basalInputDimensions=(20,),
      apicalInputDimensions=(20,),
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      formInternalBasalConnections=True)

    tm.activateCells(activeColumns = [0, 1])
    prevWinnerCells = tm.getWinnerCells()
    tm.depolarizeCells(activeCellsExternalBasal = [10, 11])
    tm.activateCells(activeColumns = [2],
                     reinforceCandidatesExternalBasal = [10, 11],
                     growthCandidatesExternalBasal = [10, 11])

    winnerCell = tm.getWinnerCells()[0]
    segments = list(tm.basalConnections.segmentsForCell(winnerCell))
    self.assertEquals(len(segments), 1,
                      "A basal segment should grow on the winner cell.")

    numCells = 40
    expectedPresynapticCells = set(prevWinnerCells) | set([numCells + 10,
                                                           numCells + 11])

    synapses = tm.basalConnections.synapsesForSegment(segments[0])
    presynapticCells = set(tm.basalConnections.dataForSynapse(s).presynapticCell
                           for s in synapses)

    self.assertEqual(presynapticCells, expectedPresynapticCells,
                     "It should grow synapses to all growth candidates.")


  def testGrowApicalSynapses(self):
    """
    Grow apical synapses from winner cells to 'growthCandidatesExternalApical'.
    """
    tm = self.constructTM(
      columnDimensions=(10,),
      basalInputDimensions=(20,),
      apicalInputDimensions=(20,),
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10)

    tm.activateCells(activeColumns = [0, 1])
    prevWinnerCells = tm.getWinnerCells()
    tm.depolarizeCells(activeCellsExternalApical = [10, 11])
    tm.activateCells(activeColumns = [2],
                     reinforceCandidatesExternalApical = [10, 11],
                     growthCandidatesExternalApical = [10, 11])

    winnerCell = tm.getWinnerCells()[0]
    segments = list(tm.apicalConnections.segmentsForCell(winnerCell))
    self.assertEquals(len(segments), 1,
                      "A apical segment should grow on the winner cell.")

    numCells = 40
    expectedPresynapticCells = set([numCells + 10, numCells + 11])

    synapses = tm.apicalConnections.synapsesForSegment(segments[0])
    presynapticCells = set(tm.apicalConnections.dataForSynapse(s).presynapticCell
                           for s in synapses)

    self.assertEqual(presynapticCells, expectedPresynapticCells,
                     "It should grow synapses to all growth candidates.")


  def testReinforceActiveBasalSynapses(self):
    """
    Reinforce basal synapses from active segments to previous active internal
    cells and to 'reinforceCandidatesExternalBasal'.
    """
    tm = self.constructTM(
      columnDimensions=(10,),
      basalInputDimensions=(20,),
      apicalInputDimensions=(20,),
      cellsPerColumn=4,
      activationThreshold=3,
      connectedPermanence=.5,
      minThreshold=2,
      permanenceIncrement=.10,
      formInternalBasalConnections=True)

    numCells = 40

    # Intentionally connect to an entire column so that we don't grow a synapse.
    segment = tm.basalConnections.createSegment(4)
    tm.basalConnections.createSynapse(segment, 0, 0.5)
    tm.basalConnections.createSynapse(segment, 1, 0.5)
    tm.basalConnections.createSynapse(segment, 2, 0.5)
    tm.basalConnections.createSynapse(segment, 3, 0.5)
    tm.basalConnections.createSynapse(segment, numCells + 10, 0.5)
    tm.basalConnections.createSynapse(segment, numCells + 11, 0.5)

    tm.activateCells(activeColumns = [0])
    tm.depolarizeCells(activeCellsExternalBasal = [10, 11])
    tm.activateCells(activeColumns = [1],
                     reinforceCandidatesExternalBasal = [10, 11],
                     growthCandidatesExternalBasal = [10, 11])

    np.testing.assert_equal(tm.getActiveCells(), [4])

    segments = list(tm.basalConnections.segmentsForCell(4))
    self.assertEquals(len(segments), 1,
                      "There should still only be one segment.")

    expectedPresynapticCells = set([0, 1, 2, 3]) | set([numCells + 10,
                                                        numCells + 11])

    synapses = tm.basalConnections.synapsesForSegment(segments[0])
    synapseData = list(tm.basalConnections.dataForSynapse(s)
                       for s in synapses)

    self.assertEqual(set(d.presynapticCell for d in synapseData),
                     expectedPresynapticCells)

    for d in synapseData:
      self.assertAlmostEqual(d.permanence, 0.6)


  def testReinforceActiveApicalSynapses(self):
    """
    Reinforce apical synapses from active segments on predictive cells to
    'reinforceCandidatesExternalApical'.
    """
    tm = self.constructTM(
      columnDimensions=(10,),
      basalInputDimensions=(20,),
      apicalInputDimensions=(20,),
      cellsPerColumn=4,
      activationThreshold=2,
      connectedPermanence=.5,
      minThreshold=1,
      permanenceIncrement=.10)

    numCells = 40

    # Also connect a basal segment so that the cell becomes depolarized.
    basalSegment = tm.basalConnections.createSegment(4)
    tm.basalConnections.createSynapse(basalSegment, numCells + 0, 0.5)
    tm.basalConnections.createSynapse(basalSegment, numCells + 1, 0.5)

    apicalSegment = tm.apicalConnections.createSegment(4)
    tm.apicalConnections.createSynapse(apicalSegment, numCells + 10, 0.5)
    tm.apicalConnections.createSynapse(apicalSegment, numCells + 11, 0.5)

    tm.depolarizeCells(activeCellsExternalBasal = [0, 1],
                       activeCellsExternalApical = [10, 11])
    tm.activateCells(activeColumns = [1],
                     reinforceCandidatesExternalBasal = [0, 1],
                     growthCandidatesExternalBasal = [0, 1],
                     reinforceCandidatesExternalApical = [10, 11],
                     growthCandidatesExternalApical = [10, 11])

    np.testing.assert_equal(tm.getActiveCells(), [4])

    segments = list(tm.apicalConnections.segmentsForCell(4))
    self.assertEquals(len(segments), 1,
                      "There should still only be one segment.")

    expectedPresynapticCells = set([numCells + 10, numCells + 11])

    synapses = tm.apicalConnections.synapsesForSegment(segments[0])
    synapseData = list(tm.apicalConnections.dataForSynapse(s)
                       for s in synapses)

    self.assertEqual(set(d.presynapticCell for d in synapseData),
                     expectedPresynapticCells)

    for d in synapseData:
      self.assertAlmostEqual(d.permanence, 0.6)
