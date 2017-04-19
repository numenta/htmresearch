#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-2016, Numenta, Inc.  Unless you have an agreement
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

import unittest

from htmresearch_core.experimental import ExtendedTemporalMemory


class TemporalMemoryUnitTest(unittest.TestCase):
  """
  The TemporalMemory unit tests, ported to work with the ExtendedTemporalMemory.
  """

  def constructTM(self, **params):
    return ExtendedTemporalMemory(**params)


  def testInitInvalidParams(self):
    # Invalid columnDimensions
    kwargs = {"columnDimensions": [], "cellsPerColumn": 32}
    self.assertRaises((ValueError, RuntimeError,),
                      ExtendedTemporalMemory, **kwargs)

    # Invalid cellsPerColumn
    kwargs = {"columnDimensions": [2048], "cellsPerColumn": 0}
    self.assertRaises((ValueError, RuntimeError,),
                      ExtendedTemporalMemory, **kwargs)
    kwargs = {"columnDimensions": [2048], "cellsPerColumn": -10}
    self.assertRaises((ValueError, RuntimeError,),
                      ExtendedTemporalMemory, **kwargs)


  def testActivateCorrectlyPredictiveCells(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0]
    activeColumns = [1]
    previousActiveCells = [0,1,2,3]
    expectedActiveCells = [4]

    activeSegment = tm.basalConnections.createSegment(expectedActiveCells[0])
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], .5)

    tm.compute(previousActiveColumns)
    self.assertEqual(expectedActiveCells, tm.getPredictiveCells())
    tm.compute(activeColumns)
    self.assertEqual(expectedActiveCells, tm.getActiveCells())


  def testBurstUnpredictedColumns(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    activeColumns = [0]
    burstingCells = [0, 1, 2, 3]

    tm.compute(activeColumns)

    self.assertEqual(burstingCells, tm.getActiveCells())


  def testZeroActiveColumns(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0]
    previousActiveCells = [0, 1, 2, 3]
    expectedActiveCells = [4]

    segment = tm.basalConnections.createSegment(expectedActiveCells[0])
    tm.basalConnections.createSynapse(segment, previousActiveCells[0], .5)
    tm.basalConnections.createSynapse(segment, previousActiveCells[1], .5)
    tm.basalConnections.createSynapse(segment, previousActiveCells[2], .5)
    tm.basalConnections.createSynapse(segment, previousActiveCells[3], .5)

    tm.compute(previousActiveColumns)
    self.assertNotEquals(len(tm.getActiveCells()), 0)
    self.assertNotEquals(len(tm.getWinnerCells()), 0)
    self.assertNotEquals(len(tm.getPredictiveCells()), 0)

    zeroColumns = []
    tm.compute(zeroColumns)

    self.assertEquals(len(tm.getActiveCells()), 0)
    self.assertEquals(len(tm.getWinnerCells()), 0)
    self.assertEquals(len(tm.getPredictiveCells()), 0)


  def testPredictedActiveCellsAreAlwaysWinners(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.5,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0]
    activeColumns = [1]
    previousActiveCells = [0, 1, 2, 3]
    expectedWinnerCells = [4, 6]

    activeSegment1 = tm.basalConnections.createSegment(expectedWinnerCells[0])
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[0], .5)
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[1], .5)
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[2], .5)

    activeSegment2 = tm.basalConnections.createSegment(expectedWinnerCells[1])
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[0], .5)
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[1], .5)
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[2], .5)

    tm.compute(previousActiveColumns, learn=False)
    tm.compute(activeColumns, learn=False)

    self.assertEqual(expectedWinnerCells, tm.getWinnerCells())


  def testReinforceCorrectlyActiveSegments(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.2,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.08,
      predictedSegmentDecrement=0.02,
      seed=42)

    prevActiveColumns = [0]
    prevActiveCells = [0,1,2,3]
    activeColumns = [1]
    activeCell = 5

    activeSegment = tm.basalConnections.createSegment(activeCell)
    as1 = tm.basalConnections.createSynapse(activeSegment, prevActiveCells[0], .5)
    as2 = tm.basalConnections.createSynapse(activeSegment, prevActiveCells[1], .5)
    as3 = tm.basalConnections.createSynapse(activeSegment, prevActiveCells[2], .5)
    is1 = tm.basalConnections.createSynapse(activeSegment, 81, .5) #inactive synapse

    tm.compute(prevActiveColumns)
    tm.compute(activeColumns)

    self.assertAlmostEqual(.6, tm.basalConnections.dataForSynapse(as1).permanence)
    self.assertAlmostEqual(.6, tm.basalConnections.dataForSynapse(as2).permanence)
    self.assertAlmostEqual(.6, tm.basalConnections.dataForSynapse(as3).permanence)
    self.assertAlmostEqual(.42, tm.basalConnections.dataForSynapse(is1).permanence)


  def testReinforceSelectedMatchingSegmentInBurstingColumn(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.08,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0]
    previousActiveCells = [0,1,2,3]
    activeColumns = [1]
    burstingCells = [4,5,6,7]

    selectedMatchingSegment = tm.basalConnections.createSegment(burstingCells[0])
    as1 = tm.basalConnections.createSynapse(selectedMatchingSegment,
                                       previousActiveCells[0], .3)
    as2 = tm.basalConnections.createSynapse(selectedMatchingSegment,
                                       previousActiveCells[1], .3)
    as3 = tm.basalConnections.createSynapse(selectedMatchingSegment,
                                       previousActiveCells[2], .3)
    is1 = tm.basalConnections.createSynapse(selectedMatchingSegment, 81, .3)

    otherMatchingSegment = tm.basalConnections.createSegment(burstingCells[1])
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                 previousActiveCells[0], .3)
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                 previousActiveCells[1], .3)
    tm.basalConnections.createSynapse(otherMatchingSegment, 81, .3)

    tm.compute(previousActiveColumns)
    tm.compute(activeColumns)

    self.assertAlmostEqual(.4, tm.basalConnections.dataForSynapse(as1).permanence)
    self.assertAlmostEqual(.4, tm.basalConnections.dataForSynapse(as2).permanence)
    self.assertAlmostEqual(.4, tm.basalConnections.dataForSynapse(as3).permanence)
    self.assertAlmostEqual(.22, tm.basalConnections.dataForSynapse(is1).permanence)


  def testNoChangeToNonselectedMatchingSegmentsInBurstingColumn(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.08,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0]
    previousActiveCells = [0,1,2,3]
    activeColumns = [1]
    burstingCells = [4,5,6,7]

    selectedMatchingSegment = tm.basalConnections.createSegment(burstingCells[0])
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                 previousActiveCells[0], .3)
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                 previousActiveCells[1], .3)
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                 previousActiveCells[2], .3)
    tm.basalConnections.createSynapse(selectedMatchingSegment, 81, .3)

    otherMatchingSegment = tm.basalConnections.createSegment(burstingCells[1])
    as1 = tm.basalConnections.createSynapse(otherMatchingSegment,
                                       previousActiveCells[0], .3)
    as2 = tm.basalConnections.createSynapse(otherMatchingSegment,
                                       previousActiveCells[1], .3)
    is1 = tm.basalConnections.createSynapse(otherMatchingSegment, 81, .3)

    tm.compute(previousActiveColumns)
    tm.compute(activeColumns)

    self.assertAlmostEqual(.3, tm.basalConnections.dataForSynapse(as1).permanence)
    self.assertAlmostEqual(.3, tm.basalConnections.dataForSynapse(as2).permanence)
    self.assertAlmostEqual(.3, tm.basalConnections.dataForSynapse(is1).permanence)


  def testNoChangeToMatchingSegmentsInPredictedActiveColumn(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0]
    activeColumns = [1]
    previousActiveCells = [0,1,2,3]
    expectedActiveCells = [4]
    otherburstingCells = [5,6,7]

    activeSegment = tm.basalConnections.createSegment(expectedActiveCells[0])
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], .5)

    matchingSegmentOnSameCell = tm.basalConnections.createSegment(
      expectedActiveCells[0])
    s1 = tm.basalConnections.createSynapse(matchingSegmentOnSameCell,
                                      previousActiveCells[0], .3)
    s2 = tm.basalConnections.createSynapse(matchingSegmentOnSameCell,
                                      previousActiveCells[1], .3)

    matchingSegmentOnOtherCell = tm.basalConnections.createSegment(
      otherburstingCells[0])
    s3 = tm.basalConnections.createSynapse(matchingSegmentOnOtherCell,
                                      previousActiveCells[0], .3)
    s4 = tm.basalConnections.createSynapse(matchingSegmentOnOtherCell,
                                      previousActiveCells[1], .3)


    tm.compute(previousActiveColumns)
    self.assertEqual(expectedActiveCells, tm.getPredictiveCells())
    tm.compute(activeColumns)

    self.assertAlmostEqual(.3, tm.basalConnections.dataForSynapse(s1).permanence)
    self.assertAlmostEqual(.3, tm.basalConnections.dataForSynapse(s2).permanence)
    self.assertAlmostEqual(.3, tm.basalConnections.dataForSynapse(s3).permanence)
    self.assertAlmostEqual(.3, tm.basalConnections.dataForSynapse(s4).permanence)


  def testNoNewSegmentIfNotEnoughWinnerCells(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    zeroColumns = []
    activeColumns = [0]

    tm.compute(zeroColumns)
    tm.compute(activeColumns)

    self.assertEqual(0, tm.basalConnections.numSegments())


  def testNewSegmentAddSynapsesToSubsetOfWinnerCells(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=2,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0, 1, 2]
    activeColumns = [4]

    tm.compute(previousActiveColumns)

    prevWinnerCells = tm.getWinnerCells() #[0, 8, 7]
    self.assertEqual(3, len(prevWinnerCells))

    tm.compute(activeColumns)

    winnerCells = tm.getWinnerCells() #[18]
    self.assertEqual(1, len(winnerCells))
    segments = list(tm.basalConnections.segmentsForCell(winnerCells[0]))
    self.assertEqual(1, len(segments))
    synapses = list(tm.basalConnections.synapsesForSegment(segments[0]))
    self.assertEqual(2, len(synapses))

    for synapse in synapses:
      synapseData = tm.basalConnections.dataForSynapse(synapse)
      self.assertAlmostEqual(.21, synapseData.permanence)
      self.assertTrue(synapseData.presynapticCell in prevWinnerCells)


  def testNewSegmentAddSynapsesToAllWinnerCells(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0, 1, 2]
    activeColumns = [4]

    tm.compute(previousActiveColumns)
    prevWinnerCells = sorted(tm.getWinnerCells())
    self.assertEqual(3, len(prevWinnerCells))

    tm.compute(activeColumns)

    winnerCells = tm.getWinnerCells()
    self.assertEqual(1, len(winnerCells))
    segments = list(tm.basalConnections.segmentsForCell(winnerCells[0]))
    self.assertEqual(1, len(segments))
    synapses = list(tm.basalConnections.synapsesForSegment(segments[0]))
    self.assertEqual(3, len(synapses))

    presynapticCells = []
    for synapse in synapses:
      synapseData = tm.basalConnections.dataForSynapse(synapse)
      self.assertAlmostEqual(.21, synapseData.permanence)
      presynapticCells.append(synapseData.presynapticCell)

    presynapticCells = sorted(presynapticCells)
    self.assertEqual(prevWinnerCells, presynapticCells)


  def testMatchingSegmentAddSynapsesToSubsetOfWinnerCells(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=1,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=1,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0, 1, 2, 3]
    prevWinnerCells = [0, 1, 2, 3]
    activeColumns = [4]

    matchingSegment = tm.basalConnections.createSegment(4)
    tm.basalConnections.createSynapse(matchingSegment, 0, .5)

    tm.compute(previousActiveColumns)
    self.assertEqual(prevWinnerCells, tm.getWinnerCells())
    tm.compute(activeColumns)

    synapses = tm.basalConnections.synapsesForSegment(matchingSegment)
    self.assertEqual(3, len(synapses))

    for synapse in synapses:
      synapseData = tm.basalConnections.dataForSynapse(synapse)
      if synapseData.presynapticCell != 0:
        self.assertAlmostEqual(.21, synapseData.permanence)
        self.assertTrue(synapseData.presynapticCell == prevWinnerCells[1] or
                        synapseData.presynapticCell == prevWinnerCells[2] or
                        synapseData.presynapticCell == prevWinnerCells[3])


  def testMatchingSegmentAddSynapsesToAllWinnerCells(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=1,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=1,
      maxNewSynapseCount=3,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    previousActiveColumns = [0, 1]
    prevWinnerCells = [0, 1]
    activeColumns = [4]

    matchingSegment = tm.basalConnections.createSegment(4)
    tm.basalConnections.createSynapse(matchingSegment, 0, .5)

    tm.compute(previousActiveColumns)
    self.assertEqual(prevWinnerCells, tm.getWinnerCells())

    tm.compute(activeColumns)

    synapses = tm.basalConnections.synapsesForSegment(matchingSegment)
    self.assertEqual(2, len(synapses))

    for synapse in synapses:
      synapseData = tm.basalConnections.dataForSynapse(synapse)
      if synapseData.presynapticCell != 0:
        self.assertAlmostEqual(.21, synapseData.permanence)
        self.assertEqual(prevWinnerCells[1], synapseData.presynapticCell)


  def testActiveSegmentGrowSynapsesAccordingToPotentialOverlap(self):
    """
    When a segment becomes active, grow synapses to previous winner cells.

    The number of grown synapses is calculated from the "matching segment"
    overlap, not the "active segment" overlap.
    """
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=1,
      activationThreshold=2,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=1,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.0,
      seed=42)

    # Use 1 cell per column so that we have easy control over the winner cells.
    previousActiveColumns = [0, 1, 2, 3, 4]
    prevWinnerCells = [0, 1, 2, 3, 4]
    activeColumns = [5]

    activeSegment = tm.basalConnections.createSegment(5)
    tm.basalConnections.createSynapse(activeSegment, 0, .5)
    tm.basalConnections.createSynapse(activeSegment, 1, .5)
    tm.basalConnections.createSynapse(activeSegment, 2, .2)

    tm.compute(previousActiveColumns)
    self.assertEqual(prevWinnerCells, tm.getWinnerCells())
    tm.compute(activeColumns)

    presynapticCells = set(
      tm.basalConnections.dataForSynapse(synapse).presynapticCell for synapse in
      tm.basalConnections.synapsesForSegment(activeSegment))
    self.assertTrue(presynapticCells == set([0, 1, 2, 3]) or
                    presynapticCells == set([0, 1, 2, 4]))


  def testDestroyWeakSynapseOnWrongPrediction(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.2,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.02,
      seed=42)

    previousActiveColumns = [0]
    previousActiveCells = [0, 1, 2, 3]
    activeColumns = [2]
    expectedActiveCells = [5]

    activeSegment = tm.basalConnections.createSegment(expectedActiveCells[0])
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], .5)

    # Weak synapse.
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], .015)

    tm.compute(previousActiveColumns)
    tm.compute(activeColumns)

    self.assertEqual(3, tm.basalConnections.numSynapses(activeSegment))


  def testDestroyWeakSynapseOnActiveReinforce(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.2,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.02,
      seed=42)

    previousActiveColumns = [0]
    previousActiveCells = [0, 1, 2, 3]
    activeColumns = [2]
    activeCell = 5

    activeSegment = tm.basalConnections.createSegment(activeCell)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], .5)
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], .5)

    # Weak inactive synapse.
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], .009)

    tm.compute(previousActiveColumns)
    tm.compute(activeColumns)

    self.assertEqual(3, tm.basalConnections.numSynapses(activeSegment))


  def testRecycleWeakestSynapseToMakeRoomForNewSynapse(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=1,
      activationThreshold=3,
      initialPermanence=.21,
      connectedPermanence=.50,
      minThreshold=1,
      maxNewSynapseCount=3,
      permanenceIncrement=.02,
      permanenceDecrement=.02,
      predictedSegmentDecrement=0.0,
      seed=42,
      maxSynapsesPerSegment=3)

    prevActiveColumns = [0, 1, 2]
    prevWinnerCells = [0, 1, 2]
    activeColumns = [4]

    matchingSegment = tm.basalConnections.createSegment(4)
    tm.basalConnections.createSynapse(matchingSegment, 81, .6)

    weakestSynapse = tm.basalConnections.createSynapse(matchingSegment, 0, .11)

    tm.compute(prevActiveColumns)
    self.assertEqual(prevWinnerCells, tm.getWinnerCells())
    tm.compute(activeColumns)

    synapses = tm.basalConnections.synapsesForSegment(matchingSegment)
    self.assertEqual(3, len(synapses))
    presynapticCells = set(
      tm.basalConnections.dataForSynapse(synapse).presynapticCell
      for synapse in synapses)
    self.assertFalse(0 in presynapticCells)


  def testRecycleLeastRecentlyActiveSegmentToMakeRoomForNewSegment(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=1,
      activationThreshold=3,
      initialPermanence=.50,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=3,
      permanenceIncrement=.02,
      permanenceDecrement=.02,
      predictedSegmentDecrement=0.0,
      seed=42,
      maxSegmentsPerCell=2)

    prevActiveColumns1 = [0, 1, 2]
    prevActiveColumns2 = [3, 4, 5]
    prevActiveColumns3 = [6, 7, 8]
    activeColumns = [9]

    tm.compute(prevActiveColumns1)
    tm.compute(activeColumns)

    self.assertEqual(1, tm.basalConnections.numSegments(9))
    oldestSegment = list(tm.basalConnections.segmentsForCell(9))[0]
    tm.reset()
    tm.compute(prevActiveColumns2)
    tm.compute(activeColumns)

    self.assertEqual(2, tm.basalConnections.numSegments(9))

    oldPresynaptic = \
      set(tm.basalConnections.dataForSynapse(synapse).presynapticCell
          for synapse in tm.basalConnections.synapsesForSegment(oldestSegment))

    tm.reset()
    tm.compute(prevActiveColumns3)
    tm.compute(activeColumns)
    self.assertEqual(2, tm.basalConnections.numSegments(9))

    # Verify none of the segments are connected to the cells the old
    # segment was connected to.

    for segment in tm.basalConnections.segmentsForCell(9):
      newPresynaptic = set(tm.basalConnections.dataForSynapse(synapse).presynapticCell
                           for synapse
                           in tm.basalConnections.synapsesForSegment(segment))
      self.assertEqual([], list(oldPresynaptic & newPresynaptic))


  def testDestroySegmentsWithTooFewSynapsesToBeMatching(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.2,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.02,
      seed=42)

    prevActiveColumns = [0]
    prevActiveCells = [0, 1, 2, 3]
    activeColumns = [2]
    expectedActiveCell = 5

    matchingSegment = tm.basalConnections.createSegment(expectedActiveCell)
    tm.basalConnections.createSynapse(matchingSegment, prevActiveCells[0], .015)
    tm.basalConnections.createSynapse(matchingSegment, prevActiveCells[1], .015)
    tm.basalConnections.createSynapse(matchingSegment, prevActiveCells[2], .015)
    tm.basalConnections.createSynapse(matchingSegment, prevActiveCells[3], .015)

    tm.compute(prevActiveColumns)
    tm.compute(activeColumns)

    self.assertEqual(0, tm.basalConnections.numSegments(expectedActiveCell))


  def testPunishMatchingSegmentsInInactiveColumns(self):
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.2,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.02,
      seed=42)

    previousActiveColumns = [0]
    previousActiveCells = [0, 1, 2, 3]
    activeColumns = [1]
    previousInactiveCell = 81

    activeSegment = tm.basalConnections.createSegment(42)
    as1 = tm.basalConnections.createSynapse(activeSegment,
                                       previousActiveCells[0], .5)
    as2 = tm.basalConnections.createSynapse(activeSegment,
                                       previousActiveCells[1], .5)
    as3 = tm.basalConnections.createSynapse(activeSegment,
                                       previousActiveCells[2], .5)
    is1 = tm.basalConnections.createSynapse(activeSegment,
                                       previousInactiveCell, .5)

    matchingSegment = tm.basalConnections.createSegment(43)
    as4 = tm.basalConnections.createSynapse(matchingSegment,
                                       previousActiveCells[0], .5)
    as5 = tm.basalConnections.createSynapse(matchingSegment,
                                       previousActiveCells[1], .5)
    is2 = tm.basalConnections.createSynapse(matchingSegment,
                                       previousInactiveCell, .5)

    tm.compute(previousActiveColumns)
    tm.compute(activeColumns)

    self.assertAlmostEqual(.48, tm.basalConnections.dataForSynapse(as1).permanence)
    self.assertAlmostEqual(.48, tm.basalConnections.dataForSynapse(as2).permanence)
    self.assertAlmostEqual(.48, tm.basalConnections.dataForSynapse(as3).permanence)
    self.assertAlmostEqual(.48, tm.basalConnections.dataForSynapse(as4).permanence)
    self.assertAlmostEqual(.48, tm.basalConnections.dataForSynapse(as5).permanence)
    self.assertAlmostEqual(.50, tm.basalConnections.dataForSynapse(is1).permanence)
    self.assertAlmostEqual(.50, tm.basalConnections.dataForSynapse(is2).permanence)


  def testAddSegmentToCellWithFewestSegments(self):
    grewOnCell1 = False
    grewOnCell2 = False
    for seed in xrange(100):
      tm = self.constructTM(
        columnDimensions=[32],
        cellsPerColumn=4,
        activationThreshold=3,
        initialPermanence=.2,
        connectedPermanence=.50,
        minThreshold=2,
        maxNewSynapseCount=4,
        permanenceIncrement=.10,
        permanenceDecrement=.10,
        predictedSegmentDecrement=0.02,
        seed=seed)

      prevActiveColumns = [1, 2, 3, 4]
      activeColumns = [0]
      prevActiveCells = [4, 5, 6, 7]
      nonMatchingCells = [0, 3]
      activeCells = [0, 1, 2, 3]

      segment1 = tm.basalConnections.createSegment(nonMatchingCells[0])
      tm.basalConnections.createSynapse(segment1, prevActiveCells[0], .5)
      segment2 = tm.basalConnections.createSegment(nonMatchingCells[1])
      tm.basalConnections.createSynapse(segment2, prevActiveCells[1], .5)

      tm.compute(prevActiveColumns)
      tm.compute(activeColumns)

      self.assertEqual(activeCells, tm.getActiveCells())

      self.assertEqual(3, tm.basalConnections.numSegments())
      self.assertEqual(1, tm.basalConnections.numSegments(0))
      self.assertEqual(1, tm.basalConnections.numSegments(3))
      self.assertEqual(1, tm.basalConnections.numSynapses(segment1))
      self.assertEqual(1, tm.basalConnections.numSynapses(segment2))

      segments = list(tm.basalConnections.segmentsForCell(1))
      if len(segments) == 0:
        segments2 = list(tm.basalConnections.segmentsForCell(2))
        self.assertNotEquals(len(segments2), 0)
        grewOnCell2 = True
        segments.append(segments2[0])
      else:
        grewOnCell1 = True

      self.assertEqual(1, len(segments))
      synapses = list(tm.basalConnections.synapsesForSegment(segments[0]))
      self.assertEqual(4, len(synapses))

      columnChecklist = set(prevActiveColumns)

      for synapse in synapses:
        synapseData = tm.basalConnections.dataForSynapse(synapse)
        self.assertAlmostEqual(.2, synapseData.permanence)

        column = tm.columnForCell(synapseData.presynapticCell)
        self.assertTrue(column in columnChecklist)
        columnChecklist.remove(column)
      self.assertEquals(len(columnChecklist), 0)

    self.assertTrue(grewOnCell1)
    self.assertTrue(grewOnCell2)


  def testColumnForCell1D(self):
    tm = self.constructTM(
      columnDimensions=[2048],
      cellsPerColumn=5
    )
    self.assertEqual(tm.columnForCell(0), 0)
    self.assertEqual(tm.columnForCell(4), 0)
    self.assertEqual(tm.columnForCell(5), 1)
    self.assertEqual(tm.columnForCell(10239), 2047)


  def testColumnForCell2D(self):
    tm = self.constructTM(
      columnDimensions=[64, 64],
      cellsPerColumn=4
    )
    self.assertEqual(tm.columnForCell(0), 0)
    self.assertEqual(tm.columnForCell(3), 0)
    self.assertEqual(tm.columnForCell(4), 1)
    self.assertEqual(tm.columnForCell(16383), 4095)


  def testColumnForCellInvalidCell(self):
    tm = self.constructTM(
      columnDimensions=[64, 64],
      cellsPerColumn=4
    )

    try:
      tm.columnForCell(16383)
    except IndexError:
      self.fail("IndexError raised unexpectedly")

    args = [16384]
    self.assertRaises((IndexError, RuntimeError), tm.columnForCell, *args)

    args = [-1]
    self.assertRaises((IndexError, RuntimeError), tm.columnForCell, *args)


  def testCellsForColumn1D(self):
    tm = self.constructTM(
      columnDimensions=[2048],
      cellsPerColumn=5
    )
    expectedCells = [5, 6, 7, 8, 9]
    self.assertEqual(tm.cellsForColumn(1), expectedCells)


  def testCellsForColumn2D(self):
    tm = self.constructTM(
      columnDimensions=[64, 64],
      cellsPerColumn=4
    )
    expectedCells = [256, 257, 258, 259]
    self.assertEqual(tm.cellsForColumn(64), expectedCells)


  def testCellsForColumnInvalidColumn(self):
    tm = self.constructTM(
      columnDimensions=[64, 64],
      cellsPerColumn=4
    )

    try:
      tm.cellsForColumn(4095)
    except IndexError:
      self.fail("IndexError raised unexpectedly")

    args = [4096]
    self.assertRaises((IndexError, RuntimeError), tm.cellsForColumn, *args)

    args = [-1]
    self.assertRaises((IndexError, RuntimeError), tm.cellsForColumn, *args)


  def testNumberOfColumns(self):
    tm = self.constructTM(
      columnDimensions=[64, 64],
      cellsPerColumn=32
    )
    self.assertEqual(tm.numberOfColumns(), 64 * 64)


  def testNumberOfCells(self):
    tm = self.constructTM(
      columnDimensions=[64, 64],
      cellsPerColumn=32
    )
    self.assertEqual(tm.numberOfCells(), 64 * 64 * 32)
