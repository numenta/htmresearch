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
Extended Temporal Memory implementation in Python.
The static methods in this file use the following parameter ordering convention:
1. Output / mutated params
2. Traditional parameters to the function, i.e. the ones that would still exist
   if this function were a method on a class
3. Model state (not mutated)
4. Model parameters (including "learn")
"""

from collections import defaultdict
from itertools import imap, tee
import operator

from nupic.bindings.math import Random
from nupic.research.connections import binSearch, Connections
from nupic.support.group_by import groupby2



_EPSILON = 0.00001  # constant error threshold to check equality of permanences
                    # to other floats
_MIN_PREDICTIVE_THRESHOLD = 2



class ExtendedTemporalMemory(object):
  """ Class implementing the Temporal Memory algorithm with the added ability of
  being able to learn from both internal and external cell activation. This
  class has an option to learn on a single cell within a column and not
  look for a new winner cell until a reset() is called.
  """

  def __init__(self,
               columnDimensions=(2048,),
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               formInternalBasalConnections=True,
               learnOnOneCell=False,
               seed=42,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               **kwargs):
    """
    @param columnDimensions (list) Dimensions of the column space
    @param cellsPerColumn (int) Number of cells per column
    @param activationThreshold (int) If the number of active connected synapses
        on a segment is at least this threshold, the segment is said to be
        active.
    @param initialPermanence (float) Initial permanence of a new synapse
    @param connectedPermanence (float) If the permanence value for a synapse is
        greater than this value, it is said to be connected.
    @param minThreshold (int) If the number of potential synapses active on a
        segment is at least this threshold, it is said to be "matching" and is
        eligible for learning.
    @param maxNewSynapseCount (int)
        The maximum number of synapses added to a segment during learning.
    @param permanenceIncrement (float)
        Amount by which permanences of synapses are incremented during learning.
    @param permanenceDecrement (float)
    Amount by which permanences of synapses are decremented during learning.
    @param predictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.
    @param formInternalBasalConnections (boolean)
    If True, the winner cell for each column will be fixed between resets.
    @param learnOnOneCell (boolean)
    If True, the winner cell for each column will be fixed between resets.
    @param maxSegmentsPerCell
    The maximum number of segments per cell.
    @param maxSynapsesPerSegment
    The maximum number of synapses per segment.
    @param seed (int)
    Seed for the random number generator.

    Notes:
    predictedSegmentDecrement: A good value is just a bit larger than
    (the column-level sparsity * permanenceIncrement). So, if column-level
    sparsity is 2% and permanenceIncrement is 0.01, this parameter should be
    something like 4% * 0.01 = 0.0004).
    """
    # Error checking
    if not len(columnDimensions):
      raise ValueError("Number of column dimensions must be greater than 0")

    if cellsPerColumn <= 0:
      raise ValueError("Number of cells per column must be greater than 0")

    if minThreshold > activationThreshold:
      raise ValueError(
        "The min threshold can't be greater than the activation threshold")

    self.columnDimensions = columnDimensions
    self.cellsPerColumn = cellsPerColumn
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.maxNewSynapseCount = maxNewSynapseCount
    self.formInternalBasalConnections = formInternalBasalConnections
    self.learnOnOneCell = learnOnOnceCell
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement

    # Initialize connections
    self.basalConnections = Connections(
      self.numberOfCells(),
      maxSegmentsPerCell=maxSegmentsPerCell,
      maxSynapsesPerSegment=maxSynapsesPerSegment)
    self.apicalConnections = Connections(self.numberOfCells(),
      maxSegmentsPerCell=maxSegmentsPerCell,
      maxSynapsesPerSegment=maxSynapsesPerSegment)

    self._random = Random(seed)

    self.activeCells = []
    self.winnerCells = []
    self.activeSegments = []
    self.matchingSegments = []

    self.numActiveConnectedSynapsesForBasalSegment = []
    self.numActivePotentialSynapsesForBasalSegment = []
    self.numActiveConnectedSynapsesForApicalSegment = []
    self.numActivePotentialSynapsesForApicalSegment = []

    self.chosenCellForColumn = {}

  # ==============================
  # Main functions
  # ==============================

  def compute(self,
              activeColumns,
              activeExternalCellsBasal,
              activeExternalCellsApical,
              learn=True):
    """ Feeds input record through TM, performing inference and learning.

    @param activeColumns (iter)
    Indices of active columns
    @param activeExternalCellsBasal (iter)
    Sorted list of active external cells for activating basal dendrites at the
    end of this time step.
    @param activeExternalCellsApical (iter)
    Sorted list of active external cells for activating apical dendrites at the
    end of this time step.
    @param learn (bool)
    Whether or not learning is enabled
    """
    self.activateCells(sorted(activeColumns), learn)
    self.activateDendrites(learn, activeExternalCellsBasal,
                           activeExternalCellsApical)


  def activateCells(self, activeColumns, learn=True):
    """ Calculate the active cells, using the current active columns and
    dendrite segments. Grow and reinforce synapses.

    @param activeColumns (iter)
        A sorted list of active column indices.
    @param learn (bool)
        If true, reinforce/punish/grow synapses.
    """
    prevActiveExternalCellsBasal = self.activeExternalCellsBasal
    prevActiveExternalCellsApical = self.activeExternalCellsApical

    prevActiveCells = self.activeCells
    prevWinnerCells = self.winnerCells
    self.activeCells = []
    self.winnerCells = []

    columnForSegment = lambda segment: int(segment.cell / self.cellsPerColumn)

    # Walk the columns -- do all cell activation and learning for each column
    # before moving on to the next
    for columnData in groupby2(activeColumns, identity,
                               self.activeBasalSegments, columnForSegment,
                               self.matchingBasalSegments, columnForSegment,
                               self.activeApicalSegments, columnForSegment,
                               self.matchingApicalSegments, columnForSegment):
      (column,
       activeColumns,
       activeBasalSegmentsOnCol,
       matchingBasalSegmentsOnCol,
       activeApicalSegmentsOnCol,
       matchingApicalSegmentsOnCol) = columnData

      if activeColumns:
        maxPredictiveScore = 0

        for cellData in groupby2(activeBasalSegmentsOnCol, cellForSegment,
                                 activeApicalSegmentsOnCol, cellForSegment):
          (cell,
           activeBasalSegmentsOnCell,
           activeApicalSegmentsOnCell) = cellData

          maxPredictiveScore = max(maxPredictiveScore,
                                   predictiveScore(activeBasalSegmentsOnCell,
                                                   activeApicalSegmentsOnCell))

        if maxPredictiveScore >= _MIN_PREDICTIVE_THRESHOLD:
          cellsToAdd = self.activatePredictedColumn(
            self.basalConnections,
            self.apicalConnections,
            self._random,
            activeBasalSegmentsOnCol,
            matchingBasalSegmentsOnCol,
            activeApicalSegmentsOnCol,
            matchingApicalSegmentsOnCol,
            maxPredictiveScore,
            prevActiveCells,
            prevWinnerCells,
            prevActiveExternalCellsBasal,
            prevActiveExternalCellsApical
            self.numActivePotentialSynapsesForBasalSegment,
            self.numActivePotentialSynapsesForApicalSegment,
            self.maxNewSynapseCount,
            self.initialPermanence,
            self.permanenceIncrement,
            self.permanenceDecrement,
            self.formInternalBasalConnections,
            learn)

          self.activeCells += cellsToAdd
          self.winnerCells += cellsToAdd

        else:
          (cellsToAdd,
           winnerCell) = self.burstColumn(
             self.basalConnections,
             self.apicalConnections,
             self._random,
             self.chosenCellForColumn,
             column,
             activeBasalSegmentsOnCol,
             matchingBasalSegmentsOnCol,
             activeApicalSegmentsOnCol,
             matchingApicalSegmentsOnCol,
             prevActiveCells,
             prevWinnerCells,
             prevActiveExternalCellsBasal,
             prevActiveExternalCellsApical,
             self.numActivePotentialSynapsesForBasalSegment,
             self.numActivePotentialSynapsesForApicalSegment,
             self.cellsPerColumn,
             self.maxNewSynapseCount,
             self.initialPermanence,
             self.permanenceIncrement,
             self.permanenceDecrement,
             self.formInternalBasalConnections,
             self.learnOnOneCell,
             learn)

          self.activeCells += cellsToAdd
          self.winnerCells.append(winnerCell)

      else:
        if learn:
          ExtendedTemporalMemory.punishPredictedColumn(
            self.basalConnections,
            matchingBasalSegmentsOnCol,
            prevActiveCells, prevActiveExternalCellsBasal,
            self.predictedSegmentDecrement)

          # Don't punish apical segments.


  def activateDendrites(self, activeExternalCellsBasal,
                        activeExternalCellsApical, learn=True):
    """ Calculate dendrite segment activity, using the current active cells.

    @param activeExternalCellsBasal (iter)
    Sorted list of active external cells for activating basal dendrites.
    @param activeExternalCellsApical (iter)
    Sorted list of active external cells for activating apical dendrites.
    @param learn (bool)
    If true, segment activations will be recorded. This information is used
    during segment cleanup.
    """

    basal = ExtendedTemporalMemory.calculateExcitations(
      self.basalConnections, self.activeCells, activeExternalCellsBasal,
      self.connectedPermanence, self.activationThreshold, self.minThreshold,
      learn)

    (self.activeBasalSegments,
     self.matchingBasalSegments,
     self.numActiveConnectedSynapsesForBasalSegment,
     self.numActivePotentialSynapsesForBasalSegment) = basal

    apical = ExtendedTemporalMemory.calculateExcitations(
      self.apicalConnections, self.activeCells, activeExternalCellsApical,
      self.connectedPermanence, self.activationThreshold, self.minThreshold,
      learn)

    (self.activeApicalSegments,
     self.matchingApicalSegments,
     self.numActiveConnectedSynapsesForApicalSegment,
     self.numActivePotentialSynapsesForApicalSegment) = apical


  # TODO: make this an instance method -- https://github.com/numenta/nupic/issues/3309
  @staticmethod
  def activatePredictedColumn(basalConnections, apicalConnections, random,
                              columnActiveBasalSegments,
                              columnMatchingBasalSegments,
                              columnActiveApicalSegments,
                              columnMatchingApicalSegments,
                              predictiveThreshold,
                              prevActiveCells, prevWinnerCells,
                              prevActiveExternalCellsBasal,
                              prevActiveExternalCellsApical,
                              numActivePotentialSynapsesForBasalSegment,
                              numActivePotentialSynapsesForApicalSegment,
                              maxNewSynapseCount, initialPermanence,
                              permanenceIncrement, permanenceDecrement,
                              formInternalBasalConnections, learn):
    """
    @param connections (Object)
    Connections for the TM. Gets mutated.

    @param random (Object)
    Random number generator. Gets mutated.
    @param columnActiveSegments (iter)
    Active segments in this column.
    @param prevActiveCells (list)
    Active cells in `t-1`.
    @param prevWinnerCells (list)
    Winner cells in `t-1`.
    @param numActivePotentialSynapsesForSegment (list)
    Number of active potential synapses per segment, indexed by the segment's
    flatIdx.
    @param maxNewSynapseCount (int)
    The maximum number of synapses added to a segment during learning
    @param initialPermanence (float)
    Initial permanence of a new synapse.
    @permanenceIncrement (float)
    Amount by which permanences of synapses are incremented during learning.
    @permanenceDecrement (float)
    Amount by which permanences of synapses are decremented during learning.
    @param learn (bool)
    If true, grow and reinforce synapses.

    @return cellsToAdd (list)
    A list of predicted cells that will be added to active cells and winner
    cells.
    """
    cellsToAdd = []

    for cellData in groupby2(columnActiveBasalSegments, cellForSegment,
                             columnMatchingBasalSegments, cellForSegment,
                             columnActiveApicalSegments, cellForSegment,
                             columnMatchingApicalSegments, cellForSegment):
      (cell,
       cellActiveBasalSegments, cellMatchingBasalSegments,
       cellActiveApicalSegments, cellMatchingApicalSegments) = cellData

      if predictiveScore(cellActiveBasalSegments,
                         cellActiveApicalSegments) >= predictiveThreshold:
        cellsToAdd.append(cell)

        if learn:

          # Do basal
          internalBasalCandidates = (  # TODO: reformat this to be readable
            prevWinnerCells if formInternalBasalConnections
            else tuple())
          ExtendedTemporalMemory.learnOnCell(
            basalConnections,
            # random,
            cell,
            prevActiveCells,
            internalBasalCandidates,
            prevActiveExternalCellsBasal,
            cellActiveBasalSegments,
            cellMatchingBasalSegments,
            maxNewSynapseCount,
            initialPermanence,
            permanenceIncrement,
            permanenceDecrement,
            ## ??
            numActivePotentialSynapsesForBasalSegment)

          # Do apical
          internalApicalCandidates = tuple()
          ExtendedTemporalMemory.learnOnCell(
            apicalConnections, random,
            cell,
            cellActiveApicalSegments, cellMatchingApicalSegments,
            prevActiveCells,
            internalApicalCandidates, prevActiveExternalCellsApical,
            numActivePotentialSynapsesForApicalSegment,
            maxNewSynapseCount,
            initialPermanence, permanenceIncrement, permanenceDecrement)

    return cellsToAdd


  @staticmethod
  def learnOnCell(connections,
                  cell,
                  prevActiveCells,
                  internalCandidates,
                  externalCandidates,
                  cellActiveSegments,
                  cellMatchingSegments,
                  maxNewSynapseCount,
                  initialPermanence,
                  permanenceIncrement,
                  permanenceDecrement):
    """

    @param connections (Object)
    Connections for the TM. Gets mutated.
    """
    # TODO: use lambda rather than itertools?
    # bySegment = lambda segment: segment  #???
    # for segmentData in groupBy2(cellActiveSegments, bySegment,
    #                             cellMatchingSegments, bySegment):
    #   # (segment,
    #   #  x, y, z) = segmentData  # what is unpacked here?
    #   segment = segmentData[0]
    for segment in itertools.chain(cellActiveSegments, cellMatchingSegments):
    # for segment in cellActiveSegments:

      TemporalMemory.adaptSegment(connections,
                                  segment,
                                  prevActiveCells,
                                  externalCandidates,
                                  permanenceIncrement,
                                  permanenceDecrement)

      active = numActivePotentialSynapsesForSegment[segment.flatIdx]
      numGrowDesired = maxNewSynapseCount - active

      if numGrowDesired > 0:
        TemporalMemory.growSynapses(connections,
                                    segment,
                                    numGrowDesired,
                                    internalCandidates,
                                    externalCandidates,
                                    initialPermanence)


  # TODO: porting from https://github.com/numenta/nupic.research/blob/master/htmresearch/algorithms/temporal_memory_phases.py#L549
  @staticmethod
  def adaptSegment(connections,
                   segment,
                   prevActiveCells,
                   externalCandidates,
                   permanenceIncrement,
                   permanenceDecrement):
    """
    Updates synapses on segment.
    Strengthens active synapses; weakens inactive synapses.
    @param segment              (int)         Segment index
    @param activeSynapses       (set)         Indices of active synapses
    @param connections          (Connections) Connectivity of layer
    @param permanenceIncrement  (float)  Amount to increment active synapses
    @param permanenceDecrement  (float)  Amount to decrement inactive synapses
    """
    for synapse in set(connections.synapsesForSegment(segment)):
      synapseData = connections.dataForSynapse(synapse)
      permanence = synapseData.permanence

      if synapse in activeSynapses:
        permanence += permanenceIncrement
      else:
        permanence -= permanenceDecrement

      # Keep permanence within min/max bounds
      permanence = max(0.0, min(1.0, permanence))

      if (permanence < EPSILON):
        connections.destroySynapse(synapse)
      else:
        connections.updateSynapsePermanence(synapse, permanence)


  @staticmethod
  def growSynapses():
    return



  # TODO: make this an instance method -- https://github.com/numenta/nupic/issues/3309
  @staticmethod
  def burstColumn(connections, random, column, columnMatchingSegments,
                  prevActiveCells, prevWinnerCells,
                  numActivePotentialSynapsesForSegment, cellsPerColumn,
                  maxNewSynapseCount, initialPermanence, permanenceIncrement,
                  permanenceDecrement, learn):
    """
    @param connections (Object)
    Connections for the TM. Gets mutated.
    @param random (Object)
    Random number generator. Gets mutated.
    @param column (int)
    Index of bursting column.
    @param columnMatchingSegments (iter)
    Matching segments in this column.
    @param prevActiveCells (list)
    Active cells in `t-1`.
    @param prevWinnerCells (list)
    Winner cells in `t-1`.
    @param numActivePotentialSynapsesForSegment (list)
    Number of active potential synapses per segment, indexed by the segment's
    flatIdx.
    @param cellsPerColumn (int)
    Number of cells per column.
    @param maxNewSynapseCount (int)
    The maximum number of synapses added to a segment during learning.
    @param initialPermanence (float)
    Initial permanence of a new synapse.
    @param permanenceIncrement (float)
    Amount by which permanences of synapses are incremented during learning.
    @param permanenceDecrement (float)
    Amount by which permanences of synapses are decremented during learning.
    @param learn (bool)
    Whether or not learning is enabled.
    @return (tuple) Contains:
                      `cells`         (iter),
                      `winnerCell`    (int),
    """
    start = cellsPerColumn * column
    cells = xrange(start, start + cellsPerColumn)

    if columnMatchingSegments is not None:
      numActive = lambda s: numActivePotentialSynapsesForSegment[s.flatIdx]
      bestMatchingSegment = max(columnMatchingSegments, key=numActive)
      winnerCell = bestMatchingSegment.cell

      if learn:
        TemporalMemory.adaptSegment(connections, bestMatchingSegment,
                                    prevActiveCells, permanenceIncrement,
                                    permanenceDecrement)

        nGrowDesired = maxNewSynapseCount - numActive(bestMatchingSegment)

        if nGrowDesired > 0:
          TemporalMemory.growSynapses(connections, random, bestMatchingSegment,
                                      nGrowDesired, prevWinnerCells,
                                      initialPermanence)
    else:
      winnerCell = TemporalMemory.leastUsedCell(random, cells, connections)
      if learn:
        nGrowExact = min(maxNewSynapseCount, len(prevWinnerCells))
        if nGrowExact > 0:
          segment = connections.createSegment(winnerCell)
          TemporalMemory.growSynapses(connections, random, segment,
                                      nGrowExact, prevWinnerCells,
                                      initialPermanence)

    return cells, winnerCell


  # TODO: make this an instance method -- https://github.com/numenta/nupic/issues/3309
  @staticmethod
  def punishPredictedColumn(connections, columnMatchingSegments,
                            predictedSegmentDecrement, prevActiveCells):
    """
    @param connections (Object)
    Connections for the TM. Gets mutated.
    @param columnMatchingSegments (iter)
    Matching segments for this column.
    @param predictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.
    @param prevActiveCells (list)
    Active cells in `t-1`.
    """
    if predictedSegmentDecrement > 0.0 and columnMatchingSegments is not None:
      for segment in columnMatchingSegments:
        TemporalMemory.adaptSegment(connections, segment,
                                    prevActiveCells,
                                    -predictedSegmentDecrement, 0.0)


  def reset(self):
    """ Indicates the start of a new sequence and resets the sequence
        state of the TM. """
    self.activeCells = []
    self.winnerCells = []
    self.activeBasalSegments = []
    self.matchingBasalSegments = []
    self.activeApicalSegments = []
    self.matchingApicalSegments = []
    self.chosenCellForColumn = {}

  # ==============================
  # Helper functions
  # ==============================

  @staticmethod
  def calculateExcitations(connections, activeCells, activeExternalCells,
                           connectedPermanence, activationThreshold,
                           minThreshold, learn):

    offset = connections.numCells()
    allActiveCells = itertools.chain(activeCells,
                                     (c + offset for c in activeExternalCells))

    (numActiveConnected,
     numActivePotential) = connections.computeActivity(
       allActiveCells,
       connectedPermanence)

    activeSegments = (
      connections.segmentForFlatIdx(i)
      for i in xrange(len(numActiveConnected))
      if numActiveConnected[i] >= activationThreshold
    )

    matchingSegments = (
      connections.segmentForFlatIdx(i)
      for i in xrange(len(numActivePotential))
      if numActivePotential[i] >= minThreshold
    )

    maxSegmentsPerCell = connections.maxSegmentsPerCell
    segmentKey = lambda segment: (segment.cell * maxSegmentsPerCell
                                  + segment.idx)

    if learn:
      for segment in activeSegments:
        connections.recordSegmentActivity(segment)
      connections.startNewIteration()

    return (sorted(activeSegments, key = segmentKey),
            sorted(matchingSegments, key = segmentKey),
            numActiveConnected,
            numActivePotential)


  @staticmethod
  def leastUsedCell(random, cells, connections):
    """ Gets the cell with the smallest number of segments.
    Break ties randomly.
    @param random (Object)
    Random number generator. Gets mutated.
    @param cells (list)
    Indices of cells.
    @param connections (Object)
    Connections instance for the TM.
    @return (int) Cell index.
    """
    leastUsedCells = []
    minNumSegments = float("inf")
    for cell in cells:
      numSegments = connections.numSegments(cell)

      if numSegments < minNumSegments:
        minNumSegments = numSegments
        leastUsedCells = []

      if numSegments == minNumSegments:
        leastUsedCells.append(cell)

    i = random.getUInt32(len(leastUsedCells))
    return leastUsedCells[i]


  @staticmethod
  def growSynapses(connections, random, segment, nDesiredNewSynapes,
                   prevWinnerCells, initialPermanence):
    """ Creates nDesiredNewSynapes synapses on the segment passed in if
    possible, choosing random cells from the previous winner cells that are
    not already on the segment.
    @param  connections        (Object) Connections instance for the tm
    @param  random             (Object) Tm object used to generate random
                                        numbers
    @param  segment            (int)    Segment to grow synapses on.
    @params nDesiredNewSynapes (int)    Desired number of synapses to grow
    @params prevWinnerCells    (list)   Winner cells in `t-1`
    @param  initialPermanence  (float)  Initial permanence of a new synapse.
    Notes: The process of writing the last value into the index in the array
    that was most recently changed is to ensure the same results that we get
    in the c++ implentation using iter_swap with vectors.
    """
    candidates = list(prevWinnerCells)
    eligibleEnd = len(candidates) - 1

    for synapse in connections.synapsesForSegment(segment):
      try:
        index = candidates[:eligibleEnd + 1].index(synapse.presynapticCell)
      except ValueError:
        index = -1
      if index != -1:
        candidates[index] = candidates[eligibleEnd]
        eligibleEnd -= 1

    candidatesLength = eligibleEnd + 1
    nActual = min(nDesiredNewSynapes, candidatesLength)

    for _ in range(nActual):
      rand = random.getUInt32(candidatesLength)
      connections.createSynapse(segment, candidates[rand],
                                initialPermanence)
      candidates[rand] = candidates[candidatesLength - 1]
      candidatesLength -= 1


  @staticmethod
  def adaptSegment(connections, segment, prevActiveCells, permanenceIncrement,
                   permanenceDecrement):
    """ Updates synapses on segment.
    Strengthens active synapses; weakens inactive synapses.
    @param connections          (Object) Connections instance for the tm
    @param segment              (int)    Segment to adapt
    @param prevActiveCells      (list)   Active cells in `t-1`
    @param permanenceIncrement  (float)  Amount to increment active synapses
    @param permanenceDecrement  (float)  Amount to decrement inactive synapses
    """

    for synapse in connections.synapsesForSegment(segment):
      permanence = synapse.permanence

      if binSearch(prevActiveCells, synapse.presynapticCell) != -1:
        permanence += permanenceIncrement
      else:
        permanence -= permanenceDecrement

      # Keep permanence within min/max bounds
      permanence = max(0.0, min(1.0, permanence))

      if permanence < _EPSILON:
        connections.destroySynapse(synapse)
      else:
        connections.updateSynapsePermanence(synapse, permanence)

    if connections.numSynapses(segment) == 0:
      connections.destroySegment(segment)


  def columnForCell(self, cell):
    """ Returns the index of the column that a cell belongs to.
    @param cell (int) Cell index
    @return (int) Column index
    """
    self._validateCell(cell)

    return int(cell / self.cellsPerColumn)


  def cellsForColumn(self, column):
    """ Returns the indices of cells that belong to a column.
    @param column (int) Column index
    @return (list) Cell indices
    """
    self._validateColumn(column)

    start = self.cellsPerColumn * column
    end = start + self.cellsPerColumn
    return range(start, end)


  def numberOfColumns(self):
    """ Returns the number of columns in this layer.
    @return (int) Number of columns
    """
    return reduce(operator.mul, self.columnDimensions, 1)


  def numberOfCells(self):
    """ Returns the number of cells in this layer.
    @return (int) Number of cells
    """
    return self.numberOfColumns() * self.cellsPerColumn


  def mapCellsToColumns(self, cells):
    """ Maps cells to the columns they belong to
    @param cells (set) Cells
    @return (dict) Mapping from columns to their cells in `cells`
    """
    cellsForColumns = defaultdict(set)

    for cell in cells:
      column = self.columnForCell(cell)
      cellsForColumns[column].add(cell)

    return cellsForColumns


  def getActiveCells(self):
    """ Returns the indices of the active cells.
    @return (list) Indices of active cells.
    """
    return self.getCellIndices(self.activeCells)


  def getPredictiveCells(self):
    """ Returns the indices of the predictive cells.
    @return (list) Indices of predictive cells.
    """
    previousCell = None
    predictiveCells = []
    for segment in self.activeSegments:
      if segment.cell != previousCell:
        predictiveCells.append(segment.cell)
        previousCell = segment.cell

    return predictiveCells


  def getWinnerCells(self):
    """ Returns the indices of the winner cells.
    @return (list) Indices of winner cells.
    """
    return self.getCellIndices(self.winnerCells)


  def getCellsPerColumn(self):
    """ Returns the number of cells per column.
    @return (int) The number of cells per column.
    """
    return self.cellsPerColumn


  def getColumnDimensions(self):
    """
    Returns the dimensions of the columns in the region.
    @return (tuple) Column dimensions
    """
    return self.columnDimensions


  def getActivationThreshold(self):
    """
    Returns the activation threshold.
    @return (int) The activation threshold.
    """
    return self.activationThreshold


  def setActivationThreshold(self, activationThreshold):
    """
    Sets the activation threshold.
    @param activationThreshold (int) activation threshold.
    """
    self.activationThreshold = activationThreshold


  def getInitialPermanence(self):
    """
    Get the initial permanence.
    @return (float) The initial permanence.
    """
    return self.initialPermanence


  def setInitialPermanence(self, initialPermanence):
    """
    Sets the initial permanence.
    @param initialPermanence (float) The initial permanence.
    """
    self.initialPermanence = initialPermanence


  def getMinThreshold(self):
    """
    Returns the min threshold.
    @return (int) The min threshold.
    """
    return self.minThreshold


  def setMinThreshold(self, minThreshold):
    """
    Sets the min threshold.
    @param minThreshold (int) min threshold.
    """
    self.minThreshold = minThreshold


  def getMaxNewSynapseCount(self):
    """
    Returns the max new synapse count.
    @return (int) The max new synapse count.
    """
    return self.maxNewSynapseCount


  def setMaxNewSynapseCount(self, maxNewSynapseCount):
    """
    Sets the max new synapse count.
    @param maxNewSynapseCount (int) Max new synapse count.
    """
    self.maxNewSynapseCount = maxNewSynapseCount


  def getPermanenceIncrement(self):
    """
    Get the permanence increment.
    @return (float) The permanence increment.
    """
    return self.permanenceIncrement


  def setPermanenceIncrement(self, permanenceIncrement):
    """
    Sets the permanence increment.
    @param permanenceIncrement (float) The permanence increment.
    """
    self.permanenceIncrement = permanenceIncrement


  def getPermanenceDecrement(self):
    """
    Get the permanence decrement.
    @return (float) The permanence decrement.
    """
    return self.permanenceDecrement


  def setPermanenceDecrement(self, permanenceDecrement):
    """
    Sets the permanence decrement.
    @param permanenceDecrement (float) The permanence decrement.
    """
    self.permanenceDecrement = permanenceDecrement


  def getPredictedSegmentDecrement(self):
    """
    Get the predicted segment decrement.
    @return (float) The predicted segment decrement.
    """
    return self.predictedSegmentDecrement


  def setPredictedSegmentDecrement(self, predictedSegmentDecrement):
    """
    Sets the predicted segment decrement.
    @param predictedSegmentDecrement (float) The predicted segment decrement.
    """
    self.predictedSegmentDecrement = predictedSegmentDecrement


  def getConnectedPermanence(self):
    """
    Get the connected permanence.
    @return (float) The connected permanence.
    """
    return self.connectedPermanence


  def setConnectedPermanence(self, connectedPermanence):
    """
    Sets the connected permanence.
    @param connectedPermanence (float) The connected permanence.
    """
    self.connectedPermanence = connectedPermanence


  def write(self, proto):
    """ Writes serialized data to proto object
    @param proto (DynamicStructBuilder) Proto object
    """
    proto.columnDimensions = self.columnDimensions
    proto.cellsPerColumn = self.cellsPerColumn
    proto.activationThreshold = self.activationThreshold
    proto.initialPermanence = self.initialPermanence
    proto.connectedPermanence = self.connectedPermanence
    proto.minThreshold = self.minThreshold
    proto.maxNewSynapseCount = self.maxNewSynapseCount
    proto.permanenceIncrement = self.permanenceIncrement
    proto.permanenceDecrement = self.permanenceDecrement
    proto.predictedSegmentDecrement = self.predictedSegmentDecrement

    self.connections.write(proto.connections)
    self._random.write(proto.random)

    proto.activeCells = list(self.activeCells)
    proto.winnerCells = list(self.winnerCells)
    activeSegmentOverlaps = \
        proto.init('activeSegmentOverlaps', len(self.activeSegments))
    for i, segment in enumerate(self.activeSegments):
      activeSegmentOverlaps[i].cell = segment.cell
      activeSegmentOverlaps[i].segment = segment.idx
      activeSegmentOverlaps[i].overlap = (
        self.numActiveConnectedSynapsesForSegment[segment.flatIdx]
      )

    matchingSegmentOverlaps = \
        proto.init('matchingSegmentOverlaps', len(self.matchingSegments))
    for i, segment in enumerate(self.matchingSegments):
      matchingSegmentOverlaps[i].cell = segment.cell
      matchingSegmentOverlaps[i].segment = segment.idx
      matchingSegmentOverlaps[i].overlap = (
        self.numActivePotentialSynapsesForSegment[segment.flatIdx]
      )



  @classmethod
  def read(cls, proto):
    """ Reads deserialized data from proto object
    @param proto (DynamicStructBuilder) Proto object
    @return (TemporalMemory) TemporalMemory instance
    """
    tm = object.__new__(cls)

    tm.columnDimensions = list(proto.columnDimensions)
    tm.cellsPerColumn = int(proto.cellsPerColumn)
    tm.activationThreshold = int(proto.activationThreshold)
    tm.initialPermanence = proto.initialPermanence
    tm.connectedPermanence = proto.connectedPermanence
    tm.minThreshold = int(proto.minThreshold)
    tm.maxNewSynapseCount = int(proto.maxNewSynapseCount)
    tm.permanenceIncrement = proto.permanenceIncrement
    tm.permanenceDecrement = proto.permanenceDecrement
    tm.predictedSegmentDecrement = proto.predictedSegmentDecrement

    tm.connections = Connections.read(proto.connections)
    #pylint: disable=W0212
    tm._random = Random()
    tm._random.read(proto.random)
    #pylint: enable=W0212

    tm.activeCells = [int(x) for x in proto.activeCells]
    tm.winnerCells = [int(x) for x in proto.winnerCells]

    flatListLength = tm.connections.segmentFlatListLength()
    tm.numActiveConnectedSynapsesForSegment = [0] * flatListLength
    tm.numActivePotentialSynapsesForSegment = [0] * flatListLength

    tm.activeSegments = []
    tm.matchingSegments = []

    for i in xrange(len(proto.activeSegmentOverlaps)):
      protoSegmentOverlap = proto.activeSegmentOverlaps[i]

      segment = tm.connections.getSegment(protoSegmentOverlap.cell,
                                          protoSegmentOverlap.segment)
      tm.activeSegments.append(segment)

      overlap = protoSegmentOverlap.overlap
      tm.numActiveConnectedSynapsesForSegment[segment.flatIdx] = overlap

    for i in xrange(len(proto.matchingSegmentOverlaps)):
      protoSegmentOverlap = proto.matchingSegmentOverlaps[i]

      segment = tm.connections.getSegment(protoSegmentOverlap.cell,
                                          protoSegmentOverlap.segment)
      tm.matchingSegments.append(segment)

      overlap = protoSegmentOverlap.overlap
      tm.numActivePotentialSynapsesForSegment[segment.flatIdx] = overlap

    return tm


  def __eq__(self, other):
    """ Equality operator for TemporalMemory instances.
    Checks if two instances are functionally identical
    (might have different internal state).
    @param other (TemporalMemory) TemporalMemory instance to compare to
    """
    if self.columnDimensions != other.columnDimensions:
      return False
    if self.cellsPerColumn != other.cellsPerColumn:
      return False
    if self.activationThreshold != other.activationThreshold:
      return False
    if abs(self.initialPermanence - other.initialPermanence) > _EPSILON:
      return False
    if abs(self.connectedPermanence - other.connectedPermanence) > _EPSILON:
      return False
    if self.minThreshold != other.minThreshold:
      return False
    if self.maxNewSynapseCount != other.maxNewSynapseCount:
      return False
    if abs(self.permanenceIncrement - other.permanenceIncrement) > _EPSILON:
      return False
    if abs(self.permanenceDecrement - other.permanenceDecrement) > _EPSILON:
      return False
    if abs(self.predictedSegmentDecrement -
           other.predictedSegmentDecrement) > _EPSILON:
      return False

    if self.connections != other.connections:
      return False
    if self.activeCells != other.activeCells:
      return False
    if self.winnerCells != other.winnerCells:
      return False

    if self.matchingSegments != other.matchingSegments:
      return False
    if self.activeSegments != other.activeSegments:
      return False

    return True


  def __ne__(self, other):
    """ Non-equality operator for TemporalMemory instances.
    Checks if two instances are not functionally identical
    (might have different internal state).
    @param other (TemporalMemory) TemporalMemory instance to compare to
    """
    return not self.__eq__(other)


  def _validateColumn(self, column):
    """ Raises an error if column index is invalid.
    @param column (int) Column index
    """
    if column >= self.numberOfColumns() or column < 0:
      raise IndexError("Invalid column")


  def _validateCell(self, cell):
    """ Raises an error if cell index is invalid.
    @param cell (int) Cell index
    """
    if cell >= self.numberOfCells() or cell < 0:
      raise IndexError("Invalid cell")


  @classmethod
  def getCellIndices(cls, cells):
    """ Returns the indices of the cells passed in.
    @param cells (list) cells to find the indices of
    """
    return [cls.getCellIndex(c) for c in cells]


  @staticmethod
  def getCellIndex(cell):
    """ Returns the index of the cell
    @param cell (int) cell to find the index of
    """
    return cell


def isSortedWithoutDuplicates(iterable):
  """ Returns True if the input is sorted and contains no duplicates.
  @param iterable (iter)
  @return (bool)
  """
  a, b = itertools.tee(iterable)
  next(b, None)
  return all(itertools.imap(operator.lt, a, b))


def identity(x):
  return x


def cellForSegment(segment):
  return segment.cell


def predictiveScore(activeBasalSegmentsOnCell, activeApicalSegmentsOnCell):
  score = 0

  if activeBasalSegmentsOnCell is not None:
    score += 2

  if activeApicalSegmentsOnCell is not None:
    score += 1

  return score