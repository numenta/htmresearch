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
General Temporal Memory implementation in Python.
"""

from collections import defaultdict
from sensorimotor.general_temporal_memory import GeneralTemporalMemory



class OrphanTemporalMemory(GeneralTemporalMemory):
  """
  Class implementing the General Temporal Memory algorithm with the added
  ability to weaken orphan segments. An orphan segment is one where
  it had some reasonably high input activity but the cell did not
  become active on the next time step.
  """

  # ==============================
  # Main functions
  # ==============================


  def __init__(self,
               permanenceOrphanDecrement = 0.004,
               **kwargs):
    """
    @param permanenceOrphanDecrement (float) Amount by which
           active permanences of orphan segments are decremented.
    """

    super(OrphanTemporalMemory, self).__init__(**kwargs)
    self.permanenceOrphanDecrement = permanenceOrphanDecrement
    self.matchingSegments = set()
    self.matchingCells = set()


  def compute(self,
              activeColumns,
              activeExternalCells=None,
              formInternalConnections=True,
              learn=True):
    """
    Feeds input record through TM, performing inference and learning.
    Updates member variables with new state.

    @param activeColumns           (set)     Indices of active columns in `t`
    @param activeExternalCells     (set)     Indices of active external inputs in `t`
    @param formInternalConnections (boolean) Flag to determine whether to form connections with internal cells within this temporal memory
    """

    if not activeExternalCells:
      activeExternalCells = set()

    activeExternalCells = self._reindexActiveExternalCells(activeExternalCells)

    (activeCells,
     winnerCells,
     activeSegments,
     predictiveCells,
     predictedColumns,
     chosenCellForColumn,
     matchingSegments,
     matchingCells) = self.computeFn(activeColumns,
                                           activeExternalCells,
                                           self.activeExternalCells,
                                           self.predictiveCells,
                                           self.activeSegments,
                                           self.activeCells,
                                           self.matchingSegments,
                                           self.matchingCells,
                                           self.winnerCells,
                                           self.connections,
                                           formInternalConnections,
                                           self.learnOnOneCell,
                                           self.chosenCellForColumn,
                                           learn=learn)

    self.activeExternalCells = activeExternalCells

    self.unpredictedActiveColumns = activeColumns - predictedColumns
    self.predictedActiveCells = self.predictiveCells & activeCells

    self.activeCells = activeCells
    self.winnerCells = winnerCells
    self.activeSegments = activeSegments
    self.predictiveCells = predictiveCells
    self.chosenCellForColumn = chosenCellForColumn
    self.matchingSegments = matchingSegments
    self.matchingCells = matchingCells


  def computeFn(self,
                activeColumns,
                activeExternalCells,
                prevActiveExternalCells,
                prevPredictiveCells,
                prevActiveSegments,
                prevActiveCells,
                prevMatchingSegments,
                prevMatchingCells,
                prevWinnerCells,
                connections,
                formInternalConnections,
                learnOnOneCell,
                chosenCellForColumn,
                learn=True):
    """
    'Functional' version of compute.
    Returns new state.

    @param activeColumns                   (set)         Indices of active columns in `t`
    @param activeExternalCells             (set)         Indices of active external cells in `t`
    @param prevActiveExternalCells         (set)         Indices of active external cells in `t-1`
    @param prevPredictiveCells             (set)         Indices of predictive cells in `t-1`
    @param prevActiveSegments              (set)         Indices of active segments in `t-1`
    @param prevActiveCells                 (set)         Indices of active cells in `t-1`
    @param prevWinnerCells                 (set)         Indices of winner cells in `t-1`
    @param connections                     (Connections) Connectivity of layer
    @param formInternalConnections         (boolean)     Flag to determine whether to form connections with internal cells within this temporal memory
    @param learnOnOneCell                  (boolean)     If True, the winner cell for each column will be fixed between resets.
    @param chosenCellForColumn             (dict)        The current winner cell for each column, if it exists.

    @return (tuple) Contains:
                      `activeCells`               (set),
                      `winnerCells`               (set),
                      `activeSegments`            (set),
                      `predictiveCells`           (set)
    """
    activeCells = set()
    winnerCells = set()

    (_activeCells,
     _winnerCells,
     predictedColumns,
     orphanCells) = self.activateCorrectlyPredictiveCells(
       prevPredictiveCells,
       prevMatchingCells,
       activeColumns)

    # if orphanCells:
    #   print "orphan columns=",self.mapCellsToColumns(orphanCells)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    (_activeCells,
     _winnerCells,
     learningSegments,
     chosenCellForColumn) = self.burstColumns(
       activeColumns,
       predictedColumns,
       prevActiveCells | prevActiveExternalCells,
       prevWinnerCells,
       learnOnOneCell,
       chosenCellForColumn,
       connections)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    if learn:
      prevCellActivity = prevActiveExternalCells

      if formInternalConnections:
        prevCellActivity.update(prevWinnerCells)

      self.learnOnSegments(prevActiveSegments,
                           learningSegments,
                           prevActiveCells | prevActiveExternalCells,
                           winnerCells,
                           prevCellActivity,
                           connections,
                           orphanCells,
                           prevMatchingSegments)

    (activeSegments,
     predictiveCells,
     matchingSegments,
     matchingCells) = self.computePredictiveCells(
       activeCells | activeExternalCells, connections)

    # print "Matching segments:"
    # for segment in matchingSegments:
    #   self.printSegment(segment, connections)


    return (activeCells,
            winnerCells,
            activeSegments,
            predictiveCells,
            predictedColumns,
            chosenCellForColumn,
            matchingSegments,
            matchingCells)


  def activateCorrectlyPredictiveCells(self,
                                       prevPredictiveCells,
                                       prevMatchingCells,
                                       activeColumns):
    """
    Phase 1: Activate the correctly predictive cells.

    Pseudocode:

      - for each prev predictive cell
        - if in active column
          - mark it as active
          - mark it as winner cell
          - mark column as predicted
        - if not in active column
          - mark it as an orphan cell

    @param prevPredictiveCells (set) Indices of predictive cells in `t-1`
    @param activeColumns       (set) Indices of active columns in `t`

    @return (tuple) Contains:
                      `activeCells`      (set),
                      `winnerCells`      (set),
                      `predictedColumns` (set),
                      `orphanCells`      (set)
    """
    activeCells = set()
    winnerCells = set()
    predictedColumns = set()
    orphanCells = set()

    for cell in prevPredictiveCells:
      column = self.columnForCell(cell)

      if column in activeColumns:
        activeCells.add(cell)
        winnerCells.add(cell)
        predictedColumns.add(column)
      elif cell in prevMatchingCells:
        orphanCells.add(cell)

    # If any cell was previously matching and we don't get bottom up activity
    # in that column, mark it as an orphan cell.
    for cell in prevMatchingCells:
      column = self.columnForCell(cell)

      if not (column in activeColumns):
        orphanCells.add(cell)

    return activeCells, winnerCells, predictedColumns, orphanCells


  def learnOnSegments(self,
                      prevActiveSegments,
                      learningSegments,
                      prevActiveCells,
                      winnerCells,
                      prevWinnerCells,
                      connections,
                      orphanCells,
                      prevMatchingSegments):
    """
    Phase 3: Perform learning by adapting segments.

    Pseudocode:

      - (learning) for each prev active or learning segment
        - if learning segment or from winner cell
          - strengthen active synapses
          - weaken inactive synapses
        - if learning segment
          - add some synapses to the segment
            - subsample from prev winner cells

      - for each previously matching segment
        - if cell is an orphan cell
          - weaken active synapses but don't touch inactive synapses

    @param prevActiveSegments           (set)         Indices of active segments in `t-1`
    @param learningSegments             (set)         Indices of learning segments in `t`
    @param prevActiveCells              (set)         Indices of active cells in `t-1`
    @param winnerCells                  (set)         Indices of winner cells in `t`
    @param prevWinnerCells              (set)         Indices of winner cells in `t-1`
    @param connections                  (Connections) Connectivity of layer
    @param orphanCells                  (set)         Indices of orphan cells
    @param prevMatchingSegments         (set)         Indices of segments with
                                                      previous reasonable input
    """
    for segment in prevActiveSegments | learningSegments:
      isLearningSegment = segment in learningSegments
      cellForSegment = connections.cellForSegment(segment)
      isFromWinnerCell = cellForSegment in winnerCells

      activeSynapses = self.activeSynapsesForSegment(
        segment, prevActiveCells, connections)

      if isLearningSegment or isFromWinnerCell:
        self.adaptSegment(segment, activeSynapses, connections,
                          self.permanenceIncrement,
                          self.permanenceDecrement)

      if isLearningSegment:
        n = self.maxNewSynapseCount - len(activeSynapses)

        for presynapticCell in self.pickCellsToLearnOn(n,
                                                       segment,
                                                       prevWinnerCells,
                                                       connections):
          connections.createSynapse(segment,
                                    presynapticCell,
                                    self.initialPermanence)

    # Decrement segments that had some input previously but where the cells
    # were marked as orphans
    for segment in prevMatchingSegments:
      isOrphanCell = connections.cellForSegment(segment) in orphanCells

      activeSynapses = self.activeSynapsesForSegment(
        segment, prevActiveCells, connections)

      if isOrphanCell:
        # print "\nHandling orphan cell"
        # print "Before:"
        # self.printSegment(segment, connections)
        self.adaptSegment(segment, activeSynapses, connections,
                          -self.permanenceOrphanDecrement,
                          0.0)
        # print "After:"
        # self.printSegment(segment, connections)
        # print "========================================="


  def computePredictiveCells(self, activeCells, connections):
    """
    Phase 4: Compute predictive cells due to lateral input
    on distal dendrites.

    Pseudocode:

      - for each distal dendrite segment with activity >= activationThreshold
        - mark the segment as active
        - mark the cell as predictive

      - for each distal dendrite segment with unconnected
      activity >= minThreshold
        - mark the segment as matching
        - mark the cell as matching

    Forward propagates activity from active cells to the synapses that touch
    them, to determine which synapses are active.

    @param activeCells (set)         Indices of active cells in `t`
    @param connections (Connections) Connectivity of layer

    @return (tuple) Contains:
                      `activeSegments`   (set),
                      `predictiveCells`  (set),
                      `matchingSegments` (set),
                      `matchingCells`    (set)
    """
    numActiveConnectedSynapsesForSegment = defaultdict(lambda: 0)
    numActiveSynapsesForSegment = defaultdict(lambda: 0)
    activeSegments = set()
    predictiveCells = set()

    matchingSegments = set()
    matchingCells = set()

    for cell in activeCells:
      for synapseData in connections.synapsesForPresynapticCell(cell).values():
        segment = synapseData.segment
        permanence = synapseData.permanence

        if permanence >= self.connectedPermanence:
          numActiveConnectedSynapsesForSegment[segment] += 1

          if (numActiveConnectedSynapsesForSegment[segment] >=
              self.activationThreshold):
            activeSegments.add(segment)
            predictiveCells.add(connections.cellForSegment(segment))

        # See how many segments have weak activity, defined as unconnected
        # active segments >= minThreshold
        numActiveSynapsesForSegment[segment] += 1

        if numActiveSynapsesForSegment[segment] >= self.minThreshold:
          matchingSegments.add(segment)
          matchingCells.add(connections.cellForSegment(segment))

    return activeSegments, predictiveCells, matchingSegments, matchingCells


  def adaptSegment(self, segment, activeSynapses, connections,
                   permanenceIncrement, permanenceDecrement):
    """
    Updates synapses on segment.
    Strengthens active synapses; weakens inactive synapses.

    @param segment        (int)         Segment index
    @param activeSynapses (set)         Indices of active synapses
    @param connections    (Connections) Connectivity of layer
    @param permanenceIncrement (float)  Amount to increment active synapses
    @param permanenceDecrement (float)  Amount to decrement inactive synapses
    """
    synapses = connections.synapsesForSegment(segment)
    for synapse in synapses:
      synapseData = connections.dataForSynapse(synapse)
      permanence = synapseData.permanence

      if synapse in activeSynapses:
        permanence += permanenceIncrement
      else:
        permanence -= permanenceDecrement

      # Keep permanence within min/max bounds
      permanence = max(0.0, min(1.0, permanence))

      connections.updateSynapsePermanence(synapse, permanence)


  def printSegment(self, segment, connections):
    cell = connections.cellForSegment(segment)
    synapses = connections.synapsesForSegment(segment)
    print "segment id=",segment
    print "   cell=",cell
    print "   col =",self.columnForCell(cell)
    print "   synapses=",
    for synapse in synapses:
      synapseData = connections.dataForSynapse(synapse)
      permanence = synapseData.permanence
      presynapticCell = synapseData.presynapticCell
      print "%d:%g" % (presynapticCell,permanence),
    print