import operator

import numpy as np
from nupic.bindings.math import Random, SegmentSparseMatrix


EMPTY_UINT_ARRAY = np.array((), dtype="uint32")


class TemporalMemory(object):
  """
  Like the ExtendedTemporalMemory, but this just handles external inputs.
  """

  def __init__(self,
               columnDimensions=(2048,),
               basalInputDimensions=(),
               apicalInputDimensions=(),
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.1,
               predictedSegmentDecrement=0.0,
               maxNewSynapseCount=None,
               maxSynapsesPerSegment=-1,
               maxSegmentsPerCell=None,
               seed=42):

    self.columnDimensions = columnDimensions
    self.numColumns = self._numPoints(columnDimensions)
    self.basalInputDimensions = basalInputDimensions
    self.apicalInputDimensions = apicalInputDimensions

    self.cellsPerColumn = cellsPerColumn
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold

    self.sampleSize = sampleSize
    if maxNewSynapseCount is not None:
      print "Parameter 'maxNewSynapseCount' is deprecated. Use 'sampleSize'."
      self.sampleSize = maxNewSynapseCount

    if maxSegmentsPerCell is not None:
      print "Warning: ignoring parameter 'maxSegmentsPerCell'"

    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.activationThreshold = activationThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment

    self.basalConnections = SegmentSparseMatrix(
      self.numColumns*cellsPerColumn, self._numPoints(basalInputDimensions))
    self.apicalConnections = SegmentSparseMatrix(
      self.numColumns*cellsPerColumn, self._numPoints(apicalInputDimensions))
    self.rng = Random(seed)
    self.activeCells = EMPTY_UINT_ARRAY
    self.winnerCells = EMPTY_UINT_ARRAY
    self.prevPredictedCells = EMPTY_UINT_ARRAY


  def reset(self):
    self.activeCells = EMPTY_UINT_ARRAY
    self.winnerCells = EMPTY_UINT_ARRAY
    self.prevPredictedCells = EMPTY_UINT_ARRAY


  def compute(self, activeColumns, lateralInput, feedbackInput, learn=True):
    # Calculate predictions for this timestep
    (activeBasalSegments,
     activeApicalSegments,
     matchingBasalSegments,
     matchingApicalSegments,
     basalPotentialExcitations,
     apicalPotentialExcitations) = self._calculateSegmentActivity(lateralInput,
                                                                  feedbackInput)
    predictedCells = self._calculatePredictedCells(activeBasalSegments,
                                                   activeApicalSegments)

    # Calculate active cells
    (correctPredictedCells,
     burstingColumns) = self._getColumnCoverage(predictedCells, activeColumns,
                                                self.cellsPerColumn)
    newActiveCells = np.concatenate((correctPredictedCells,
                                     self._getAllCellsInColumns(
                                       burstingColumns, self.cellsPerColumn)))
    newActiveCells.sort()

    # Calculate learning
    (learningActiveBasalSegments,
     learningActiveApicalSegments,
     learningMatchingBasalSegments,
     learningMatchingApicalSegments,
     punishedBasalSegments,
     punishedApicalSegments,
     newBasalSegmentCells,
     newApicalSegmentCells,
     learningCells) = self._calculateLearning(activeColumns,
                                              burstingColumns,
                                              correctPredictedCells,
                                              activeBasalSegments,
                                              activeApicalSegments,
                                              matchingBasalSegments,
                                              matchingApicalSegments,
                                              basalPotentialExcitations,
                                              apicalPotentialExcitations)

    # Learn
    if learn:
      self._learn(learningActiveBasalSegments,
                  learningActiveApicalSegments,
                  learningMatchingBasalSegments,
                  learningMatchingApicalSegments,
                  punishedBasalSegments,
                  punishedApicalSegments,
                  newBasalSegmentCells,
                  newApicalSegmentCells,
                  lateralInput,
                  feedbackInput,
                  basalPotentialExcitations,
                  apicalPotentialExcitations)


    # Save the results
    self.prevPredictedCells = predictedCells
    self.activeCells = newActiveCells
    self.winnerCells = learningCells


  def _calculateLearning(self,
                         activeColumns,
                         burstingColumns,
                         correctPredictedCells,
                         activeBasalSegments,
                         activeApicalSegments,
                         matchingBasalSegments,
                         matchingApicalSegments,
                         basalPotentialExcitations,
                         apicalPotentialExcitations):
    """
    This is basically just the TemporalMemory's basal learning with apical
    learning bolted on.

    First, calculate all basal learning. Correctly predicted cells always have
    active basal and segments, and we learn on these segments. In bursting
    columns, we either learn on an existing basal segment, or we grow a new one.

    Next, now that we know all the cells doing basal learning, calculate the
    apical learning.

    The only influence apical dendrites have on basal learning is: the apical
    dendrites influence which cells are considered "predicted". So an active
    apical dendrite can keep some basal segments in active columns from
    learning.

    @param correctPredictedCells (numpy array)
    @param burstingColumns (numpy array)
    @param activeBasalSegments (numpy array)
    @param activeApicalSegments (numpy array)
    @param matchingBasalSegments (numpy array)
    @param matchingApicalSegments (numpy array)
    @param basalPotentialExcitations (numpy array)
    @param apicalPotentialExcitations (numpy array)

    @return (tuple)
    - learningActiveBasalSegments (numpy array)
      Active basal segments on correct predicted cells

    - learningActiveApicalSegments (numpy array)
      Active apical segments on correct predicted cells

    - learningMatchingBasalSegments (numpy array)
      Matching basal segments selected for learning in bursting columns

    - learningMatchingApicalSegments (numpy array)
      Matching apical segments selected for learning in bursting columns

    - punishedBasalSegments (numpy array)
      Basal segments that should be punished for predicting an inactive column

    - punishedApicalSegments (numpy array)
      Apical segments that should be punished for predicting an inactive column

    - newSegmentCells (numpy array)
      Cells in bursting columns that were selected to grow new segments
    """

    # Correctly predicted columns
    learningActiveBasalSegments = self.basalConnections.filterSegmentsByCell(
      activeBasalSegments, correctPredictedCells)

    cellsForMatchingBasal = self.basalConnections.mapSegmentsToCells(
      matchingBasalSegments)
    matchingCells = np.unique(cellsForMatchingBasal)
    (matchingCellsInBurstingColumns,
     burstingColumnsWithNoMatch) = self._getColumnCoverage(matchingCells,
                                                           burstingColumns,
                                                           self.cellsPerColumn)
    learningMatchingBasalSegments = self._chooseBestSegmentPerColumn(
      self.basalConnections, matchingCellsInBurstingColumns,
      matchingBasalSegments, basalPotentialExcitations, self.cellsPerColumn)
    newBasalSegmentCells = self._getCellsWithFewestSegments(
      self.basalConnections, burstingColumnsWithNoMatch, self.cellsPerColumn)

    # Learning cells were determined completely from basal segments.
    # Do all apical learning on the same cells.
    learningCells = np.concatenate(
      (self.basalConnections.mapSegmentsToCells(learningActiveBasalSegments),
       self.basalConnections.mapSegmentsToCells(learningMatchingBasalSegments),
       newBasalSegmentCells))
    learningCells.sort()

    # Cells with active apical segments
    learningActiveApicalSegments = self.apicalConnections.filterSegmentsByCell(
      activeApicalSegments, learningCells)

    # Cells with matching apical segments
    learningCellsWithoutActiveApical = np.setdiff1d(
      learningCells,
      self.apicalConnections.mapSegmentsToCells(learningActiveApicalSegments))
    cellsForMatchingApical = self.apicalConnections.mapSegmentsToCells(
      matchingApicalSegments)
    learningCellsWithMatchingApical = np.intersect1d(
      learningCellsWithoutActiveApical, cellsForMatchingApical)
    learningMatchingApicalSegments = self._chooseBestSegmentPerCell(
      self.apicalConnections, learningCellsWithMatchingApical,
      matchingApicalSegments, apicalPotentialExcitations)

    # Cells that need to grow an apical segment
    newApicalSegmentCells = np.setdiff1d(learningCellsWithoutActiveApical,
                                         learningCellsWithMatchingApical)

    learningActiveApicalSegments = self.apicalConnections.filterSegmentsByCell(
      activeApicalSegments, correctPredictedCells)

    # Incorrectly predicted columns
    if self.predictedSegmentDecrement > 0.0:
      correctMatchingBasalMask = np.in1d(
        cellsForMatchingBasal / self.cellsPerColumn, activeColumns)
      correctMatchingApicalMask = np.in1d(
        cellsForMatchingApical / self.cellsPerColumn, activeColumns)

      punishedBasalSegments = matchingBasalSegments[~correctMatchingBasalMask]
      punishedApicalSegments = matchingApicalSegments[~correctMatchingApicalMask]
    else:
      punishedBasalSegments = EMPTY_UINT_ARRAY
      punishedApicalSegments = EMPTY_UINT_ARRAY

    return (learningActiveBasalSegments,
            learningActiveApicalSegments,
            learningMatchingBasalSegments,
            learningMatchingApicalSegments,
            punishedBasalSegments,
            punishedApicalSegments,
            newBasalSegmentCells,
            newApicalSegmentCells,
            learningCells)


  def _calculateSegmentActivity(self, lateralInput, feedbackInput):
    """
    Calculate the active and matching segments for this timestep.

    @param lateralInput (numpy array)
    @param feedbackInput (numpy array)

    @return (tuple)
    - activeBasalSegments (numpy array)
      Basal dendrite segments with enough active connected synapses to cause a
      dendritic spike

    - activeApicalSegments (numpy array)
      Apical dendrite segments with enough active connected synapses to cause a
      dendritic spike

    - matchingBasalSegments (numpy array)
      Basal dendrite segments with enough active potential synapses to be
      selected for learning in a bursting column

    - matchingApicalSegments (numpy array)
      Apical dendrite segments with enough active potential synapses to be
      selected for learning in a bursting column

    - basalPotentialExcitations (numpy array)
      The number of active potential synapses for each basal segment.
      Includes counts for active, matching, and nonmatching segments.

    - apicalPotentialExcitations (numpy array)
      The number of active potential synapses for each apical segment
      Includes counts for active, matching, and nonmatching segments.
    """

    basalPermanences = self.basalConnections.matrix
    apicalPermanences = self.apicalConnections.matrix

    # Active basal
    basalExcitations = basalPermanences.rightVecSumAtNZGteThresholdSparse(
      lateralInput, self.connectedPermanence)
    activeBasalSegments = np.flatnonzero(
      basalExcitations >= self.activationThreshold).astype("uint32")
    self.basalConnections.sortSegmentsByCell(activeBasalSegments)

    # Matching basal
    basalPotentialExcitations = basalPermanences.rightVecSumAtNZSparse(
      lateralInput)
    matchingBasalSegments = np.flatnonzero(
      basalPotentialExcitations >= self.minThreshold).astype("uint32")
    self.basalConnections.sortSegmentsByCell(matchingBasalSegments)

    # Active apical
    apicalExcitations = apicalPermanences.rightVecSumAtNZGteThresholdSparse(
      feedbackInput, self.connectedPermanence)
    activeApicalSegments = np.flatnonzero(
      apicalExcitations >= self.activationThreshold).astype("uint32")
    self.apicalConnections.sortSegmentsByCell(activeApicalSegments)

    # Matching apical
    apicalPotentialExcitations = apicalPermanences.rightVecSumAtNZSparse(
      feedbackInput)
    matchingApicalSegments = np.flatnonzero(
      apicalPotentialExcitations >= self.minThreshold).astype("uint32")
    self.apicalConnections.sortSegmentsByCell(matchingApicalSegments)

    return (activeBasalSegments,
            activeApicalSegments,
            matchingBasalSegments,
            matchingApicalSegments,
            basalPotentialExcitations,
            apicalPotentialExcitations)


  def _calculatePredictedCells(self, activeBasalSegments, activeApicalSegments):
    """
    Calculate the predicted cells, given the set of active segments.

    An active basal segment is enough to predict a cell.
    An active apical segment is *not* enough to predict a cell.

    When a cell has both types of segments active, other cells in its minicolumn
    must also have both types of segments to be considered predictive.

    @param activeBasalSegments (numpy array)
    @param activeApicalSegments (numpy array)

    @return (numpy array)
    """

    cellsForBasalSegments = self.basalConnections.mapSegmentsToCells(
      activeBasalSegments)
    cellsForApicalSegments = self.apicalConnections.mapSegmentsToCells(
      activeApicalSegments)

    fullyDepolarizedCells = np.intersect1d(cellsForBasalSegments,
                                           cellsForApicalSegments)
    partlyDepolarizedCells = np.setdiff1d(cellsForBasalSegments,
                                          fullyDepolarizedCells)

    inhibitedMask = np.in1d(partlyDepolarizedCells / self.cellsPerColumn,
                            fullyDepolarizedCells / self.cellsPerColumn)
    predictedCells = np.append(fullyDepolarizedCells,
                               partlyDepolarizedCells[~inhibitedMask])
    predictedCells.sort()

    return predictedCells


  def _learn(self,
             learningActiveBasalSegments,
             learningActiveApicalSegments,
             learningMatchingBasalSegments,
             learningMatchingApicalSegments,
             punishedBasalSegments,
             punishedApicalSegments,
             newBasalSegmentCells,
             newApicalSegmentCells,
             lateralInput,
             feedbackInput,
             basalPotentialExcitations,
             apicalPotentialExcitations):
    """
    Adjust synapse permanences, grow new synapses, and grow new segments.

    @param learningActiveBasalSegments (numpy array)
    @param learningActiveApicalSegments (numpy array)
    @param learningMatchingBasalSegments (numpy array)
    @param learningMatchingApicalSegments (numpy array)
    @param punishedBasalSegments (numpy array)
    @param punishedApicalSegments (numpy array)
    @param newBasalSegmentCells (numpy array)
    @param newApicalSegmentCells (numpy array)
    @param lateralInput (numpy array)
    @param feedbackInput (numpy array)
    @param basalPotentialExcitations (numpy array)
    @param apicalPotentialExcitations (numpy array)
    """

    basalPermanences = self.basalConnections.matrix
    apicalPermanences = self.apicalConnections.matrix

    # Existing basal
    self._learnOnExistingSegments(basalPermanences, self.rng,
                                  learningActiveBasalSegments, lateralInput,
                                  basalPotentialExcitations, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)
    self._learnOnExistingSegments(basalPermanences, self.rng,
                                  learningMatchingBasalSegments, lateralInput,
                                  basalPotentialExcitations, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)

    # Existing apical
    self._learnOnExistingSegments(apicalPermanences, self.rng,
                                  learningActiveApicalSegments, feedbackInput,
                                  apicalPotentialExcitations, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)
    self._learnOnExistingSegments(apicalPermanences, self.rng,
                                  learningMatchingApicalSegments, feedbackInput,
                                  apicalPotentialExcitations, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)

    # New basal
    if len(lateralInput) > 0:
      newBasalSegments = self.basalConnections.createSegments(
        newBasalSegmentCells)
      self._learnOnNewSegments(basalPermanences, self.rng, newBasalSegments,
                               lateralInput, self.sampleSize,
                               self.initialPermanence,
                               self.maxSynapsesPerSegment)

    # New apical
    if len(feedbackInput) > 0:
      newApicalSegments = self.apicalConnections.createSegments(
        newApicalSegmentCells)
      self._learnOnNewSegments(apicalPermanences, self.rng, newApicalSegments,
                               feedbackInput, self.sampleSize,
                               self.initialPermanence,
                               self.maxSynapsesPerSegment)

    # Punish incorrect predictions.
    self._punishSegments(basalPermanences, punishedBasalSegments,
                         lateralInput, self.predictedSegmentDecrement)
    self._punishSegments(apicalPermanences, punishedApicalSegments,
                         feedbackInput, self.predictedSegmentDecrement)


  @classmethod
  def _chooseBestSegmentPerCell(cls,
                                connections,
                                cells,
                                allMatchingSegments,
                                potentialExcitations):
    """
    For each specified cell, choose its matching segment with largest number
    of active potential synapses. When there's a tie, the first segment wins.

    @param connections (SegmentSparseMatrix)
    @param cells (numpy array)
    @param allMatchingSegments (numpy array)
    @param potentialExcitations (numpy array)

    @return (numpy array)
    One segment per cell
    """

    candidateSegments = connections.filterSegmentsByCell(allMatchingSegments,
                                                         cells)

    # Narrow it down to one pair per cell.
    onePerCellFilter = cls._argmaxMulti(potentialExcitations[candidateSegments],
                                        connections.mapSegmentsToCells(
                                          candidateSegments))
    learningSegments = candidateSegments[onePerCellFilter]

    return learningSegments


  @classmethod
  def _chooseBestSegmentPerColumn(cls, connections, matchingCells,
                                  allMatchingSegments, potentialExcitations,
                                  cellsPerColumn):
    """
    For all the columns covered by 'matchingCells', choose the column's matching
    segment with largest number of active potential synapses. When there's a
    tie, the first segment wins.

    @param connections (SegmentSparseMatrix)
    @param matchingCells (numpy array)
    @param allMatchingSegments (numpy array)
    @param potentialExcitations (numpy array)
    """

    candidateSegments = connections.filterSegmentsByCell(allMatchingSegments,
                                                         matchingCells)

    # Narrow it down to one segment per column.
    cellScores = potentialExcitations[candidateSegments]
    columnsForCandidates = (connections.mapSegmentsToCells(candidateSegments) /
                            cellsPerColumn)
    onePerColumnFilter = cls._argmaxMulti(cellScores, columnsForCandidates)

    learningSegments = candidateSegments[onePerColumnFilter]

    return learningSegments


  @classmethod
  def _getCellsWithFewestSegments(cls, connections, columns, cellsPerColumn):
    """
    For each column, get the cell that has the fewest total basal segments.
    Break ties randomly.

    @param columns (numpy array)
    Columns to check

    @return (numpy array)
    One cell for each of the provided columns
    """
    candidateCells = cls._getAllCellsInColumns(columns, cellsPerColumn)

    # Arrange the segment counts into one row per minicolumn.
    segmentCounts = np.reshape(connections.getSegmentCounts(candidateCells),
                               newshape=(len(columns),
                                         cellsPerColumn))

    # Filter to just the cells that are tied for fewest in their minicolumn.
    minSegmentCounts = np.amin(segmentCounts, axis=1, keepdims=True)
    candidateCells = candidateCells[np.flatnonzero(segmentCounts ==
                                                   minSegmentCounts)]

    # Filter to one cell per column, choosing randomly from the minimums.
    # To do the random choice, add a random offset to each index in-place, using
    # casting to floor the result.
    (_,
     onePerColumnFilter,
     numCandidatesInColumns) = np.unique(candidateCells / cellsPerColumn,
                                         return_index=True, return_counts=True)
    np.add(onePerColumnFilter,
           np.random.rand(len(columns))*numCandidatesInColumns,
           out=onePerColumnFilter,
           casting="unsafe")

    return candidateCells[onePerColumnFilter]


  @staticmethod
  def _argmaxMulti(a, groupKeys):
    """
    This is like numpy's argmax, but it returns multiple maximums.

    It gets the indices of the max values of each group in 'a', grouping the
    elements by their corresponding value in 'groupKeys'.

    @param a (numpy array)
    An array of values that will be compared

    @param groupKeys (numpy array)
    An array with the same length of 'a'. Each entry identifies the group for
    each 'a' value. Group numbers must be organized together (e.g. sorted).

    @return (numpy array)
    The indices of one maximum value per group

    @example
      _argmaxMulti([5, 4, 7, 2, 9, 8],
                   [0, 0, 0, 1, 1, 1])
    returns
      [2, 4]
    """
    _, indices, lengths = np.unique(groupKeys, return_index=True,
                                    return_counts=True)

    maxValues = np.maximum.reduceat(a, indices)
    allMaxIndices = np.flatnonzero(np.repeat(maxValues, lengths) == a)

    # Break ties by finding the insertion points of the the group start indices
    # and using the values currently at those points. This approach will choose
    # the first occurrence of each max value.
    return allMaxIndices[np.searchsorted(allMaxIndices, indices)]


  @staticmethod
  def _getColumnCoverage(cells, columns, cellsPerColumn):
    """
    Get the cells that fall within the specified columns and the columns that
    don't contain any of the specified cells

    @param cells (numpy array)
    @param columns (numpy array)
    @param cellsPerColumn (int)

    @return (tuple)
    - cellsInColumns (numpy array)
      The provided cells that are within these columns

    - columnsWithoutCells (numpy array)
      The provided columns that weren't covered by any of the provided cells.
    """
    columnsWithCells = np.intersect1d(columns, cells / cellsPerColumn)
    cellsInColumnsMask = np.in1d(cells / cellsPerColumn, columnsWithCells)
    cellsInColumns = cells[cellsInColumnsMask]
    columnsWithoutCells = np.setdiff1d(columns, columnsWithCells)

    return cellsInColumns, columnsWithoutCells


  @staticmethod
  def _getAllCellsInColumns(columns, cellsPerColumn):
    """
    Get all cells in the specified columns.

    @param columns (numpy array)
    @param cellsPerColumn (int)

    @return (numpy array)
    All cells within the specified columns. The cells are in the same order as the
    provided columns, so they're sorted if the columns are sorted.
    """

    # Add
    #   [[beginningOfColumn0],
    #    [beginningOfColumn1],
    #     ...]
    # to
    #   [0, 1, 2, ..., cellsPerColumn - 1]
    # to get
    #   [beginningOfColumn0 + 0, beginningOfColumn0 + 1, ...
    #    beginningOfColumn1 + 0, ...
    #    ...]
    # then flatten it.
    return ((columns * cellsPerColumn).reshape((-1, 1)) +
            np.arange(cellsPerColumn, dtype="uint32")).flatten()


  @classmethod
  def _learnOnExistingSegments(cls, permanences, rng,
                               learningSegments, activeInput,
                               potentialExcitations,
                               sampleSize, initialPermanence,
                               permanenceIncrement, permanenceDecrement,
                               maxSynapsesPerSegment):
    """
    Learn on segments. Reinforce active synapses, punish inactive synapses, and
    grow new synapses.

    @param permanences (SparseMatrix)
    @param rng (Random)
    @param learningSegments (numpy array)
    @param activeInput (numpy array)
    """
    permanences.incrementNonZerosOnOuter(
      learningSegments, activeInput, permanenceIncrement)
    permanences.incrementNonZerosOnRowsExcludingCols(
      learningSegments, activeInput, -permanenceDecrement)
    permanences.clipRowsBelowAndAbove(
      learningSegments, 0.0, 1.0)

    numNewNonzeros = cls._getSynapseGrowthCounts(
      permanences, learningSegments, activeInput, potentialExcitations,
      sampleSize, maxSynapsesPerSegment)

    permanences.setRandomZerosOnOuter(
      learningSegments, activeInput, numNewNonzeros, initialPermanence, rng)


  @staticmethod
  def _getSynapseGrowthCounts(permanences, learningSegments, activeInput,
                              potentialExcitations, sampleSize,
                              maxSynapsesPerSegment):
    """
    Calculate the number of new synapses to grow for each segment, considering
    the sampleSize and maxSynapsesPerSegment parameters.

    @param permanences (SparseMatrix)
    @param learningSegments (numpy array)
    @param activeInput (numpy array)
    @param potentialExcitations (numpy array)

    @return (numpy array)
    """

    # Use signed integers to handle differences, then zero any negative numbers
    # and convert back to unsigned.
    activeSynapsesBySegment = potentialExcitations[
      learningSegments].astype("int32")

    if sampleSize == -1:
      numNew = len(activeInput) - activeSynapsesBySegment
    else:
      numNew = sampleSize - activeSynapsesBySegment

    if maxSynapsesPerSegment != -1:
      totalSynapsesPerSegment = permanences.nNonZerosPerRow(
        learningSegments).astype("int32")
      numSynapsesToReachMax = maxSynapsesPerSegment - totalSynapsesPerSegment
      numNew = np.where(numNew <= numSynapsesToReachMax,
                        numNew, numSynapsesToReachMax)

    numNewUnsigned = np.empty(len(learningSegments), dtype="uint32")
    np.clip(numNew, 0, float("inf"), out=numNewUnsigned)

    return numNewUnsigned


  @staticmethod
  def _learnOnNewSegments(permanences, rng, newSegments, activeInput,
                          sampleSize, initialPermanence, maxSynapsesPerSegment):
    """
    Grow synapses on the provided segments.

    Because each segment has no synapses, we don't have to calculate how many
    synapses to grow.

    @param permanences (SparseMatrix)
    @param rng (Random)
    @param newSegments (numpy array)
    @param activeInput (numpy array)
    """

    numGrow = len(activeInput)

    if sampleSize != -1:
      numGrow = min(numGrow, sampleSize)

    if maxSynapsesPerSegment != -1:
      numGrow = min(numGrow, maxSynapsesPerSegment)

    permanences.setRandomZerosOnOuter(
      newSegments, activeInput, numGrow, initialPermanence, rng)


  @staticmethod
  def _punishSegments(permanences, punishedSegments, activeInput,
                      predictedSegmentDecrement):
    """
    Weaken active synapses on the provided segments.

    @param permanences (SparseMatrix)
    @param punishedSegments (numpy array)
    @param activeInput (numpy array)
    """
    permanences.incrementNonZerosOnOuter(
      punishedSegments, activeInput, -predictedSegmentDecrement)
    permanences.clipRowsBelowAndAbove(
      punishedSegments, 0.0, 1.0)


  @staticmethod
  def _numPoints(dimensions):
    """
    Get the number of discrete points in a set of dimensions.

    @param dimensions (sequence of integers)
    @return (int)
    """
    if len(dimensions) == 0:
      return 0
    else:
      return reduce(operator.mul, dimensions, 1)


  def getActiveCells(self):
    return self.activeCells


  def getWinnerCells(self):
    return self.winnerCells


  def getPreviouslyPredictedCells(self):
    return self.prevPredictedCells
