import operator

import numpy as np
from nupic.bindings.math import Random, SegmentSparseMatrix


EMPTY_UINT_ARRAY = np.array((), dtype="uint32")


class TemporalMemory(object):
  """
  TemporalMemory with basal and apical connections, and with the ability to
  connect to external cells.

  Basal connections are used to implement traditional Temporal Memory.

  The apical connections are used for further disambiguation. If multiple cells
  in a minicolumn have active basal segments, each of those cells is predicted,
  unless one of them also has an active apical segment, in which case only the
  cells with active basal and apical segments are predicted.

  This TemporalMemory is unaware of whether its basalInput or apicalInput are
  from internal or external cells. They are just cell numbers. The caller knows
  what these cell numbers mean, but the TemporalMemory doesn't. This allows the
  same code to work for various algorithms.

  To implement sequence memory, use

    basalInputDimensions=(numColumns*cellsPerColumn,)

  and call compute like this:

    tm.compute(activeColumns, tm.getActiveCells(), tm.getWinnerCells())

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


  def compute(self,
              activeColumns,
              basalInput,
              basalGrowthCandidates,
              apicalInput=EMPTY_UINT_ARRAY,
              apicalGrowthCandidates=EMPTY_UINT_ARRAY,
              learn=True):
    """
    @param activeColumns (numpy array)
    @param basalInput (numpy array)
    @param basalGrowthCandidates (numpy array)
    @param apicalInput (numpy array)
    @param apicalGrowthCandidates (numpy array)
    @param learn (bool)
    """
    # Calculate predictions for this timestep
    (activeBasalSegments,
     activeApicalSegments,
     matchingBasalSegments,
     matchingApicalSegments,
     basalPotentialOverlaps,
     apicalPotentialOverlaps) = self._calculateSegmentActivity(basalInput,
                                                               apicalInput)
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
     basalSegmentsToPunish,
     apicalSegmentsToPunish,
     newBasalSegmentCells,
     newApicalSegmentCells,
     learningCells) = self._calculateLearning(activeColumns,
                                              burstingColumns,
                                              correctPredictedCells,
                                              activeBasalSegments,
                                              activeApicalSegments,
                                              matchingBasalSegments,
                                              matchingApicalSegments,
                                              basalPotentialOverlaps,
                                              apicalPotentialOverlaps)

    # Learn
    if learn:
      self._learn(learningActiveBasalSegments,
                  learningActiveApicalSegments,
                  learningMatchingBasalSegments,
                  learningMatchingApicalSegments,
                  basalSegmentsToPunish,
                  apicalSegmentsToPunish,
                  newBasalSegmentCells,
                  newApicalSegmentCells,
                  basalInput,
                  basalGrowthCandidates,
                  apicalInput,
                  apicalGrowthCandidates,
                  basalPotentialOverlaps,
                  apicalPotentialOverlaps)

    # Save the results
    self.activeCells = newActiveCells
    self.winnerCells = learningCells
    self.prevPredictedCells = predictedCells


  def _calculateLearning(self,
                         activeColumns,
                         burstingColumns,
                         correctPredictedCells,
                         activeBasalSegments,
                         activeApicalSegments,
                         matchingBasalSegments,
                         matchingApicalSegments,
                         basalPotentialOverlaps,
                         apicalPotentialOverlaps):
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
    @param basalPotentialOverlaps (numpy array)
    @param apicalPotentialOverlaps (numpy array)

    @return (tuple)
    - learningActiveBasalSegments (numpy array)
      Active basal segments on correct predicted cells

    - learningActiveApicalSegments (numpy array)
      Active apical segments on correct predicted cells

    - learningMatchingBasalSegments (numpy array)
      Matching basal segments selected for learning in bursting columns

    - learningMatchingApicalSegments (numpy array)
      Matching apical segments selected for learning in bursting columns

    - basalSegmentsToPunish (numpy array)
      Basal segments that should be punished for predicting an inactive column

    - apicalSegmentsToPunish (numpy array)
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
      matchingBasalSegments, basalPotentialOverlaps, self.cellsPerColumn)
    newBasalSegmentCells = self._getCellsWithFewestSegments(
      self.basalConnections, self.rng, burstingColumnsWithNoMatch,
      self.cellsPerColumn)

    # Learning cells were determined completely from basal segments.
    # Do all apical learning on the same cells.
    learningCells = np.concatenate(
      (correctPredictedCells,
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
      matchingApicalSegments, apicalPotentialOverlaps)

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

      basalSegmentsToPunish = matchingBasalSegments[~correctMatchingBasalMask]
      apicalSegmentsToPunish = matchingApicalSegments[~correctMatchingApicalMask]
    else:
      basalSegmentsToPunish = EMPTY_UINT_ARRAY
      apicalSegmentsToPunish = EMPTY_UINT_ARRAY

    return (learningActiveBasalSegments,
            learningActiveApicalSegments,
            learningMatchingBasalSegments,
            learningMatchingApicalSegments,
            basalSegmentsToPunish,
            apicalSegmentsToPunish,
            newBasalSegmentCells,
            newApicalSegmentCells,
            learningCells)


  def _calculateSegmentActivity(self, basalInput, apicalInput):
    """
    Calculate the active and matching segments for this timestep.

    @param basalInput (numpy array)
    @param apicalInput (numpy array)

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

    - basalPotentialOverlaps (numpy array)
      The number of active potential synapses for each basal segment.
      Includes counts for active, matching, and nonmatching segments.

    - apicalPotentialOverlaps (numpy array)
      The number of active potential synapses for each apical segment
      Includes counts for active, matching, and nonmatching segments.
    """

    basalPermanences = self.basalConnections.matrix
    apicalPermanences = self.apicalConnections.matrix

    # Active basal
    basalOverlaps = basalPermanences.rightVecSumAtNZGteThresholdSparse(
      basalInput, self.connectedPermanence)
    activeBasalSegments = np.flatnonzero(
      basalOverlaps >= self.activationThreshold).astype("uint32")
    self.basalConnections.sortSegmentsByCell(activeBasalSegments)

    # Matching basal
    basalPotentialOverlaps = basalPermanences.rightVecSumAtNZSparse(
      basalInput)
    matchingBasalSegments = np.flatnonzero(
      basalPotentialOverlaps >= self.minThreshold).astype("uint32")
    self.basalConnections.sortSegmentsByCell(matchingBasalSegments)

    # Active apical
    apicalOverlaps = apicalPermanences.rightVecSumAtNZGteThresholdSparse(
      apicalInput, self.connectedPermanence)
    activeApicalSegments = np.flatnonzero(
      apicalOverlaps >= self.activationThreshold).astype("uint32")
    self.apicalConnections.sortSegmentsByCell(activeApicalSegments)

    # Matching apical
    apicalPotentialOverlaps = apicalPermanences.rightVecSumAtNZSparse(
      apicalInput)
    matchingApicalSegments = np.flatnonzero(
      apicalPotentialOverlaps >= self.minThreshold).astype("uint32")
    self.apicalConnections.sortSegmentsByCell(matchingApicalSegments)

    return (activeBasalSegments,
            activeApicalSegments,
            matchingBasalSegments,
            matchingApicalSegments,
            basalPotentialOverlaps,
            apicalPotentialOverlaps)


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
             basalSegmentsToPunish,
             apicalSegmentsToPunish,
             newBasalSegmentCells,
             newApicalSegmentCells,
             basalInput,
             basalGrowthCandidates,
             apicalInput,
             apicalGrowthCandidates,
             basalPotentialOverlaps,
             apicalPotentialOverlaps):
    """
    Adjust synapse permanences, grow new synapses, and grow new segments.

    @param learningActiveBasalSegments (numpy array)
    @param learningActiveApicalSegments (numpy array)
    @param learningMatchingBasalSegments (numpy array)
    @param learningMatchingApicalSegments (numpy array)
    @param basalSegmentsToPunish (numpy array)
    @param apicalSegmentsToPunish (numpy array)
    @param newBasalSegmentCells (numpy array)
    @param newApicalSegmentCells (numpy array)
    @param basalInput (numpy array)
    @param basalGrowthCandidates (numpy array)
    @param apicalInput (numpy array)
    @param apicalGrowthCandidates (numpy array)
    @param basalPotentialOverlaps (numpy array)
    @param apicalPotentialOverlaps (numpy array)
    """

    basalPermanences = self.basalConnections.matrix
    apicalPermanences = self.apicalConnections.matrix

    # Existing basal
    self._learnOnExistingSegments(basalPermanences, self.rng,
                                  learningActiveBasalSegments, basalInput,
                                  basalGrowthCandidates, basalPotentialOverlaps,
                                  self.sampleSize, self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)
    self._learnOnExistingSegments(basalPermanences, self.rng,
                                  learningMatchingBasalSegments, basalInput,
                                  basalGrowthCandidates, basalPotentialOverlaps,
                                  self.sampleSize, self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)

    # Existing apical
    self._learnOnExistingSegments(apicalPermanences, self.rng,
                                  learningActiveApicalSegments, apicalInput,
                                  apicalGrowthCandidates,
                                  apicalPotentialOverlaps, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)
    self._learnOnExistingSegments(apicalPermanences, self.rng,
                                  learningMatchingApicalSegments, apicalInput,
                                  apicalGrowthCandidates,
                                  apicalPotentialOverlaps, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)

    # New basal
    if len(basalGrowthCandidates) > 0:
      newBasalSegments = self.basalConnections.createSegments(
        newBasalSegmentCells)
      self._learnOnNewSegments(basalPermanences, self.rng, newBasalSegments,
                               basalGrowthCandidates, self.sampleSize,
                               self.initialPermanence,
                               self.maxSynapsesPerSegment)

    # New apical
    if len(apicalGrowthCandidates) > 0:
      newApicalSegments = self.apicalConnections.createSegments(
        newApicalSegmentCells)
      self._learnOnNewSegments(apicalPermanences, self.rng, newApicalSegments,
                               apicalGrowthCandidates, self.sampleSize,
                               self.initialPermanence,
                               self.maxSynapsesPerSegment)

    # Punish incorrect predictions.
    self._punishSegments(basalPermanences, basalSegmentsToPunish,
                         basalInput, self.predictedSegmentDecrement)
    self._punishSegments(apicalPermanences, apicalSegmentsToPunish,
                         apicalInput, self.predictedSegmentDecrement)


  @classmethod
  def _chooseBestSegmentPerCell(cls,
                                connections,
                                cells,
                                allMatchingSegments,
                                potentialOverlaps):
    """
    For each specified cell, choose its matching segment with largest number
    of active potential synapses. When there's a tie, the first segment wins.

    @param connections (SegmentSparseMatrix)
    @param cells (numpy array)
    @param allMatchingSegments (numpy array)
    @param potentialOverlaps (numpy array)

    @return (numpy array)
    One segment per cell
    """

    candidateSegments = connections.filterSegmentsByCell(allMatchingSegments,
                                                         cells)

    # Narrow it down to one pair per cell.
    onePerCellFilter = cls._argmaxMulti(potentialOverlaps[candidateSegments],
                                        connections.mapSegmentsToCells(
                                          candidateSegments))
    learningSegments = candidateSegments[onePerCellFilter]

    return learningSegments


  @classmethod
  def _chooseBestSegmentPerColumn(cls, connections, matchingCells,
                                  allMatchingSegments, potentialOverlaps,
                                  cellsPerColumn):
    """
    For all the columns covered by 'matchingCells', choose the column's matching
    segment with largest number of active potential synapses. When there's a
    tie, the first segment wins.

    @param connections (SegmentSparseMatrix)
    @param matchingCells (numpy array)
    @param allMatchingSegments (numpy array)
    @param potentialOverlaps (numpy array)
    """

    candidateSegments = connections.filterSegmentsByCell(allMatchingSegments,
                                                         matchingCells)

    # Narrow it down to one segment per column.
    cellScores = potentialOverlaps[candidateSegments]
    columnsForCandidates = (connections.mapSegmentsToCells(candidateSegments) /
                            cellsPerColumn)
    onePerColumnFilter = cls._argmaxMulti(cellScores, columnsForCandidates)

    learningSegments = candidateSegments[onePerColumnFilter]

    return learningSegments


  @classmethod
  def _getCellsWithFewestSegments(cls, connections, rng, columns,
                                  cellsPerColumn):
    """
    For each column, get the cell that has the fewest total basal segments.
    Break ties randomly.

    @param connections (SegmentSparseMatrix)
    @param rng (Random)
    @param columns (numpy array) Columns to check

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

    offsetPercents = np.empty(len(columns), dtype="float32")
    rng.initializeReal32Array(offsetPercents)

    np.add(onePerColumnFilter,
           offsetPercents*numCandidatesInColumns,
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
                               learningSegments,
                               activeInput, growthCandidates,
                               potentialOverlaps,
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

    maxNewNonzeros = cls._getMaxSynapseGrowthCounts(
      permanences, learningSegments, growthCandidates, potentialOverlaps,
      sampleSize, maxSynapsesPerSegment)

    permanences.setRandomZerosOnOuter(
      learningSegments, growthCandidates, maxNewNonzeros, initialPermanence,
      rng)


  @staticmethod
  def _getMaxSynapseGrowthCounts(permanences, learningSegments,
                                 growthCandidates, potentialOverlaps,
                                 sampleSize, maxSynapsesPerSegment):
    """
    Calculate the number of new synapses to attempt to grow for each segment,
    considering the sampleSize and maxSynapsesPerSegment parameters.

    Because the growth candidates are a subset of the active cells, we can't
    actually calculate the number of synapses to grow. We don't know how many
    of the active synapses are to winner cells. We can only calculate the
    maximums.

    @param permanences (SparseMatrix)
    @param learningSegments (numpy array)
    @param growthCandidates (numpy array)
    @param potentialOverlaps (numpy array)

    @return (numpy array) or (int)
    """

    if sampleSize != -1 or maxSynapsesPerSegment != -1:
      # Use signed integers to handle differences, then zero any negative numbers
      # and convert back to unsigned.
      if sampleSize == -1:
        maxNew = np.full(len(learningSegments), len(winnerInput), dtype="int32")
      else:
        numActiveSynapsesBySegment = potentialOverlaps[
          learningSegments].astype("int32")
        maxNew = sampleSize - numActiveSynapsesBySegment

      if maxSynapsesPerSegment != -1:
        totalSynapsesPerSegment = permanences.nNonZerosPerRow(
          learningSegments).astype("int32")
        numSynapsesToReachMax = maxSynapsesPerSegment - totalSynapsesPerSegment
        maxNew = np.where(maxNew <= numSynapsesToReachMax,
                          maxNew, numSynapsesToReachMax)

      maxNewUnsigned = np.empty(len(learningSegments), dtype="uint32")
      np.clip(maxNew, 0, float("inf"), out=maxNewUnsigned)

      return maxNewUnsigned
    else:
      return len(winnerInput)


  @staticmethod
  def _learnOnNewSegments(permanences, rng, newSegments, growthCandidates,
                          sampleSize, initialPermanence, maxSynapsesPerSegment):
    """
    Grow synapses on the provided segments.

    Because each segment has no synapses, we don't have to calculate how many
    synapses to grow.

    @param permanences (SparseMatrix)
    @param rng (Random)
    @param newSegments (numpy array)
    @param growthCandidates (numpy array)
    """

    numGrow = len(growthCandidates)

    if sampleSize != -1:
      numGrow = min(numGrow, sampleSize)

    if maxSynapsesPerSegment != -1:
      numGrow = min(numGrow, maxSynapsesPerSegment)

    permanences.setRandomZerosOnOuter(
      newSegments, growthCandidates, numGrow, initialPermanence, rng)


  @staticmethod
  def _punishSegments(permanences, segmentsToPunish, activeInput,
                      predictedSegmentDecrement):
    """
    Weaken active synapses on the provided segments.

    @param permanences (SparseMatrix)
    @param segmentsToPunish (numpy array)
    @param activeInput (numpy array)
    """
    permanences.incrementNonZerosOnOuter(
      segmentsToPunish, activeInput, -predictedSegmentDecrement)
    permanences.clipRowsBelowAndAbove(
      segmentsToPunish, 0.0, 1.0)


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
