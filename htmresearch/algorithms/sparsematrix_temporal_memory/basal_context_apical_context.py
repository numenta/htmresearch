import operator

import numpy as np
from nupic.bindings.math import Random, SparseMatrixConnections


EMPTY_UINT_ARRAY = np.array((), dtype="uint32")


class TemporalMemory(object):
  """
  An alternate approach to apical dendrites. Every cell SDR is specific to both
  the basal the apical input. Prediction requires both basal and apical support.
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

    self.basalConnections = SparseMatrixConnections(
      self.numColumns*cellsPerColumn, self._numPoints(basalInputDimensions))
    self.apicalConnections = SparseMatrixConnections(
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
              apicalInput,
              apicalGrowthCandidates,
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
    predictedCells = np.intersect1d(
      self.basalConnections.mapSegmentsToCells(activeBasalSegments),
      self.apicalConnections.mapSegmentsToCells(activeApicalSegments))

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
     newSegmentCells,
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
                  newSegmentCells,
                  basalInput,
                  basalGrowthCandidates,
                  apicalInput,
                  apicalGrowthCandidates,
                  basalPotentialOverlaps,
                  apicalPotentialOverlaps)


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
                         basalPotentialOverlaps,
                         apicalPotentialOverlaps):
    """
    Learning occurs on pairs of segments. Correctly predicted cells always have
    active basal and apical segments, and we learn on these segments. In
    bursting columns, we either learn on an existing segment pair, or we grow a
    new pair of segments.

    @param activeColumns (numpy array)
    @param burstingColumns (numpy array)
    @param correctPredictedCells (numpy array)
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

    - learningCells (numpy array)
      Every cell that has a learning segment or was selected to grow a segment
    """

    # Correctly predicted columns
    learningActiveBasalSegments = self.basalConnections.filterSegmentsByCell(
      activeBasalSegments, correctPredictedCells)
    learningActiveApicalSegments = self.apicalConnections.filterSegmentsByCell(
      activeApicalSegments, correctPredictedCells)

    # Bursting columns
    cellsForMatchingBasal = self.basalConnections.mapSegmentsToCells(
      matchingBasalSegments)
    cellsForMatchingApical = self.apicalConnections.mapSegmentsToCells(
      matchingApicalSegments)
    matchingCells = np.intersect1d(
      cellsForMatchingBasal, cellsForMatchingApical)
    (matchingCellsInBurstingColumns,
     burstingColumnsWithNoMatch) = self._getColumnCoverage(matchingCells,
                                                           burstingColumns,
                                                           self.cellsPerColumn)
    (learningMatchingBasalSegments,
     learningMatchingApicalSegments) = self._chooseBestSegmentPairPerColumn(
       matchingCellsInBurstingColumns, matchingBasalSegments,
       matchingApicalSegments, basalPotentialOverlaps, apicalPotentialOverlaps)
    newSegmentCells = self._getCellsWithFewestSegments(
      burstingColumnsWithNoMatch)

    # Incorrectly predicted columns
    if self.predictedSegmentDecrement > 0.0:
      correctMatchingBasalMask = np.in1d(
        cellsForMatchingBasal / self.cellsPerColumn, activeColumns)
      correctMatchingApicalMask = np.in1d(
        cellsForMatchingApical / self.cellsPerColumn, activeColumns)

      basalSegmentsToPunish = matchingBasalSegments[~correctMatchingBasalMask]
      apicalSegmentsToPunish = matchingApicalSegments[~correctMatchingApicalMask]
    else:
      basalSegmentsToPunish = ()
      apicalSegmentsToPunish = ()

    # Make a list of every cell that is learning
    learningCells =  np.concatenate(
      (correctPredictedCells,
       self.basalConnections.mapSegmentsToCells(learningMatchingBasalSegments),
       newSegmentCells))

    return (learningActiveBasalSegments,
            learningActiveApicalSegments,
            learningMatchingBasalSegments,
            learningMatchingApicalSegments,
            basalSegmentsToPunish,
            apicalSegmentsToPunish,
            newSegmentCells,
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

    # Active basal
    basalOverlaps = self.basalConnections.computeActivity(
      basalInput, self.connectedPermanence)
    activeBasalSegments = np.flatnonzero(
      basalOverlaps >= self.activationThreshold).astype("uint32")
    self.basalConnections.sortSegmentsByCell(activeBasalSegments)

    # Matching basal
    basalPotentialOverlaps = self.basalConnections.computeActivity(
      basalInput)
    matchingBasalSegments = np.flatnonzero(
      basalPotentialOverlaps >= self.minThreshold).astype("uint32")
    self.basalConnections.sortSegmentsByCell(matchingBasalSegments)

    # Active apical
    apicalOverlaps = self.apicalConnections.computeActivity(
      apicalInput, self.connectedPermanence)
    activeApicalSegments = np.flatnonzero(
      apicalOverlaps >= self.activationThreshold).astype("uint32")
    self.apicalConnections.sortSegmentsByCell(activeApicalSegments)

    # Matching apical
    apicalPotentialOverlaps = self.apicalConnections.computeActivity(
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


  def _learn(self,
             learningActiveBasalSegments,
             learningActiveApicalSegments,
             learningMatchingBasalSegments,
             learningMatchingApicalSegments,
             basalSegmentsToPunish,
             apicalSegmentsToPunish,
             newSegmentCells,
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
    @param newSegmentCells (numpy array)
    @param basalInput (numpy array)
    @param basalGrowthCandidates (numpy array)
    @param apicalInput (numpy array)
    @param apicalGrowthCandidates (numpy array)
    @param basalPotentialOverlaps (numpy array)
    @param apicalPotentialOverlaps (numpy array)
    """

    # Existing basal
    self._learnOnExistingSegments(self.basalConnections, self.rng,
                                  learningActiveBasalSegments, basalInput,
                                  basalGrowthCandidates,
                                  basalPotentialOverlaps, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)
    self._learnOnExistingSegments(self.basalConnections, self.rng,
                                  learningMatchingBasalSegments, basalInput,
                                  basalGrowthCandidates,
                                  basalPotentialOverlaps, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)

    # Existing apical
    self._learnOnExistingSegments(self.apicalConnections, self.rng,
                                  learningActiveApicalSegments, apicalInput,
                                  apicalGrowthCandidates,
                                  apicalPotentialOverlaps, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)
    self._learnOnExistingSegments(self.apicalConnections, self.rng,
                                  learningMatchingApicalSegments, apicalInput,
                                  apicalGrowthCandidates,
                                  apicalPotentialOverlaps, self.sampleSize,
                                  self.initialPermanence,
                                  self.permanenceIncrement,
                                  self.permanenceDecrement,
                                  self.maxSynapsesPerSegment)

    # New segments: Only grow segments if there is basal *and* apical input.
    if len(basalInput) > 0 and len(apicalInput) > 0:
      newBasalSegments = self.basalConnections.createSegments(newSegmentCells)
      self._learnOnNewSegments(self.basalConnections, self.rng, newBasalSegments,
                               basalGrowthCandidates, self.sampleSize,
                               self.initialPermanence,
                               self.maxSynapsesPerSegment)
      newApicalSegments = self.apicalConnections.createSegments(newSegmentCells)
      self._learnOnNewSegments(self.apicalConnections, self.rng, newApicalSegments,
                               apicalGrowthCandidates, self.sampleSize,
                               self.initialPermanence,
                               self.maxSynapsesPerSegment)

    # Punish incorrect predictions.
    self._punishSegments(self.basalConnections, basalSegmentsToPunish,
                         basalInput, self.predictedSegmentDecrement)
    self._punishSegments(self.apicalConnections, apicalSegmentsToPunish,
                         apicalInput, self.predictedSegmentDecrement)


  def _chooseBestSegmentPairPerColumn(self,
                                      matchingCellsInBurstingColumns,
                                      matchingBasalSegments,
                                      matchingApicalSegments,
                                      basalPotentialOverlaps,
                                      apicalPotentialOverlaps):
    """
    Choose the best pair of matching segments - one basal and one apical - for
    each column. Pairs are ranked by the sum of their potential overlaps.
    When there's a tie, the first pair wins.

    @param matchingCellsInBurstingColumns (numpy array)
    Cells in bursting columns that have at least one matching basal segment and
    at least one matching apical segment

    @param matchingBasalSegments (numpy array)
    @param matchingApicalSegments (numpy array)
    @param basalPotentialOverlaps (numpy array)
    @param apicalPotentialOverlaps (numpy array)

    @return (tuple)
    - learningBasalSegments (numpy array)
      The selected basal segments

    - learningApicalSegments (numpy array)
      The selected apical segments
    """

    basalCandidateSegments = self.basalConnections.filterSegmentsByCell(
      matchingBasalSegments, matchingCellsInBurstingColumns)
    apicalCandidateSegments = self.apicalConnections.filterSegmentsByCell(
      matchingApicalSegments, matchingCellsInBurstingColumns)

    # Narrow it down to one pair per cell.
    oneBasalPerCellFilter = self._argmaxMulti(
      basalPotentialOverlaps[basalCandidateSegments],
      self.basalConnections.mapSegmentsToCells(basalCandidateSegments))
    basalCandidateSegments = basalCandidateSegments[oneBasalPerCellFilter]
    oneApicalPerCellFilter = self._argmaxMulti(
      apicalPotentialOverlaps[apicalCandidateSegments],
      self.apicalConnections.mapSegmentsToCells(apicalCandidateSegments))
    apicalCandidateSegments = apicalCandidateSegments[oneApicalPerCellFilter]

    # Narrow it down to one pair per column.
    cellScores = (basalPotentialOverlaps[basalCandidateSegments] +
                  apicalPotentialOverlaps[apicalCandidateSegments])
    columnsForCandidates = (
      self.basalConnections.mapSegmentsToCells(basalCandidateSegments) /
      self.cellsPerColumn)
    onePerColumnFilter = self._argmaxMulti(cellScores, columnsForCandidates)

    learningBasalSegments = basalCandidateSegments[onePerColumnFilter]
    learningApicalSegments = apicalCandidateSegments[onePerColumnFilter]

    return (learningBasalSegments,
            learningApicalSegments)


  def _getCellsWithFewestSegments(self, columns):
    """
    For each column, get the cell that has the fewest total segments (basal or
    apical). Break ties randomly.

    @param columns (numpy array)
    Columns to check

    @return (numpy array)
    One cell for each of the provided columns
    """
    candidateCells = self._getAllCellsInColumns(columns, self.cellsPerColumn)

    # Arrange the segment counts into one row per minicolumn.
    segmentCounts = np.reshape(
      self.basalConnections.getSegmentCounts(candidateCells) +
      self.apicalConnections.getSegmentCounts(candidateCells),
      newshape=(len(columns),
                self.cellsPerColumn))

    # Filter to just the cells that are tied for fewest in their minicolumn.
    minSegmentCounts = np.amin(segmentCounts, axis=1, keepdims=True)
    candidateCells = candidateCells[np.flatnonzero(segmentCounts ==
                                                   minSegmentCounts)]

    # Filter to one cell per column, choosing randomly from the minimums.
    # To do the random choice, add a random offset to each index in-place, using
    # casting to floor the result.
    (_,
     onePerColumnFilter,
     numCandidatesInColumns) = np.unique(candidateCells / self.cellsPerColumn,
                                         return_index=True, return_counts=True)

    offsetPercents = np.empty(len(columns), dtype="float32")
    self.rng.initializeReal32Array(offsetPercents)

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
  def _learnOnExistingSegments(cls, connections, rng,
                               learningSegments,
                               activeInput, growthCandidates,
                               potentialOverlaps,
                               sampleSize, initialPermanence,
                               permanenceIncrement, permanenceDecrement,
                               maxSynapsesPerSegment):
    """
    Learn on segments. Reinforce active synapses, punish inactive synapses, and
    grow new synapses.

    @param connections (SparseMatrixConnections)
    @param rng (Random)
    @param learningSegments (numpy array)
    @param activeInput (numpy array)
    """
    connections.adjustSynapses(learningSegments, activeInput,
                               permanenceIncrement, -permanenceDecrement)

    maxNewNonzeros = cls._getMaxSynapseGrowthCounts(
      connections, learningSegments, growthCandidates, potentialOverlaps,
      sampleSize, maxSynapsesPerSegment)

    connections.growSynapsesToSample(learningSegments, growthCandidates,
                                     maxNewNonzeros, initialPermanence, rng)


  @staticmethod
  def _getMaxSynapseGrowthCounts(connections, learningSegments, growthCandidates,
                                 potentialOverlaps, sampleSize,
                                 maxSynapsesPerSegment):
    """
    Calculate the number of new synapses to attempt to grow for each segment,
    considering the sampleSize and maxSynapsesPerSegment parameters.

    Because the growth candidates are a subset of the active cells, we can't
    actually calculate the number of synapses to grow. We don't know how many
    of the active synapses are to winner cells. We can only calculate the
    maximums.

    @param connections (SparseMatrixConnections)
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
        totalSynapsesPerSegment = connections.mapSegmentsToSynapseCounts(
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
  def _learnOnNewSegments(connections, rng, newSegments, growthCandidates,
                          sampleSize, initialPermanence, maxSynapsesPerSegment):
    """
    Grow synapses on the provided segments.

    Because each segment has no synapses, we don't have to calculate how many
    synapses to grow.

    @param connections (SparseMatrixConnections)
    @param rng (Random)
    @param newSegments (numpy array)
    @param growthCandidates (numpy array)
    """

    numGrow = len(growthCandidates)

    if sampleSize != -1:
      numGrow = min(numGrow, sampleSize)

    if maxSynapsesPerSegment != -1:
      numGrow = min(numGrow, maxSynapsesPerSegment)

    connections.growSynapsesToSample(newSegments, growthCandidates, numGrow,
                                     initialPermanence, rng)


  @staticmethod
  def _punishSegments(connections, segmentsToPunish, activeInput,
                      predictedSegmentDecrement):
    """
    Weaken active synapses on the provided segments.

    @param connections (SparseMatrixConnections)
    @param segmentsToPunish (numpy array)
    @param activeInput (numpy array)
    """
    connections.adjustActiveSynapses(segmentsToPunish, activeInput,
                                     -predictedSegmentDecrement)


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
