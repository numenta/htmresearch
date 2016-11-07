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

import itertools

import numpy

from nupic.bindings.math import SparseMatrix, GetNTAReal, Random



class ColumnPooler(object):
  """
  This class constitutes a temporary implementation for a cross-column pooler.
  The implementation goal of this class is to prove basic properties before
  creating a cleaner implementation.
  """

  def __init__(self,
               inputWidth,
               lateralInputWidths=(),
               cellCount=4096,
               numActiveColumnsPerInhArea=40,

               lateralConnectionsImpl="PairwiseSegments",

               # Proximal
               synPermProximalInc=0.1,
               synPermProximalDec=0.001,
               initialProximalPermanence=0.6,
               sampleSizeProximal=20,
               minThresholdProximal=10,
               connectedPermanenceProximal=0.50,

               # Distal
               synPermDistalInc=0.1,
               synPermDistalDec=0.001,
               initialDistalPermanence=0.6,
               sampleSizeDistal=20,
               minThresholdDistal=13,
               connectedPermanenceDistal=0.50,

               seed=42):
    """
    Parameters:
    ----------------------------
    @param  inputWidth (int)
            The number of proximal inputs into this layer

    @param  lateralInputWidths (list of ints)
            The number of input bits in each lateral input layer.

    @param  numActiveColumnsPerInhArea (int)
            Target number of active cells

    @param  synPermProximalInc (float)
            Permanence increment for proximal synapses

    @param  synPermProximalDec (float)
            Permanence decrement for proximal synapses

    @param  initialProximalPermanence (float)
            Initial permanence value for proximal synapses

    @param  sampleSizeProximal (int)
            Number of proximal synapses a cell should grow to each feedforward
            pattern, or -1 to connect to every active bit

    @param  minThresholdProximal (int)
            Number of active synapses required for a cell to have feedforward
            support

    @param  connectedPermanenceProximal (float)
            Permanence required for a proximal synapse to be connected

    @param  synPermDistalInc (float)
            Permanence increment for distal synapses

    @param  synPermDistalDec (float)
            Permanence decrement for distal synapses

    @param  sampleSizeDistal (int)
            Number of distal synapses a cell should grow to each lateral
            pattern, or -1 to connect to every active bit

    @param  initialDistalPermanence (float)
            Initial permanence value for distal synapses

    @param  minThresholdDistal (int)
            Number of active synapses required to activate a distal segment

    @param  connectedPermanenceDistal (float)
            Permanence required for a distal synapse to be connected

    @param  seed (int)
            Random number generator seed
    """

    self.inputWidth = inputWidth
    self.cellCount = cellCount
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.initialProximalPermanence = initialProximalPermanence
    self.connectedPermanenceProximal = connectedPermanenceProximal
    self.sampleSizeProximal = sampleSizeProximal
    self.minThresholdProximal = minThresholdProximal

    self.activeCells = ()
    self._random = Random(seed)

    # These sparse matrices will hold the synapses for each segment.
    # Each row represents one segment on a cell, so each cell potentially has
    # 1 proximal segment and 1+len(lateralInputWidths) distal segments.
    self.proximalPermanences = SparseMatrix(cellCount, inputWidth)

    if lateralConnectionsImpl == "PairwiseSegments":
      self.lateralConnections = PairwiseSegments(
        cellCount,
        lateralInputWidths,
        synPermDistalInc,
        synPermDistalDec,
        initialDistalPermanence,
        sampleSizeDistal,
        minThresholdDistal,
        connectedPermanenceDistal,
        self._random
      )
    elif lateralConnectionsImpl == "TwoSegmentsPerCell":
      self.lateralConnections = TwoSegmentsPerCell(
        cellCount,
        sum(lateralInputWidths),
        synPermDistalInc,
        synPermDistalDec,
        initialDistalPermanence,
        sampleSizeDistal,
        minThresholdDistal,
        connectedPermanenceDistal,
        self._random
      )
    else:
      raise ValueError("Unknown lateralConnectionsImpl", lateralConnectionsImpl)


  def compute(self, feedforwardInput=(), lateralInputs=(), learn=True):
    """
    Runs one time step of the column pooler algorithm.

    @param  feedforwardInput (iterable)
            Indices of active feedforward input bits

    @param  lateralInputs (varying type based on lateralConnectionsImpl)
            Input to the lateralConnections.

    @param  learn (bool)
            If True, we are learning a new object
    """

    if learn:
      self._computeLearningMode(feedforwardInput, lateralInputs)
    else:
      self._computeInferenceMode(feedforwardInput, lateralInputs)


  def _computeLearningMode(self, feedforwardInput, lateralInputs):
    """
    Learning mode: we are learning a new object. If there is no prior
    activity, we randomly activate 1% of cells and create connections to
    incoming input. If there was prior activity, we maintain it.

    These cells will represent the object and learn distal connections to each
    other and to lateral cortical columns.

    Parameters:
    ----------------------------
    @param  feedforwardInput (iterable)
            List of indices of active feedforward input bits

    @param  lateralInputs (varying type based on lateralConnectionsImpl)
            Input to the lateralConnections.
            Lists of indices of active lateral input bits, one per lateral layer
    """

    prevActiveCells = self.activeCells

    # If there are no previously active cells, select random subset of cells.
    # Else we maintain previous activity.
    if len(self.activeCells) == 0:
      self.activeCells = sorted(_sampleRange(self._random, 0,
                                             self.numberOfCells(),
                                             1,
                                             self.numActiveColumnsPerInhArea))

    if len(feedforwardInput) > 0:
      # Proximal learning
      _learn(self.proximalPermanences, self._random,
             self.activeCells, sorted(feedforwardInput),
             self.sampleSizeProximal, self.initialProximalPermanence,
             self.synPermProximalInc, self.synPermProximalDec,
             self.connectedPermanenceProximal)

      self.lateralConnections.learn(self.activeCells, prevActiveCells,
                                    lateralInputs)


  def _computeInferenceMode(self, feedforwardInput, lateralInputs):
    """
    Inference mode: if there is some feedforward activity, perform
    spatial pooling on it to recognize previously known objects, then use
    lateral activity to activate a subset of the cells with feedforward
    support. If there is no feedforward activity, use lateral activity to
    activate a subset of the previous active cells.

    Parameters:
    ----------------------------
    @param  feedforwardInput (iterable)
            Indices of active feedforward input bits

    @param  lateralInputs (varying type based on lateralConnectionsImpl)
            Input to lateralConnections
    """

    prevActiveCells = self.activeCells

    # Calculate feedforward support
    overlaps = _rightVecSumAtNZGtThreshold_sparse(
      self.proximalPermanences, sorted(feedforwardInput),
      self.connectedPermanenceProximal)
    feedforwardSupportedCells = set(
      numpy.where(overlaps >= self.minThresholdProximal)[0])

    # Calculate lateral support
    lateralScores = self.lateralConnections.depolarizeCells(prevActiveCells,
                                                            lateralInputs)

    activeCells = []

    # First, activate cells that have feedforward support, ranking them by
    # lateral support.
    orderedCandidates = sorted((cell for cell in feedforwardSupportedCells),
                               key=lateralScores.__getitem__,
                               reverse=True)
    for _, cells in itertools.groupby(orderedCandidates,
                                      lateralScores.__getitem__):
      activeCells.extend(cells)
      if len(activeCells) >= self.numActiveColumnsPerInhArea:
        break

    # If necessary, activate cells that were previously active and have lateral
    # support.
    if len(activeCells) < self.numActiveColumnsPerInhArea:
      orderedCandidates = sorted((cell for cell in prevActiveCells
                                  if cell not in feedforwardSupportedCells
                                  and lateralScores[cell] > 0),
                                 key=lateralScores.__getitem__,
                                 reverse=True)
      for _, cells in itertools.groupby(orderedCandidates,
                                        lateralScores.__getitem__):
        activeCells.extend(cells)
        if len(activeCells) >= self.numActiveColumnsPerInhArea:
          break

    self.activeCells = sorted(activeCells)


  def numberOfInputs(self):
    """
    Returns the number of inputs into this layer
    """
    return self.inputWidth


  def numberOfCells(self):
    """
    Returns the number of cells in this layer.
    @return (int) Number of cells
    """
    return self.cellCount


  def getActiveCells(self):
    """
    Returns the indices of the active cells.
    @return (list) Indices of active cells.
    """
    return self.activeCells


  def numberOfConnectedProximalSynapses(self, cells=None):
    """
    Returns the number of proximal connected synapses on these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    return _countWhereGreaterEqualInRows(self.proximalPermanences, cells,
                                         self.connectedPermanenceProximal)


  def numberOfProximalSynapses(self, cells=None):
    """
    Returns the number of proximal synapses with permanence>0 on these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    n = 0
    for cell in cells:
      n += self.proximalPermanences.nNonZerosOnRow(cell)
    return n


  def numberOfDistalSegments(self, cells=None):
    """
    Returns the total number of distal segments for these cells.

    A segment "exists" if its row in the matrix has any permanence values > 0.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    return self.lateralConnections.numberOfSegments(cells)


  def numberOfConnectedDistalSynapses(self, cells=None):
    """
    Returns the number of connected distal synapses on these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    return self.lateralConnections.numberOfConnectedSynapses(cells)


  def numberOfDistalSynapses(self, cells=None):
    """
    Returns the total number of distal synapses for these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    return self.lateralConnections.numberOfSynapses(cells)


  def reset(self):
    """
    Reset internal states. When learning this signifies we are to learn a
    unique new object.
    """
    self.activeCells = ()



class TwoSegmentsPerCell(object):
  """
  An implementation of lateral connections that uses two dendrite segments per
  cell. One segment is dedicated to connections within this cortical column,
  and the other is dedicated to external connections.
  """
  def __init__(self,
               cellCount,
               lateralInputWidth,
               synPermInc,
               synPermDec,
               initialPermanence,
               sampleSize,
               minThreshold,
               connectedPermanence,
               rng):
    self.synPermInc = synPermInc
    self.synPermDec = synPermDec
    self.initialPermanence = initialPermanence
    self.sampleSize = sampleSize
    self.minThreshold = minThreshold
    self.connectedPermanence = connectedPermanence
    self.rng = rng

    self.internalPermanences = SparseMatrix(cellCount, cellCount)
    self.externalPermanences = SparseMatrix(cellCount, lateralInputWidth)


  def learn(self, learningCells, prevActiveCells, lateralInputs):
    # Internal learning
    if len(prevActiveCells) > 0:
      _learn(self.internalPermanences, self.rng,
             learningCells, prevActiveCells,
             self.sampleSize, self.initialPermanence,
             self.synPermInc, self.synPermDec,
             self.connectedPermanence)

    # External learning
    _learn(self.externalPermanences, self.rng,
           learningCells, sorted(lateralInputs),
           self.sampleSize, self.initialPermanence,
           self.synPermInc, self.synPermDec,
           self.connectedPermanence)


  def depolarizeCells(self, prevActiveCells, lateralInputs):
    """
    @param  lateralInputs (list of iterables)
            Sets of indices of active lateral input bits, one per lateral layer

    @returns (numpy array)
    A score for every cell.
    """

    numActiveSegmentsByCell = numpy.zeros(self.internalPermanences.nRows(),
                                          dtype="uint32")
    overlaps = _rightVecSumAtNZGtThreshold_sparse(
      self.internalPermanences, prevActiveCells,
      self.connectedPermanence)
    numActiveSegmentsByCell[overlaps >= self.minThreshold] += 1

    overlaps = _rightVecSumAtNZGtThreshold_sparse(
      self.externalPermanences, lateralInputs,
      self.connectedPermanence)
    numActiveSegmentsByCell[overlaps >= self.minThreshold] += 1

    return numActiveSegmentsByCell


  def numberOfSegments(self, cells):
    n = 0

    for cell in cells:
      if self.internalPermanences.nNonZerosOnRow(cell) > 0:
        n += 1

      if self.externalPermanences.nNonZerosOnRow(cell) > 0:
        n += 1

    return n


  def numberOfConnectedSynapses(self, cells):
    n = _countWhereGreaterEqualInRows(self.internalPermanences, cells,
                                      self.connectedPermanence)

    n += _countWhereGreaterEqualInRows(self.externalPermanences, cells,
                                       self.connectedPermanence)

    return n


  def numberOfSynapses(self, cells):
    n = 0
    for cell in cells:
      n += self.internalPermanences.nNonZerosOnRow(cell)
      n += self.externalPermanences.nNonZerosOnRow(cell)
    return n



class PairwiseSegments(object):
  """
  Each cell in a cortical column will devote one dendrite segment to each
  cortical column that it connects to laterally. So, one segment will be used
  for distal connections within the layer, and an additional segment will be
  used for each external layer.
  """

  def __init__(self,
               cellCount,
               lateralInputWidths,
               synPermInc,
               synPermDec,
               initialPermanence,
               sampleSize,
               minThreshold,
               connectedPermanence,
               rng):

    self.lateralInputWidths = lateralInputWidths

    self.synPermInc = synPermInc
    self.synPermDec = synPermDec
    self.initialPermanence = initialPermanence
    self.sampleSize = sampleSize
    self.minThreshold = minThreshold
    self.connectedPermanence = connectedPermanence
    self.rng = rng

    self.internalPermanences = SparseMatrix(cellCount, cellCount)
    self.externalPermanences = tuple(SparseMatrix(cellCount, n)
                                     for n in lateralInputWidths)


  def learn(self, learningCells, prevActiveCells, lateralInputs):
    # Internal learning
    if len(prevActiveCells) > 0:
      _learn(self.internalPermanences, self.rng,
             learningCells, prevActiveCells,
             self.sampleSize, self.initialPermanence,
             self.synPermInc, self.synPermDec,
             self.connectedPermanence)

    # External learning
    for i, lateralInput in enumerate(lateralInputs):
      _learn(self.externalPermanences[i], self.rng,
             learningCells, sorted(lateralInput),
             self.sampleSize, self.initialPermanence,
             self.synPermInc, self.synPermDec,
             self.connectedPermanence)


  def depolarizeCells(self, prevActiveCells, lateralInputs):
    """
    @param  lateralInput (list of iterables)
            Sets of indices of active lateral input bits, one per lateral layer

    @returns (numpy array)
    A score for every cell.
    """

    numActiveSegmentsByCell = numpy.zeros(self.internalPermanences.nRows(),
                                          dtype="uint32")
    overlaps = _rightVecSumAtNZGtThreshold_sparse(
      self.internalPermanences, prevActiveCells,
      self.connectedPermanence)
    numActiveSegmentsByCell[overlaps >= self.minThreshold] += 1

    for i, lateralInput in enumerate(lateralInputs):
      overlaps = _rightVecSumAtNZGtThreshold_sparse(
        self.externalPermanences[i], sorted(lateralInput),
        self.connectedPermanence)
      numActiveSegmentsByCell[overlaps >= self.minThreshold] += 1

    return numActiveSegmentsByCell


  def numberOfSegments(self, cells):
    n = 0

    for cell in cells:
      if self.internalPermanences.nNonZerosOnRow(cell) > 0:
        n += 1

      for permanences in self.externalPermanences:
        if permanences.nNonZerosOnRow(cell) > 0:
          n += 1

    return n


  def numberOfConnectedSynapses(self, cells):
    n = _countWhereGreaterEqualInRows(self.internalPermanences, cells,
                                      self.connectedPermanence)

    for permanences in self.externalPermanences:
      n += _countWhereGreaterEqualInRows(permanences, cells,
                                         self.connectedPermanence)

    return n


  def numberOfSynapses(self, cells):
    n = 0
    for cell in cells:
      n += self.internalPermanences.nNonZerosOnRow(cell)

      for permanences in self.externalPermanences:
        n += permanences.nNonZerosOnRow(cell)
    return n


def _learn(# mutated args
           permanences, rng,

           # activity
           activeCells, activeInput,

           # configuration
           sampleSize, initialPermanence, permanenceIncrement,
           permanenceDecrement, connectedPermanence):
  """
  For each active cell, reinforce active synapses, punish inactive synapses,
  and grow new synapses to a subset of the active input bits that the cell
  isn't already connected to.

  Parameters:
  ----------------------------
  @param  permanences (SparseMatrix)
          Matrix of permanences, with cells as rows and inputs as columns

  @param  rng (Random)
          Random number generator

  @param  activeCells (sorted sequence)
          Sorted list of the cells that are learning

  @param  activeInput (sorted sequence)
          Sorted list of active bits in the input

  For remaining parameters, see the __init__ docstring.
  """

  permanences.incrementNonZerosOnOuter(
    activeCells, activeInput, permanenceIncrement)
  permanences.incrementNonZerosOnRowsExcludingCols(
    activeCells, activeInput, -permanenceDecrement)
  permanences.clipRowsBelowAndAbove(
    activeCells, 0.0, 1.0)
  if sampleSize == -1:
    permanences.setZerosOnOuter(
      activeCells, activeInput, initialPermanence)
  else:
    permanences.increaseRowNonZeroCountsOnOuterTo(
      activeCells, activeInput, sampleSize, initialPermanence, rng)


#
# Functionality that could be added to the C code or bindings
#

def _sampleRange(rng, start, end, step, k):
  """
  Equivalent to:

  random.sample(xrange(start, end, step), k)

  except it uses our random number generator.

  This wouldn't need to create the arange if it were implemented in C.
  """
  array = numpy.empty(k, dtype="uint32")
  rng.sample(numpy.arange(start, end, step, dtype="uint32"), array)
  return array


def _rightVecSumAtNZGtThreshold_sparse(sparseMatrix,
                                       sparseBinaryArray,
                                       threshold):
  """
  Like rightVecSumAtNZGtThreshold, but it supports sparse binary arrays.

  @param sparseBinaryArray (sorted sequence)
  A sorted list of indices.

  Note: this Python implementation doesn't require the list to be sorted, but
  an eventual C implementation would.
  """
  denseArray = numpy.zeros(sparseMatrix.nCols(), dtype=GetNTAReal())
  denseArray[sparseBinaryArray] = 1
  return sparseMatrix.rightVecSumAtNZGtThreshold(denseArray, threshold)


def _countWhereGreaterEqualInRows(sparseMatrix, rows, threshold):
  """
  Like countWhereGreaterOrEqual, but for an arbitrary selection of rows, and
  without any column filtering.
  """
  return sum(sparseMatrix.countWhereGreaterOrEqual(row, row+1,
                                                   0, sparseMatrix.nCols(),
                                                   threshold)
             for row in rows)
