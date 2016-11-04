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
    self.synPermDistalInc = synPermDistalInc
    self.synPermDistalDec = synPermDistalDec
    self.initialDistalPermanence = initialDistalPermanence
    self.connectedPermanenceDistal = connectedPermanenceDistal
    self.sampleSizeDistal = sampleSizeDistal
    self.minThresholdDistal = minThresholdDistal

    self.activeCells = ()
    self._random = Random(seed)

    # These sparse matrices will hold the synapses for each segment.
    # Each row represents one segment on a cell, so each cell potentially has
    # 1 proximal segment and 1+len(lateralInputWidths) distal segments.
    self.proximalPermanences = SparseMatrix(cellCount, inputWidth)
    self.internalDistalPermanences = SparseMatrix(cellCount, cellCount)
    self.distalPermanences = tuple(SparseMatrix(cellCount, n)
                                   for n in lateralInputWidths)


  def compute(self, feedforwardInput=(), lateralInputs=(), learn=True):
    """
    Runs one time step of the column pooler algorithm.

    @param  feedforwardInput (iterable)
            Indices of active feedforward input bits

    @param  lateralInputs (list of iterables)
            Sets of indices of active lateral input bits, one per lateral layer

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

    @param  lateralInputs (list of iterables)
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
      self._learn(self.proximalPermanences, self._random,
                  self.activeCells, sorted(feedforwardInput),
                  self.sampleSizeProximal, self.initialProximalPermanence,
                  self.synPermProximalInc, self.synPermProximalDec,
                  self.connectedPermanenceProximal)

      # Internal distal learning
      if len(prevActiveCells) > 0:
        self._learn(self.internalDistalPermanences, self._random,
                    self.activeCells, prevActiveCells,
                    self.sampleSizeDistal, self.initialDistalPermanence,
                    self.synPermDistalInc, self.synPermDistalDec,
                    self.connectedPermanenceDistal)

      # External distal learning
      for i, lateralInput in enumerate(lateralInputs):
        self._learn(self.distalPermanences[i], self._random,
                    self.activeCells, sorted(lateralInput),
                    self.sampleSizeDistal, self.initialDistalPermanence,
                    self.synPermDistalInc, self.synPermDistalDec,
                    self.connectedPermanenceDistal)


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

    @param  lateralInputs (list of iterables)
            Sets of indices of active lateral input bits, one per lateral layer
    """

    prevActiveCells = self.activeCells

    # Calculate feedforward support
    overlaps = _rightVecSumAtNZGtThreshold_sparse(
      self.proximalPermanences, sorted(feedforwardInput),
      self.connectedPermanenceProximal)
    feedforwardSupportedCells = set(
      numpy.where(overlaps >= self.minThresholdProximal)[0])

    # Calculate lateral support
    numActiveSegmentsByCell = numpy.zeros(self.cellCount, dtype="int")
    overlaps = _rightVecSumAtNZGtThreshold_sparse(
      self.internalDistalPermanences, prevActiveCells,
      self.connectedPermanenceDistal)
    numActiveSegmentsByCell[overlaps >= self.minThresholdDistal] += 1
    for i, lateralInput in enumerate(lateralInputs):
      overlaps = _rightVecSumAtNZGtThreshold_sparse(
        self.distalPermanences[i], sorted(lateralInput),
        self.connectedPermanenceDistal)
      numActiveSegmentsByCell[overlaps >= self.minThresholdDistal] += 1


    activeCells = []

    # First, activate cells that have feedforward support, ranking them by
    # lateral support.
    orderedCandidates = sorted((cell for cell in feedforwardSupportedCells),
                               key=numActiveSegmentsByCell.__getitem__,
                               reverse=True)
    for _, cells in itertools.groupby(orderedCandidates,
                                      numActiveSegmentsByCell.__getitem__):
      activeCells.extend(cells)
      if len(activeCells) >= self.numActiveColumnsPerInhArea:
        break

    # If necessary, activate cells that were previously active and have lateral
    # support.
    if len(activeCells) < self.numActiveColumnsPerInhArea:
      orderedCandidates = sorted((cell for cell in prevActiveCells
                                  if cell not in feedforwardSupportedCells
                                  and numActiveSegmentsByCell[cell] > 0),
                                 key=numActiveSegmentsByCell.__getitem__,
                                 reverse=True)
      for _, cells in itertools.groupby(orderedCandidates,
                                        numActiveSegmentsByCell.__getitem__):
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

    n = 0

    for cell in cells:
      if self.internalDistalPermanences.nNonZerosOnRow(cell) > 0:
        n += 1

      for permanences in self.distalPermanences:
        if permanences.nNonZerosOnRow(cell) > 0:
          n += 1

    return n


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

    n = _countWhereGreaterEqualInRows(self.internalDistalPermanences, cells,
                                      self.connectedPermanenceDistal)

    for permanences in self.distalPermanences:
      n += _countWhereGreaterEqualInRows(permanences, cells,
                                         self.connectedPermanenceDistal)

    return n


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
    n = 0
    for cell in cells:
      n += self.internalDistalPermanences.nNonZerosOnRow(cell)

      for permanences in self.distalPermanences:
        n += permanences.nNonZerosOnRow(cell)
    return n


  def reset(self):
    """
    Reset internal states. When learning this signifies we are to learn a
    unique new object.
    """
    self.activeCells = ()


  @staticmethod
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
