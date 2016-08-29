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

import numpy

from nupic.bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix,
                                 GetNTAReal, Random)
from htmresearch.algorithms.extended_temporal_memory import (
  ExtendedTemporalMemory
)

realDType = GetNTAReal()
uintType = "uint32"



class ColumnPooler(object):
  """
  This class constitutes a temporary implementation for a cross-column pooler.
  The implementation goal of this class is to prove basic properties before
  creating a cleaner implementation.
  """

  def __init__(self,
               inputWidth,
               numActiveColumnsPerInhArea=40,
               synPermProximalInc=0.1,
               synPermProximalDec=0.001,
               initialProximalPermanence=0.6,
               columnDimensions=(2048,),
               activationThreshold=13,
               initialPermanence=0.41,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               seed=42):
    """
    Please see ExtendedTemporalMemory for descriptions of common constructor
    parameters.

    Parameters:
    ----------------------------
    @param  inputWidth (int)
            The number of proximal inputs into this layer

    @param  numActiveColumnsPerInhArea (int)
            Number of active cells

    @param  synPermProximalInc (float)
            Permanence increment for proximal synapses

    @param  synPermProximalDec (float)
            Permanence decrement for proximal synapses

    @param  initialProximalPermanence (float)
            Initial permanence value for proximal segments
    """

    self.inputWidth = inputWidth
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.initialProximalPermanence = initialProximalPermanence
    self.connectedPermanence = connectedPermanence
    self.maxNewSynapseCount = maxNewSynapseCount
    self.minThreshold = minThreshold
    self.activeCells = set()
    self._random = Random(seed)

    # Create our own instance of extended temporal memory to handle distal
    # segments.
    self.tm = ExtendedTemporalMemory(
                      columnDimensions=columnDimensions,
                      cellsPerColumn=1,
                      activationThreshold=activationThreshold,
                      initialPermanence=initialPermanence,
                      connectedPermanence=connectedPermanence,
                      minThreshold=minThreshold,
                      maxNewSynapseCount=maxNewSynapseCount,
                      permanenceIncrement=permanenceIncrement,
                      permanenceDecrement=permanenceDecrement,
                      predictedSegmentDecrement=predictedSegmentDecrement,
                      maxSegmentsPerCell=maxSegmentsPerCell,
                      maxSynapsesPerSegment=maxSynapsesPerSegment,
                      seed=seed,
                      learnOnOneCell=False,
    )

    # These sparse matrices will hold the synapses for each proximal segment.
    #
    # proximalPermanences - SparseMatrix with permanence values
    # proximalConnections - SparseBinaryMatrix of connected synapses

    self.proximalPermanences = SparseMatrix(self.numberOfColumns(),
                                               inputWidth)
    self.proximalConnections = SparseBinaryMatrix(inputWidth)
    self.proximalConnections.resize(self.numberOfColumns(), inputWidth)



  def compute(self,
              feedforwardInput=None,
              activeExternalCells=None,
              learn=True):
    """

    Parameters:
    ----------------------------
    @param  feedforwardInput     (set)
            Indices of active input bits

    @param  activeExternalCells  (set)
            Indices of active cells that will form connections to distal
            segments.

    @param  activeApicalCells (set)
            Indices of active cells that will form connections to apical
            segments. Unused in this implementation.

    @param  formInternalConnections (bool)
            If True, cells will form

    @param learn                    (bool)
            If True, we are learning a new object

    """
    if activeExternalCells is None:
      activeExternalCells = set()

    if learn:
      self._computeLearningMode(feedforwardInput=feedforwardInput,
                               lateralInput=activeExternalCells)

    else:
      self._computeInferenceMode(feedforwardInput=feedforwardInput,
                                 lateralInput=activeExternalCells)


  def _computeLearningMode(self, feedforwardInput, lateralInput):
    """
    Learning mode: we are learning a new object. If there is no prior
    activity, we randomly activate 2% of cells and create connections to
    incoming input. If there was prior activity, we maintain it.

    These cells will represent the object and learn distal connections to
    lateral cortical columns.

    Parameters:
    ----------------------------
    @param  feedforwardInput (set)
            Indices of active input bits

    @param  lateralInput (list of lists)
            A list of list of active cells from neighboring columns.
            len(lateralInput) == number of connected neighboring cortical
            columns.

    """
    # If there are no previously active cells, select random subset of cells
    if len(self.activeCells) == 0:
      self.activeCells = set(self._random.shuffle(
            numpy.array(range(self.numberOfCells()),
                        dtype="uint32"))[0:self.numActiveColumnsPerInhArea])

    # else we maintain previous activity, nothing to do.

    # Those cells that remain active will learn on their proximal and distal
    # dendrites as long as there is some input.  If there are no
    # cells active, no learning happens.  This only happens in the very
    # beginning if there has been no bottom up activity at all.
    if len(self.activeCells) > 0:

      # Learn on proximal dendrite if appropriate
      if len(feedforwardInput) > 0:
        self._learnProximal(feedforwardInput, self.activeCells,
                            self.maxNewSynapseCount, self.proximalPermanences,
                            self.proximalConnections,
                            self.initialProximalPermanence,
                            self.synPermProximalInc, self.synPermProximalDec,
                            self.connectedPermanence)

      # Learn on distal dendrites if appropriate
      self.tm.compute(activeColumns=self.activeCells,
                      activeExternalCells=lateralInput,
                      formInternalConnections=False,
                      learn=True)


  def _computeInferenceMode(self, feedforwardInput, lateralInput):
    """
    Inference mode: if there is some feedforward activity, perform
    spatial pooling on it to recognize previously known objects. If there
    is no feedforward activity, maintain previous activity.

    Parameters:
    ----------------------------
    @param  feedforwardInput (set)
            Indices of active input bits

    @param  lateralInput (list of lists)
            A list of list of active cells from neighboring columns.
            len(lateralInput) == number of connected neighboring cortical
            columns.

    """
    # Figure out which cells are active due to feedforward proximal inputs
    # In order to form unions, we keep all cells that are over threshold
    inputVector = numpy.zeros(self.numberOfInputs(), dtype=realDType)
    inputVector[list(feedforwardInput)] = 1
    overlaps = numpy.zeros(self.numberOfColumns(), dtype=realDType)
    self.proximalConnections.rightVecSumAtNZ_fast(inputVector.astype(realDType),
                                                 overlaps)
    overlaps[overlaps < self.minThreshold] = 0
    bottomUpActivity =  set(overlaps.nonzero()[0])

    # If there is insufficient current bottom up activity, we incorporate all
    # previous activity. We set their overlaps so they are sure to win.
    if len(bottomUpActivity) < self.numActiveColumnsPerInhArea:
      bottomUpActivity = bottomUpActivity.union(self.activeCells)
      maxOverlap = overlaps.max()
      overlaps[self.getActiveCells()] = maxOverlap+1

    # Narrow down list of active cells based on lateral activity
    self.activeCells = self._winnersBasedOnLateralActivity(
      bottomUpActivity,
      self.getPredictiveCells(),
      overlaps,
      self.numActiveColumnsPerInhArea
    )

    # Update predictive cells for next time step
    self.tm.compute(activeColumns=self.activeCells,
                    activeExternalCells=lateralInput,
                    formInternalConnections=False,
                    learn=False)


  def numberOfInputs(self):
    """
    Returns the number of inputs into this layer
    """
    return self.inputWidth


  def numberOfColumns(self):
    """
    Returns the number of columns in this layer.
    @return (int) Number of columns
    """
    return self.tm.numberOfColumns()


  def numberOfCells(self):
    """
    Returns the number of cells in this layer.
    @return (int) Number of cells
    """
    return self.tm.numberOfCells()


  def getActiveCells(self):
    """
    Returns the indices of the active cells.
    @return (set) Indices of active cells.
    """
    return self.getCellIndices(self.activeCells)


  @classmethod
  def getCellIndices(cls, cells):
    return [cls.getCellIndex(c) for c in cells]


  @staticmethod
  def getCellIndex(cell):
    return cell


  def numberOfConnectedSynapses(self, cells=None):
    """
    Returns the number of proximal connected synapses on these cells.

    Parameters:
    ----------------------------
    @param  cells (set or list)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())
    n = 0
    for cell in cells:
      n += self.proximalConnections.nNonZerosOnRow(cell)
    return n


  def numberOfSynapses(self, cells=None):
    """
    Returns the number of proximal synapses with permanence>0 on these cells.

    Parameters:
    ----------------------------
    @param  cells (set or list)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())
    n = 0
    for cell in cells:
      n += self.proximalPermanences.nNonZerosOnRow(cell)
    return n


  def numberOfDistalSegments(self, cells):
    """
    Returns the total number of distal segments for these cells.

    Parameters:
    ----------------------------
    @param  cells (set or list)
            Indices of the cells
    """
    n = 0
    for cell in cells:
      n += len(self.tm.connections.segmentsForCell(cell))
    return n


  def numberOfDistalSynapses(self, cells):
    """
    Returns the total number of distal synapses for these cells.

    Parameters:
    ----------------------------
    @param  cells (set or list)
            Indices of the cells
    """
    n = 0
    for cell in cells:
      segments = self.tm.connections.segmentsForCell(cell)
      for segment in segments:
        n += len(self.tm.connections.synapsesForSegment(segment))
    return n


  def reset(self):
    """
    Reset internal states. When learning this signifies we are to learn a
    unique new object.
    """
    self.activeCells = set()
    self.tm.reset()


  def getPredictiveCells(self):
    """
    Get the set of distally predictive cells as a set.

    @return (list) A list containing indices of the current distally predicted
    cells.
    """
    return self.tm.getPredictiveCells()


  def getPredictedActiveCells(self):
    """
    Get the set of cells that were predicted previously then became active

    @return (set) A set containing indices.
    """
    return self.tm.predictedActiveCellsIndices()


  def getConnections(self):
    """
    Get the Connections structure associated with our TM. Beware of using
    this as it is implementation specific and may change.

    @return (object) A Connections object
    """
    return self.tm.connections


  def _learnProximal(self,
             activeInputs, activeCells, maxNewSynapseCount, proximalPermanences,
             proximalConnections, initialPermanence, synPermProximalInc,
             synPermProximalDec, connectedPermanence):
    """
    Learn on proximal dendrites of active cells.  Updates proximalPermanences
    """
    for cell in activeCells:
      cellPermanencesDense = proximalPermanences.getRow(cell)
      cellNonZeroIndices, _ = proximalPermanences.rowNonZeros(cell)
      cellNonZeroIndices = list(cellNonZeroIndices)

      # Get new and existing connections for this segment
      newInputs, existingInputs = self._pickProximalInputsToLearnOn(
        maxNewSynapseCount, activeInputs, cellNonZeroIndices
      )

      # Adjust existing connections appropriately
      # First we decrement all existing permanences
      if len(cellNonZeroIndices) > 0:
        cellPermanencesDense[cellNonZeroIndices] -= synPermProximalDec

      # Then we add inc + dec to existing active synapses
      if len(existingInputs) > 0:
        cellPermanencesDense[existingInputs] += synPermProximalInc + synPermProximalDec

      # Add new connections
      if len(newInputs) > 0:
        cellPermanencesDense[newInputs] += initialPermanence

      # Update proximalPermanences and proximalConnections
      proximalPermanences.setRowFromDense(cell, cellPermanencesDense)
      newConnected = numpy.where(cellPermanencesDense >= connectedPermanence)[0]
      proximalConnections.replaceSparseRow(cell, newConnected)

      cellNonZeroIndices, _ = proximalPermanences.rowNonZeros(cell)


  def _pickProximalInputsToLearnOn(self, newSynapseCount, activeInputs,
                                  cellNonZeros):
    """
    Pick inputs to form proximal connections to. We just randomly subsample
    from activeInputs, regardless of whether they are already connected.

    We return a list of up to newSynapseCount input indices from activeInputs
    that are valid new connections for this cell. We also return a list
    containing all inputs in activeInputs that are already connected to this
    cell.

    Parameters:
    ----------------------------
    @param newSynapseCount  (int)        Number of inputs to pick
    @param cell             (int)        Cell index
    @param activeInputs     (set)        Indices of active inputs
    @param proximalPermanences (sparse)  The matrix of proximal connections

    @return (list, list) Indices of new inputs to connect to, inputs already
                         connected
    """
    candidates = []
    alreadyConnected = []

    # Collect inputs that already have synapses and list of new candidate inputs
    for inputIdx in activeInputs:
      if inputIdx in cellNonZeros:
        alreadyConnected += [inputIdx]
      else:
        candidates += [inputIdx]

    # Select min(newSynapseCount, len(candidates)) new inputs to connect to
    if newSynapseCount >= len(candidates):
      return candidates, alreadyConnected

    else:
      # Pick newSynapseCount cells randomly
      # TODO: we could maybe implement this more efficiently with shuffle.
      inputs = []
      for _ in range(newSynapseCount):
        i = self._random.getUInt32(len(candidates))
        inputs += [candidates[i]]
        candidates.remove(candidates[i])

      return inputs, alreadyConnected


  def _winnersBasedOnLateralActivity(self,
                                     activeCells,
                                     predictiveCells,
                                     overlaps,
                                     targetActiveCells):
    """
    Given the set of cells active due to feedforward input, narrow down the
    list of active cells based on predictions due to previous lateralInput.

    Parameters:
    ----------------------------
    @param    activeCells           (set)
              Indices of cells activated by bottom-up input.

    @param    predictiveCells       (set)
              Indices of cells that are laterally predicted.

    @param    overlaps              (numpy array)
              Bottom up overlap scores for each proximal segment. This is used
              to select additional cells if the narrowed down list contains less
              than targetActiveCells.

    @param    targetActiveCells     (int)
              The number of active cells we want to have active.

    @return (set) List of new winner cell indices

    """
    # No TM accessors that return set so access internal member directly
    predictedActiveCells = activeCells.intersection(predictiveCells)

    # If predicted cells don't intersect at all with active cells, we go with
    # bottom up input. In these cases we can stick with existing active cells
    # and skip the overlap sorting
    if len(predictedActiveCells) == 0:
      predictedActiveCells = activeCells

    # We want to keep all cells that were predicted and currently active due to
    # feedforward input. This set could be larger than our target number of
    # active cells due to unions, which is ok. However if there are insufficient
    # cells active after this intersection, we fill in with those currently
    # active cells that have highest overlap.
    elif len(predictedActiveCells) < targetActiveCells:
      # Don't want to consider cells already chosen
      overlaps[list(predictedActiveCells)] = 0

      # Add in the desired number of cells with highest activity
      numActive = targetActiveCells - len(predictedActiveCells)
      winnerIndices = numpy.argsort(overlaps, kind='mergesort')
      sortedWinnerIndices = winnerIndices[-numActive:][::-1]
      predictedActiveCells = predictedActiveCells.union(set(sortedWinnerIndices))

    return predictedActiveCells
