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

import scipy.sparse as sparse

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory

realDType = numpy.float32
uintType = "uint32"



class ColumnPooler(ExtendedTemporalMemory):
  """
  This class constitutes a temporary implementation for a cross-column pooler.
  """

  def __init__(self,
               inputWidth,
               numActiveColumnsPerInhArea=40,
               synPermProximalInc=0.1,
               synPermProximalDec=0.001,
               **kwargs):
    """
    Please see ExtendedTemporalMemory for descriptions of common constructor
    parameters.

    @param inputWidth                 (int) The number of proximal inputs into
                                            this layer
    @param numActiveColumnsPerInhArea (int) Number of active cells
    @param synPermProximalInc       (float) Permanence increment for proximal
                                            synapses
    @param synPermProximalDec       (float) Permanence decrement for proximal
                                            synapses
    """

    # Override: we only support one cell per column for now
    kwargs['cellsPerColumn'] = 1
    super(ColumnPooler, self).__init__(**kwargs)

    self.inputWidth = inputWidth
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.previousOverlaps = None

    # This sparse matrix will hold the proximal segment for each cell.
    # self.proximalSegment[cell] = sparse vector of permanence values for cell
    # self.proximalSegments[cell, i] = permanence for the i'th input to cell
    self.proximalSegments = sparse.lil_matrix((self.numberOfCells(),inputWidth),
                                             dtype=realDType)


  def compute(self,
              feedforwardInput=None,
              activeExternalCells=None,
              activeApicalCells=None,
              formInternalConnections=True,
              learn=True):
    """

    @param feedforwardInput     (set) Indices of active input bits

    @param learn                If True, we are learning a new object

    """
    if activeExternalCells is None:
      activeExternalCells = set()

    if learn:
      self._computeLearningMode(feedforwardInput=feedforwardInput,
                               lateralInput=activeExternalCells)


  def _computeLearningMode(self, feedforwardInput, lateralInput):
    """
    Computes when learning new object

    Learning mode: we are learning a new object. If there is no prior
    activity, we randomly activate 2% of cells and create connections to
    incoming input. If there was prior activity, we maintain it.

    These cells will represent the object and learn distal connections to
    lateral cortical columns.

    @param feedforwardInput     (set) Indices of active input bits

    @param lateralInput         A list of list of active cells from neighboring
                                columns. len(lateralInput) == number of
                                connected neighboring cortical columns.

    """

    # Figure out which cells are active due to feedforward proximal inputs
    ffInput = numpy.zeros(self.numberOfInputs())
    ffInput[list(feedforwardInput)] = 1
    overlaps = self.proximalSegments.dot(ffInput)

    # If we have bottom up input and there are no previously active cells,
    # select a random subset of the cells
    if overlaps.max() < self.minThreshold:
      if len(self.activeCells) == 0:
        # No previously active cells, need to create new SDR
        self.activeCells = set(self._random.shuffle(
              numpy.array(range(self.numberOfCells()),
                          dtype="uint32"))[0:self.numActiveColumnsPerInhArea])

    # else: we maintain previous activity

    # Compute distal segment activity for each cell
    if len(lateralInput) > 0:
      # Figure out distal input into active cells
      pass

    # Reconcile and select the cells with sufficient bottom up activity plus
    # maximal lateral activity
    # print "Max overlap=", overlaps.max()

    # Calculate winners using stable sort algorithm (mergesort)
    # for compatibility with C++
    if overlaps.max() >= self.minThreshold:
      winnerIndices = numpy.argsort(overlaps, kind='mergesort')
      sortedWinnerIndices = winnerIndices[-self.numActiveColumnsPerInhArea:][::-1]
      print sortedWinnerIndices


    # Those cells that remain active will learn on their proximal and distal
    # dendrites as long as there is some input.  If there are no cells active,
    # learning happens.
    if len(self.activeCells) > 0:

      # Learn on proximal dendrite if appropriate
      if len(feedforwardInput) > 0:
        self._learnProximal(feedforwardInput, self.activeCells,
                            self.maxNewSynapseCount)

      # Learn on distal dendrites if appropriate
      if len(lateralInput) > 0:
        self._learnDistal(lateralInput, self.activeCells)


  def computeInferenceMode(self, lateralInput, learn):
    """
    Inference mode: if there is some feedforward activity, perform
    spatial pooling on it to recognize previously known objects. If there
    is no feedforward activity, maintain previous activity.


    @param feedforwardInput     A numpy array of 0's and 1's that comprises
                                the input (typically the active cells in TM)
    @param lateralInput         A list of list of active cells from neighboring
                                columns. len(lateralInput) == number of
                                connected neighboring cortical columns.

    @param learn                If true, learn on distal segments.

    """
    pass


  def numberOfInputs(self):
    """
    Returns the number of inputs into this layer
    @return (int) Number of inputs
    """
    return self.inputWidth


  def _learnProximal(self, activeInputs, activeCells, maxNewSynapseCount):
    """Learn on proximal dendrites of active cells"""
    # print "activeInputs=",activeInputs
    # print "activeCells=",activeCells
    activeCellsShuffled = numpy.array(list(activeCells), dtype="uint32")
    for cell in activeCells:
      # For existing connections, increment/decrement permanences appropriately

      # Sample up to maxNewSynapseCount inputs and add to current segment with
      # permanence initialPermanence.

      # self._random.shuffle(activeCellsShuffled)[0:maxNewSynapseCount]
      # print activeCellsShuffled
      # self.proximalSegments[cell,]
      pass


  def _learnDistal(self, lateralInput, activeCells):
    pass


  def _pickProximalInputsToLearnOn(self, newSynapseCount, cell, activeInputs,
                                  proximalSegments):
    """
    Pick inputs to form proximal connections to. We return

    @param newSynapseCount  (int)        Number of inputs to pick
    @param cell             (int)        Cell index
    @param activeInputs     (set)        Indices of active inputs
    @param proximalSegments (sparse)     The matrix of proximal connections

    @return (list) Indices of cells picked
    """
    candidates = set(activeInputs)

    # Remove input indices that are already connected to this cell
    for inputIdx in proximalSegments[cell].nonzero()[1]:
      if inputIdx in candidates:
        candidates.remove(inputIdx)

    # Select min(newSynapseCount, len(candidates)) new inputs to connect to
    if newSynapseCount >= len(candidates):
      return candidates

    else:
      candidates = sorted(candidates)

      # Pick newSynapseCount cells randomly
      # TODO: we could implement this more efficiently with shuffle.
      inputs = set()
      for _ in range(newSynapseCount):
        i = self._random.getUInt32(len(candidates))
        inputs.add(candidates[i])
        del candidates[i]

      return inputs


