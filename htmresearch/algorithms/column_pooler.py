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
  The implementation goal of this class is to prove basic properties before
  creating a cleaner implementation.
  """

  def __init__(self,
               inputWidth,
               numActiveColumnsPerInhArea=40,
               synPermProximalInc=0.1,
               synPermProximalDec=0.001,
               initialProximalPermanence=0.51,
               **kwargs):
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

    # Override: we only support one cell per column for now
    kwargs['cellsPerColumn'] = 1
    super(ColumnPooler, self).__init__(**kwargs)

    self.inputWidth = inputWidth
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.initialProximalPermanence = initialProximalPermanence
    self.previousOverlaps = None

    # These sparse matrices will hold the synapses for each proximal segment.
    #
    # proximalPermanences[cell] = sparse vector of permanence values for cell
    # proximalPermanences[cell, i] = permanence for the i'th input to cell
    #
    # proximalConnections[cell] = sparse vector of connected inputs into cell
    # proximalConnections[cell, i] = 1 iff the permanence for the i'th input to
    #                                 cell is above connectedPerm
    self.proximalPermanences = sparse.lil_matrix(
      (self.numberOfCells(), inputWidth), dtype=realDType)
    self.proximalConnections = sparse.lil_matrix(
      (self.numberOfCells(),inputWidth), dtype=realDType)


  def compute(self,
              feedforwardInput=None,
              activeExternalCells=None,
              activeApicalCells=None,
              formInternalConnections=True,
              learn=True):
    """

    Parameters:
    ----------------------------
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

    # Incorporate distal segment activity and update list of active cells
    self.activeCells = self._winnersBasedOnLateralActivity(
      self.activeCells, lateralInput, self.minThreshold
    )

    # Those cells that remain active will learn on their proximal and distal
    # dendrites as long as there is some input.  If there are no
    # cells active, no learning happens.
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
      if len(lateralInput) > 0:
        self._learnDistal(lateralInput, self.activeCells)


  def computeInferenceMode(self, feedforwardInput, lateralInput, learn):
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

    @param  learn (bool)
            If true, learn on distal segments.

    """
    # Figure out which cells are active due to feedforward proximal inputs
    ffInput = numpy.zeros(self.numberOfInputs())
    ffInput[list(feedforwardInput)] = 1
    overlaps = self.proximalConnections.dot(ffInput)


  def numberOfInputs(self):
    """
    Returns the number of inputs into this layer
    """
    return self.inputWidth


  def numberOfConnectedSynapses(self, cells):
    """
    Returns the number of connected synapses on these cells.

    Parameters:
    ----------------------------
    @param  cells (set or list)
            Indices of the cells
    """
    return self.proximalConnections[list(cells)].nnz


  def numberOfSynapses(self, cells):
    """
    Returns the number of synapses with permanence>0 on these cells.

    Parameters:
    ----------------------------
    @param  cells (set or list)
            Indices of the cells
    """
    return self.proximalPermanences[list(cells)].nnz


  def _learnProximal(self,
             activeInputs, activeCells, maxNewSynapseCount, proximalPermanences,
             proximalConnections,
             initialPermanence, synPermProximalInc, synPermProximalDec,
             connectedPermanence):
    """
    Learn on proximal dendrites of active cells.  Updates proximalPermanences
    """
    for cell in activeCells:

      # Get new and existing connections for this segment
      newInputs, existingInputs = self._pickProximalInputsToLearnOn(
        maxNewSynapseCount, cell, activeInputs, proximalPermanences
      )

      # Adjust existing connections appropriately
      # First we decrement all permanences
      nz = proximalPermanences[cell].nonzero()[1]  # slowest line??
      if len(nz) > 0:
        t = proximalPermanences[cell, nz].todense()
        t -= synPermProximalDec
        proximalPermanences[cell, nz] = t

      # Then we add inc + dec to existing active synapses
      if len(existingInputs) > 0:
        t = proximalPermanences[cell, existingInputs].todense()
        t += synPermProximalInc + synPermProximalDec
        proximalPermanences[cell, existingInputs] = t

      # Add new connections
      if len(newInputs) > 0:
        proximalPermanences[cell, newInputs] = initialPermanence

      # Update proximalConnections
      proximalConnections[cell,:] = 0
      nz = (proximalPermanences[cell]>=connectedPermanence).nonzero()[1]
      proximalConnections[cell,nz] = 1.0


  def _winnersBasedOnLateralActivity(self,
                                     activeCells,
                                     lateralInput,
                                     minThreshold):
    """
    Incorporate effect of lateral activity, if any, and update the set of
    winners.

    UNIMPLEMENTED

    @return (set) list of new winner cell indices
    """
    if len(lateralInput) == 0:
      return activeCells

    sortedWinnerIndices = activeCells

    # Figure out distal input into active cells

    # TODO: Reconcile and select the cells with sufficient bottom up activity
    # plus maximal lateral activity

    # Calculate winners using stable sort algorithm (mergesort)
    # for compatibility with C++
    # if overlaps.max() >= minThreshold:
    #   winnerIndices = numpy.argsort(overlaps, kind='mergesort')
    #   sortedWinnerIndices = winnerIndices[
    #                         -self.numActiveColumnsPerInhArea:][::-1]
    #   sortedWinnerIndices = set(sortedWinnerIndices)

    return sortedWinnerIndices


  def _pickProximalInputsToLearnOn(self, newSynapseCount, cell, activeInputs,
                                  proximalPermanences):
    """
    Pick inputs to form proximal connections to. We return a list of up to
    newSynapseCount input indices from activeInputs that are valid new
    connections for this cell. We also return a list containing all inputs
    in activeInputs that are already connected to this cell.

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

    # Collect inputs already connected or new candidates
    nz = proximalPermanences[cell].nonzero()[1]  # Slowest line - need it?
    for inputIdx in activeInputs:
      if inputIdx in nz:
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


  def _learnDistal(self, lateralInput, activeCells):
    pass


