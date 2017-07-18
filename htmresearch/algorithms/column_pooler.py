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
               sdrSize=40,

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
               activationThresholdDistal=13,
               connectedPermanenceDistal=0.50,
               distalSegmentInhibitionFactor=0.999,
               inertiaFactor=1.,

               seed=42):
    """
    Parameters:
    ----------------------------
    @param  inputWidth (int)
            The number of bits in the feedforward input

    @param  lateralInputWidths (list of ints)
            The number of bits in each lateral input

    @param  sdrSize (int)
            The number of active cells in an object SDR

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

    @param  activationThresholdDistal (int)
            Number of active synapses required to activate a distal segment

    @param  connectedPermanenceDistal (float)
            Permanence required for a distal synapse to be connected

    @param  distalSegmentInhibitionFactor (float)
            The proportion of the highest number of active lateral segments
            necessary for a cell not to be inhibited (must be <1).

    @param  inertiaFactor (float)
            The proportion of previously active cells that remain
            active in the next timestep due to inertia (in the absence of
            inhibition).

    @param  seed (int)
            Random number generator seed
    """

    assert distalSegmentInhibitionFactor > 0.0
    assert distalSegmentInhibitionFactor < 1.0

    self.inputWidth = inputWidth
    self.cellCount = cellCount
    self.sdrSize = sdrSize
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
    self.activationThresholdDistal = activationThresholdDistal
    self.distalSegmentInhibitionFactor = distalSegmentInhibitionFactor
    self.inertiaFactor = inertiaFactor
    self.counter = 0

    self.activeCells = numpy.empty(0, dtype="uint32")
    self._random = Random(seed)

    # These sparse matrices will hold the synapses for each segment.
    # Each row represents one segment on a cell, so each cell potentially has
    # 1 proximal segment and 1+len(lateralInputWidths) distal segments.
    self.proximalPermanences = SparseMatrix(cellCount, inputWidth)
    self.internalDistalPermanences = SparseMatrix(cellCount, cellCount)
    self.distalPermanences = tuple(SparseMatrix(cellCount, n)
                                   for n in lateralInputWidths)

    self.useInertia=True


  def compute(self, feedforwardInput=(), lateralInputs=(),
              feedforwardGrowthCandidates=None, learn=True, bursting = False):
    """
    Runs one time step of the column pooler algorithm.

    @param  feedforwardInput (sequence)
            Sorted indices of active feedforward input bits

    @param  lateralInputs (list of sequences)
            For each lateral layer, a list of sorted indices of active lateral
            input bits

    @param  feedforwardGrowthCandidates (sequence or None)
            Sorted indices of feedforward input bits that active cells may grow
            new synapses to. If None, the entire feedforwardInput is used.

    @param  learn (bool)
            If True, we are learning a new object
    """
    if not learn:
      self._computeInferenceMode(feedforwardInput, lateralInputs)
    else:
      if bursting:
        self.activeCells = numpy.asarray([], dtype = "int")
        self._computeLearningMode(feedforwardInput, lateralInputs,
                                  feedforwardGrowthCandidates)
      elif numpy.abs(len(self.activeCells) - self.sdrSize) > (0.25*self.sdrSize):
        self._computeInferenceMode(feedforwardInput, lateralInputs)
        if len(self.activeCells) < (0.75*self.sdrSize):
          self.activeCells = numpy.asarray([], dtype = "int")
          self._computeLearningMode(feedforwardInput, lateralInputs,
                                    feedforwardGrowthCandidates)
      else:
        self._computeLearningMode(feedforwardInput, lateralInputs,
                                  feedforwardGrowthCandidates)

  def _computeLearningMode(self, feedforwardInput, lateralInputs,
                           feedforwardGrowthCandidates):
    """
    Learning mode: we are learning a new object. If there is no prior
    activity, we randomly activate 'sdrSize' cells and create connections to
    incoming input. If there was prior activity, we maintain it.

    These cells will represent the object and learn distal connections to each
    other and to lateral cortical columns.

    Parameters:
    ----------------------------
    @param  feedforwardInput (sequence)
            Sorted indices of active feedforward input bits

    @param  lateralInputs (list of sequences)
            For each lateral layer, a list of sorted indices of active lateral
            input bits

    @param  feedforwardGrowthCandidates (sequence or None)
            Sorted indices of feedforward input bits that the active cells may
            grow new synapses to.  This is assumed to be the predicted active
            cells of the input layer.
    """
    # If we have a large amount of unpredicted input, we should avoid learning.
    # In this case, it is possible that we are on a sequence which has not
    # fully been mastered by the TM, and it may take it some time to learn a
    # stable representation.

    #import ipdb; ipdb.set_trace()

    prevActiveCells = self.activeCells
    cellsToLearn = set(self.activeCells)

    # If there is some current activity, pick cells that have lateral input and
    # activate them, reaching quota if possible.  This also allows for one
    # column to infer what object it is currently learning from another, more
    # confident column.
    if len(cellsToLearn) < self.sdrSize:
      numActiveSegmentsByCell = numpy.zeros(self.cellCount, dtype="int")
      for i, lateralInput in enumerate(lateralInputs):
        overlaps = self.distalPermanences[i].rightVecSumAtNZGteThresholdSparse(
          lateralInput, self.connectedPermanenceDistal)
        numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1

      cellsToAdd = numpy.argsort(numActiveSegmentsByCell)[::-1]
      for cell in cellsToAdd:
        if (len(cellsToLearn) > self.sdrSize or
            numActiveSegmentsByCell[cell] < 1):
          break
        elif cell in cellsToLearn:
          continue
        else:
          cellsToLearn.add(cell)

    # If there are not enough previously active cells, select random subset of
    # cells to learn on.  If there were only some active cells, we assume that
    # we are learning an object related to what is represented by those cells;
    # if there were none, we are learning a new object.
    # This case is the only way different object representations are created.
    if len(cellsToLearn) < self.sdrSize:
      numToAdd = self.sdrSize - len(cellsToLearn)
      newCells = _sampleRange(self._random,
                              0, self.numberOfCells(),
                              step=1, k=numToAdd)
      cellsToLearn |= set(newCells)

      cellsToLearn = numpy.asarray(list(cellsToLearn))
      cellsToLearn.sort()
      #print len(self.activeCells), len(numpy.intersect1d(cellsToLearn, self.activeCells))
      self.activeCells = numpy.union1d(self.activeCells, cellsToLearn)
      self.activeCells.sort()

      # Internal distal learning
      self._learn(self.internalDistalPermanences, self._random,
                  self.activeCells, self.activeCells, self.activeCells,
                  self.sampleSizeDistal, self.initialDistalPermanence,
                  self.synPermDistalInc, self.synPermDistalDec,
                  self.connectedPermanenceDistal)

    else:
      cellsToLearn = numpy.asarray(list(cellsToLearn))
      cellsToLearn.sort()

    if len(cellsToLearn) > self.sdrSize:
      numActiveSegmentsByCell = numpy.zeros(self.cellCount, dtype="int")
      for i, lateralInput in enumerate(lateralInputs):
        overlaps = self.distalPermanences[i].rightVecSumAtNZGteThresholdSparse(
          lateralInput, self.connectedPermanenceDistal)
        numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1
      numActiveSegmentsByCell = numActiveSegmentsByCell[cellsToLearn]
      indices = numpy.argsort(numActiveSegmentsByCell)[:self.sdrSize]
      cellsToLearn = cellsToLearn[indices]

    cellsToLearn.sort()
    #print len(self.activeCells), len(numpy.intersect1d(cellsToLearn, self.activeCells))
    self.activeCells = numpy.union1d(self.activeCells, cellsToLearn)
    self.activeCells.sort()

    # Finally, now that we have decided which cells we should be learning on, do
    # the actual learning.
    if (len(feedforwardInput) > 0):# and
          #len(feedforwardGrowthCandidates) > self.minThresholdProximal):
      # Proximal learning
      self._learn(self.proximalPermanences, self._random,
                  cellsToLearn, feedforwardInput,
                  feedforwardGrowthCandidates, self.sampleSizeProximal,
                  self.initialProximalPermanence, self.synPermProximalInc,
                  self.synPermProximalDec, self.connectedPermanenceProximal)

      # Internal distal learning
      # Don't do any if we haven't gotten predicted input, i.e. if we aren't
      # learning anything proximally.
      if False: #(len(prevActiveCells) > 0): #and
            #len(feedforwardGrowthCandidates) > self.minThresholdProximal):
        self._learn(self.internalDistalPermanences, self._random,
                    cellsToLearn, prevActiveCells, prevActiveCells,
                    self.sampleSizeDistal, self.initialDistalPermanence,
                    self.synPermDistalInc, self.synPermDistalDec,
                    self.connectedPermanenceDistal)

      # External distal learning
      # We should do this no matter what, since other columns might still be
      # learning useful things even if we're not.
      for i, lateralInput in enumerate(lateralInputs):
        self._learn(self.distalPermanences[i], self._random,
                    cellsToLearn, lateralInput, lateralInput,
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
    @param  feedforwardInput (sequence)
            Sorted indices of active feedforward input bits

    @param  lateralInputs (list of sequences)
            For each lateral layer, a list of sorted indices of active lateral
            input bits
    """

    prevActiveCells = self.activeCells

    # Calculate the feedforward supported cells
    overlaps = self.proximalPermanences.rightVecSumAtNZGteThresholdSparse(
      feedforwardInput, self.connectedPermanenceProximal)
    feedforwardSupportedCells = numpy.where(
      overlaps >= self.minThresholdProximal)[0]

    # Calculate the number of active segments on each cell
    numActiveSegmentsByCell = numpy.zeros(self.cellCount, dtype="int")
    overlaps = self.internalDistalPermanences.rightVecSumAtNZGteThresholdSparse(
      prevActiveCells, self.connectedPermanenceDistal)
    numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1
    for i, lateralInput in enumerate(lateralInputs):
      overlaps = self.distalPermanences[i].rightVecSumAtNZGteThresholdSparse(
        lateralInput, self.connectedPermanenceDistal)
      numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1

    chosenCells = []
    minNumActiveCells = int(self.sdrSize * 0.75)

    numActiveSegsForFFSuppCells = numActiveSegmentsByCell[
        feedforwardSupportedCells]

    # First, activate the FF-supported cells that have the highest number of
    # lateral active segments (as long as it's not 0)
    if len(feedforwardSupportedCells) == 0:
      pass
    else:
      # This loop will select the FF-supported AND laterally-active cells, in
      # order of descending lateral activation, until we exceed the
      # minNumActiveCells quorum - but will exclude cells with 0 lateral
      # active segments.
      ttop = numpy.max(numActiveSegsForFFSuppCells)
      while ttop > 0 and len(chosenCells) <= minNumActiveCells:
        chosenCells = numpy.union1d(chosenCells,
                    feedforwardSupportedCells[numActiveSegsForFFSuppCells >
                    self.distalSegmentInhibitionFactor * ttop])
        ttop -= 1

    # If we still haven't filled the minNumActiveCells quorum, add in the
    # FF-supported cells with 0 lateral support AND the inertia cells.
    if len(chosenCells) < minNumActiveCells:
      if self.useInertia:
        prevCells = numpy.setdiff1d(prevActiveCells, chosenCells)
        inertialCap = int(len(prevCells) * self.inertiaFactor)
        if inertialCap > 0:
          numActiveSegsForPrevCells = numActiveSegmentsByCell[prevCells]
          # We sort the previously-active cells by number of active lateral
          # segments (this really helps).  We then activate them in order of
          # descending lateral activation.
          sortIndices = numpy.argsort(numActiveSegsForPrevCells)[::-1]
          prevCells = prevCells[sortIndices]
          numActiveSegsForPrevCells = numActiveSegsForPrevCells[sortIndices]

          # We use inertiaFactor to limit the number of previously-active cells
          # which can become active, forcing decay even if we are below quota.
          prevCells = prevCells[:inertialCap]
          numActiveSegsForPrevCells = numActiveSegsForPrevCells[:inertialCap]

          # Activate groups of previously active cells by order of their lateral
          # support until we either meet quota or run out of cells.
          ttop = numpy.max(numActiveSegsForPrevCells)
          while ttop >= 0 and len(chosenCells) <= minNumActiveCells:
            chosenCells = numpy.union1d(chosenCells,
                        prevCells[numActiveSegsForPrevCells >
                        self.distalSegmentInhibitionFactor * ttop])
            ttop -= 1

      # Finally, add remaining cells with feedforward support
      remFFcells = numpy.setdiff1d(feedforwardSupportedCells, chosenCells)
      # Note that this is 100% of the remaining FF-supported cells, there is no
      # attempt to select only certain ones or limit how many come active.
      chosenCells = numpy.append(chosenCells, remFFcells)

    chosenCells.sort()
    self.activeCells = numpy.asarray(chosenCells, dtype="uint32")


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
    self.activeCells = numpy.empty(0, dtype="uint32")

  def getUseInertia(self):
    """
    Get whether we actually use inertia  (i.e. a fraction of the
    previously active cells remain active at the next time step unless
    inhibited by cells with both feedforward and lateral support).
    @return (Bool) Whether inertia is used.
    """
    return self.useInertia

  def setUseInertia(self, useInertia):
    """
    Sets whether we actually use inertia (i.e. a fraction of the
    previously active cells remain active at the next time step unless
    inhibited by cells with both feedforward and lateral support).
    @param useInertia (Bool) Whether inertia is used.
    """
    self.useInertia = useInertia

  @staticmethod
  def _learn(# mutated args
             permanences, rng,

             # activity
             activeCells, activeInput, growthCandidateInput,

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

    @param  growthCandidateInput (sorted sequence)
            Sorted list of active bits in the input that the activeCells may
            grow new synapses to

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
      existingSynapseCounts = permanences.nNonZerosPerRowOnCols(
        activeCells, activeInput)

      maxNewByCell = numpy.empty(len(activeCells), dtype="int32")
      numpy.subtract(sampleSize, existingSynapseCounts, out=maxNewByCell)

      permanences.setRandomZerosOnOuter(
        activeCells, growthCandidateInput, maxNewByCell, initialPermanence, rng)


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



def _countWhereGreaterEqualInRows(sparseMatrix, rows, threshold):
  """
  Like countWhereGreaterOrEqual, but for an arbitrary selection of rows, and
  without any column filtering.
  """
  return sum(sparseMatrix.countWhereGreaterOrEqual(row, row+1,
                                                   0, sparseMatrix.nCols(),
                                                   threshold)
             for row in rows)
