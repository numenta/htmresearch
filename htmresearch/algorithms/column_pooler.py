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
  """,
  This class constitutes a temporary implementation for a cross-column pooler.
  The implementation goal of this class is to prove basic properties before
  creating a cleaner implementation.
  """

  def __init__(self,
               inputWidth,
               lateralInputWidths=(),
               cellCount=4096,
               sdrSize=40,
               onlineLearning = False,
               maxSdrSize = None,
               minSdrSize = None,

               # Proximal
               synPermProximalInc=0.1,
               synPermProximalDec=0.001,
               initialProximalPermanence=0.6,
               sampleSizeProximal=20,
               minThresholdProximal=10,
               connectedPermanenceProximal=0.50,
               predictedInhibitionThreshold=20,

               # Distal
               synPermDistalInc=0.1,
               synPermDistalDec=0.001,
               initialDistalPermanence=0.6,
               sampleSizeDistal=20,
               activationThresholdDistal=13,
               connectedPermanenceDistal=0.50,
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

    @param  onlineLearning (Bool)
            Whether or not the column pooler should learn in online mode.

    @param  maxSdrSize (int)
            The maximum SDR size for learning.  If the column pooler has more
            than this many cells active, it will refuse to learn.  This serves
            to stop the pooler from learning when it is uncertain of what object
            it is sensing.

    @param  minSdrSize (int)
            The minimum SDR size for learning.  If the column pooler has fewer
            than this many active cells, it will create a new representation
            and learn that instead.  This serves to create separate
            representations for different objects and sequences.

            If online learning is enabled, this parameter should be at least
            inertiaFactor*sdrSize.  Otherwise, two different objects may be
            incorrectly inferred to be the same, as SDRs may still be active
            enough to learn even after inertial decay.

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

    @param  predictedInhibitionThreshold (int)
            How much predicted input must be present for inhibitory behavior
            to be triggered.  Only has effects if onlineLearning is true.

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

    @param  inertiaFactor (float)
            The proportion of previously active cells that remain
            active in the next timestep due to inertia (in the absence of
            inhibition).  If onlineLearning is enabled, should be at most
            1 - learningTolerance, or representations may incorrectly become
            mixed.

    @param  seed (int)
            Random number generator seed
    """

    assert maxSdrSize is None or maxSdrSize >= sdrSize
    assert minSdrSize is None or minSdrSize <= sdrSize

    self.inputWidth = inputWidth
    self.cellCount = cellCount
    self.sdrSize = sdrSize
    self.onlineLearning = onlineLearning
    if maxSdrSize is None:
      self.maxSdrSize = sdrSize
    else:
      self.maxSdrSize = maxSdrSize
    if minSdrSize is None:
      self.minSdrSize = sdrSize
    else:
      self.minSdrSize = minSdrSize
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.initialProximalPermanence = initialProximalPermanence
    self.connectedPermanenceProximal = connectedPermanenceProximal
    self.sampleSizeProximal = sampleSizeProximal
    self.minThresholdProximal = minThresholdProximal
    self.predictedInhibitionThreshold = predictedInhibitionThreshold
    self.synPermDistalInc = synPermDistalInc
    self.synPermDistalDec = synPermDistalDec
    self.initialDistalPermanence = initialDistalPermanence
    self.connectedPermanenceDistal = connectedPermanenceDistal
    self.sampleSizeDistal = sampleSizeDistal
    self.activationThresholdDistal = activationThresholdDistal
    self.inertiaFactor = inertiaFactor

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
              feedforwardGrowthCandidates=None, learn=True,
              predictedInput = None,):
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

    @param predictedInput (sequence)
           Sorted indices of predicted cells in the TM layer.
    """

    if feedforwardGrowthCandidates is None:
      feedforwardGrowthCandidates = feedforwardInput

    # inference step
    if not learn: # inference
      self._computeInferenceMode(feedforwardInput, lateralInputs)

    # learning step
    elif not self.onlineLearning:
      self._computeLearningMode(feedforwardInput, lateralInputs,
                                feedforwardGrowthCandidates)
    # online learning step
    else:
      if (predictedInput is not None and
          len(predictedInput) > self.predictedInhibitionThreshold):
        predictedActiveInput = numpy.intersect1d(feedforwardInput,
                                                 predictedInput)
        predictedGrowthCandidates = numpy.intersect1d(
            feedforwardGrowthCandidates, predictedInput)
        self._computeInferenceMode(predictedActiveInput, lateralInputs)
        self._computeLearningMode(predictedActiveInput, lateralInputs,
                                  feedforwardGrowthCandidates)
      elif not self.minSdrSize <= len(self.activeCells) <= self.maxSdrSize:
        # If the pooler doesn't have a single representation, try to infer one,
        # before actually attempting to learn.
        self._computeInferenceMode(feedforwardInput, lateralInputs)
        self._computeLearningMode(feedforwardInput, lateralInputs,
                                  feedforwardGrowthCandidates)
      else:
        # If there isn't predicted input and we have a single SDR,
        # we are extending that representation and should just learn.
        self._computeLearningMode(feedforwardInput, lateralInputs,
                                  feedforwardGrowthCandidates)


  def _computeLearningMode(self, feedforwardInput, lateralInputs,
                                 feedforwardGrowthCandidates):
    """
    Learning mode: we are learning a new object in an online fashion. If there
    is no prior activity, we randomly activate 'sdrSize' cells and create
    connections to incoming input. If there was prior activity, we maintain it.
    If we have a union, we simply do not learn at all.

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
    prevActiveCells = self.activeCells

    # If there are not enough previously active cells, then we are no longer on
    # a familiar object.  Either our representation decayed due to the passage
    # of time (i.e. we moved somewhere else) or we were mistaken.  Either way,
    # create a new SDR and learn on it.
    # This case is the only way different object representations are created.
    # enforce the active cells in the output layer
    if len(self.activeCells) < self.minSdrSize:
      self.activeCells = _sampleRange(self._random,
                                      0, self.numberOfCells(),
                                      step=1, k=self.sdrSize)
      self.activeCells.sort()

    # If we have a union of cells active, don't learn.  This primarily affects
    # online learning.
    if len(self.activeCells) > self.maxSdrSize:
      return

    # Finally, now that we have decided which cells we should be learning on, do
    # the actual learning.
    if len(feedforwardInput) > 0:
      self._learn(self.proximalPermanences, self._random,
                  self.activeCells, feedforwardInput,
                  feedforwardGrowthCandidates, self.sampleSizeProximal,
                  self.initialProximalPermanence, self.synPermProximalInc,
                  self.synPermProximalDec, self.connectedPermanenceProximal)

      # External distal learning cross column, segments
      for i, lateralInput in enumerate(lateralInputs):
        self._learn(self.distalPermanences[i], self._random,
                    self.activeCells, lateralInput, lateralInput,
                    self.sampleSizeDistal, self.initialDistalPermanence,
                    self.synPermDistalInc, self.synPermDistalDec,
                    self.connectedPermanenceDistal)

      # Internal distal learning within the same column
      self._learn(self.internalDistalPermanences, self._random,
                  self.activeCells, prevActiveCells, prevActiveCells,
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
    print self.activeCells

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

    # First, activate the FF-supported cells that have the highest number of
    if len(feedforwardSupportedCells) == 0:
    # lateral active segments (as long as it's not 0)
      pass
    else:
      numActiveSegsForFFSuppCells = numActiveSegmentsByCell[
        feedforwardSupportedCells]

      # This loop will select the FF-supported AND laterally-active cells, in
      # order of descending lateral activation, until we exceed the sdrSize
      # quorum - but will exclude cells with 0 lateral active segments.
      ttop = numpy.max(numActiveSegsForFFSuppCells)
      while ttop > 0 and len(chosenCells) < self.sdrSize:
        chosenCells = numpy.union1d(chosenCells,
                    feedforwardSupportedCells[numActiveSegsForFFSuppCells > ttop])
        ttop -= 1

    # If we haven't filled the sdrSize quorum, add in inertial cells.
    if len(chosenCells) < self.sdrSize:
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
          while ttop >= 0 and len(chosenCells) < self.sdrSize:
            chosenCells = numpy.union1d(chosenCells,
                        prevCells[numActiveSegsForPrevCells > ttop])
            ttop -= 1

    # If we haven't filled the sdrSize quorum, add cells that have feedforward
    # support and no lateral support.
    discrepancy = self.sdrSize - len(chosenCells)
    if discrepancy > 0:
      remFFcells = numpy.setdiff1d(feedforwardSupportedCells, chosenCells)
      if len(remFFcells) > discrepancy:
        # Inhibit cells proportionally to the number of cells that have already
        # been chosen. If ~0 have been chosen activate ~all of the feedforward
        # supported cells. If ~sdrSize have been chosen, activate very few of
        # the feedforward supported cells.
        n = min(max(discrepancy,
                    len(remFFcells) * discrepancy / self.sdrSize),
                len(remFFcells))
        selected = numpy.empty(n, dtype="uint32")
        self._random.sample(numpy.asarray(remFFcells, dtype="uint32"),
                            selected)
        chosenCells = numpy.append(chosenCells, selected)
      else:
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
