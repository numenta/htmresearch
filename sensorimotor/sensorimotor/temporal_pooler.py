# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy

from nupic.bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix)
from nupic.research.spatial_pooler import SpatialPooler

realDType = numpy.float32
uintType = "uint32"


class TemporalPooler(SpatialPooler):
  """
  This is the default implementation of the new temporal pooler. It attempts
  to form stable and unique representations of input sequences of active
  cells from Temporal Memory, but only if those cells were correctly
  predicted by Temporal Memory. If the active cell sequence was not predicted,
  a competition, similar to that of the spatial pooler's, is used to select
  active columns.

  If a column is sufficiently activated by inputs that were predicted in the
  Temporal Memory, it enters a "pooling mode." A pooling
  column can, for a limited period, maintain active column status even without
  receiving any bottom-up input activity. This is implemented using a kind of
  timer for each column. When the timer "runs out," it loses its pooling status.
  Whenever significant predicted input returns to a column's synapses,
  its timer is reset. Note, that in this temporal pooler there is only one cell
  per column and there is no temporal memory.
  """

  # TODO: RM: Why does this subclass of SpatialPooler inherit so little from
  # its parent? Especially in the constructor / initialization? At least,
  # there should be some documentation explaining the reason(s) for the
  # divergences. It's difficult to understand what is specific to this
  # class. Also there are a lot of documentation
  # passages that  are repeated from  the spatial pooler, which would be
  # alleviated using inheritance.
  def __init__(self,
               inputDimensions=[32,32],
               columnDimensions=[64,64],
               potentialRadius=16,
               potentialPct=0.9,
               globalInhibition=True,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=10.0,
               stimulusThreshold=2,
               synPermInactiveDec=0.01,
               synPermActiveInc=0.03,
               synPredictedInc=0.5,
               synPermConnected=0.3,
               minPctOverlapDutyCycle=0.001,
               minPctActiveDutyCycle=0.001,
               dutyCyclePeriod=1000,
               maxBoost=1.0,
               useBurstingRule = False,
               usePoolingRule = True,
               maxPoolingTime = 1000,
               poolingThreshUnpredicted = 0.0,
               initConnectedPct = 0.2,
               seed=-1,
               spVerbosity=0,
               wrapAround=True):
    """
    Please see spatial_pooler.py in NuPIC for descriptions of common
    constructor parameters.

    Class-specific parameters:
    -------------------------------------

    @param synPredictedInc:
      Define a metabotropically active synapse as an active synapse
      whose input originates from a correctly predicted cell.
      synPredictedInc is then the amount by which a metabotropically active
      synapse is incremented in each round

    @param useBurstingRule:
      A Boolean indicating whether bursting columns in the TM will have a
      (strong) effect on overlap calculation

    @param usePoolingRule:
       A Boolean indicating whether inputs representing correctly predicted
       cells in the TM will contribute to column overlap calculation

    @param maxPoolingTime:
      The maximum number of timesteps that a pooling column will pool for
      in the absence of any predicted input.

    @param poolingThreshUnpredicted:
      A threshold, ranging from 0 to 1, on the fraction of
      bottom-up input that is unpredicted. If this is exceeded, the temporal
      pooler will stop pooling.

    @param initConnectedPct:
      A value between 0 or 1 governing the chance, for each permanence,
      that the initial permanence value will be a value that is considered
      connected.
    """
    self.initialize(inputDimensions,
                    columnDimensions,
                    potentialRadius,
                    potentialPct,
                    globalInhibition,
                    localAreaDensity,
                    numActiveColumnsPerInhArea,
                    stimulusThreshold,
                    synPermInactiveDec,
                    synPermActiveInc,
                    synPredictedInc,
                    synPermConnected,
                    minPctOverlapDutyCycle,
                    minPctActiveDutyCycle,
                    dutyCyclePeriod,
                    maxBoost,
                    useBurstingRule,
                    usePoolingRule,
                    maxPoolingTime,
                    poolingThreshUnpredicted,
                    seed,      
                    initConnectedPct,              
                    spVerbosity,
                    wrapAround)


  def initialize(self,
               inputDimensions=[32,32],
               columnDimensions=[64,64],
               potentialRadius=16,
               potentialPct=0.5,
               globalInhibition=False,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=10.0,
               stimulusThreshold=0,
               synPermInactiveDec=0.01,
               synPermActiveInc=0.1,
               synPredictedInc=0.1,
               synPermConnected=0.10,
               minPctOverlapDutyCycle=0.001,
               minPctActiveDutyCycle=0.001,
               dutyCyclePeriod=1000,
               maxBoost=10.0,
               useBurstingRule=True,
               usePoolingRule=True,
               maxPoolingTime=1000,
               poolingThreshUnpredicted=0.0,
               seed=-1,
               initConnectedPct=0.1,
               spVerbosity=0,
               wrapAround=True):

    # Verify input is valid
    inputDimensions = numpy.array(inputDimensions)
    columnDimensions = numpy.array(columnDimensions)
    numColumns = columnDimensions.prod()
    numInputs = inputDimensions.prod()

    assert(numColumns > 0)
    assert(numInputs > 0)
    assert (numActiveColumnsPerInhArea > 0 or 
           (localAreaDensity > 0 and localAreaDensity <= 0.5))

    # save arguments
    self._numInputs = int(numInputs)
    self._numColumns = int(numColumns)
    self._columnDimensions = columnDimensions
    self._inputDimensions = inputDimensions
    self._potentialRadius = int(min(potentialRadius, numInputs))
    self._potentialPct = potentialPct
    self._globalInhibition = globalInhibition
    self._numActiveColumnsPerInhArea = int(numActiveColumnsPerInhArea)
    self._localAreaDensity = localAreaDensity
    self._stimulusThreshold = stimulusThreshold
    self._synPermInactiveDec = synPermInactiveDec
    self._synPermActiveInc = synPermActiveInc
    self._synPermBelowStimulusInc = synPermConnected / 10.0
    self._synPermConnected = synPermConnected
    self._minPctOverlapDutyCycles = minPctOverlapDutyCycle
    self._minPctActiveDutyCycles = minPctActiveDutyCycle
    self._dutyCyclePeriod = dutyCyclePeriod
    self._maxBoost = maxBoost
    self._spVerbosity = spVerbosity
    self._wrapAround = wrapAround

    # ------ Specific to temporal_pooler.py --------
    self._synPredictedInc = synPredictedInc
    self.useBurstingRule = useBurstingRule
    self.usePoolingRule = usePoolingRule
    self._maxPoolingTime = maxPoolingTime
    self._poolingThreshUnpredicted = poolingThreshUnpredicted
    # This also appears in the SP but as a local and not a class variable
    self._initConnectedPct = initConnectedPct

    # A column will enter pooling state if it receives enough predicted inputs
    # pooling columns have priority during competition
    self._poolingActivation = numpy.zeros(self._numColumns, dtype="int32")
    self._poolingColumns = []

    # ------ /Specific to temporal_pooler.py --------

    # Extra parameter settings
    self._synPermMin = 0.0
    self._synPermMax = 1.0
    self._synPermTrimThreshold = synPermActiveInc / 2.0
    assert(self._synPermTrimThreshold < self._synPermConnected)
    self._updatePeriod = 20

    # Internal state
    self._version = 1.0
    self._iterationNum = 0
    self._iterationLearnNum = 0

    # initialize the random number generators
    self._seed(seed)

    # Initialize a tiny random tie breaker. This is used to determine winning
    # columns where the overlaps are identical.
    self._tieBreaker = 0.01 * numpy.array([self._random.getReal64() for i in
                                          xrange(self._numColumns)])

    # initialize connection matrix
    self.initializeConnections()
    self._overlapDutyCycles = numpy.zeros(numColumns, dtype=realDType)
    self._activeDutyCycles = numpy.zeros(numColumns, dtype=realDType)
    self._minOverlapDutyCycles = numpy.zeros(numColumns, 
                                             dtype=realDType)
    self._minActiveDutyCycles = numpy.zeros(numColumns,
                                            dtype=realDType)
    self._boostFactors = numpy.ones(numColumns, dtype=realDType)

    # The inhibition radius determines the size of a column's local 
    # neighborhood. A cortical minicolumn must overcome the overlap
    # score of columns in its neighborhood in order to become active. This
    # radius is updated every _updatePeriod iterations. It grows and shrinks
    # with the average number of connected synapses per column.
    self._inhibitionRadius = 0
    self._updateInhibitionRadius()
    
    if self._spVerbosity > 0:
      self.printParameters()

  def initializeConnections(self):
    """
    Initialize connection matrix, including:

    _permanences        permanence of synaptic connections (sparse matrix)
    _potentialPools     potential pool of connections for each cell
                        (sparse binary matrix)
    _connectedSynapses  connected synapses (binary sparse matrix)
    _connectedCounts    number of connections per cell (numpy array)
    """
    numColumns = self._numColumns
    numInputs = self._numInputs

    # TODO: Review comment with someone, this seems specific to this class...
    # The SP should be setup so that stimulusThreshold is reasonably high,
    # similar to a TP's activation threshold. The pct of connected cells may
    # need to be low to avoid false positives given the monosynaptic rule 
    initConnectedPct = self._initConnectedPct

    # Store the set of all inputs that are within each column's potential pool.
    # 'potentialPools' is a matrix, whose rows represent cortical columns, and
    # whose columns represent the input bits. if potentialPools[i][j] == 1,
    # then input bit 'j' is in column 'i's potential pool. A column can only be
    # connected to inputs in its potential pool. The indices refer to a
    # flattened version of both the inputs and columns. Namely, irrespective
    # of the topology of the inputs and columns, they are treated as being a
    # one dimensional array. Since a column is typically connected to only a
    # subset of the inputs, many of the entries in the matrix are 0. Therefore
    # the the potentialPool matrix is stored using the SparseBinaryMatrix
    # class, to reduce memory footprint and compuation time of algorithms that
    # require iterating over the data strcuture.
    self._potentialPools = SparseBinaryMatrix(numInputs)
    self._potentialPools.resize(numColumns, numInputs)

    # Initialize the permanences for each column. Similar to the 
    # 'self._potentialPools', the permanences are stored in a matrix whose rows
    # represent the cortial columns, and whose columns represent the input 
    # bits. if self._permanences[i][j] = 0.2, then the synapse connecting 
    # cortical column 'i' to input bit 'j'  has a permanence of 0.2. Here we 
    # also use the SparseMatrix class to reduce the memory footprint and 
    # computation time of algorithms that require iterating over the data 
    # structure. This permanence matrix is only allowed to have non-zero 
    # elements where the potential pool is non-zero.
    self._permanences = SparseMatrix(numColumns, numInputs)

    # 'self._connectedSynapses' is a similar matrix to 'self._permanences' 
    # (rows represent cortical columns, columns represent input bits) whose
    # entries represent whether the cortial column is connected to the input 
    # bit, i.e. its permanence value is greater than 'synPermConnected'. While 
    # this information is readily available from the 'self._permanence' matrix, 
    # it is stored separately for efficiency purposes.
    self._connectedSynapses = SparseBinaryMatrix(numInputs)
    self._connectedSynapses.resize(numColumns, numInputs)

    # Stores the number of connected synapses for each column. This is simply
    # a sum of each row of 'self._connectedSynapses'. again, while this 
    # information is readily available from 'self._connectedSynapses', it is
    # stored separately for efficiency purposes.
    self._connectedCounts = numpy.zeros(numColumns, dtype=realDType)

    # Initialize the set of permanence values for each column. Ensure that
    # each column is connected to enough input bits to allow it to be
    # activated.
    for i in xrange(numColumns):
      potential = self._mapPotential(i, wrapAround=self._wrapAround)
      self._potentialPools.replaceSparseRow(i, potential.nonzero()[0])
      perm = self._initPermanence(potential, initConnectedPct)
      self._updatePermanencesForColumn(perm, i, raisePerm=True)

    # TODO: RM: Can we remove this?
    # TODO: The permanence initialization code below runs faster but is not
    # deterministic (doesn't use our RNG or the seed).  The speed is
    # particularly important for temporal pooling because the number if inputs =
    # number of cells in the prevous level. We should consider cleaning up the
    # code and moving it to the base spatial pooler class itself.

    # for i in xrange(numColumns):
    #   # NEW indices for inputs within _potentialRadius of the current column
    #   indices = numpy.array(range(2*self._potentialRadius+1))
    #   indices += i
    #   indices -= self._potentialRadius
    #   indices %= self._numInputs # periodic boundary conditions
    #   indices = numpy.array(list(set(indices)))
    #
    #   # NEW Select a subset of the receptive field to serve as the
    #   # potential pool
    #   sample = numpy.empty((self._inputDimensions * self._potentialPct)\
    #                         .astype("int32"),dtype=uintType)
    #   self._random.getUInt32Sample(indices.astype(uintType),sample)
    #   potential = numpy.zeros(self._numInputs)
    #   potential[sample] = 1
    #
    #   # update potentialPool
    #   self._potentialPools.replaceSparseRow(i, potential.nonzero()[0])
    #
    #   # NEW generate indicator for connected/unconnected
    #   connected = numpy.random.rand(self._numInputs) < initConnectedPct
    #
    #   # set permanence value for connected synapses to be slightly above _synPermConnected
    #   permConnected =  (self._synPermConnected + numpy.random.rand(self._numInputs) *
    #     self._synPermActiveInc / 4.0)
    #   # set permanence value for unconnected synapses below _synPermConnected
    #   permNotConnected = self._synPermConnected * numpy.random.rand(self._numInputs)
    #
    #   # update permamnce value
    #   perm = permNotConnected
    #   perm[connected] = permConnected[connected]
    #   perm[potential < 1] = 0 # permanence value for cells not in the potential
    #   # pool
    #
    #   self._updatePermanencesForColumn(perm, i, raisePerm=True)


  def reset(self):
    """
    Reset the state of temporal pooler
    """
    self._poolingActivation = numpy.zeros(self._numColumns, dtype="int32")
    self._poolingColumns = []
    self._overlapDutyCycles = numpy.zeros(self._numColumns, dtype=realDType)
    self._activeDutyCycles = numpy.zeros(self._numColumns, dtype=realDType)
    self._minOverlapDutyCycles = numpy.zeros(self._numColumns, 
                                             dtype=realDType)
    self._minActiveDutyCycles = numpy.zeros(self._numColumns,
                                            dtype=realDType)
    self._boostFactors = numpy.ones(self._numColumns, dtype=realDType)


  def compute(self, inputVector, learning, activeArray, burstingColumns,
              correctlyPredicted):
    """
    This is the primary public method of the class. This function takes an input
    vector and outputs the indices of the active columns.

    @param inputVector Here, the input is the active cells from the TM
    @param learning Boolean specifying whether learning is on
    @param activeArray An array representing the active columns produced by this
                      method
    @param burstingColumns
      A numpy array with numColumns elements where each 1 represent a
      currently bursting column in the TM.
    @param correctlyPredicted
      A numpy array with numInputs elements. A 1 indicates that this cell
      switching from predicted state in the previous time step to active state
      in the current timestep
    """
    assert (numpy.size(inputVector) == self._numInputs)
    assert (numpy.size(correctlyPredicted) == self._numInputs)

    self._updateBookeepingVars(learning)
    inputVector = numpy.array(inputVector, dtype=realDType)
    correctlyPredicted = numpy.array(correctlyPredicted, dtype=realDType)
    inputVector.reshape(-1)

    if self._spVerbosity > 3:
      print " Input bits: ", inputVector.nonzero()[0]
      print " Correctly predicted cells: ", correctlyPredicted.nonzero()[0]

    # Phase 1: Calculate overlap scores
    # The overlap score has 4 components:
    # (1) Overlap between correctly predicted input cells and pooling columns
    # (2) Overlap between active cells input and all columns (standard)
    # (3) Overlap between correctly predicted input cells and all columns
    # (4) Overlap sent from bursting columns to all columns

    # 1) Calculate pooling overlap
    if self.usePoolingRule:
      overlapsPooling = self._calculatePoolingOverlap(correctlyPredicted,
                                                      learning)

      if self._spVerbosity > 4:
        print "usePoolingRule: Overlaps after step 1:"
        print "   ", overlapsPooling
    else:
      overlapsPooling = 0
  
    # 2) Calculate overlap between standard input and connected synapses
    overlapsAllInput = self._calculateOverlap(inputVector)

    # 3) overlap with predicted inputs
    # NEW: Isn't this redundant with 1 and 2)? This looks at connected synapses
    # only.
    # RM If 1) is called with learning=False connected synapses are used and
    # it is somewhat redundant although there is a boosting factor in 1) which
    # makes 1's effect stronger. If 1) is called with learning=True it's less
    # redundant
    overlapsPredicted = self._calculateOverlap(correctlyPredicted)

    if self._spVerbosity > 4:
      print "Overlaps with all inputs:"
      print " Number of On Bits: ", inputVector.sum()
      print "   ", overlapsAllInput
      print "Overlaps with predicted inputs:"
      print "   ", overlapsPredicted

    # 4) consider bursting columns
    if self.useBurstingRule:
      overlapsBursting = self._calculateBurstingColumns(burstingColumns)

      if self._spVerbosity > 4:
        print "Overlaps with bursting inputs:"
        print "   ", overlapsBursting
    else:
      overlapsBursting = 0

    overlaps = (overlapsPooling + overlapsPredicted + overlapsAllInput +
                overlapsBursting)

    # Apply boosting when learning is on
    if learning:
      boostedOverlaps = self._boostFactors * overlaps
      if self._spVerbosity > 4:
        print "Overlaps after boosting:"
        print "   ", boostedOverlaps      

    else:
      boostedOverlaps = overlaps

    # Apply inhibition to determine the winning columns
    activeColumns = self._inhibitColumns(boostedOverlaps)    
      
    if learning:
      self._adaptSynapses(inputVector, correctlyPredicted, activeColumns)
      self._updateDutyCycles(overlaps, activeColumns)
      self._bumpUpWeakColumns() 
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    # TODO: Why do anything with activeArray if returning activeColumns?
    activeArray.fill(0)
    if activeColumns.size > 0:
      activeArray[activeColumns] = 1

    # Update pooling state of columns
    poolingColumnsUpdate = activeColumns[numpy.where(
                                      overlapsPredicted[activeColumns] > 0)[0]]
    numUnpredictedInputs = float(len(burstingColumns.nonzero()[0]))
    numPredictedInputs = float(len(correctlyPredicted))

    # TODO: I don't like this. Here we are combining two different kinds of
    # things, some number of columns and some number of cells -- so the behavior
    # of this changes as the ratio of cells to columns changes. This makes
    # picking an appropriate _poolingThreshUnpredicted value harder.
    unpredictedFraction = numUnpredictedInputs / (numUnpredictedInputs +
                                                  numPredictedInputs)
    self._updatePoolingState(poolingColumnsUpdate,
                             unpredictedFraction)

    if self._spVerbosity > 2:
      activeColumns.sort()
      print "The following columns are finally active:"
      print "   ",activeColumns
      print "The following columns are in pooling state:"
      print "   ",self._poolingActivation.nonzero()[0]
      # print "Inputs to pooling columns"
      # print "   ",overlapsPredicted[self._poolingColumns]

    return activeColumns


  def _calculatePoolingOverlap(self, correctlyPredictedInput, learning):
    """
    Determines each column's overlap with inputs coming from
    predicted cells. If learning, overlap is calculated between predicted
    input cells and potential synapses. Otherwise, overlap is calculated
    between predicted input cells and connected synapses.
    The overlap of a column that was previously active and
    has even one active predicted synapse is set to (_numInputs + 1) so they are
    guaranteed to win during inhibition.
    
    TODO: check with Jeff, what happens in biology if a column was not
    previously
    active but receives metabotropic input?  Does it turn on, or does the met.
    input just extend the activity of already active columns? Does it matter
    whether the synapses are connected or not? Currently we are assuming no
    because most connected synapses become disconnected in the previous time
    step. If column i at time t is bursting and performs learning, the synapses
    from t+1 aren't active at time t, so their permanence will decrease.

    Parameters:
    ----------------------------
    correctlyPredictedInput: a numpy array with numInputs elements. A 1
                             indicates that
                    this cell switching from predicted state in the previous
                    time step to active state in the current timestep

    @returns        an array of overlap values due predicted cells
    """
    overlaps = numpy.zeros(self._numColumns).astype(realDType)

    # If no columns in pooling state or no predicted inputs, return zero
    if sum(self._poolingActivation) == 0 or len(correctlyPredictedInput.nonzero()[0]) == 0:
      return overlaps

    if learning:
      # During learning, overlap is calculated based on potential synapses. 
      self._potentialPools.rightVecSumAtNZ_fast(correctlyPredictedInput, overlaps)
    else:
      # At inference stage, overlap is calculated based on connected synapses.
      self._connectedSynapses.rightVecSumAtNZ_fast(correctlyPredictedInput, overlaps)
    # TODO: ^^ Why was this done? to accelerate learning?

    # Only consider columns that are in pooling state
    mask = numpy.zeros(self._numColumns).astype(realDType)

    # TODO: Can't you just put boostFactorPooling here?
    mask[self._poolingColumns] = 1
    overlaps *= mask

    # Columns that are in the pooling state and receive predicted input
    # will have their overlap boosted by a large factor so that they are likely
    # to win the inter-column inhibition
    boostFactorPooling = self._maxBoost * self._numInputs
    overlaps *= boostFactorPooling

    if self._spVerbosity > 3:
      print "\n============== In _calculatePoolingActivity ======"
      print "Received predicted cell inputs from following indices:"
      print "   ", correctlyPredictedInput.nonzero()[0]
      print "The following column indices are in pooling state:"
      print "   ", self._poolingColumns
      print "Overlap score of pooling columns:"
      print "   ", overlaps[self._poolingColumns]
      print "============== Leaving _calculatePoolingActivity  ======\n"

    return overlaps


  def _calculateBurstingColumns(self, burstingColumns):
    """
    Returns the contribution to overlap due to bursting columns. If any
    column is bursting, its overlap score is set to stimulusThreshold. This
    means it will be guaranteed to win as long as no other column is
    metabotropic or has input > stimulusThreshold. 

    Parameters:
    ----------------------------
    burstingColumns: a numpy array with numColumns elements. A 1 indicates that
                     column is currently bursting.
    """
    overlaps = burstingColumns * self._stimulusThreshold
    return overlaps


  def _adaptSynapses(self, input, correctlyPredictedInput, activeColumns):
    """
    Updates synapses' permanence based on the bottom-up input to the TP
    (from the TM) and the TP's active columns.
    For each active column, its synapses' permanences are updated as follows:

    1. if pre-synaptic input is ON due to a correctly predicted cell,
       increase permanence by _synPredictedInc
    2. else if input is ON due to an active cell, increase permanence by
       _synPermActiveInc
    3. else input is OFF, decrease permanence by _synPermInactiveDec

    Parameters:
    ----------------------------
    input:    a numpy array, input to the temporal pooler, whose ON bits
                represent the active cells from temporal memory
    correctlyPredictedInput: a numpy array of those cells in temporal memory that
                      just switched from predictive to active in
                      the current timestep
    activeColumns:  an array containing the indices of the columns that
                    won the inhibition step
    """
    inputIndices = numpy.where(input > 0)[0]
    predictedIndices = numpy.where(correctlyPredictedInput > 0)[0]
    permChanges = numpy.zeros(self._numInputs)

    # Decrement inactive cells -> active columns
    permChanges.fill(-1 * self._synPermInactiveDec)

    # Increment active cells -> active columns
    permChanges[inputIndices] = self._synPermActiveInc

    # Increment correctly predicted cells -> active columns
    permChanges[predictedIndices] = self._synPredictedInc

    if self._spVerbosity > 4:
      print "\n============== _adaptSynapses ======"
      print "Active input indices:",inputIndices
      print "predicted input indices:",predictedIndices
      # print "permChanges:"
      # print sm_utils.formatRow(permChanges, "%9.4f", 20)
      print "\n============== _adaptSynapses ======\n"

    for i in activeColumns:
      # Get the permanences of the synapses of column i
      perm = self._permanences.getRow(i)

      # Only consider connections in column's potential pool (receptive field)
      mask = numpy.where(self._potentialPools.getRow(i) > 0)[0]
      perm[mask] += permChanges[mask]
      self._updatePermanencesForColumn(perm, i, raisePerm=False)


  def _updatePoolingState(self, poolingColumnsUpdate,
                          fractionUnpredicted):
    """
    This function updates the pooling state of columns. A column will stop
    pooling if:
    (1) It hasn't received any predicted input in the last self._maxPoolingTime
    steps
    or
    (2) the fraction of unpredicted input is above poolingThreshUnpredicted

    @param poolingColumnsUpdate
    @param fractionUnpredicted
    """
    if fractionUnpredicted > self._poolingThreshUnpredicted:
      # Reset pooling activation if the fraction of unpredicted input
      # is above the threshold
      self._poolingActivation = numpy.zeros(self._numColumns, dtype="int32")
      if self._spVerbosity > 3:
        print " reset pooling state for all columns"
    else:
      # decrement activation of all pooling columns
      self._poolingActivation[self._poolingColumns] -= 1

      # reset activation of columns that are receiving predicted input
      self._poolingActivation[poolingColumnsUpdate] = self._maxPoolingTime

    self._poolingColumns = self._poolingActivation.nonzero()[0]


  def printParameters(self):
    """
    Useful for debugging.
    """
    print "------------PY  TemporalPooler Parameters ------------------"
    print "numInputs                  = ", self.getNumInputs()
    print "numColumns                 = ", self.getNumColumns()
    print "columnDimensions           = ", self._columnDimensions
    print "numActiveColumnsPerInhArea = ", self.getNumActiveColumnsPerInhArea()
    print "potentialPct               = ", self.getPotentialPct()
    print "globalInhibition           = ", self.getGlobalInhibition()
    print "localAreaDensity           = ", self.getLocalAreaDensity()
    print "stimulusThreshold          = ", self.getStimulusThreshold()
    print "synPermActiveInc           = ", self.getSynPermActiveInc()
    print "synPermInactiveDec         = ", self.getSynPermInactiveDec()
    print "synPermConnected           = ", self.getSynPermConnected()
    print "minPctOverlapDutyCycle     = ", self.getMinPctOverlapDutyCycles()
    print "minPctActiveDutyCycle      = ", self.getMinPctActiveDutyCycles()
    print "dutyCyclePeriod            = ", self.getDutyCyclePeriod()
    print "maxBoost                   = ", self.getMaxBoost()
    print "spVerbosity                = ", self.getSpVerbosity()
    print "version                    = ", self._version


  # TODO: Probably can remove this. There are no calls to it! It seems to
  # have been replaced by a similar method in sensorimotor_experiment_runner
  def getInputForTP(self, tm):
    """
    Gets inputs for TP from specified temporal memory.
    Three pieces of information are returned:
    1. Cells correctly predicted by the temporal memory
    2. All active cells
    3. Bursting cells (representing unpredicted input)
    """

    # Get correctly predicted cells in layer 4
    correctlyPredictedCells = numpy.zeros(self._inputDimensions).astype(
                              realDType)
    idx = (tm.predictedState["t-1"] + tm.activeState["t"]) == 2
    idx = idx.reshape(self._inputDimensions)
    correctlyPredictedCells[idx] = 1.0
    # print "Predicted->active cells=",correctlyPredictedCells.nonzero()[0]

    # all currently active cells in layer 4
    spInputVector = tm.learnState["t"].reshape(self._inputDimensions)
    # spInputVector = tm.activeState["t"].reshape(self._inputDimensions)

    # bursting cells in layer 4
    burstingColumns = tm.activeState["t"].sum(axis=1)
    burstingColumns[burstingColumns < tm.cellsPerColumn] = 0
    burstingColumns[burstingColumns == tm.cellsPerColumn] = 1
    # print "Bursting column indices=",burstingColumns.nonzero()[0]

    return correctlyPredictedCells, spInputVector, burstingColumns
