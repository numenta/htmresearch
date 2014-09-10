# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

import copy
import numpy

from nupic.bindings.math import (SM32 as SparseMatrix,
                                SM_01_32_32 as SparseBinaryMatrix,
                                GetNTAReal,
                                Random as NupicRandom)
from nupic.research.spatial_pooler import SpatialPooler

# realDType = GetNTAReal()
realDType = numpy.float32
uintType = "uint32"


def formatRow(x, formatString = "%d", rowSize = 100):
  """
  Utility routine for pretty printing large vectors
  """
  s = ''
  for c,v in enumerate(x):
    if c > 0 and c % 10 == 0:
      s += ' '
    if c > 0 and c % rowSize == 0:
      s += '\n'
    s += formatString % v
  s += ' '
  return s


class SPTP(SpatialPooler):
  """
  This class implements the new temporal pooler. It tries to form stable and
  unique representations for sequences that are correctly predicted. If the
  input is not predicted, it used similar competition rule as the spatial
  pooler to select active cells
  """

  def __init__(self,
               inputDimensions=[32,32],
               columnDimensions=[64,64],
               cellsPerColumn = 8,
               potentialRadius=16,
               potentialPct=0.5,
               globalInhibition=False,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=10.0,
               stimulusThreshold=0,
               synPermInactiveDec=0.01,
               synPermActiveInc=0.03,
               synPermActiveInactiveDec = 0,
               synPredictedInc=0.5,
               synPermConnected=0.10,
               minPctOverlapDutyCycle=0.001,
               minPctActiveDutyCycle=0.001,
               dutyCyclePeriod=1000,
               maxBoost=10.0,
               useBurstingRule = True,
               usePoolingRule = True,
               poolingLife = 1000,
               poolingThreshUnpredicted = 0.0,
               initConnectedPct = 0.1,
               seed=-1,
               spVerbosity=0,
               wrapAround=True
               ):
    """
    Please see spatial_pooler.py in NuPIC for descriptions of common
    constructor parameters.

    New parameters defined in this class:
    -------------------------------------
    @param synPermActiveInactiveDec:
      For inactive columns, synapses connected to input bits that are on are
      decreased by synPermActiveInactiveDec.

    @param synPredictedInc:
      The amount by which a metabotropically active synapse is incremented in
      each round. These are active synapses originating from a previously
      predicted cell.

    @param useBurstingRule:
      A bool value indicating whether to use bursting rule

    @param usePoolingRule:
       A bool value indicating whether to use pooling rule

    @param poolingLife:
      A pooling cell will stop pooling if it hasn't received any predicted
      input in poolingLife steps

    @param poolingThreshUnpredicted:
      A number between 0 and 1. The temporal pooler will stop pooling if the
      fraction of unpredicted input exceeds this threshold.

    @param initConnectedPct:
      A number between 0 and 1, indicating fraction of the inputs that are
      initially connected.
    """
    self.initialize(inputDimensions,
                    columnDimensions,
                    cellsPerColumn,
                    potentialRadius,
                    potentialPct,
                    globalInhibition,
                    localAreaDensity,
                    numActiveColumnsPerInhArea,
                    stimulusThreshold,
                    synPermInactiveDec,
                    synPermActiveInc,
                    synPermActiveInactiveDec,
                    synPredictedInc,
                    synPermConnected,
                    minPctOverlapDutyCycle,
                    minPctActiveDutyCycle,
                    dutyCyclePeriod,
                    maxBoost,
                    useBurstingRule,
                    usePoolingRule,
                    poolingLife,
                    poolingThreshUnpredicted,
                    seed,      
                    initConnectedPct,              
                    spVerbosity,
                    wrapAround)


  def initialize(self,
               inputDimensions=[32,32],
               columnDimensions=[64,64],
               cellsPerColumn = 8,
               potentialRadius=16,
               potentialPct=0.5,
               globalInhibition=False,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=10.0,
               stimulusThreshold=0,
               synPermInactiveDec=0.01,
               synPermActiveInc=0.1,
               synPermActiveInactiveDec =0,
               synPredictedInc=0.1,
               synPermConnected=0.10,
               minPctOverlapDutyCycle=0.001,
               minPctActiveDutyCycle=0.001,
               dutyCyclePeriod=1000,
               maxBoost=10.0,
               useBurstingRule=True,
               usePoolingRule=True,
               poolingLife=1000,
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
    self._synPredictedInc = synPredictedInc
    self._synPermBelowStimulusInc = synPermConnected / 10.0
    self._synPermConnected = synPermConnected
    self._minPctOverlapDutyCycles = minPctOverlapDutyCycle
    self._minPctActiveDutyCycles = minPctActiveDutyCycle
    self._dutyCyclePeriod = dutyCyclePeriod
    self._maxBoost = maxBoost
    self._spVerbosity = spVerbosity
    self._wrapAround = wrapAround
    self._synPermActiveInactiveDec = synPermActiveInactiveDec
    self.useBurstingRule = useBurstingRule
    self.usePoolingRule = usePoolingRule
    self._poolingLife = poolingLife
    self._initConnectedPct = initConnectedPct
    # Extra parameter settings
    self._synPermMin = 0.0
    self._synPermMax = 1.0
    self._synPermTrimThreshold = synPermActiveInc / 2.0
    assert(self._synPermTrimThreshold < self._synPermConnected)
    self._updatePeriod = 20
    self._poolingThreshUnpredicted = poolingThreshUnpredicted

    # Internal state
    self._version = 1.0
    self._iterationNum = 0
    self._iterationLearnNum = 0

    # initialize the random number generators
    self._seed(seed)
    
    # A cell will enter pooling state if it receives enough predicted inputs
    # pooling cells have priority during competition
    self._poolingState = numpy.zeros((self._numColumns), dtype='int32')
    self._poolingColumns = []
    # Initialize a tiny random tie breaker. This is used to determine winning
    # columns where the overlaps are identical.
    self._tieBreaker = 0.01*numpy.array([self._random.getReal64() for i in
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
    # neighborhood. of a column. A cortical column must overcome the overlap 
    # score of columns in its neighborhood in order to become active. This
    # radius is updated every updatePeriod iterations. It grows and shrinks
    # with the average number of connected synapses per column.
    self._inhibitionRadius = 0
    self._updateInhibitionRadius()
    
    if self._spVerbosity > 0:
      self.printParameters()

  def initializeConnections(self):

    '''
    Initialize connection matrix, including:

    _permanences : permanence of synaptic connections (sparse matrix)
    _potentialPools: potential pool of connections for each cell
                    (sparse binary matrix)
    _connectedSynapses: connected synapses (binary sparse matrix)
    _connectedCounts: number of connections per cell (numpy array)
    _permanenceDecCache: a cache for permanence decremant. 
    ''' 

    numColumns = self._numColumns
    numInputs = self._numInputs

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

    # NEW. A cache for permanence decrements of (Active->Inactive Type)
    # Permanence decrements won't be initiated until the next time
    # a cell fire
    self._permanenceDecCache = SparseMatrix(numColumns, numInputs)


    # 'self._connectedSynapses' is a similar matrix to 'self._permanences' 
    # (rows represent cortial columns, columns represent input bits) whose 
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


    # TODO: The permanence initialiation code below runs faster but is not
    # deterministic (doesn't use our random seed.  We should clean up the code
    # and consider moving it to the base spatial pooler class itself.

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
    #                         .astype('int32'),dtype=uintType)
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
    Reset the status of temporal pooler
    """
    self._poolingState = numpy.zeros((self._numColumns), dtype='int32')
    self._poolingColumns = []
    self._overlapDutyCycles = numpy.zeros(self._numColumns, dtype=realDType)
    self._activeDutyCycles = numpy.zeros(self._numColumns, dtype=realDType)
    self._minOverlapDutyCycles = numpy.zeros(self._numColumns, 
                                             dtype=realDType)
    self._minActiveDutyCycles = numpy.zeros(self._numColumns,
                                            dtype=realDType)
    self._boostFactors = numpy.ones(self._numColumns, dtype=realDType)

  def compute(self, inputVector, learn, activeArray, burstingColumns,
              predictedCells):
    """
    This is the primary public method of the SpatialPooler class. This 
    function takes a input vector and outputs the indices of the active columns.
    If 'learn' is set to True, this method also updates the permanences of the
    columns.

    Parameters:
    ----------------------------
    inputVector:    a numpy array of 0's and 1's thata comprises the input to 
                    the spatial pooler. The array will be treated as a one
                    dimensional array, therefore the dimensions of the array
                    do not have to much the exact dimensions specified in the 
                    class constructor. In fact, even a list would suffice. 
                    The number of input bits in the vector must, however, 
                    match the number of bits specified by the call to the 
                    constructor. Therefore there must be a '0' or '1' in the
                    array for every input bit.
    learn:          a boolean value indicating whether learning should be 
                    performed. Learning entails updating the  permanence 
                    values of the synapses, and hence modifying the 'state' 
                    of the model. Setting learning to 'off' freezes the SP
                    and has many uses. For example, you might want to feed in
                    various inputs and examine the resulting SDR's.
    activeArray:    an array whose size is equal to the number of columns. 
                    Before the function returns this array will be populated 
                    with 1's at the indices of the active columns, and 0's 
                    everywhere else.
    burstingColumns: a numpy array with numColumns elements. A 1 indicates that
                    column is currently bursting.
    predictedCells: a numpy array with numInputs elements. A 1 indicates that
                    this cell switching from predicted state in the previous
                    time step to active state in the current timestep
    """
    assert (numpy.size(inputVector) == self._numInputs)
    assert (numpy.size(predictedCells) == self._numInputs)

    self._updateBookeepingVars(learn)
    inputVector = numpy.array(inputVector, dtype=realDType)
    predictedCells = numpy.array(predictedCells, dtype=realDType)
    inputVector.reshape(-1)

    if self._spVerbosity > 3:
      print " Input bits: ", inputVector.nonzero()[0]
      print " predictedCells: ", predictedCells.nonzero()[0]

    # Phase 1: Calculate overlap scores
    # The overlap score has 4 components
    # (1) Overlap with correctly predicted inputs for pooling cells 
    # (2) Overlap with correctly predicted inputs for all cells 
    # (3) Overlap with all inputs
    # (4) Overlap with cells in the bursting column

    # 1) Check pooling rule.
    if self.usePoolingRule:
      overlapsPooling = self._calculatePoolingActivity(predictedCells, learn)

      if self._spVerbosity > 4:
        print "Use Pooling Rule: Overlaps after step a:"
        print "   ", overlapsPooling

    else:
      overlapsPooling = 0
  
    # 2) Calculate overlap between input and connected synapses
    overlapsAllInput = self._calculateOverlap(inputVector)

    # 3) overlap with predicted inputs
    overlapsPredicted = self._calculateOverlap(predictedCells)      

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

    
    overlaps = overlapsPooling + overlapsPredicted + \
              overlapsAllInput + overlapsBursting

    # Apply boosting when learning is on
    if learn:
      boostedOverlaps = self._boostFactors * overlaps
      if self._spVerbosity > 4:
        print "Overlaps after boosting:"
        print "   ", boostedOverlaps      

    else:
      boostedOverlaps = overlaps
    

    # Apply inhibition to determine the winning columns
    activeColumns = self._inhibitColumns(boostedOverlaps)    
      
    if learn:
      self._adaptSynapses(inputVector, activeColumns, predictedCells)
      self._updateDutyCycles(overlaps, activeColumns)
      self._bumpUpWeakColumns() 
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()
    
    activeArray.fill(0)
    if activeColumns.size > 0:
      activeArray[activeColumns] = 1

    # update pooling state of cells
    activeColWithPredictedInput = activeColumns[numpy.where(\
                                overlapsPredicted[activeColumns]>0)[0]]
    
    numUnPredictedInput = float(len(burstingColumns.nonzero()[0]))
    numPredictedInput = float(len(predictedCells))
    fracUnPredicted = numUnPredictedInput/(numUnPredictedInput + numPredictedInput)

    self._updatePoolingState(activeColWithPredictedInput, fracUnPredicted)

    if self._spVerbosity > 2:
      activeColumns.sort()
      print "The following columns are finally active:"
      print "   ",activeColumns
      print "The following columns are in pooling state:"
      print "   ",self._poolingState.nonzero()[0]
      # print "Inputs to pooling columns"
      # print "   ",overlapsPredicted[self._poolingColumns]


  def _updatePoolingState(self, activeColWithPredictedInput, fracUnPredicted):
    """
    This function update pooling state of cells

    A cell will stop pooling if 
    (1) it hasn't received any predicted input in the last self._poolingLife steps
    or
    (2) the fraction of unpredicted input is above poolingThreshUnpredicted
    """
    

    if fracUnPredicted>self._poolingThreshUnpredicted:
      # reset pooling state if the fraction of unpredicted input 
      # is above the threshold
      if self._spVerbosity > 3:
        print " reset pooling state for all cells"
      self._poolingState = numpy.zeros((self._numColumns), dtype='int32')
    else:
      # decremant life of all pooling cells
      self._poolingState[self._poolingColumns] -= 1
      # reset life of cells that are receiving predicted input
      self._poolingState[activeColWithPredictedInput] = self._poolingLife

    self._poolingColumns = self._poolingState.nonzero()[0]


  def _calculatePoolingActivity(self, predictedCells, learn):
    """
    This function determines each column's overlap with metabotropically
    activated inputs. Overlap is calculated based on potential synapses. The
    overlap of a column that was previously active and has even one active
    metabotropic synapses is set to _numInputs+1 so they are guaranteed to win
    during inhibition.
    
    TODO: check with Jeff, what happens in biology if a cell was not previously
    active but receives metabotropic input?  Does it turn on, or does the met.
    input just extend the activity of already active cells? Does it matter
    whether the synapses are connected or not? Currently we are assuming no
    because most connected synapses become disconnected in the previous time
    step. If column i at time t is bursting and performs learning, the synapses
    from t+1 aren't active at time t, so their permanence will decrease.

    Parameters:
    ----------------------------
    predictedCells: a numpy array with numInputs elements. A 1 indicates that
                    this cell switching from predicted state in the previous
                    time step to active state in the current timestep  
    """


    overlaps = numpy.zeros(self._numColumns).astype(realDType)

    # no predicted inputs or no cell in pooling state, return zero
    if sum(self._poolingState)==0 or len(predictedCells.nonzero()[0])==0:
      return overlaps


    # self._connectedSynapses.rightVecSumAtNZ_fast(predictedCells, overlaps)      
    if learn:
      # During learning, overlap is calculated based on potential synapses. 
      self._potentialPools.rightVecSumAtNZ_fast(predictedCells, overlaps)
    else:
      # At inference stage, overlap is calculated based on connected synapses.
      self._connectedSynapses.rightVecSumAtNZ_fast(predictedCells, overlaps)      

    poolingColumns = self._poolingColumns

    # # only consider columns that are in pooling state
    mask = numpy.zeros(self._numColumns).astype(realDType)
    mask[poolingColumns] = 1    
    overlaps = overlaps * mask
    # columns that are in polling state and receives predicted input
    # will be boosted by a large factor
    boostFactorPooling = self._maxBoost*self._numInputs
    overlaps = boostFactorPooling * overlaps

    if self._spVerbosity > 3:
      print "\n============== Entering _calculateMetabotropicActivity ======"
      print "Received metabotropic inputs from following indices:"
      print "   ",predictedCells.nonzero()[0]
      print "The following column indices are in pooling state:"
      print "   ",poolingColumns
      print "Overlap score of pooling columns:"
      print "   ",overlaps[poolingColumns]
      print "============== Leaving _calculateMetabotropicActivity ======\n"

    return overlaps
  
  def _calculateBurstingColumns(self, burstingColumns):
    """
    Return the contribution to overlap due to bursting columns. If any column is
    bursting, its overlap score is set to stimulusThreshold. This
    means it will be guaranteed to win as long as no other column is
    metabotropic or has input > stimulusThreshold. 

    Parameters:
    ----------------------------
    burstingColumns: a numpy array with numColumns elements. A 1 indicates that
                    column is currently bursting.
    """
    overlaps = burstingColumns * self._stimulusThreshold
    return overlaps


  def _adaptSynapses(self, inputVector, activeColumns, predictedCells):
    """
    The primary method in charge of learning. Adapts the permanence values of 
    the synapses based on the input vector, and the chosen columns after 
    inhibition round. 
    The following rules applies to synapse adaptation:

    For active cells:

    1. synapses connected to input bits that are on are increased by synPermActiveInc
    2. synapses connected to input bits that are off are decreased by synPermInactiveDec  
    3. synapses connected to inputs bits that are on due to predicted inputs
    are increased by synPredictedInc. 

    For inactive cells:
    4. synapses connected to input bits that are on are decreased by synPermActiveInactiveDec


    Parameters:
    ----------------------------
    inputVector:    a numpy array of 0's and 1's thata comprises the input to 
                    the spatial pooler. There exists an entry in the array 
                    for every input bit.
    activeColumns:  an array containing the indices of the columns that 
                    survived inhibition.
    predictedCells: a numpy array with numInputs elements. A 1 indicates that
                    this cell switching from predicted state in the previous
                    time step to active state in the current timestep
    """
    inputIndices = numpy.where(inputVector > 0)[0]
    predictedIndices = numpy.where(predictedCells > 0)[0]
    permChanges = numpy.zeros(self._numInputs)

    # decrement Inactive -> active connections
    permChanges.fill(-1 * self._synPermInactiveDec)
    # increment active -> active connections
    permChanges[inputIndices] = self._synPermActiveInc
    # increment correctly predicted cells -> active connections
    permChanges[predictedIndices] = self._synPredictedInc

    if self._spVerbosity > 4:
      print "\n============== _adaptSynapses ======"
      print "Active input indices:",inputIndices
      print "predicted input indices:",predictedIndices
      # print "permChanges:"
      # print formatRow(permChanges, "%9.4f", 20)
      print "\n============== _adaptSynapses ======\n"

    for i in activeColumns:
      # input connections to column i
      perm = self._permanences.getRow(i)

      # decremant cached (active->inactive connections)
      permChangesBinary = self._permanenceDecCache.getRow(i)
      perm = numpy.where(permChangesBinary>0, perm-self._synPermActiveInactiveDec, perm)
      self._permanenceDecCache.setRowToZero(i)

      # only consider connections in potential pool
      maskPotential = numpy.where(self._potentialPools.getRow(i) > 0)[0]
      perm[maskPotential] += permChanges[maskPotential]
      self._updatePermanencesForColumn(perm, i, raisePerm=False)

    # decrement active -> inactive connections
    if self._synPermActiveInactiveDec > 0:

      for i in inputIndices: 
        # go through all active inputs
        if self._spVerbosity > 5:
          print "Active Input: ", i
          print "Current Connection: ", self._connectedSynapses.getRow(i)
                          
        # push permanance decremant to cache
        permChangesBinary = numpy.zeros(self._numColumns)
        permChangesBinary.fill(1) 
        permChangesBinary[activeColumns] = 0        
        self._permanenceDecCache.setColFromDense(i, permChangesBinary)     


  def printParameters(self):
    """
    Useful for debugging.
    """
    print "------------PY  SPTP Parameters ------------------"
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


  def extractInputForTP(self, tm):  
    """
    Extract inputs for TP from the state of temporal memory
    three information are extracted
    1. correctly predicted cells
    2. all active cells
    3. bursting cells (unpredicted input)
    """

    # bursting cells in layer 4
    burstingColumns = tm.activeState['t'].sum(axis=1)
    burstingColumns[ burstingColumns < tm.cellsPerColumn ] = 0
    burstingColumns[ burstingColumns == tm.cellsPerColumn ] = 1
    # print "Bursting column indices=",burstingColumns.nonzero()[0]  
    
    # correctly predicted cells in layer 4
    correctlyPredictedCells = numpy.zeros(self._inputDimensions).astype(realDType)
    idx = (tm.predictedState['t-1'] + tm.activeState['t']) == 2
    idx = idx.reshape(self._inputDimensions)
    correctlyPredictedCells[idx] = 1.0
    # print "Predicted->active cell indices=",correctlyPredictedCells.nonzero()[0]

    # all currently active cells in layer 4
    spInputVector = tm.learnState['t'].reshape(self._inputDimensions)  
    # spInputVector = tm.activeState['t'].reshape(self._inputDimensions)      
    
    return (correctlyPredictedCells, spInputVector, burstingColumns)  
