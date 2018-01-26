# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
from nupic.algorithms.spatial_pooler import SpatialPooler 
import numpy as np
import numpy
from nupic.bindings.math import GetNTAReal
realDType = GetNTAReal()
PERMANENCE_EPSILON = 0.000001



class LateralPooler(SpatialPooler):
  """
  An experimental spatial pooler implementation
  with learned lateral inhibitory connections.
  """
  def __init__(self, lateralLearningRate = 1.0, lateralDutyCyclePeriod=None, enforceDesiredWeight=True, **spArgs):


    super(LateralPooler, self).__init__(**spArgs)

    self.shape      = (self._numColumns, self._numInputs)
    self.codeWeight = self._numActiveColumnsPerInhArea
    self.sparsity   = float(self.codeWeight)/float(self._numColumns)

    # If true we activate `codeWeight` 
    # columns at most
    self.enforceDesiredWeight = enforceDesiredWeight

    # The new lateral inhibitory connections
    # and learning rates
    n = self._numColumns
    self.lateralConnections = np.ones((n,n))/float(n-1)
    np.fill_diagonal(self.lateralConnections, 0.0)
    self.lateralLearningRate = lateralLearningRate
    if lateralDutyCyclePeriod == None:
      self.lateralDutyCyclePeriod = self._dutyCyclePeriod
    else:
      self.lateralDutyCyclePeriod = lateralDutyCyclePeriod

    # Varibale to store average pairwise activities
    s = self.sparsity
    self.avgActivityPairs = np.ones((n,n))*(s**2)
    np.fill_diagonal(self.avgActivityPairs, s)

    # experimental boosting
    self._beta = 0.0
    
  def _inhibitColumnsWithLateral(self, overlaps, lateralConnections):
    """
    Performs an experimentatl local inhibition. Local inhibition is 
    iteratively performed on a column by column basis.
    """
    n,m = self.shape
    y   = np.zeros(n)
    s   = self.sparsity
    L   = lateralConnections
    desiredWeight = self.codeWeight
    inhSignal     = np.zeros(n)
    sortedIndices = np.argsort(overlaps, kind='mergesort')[::-1]

    currentWeight = 0
    for i in sortedIndices:

      if overlaps[i] < self._stimulusThreshold:
        break

      inhTooStrong = ( inhSignal[i] >= s )

      if not inhTooStrong:
        y[i]              = 1.
        currentWeight    += 1
        inhSignal[:]     += L[i,:]

      if self.enforceDesiredWeight and currentWeight == desiredWeight:
        break

    activeColumns = np.where(y==1.0)[0]

    return activeColumns    


  def _updateAvgActivityPairs(self, activeArray):
    """
    Updates the average firing activity of pairs of 
    columns.
    """
    n, m = self.shape
    Y    = activeArray.reshape((n,1))
    beta = 1.0 - 1.0/self._dutyCyclePeriod

    Q = np.dot(Y, Y.T) 

    self.avgActivityPairs = beta*self.avgActivityPairs + (1-beta)*Q



  def _updateLateralConnections(self, epsilon, avgActivityPairs):
    """
    Sets the weights of the lateral connections based on 
    average pairwise activity of the SP's columns. Intuitively: The more 
    two columns fire together on average the stronger the inhibitory
    connection gets. 
    """
    oldL = self.lateralConnections
    newL = avgActivityPairs.copy()
    np.fill_diagonal(newL, 0.0)
    newL = newL/np.sum(newL, axis=1, keepdims=True)

    self.lateralConnections[:,:] = (1 - epsilon)*oldL + epsilon*newL



  def compute(self, inputVector, learn, activeArray, applyLateralInhibition=True):
    """
    This is the primary public method of the LateralPooler class. This
    function takes a input vector and outputs the indices of the active columns.
    If 'learn' is set to True, this method also updates the permanences of the
    columns and their lateral inhibitory connection weights.
    """
    if not isinstance(inputVector, np.ndarray):
      raise TypeError("Input vector must be a numpy array, not %s" %
                      str(type(inputVector)))

    if inputVector.size != self._numInputs:
      raise ValueError(
          "Input vector dimensions don't match. Expecting %s but got %s" % (
              inputVector.size, self._numInputs))

    self._updateBookeepingVars(learn)
    inputVector = np.array(inputVector, dtype=realDType)
    inputVector.reshape(-1)
    self._overlaps = self._calculateOverlap(inputVector)

    # Apply boosting when learning is on
    if learn:
      self._boostedOverlaps = self._boostFactors * self._overlaps
    else:
      self._boostedOverlaps = self._overlaps

    # Apply inhibition to determine the winning columns
    if applyLateralInhibition == True:
      activeColumns = self._inhibitColumnsWithLateral(self._boostedOverlaps, self.lateralConnections)
    else:
      activeColumns = self._inhibitColumns(self._boostedOverlaps)
    activeArray.fill(0)
    activeArray[activeColumns] = 1.0

    if learn:
      self._adaptSynapses(inputVector, activeColumns, self._boostedOverlaps)
      self._updateDutyCycles(self._overlaps, activeColumns)
      self._bumpUpWeakColumns()
      self._updateBoostFactors()
      self._updateAvgActivityPairs(activeArray)

      epsilon = self.lateralLearningRate
      if epsilon > 0:
        self._updateLateralConnections(epsilon, self.avgActivityPairs)

      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    return activeArray


  def encode(self, X, applyLateralInhibition=True):
    """
    This method encodes a batch of input vectors.
    Note the inputs are assumed to be given as the 
    columns of the matrix X (not the rows).
    """
    d = X.shape[1]
    n = self._numColumns
    Y = np.zeros((n,d))
    for t in range(d):
        self.compute(X[:,t], False, Y[:,t], applyLateralInhibition)
        
    return Y


  @property
  def feedforward(self):
    """
    Soon to be depriciated.
    Needed to make the SP implementation compatible 
    with some older code.
    """
    m = self._numInputs
    n = self._numColumns
    W = np.zeros((n, m))
    for i in range(self._numColumns):
        self.getPermanence(i, W[i, :])

    return W

  @property
  def code_weight(self):
    """
    Soon to be depriciated.
    Needed to make the SP implementation compatible 
    with some older code.
    """
    return self._numActiveColumnsPerInhArea


  @property
  def smoothing_period(self):
    """
    Soon to be depriciated.
    Needed to make the SP implementation compatible 
    with some older code.
    """
    return self._dutyCyclePeriod


  @property
  def avg_activity_pairs(self):
    """
    Soon to be depriciated.
    Needed to make the SP implementation compatible 
    with some older code.
    """
    return self.avgActivityPairs


  def _updateBoostFactorsGlobal(self):
    """
    Update boost factors when global inhibition is used
    """
    # When global inhibition is enabled, the target activation level is
    # the sparsity of the spatial pooler
    if (self._localAreaDensity > 0):
      targetDensity = self._localAreaDensity
    else:
      inhibitionArea = ((2 * self._inhibitionRadius + 1)
                        ** self._columnDimensions.size)
      inhibitionArea = min(self._numColumns, inhibitionArea)
      targetDensity = float(self._numActiveColumnsPerInhArea) / inhibitionArea
      targetDensity = min(targetDensity, 0.5)


    # Usual definition
    self._beta = (targetDensity - self._activeDutyCycles)

    # Experimental setting
    # self._beta += 0.001*(targetDensity - self._activeDutyCycles)
    
    self._boostFactors = np.exp(self._beta * self._boostStrength)


  def _adaptSynapses(self, inputVector, activeColumns, overlaps):
    """
    The primary method in charge of learning. Adapts the permanence values of
    the synapses based on the input vector, and the chosen columns after
    inhibition round. Permanence values are increased for synapses connected to
    input bits that are turned on, and decreased for synapses connected to
    inputs bits that are turned off.

    Parameters:
    ----------------------------
    :param inputVector:
                    A numpy array of 0's and 1's that comprises the input to
                    the spatial pooler. There exists an entry in the array
                    for every input bit.
    :param activeColumns:
                    An array containing the indices of the columns that
                    survived inhibition.
    """
    inputIndices = np.where(inputVector > 0)[0]

    permChanges = np.zeros(self._numInputs, dtype=realDType)
    permChanges.fill(-1 * self._synPermInactiveDec)
    permChanges[inputIndices] = self._synPermActiveInc
    for columnIndex in activeColumns:
      perm = self._permanences[columnIndex]
      maskPotential = np.where(self._potentialPools[columnIndex] > 0)[0]
      perm[maskPotential] += permChanges[maskPotential]
      self._updatePermanencesForColumn(perm, columnIndex, raisePerm=True)






