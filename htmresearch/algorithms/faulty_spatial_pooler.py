# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
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

"""
Faulty Spatial Pooler implementation in Python.
"""
import itertools
import numpy
from collections import defaultdict
from nupic.research.spatial_pooler import SpatialPooler
import nupic.math.topology as topology
from nupic.bindings.math import GetNTAReal

realDType = GetNTAReal()
uintType = "uint32"


class FaultySpatialPooler(SpatialPooler):
  """
  Class implementing a fallible Spatial Pooler class. This class allows the
  user to kill a certain number of cells. The dead cells cannot become active,
  will no longer participate in competition, and will not learn new connections
  """


  def __init__(self,
               **kwargs):

    self.deadCols = numpy.array([])
    self.zombiePermutation = None  # Contains the order in which cells
    # will be killed
    self.numDead = 0
    self.deadColumnInputSpan = None
    self.targetDensity = None
    super(FaultySpatialPooler, self).__init__(**kwargs)


  def killCells(self, percent=0.05):
    """
    Changes the percentage of cells that are now considered dead. The first
    time you call this method a permutation list is set up. Calls change the
    number of cells considered dead.
    """
    numColumns = numpy.prod(self.getColumnDimensions())

    if self.zombiePermutation is None:
      self.zombiePermutation = numpy.random.permutation(numColumns)

    self.numDead = round(percent * numColumns)

    if self.numDead > 0:
      self.deadCols = self.zombiePermutation[0:self.numDead]
    else:
      self.deadCols = numpy.array([])

    self.deadColumnInputSpan = self.getConnectedSpan(self.deadCols)
    self.removeDeadColumns()


  def killCellRegion(self, centerColumn, radius):
    """
    Kill cells around a centerColumn, within radius
    """
    self.deadCols = topology.wrappingNeighborhood(centerColumn,
                                          radius,
                                          self._columnDimensions)
    self.deadColumnInputSpan = self.getConnectedSpan(self.deadCols)
    self.removeDeadColumns()


  def removeDeadColumns(self):
    print "Total number of dead cells = {}".format(len(self.deadCols))
    for columnIndex in self.deadCols:
      potential = numpy.zeros(self._numInputs, dtype=uintType)
      self._potentialPools.replace(columnIndex, potential.nonzero()[0])

      perm = numpy.zeros(self._numInputs, dtype=realDType)
      self._updatePermanencesForColumn(perm, columnIndex, raisePerm=False)


  def getConnectedSpan(self, columns):
    dimensions = self._inputDimensions

    maxCoord = numpy.empty(self._inputDimensions.size)
    minCoord = numpy.empty(self._inputDimensions.size)
    maxCoord.fill(-1)
    minCoord.fill(max(self._inputDimensions))
    for columnIndex in columns:
      connected = self._connectedSynapses[columnIndex].nonzero()[0]
      if connected.size == 0:
        continue
      for i in connected:
        maxCoord = numpy.maximum(maxCoord, numpy.unravel_index(i, dimensions))
        minCoord = numpy.minimum(minCoord, numpy.unravel_index(i, dimensions))
    return (minCoord, maxCoord)


  def updatePotentialRadius(self, newPotentialRadius):
    """
    Change the potential radius for all columns
    :return:
    """
    oldPotentialRadius = self._potentialRadius
    self._potentialRadius = newPotentialRadius
    numColumns = numpy.prod(self.getColumnDimensions())
    for columnIndex in xrange(numColumns):
      potential = self._mapPotential(columnIndex)
      self._potentialPools.replace(columnIndex, potential.nonzero()[0])

    self._updateInhibitionRadius()



  def compute(self, inputVector, learn, activeArray):
    """
    This is the primary public method of the SpatialPooler class. This
    function takes a input vector and outputs the indices of the active columns.
    If 'learn' is set to True, this method also updates the permanences of the
    columns.

    @param inputVector: A numpy array of 0's and 1's that comprises the input
        to the spatial pooler. The array will be treated as a one dimensional
        array, therefore the dimensions of the array do not have to match the
        exact dimensions specified in the class constructor. In fact, even a
        list would suffice. The number of input bits in the vector must,
        however, match the number of bits specified by the call to the
        constructor. Therefore there must be a '0' or '1' in the array for
        every input bit.
    @param learn: A boolean value indicating whether learning should be
        performed. Learning entails updating the  permanence values of the
        synapses, and hence modifying the 'state' of the model. Setting
        learning to 'off' freezes the SP and has many uses. For example, you
        might want to feed in various inputs and examine the resulting SDR's.
    @param activeArray: An array whose size is equal to the number of columns.
        Before the function returns this array will be populated with 1's at
        the indices of the active columns, and 0's everywhere else.
    """
    if not isinstance(inputVector, numpy.ndarray):
      raise TypeError("Input vector must be a numpy array, not %s" %
                      str(type(inputVector)))

    if inputVector.size != self._numInputs:
      raise ValueError(
          "Input vector dimensions don't match. Expecting %s but got %s" % (
              inputVector.size, self._numInputs))

    self._updateBookeepingVars(learn)
    inputVector = numpy.array(inputVector, dtype=realDType)
    inputVector.reshape(-1)
    self._overlaps = self._calculateOverlap(inputVector)
    # self._overlaps[self.deadCols] = 0

    # Apply boosting when learning is on
    if learn:
      self._boostedOverlaps = self._boostFactors * self._overlaps
    else:
      self._boostedOverlaps = self._overlaps

    # Apply inhibition to determine the winning columns
    activeColumns = self._inhibitColumns(self._boostedOverlaps)

    if learn:
      self._adaptSynapses(inputVector, activeColumns)
      self._updateDutyCycles(self._overlaps, activeColumns)
      self._bumpUpWeakColumns()
      self._updateTargetActivityDensity()
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

      # self.growRandomSynapses()

    activeArray.fill(0)
    activeArray[activeColumns] = 1


  def growRandomSynapses(self):
    columnFraction = 0.02

    selectColumns = numpy.random.choice(self._numColumns,
                                        size=int(columnFraction * self._numColumns),
                                        replace=False)
    for columnIndex in selectColumns:
      perm = self._permanences[columnIndex]
      unConnectedSyn = numpy.where(numpy.logical_and(
        self._potentialPools[columnIndex] > 0,
        perm < self._synPermConnected))[0]
      if len(unConnectedSyn) == 0:
        continue
      selectSyn = numpy.random.choice(unConnectedSyn)
      perm[selectSyn] = self._synPermConnected + self._synPermActiveInc
      self._updatePermanencesForColumn(perm, columnIndex, raisePerm=False)


  def _updateBoostFactors(self):
    """
    Update the boost factors for all columns. The boost factors are used to
    increase the overlap of inactive columns to improve their chances of
    becoming active. and hence encourage participation of more columns in the
    learning process. This is a line defined as: y = mx + b boost =
    (1-maxBoost)/minDuty * dutyCycle + maxFiringBoost. Intuitively this means
    that columns that have been active enough have a boost factor of 1, meaning
    their overlap is not boosted. Columns whose active duty cycle drops too much
    below that of their neighbors are boosted depending on how infrequently they
    have been active. The more infrequent, the more they are boosted. The exact
    boost factor is linearly interpolated between the points (dutyCycle:0,
    boost:maxFiringBoost) and (dutyCycle:minDuty, boost:1.0).

            boostFactor
                ^
    maxBoost _  |
                |\
                | \
          1  _  |  \ _ _ _ _ _ _ _
                |
                +--------------------> activeDutyCycle
                   |
            minActiveDutyCycle
    """
    if self._maxBoost > 1:
      self._boostFactors = numpy.exp(-(
        self._activeDutyCycles-self.targetDensity) * self._maxBoost)
    else:
      pass


  def getAliveColumns(self):
    numColumns = numpy.prod(self.getColumnDimensions())
    aliveColumns = numpy.ones(numColumns)
    aliveColumns[self.deadCols] = 0
    return aliveColumns.nonzero()[0]


  def _updateTargetActivityDensity(self):
    if (self._localAreaDensity > 0):
      density = self._localAreaDensity
    else:
      inhibitionArea = ((2 * self._inhibitionRadius + 1)
                        ** self._columnDimensions.size)
      inhibitionArea = min(self._numColumns, inhibitionArea)
      density = float(self._numActiveColumnsPerInhArea) / inhibitionArea
      density = min(density, 0.5)

    if self._globalInhibition:
      targetDensity = density * numpy.ones(self._numColumns, dtype=realDType)
    else:
      targetDensity = numpy.zeros(self._numColumns, dtype=realDType)
      for i in xrange(self._numColumns):
        if i in self.deadCols:
          continue

        maskNeighbors = self._getColumnNeighborhood(i)
        targetDensity[i] = numpy.mean(self._activeDutyCycles[maskNeighbors])
    self.targetDensity = targetDensity