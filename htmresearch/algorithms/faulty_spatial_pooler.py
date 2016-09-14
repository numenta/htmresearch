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

    print "Total number of dead cells = {}".format(len(self.deadCols))

    for columnIndex in self.deadCols:
      potential = numpy.zeros(self._numInputs, dtype=uintType)
      self._potentialPools.replace(columnIndex, potential.nonzero()[0])

      perm = numpy.zeros(self._numInputs, dtype=realDType)
      self._updatePermanencesForColumn(perm, columnIndex, raisePerm=False)



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
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    activeArray.fill(0)
    activeArray[activeColumns] = 1



  def _getNeighborsND(self, columnIndex, dimensions, radius, wrapAround=False):
    """
    Similar to _getNeighbors1D and _getNeighbors2D, this function Returns a
    list of indices corresponding to the neighbors of a given column. Since the
    permanence values are stored in such a way that information about topology
    is lost. This method allows for reconstructing the topology of the inputs,
    which are flattened to one array. Given a column's index, its neighbors are
    defined as those columns that are 'radius' indices away from it in each
    dimension. The method returns a list of the flat indices of these columns.
    Parameters:
    ----------------------------
    @param columnIndex: The index identifying a column in the permanence, potential
                    and connectivity matrices.
    @param dimensions: An array containing a dimensions for the column space. A 2x3
                    grid will be represented by [2,3].
    @param radius:  Indicates how far away from a given column are other
                    columns to be considered its neighbors. In the previous 2x3
                    example, each column with coordinates:
                    [2+/-radius, 3+/-radius] is considered a neighbor.
    @param wrapAround: A boolean value indicating whether to consider columns at
                    the border of a dimensions to be adjacent to columns at the
                    other end of the dimension. For example, if the columns are
                    laid out in one dimension, columns 1 and 10 will be
                    considered adjacent if wrapAround is set to true:
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    assert(dimensions.size > 0)

    columnCoords = numpy.unravel_index(columnIndex, dimensions)
    rangeND = []
    for i in xrange(dimensions.size):
      if wrapAround:
        curRange = numpy.array(range(columnCoords[i]-radius,
                                     columnCoords[i]+radius+1)) % dimensions[i]
      else:
        curRange = numpy.array(range(columnCoords[i]-radius,
                                     columnCoords[i]+radius+1))
        curRange = curRange[
          numpy.logical_and(curRange >= 0, curRange < dimensions[i])]

      rangeND.append(numpy.unique(curRange))

    neighbors = numpy.ravel_multi_index(
      numpy.array(list(itertools.product(*rangeND))).T,
      dimensions).tolist()

    neighbors.remove(columnIndex)

    aliveNeighbors = []
    for columnIndex in neighbors:
      if columnIndex not in self.deadCols:
        aliveNeighbors.append(columnIndex)
    neighbors = aliveNeighbors

    return neighbors


  def getAliveColumns(self):
    numColumns = numpy.prod(self.getColumnDimensions())
    aliveColumns = numpy.ones(numColumns)
    aliveColumns[self.deadCols] = 0
    aliveColumnIdx = aliveColumns.nonzero()[0]
    return aliveColumnIdx