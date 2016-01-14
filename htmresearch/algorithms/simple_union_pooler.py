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

class SimpleUnionPooler(object):
  """
  Experimental Simple Union Pooler Python Implementation.
  The simple union pooler computes a union of the last N SDRs
  """

  def __init__(self,
               numInputs=2048,
               historyLength=10):
    """
    Parameters:
    ----------------------------
    @param numInputs: The length of the input SDRs
    @param historyLength: The union window length. For a union of the last
    10 steps, use historyLength=10
    """

    self._historyLength = historyLength
    self._numInputs = numInputs
    self.reset()


  def reset(self):
    """
    Reset Union Pooler, clear active cell history
    """
    self._unionSDR = numpy.zeros(shape=(self._numInputs,))
    self._activeCellsHistory = []


  def updateHistory(self, activeCells):
    """
    Computes one cycle of the Union Pooler algorithm. Return the union SDR
    Parameters:
    ----------------------------
    @param activeCells: A list that stores indices of active cells
    """
    self._activeCellsHistory.append(activeCells)
    if len(self._activeCellsHistory) > self._historyLength:
      self._activeCellsHistory.pop(0)

    self._unionSDR = numpy.zeros(shape=(self._numInputs,))
    for i in self._activeCellsHistory:
      self._unionSDR[i] = 1

    return self._unionSDR


  def unionIntoArray(self, inputVector, outputVector):
    """
    Create a union of the inputVector and copy the result into the outputVector
    Parameters:
    ----------------------------
    @param inputVector: The inputVector can be either a full numpy array
    containing 0's and 1's, or a list of non-zero entry indices
    @param outputVector: A numpy array that matches the length of the
    union pooler.
    """
    if isinstance(inputVector, numpy.ndarray):
      if inputVector.size == self._numInputs:
        activeBits = numpy.where(inputVector)[0]
      else:
        raise ValueError(
          "Input vector dimensions don't match. Expecting %s but got %s" % (
            self._numInputs, inputVector.size))
    elif isinstance(inputVector, list):
      if max(inputVector) >= self._numInputs:
        raise ValueError(
          "Non-zero entry indices exceed input dimension of union pooler. "
          "Expecting %s but got %s" % (self._numInputs, max(inputVector)))
      else:
        activeBits = inputVector
    else:
      raise TypeError("Unsuported input types")

    if len(outputVector) != self._numInputs:
      raise ValueError(
        "Output vector dimension does match dimension of union pooler "
        "Expecting %s but got %s" % (self._numInputs, len(outputVector)))

    unionSDR = self.updateHistory(activeBits)

    for i in xrange(len(unionSDR)):
      outputVector[i] = unionSDR[i]


  def getSparsity(self):
    """
    Return the sparsity of the current union SDR
    """
    sparsity = numpy.sum(self._unionSDR) / self._numInputs
    return sparsity