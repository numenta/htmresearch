# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
from nupic.bindings.math import GetNTAReal
REAL_DTYPE = GetNTAReal()
UINT_DTYPE = "uint32"

class SimpleUnionPooler(object):
	"""
	Experimental Simple Union Pooler Python Implementation.
	The simple union pooler computes a union of the last N SDRs
	"""


	def __init__(self,
							 inputDimensions=(2048,),
							 historyLength=10):
		"""
		Parameters:
		----------------------------
		@param inputDimensions:
			A sequence representing the dimensions of the input vector. Format is
			(height, width, depth, ...), where each value represents the size of the
			dimension.  For a topology of one dimension with 100 inputs use 100, or
			(100,). For a two dimensional topology of 10x5 use (10,5).

		@param historyLength: The union window length. For a union of the last
		10 steps, use historyLength=10
		"""
		inputDimensions = numpy.array(inputDimensions, ndmin=1)

		self._numInputs = inputDimensions.prod()
		self._historyLength = historyLength

		# Current union SDR; the output of the union pooler algorithm
		self._unionSDR = numpy.zeros(shape=inputDimensions, dtype=UINT_DTYPE)

		# Indices of currently active cells
		self._activeCells = set()

		# Activation time of all cells
		self._activationTimer = numpy.zeros(self._numInputs, dtype=REAL_DTYPE)


	def reset(self):
		self._unionSDR = numpy.array([], dtype=UINT_DTYPE)
		self._activeCells = set()
		self._activationTimer = numpy.zeros(self._numInputs, dtype=REAL_DTYPE)


	def compute(self, inputVector):
		"""
		Computes one cycle of the Union Pooler algorithm.
		@param inputVector: A numpy array of 0's and 1's that comprises the
		input to the union pooler. The array will be treated as a one dimensional array
		"""

		if not isinstance(inputVector, numpy.ndarray):
			raise TypeError("Input vector must be a numpy array, not %s" %
											str(type(inputVector)))

		if inputVector.size != self._numInputs:
			raise ValueError(
					"Input vector dimensions don't match. Expecting %s but got %s" % (
							inputVector.size, self._numInputs))

		inputVector = numpy.array(inputVector, dtype=REAL_DTYPE)
		inputVector.reshape(-1)

		activeBits = set(numpy.where(inputVector)[0])

		self._activeCells = self._activeCells | activeBits
		self._activationTimer[list(self._activeCells)] += 1

		cellsPastHistoryLength = set(numpy.where(self._activationTimer > self._historyLength)[0])
		self._activeCells = self._activeCells - cellsPastHistoryLength


	def getUnionSDR(self):
		self._unionSDR = numpy.zeros(shape=(self._numInputs,), dtype=UINT_DTYPE)
		self._unionSDR[list(self._activeCells)] = 1
		return self._unionSDR
