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

		self._inputDimensions = inputDimensions
		self._historyLength = historyLength
		self._numInputs = 1
		for d in inputDimensions:
			self._numInputs *= d

		# Current union SDR; the output of the union pooler algorithm
		self._unionSDR = numpy.zeros(shape=self._inputDimensions, dtype=UINT_DTYPE)

		# Indices of currently active cells
		self._activeCellsHistory = []


	def reset(self):
		"""
		Reset Union Pooler, clear active cell history
		"""
		self._unionSDR = numpy.zeros(shape=self._inputDimensions, dtype=UINT_DTYPE)
		self._activeCellsHistory = []


	def compute(self, activeCells):
		"""
		Computes one cycle of the Union Pooler algorithm. Return the union SDR
		Parameters:
		----------------------------
		@param activeCells: A list that stores indices of active cells
		"""
		self._activeCellsHistory.append(activeCells)
		if len(self._activeCellsHistory) > self._historyLength:
			self._activeCellsHistory.pop(0)

		self.createUnionSDR()
		return self._unionSDR


	def createUnionSDR(self):
		"""
		Create union SDR from a history of active cells
		"""
		self._unionSDR = numpy.zeros(shape=self._inputDimensions, dtype=UINT_DTYPE)
		for i in self._activeCellsHistory:
			self._unionSDR[i] = 1


	def unionIntoArray(self, inputVector, outputVector):
		"""
		Create a union of the inputVector and copy the result into the output Vector
		Parameters:
		----------------------------
		@param inputVector: The inputVector can be either a full numpy array containing
		0's and 1's, or a list of non-zero entry indices
		@param outputVector: A numpy array that matches the inputDimensions of the
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
					"Non-zero entry indice exceeds input dimension of union pooler. "
					"Expecting %s but got %s" % (self._numInputs, max(inputVector)))
			else:
				activeBits = inputVector
		else:
			raise ValueError("Unsuported input types")

		self.compute(activeBits)

		outputVector[:] = 0
		for i in self._activeCellsHistory:
			outputVector[i] = 1
