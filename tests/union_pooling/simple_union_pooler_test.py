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


import unittest
import numpy
import numpy.testing as npt
from htmresearch.algorithms.simple_union_pooler import SimpleUnionPooler

REAL_DTYPE = numpy.float32


class SimpleUnionPoolerTest(unittest.TestCase):
	def setUp(self):
		self.unionPooler = SimpleUnionPooler(inputDimensions=(2048,),
		                                     historyLength=10)

	def testUnionCompute(self):
		activeCells = []
		activeCells.append([1, 3, 4])
		activeCells.append([101, 302, 405])

		for i in xrange(len(activeCells)):
			inputVector = numpy.zeros(shape=(2048,))
			inputVector[numpy.array(activeCells[i])] = 1
			self.unionPooler.compute(inputVector)

		activeCellsUnion = set()
		for i in xrange(len(activeCells)):
			activeCellsUnion = activeCellsUnion | set(activeCells[i])
		self.assertSetEqual(self.unionPooler._activeCells, activeCellsUnion)

		for i in xrange(len(activeCells)):
			npt.assert_allclose(self.unionPooler._activationTimer[numpy.array(activeCells[i])],
			                    len(activeCells)-i)

	def testHistoryLength(self):
		self.unionPooler = SimpleUnionPooler(inputDimensions=(2048,),
		                                     historyLength=2)
		activeCells = []
		activeCells.append([1, 3, 4])
		activeCells.append([101, 302, 405])
		activeCells.append([240, 903, 858])

		for i in xrange(len(activeCells)):
			inputVector = numpy.zeros(shape=(2048,))
			inputVector[numpy.array(activeCells[i])] = 1
			self.unionPooler.compute(inputVector)

		activeCellsUnion = set()
		for i in xrange(1, len(activeCells)):
			activeCellsUnion = activeCellsUnion | set(activeCells[i])
		self.assertSetEqual(self.unionPooler._activeCells, activeCellsUnion)


if __name__ == '__main__':
	unittest.main()