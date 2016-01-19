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
from htmresearch.regions.TemporalPoolerRegion import TemporalPoolerRegion



class TemporalPoolerRegionTest(unittest.TestCase):
	def setUp(self):
		self.tpRegion = None


	def testSimpleUnion(self):
		self.tpRegion = TemporalPoolerRegion(1024, 1024, 10, "simpleUnion")
		self.tpRegion.initialize([], [])

		outputs = {"mostActiveCells": numpy.zeros((self.tpRegion._inputWidth,))}

		activeCells = numpy.zeros(self.tpRegion._inputWidth)
		activeCells[[1, 5, 10]] = 1
		inputs = {"activeCells": activeCells}

		self.tpRegion.compute(inputs, outputs)

		activeCells = numpy.zeros(self.tpRegion._inputWidth)
		activeCells[[3, 17, 40]] = 1
		inputs = {"activeCells": activeCells}

		self.tpRegion.compute(inputs, outputs)

		self.assertSetEqual(set(numpy.where(outputs["mostActiveCells"])[0]),
		                    {1, 5, 10, 3, 17, 40})



if __name__ == "__main__":
	unittest.main()
