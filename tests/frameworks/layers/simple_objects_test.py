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

import unittest

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)



class SimpleObjectsTest(unittest.TestCase):
  """(Incomplete and really simple) unit tests for simple object machine."""


  def testCreateRandom(self):
    """Simple construction test."""
    objects= createObjectMachine(
      machineType="simple",
      seed=42
    )
    objects.createRandomObjects(numObjects=10,
                                numPoints=10, numLocations=10, numFeatures=10)
    self.assertEqual(len(objects), 10)

    # Num locations must be >= num points
    with self.assertRaises(AssertionError):
      objects.createRandomObjects(numObjects=10,
                                  numPoints=10, numLocations=9, numFeatures=10)


  def testGetDistinctPairs(self):
    """Ensures we can compute unique pairs."""
    pairObjects = createObjectMachine(
      machineType="simple",
      numInputBits=20,
      sensorInputSize=150,
      externalInputSize=2400,
      numCorticalColumns=3,
      numFeatures=5,
      numLocations=10,
      seed=42
    )

    pairObjects.addObject([(1, 3)], 0)
    pairObjects.addObject([(3, 1), (1, 3)], 1)
    pairObjects.addObject([(2, 4)], 2)

    distinctPairs = pairObjects.getDistinctPairs()
    self.assertEqual(len(distinctPairs), 3)

    pairObjects.addObject([(2, 4), (1, 3)], 3)
    distinctPairs = pairObjects.getDistinctPairs()
    self.assertEqual(len(distinctPairs), 3)

    pairObjects.addObject([(2, 4), (1, 3), (1, 1)], 4)
    distinctPairs = pairObjects.getDistinctPairs()
    self.assertEqual(len(distinctPairs), 4)



if __name__ == "__main__":
  unittest.main()
